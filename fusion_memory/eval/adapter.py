from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from fusion_memory import MemoryService, Scope
from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import EvidencePack
from fusion_memory.core.text import compact_summary, keyword_score


@dataclass
class EvalDocument:
    id: str
    content: str
    timestamp: datetime | None = None
    speaker: str = "document"


@dataclass
class EvalQuery:
    id: str
    query: str
    gold_answers: list[str]
    category: str | None = None


@dataclass
class EvalResult:
    query_id: str
    answer_policy: str
    retrieved_source_span_ids: list[str]
    matched_gold: bool
    category: str | None
    query_type: str | None = None
    source_span_quota_met: bool | None = None
    coverage_insufficient: bool | None = None
    query_text: str = ""
    answer: str = ""
    evidence_pack: dict[str, Any] = field(default_factory=dict)
    evidence_matched_gold: bool = False
    answer_model: str = "local_extractive_v0"
    judge_model: str = "lexical_contains_v0"
    mode: str = "fast"
    tokens_query: int = 0
    retrieval_latency_ms: float = 0.0
    llm_calls: int = 0


class LocalExtractiveAnswerModel:
    version = "local_extractive_v0"

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def answer(self, query: str, pack: EvidencePack) -> str:
        if pack.answer_policy == "abstain_if_not_supported":
            return "Not enough supported memory to answer."
        candidates: list[str] = []
        candidates.extend(str(item.get("text", "")) for item in pack.current_views)
        candidates.extend(str(item.get("text", "")) for item in pack.entity_profiles)
        candidates.extend(str(item.get("text", "")) for item in pack.facts)
        candidates.extend(str(item.get("description", "")) for item in pack.events)
        candidates.extend(str(item.get("content", "")) for item in pack.source_spans)
        candidates = [candidate for candidate in candidates if candidate.strip()]
        if not candidates:
            return "Not enough supported memory to answer."
        candidates.sort(key=lambda candidate: keyword_score(query, candidate), reverse=True)
        return compact_summary(candidates[0], self.config.local_answer_summary_chars)


class LexicalContainsJudge:
    version = "lexical_contains_v0"

    def score(self, answer: str, gold_answers: list[str]) -> bool:
        answer_lower = answer.lower()
        return any(gold.lower() in answer_lower for gold in gold_answers)


class BenchmarkAdapter:
    """Minimal local benchmark adapter.

    It validates retrieval/evidence-pack behavior without pretending to replace
    BEAM's LLM answer and judge harness. Later adapters can subclass this and
    plug in benchmark-specific scoring.
    """

    def __init__(self, service: MemoryService, scope: Scope, answer_model: Any | None = None, judge_model: Any | None = None) -> None:
        self.service = service
        self.scope = scope
        self.answer_model = answer_model or LocalExtractiveAnswerModel(service.config)
        self.judge_model = judge_model or LexicalContainsJudge()

    def ingest_dataset(self, dataset_path: str | Path, split: str | None = None) -> dict[str, Any]:
        documents, _ = load_dataset(dataset_path, split=split)
        span_ids = self.ingest_documents(documents)
        return {"documents": len(documents), "span_ids": span_ids, "span_count": len(span_ids), "split": split}

    def build_queries(self, dataset_path: str | Path, split: str | None = None) -> list[EvalQuery]:
        _, queries = load_dataset(dataset_path, split=split)
        return queries

    def answer_query(self, query: EvalQuery, budget: dict[str, Any] | None = None) -> EvalResult:
        budget = budget or {}
        started = perf_counter()
        pack = self.service.answer_context(query.query, self.scope, budget=budget)
        latency_ms = (perf_counter() - started) * 1000
        model_call_mark = _model_call_count(self.answer_model, self.judge_model)
        answer = self.answer_model.answer(query.query, pack)
        evidence_blob = " ".join(span["content"] for span in pack.source_spans)
        evidence_blob += " " + str(pack.facts) + " " + str(pack.current_views) + " " + str(pack.events)
        evidence_matched = any(gold.lower() in evidence_blob.lower() for gold in query.gold_answers)
        answer_matched = self.judge_model.score(answer, query.gold_answers)
        llm_calls = _model_call_count(self.answer_model, self.judge_model) - model_call_mark
        return EvalResult(
            query_id=query.id,
            answer_policy=pack.answer_policy,
            retrieved_source_span_ids=[span["id"] for span in pack.source_spans],
            matched_gold=answer_matched,
            category=query.category,
            query_type=pack.coverage.get("query_type"),
            source_span_quota_met=pack.coverage.get("source_span_quota_met"),
            coverage_insufficient=pack.coverage.get("coverage_insufficient"),
            query_text=query.query,
            answer=answer,
            evidence_pack=_pack_summary(pack),
            evidence_matched_gold=evidence_matched,
            answer_model=getattr(self.answer_model, "version", self.answer_model.__class__.__name__),
            judge_model=getattr(self.judge_model, "version", self.judge_model.__class__.__name__),
            mode=str(budget.get("mode", "fast")),
            tokens_query=_approx_tokens(query.query, pack, answer),
            retrieval_latency_ms=latency_ms,
            llm_calls=llm_calls,
        )

    def ingest_documents(self, documents: list[EvalDocument]) -> list[str]:
        span_ids: list[str] = []
        for doc in documents:
            result = self.service.add(
                {
                    "role": doc.speaker,
                    "content": doc.content,
                    "turn_id": doc.id,
                    "timestamp": (doc.timestamp or datetime.now(timezone.utc)).isoformat(),
                },
                self.scope,
                doc.timestamp or datetime.now(timezone.utc),
                {"source_uri": doc.id},
            )
            span_ids.extend(result.span_ids)
        return span_ids

    def run_queries(self, queries: list[EvalQuery], budget: dict[str, Any] | None = None) -> list[EvalResult]:
        return [self.answer_query(query, budget=budget) for query in queries]

    def run_ablation(self, queries: list[EvalQuery], modes: list[str] | None = None) -> dict[str, Any]:
        modes = modes or ["fast", "balanced", "benchmark"]
        return {mode: self.report(self.run_queries(queries, budget={"mode": mode})) for mode in modes}

    def run_component_ablation(self, queries: list[EvalQuery]) -> dict[str, Any]:
        source_sets = {
            "L0": ["raw", "exact", "entities"],
            "L0+L1": ["raw", "exact", "entities", "facts"],
            "L0+L1+L2": ["raw", "exact", "entities", "facts", "events"],
            "Full": ["raw", "exact", "entities", "facts", "events", "views", "profiles"],
        }
        return {
            name: self.report(self.run_queries(queries, budget={"enabled_sources": enabled_sources}))
            for name, enabled_sources in source_sets.items()
        }

    def report(self, results: list[EvalResult]) -> dict[str, Any]:
        total = len(results)
        matched = sum(1 for result in results if result.matched_gold)
        evidence_matched = sum(1 for result in results if result.evidence_matched_gold)
        by_category: dict[str, dict[str, int]] = {}
        quota_met = sum(1 for result in results if result.source_span_quota_met)
        abstentions = sum(1 for result in results if result.answer_policy == "abstain_if_not_supported")
        latencies = [result.retrieval_latency_ms for result in results]
        tokens = [result.tokens_query for result in results]
        modes = sorted({result.mode for result in results})
        failures = [
            {
                "query_id": result.query_id,
                "category": result.category,
                "answer_policy": result.answer_policy,
                "answer": result.answer,
                "retrieved_source_span_ids": result.retrieved_source_span_ids,
            }
            for result in results
            if not result.matched_gold
        ][:20]
        for result in results:
            key = result.category or "uncategorized"
            by_category.setdefault(key, {"matched": 0, "evidence_matched": 0, "total": 0})
            by_category[key]["total"] += 1
            by_category[key]["matched"] += int(result.matched_gold)
            by_category[key]["evidence_matched"] += int(result.evidence_matched_gold)
        return {
            "total": total,
            "matched": matched,
            "evidence_matched": evidence_matched,
            "retrieval_match_rate": evidence_matched / total if total else 0.0,
            "answer_match_rate": matched / total if total else 0.0,
            "raw_evidence_quota_hit_rate": quota_met / total if total else 0.0,
            "abstentions": abstentions,
            "by_category": {
                key: {
                    **value,
                    "rate": value["evidence_matched"] / value["total"] if value["total"] else 0.0,
                    "answer_rate": value["matched"] / value["total"] if value["total"] else 0.0,
                    "evidence_rate": value["evidence_matched"] / value["total"] if value["total"] else 0.0,
                }
                for key, value in by_category.items()
            },
            "latency_ms": _latency_report(latencies),
            "avg_tokens_query": sum(tokens) / total if total else 0.0,
            "llm_calls_query": sum(result.llm_calls for result in results) / total if total else 0.0,
            "answer_model": getattr(self.answer_model, "version", self.answer_model.__class__.__name__),
            "judge_model": getattr(self.judge_model, "version", self.judge_model.__class__.__name__),
            "mode": modes[0] if len(modes) == 1 else "mixed",
            "config": self.service.config.snapshot(),
            "encoding_report": self.service.encoding_report(self.scope),
            "profile_report": self.service.profile_report(self.scope),
            "failure_samples": failures,
        }


def load_dataset(dataset_path: str | Path, split: str | None = None) -> tuple[list[EvalDocument], list[EvalQuery]]:
    path = Path(dataset_path)
    if path.is_dir():
        base = path / split if split and (path / split).is_dir() else path
        documents = _load_records(_first_existing(base, ["documents.jsonl", "docs.jsonl", "documents.json", "docs.json"]))
        queries = _load_records(_first_existing(base, ["queries.jsonl", "questions.jsonl", "queries.json", "questions.json"]))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            documents = data.get("documents") or data.get("docs") or []
            queries = data.get("queries") or data.get("questions") or []
        else:
            raise ValueError("single-file dataset must be a JSON object with documents/docs and queries/questions")
    return [_to_document(item) for item in documents], [_to_query(item) for item in queries]


def _first_existing(base: Path, names: list[str]) -> Path:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"none of {names} found under {base}")


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["documents", "docs", "queries", "questions", "items"]:
            if key in data:
                return list(data[key])
    raise ValueError(f"unsupported dataset file shape: {path}")


def _to_document(item: dict[str, Any]) -> EvalDocument:
    timestamp = item.get("timestamp") or item.get("time") or item.get("created_at")
    return EvalDocument(
        id=str(item.get("id") or item.get("doc_id") or item.get("document_id")),
        content=str(item.get("content") or item.get("text") or ""),
        timestamp=_parse_time(timestamp),
        speaker=str(item.get("speaker") or item.get("role") or "document"),
    )


def _to_query(item: dict[str, Any]) -> EvalQuery:
    gold = item.get("gold_answers") or item.get("answers") or item.get("gold") or []
    if isinstance(gold, str):
        gold = [gold]
    category = item.get("category")
    if not category and isinstance(item.get("meta"), dict):
        category = item["meta"].get("category") or item["meta"].get("query_type")
    return EvalQuery(
        id=str(item.get("id") or item.get("query_id") or item.get("question_id")),
        query=str(item.get("query") or item.get("question") or ""),
        gold_answers=[str(answer) for answer in gold],
        category=category,
    )


def _pack_summary(pack: EvidencePack) -> dict[str, Any]:
    return {
        "answer_policy": pack.answer_policy,
        "coverage": pack.coverage,
        "current_view_count": len(pack.current_views),
        "entity_profile_count": len(pack.entity_profiles),
        "fact_count": len(pack.facts),
        "event_count": len(pack.events),
        "source_span_count": len(pack.source_spans),
        "source_span_ids": [span["id"] for span in pack.source_spans],
    }


def _approx_tokens(query: str, pack: EvidencePack, answer: str) -> int:
    text_parts = [
        query,
        answer,
        str(pack.current_views),
        str(pack.entity_profiles),
        str(pack.facts),
        str(pack.events),
        " ".join(str(span.get("content", "")) for span in pack.source_spans),
    ]
    return sum(len(part.split()) for part in text_parts)


def _latency_report(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(latencies)
    return {"p50": _percentile(ordered, 0.50), "p95": _percentile(ordered, 0.95)}


def _model_call_count(*models: Any) -> int:
    collections: list[Any] = []
    for model in models:
        calls = getattr(model, "calls", None)
        if isinstance(calls, list):
            collections.append(calls)
        client = getattr(model, "client", None)
        client_calls = getattr(client, "calls", None) if client is not None else None
        if isinstance(client_calls, list):
            collections.append(client_calls)
    seen: set[int] = set()
    total = 0
    for calls in collections:
        marker = id(calls)
        if marker in seen:
            continue
        seen.add(marker)
        total += len(calls)
    return total


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
