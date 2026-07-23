from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from fusion_memory import MemoryService, Scope
from fusion_memory.eval.adapter import (
    BenchmarkAdapter,
    EvalDocument,
    EvalQuery,
    EvalResult,
    _approx_tokens,
    _latency_report,
    _model_call_count,
    _pack_summary,
)
from fusion_memory.eval.beam.engine import BeamRetrievalEngine
from fusion_memory.eval.beam.model_adapters import as_beam_answer_model


BEAM_SPLITS = {"small", "dev", "100k", "500k", "1m", "10m"}
BEAM_SPLIT_DIRS = {
    "small": "100K",
    "dev": "100K",
    "100k": "100K",
    "500k": "500K",
    "1m": "1M",
    "10m": "10M",
}


class BeamAdapter(BenchmarkAdapter):
    """BEAM-oriented harness built on the generic local benchmark adapter.

    The loader accepts the same JSON/JSONL shapes as `BenchmarkAdapter`, but
    requires a BEAM split label so reports can be compared and replayed by split.
    Production answer/judge models can be injected through the base adapter
    constructor.
    """

    benchmark = "BEAM"

    def __init__(
        self,
        service: MemoryService,
        scope: Scope,
        split: str = "small",
        answer_model: Any | None = None,
        judge_model: Any | None = None,
        retrieval_engine: Any | None = None,
    ) -> None:
        validate_beam_split(split)
        super().__init__(
            service,
            scope,
            answer_model=as_beam_answer_model(answer_model),
            judge_model=judge_model,
        )
        self.split = split
        self.retrieval_engine = (
            retrieval_engine
            if retrieval_engine is not None
            else BeamRetrievalEngine.from_service(service)
        )

    def ingest_dataset(self, dataset_path: str | Path, split: str | None = None) -> dict[str, Any]:
        effective_split = self._effective_split(split)
        official = _load_official_beam_dataset(dataset_path, effective_split)
        if official:
            documents, _ = official
            span_ids = self.ingest_documents(documents)
            report = {"documents": len(documents), "span_ids": span_ids, "span_count": len(span_ids), "split": effective_split}
        else:
            report = super().ingest_dataset(dataset_path, split=effective_split)
        return {"benchmark": self.benchmark, **report, "split": effective_split}

    def build_queries(self, dataset_path: str | Path, split: str | None = None) -> list[EvalQuery]:
        effective_split = self._effective_split(split)
        official = _load_official_beam_dataset(dataset_path, effective_split)
        if official:
            _, queries = official
            return queries
        return super().build_queries(dataset_path, split=effective_split)

    def answer_query(self, query: EvalQuery, budget: dict[str, Any] | None = None) -> EvalResult:
        budget = {"mode": "balanced", **(budget or {})}
        started = perf_counter()
        query_scope = self._beam_scope(query.id)
        pack = self.retrieval_engine.answer_context(
            query.query,
            query_scope,
            query.category,
            budget=budget,
        )
        latency_ms = (perf_counter() - started) * 1000
        model_call_mark = _model_call_count(self.answer_model, self.judge_model)
        answer_failed = False
        try:
            answer = _beam_answer(self.answer_model, query, pack)
            score, judge_reason = _beam_score(query, answer, self.judge_model)
        except Exception as exc:
            answer_failed = True
            answer = ""
            score = 0.0
            judge_reason = f"answer generation failed: {_safe_error(exc)}"
        llm_calls = _model_call_count(self.answer_model, self.judge_model) - model_call_mark
        return EvalResult(
            query_id=query.id,
            answer_policy=pack.answer_policy,
            retrieved_source_span_ids=[span["id"] for span in pack.source_spans],
            matched_gold=score >= 0.5,
            category=query.category,
            query_type=pack.coverage.get("query_type"),
            source_span_quota_met=pack.coverage.get("source_span_quota_met"),
            coverage_insufficient=pack.coverage.get("coverage_insufficient"),
            query_text=query.query,
            answer=answer,
            evidence_pack=_pack_summary(pack),
            evidence_matched_gold=False,
            answer_model=getattr(self.answer_model, "version", self.answer_model.__class__.__name__),
            judge_model=getattr(self.judge_model, "version", self.judge_model.__class__.__name__),
            mode=str(budget.get("mode", "balanced")),
            tokens_query=_approx_tokens(query.query, pack, answer),
            retrieval_latency_ms=latency_ms,
            llm_calls=llm_calls,
            score=score,
            judge_reason=judge_reason,
            answer_failed=answer_failed,
        )

    def ingest_documents(self, documents: list[EvalDocument]) -> list[str]:
        span_ids: list[str] = []
        for doc in documents:
            session_time = doc.timestamp or datetime.now(timezone.utc)
            result = self.service.add(
                {
                    "role": doc.speaker,
                    "content": doc.content,
                    "turn_id": doc.id,
                    "timestamp": session_time.isoformat(),
                },
                self._beam_scope(doc.id),
                session_time,
                {"source_uri": doc.id},
            )
            span_ids.extend(result.span_ids)
        return span_ids

    def run_queries(self, queries: list[EvalQuery], budget: dict[str, Any] | None = None) -> list[EvalResult]:
        return [self.answer_query(query, budget=budget) for query in queries]

    def report(self, results: list[EvalResult]) -> dict[str, Any]:
        total = len(results)
        scores = [float(result.score or 0.0) for result in results]
        matched = sum(1 for result in results if result.matched_gold)
        quota_met = sum(1 for result in results if result.source_span_quota_met)
        latencies = [result.retrieval_latency_ms for result in results]
        tokens = [result.tokens_query for result in results]
        by_category = _beam_category_report(results)
        modes = sorted({result.mode for result in results})
        judge_failures = [result for result in results if _judge_failed(result.judge_reason)]
        answer_failures = [result for result in results if result.answer_failed]
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            "scoring": {
                "method": "beam_official_rubric_v1",
                "primary_metric": "accuracy",
                "accuracy_definition": "mean continuous BEAM score across queries",
                "non_ordering": "mean of rubric item scores in {0, 0.5, 1}",
                "event_ordering": "normalized Kendall tau over ordered items",
            },
            "total": total,
            "correct": matched,
            "accuracy": sum(scores) / total if total else 0.0,
            "answer_match_rate": matched / total if total else 0.0,
            "raw_evidence_quota_hit_rate": quota_met / total if total else 0.0,
            "by_category": by_category,
            "latency_ms": _latency_report(latencies),
            "avg_tokens_query": sum(tokens) / total if total else 0.0,
            "llm_calls_query": sum(result.llm_calls for result in results) / total if total else 0.0,
            "answer_model": getattr(self.answer_model, "version", self.answer_model.__class__.__name__),
            "judge_model": getattr(self.judge_model, "version", self.judge_model.__class__.__name__),
            "mode": modes[0] if len(modes) == 1 else "mixed",
            "config": self.service.config.snapshot(),
            "encoding_report": self.service.encoding_report(self.scope),
            "profile_report": self.service.profile_report(self.scope),
            "failure_samples": _beam_failure_samples(results),
            "judge_failures": {
                "count": len(judge_failures),
                "rate": len(judge_failures) / total if total else 0.0,
                "samples": _judge_failure_samples(judge_failures),
            },
            "answer_failures": {
                "count": len(answer_failures),
                "rate": len(answer_failures) / total if total else 0.0,
                "samples": _judge_failure_samples(answer_failures),
            },
            "query_type_mapping": _query_type_mapping(results),
            "evidence_pack_trace_coverage": _evidence_pack_trace_coverage(results),
            "answers": [_answer_record(result) for result in results],
        }

    def run_dataset(self, dataset_path: str | Path, *, split: str | None = None, ablate: bool = False) -> dict[str, Any]:
        effective_split = self._effective_split(split)
        ingest = self.ingest_dataset(dataset_path, split=effective_split)
        queries = self.build_queries(dataset_path, split=effective_split)
        results = self.run_queries(queries)
        output: dict[str, Any] = {"ingest": ingest, "report": self.report(results)}
        if ablate:
            output["ablation"] = {
                "retrieval_modes": self.run_ablation(queries),
            }
        return output

    def _effective_split(self, split: str | None) -> str:
        effective_split = split or self.split
        validate_beam_split(effective_split)
        self.split = effective_split
        return effective_split

    def _beam_scope(self, item_id: str) -> Scope:
        session_id = _beam_session_id_from_id(item_id)
        return Scope(
            workspace_id=self.scope.workspace_id,
            user_id=self.scope.user_id,
            agent_id=self.scope.agent_id,
            run_id=self.scope.run_id,
            session_id=session_id or self.scope.session_id,
            app_id=self.scope.app_id,
        )


def validate_beam_split(split: str) -> None:
    if split not in BEAM_SPLITS:
        raise ValueError(f"unsupported BEAM split {split!r}; expected one of {sorted(BEAM_SPLITS)}")


def _load_official_beam_dataset(dataset_path: str | Path, split: str) -> tuple[list[EvalDocument], list[EvalQuery]] | None:
    root = Path(dataset_path)
    chats_root = root / "chats"
    if not chats_root.is_dir():
        raise ValueError(f"BEAM dataset must use the official layout with a chats/ directory: {root}")
    split_dir = chats_root / BEAM_SPLIT_DIRS[split]
    if not split_dir.is_dir():
        raise ValueError(f"BEAM split directory not found: {split_dir}")
    chat_dirs = _chat_dirs(split_dir)
    if split == "small":
        chat_dirs = chat_dirs[:1]
    elif split == "dev":
        chat_dirs = chat_dirs[:3]
    documents: list[EvalDocument] = []
    queries: list[EvalQuery] = []
    for chat_dir in chat_dirs:
        documents.extend(_load_chat_documents(chat_dir, split))
        queries.extend(_load_chat_queries(chat_dir, split))
    return documents, queries


def _beam_session_id_from_id(item_id: str | None) -> str | None:
    if not item_id:
        return None
    match = re.match(r"^(beam:[^:]+:\d+):", str(item_id))
    return match.group(1) if match else None


def _chat_dirs(split_dir: Path) -> list[Path]:
    return sorted(
        [path for path in split_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def _load_chat_documents(chat_dir: Path, split: str) -> list[EvalDocument]:
    chat_path = chat_dir / "chat.json"
    if not chat_path.exists():
        return []
    data = json.loads(chat_path.read_text(encoding="utf-8"))
    documents: list[EvalDocument] = []
    for batch_index, batch in enumerate(data if isinstance(data, list) else []):
        batch_number = batch.get("batch_number", batch_index + 1) if isinstance(batch, dict) else batch_index + 1
        turns = batch.get("turns", []) if isinstance(batch, dict) else []
        for turn_index, turn in enumerate(turns):
            messages = turn if isinstance(turn, list) else [turn]
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue
                content = str(message.get("content") or "").strip()
                if not content:
                    continue
                message_id = message.get("id", f"{batch_number}-{turn_index}-{message_index}")
                documents.append(
                    EvalDocument(
                        id=f"beam:{split}:{chat_dir.name}:batch{batch_number}:msg{message_id}",
                        content=content,
                        timestamp=_parse_beam_time(message.get("time_anchor")),
                        speaker=str(message.get("role") or "document"),
                    )
                )
    return documents


def _load_chat_queries(chat_dir: Path, split: str) -> list[EvalQuery]:
    questions_path = chat_dir / "probing_questions" / "probing_questions.json"
    if not questions_path.exists():
        return []
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    queries: list[EvalQuery] = []
    if not isinstance(data, dict):
        return queries
    for category, items in data.items():
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            gold_answers = _beam_gold_answers(item)
            queries.append(
                EvalQuery(
                    id=f"beam:{split}:{chat_dir.name}:{category}:{index}",
                    query=question,
                    gold_answers=gold_answers,
                    category=str(category),
                    metadata=_beam_query_metadata(item),
                )
            )
    return queries


def _beam_gold_answers(item: dict[str, Any]) -> list[str]:
    gold = (
        item.get("gold_answers")
        or item.get("answers")
        or item.get("answer")
        or item.get("ideal_answer")
        or item.get("ideal_response")
        or item.get("expected_compliance")
        or item.get("rubric")
        or []
    )
    if isinstance(gold, str):
        return [gold] if gold.strip() else []
    if isinstance(gold, list):
        return [str(answer) for answer in gold if str(answer).strip()]
    if gold:
        return [str(gold)]
    return []


def _beam_query_metadata(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rubric",
        "difficulty",
        "source_chat_ids",
        "conversation_references",
        "plan_reference",
        "answer",
        "ideal_response",
        "expected_compliance",
        "compliance_indicators",
        "non_compliance_signs",
        "calculation_required",
        "ordering_tested",
        "total_mentions",
        "time_points",
    ]
    return {key: item[key] for key in keys if key in item}


def _parse_beam_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _query_type_mapping(results: list[EvalResult]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        category = result.category or "uncategorized"
        entry = by_category.setdefault(category, {"query_type": result.query_type, "count": 0, "query_ids": []})
        entry["count"] += 1
        entry["query_ids"].append(result.query_id)
    return by_category


def _evidence_pack_trace_coverage(results: list[EvalResult]) -> float:
    if not results:
        return 0.0
    traced = sum(1 for result in results if result.evidence_pack and "source_span_ids" in result.evidence_pack)
    return traced / len(results)


def _answer_record(result: EvalResult) -> dict[str, Any]:
    return {
        "query_id": result.query_id,
        "query_text": result.query_text,
        "category": result.category,
        "query_type": result.query_type,
        "answer": result.answer,
        "answer_policy": result.answer_policy,
        "matched_gold": result.matched_gold,
        "score": result.score,
        "judge_reason": result.judge_reason,
        "judge_failed": _judge_failed(result.judge_reason),
        "answer_failed": result.answer_failed,
        "source_span_quota_met": result.source_span_quota_met,
        "coverage_insufficient": result.coverage_insufficient,
        "retrieved_source_span_ids": result.retrieved_source_span_ids,
        "evidence_pack": result.evidence_pack,
        "evidence_matched_gold": result.evidence_matched_gold,
        "answer_model": result.answer_model,
        "judge_model": result.judge_model,
        "mode": result.mode,
        "tokens_query": result.tokens_query,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "llm_calls": result.llm_calls,
    }


def _beam_score(query: EvalQuery, answer: str, judge_model: Any) -> tuple[float, str]:
    if query.category == "event_ordering":
        reference = _event_order_reference(query)
        system = _extract_ordered_items(answer)
        score = _event_ordering_score(reference, system)
        return score, f"event_ordering_tau_norm={score:.3f}"
    rubric = _rubric_items(query)
    if not rubric:
        return 0.0, "no rubric or gold answer available"
    item_scores: list[float] = []
    reasons: list[str] = []
    for item in rubric:
        score, reason = _rubric_item_score(query.query, answer, item, judge_model)
        item_scores.append(score)
        reasons.append(f"{score:.1f}:{reason}")
    avg = sum(item_scores) / len(item_scores) if item_scores else 0.0
    return avg, "; ".join(reasons[:5])


def _rubric_items(query: EvalQuery) -> list[str]:
    rubric = query.metadata.get("rubric")
    if isinstance(rubric, str) and rubric.strip():
        return [rubric]
    if isinstance(rubric, list):
        return [str(item) for item in rubric if str(item).strip()]
    return [f"LLM response should contain: {answer}" for answer in query.gold_answers if answer.strip()]


def _beam_answer(answer_model: Any, query: EvalQuery, pack: Any) -> str:
    contextual_answer = getattr(answer_model, "answer_with_context", None)
    if callable(contextual_answer):
        return contextual_answer(
            query.query,
            pack,
            benchmark="BEAM",
            category=query.category,
            metadata={},
        )
    return answer_model.answer(query.query, pack)


def _rubric_item_score(query: str, answer: str, rubric_item: str, judge_model: Any) -> tuple[float, str]:
    scorer = getattr(judge_model, "rubric_score", None)
    if callable(scorer):
        return scorer(query, answer, rubric_item)
    needle = _rubric_needle(rubric_item)
    if not needle:
        return 0.0, "empty rubric item"
    answer_lower = answer.lower()
    needle_lower = needle.lower()
    if needle_lower in answer_lower:
        return 1.0, "exact rubric phrase found"
    tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", needle_lower) if len(token) > 2]
    if not tokens:
        return 0.0, "no comparable rubric tokens"
    hits = sum(1 for token in tokens if token in answer_lower)
    ratio = hits / len(tokens)
    if ratio >= 0.75:
        return 1.0, f"token overlap {ratio:.2f}"
    if ratio >= 0.35:
        return 0.5, f"partial token overlap {ratio:.2f}"
    return 0.0, f"low token overlap {ratio:.2f}"


def _rubric_needle(item: str) -> str:
    return item.split(": ", 1)[-1].strip() if ": " in item else item.strip()


def _event_order_reference(query: EvalQuery) -> list[str]:
    ordering = query.metadata.get("ordering_tested")
    if isinstance(ordering, list):
        return [str(item) for item in ordering if str(item).strip()]
    if query.gold_answers:
        return _extract_ordered_items(query.gold_answers[0])
    return []


def _extract_ordered_items(text: str) -> list[str]:
    items: list[str] = []
    for line in text.strip().splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", line).strip()
        if cleaned:
            items.append(cleaned)
    if len(items) <= 1:
        matches = re.findall(r"(?:^|[,;])\s*(?:\d+[.)]\s*)?([^,;]+)", text)
        items = [match.strip() for match in matches if match.strip()]
    return items


def _event_ordering_score(reference: list[str], system: list[str]) -> float:
    if not reference or not system:
        return 0.0
    reference_norm = [_normalize_order_item(item) for item in reference]
    system_norm = _align_order_items(reference_norm, [_normalize_order_item(item) for item in system])
    union = list(dict.fromkeys(reference_norm + system_norm))
    tie_rank = len(union) + 1

    def to_rank(sequence: list[str]) -> list[int]:
        ranks = {item: index + 1 for index, item in enumerate(sequence)}
        return [ranks.get(item, tie_rank) for item in union]

    tau = _kendall_tau_b(to_rank(reference_norm), to_rank(system_norm))
    return (tau + 1.0) / 2.0


def _normalize_order_item(value: str) -> str:
    value = re.sub(r"^\s*(?:\d+(?:st|nd|rd|th)?|[a-z])\s*[:.)-]\s*", "", value.strip(), flags=re.I)
    if ":" in value:
        label, rest = value.split(":", 1)
        if 0 < len(re.findall(r"[a-zA-Z0-9]+", label)) <= 6 and rest.strip():
            value = label
    return " ".join(re.findall(r"[a-zA-Z0-9]+", value.lower()))


def _align_order_items(reference_norm: list[str], system_norm: list[str]) -> list[str]:
    aligned: list[str] = []
    used: set[str] = set()
    for item in system_norm:
        match = _best_reference_match(item, [reference for reference in reference_norm if reference not in used])
        if match:
            aligned.append(match)
            used.add(match)
        else:
            aligned.append(item)
    return aligned


def _best_reference_match(item: str, references: list[str]) -> str | None:
    item_tokens = set(item.split())
    if not item_tokens:
        return None
    best: tuple[float, str] | None = None
    for reference in references:
        if item == reference or item.startswith(reference):
            return reference
        reference_tokens = set(reference.split())
        if not reference_tokens:
            continue
        coverage = len(reference_tokens & item_tokens) / len(reference_tokens)
        precision = len(reference_tokens & item_tokens) / len(item_tokens)
        score = (coverage * 0.75) + (precision * 0.25)
        if coverage >= 0.80 and (best is None or score > best[0]):
            best = (score, reference)
    return best[1] if best else None


def _kendall_tau_b(left: list[int], right: list[int]) -> float:
    concordant = discordant = ties_left = ties_right = 0
    n = len(left)
    for i in range(n):
        for j in range(i + 1, n):
            dx = (left[i] > left[j]) - (left[i] < left[j])
            dy = (right[i] > right[j]) - (right[i] < right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom_left = concordant + discordant + ties_left
    denom_right = concordant + discordant + ties_right
    denominator = (denom_left * denom_right) ** 0.5
    if denominator == 0:
        return 0.0
    return (concordant - discordant) / denominator


def _beam_category_report(results: list[EvalResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for result in results:
        key = result.category or "uncategorized"
        entry = out.setdefault(key, {"total": 0, "correct": 0, "score_sum": 0.0})
        entry["total"] += 1
        entry["correct"] += int(result.matched_gold)
        entry["score_sum"] += float(result.score or 0.0)
    return {
        key: {
            "total": value["total"],
            "correct": value["correct"],
            "accuracy": value["score_sum"] / value["total"] if value["total"] else 0.0,
            "answer_match_rate": value["correct"] / value["total"] if value["total"] else 0.0,
        }
        for key, value in out.items()
    }


def _beam_failure_samples(results: list[EvalResult]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": result.query_id,
            "category": result.category,
            "score": result.score,
            "judge_reason": result.judge_reason,
            "answer": result.answer,
            "retrieved_source_span_ids": result.retrieved_source_span_ids,
        }
        for result in results
        if not result.matched_gold
    ][:20]


def _judge_failed(reason: str) -> bool:
    return "rubric scoring failed" in reason.lower()


def _safe_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    text = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "sk-...", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1...", text, flags=re.I)
    return text[:300] + ("..." if len(text) > 300 else "")


def _judge_failure_samples(results: list[EvalResult]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": result.query_id,
            "category": result.category,
            "judge_reason": result.judge_reason,
            "answer": result.answer,
        }
        for result in results[:20]
    ]
