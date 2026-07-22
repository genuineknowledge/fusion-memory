from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from fusion_memory import MemoryService, Scope
from fusion_memory.eval.adapter import BenchmarkAdapter, EvalResult, _approx_tokens, _latency_report, _model_call_count, _pack_summary, _parse_time


LONGMEMEVAL_SPLITS = {"dev", "test", "longmemeval_s", "longmemeval_m", "longmemeval_oracle"}


@dataclass
class LongMemEvalSession:
    session_id: str
    date: datetime | None
    messages: list[dict[str, Any]]


@dataclass
class LongMemEvalItem:
    question_id: str
    question: str
    answer: str
    question_type: str
    question_date: datetime | None
    haystack_sessions: list[LongMemEvalSession]
    haystack_session_ids: list[str] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)


@dataclass
class LongMemEvalResult:
    item: LongMemEvalItem
    eval_result: EvalResult
    retrieved_session_ids: list[str]
    answer_session_hit: bool
    answer_session_recall: float


class LongMemEvalAdapter(BenchmarkAdapter):
    """Local LongMemEval harness.

    Each LongMemEval record is isolated under `run_id=question_id`, because
    every question ships with its own haystack sessions. Session ids are kept on
    ingestion so reports can measure whether retrieved evidence came from the
    annotated answer sessions.
    """

    benchmark = "LongMemEval"

    def __init__(
        self,
        service: MemoryService,
        scope: Scope,
        split: str = "dev",
        answer_model: Any | None = None,
        judge_model: Any | None = None,
    ) -> None:
        validate_longmemeval_split(split)
        super().__init__(service, scope, answer_model=answer_model, judge_model=judge_model)
        self.split = split
        self._ingested_question_ids: set[str] = set()

    def load_items(self, dataset_path: str | Path, split: str | None = None) -> list[LongMemEvalItem]:
        return load_longmemeval_dataset(dataset_path, split=self._effective_split(split))

    def run_dataset(self, dataset_path: str | Path, *, split: str | None = None, ablate: bool = False) -> dict[str, Any]:
        effective_split = self._effective_split(split)
        items = self.load_items(dataset_path, split=effective_split)
        results = self.run_items(items)
        output: dict[str, Any] = {
            "ingest": {
                "benchmark": self.benchmark,
                "split": effective_split,
                "questions": len(items),
                "haystack_sessions": sum(len(item.haystack_sessions) for item in items),
            },
            "report": self.report(results),
        }
        if ablate:
            output["ablation"] = {
                "retrieval_modes": self.run_ablation(items),
                "components": self.run_component_ablation(items),
            }
        return output

    def run_items(self, items: list[LongMemEvalItem], budget: dict[str, Any] | None = None) -> list[LongMemEvalResult]:
        return [self.answer_item(item, budget=budget) for item in items]

    def answer_item(self, item: LongMemEvalItem, budget: dict[str, Any] | None = None) -> LongMemEvalResult:
        budget = {"mode": "balanced", "allow_cross_session": True, **(budget or {})}
        query_scope = self._question_scope(item.question_id)
        self._ingest_item(item)
        started = perf_counter()
        pack = self.service.answer_context(item.question, query_scope, budget=budget)
        latency_ms = (perf_counter() - started) * 1000
        model_call_mark = _model_call_count(self.answer_model, self.judge_model)
        answer = self.answer_model.answer(item.question, pack)
        gold_answers = [item.answer] if item.answer else []
        evidence_blob = " ".join(span["content"] for span in pack.source_spans)
        evidence_blob += " " + str(pack.facts) + " " + str(pack.current_views) + " " + str(pack.events)
        evidence_matched = any(gold.lower() in evidence_blob.lower() for gold in gold_answers)
        answer_matched = _is_abstention_item(item) and pack.answer_policy == "abstain_if_not_supported"
        if not answer_matched:
            answer_matched = self.judge_model.score(answer, gold_answers)
        llm_calls = _model_call_count(self.answer_model, self.judge_model) - model_call_mark
        retrieved_session_ids = _retrieved_session_ids(pack.source_spans)
        answer_session_recall = _session_recall(retrieved_session_ids, item.answer_session_ids)
        result = EvalResult(
            query_id=item.question_id,
            answer_policy=pack.answer_policy,
            retrieved_source_span_ids=[span["id"] for span in pack.source_spans],
            matched_gold=answer_matched,
            category=item.question_type,
            query_type=pack.coverage.get("query_type"),
            source_span_quota_met=pack.coverage.get("source_span_quota_met"),
            coverage_insufficient=pack.coverage.get("coverage_insufficient"),
            query_text=item.question,
            answer=answer,
            evidence_pack=_pack_summary(pack),
            evidence_matched_gold=evidence_matched,
            answer_model=getattr(self.answer_model, "version", self.answer_model.__class__.__name__),
            judge_model=getattr(self.judge_model, "version", self.judge_model.__class__.__name__),
            mode=str(budget.get("mode", "balanced")),
            tokens_query=_approx_tokens(item.question, pack, answer),
            retrieval_latency_ms=latency_ms,
            llm_calls=llm_calls,
        )
        return LongMemEvalResult(
            item=item,
            eval_result=result,
            retrieved_session_ids=retrieved_session_ids,
            answer_session_hit=bool(set(retrieved_session_ids).intersection(item.answer_session_ids)),
            answer_session_recall=answer_session_recall,
        )

    def run_ablation(self, items: list[LongMemEvalItem], modes: list[str] | None = None) -> dict[str, Any]:
        modes = modes or ["fast", "balanced"]
        if any(mode not in {"fast", "balanced"} for mode in modes):
            raise ValueError("eval retrieval mode must be fast or balanced")
        return {mode: self.report(self.run_items(items, budget={"mode": mode})) for mode in modes}

    def run_component_ablation(self, items: list[LongMemEvalItem]) -> dict[str, Any]:
        source_sets = {
            "L0": ["raw", "exact", "entities"],
            "L0+L1": ["raw", "exact", "entities", "facts"],
            "L0+L1+L2": ["raw", "exact", "entities", "facts", "events"],
            "Full": ["raw", "exact", "entities", "facts", "events", "views", "profiles"],
        }
        return {
            name: self.report(self.run_items(items, budget={"enabled_sources": enabled_sources}))
            for name, enabled_sources in source_sets.items()
        }

    def report(self, results: list[LongMemEvalResult]) -> dict[str, Any]:  # type: ignore[override]
        eval_results = [result.eval_result for result in results]
        base = super().report(eval_results)
        total = len(results)
        answer_session_hits = sum(1 for result in results if result.answer_session_hit)
        abstention_items = [result for result in results if _is_abstention_item(result.item)]
        abstention_correct = sum(1 for result in abstention_items if result.eval_result.answer_policy == "abstain_if_not_supported")
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            **base,
            "answer_session_hit_rate": answer_session_hits / total if total else 0.0,
            "answer_session_recall": sum(result.answer_session_recall for result in results) / total if total else 0.0,
            "abstention_accuracy": abstention_correct / len(abstention_items) if abstention_items else None,
            "question_type_mapping": _question_type_mapping(results),
            "answers": [_answer_record(result) for result in results],
        }

    def _ingest_item(self, item: LongMemEvalItem) -> None:
        if item.question_id in self._ingested_question_ids:
            return
        for session in item.haystack_sessions:
            scope = self._question_scope(item.question_id, session_id=session.session_id)
            session_time = session.date or item.question_date or datetime.now(timezone.utc)
            self.service.add(
                {"messages": session.messages},
                scope,
                session_time,
                {"source_uri": session.session_id, "longmemeval_question_id": item.question_id},
            )
        self._ingested_question_ids.add(item.question_id)

    def _question_scope(self, question_id: str, session_id: str | None = None) -> Scope:
        return Scope(
            workspace_id=self.scope.workspace_id,
            user_id=self.scope.user_id,
            agent_id=self.scope.agent_id,
            run_id=question_id,
            session_id=session_id,
            app_id=self.scope.app_id,
        )

    def _effective_split(self, split: str | None) -> str:
        effective_split = split or self.split
        validate_longmemeval_split(effective_split)
        self.split = effective_split
        return effective_split


def load_longmemeval_dataset(dataset_path: str | Path, split: str | None = None) -> list[LongMemEvalItem]:
    path = Path(dataset_path)
    records = _load_longmemeval_records(path, split=split)
    return [_to_longmemeval_item(record, index) for index, record in enumerate(records)]


def validate_longmemeval_split(split: str) -> None:
    if split not in LONGMEMEVAL_SPLITS:
        raise ValueError(f"unsupported LongMemEval split {split!r}; expected one of {sorted(LONGMEMEVAL_SPLITS)}")


def _load_longmemeval_records(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    if path.is_dir():
        candidates = []
        if split:
            candidates.extend(
                [
                    path / f"{split}.jsonl",
                    path / f"{split}.json",
                    path / split / "data.jsonl",
                    path / split / "data.json",
                    path / split / "questions.jsonl",
                    path / split / "questions.json",
                ]
            )
        candidates.extend([path / "data.jsonl", path / "data.json", path / "longmemeval.jsonl", path / "longmemeval.json"])
        for candidate in candidates:
            if candidate.exists():
                return _read_records(candidate)
        raise FileNotFoundError(f"no LongMemEval JSON/JSONL file found under {path}")
    return _read_records(path)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["items", "data", "questions", "examples"]:
            if isinstance(data.get(key), list):
                return list(data[key])
    raise ValueError(f"unsupported LongMemEval file shape: {path}")


def _to_longmemeval_item(record: dict[str, Any], index: int) -> LongMemEvalItem:
    question_id = str(record.get("question_id") or record.get("id") or f"longmem_{index}")
    question_type = str(record.get("question_type") or record.get("category") or "unknown")
    answer = record.get("answer") or record.get("gold_answer") or record.get("gold") or ""
    answer_session_ids = _string_list(record.get("answer_session_ids") or record.get("answer_session_id") or [])
    haystack_session_ids = _string_list(record.get("haystack_session_ids") or [])
    haystack_dates = _list(record.get("haystack_dates") or record.get("session_dates") or [])
    sessions_raw = _list(record.get("haystack_sessions") or record.get("sessions") or record.get("haystack") or [])
    sessions: list[LongMemEvalSession] = []
    for session_index, raw_session in enumerate(sessions_raw):
        session_id = _session_id(raw_session, haystack_session_ids, session_index)
        session_date = _session_date(raw_session, haystack_dates, session_index)
        sessions.append(
            LongMemEvalSession(
                session_id=session_id,
                date=session_date,
                messages=_session_messages(raw_session, session_id=session_id, timestamp=session_date),
            )
        )
    return LongMemEvalItem(
        question_id=question_id,
        question=str(record.get("question") or record.get("query") or ""),
        answer=str(answer),
        question_type=question_type,
        question_date=_parse_time(record.get("question_date") or record.get("query_date")),
        haystack_sessions=sessions,
        haystack_session_ids=haystack_session_ids or [session.session_id for session in sessions],
        answer_session_ids=answer_session_ids,
    )


def _session_id(raw_session: Any, haystack_session_ids: list[str], index: int) -> str:
    if index < len(haystack_session_ids):
        return haystack_session_ids[index]
    if isinstance(raw_session, dict):
        value = raw_session.get("session_id") or raw_session.get("id") or raw_session.get("conversation_id")
        if value:
            return str(value)
    return f"session_{index}"


def _session_date(raw_session: Any, haystack_dates: list[Any], index: int) -> datetime | None:
    if isinstance(raw_session, dict):
        value = raw_session.get("date") or raw_session.get("timestamp") or raw_session.get("created_at")
        if value:
            return _parse_time(value)
    if index < len(haystack_dates):
        return _parse_time(haystack_dates[index])
    return None


def _session_messages(raw_session: Any, *, session_id: str, timestamp: datetime | None) -> list[dict[str, Any]]:
    if isinstance(raw_session, str):
        return [{"role": "user", "content": raw_session, "turn_id": f"{session_id}_0", "timestamp": _ts(timestamp)}]
    if isinstance(raw_session, list):
        return [_message(item, session_id=session_id, index=index, timestamp=timestamp) for index, item in enumerate(raw_session)]
    if isinstance(raw_session, dict):
        for key in ["messages", "conversation", "turns"]:
            if isinstance(raw_session.get(key), list):
                return [_message(item, session_id=session_id, index=index, timestamp=timestamp) for index, item in enumerate(raw_session[key])]
        content = raw_session.get("content") or raw_session.get("text") or raw_session.get("summary") or ""
        return [{"role": raw_session.get("role") or raw_session.get("speaker") or "user", "content": str(content), "turn_id": f"{session_id}_0", "timestamp": _ts(timestamp)}]
    return [{"role": "user", "content": str(raw_session), "turn_id": f"{session_id}_0", "timestamp": _ts(timestamp)}]


def _message(item: Any, *, session_id: str, index: int, timestamp: datetime | None) -> dict[str, Any]:
    if isinstance(item, str):
        return {"role": "user", "content": item, "turn_id": f"{session_id}_{index}", "timestamp": _ts(timestamp)}
    if isinstance(item, dict):
        role = item.get("role") or item.get("speaker") or "user"
        content = item.get("content") or item.get("text") or item.get("message") or ""
        turn_id = item.get("turn_id") or item.get("id") or f"{session_id}_{index}"
        ts = item.get("timestamp") or item.get("time") or _ts(timestamp)
        return {"role": role, "content": str(content), "turn_id": str(turn_id), "timestamp": ts}
    return {"role": "user", "content": str(item), "turn_id": f"{session_id}_{index}", "timestamp": _ts(timestamp)}


def _retrieved_session_ids(source_spans: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for span in source_spans:
        session_id = span.get("session_id") or span.get("source_uri")
        if session_id:
            values.append(str(session_id))
    return list(dict.fromkeys(values))


def _session_recall(retrieved_session_ids: list[str], answer_session_ids: list[str]) -> float:
    if not answer_session_ids:
        return 0.0
    return len(set(retrieved_session_ids).intersection(answer_session_ids)) / len(set(answer_session_ids))


def _question_type_mapping(results: list[LongMemEvalResult]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for result in results:
        key = result.item.question_type
        entry = mapping.setdefault(key, {"query_type": result.eval_result.query_type, "count": 0, "question_ids": []})
        entry["count"] += 1
        entry["question_ids"].append(result.item.question_id)
    return mapping


def _answer_record(result: LongMemEvalResult) -> dict[str, Any]:
    return {
        "question_id": result.item.question_id,
        "question": result.item.question,
        "question_type": result.item.question_type,
        "answer": result.eval_result.answer,
        "gold_answer": result.item.answer,
        "answer_policy": result.eval_result.answer_policy,
        "matched_gold": result.eval_result.matched_gold,
        "evidence_matched_gold": result.eval_result.evidence_matched_gold,
        "answer_session_ids": result.item.answer_session_ids,
        "retrieved_session_ids": result.retrieved_session_ids,
        "answer_session_hit": result.answer_session_hit,
        "answer_session_recall": result.answer_session_recall,
        "retrieved_source_span_ids": result.eval_result.retrieved_source_span_ids,
        "evidence_pack": result.eval_result.evidence_pack,
        "tokens_query": result.eval_result.tokens_query,
        "retrieval_latency_ms": result.eval_result.retrieval_latency_ms,
    }


def _is_abstention_item(item: LongMemEvalItem) -> bool:
    return item.question_type.endswith("_abs")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _ts(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
