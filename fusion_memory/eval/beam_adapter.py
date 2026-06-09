from __future__ import annotations

from pathlib import Path
from typing import Any

from fusion_memory import MemoryService, Scope
from fusion_memory.eval.adapter import BenchmarkAdapter, EvalQuery, EvalResult


BEAM_SPLITS = {"small", "dev", "100k", "500k", "1m", "10m"}


class BeamAdapter(BenchmarkAdapter):
    """BEAM-oriented harness built on the generic local benchmark adapter.

    The loader accepts the same JSON/JSONL shapes as `BenchmarkAdapter`, but
    requires a BEAM split label so reports can be compared and replayed by split.
    Production answer/judge models can be injected through the base adapter
    constructor.
    """

    benchmark = "BEAM"

    def __init__(self, service: MemoryService, scope: Scope, split: str = "small", answer_model: Any | None = None, judge_model: Any | None = None) -> None:
        validate_beam_split(split)
        super().__init__(service, scope, answer_model=answer_model, judge_model=judge_model)
        self.split = split

    def ingest_dataset(self, dataset_path: str | Path, split: str | None = None) -> dict[str, Any]:
        effective_split = self._effective_split(split)
        report = super().ingest_dataset(dataset_path, split=effective_split)
        return {"benchmark": self.benchmark, **report, "split": effective_split}

    def build_queries(self, dataset_path: str | Path, split: str | None = None) -> list[EvalQuery]:
        return super().build_queries(dataset_path, split=self._effective_split(split))

    def run_queries(self, queries: list[EvalQuery], budget: dict[str, Any] | None = None) -> list[EvalResult]:
        effective_budget = {"mode": "benchmark", **(budget or {})}
        return super().run_queries(queries, budget=effective_budget)

    def report(self, results: list[EvalResult]) -> dict[str, Any]:
        base = super().report(results)
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            **base,
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
                "components": self.run_component_ablation(queries),
            }
        return output

    def _effective_split(self, split: str | None) -> str:
        effective_split = split or self.split
        validate_beam_split(effective_split)
        self.split = effective_split
        return effective_split


def validate_beam_split(split: str) -> None:
    if split not in BEAM_SPLITS:
        raise ValueError(f"unsupported BEAM split {split!r}; expected one of {sorted(BEAM_SPLITS)}")


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
        "evidence_matched_gold": result.evidence_matched_gold,
        "retrieved_source_span_ids": result.retrieved_source_span_ids,
        "evidence_pack": result.evidence_pack,
        "tokens_query": result.tokens_query,
        "retrieval_latency_ms": result.retrieval_latency_ms,
    }
