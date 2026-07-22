from __future__ import annotations

from typing import Any, Protocol

from fusion_memory.core.models import EvidencePack
from fusion_memory.retrieval.context import (
    ProductQueryPlan,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.tracing import sanitize_dimension


class RetrievalUnavailable(RuntimeError):
    pass


_TRACE_STAGES = ("plan", "recall", "fusion", "selection")
_PROVIDER_KINDS = frozenset({"vector", "lexical", "temporal", "entity", "chronology"})


def sanitize_retrieval_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Keep only the product trace contract, which contains no raw memory text."""
    sanitized: dict[str, Any] = {}
    stages = trace.get("stages")
    if isinstance(stages, list | tuple):
        sanitized["stages"] = [stage for stage in stages if stage in _TRACE_STAGES]
    mode = trace.get("mode")
    if mode in {"fast", "balanced"}:
        sanitized["mode"] = mode
    if "intent" in trace:
        sanitized["intent"] = sanitize_dimension(trace["intent"])

    providers: list[dict[str, Any]] = []
    provider_rows = trace.get("providers")
    if isinstance(provider_rows, list | tuple):
        for row in provider_rows:
            if not isinstance(row, dict):
                continue
            kind = row.get("kind")
            provider: dict[str, Any] = {
                "kind": (
                    kind
                    if isinstance(kind, str) and kind in _PROVIDER_KINDS
                    else sanitize_dimension(kind)
                )
            }
            count = row.get("count")
            if isinstance(count, int) and not isinstance(count, bool):
                provider["count"] = max(0, count)
            elapsed_ms = row.get("elapsed_ms")
            if isinstance(elapsed_ms, int | float) and not isinstance(elapsed_ms, bool):
                provider["elapsed_ms"] = max(0.0, float(elapsed_ms))
            failure_code = row.get("failure_code")
            provider["failure_code"] = (
                None if failure_code is None else sanitize_dimension(failure_code)
            )
            providers.append(provider)
    if providers:
        sanitized["providers"] = providers

    filtered_count = trace.get("filtered_count")
    if isinstance(filtered_count, int) and not isinstance(filtered_count, bool):
        sanitized["filtered_count"] = max(0, filtered_count)
    selected_ids = trace.get("selected_ids")
    if isinstance(selected_ids, list | tuple):
        sanitized["selected_ids"] = [
            value
            for value in selected_ids
            if isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ]
    durations = trace.get("stage_durations_ms")
    if isinstance(durations, dict):
        sanitized["stage_durations_ms"] = {
            stage: max(0.0, float(durations[stage]))
            for stage in _TRACE_STAGES
            if isinstance(durations.get(stage), int | float)
            and not isinstance(durations.get(stage), bool)
        }
    if trace.get("reranker_failure") is not None:
        sanitized["reranker_failure"] = sanitize_dimension(trace["reranker_failure"])
    if trace.get("planner_fallback") is not None:
        fallback = trace["planner_fallback"]
        sanitized["planner_fallback"] = (
            fallback if fallback == "invalid_plan" else sanitize_dimension(fallback)
        )
    return sanitized


class RetrievalEngine(Protocol):
    def search(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan | None = None,
    ) -> RetrievalResult: ...

    def search_with_plan(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan,
    ) -> RetrievalResult: ...

    def build_evidence_pack(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack: ...
