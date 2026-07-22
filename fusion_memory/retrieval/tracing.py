from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fusion_memory.core.models import Candidate
from fusion_memory.core.text import stable_hash
from fusion_memory.retrieval.context import ProductQueryPlan, SearchRequest
from fusion_memory.retrieval.providers.product_base import ProviderOutcome


_SAFE_DIMENSIONS = {
    "aggregation",
    "chronology",
    "conflict",
    "current_state",
    "factual",
    "instruction",
    "model_unavailable",
    "reranker_unavailable",
    "summary",
    "temporal",
}


def validate_product_plan(plan: object) -> bool:
    return isinstance(plan, ProductQueryPlan) and bool(plan.provider_requests)


def build_retrieval_trace(
    context: object,
    request: SearchRequest,
    plan: ProductQueryPlan,
    outcomes: tuple[ProviderOutcome, ...] | list[ProviderOutcome],
    selected: list[Candidate],
    *,
    filtered_count: int = 0,
    stage_durations_ms: Mapping[str, float] | None = None,
    reranker_failure: str | None = None,
) -> dict[str, Any]:
    durations = stage_durations_ms or {}
    trace: dict[str, Any] = {
        "stages": ["plan", "recall", "fusion", "selection"],
        "mode": request.mode,
        "intent": _safe_dimension(plan.intent),
        "providers": [
            {
                "kind": outcome.provider.value,
                "count": len(outcome.candidates),
                "elapsed_ms": round(max(0.0, float(outcome.elapsed_ms)), 3),
                "failure_code": (
                    _safe_dimension(outcome.failure.error_code)
                    if outcome.failure is not None
                    else None
                ),
            }
            for outcome in outcomes
        ],
        "filtered_count": max(0, int(filtered_count)),
        "selected_ids": [stable_hash(candidate.id) for candidate in selected],
        "stage_durations_ms": {
            stage: round(max(0.0, float(durations.get(stage, 0.0))), 3)
            for stage in ("plan", "recall", "fusion", "selection")
        },
    }
    if reranker_failure is not None:
        trace["reranker_failure"] = reranker_failure
    return trace


def _safe_dimension(value: object) -> str:
    label = str(value)
    if label in _SAFE_DIMENSIONS:
        return label
    return f"hashed_{stable_hash(label)[:16]}"
