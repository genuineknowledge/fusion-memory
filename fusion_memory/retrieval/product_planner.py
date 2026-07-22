from __future__ import annotations

from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderKind,
    ProviderRequest,
    SearchRequest,
)
from fusion_memory.retrieval.query_intent import QueryIntent, analyze_query_intent


class ProductQueryPlanner:
    """Build product retrieval plans from deterministic query capabilities."""

    def plan(self, request: SearchRequest) -> ProductQueryPlan:
        intent = analyze_query_intent(request.query)
        return ProductQueryPlan(
            intent=_intent_label(intent),
            provider_requests=_provider_requests(intent, request.limit),
            time_range=request.time_range,
            entities=tuple(intent.entities),
            speaker=None if intent.speaker_scope == "any" else intent.speaker_scope,
            ordering=_ordering(intent),
            use_reranker=request.mode == "balanced",
            query_intent=intent.to_dict(),
        )

    def safe_default(self, request: SearchRequest) -> ProductQueryPlan:
        return ProductQueryPlan(
            intent="factual",
            provider_requests=(
                ProviderRequest(ProviderKind.VECTOR, max(request.limit * 2, 12)),
                ProviderRequest(ProviderKind.LEXICAL, max(request.limit * 2, 12)),
            ),
            time_range=request.time_range,
            entities=(),
            speaker=None,
            ordering=OrderingMode.RELEVANCE,
            use_reranker=request.mode == "balanced",
        )


def _intent_label(intent: QueryIntent) -> str:
    if intent.temporal.requires_order:
        return "chronology"
    if intent.needs_current_state:
        return "current_state"
    if intent.needs_conflict_check:
        return "conflict"
    if intent.answer_shape == "summary":
        return "summary"
    if intent.aggregation.operation != "none":
        return "aggregation"
    if intent.answer_shape == "instruction":
        return "instruction"
    if intent.temporal.requires_time:
        return "temporal"
    return "factual"


def _provider_requests(intent: QueryIntent, limit: int) -> tuple[ProviderRequest, ...]:
    kinds = [ProviderKind.VECTOR, ProviderKind.LEXICAL]
    if intent.entities:
        kinds.append(ProviderKind.ENTITY)
    if intent.temporal.requires_time or intent.needs_current_state:
        kinds.append(ProviderKind.TEMPORAL)
    if intent.temporal.requires_order:
        kinds.append(ProviderKind.CHRONOLOGY)
    if not intent.entities:
        kinds.append(ProviderKind.ENTITY)
    return tuple(ProviderRequest(kind, max(limit * 2, 12)) for kind in dict.fromkeys(kinds))


def _ordering(intent: QueryIntent) -> OrderingMode:
    if intent.temporal.requires_order:
        return OrderingMode.CHRONOLOGICAL
    if intent.needs_current_state:
        return OrderingMode.RECENCY
    return OrderingMode.RELEVANCE
