from __future__ import annotations

from dataclasses import replace

from fusion_memory.retrieval.context import (
    ProductQueryPlan,
    ProviderKind,
    ProviderRequest,
    SearchRequest,
)
from fusion_memory.retrieval.product_planner import ProductQueryPlanner


CATEGORY_PROVIDERS = {
    "event_ordering": (ProviderKind.CHRONOLOGY, ProviderKind.TEMPORAL),
    "temporal_reasoning": (ProviderKind.TEMPORAL,),
    "contradiction_resolution": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "knowledge_update": (ProviderKind.TEMPORAL, ProviderKind.LEXICAL),
    "multi_session_reasoning": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "preference_following": (ProviderKind.LEXICAL, ProviderKind.ENTITY),
    "instruction_following": (ProviderKind.LEXICAL,),
    "information_extraction": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "summarization": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "abstention": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
}


class BeamQueryPlanner:
    def __init__(self, product_planner: ProductQueryPlanner | None = None) -> None:
        self.product_planner = product_planner or ProductQueryPlanner()

    def plan(self, query: str, category: str | None, limit: int) -> ProductQueryPlan:
        request = SearchRequest(query=query, limit=limit, mode="balanced")
        plan = self.product_planner.plan(request)
        provider_requests = list(plan.provider_requests)
        planned_kinds = {provider_request.kind for provider_request in provider_requests}
        provider_limit = max(limit * 2, 12)
        for provider_kind in CATEGORY_PROVIDERS.get(category or "", ()):
            if provider_kind in planned_kinds:
                continue
            provider_requests.append(ProviderRequest(provider_kind, provider_limit))
            planned_kinds.add(provider_kind)
        return replace(plan, provider_requests=tuple(provider_requests))
