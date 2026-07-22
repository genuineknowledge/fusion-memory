from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fusion_memory.core.models import EvidencePack, Scope, new_id
from fusion_memory.eval.beam.query_planner import BeamQueryPlanner
from fusion_memory.retrieval.context import RetrievalContext, SearchRequest
from fusion_memory.retrieval.product_engine import ProductRetrievalEngine
from fusion_memory.retrieval.product_evidence_pack import ProductEvidencePackBuilder
from fusion_memory.retrieval.product_planner import ProductQueryPlanner
from fusion_memory.retrieval.providers.chronology import ChronologyProvider
from fusion_memory.retrieval.providers.entity import EntityProvider
from fusion_memory.retrieval.providers.lexical import LexicalProvider
from fusion_memory.retrieval.providers.product_registry import ProductProviderRegistry
from fusion_memory.retrieval.providers.temporal import TemporalProvider
from fusion_memory.retrieval.providers.vector import VectorProvider


class BeamRetrievalEngine:
    def __init__(
        self,
        *,
        product_engine: ProductRetrievalEngine,
        pack_builder: ProductEvidencePackBuilder,
        planner: BeamQueryPlanner | None = None,
    ) -> None:
        self.product_engine = product_engine
        self.pack_builder = pack_builder
        self.planner = planner or BeamQueryPlanner()

    @classmethod
    def from_service(cls, service: Any) -> "BeamRetrievalEngine":
        repository = service.store
        pack_builder = ProductEvidencePackBuilder(repository, service.config)
        registry = ProductProviderRegistry(
            [
                VectorProvider(repository),
                LexicalProvider(repository),
                TemporalProvider(repository),
                EntityProvider(repository),
                ChronologyProvider(repository),
            ]
        )
        product_engine = ProductRetrievalEngine(
            ProductQueryPlanner(),
            registry,
            pack_builder=pack_builder,
            reranker=service.reranker,
            mmr_lambda=service.config.mmr_lambda,
        )
        return cls(product_engine=product_engine, pack_builder=pack_builder)

    def answer_context(
        self,
        query: str,
        scope: Scope,
        category: str | None,
        budget: dict[str, Any] | None = None,
    ) -> EvidencePack:
        budget = dict(budget or {})
        scope.validate_for_read()
        minimum_limit = 24 if category == "event_ordering" else 50
        requested_limit = _non_negative_int(budget.get("limit"))
        limit = max(minimum_limit, requested_limit)
        token_budget = max(24000, _non_negative_int(budget.get("token_budget")))
        request = SearchRequest(
            query=query,
            limit=limit,
            mode="balanced",
            time_range=budget.get("time_range"),
            include_trace=bool(budget.get("include_trace", True)),
        )
        context = RetrievalContext(
            scope=scope,
            user_id=scope.user_id,
            now=datetime.now(timezone.utc),
            trace_id=new_id("trace"),
            deadline=budget.get("deadline"),
            include_session=True,
        )
        plan = self.planner.plan(query, category, limit)
        result = self.product_engine.search_with_plan(context, request, plan)
        pack = self.pack_builder.build(context, request, result, token_budget)
        pack.coverage["benchmark"] = "BEAM"
        pack.coverage["benchmark_category"] = category
        pack.coverage["query_type"] = category
        return pack


def _non_negative_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return 0
    return value
