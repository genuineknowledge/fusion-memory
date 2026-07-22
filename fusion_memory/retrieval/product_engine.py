from __future__ import annotations

from time import perf_counter
from typing import Protocol

from fusion_memory.core.models import EvidencePack
from fusion_memory.retrieval.context import (
    ProductQueryPlan,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.engine import RetrievalUnavailable
from fusion_memory.retrieval.providers.base import ProviderOutcome
from fusion_memory.retrieval.query_planner import ProductQueryPlanner
from fusion_memory.retrieval.reranker import LexicalCrossEncoderReranker, Reranker
from fusion_memory.retrieval.selection import select_candidates
from fusion_memory.retrieval.tracing import (
    build_retrieval_trace,
    sanitize_dimension,
    validate_product_plan,
)


class ProductPlanner(Protocol):
    def plan(self, request: SearchRequest) -> object: ...

    def safe_default(self, request: SearchRequest) -> ProductQueryPlan: ...


class ProductRegistry(Protocol):
    def run(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan,
    ) -> tuple[ProviderOutcome, ...]: ...


class ProductPackBuilder(Protocol):
    def build(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack: ...


class ProductRetrievalEngine:
    def __init__(
        self,
        planner: ProductPlanner | None,
        registry: ProductRegistry,
        *,
        pack_builder: ProductPackBuilder,
        reranker: Reranker | None = None,
        mmr_lambda: float = 0.72,
    ) -> None:
        self.planner = planner if planner is not None else ProductQueryPlanner()
        self.registry = registry
        self.reranker = reranker or LexicalCrossEncoderReranker()
        self.mmr_lambda = mmr_lambda
        self.pack_builder = pack_builder

    def build_evidence_pack(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack:
        return self.pack_builder.build(context, request, result, token_budget)

    def search(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan | None = None,
    ) -> RetrievalResult:
        plan_started = perf_counter()
        planned = plan or self.planner.plan(request)
        plan_elapsed_ms = (perf_counter() - plan_started) * 1000
        if validate_product_plan(planned):
            return self._search_with_plan(context, request, planned, plan_elapsed_ms=plan_elapsed_ms)

        fallback = self.planner.safe_default(request)
        plan_elapsed_ms = (perf_counter() - plan_started) * 1000
        result = self._search_with_plan(context, request, fallback, plan_elapsed_ms=plan_elapsed_ms)
        return RetrievalResult(
            candidates=result.candidates,
            coverage={**result.coverage, "planner_fallback": "invalid_plan"},
            trace={**result.trace, "planner_fallback": "invalid_plan"},
            plan=result.plan,
        )

    def search_with_plan(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan,
    ) -> RetrievalResult:
        if not validate_product_plan(plan):
            raise ValueError("search_with_plan requires a valid ProductQueryPlan")
        return self._search_with_plan(context, request, plan, plan_elapsed_ms=0.0)

    def _search_with_plan(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan,
        *,
        plan_elapsed_ms: float,
    ) -> RetrievalResult:
        context.check_deadline()
        recall_started = perf_counter()
        outcomes = self.registry.run(context, request, plan)
        recall_elapsed_ms = (perf_counter() - recall_started) * 1000
        context.check_deadline()

        successful = [outcome for outcome in outcomes if outcome.failure is None]
        if not successful:
            raise RetrievalUnavailable("all planned providers failed")

        candidate_lists = [
            list(outcome.candidates) for outcome in successful if outcome.candidates
        ]
        selection_status: dict[str, object] = {}
        selected = select_candidates(
            request.query,
            candidate_lists,
            limit=request.limit,
            use_reranker=plan.use_reranker,
            reranker=self.reranker,
            mmr_lambda=self.mmr_lambda,
            status=selection_status,
        )

        failures = [
            sanitize_dimension(outcome.failure.error_code)
            for outcome in outcomes
            if outcome.failure is not None
        ]
        reranker_failure_value = selection_status.get("reranker_failure")
        reranker_failure = (
            str(reranker_failure_value) if reranker_failure_value is not None else None
        )
        coverage = {
            "intent": sanitize_dimension(plan.intent),
            "degraded": bool(failures) or reranker_failure is not None,
            "provider_failures": failures,
            "provider_counts": {
                outcome.provider.value: len(outcome.candidates) for outcome in outcomes
            },
        }
        if reranker_failure is not None:
            coverage["reranker_unavailable"] = True

        fused_count = int(selection_status.get("fused_count", 0))
        trace = build_retrieval_trace(
            context,
            request,
            plan,
            outcomes,
            selected,
            filtered_count=max(0, fused_count - len(selected)),
            stage_durations_ms={
                "plan": plan_elapsed_ms,
                "recall": recall_elapsed_ms,
                "fusion": float(selection_status.get("fusion_elapsed_ms", 0.0)),
                "selection": float(selection_status.get("selection_elapsed_ms", 0.0)),
            },
            reranker_failure=reranker_failure,
        )
        return RetrievalResult(tuple(selected), coverage, trace, plan)
