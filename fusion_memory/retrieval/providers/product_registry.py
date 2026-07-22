from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from fusion_memory.retrieval.context import ProductQueryPlan, ProviderFailure, RetrievalContext, SearchRequest
from fusion_memory.retrieval.providers.product_base import CandidateProvider, ProviderContext, ProviderOutcome, ProviderUnavailable


class ProductProviderRegistry:
    def __init__(self, providers: Iterable[CandidateProvider]) -> None:
        self._providers = {provider.kind: provider for provider in providers}

    def run(
        self,
        runtime: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan,
    ) -> tuple[ProviderOutcome, ...]:
        outcomes: list[ProviderOutcome] = []
        for provider_request in plan.provider_requests:
            runtime.check_deadline()
            if request.enabled_providers is not None and provider_request.kind not in request.enabled_providers:
                continue
            provider = self._providers[provider_request.kind]
            context = ProviderContext(
                runtime=runtime,
                request=request,
                plan=plan,
                repository=provider.repository,
                provider=provider_request.kind,
                limit=provider_request.limit,
            )
            started = perf_counter()
            try:
                outcomes.append(provider.recall(context))
            except ProviderUnavailable as exc:
                outcomes.append(
                    ProviderOutcome(
                        provider=provider_request.kind,
                        candidates=(),
                        elapsed_ms=(perf_counter() - started) * 1000,
                        failure=ProviderFailure(
                            provider=provider_request.kind,
                            error_code=exc.code,
                            retryable=True,
                        ),
                    )
                )
        return tuple(outcomes)
