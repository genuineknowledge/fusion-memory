from __future__ import annotations

from time import perf_counter

from fusion_memory.core.models import Candidate
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.retrieval.context import ProviderKind
from fusion_memory.retrieval.ports import MemorySearchRepository
from fusion_memory.retrieval.providers.product_base import ProviderContext, ProviderOutcome, ProviderUnavailable


VECTOR_SOURCE = "product_vector"


class VectorProvider:
    kind = ProviderKind.VECTOR

    def __init__(self, repository: MemorySearchRepository) -> None:
        self._repository = repository

    @property
    def repository(self) -> MemorySearchRepository:
        return self._repository

    def recall(self, context: ProviderContext) -> ProviderOutcome:
        started = perf_counter()
        try:
            results = self._repository.search_spans(
                context.request.query,
                context.runtime.scope,
                limit=context.limit,
                speaker=context.plan.speaker,
                include_session=context.runtime.include_session,
            )
        except EndpointUnavailable as exc:
            raise ProviderUnavailable("model_unavailable") from exc

        candidates = tuple(
            Candidate(
                id=span.span_id,
                type="span",
                text=span.content,
                source=VECTOR_SOURCE,
                scores={
                    "semantic_score": float(scores.get("semantic_score", 0.0)),
                    "bm25_score": float(scores.get("bm25_score", 0.0)),
                    "score": float(scores.get("score", 0.0)),
                },
                source_span_ids=[span.span_id],
                metadata={
                    "speaker": span.speaker,
                    "span_type": span.span_type,
                    "timestamp": span.timestamp.isoformat(),
                },
            )
            for span, scores in results
            if scores.get("semantic_score", 0.0) > 0
        )
        return ProviderOutcome(
            provider=context.provider,
            candidates=candidates[: context.limit],
            elapsed_ms=(perf_counter() - started) * 1000,
        )
