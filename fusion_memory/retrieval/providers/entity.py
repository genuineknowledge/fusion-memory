from __future__ import annotations

from time import perf_counter

from fusion_memory.core.models import Candidate
from fusion_memory.retrieval.context import ProviderKind
from fusion_memory.retrieval.ports import MemorySearchRepository
from fusion_memory.retrieval.providers.base import ProviderContext, ProviderOutcome


ENTITY_SOURCE = "product_entity"


class EntityProvider:
    kind = ProviderKind.ENTITY

    def __init__(self, repository: MemorySearchRepository) -> None:
        self._repository = repository

    @property
    def repository(self) -> MemorySearchRepository:
        return self._repository

    def recall(self, context: ProviderContext) -> ProviderOutcome:
        started = perf_counter()
        entities = self._repository.search_entities(
            context.request.query,
            context.runtime.scope,
            limit=context.limit,
            include_session=context.runtime.include_session,
        )
        candidates: list[Candidate] = []
        seen_span_ids: set[str] = set()
        for entity, scores in entities:
            for span_id in entity.source_span_ids:
                if span_id in seen_span_ids:
                    continue
                span = self._repository.get_span(
                    span_id,
                    context.runtime.scope,
                    include_session=context.runtime.include_session,
                )
                if span is None:
                    continue
                seen_span_ids.add(span_id)
                candidates.append(
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source=ENTITY_SOURCE,
                        scores=dict(scores),
                        source_span_ids=[span.span_id],
                        metadata={
                            "entity_name": entity.name,
                            "entity_id": entity.entity_id,
                            "speaker": span.speaker,
                            "span_type": span.span_type,
                            "timestamp": span.timestamp.isoformat(),
                        },
                    )
                )
                if len(candidates) >= context.limit:
                    return ProviderOutcome(
                        provider=context.provider,
                        candidates=tuple(candidates),
                        elapsed_ms=(perf_counter() - started) * 1000,
                    )
        return ProviderOutcome(
            provider=context.provider,
            candidates=tuple(candidates),
            elapsed_ms=(perf_counter() - started) * 1000,
        )
