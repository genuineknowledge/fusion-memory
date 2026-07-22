from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from fusion_memory.core.models import Candidate, MemoryEvent
from fusion_memory.retrieval.chronology_selector import select_persisted_graph_event_ordering_candidates
from fusion_memory.retrieval.context import ProviderKind
from fusion_memory.retrieval.ports import MemorySearchRepository
from fusion_memory.retrieval.providers.base import ProviderContext, ProviderOutcome


CHRONOLOGY_SOURCE = "product_chronology"


class ChronologyProvider:
    kind = ProviderKind.CHRONOLOGY

    def __init__(self, repository: MemorySearchRepository) -> None:
        self._repository = repository

    @property
    def repository(self) -> MemorySearchRepository:
        return self._repository

    def recall(self, context: ProviderContext) -> ProviderOutcome:
        started = perf_counter()
        graph_candidates, _telemetry = select_persisted_graph_event_ordering_candidates(
            context.request.query,
            context.runtime.scope,
            self._repository,
            context.limit,
            include_session=context.runtime.include_session,
        )
        if graph_candidates:
            candidates = tuple(_product_graph_candidate(candidate) for candidate in graph_candidates[: context.limit])
        else:
            candidates = tuple(
                _event_candidate(event)
                for event in sorted(
                    self._repository.list_events(
                        context.runtime.scope,
                        include_session=context.runtime.include_session,
                    ),
                    key=lambda event: (event.time_start is None, event.time_start, event.event_id),
                )[: context.limit]
            )
        return ProviderOutcome(
            provider=context.provider,
            candidates=candidates,
            elapsed_ms=(perf_counter() - started) * 1000,
        )


def _product_graph_candidate(candidate: Candidate) -> Candidate:
    metadata = dict(candidate.metadata)
    metadata.pop("must_preserve_reason", None)
    return replace(candidate, source=CHRONOLOGY_SOURCE, metadata=metadata)


def _event_candidate(event: MemoryEvent) -> Candidate:
    return Candidate(
        id=event.event_id,
        type="event",
        text=event.description,
        source=CHRONOLOGY_SOURCE,
        scores={"score": 1.0},
        source_span_ids=list(event.source_span_ids),
        metadata={
            "event_type": event.event_type,
            "participants": list(event.participants),
            "time_start": event.time_start.isoformat() if event.time_start else None,
            "time_end": event.time_end.isoformat() if event.time_end else None,
        },
    )
