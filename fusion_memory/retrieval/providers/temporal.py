from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

from fusion_memory.core.models import Candidate
from fusion_memory.core.text import keyword_score
from fusion_memory.retrieval.context import ProviderKind, TimeRange
from fusion_memory.retrieval.ports import MemorySearchRepository
from fusion_memory.retrieval.providers.product_base import ProviderContext, ProviderOutcome


TEMPORAL_SOURCE = "product_temporal"


class TemporalProvider:
    kind = ProviderKind.TEMPORAL

    def __init__(self, repository: MemorySearchRepository) -> None:
        self.repository = repository

    def recall(self, context: ProviderContext) -> ProviderOutcome:
        started = perf_counter()
        scope = context.runtime.scope
        include_session = context.runtime.include_session
        time_range = context.request.time_range
        records: list[tuple[datetime, Candidate]] = []

        for span in self.repository.list_spans(scope, include_session=include_session):
            if not _include_record(context.request.query, span.content, span.timestamp, time_range):
                continue
            records.append(
                (
                    span.timestamp,
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source=TEMPORAL_SOURCE,
                        scores={"temporal_score": _temporal_score(span.timestamp, context.runtime.now)},
                        source_span_ids=[span.span_id],
                        metadata={
                            "speaker": span.speaker,
                            "span_type": span.span_type,
                            "timestamp": span.timestamp.isoformat(),
                        },
                    ),
                )
            )

        for event in self.repository.list_events(scope, include_session=include_session):
            timestamp = event.time_start or event.time_end
            if timestamp is None or not _include_record(context.request.query, event.description, timestamp, time_range):
                continue
            records.append(
                (
                    timestamp,
                    Candidate(
                        id=event.event_id,
                        type="event",
                        text=event.description,
                        source=TEMPORAL_SOURCE,
                        scores={"temporal_score": _temporal_score(timestamp, context.runtime.now)},
                        source_span_ids=list(event.source_span_ids),
                        metadata={
                            "event_type": event.event_type,
                            "participants": list(event.participants),
                            "time_start": event.time_start.isoformat() if event.time_start else None,
                            "time_end": event.time_end.isoformat() if event.time_end else None,
                        },
                    ),
                )
            )

        records.sort(key=lambda item: (-_timestamp_value(item[0]), item[1].id))
        return ProviderOutcome(
            provider=context.provider,
            candidates=tuple(candidate for _timestamp, candidate in records[: context.limit]),
            elapsed_ms=(perf_counter() - started) * 1000,
        )


def _include_record(query: str, text: str, timestamp: datetime, time_range: TimeRange | None) -> bool:
    if time_range is not None:
        return time_range.contains(timestamp)
    return keyword_score(query, text) > 0


def _temporal_score(timestamp: datetime, now: datetime) -> float:
    age_seconds = max(0.0, _timestamp_value(now) - _timestamp_value(timestamp))
    return 1.0 / (1.0 + age_seconds / 86_400)


def _timestamp_value(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()
