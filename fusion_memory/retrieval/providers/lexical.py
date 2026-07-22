from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from fusion_memory.core.models import Candidate
from fusion_memory.core.text import keyword_score
from fusion_memory.retrieval.ports import MemorySearchRepository
from fusion_memory.retrieval.providers.product_base import ProviderContext, ProviderOutcome


LEXICAL_SOURCE = "product_lexical"


class LexicalProvider:
    def __init__(self, repository: MemorySearchRepository) -> None:
        self._repository = repository

    def recall(self, context: ProviderContext) -> ProviderOutcome:
        started = perf_counter()
        scope = context.runtime.scope
        include_session = context.runtime.include_session
        spans = self._repository.list_spans(scope, include_session=include_session)
        facts = self._repository.list_facts(scope, include_session=include_session)
        events = self._repository.list_events(scope, include_session=include_session)
        views = self._repository.list_current_views(scope, include_session=include_session)
        profiles = self._repository.search_entity_profiles(
            context.request.query,
            scope,
            limit=context.limit,
            include_session=include_session,
        )

        scored = [
            _scored_candidate(
                context.request.query,
                "span",
                span.span_id,
                span.content,
                [span.span_id],
                span.timestamp,
                {
                    "speaker": span.speaker,
                    "span_type": span.span_type,
                    "timestamp": span.timestamp.isoformat(),
                },
            )
            for span in spans
        ]
        scored.extend(
            _scored_candidate(
                context.request.query,
                "fact",
                fact.fact_id,
                fact.text,
                fact.source_span_ids,
                fact.observed_at or fact.created_at,
                {"category": fact.category, "confidence": fact.confidence},
            )
            for fact in facts
        )
        scored.extend(
            _scored_candidate(
                context.request.query,
                "event",
                event.event_id,
                event.description,
                event.source_span_ids,
                event.time_start or event.time_end,
                {"event_type": event.event_type, "participants": list(event.participants)},
            )
            for event in events
        )
        scored.extend(
            _scored_candidate(
                context.request.query,
                "view",
                view.view_id,
                view.text,
                view.source_span_ids,
                view.updated_at,
                {"view_type": view.view_type, "confidence": view.confidence},
            )
            for view in views
        )
        scored.extend(
            _scored_candidate(
                context.request.query,
                "profile",
                profile.profile_id,
                profile.text,
                profile.source_span_ids,
                profile.updated_at,
                {"profile_type": profile.profile_type, "confidence": profile.confidence},
            )
            for profile, _ in profiles
        )

        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-int(item[0]), -item[1], -_timestamp_value(item[2]), item[3].id))
        return ProviderOutcome(
            provider=context.provider,
            candidates=tuple(item[3] for item in scored[: context.limit]),
            elapsed_ms=(perf_counter() - started) * 1000,
        )


def _scored_candidate(
    query: str,
    candidate_type: str,
    record_id: str,
    text: str,
    source_span_ids: list[str],
    timestamp: datetime | None,
    metadata: dict[str, Any],
) -> tuple[bool, float, datetime | None, Candidate]:
    score = keyword_score(query, text)
    return (
        query.casefold() in text.casefold(),
        score,
        timestamp,
        Candidate(
            id=record_id,
            type=candidate_type,
            text=text,
            source=LEXICAL_SOURCE,
            scores={"bm25_score": score, "score": score},
            source_span_ids=list(source_span_ids),
            metadata=metadata,
        ),
    )


def _timestamp_value(value: datetime | None) -> float:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc).timestamp()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()
