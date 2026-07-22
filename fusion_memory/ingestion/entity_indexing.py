from __future__ import annotations

from typing import Any

from fusion_memory.core.models import EvidenceSpan, MemoryEvent, MemoryFact
from fusion_memory.core.text import extract_entities


class EntityIndexer:
    def upsert_span(self, store: Any, span: EvidenceSpan) -> None:
        for entity in span.entities:
            store.upsert_entity(
                span.scope,
                entity,
                entity_type="span_entity",
                source_span_ids=[span.span_id],
                observed_at=span.timestamp,
            )

    def upsert_fact(self, store: Any, fact: MemoryFact) -> None:
        names = extract_entities(fact.text + " " + fact.object)
        if fact.subject and fact.subject not in {
            "user",
            "assistant",
            "agent",
            "tool",
        }:
            names.append(fact.subject)
        for entity in dict.fromkeys(names):
            store.upsert_entity(
                fact.scope,
                entity,
                entity_type="fact_entity",
                source_span_ids=fact.source_span_ids,
                observed_at=fact.observed_at or fact.created_at,
            )

    def upsert_event(self, store: Any, event: MemoryEvent) -> None:
        names = list(event.participants) + extract_entities(event.description)
        for entity in dict.fromkeys(name for name in names if name):
            store.upsert_entity(
                event.scope,
                entity,
                entity_type="event_participant",
                source_span_ids=event.source_span_ids,
                observed_at=event.time_start,
            )
