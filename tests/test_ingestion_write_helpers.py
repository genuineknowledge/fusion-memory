from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fusion_memory import MemoryService
from fusion_memory.core.models import EvidenceSpan, MemoryEvent, MemoryFact, Scope
from fusion_memory.ingestion.candidate_records import (
    candidate_to_event,
    candidate_to_fact,
    candidate_to_relation,
)
from fusion_memory.ingestion.entity_indexing import EntityIndexer


class RecordingEntityStore:
    def __init__(self) -> None:
        self.calls: list[tuple[Scope, str, str, list[str], datetime | None]] = []

    def upsert_entity(
        self,
        scope: Scope,
        name: str,
        *,
        entity_type: str,
        source_span_ids: list[str],
        observed_at: datetime | None,
    ) -> None:
        self.calls.append(
            (scope, name, entity_type, source_span_ids, observed_at)
        )


def test_entity_indexer_preserves_span_fact_and_event_upsert_semantics() -> None:
    scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
    observed_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    span = EvidenceSpan(
        span_id="span-1",
        scope=scope,
        turn_id="turn-1",
        speaker="user",
        span_type="turn",
        content="Atlas and Qdrant",
        content_hash="hash-1",
        timestamp=observed_at,
        entities=["Atlas", "Atlas"],
    )
    fact = MemoryFact(
        fact_id="fact-1",
        scope=scope,
        subject="Father",
        predicate="prefers",
        object="Qdrant",
        text="Father prefers Qdrant for Atlas",
        category="preference",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[span.span_id],
        observed_at=observed_at,
    )
    event = MemoryEvent(
        event_id="event-1",
        scope=scope,
        event_type="decision",
        description="Father selected Qdrant for Atlas",
        participants=["Father", ""],
        source_span_ids=[span.span_id],
        time_start=observed_at,
    )
    store = RecordingEntityStore()
    indexer = EntityIndexer()

    with patch(
        "fusion_memory.ingestion.entity_indexing.extract_entities",
        side_effect=[
            ["Qdrant", "Atlas", "Qdrant"],
            ["Atlas", "Father"],
        ],
    ):
        indexer.upsert_span(store, span)
        indexer.upsert_fact(store, fact)
        indexer.upsert_event(store, event)

    assert store.calls == [
        (scope, "Atlas", "span_entity", ["span-1"], observed_at),
        (scope, "Atlas", "span_entity", ["span-1"], observed_at),
        (scope, "Qdrant", "fact_entity", ["span-1"], observed_at),
        (scope, "Atlas", "fact_entity", ["span-1"], observed_at),
        (scope, "Father", "fact_entity", ["span-1"], observed_at),
        (scope, "Father", "event_participant", ["span-1"], observed_at),
        (scope, "Atlas", "event_participant", ["span-1"], observed_at),
    ]


def test_memory_service_entity_indexer_uses_the_current_store() -> None:
    memory = MemoryService()

    class RecordingStore:
        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped
            self.upserted_entities: list[str] = []

        def __getattr__(self, name: str) -> object:
            return getattr(self.wrapped, name)

        def upsert_entity(self, scope: Scope, name: str, **kwargs: object) -> None:
            self.upserted_entities.append(name)
            self.wrapped.upsert_entity(scope, name, **kwargs)

    current_store = RecordingStore(memory.store)
    memory.store = current_store
    try:
        memory.add(
            "I prefer Qdrant for Atlas retrieval.",
            Scope(user_id="u", session_id="s"),
            datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
    finally:
        memory.close()

    assert current_store.upserted_entities


def test_candidate_to_fact_preserves_write_record_shape() -> None:
    scope = Scope(user_id="u", session_id="s")
    observed_at = datetime(2026, 7, 2, tzinfo=timezone.utc)
    candidate = SimpleNamespace(
        local_id="local-fact",
        text="Father prefers Qdrant",
        confidence=0.7,
        source_span_ids=["span-1", "span-1"],
        structured={
            "subject": "Father",
            "predicate": "prefers",
            "object": "Qdrant",
            "category": "preference",
            "confidence": 0.9,
            "salience": 0.8,
            "polarity": "positive",
            "value_mentions": ["Qdrant"],
            "topic_terms": ["Atlas"],
        },
    )

    with patch(
        "fusion_memory.ingestion.candidate_records.new_id",
        return_value="fact-1",
    ):
        fact = candidate_to_fact(scope, candidate, observed_at)

    assert fact.fact_id == "fact-1"
    assert fact.scope == scope
    assert fact.subject == "Father"
    assert fact.predicate == "prefers"
    assert fact.object == "Qdrant"
    assert fact.text == candidate.text
    assert fact.category == "preference"
    assert fact.confidence == 0.9
    assert fact.salience == 0.8
    assert fact.observed_at == observed_at
    assert fact.valid_from == observed_at
    assert fact.valid_to is None
    assert fact.polarity == "positive"
    assert fact.source_span_ids == ["span-1"]
    assert fact.metadata == {
        "hash": "4f22d605aee879cb14722c470239dd576cf0a528a9ad560f150febc9cbbcebd1",
        "candidate_local_id": "local-fact",
        "value_mentions": ["Qdrant"],
        "topic_terms": ["Atlas"],
    }


def test_candidate_to_event_and_relation_preserve_write_record_shapes() -> None:
    scope = Scope(user_id="u", session_id="s")
    candidate = SimpleNamespace(
        text="Father switched Atlas to Qdrant",
        confidence=0.7,
        source_span_ids=["span-1", "span-1"],
        structured={
            "event_type": "preference_change",
            "participants": ["Father", "Atlas"],
            "description": "Father switched Atlas to Qdrant",
            "time_start": "2026-07-03T10:00:00+00:00",
            "time_end": None,
            "time_granularity": "day",
            "time_source": "explicit",
            "confidence": 0.85,
        },
    )
    relation_candidate = SimpleNamespace(
        confidence=0.8,
        source_span_ids=["span-1", "span-1"],
        structured={
            "from_local_id": "local-fact",
            "to_fact_id": "fact-old",
            "relation_type": "supersedes",
            "confidence": 0.95,
        },
    )

    with patch(
        "fusion_memory.ingestion.candidate_records.new_id",
        side_effect=["event-1", "relation-1"],
    ):
        event = candidate_to_event(scope, candidate)
        relation = candidate_to_relation(
            relation_candidate,
            {"local-fact": "fact-new"},
        )

    assert event.event_id == "event-1"
    assert event.scope == scope
    assert event.event_type == "preference_change"
    assert event.participants == ["Father", "Atlas"]
    assert event.description == candidate.text
    assert event.time_start == datetime(2026, 7, 3, 10, tzinfo=timezone.utc)
    assert event.time_end is None
    assert event.time_granularity == "day"
    assert event.time_source == "explicit"
    assert event.source_span_ids == ["span-1"]
    assert event.confidence == 0.85

    assert relation is not None
    assert relation.relation_id == "relation-1"
    assert relation.from_fact_id == "fact-new"
    assert relation.to_fact_id == "fact-old"
    assert relation.relation_type == "supersedes"
    assert relation.source_span_ids == ["span-1"]
    assert relation.confidence == 0.95
    assert candidate_to_relation(
        relation_candidate,
        {},
    ) is None
