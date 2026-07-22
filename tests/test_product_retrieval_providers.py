from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from fusion_memory.core.models import (
    Candidate,
    CurrentView,
    EntityProfile,
    EntityRecord,
    EvidenceSpan,
    MemoryEvent,
    MemoryFact,
    Scope,
)
from fusion_memory.core.auth import AuthorizationError
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderKind,
    ProviderRequest,
    RetrievalContext,
    SearchRequest,
    TimeRange,
)
from fusion_memory.retrieval.providers.chronology import ChronologyProvider
from fusion_memory.retrieval.providers.entity import EntityProvider
from fusion_memory.retrieval.providers.lexical import LexicalProvider
from fusion_memory.retrieval.providers.product_base import (
    ProviderContext,
    ProviderOutcome,
    ProviderUnavailable,
)
from fusion_memory.retrieval.providers.product_registry import ProductProviderRegistry
from fusion_memory.retrieval.providers.temporal import TemporalProvider
from fusion_memory.retrieval.providers.vector import VectorProvider


class RepositoryFake:
    def __init__(self) -> None:
        self.spans: list[EvidenceSpan] = []
        self.facts: list[MemoryFact] = []
        self.events: list[MemoryEvent] = []
        self.views: list[CurrentView] = []
        self.profiles: list[tuple[EntityProfile, dict[str, float]]] = []
        self.entities: list[tuple[EntityRecord, dict[str, float]]] = []
        self.span_scores: list[tuple[EvidenceSpan, dict[str, float]]] = []
        self.search_spans_error: Exception | None = None
        self.raise_endpoint_unavailable = False
        self.calls: list[tuple[str, dict[str, object]]] = []

    def search_spans(self, query, scope, limit=20, speaker=None, *, include_session=False):
        self.calls.append(
            (
                "search_spans",
                {
                    "query": query,
                    "scope": scope,
                    "limit": limit,
                    "speaker": speaker,
                    "include_session": include_session,
                },
            )
        )
        if self.search_spans_error is not None:
            raise self.search_spans_error
        if self.raise_endpoint_unavailable:
            raise EndpointUnavailable("all endpoints unavailable")
        return self.span_scores[:limit]

    def list_spans(self, scope, *, include_session=False):
        self.calls.append(("list_spans", {"scope": scope, "include_session": include_session}))
        return list(self.spans)

    def list_facts(self, scope, category=None, *, include_session=False):
        self.calls.append(
            ("list_facts", {"scope": scope, "category": category, "include_session": include_session})
        )
        return list(self.facts)

    def list_events(self, scope, *, include_session=False):
        self.calls.append(("list_events", {"scope": scope, "include_session": include_session}))
        return list(self.events)

    def list_chronology_topics(self, scope, include_session=False):
        self.calls.append(
            ("list_chronology_topics", {"scope": scope, "include_session": include_session})
        )
        return []

    def list_current_views(self, scope, view_type=None, *, include_session=False):
        self.calls.append(
            ("list_current_views", {"scope": scope, "view_type": view_type, "include_session": include_session})
        )
        return list(self.views)

    def search_entity_profiles(self, query, scope, limit=20, *, include_session=False):
        self.calls.append(
            (
                "search_entity_profiles",
                {"query": query, "scope": scope, "limit": limit, "include_session": include_session},
            )
        )
        return self.profiles[:limit]

    def search_entities(self, query, scope, limit=20, *, include_session=False):
        self.calls.append(
            (
                "search_entities",
                {"query": query, "scope": scope, "limit": limit, "include_session": include_session},
            )
        )
        return self.entities[:limit]

    def get_span(self, span_id, scope=None, *, include_session=False):
        self.calls.append(
            (
                "get_span",
                {"span_id": span_id, "scope": scope, "include_session": include_session},
            )
        )
        return next((span for span in self.spans if span.span_id == span_id and span.scope == scope), None)


@pytest.fixture
def repository_fake() -> RepositoryFake:
    return RepositoryFake()


@pytest.fixture
def product_provider_context(repository_fake):
    scope = Scope(user_id="user-a")
    runtime = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-1",
        deadline=None,
        include_session=False,
    )
    plan = ProductQueryPlan(
        intent="factual",
        provider_requests=(ProviderRequest(ProviderKind.VECTOR, 4),),
        time_range=None,
        entities=("Atlas",),
        speaker=None,
        ordering=OrderingMode.RELEVANCE,
        use_reranker=False,
    )

    def build(query: str, provider: ProviderKind, limit: int = 4) -> ProviderContext:
        return ProviderContext(
            runtime=runtime,
            request=SearchRequest(query, limit),
            plan=plan,
            repository=repository_fake,
            provider=provider,
            limit=limit,
        )

    return build


def _span(span_id: str, content: str, scope: Scope, *, timestamp: datetime | None = None) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        scope=scope,
        turn_id="turn-1",
        speaker="user",
        span_type="turn",
        content=content,
        content_hash=f"hash-{span_id}",
        timestamp=timestamp or datetime.now(timezone.utc),
    )


def _with_session(context: ProviderContext) -> ProviderContext:
    return replace(context, runtime=replace(context.runtime, include_session=True))


def test_lexical_provider_reads_repository_without_service(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    repository_fake.spans = [_span("span-1", "Atlas uses Qdrant for retrieval.", scope)]

    outcome = LexicalProvider(repository_fake).recall(product_provider_context("Atlas Qdrant", ProviderKind.LEXICAL))

    assert [candidate.id for candidate in outcome.candidates] == ["span-1"]
    assert outcome.candidates[0].source == "product_lexical"
    assert not hasattr(outcome, "service")
    assert {name for name, _ in repository_fake.calls} == {
        "list_spans",
        "list_facts",
        "list_events",
        "list_current_views",
        "search_entity_profiles",
    }


def test_vector_provider_keeps_only_positive_semantic_scores(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    positive = _span("span-positive", "Atlas uses Qdrant", scope)
    zero = _span("span-zero", "Atlas also has a cache", scope)
    repository_fake.span_scores = [
        (positive, {"semantic_score": 0.8, "bm25_score": 0.5, "score": 0.7}),
        (zero, {"semantic_score": 0.0, "bm25_score": 1.0, "score": 0.45}),
    ]

    outcome = VectorProvider(repository_fake).recall(product_provider_context("Atlas", ProviderKind.VECTOR))

    assert [candidate.id for candidate in outcome.candidates] == ["span-positive"]
    assert outcome.candidates[0].source == "product_vector"
    assert outcome.candidates[0].scores == {"semantic_score": 0.8, "bm25_score": 0.5, "score": 0.7}
    assert outcome.candidates[0].metadata["speaker"] == "user"
    assert outcome.candidates[0].metadata["span_type"] == "turn"
    assert outcome.candidates[0].metadata["timestamp"] == positive.timestamp.isoformat()


def test_vector_provider_converts_only_model_endpoint_unavailability(product_provider_context, repository_fake) -> None:
    repository_fake.raise_endpoint_unavailable = True

    with pytest.raises(ProviderUnavailable, match="model_unavailable") as error:
        VectorProvider(repository_fake).recall(product_provider_context("Atlas", ProviderKind.VECTOR))

    assert error.value.code == "model_unavailable"


@pytest.mark.parametrize(
    "repository_error",
    [
        sqlite3.OperationalError("database is locked"),
        AuthorizationError("memory access denied"),
        TypeError("repository contract violation"),
    ],
    ids=["storage", "authorization", "programming"],
)
def test_vector_provider_propagates_non_endpoint_errors_unchanged(
    product_provider_context, repository_fake, repository_error: Exception
) -> None:
    repository_fake.search_spans_error = repository_error

    with pytest.raises(type(repository_error)) as error:
        VectorProvider(repository_fake).recall(product_provider_context("Atlas", ProviderKind.VECTOR))

    assert error.value is repository_error


def test_vector_provider_passes_true_session_visibility_to_repository(product_provider_context, repository_fake) -> None:
    context = _with_session(product_provider_context("Atlas", ProviderKind.VECTOR))

    VectorProvider(repository_fake).recall(context)

    assert repository_fake.calls == [
        (
            "search_spans",
            {
                "query": "Atlas",
                "scope": context.runtime.scope,
                "limit": context.limit,
                "speaker": context.plan.speaker,
                "include_session": True,
            },
        )
    ]


def test_entity_provider_hydrates_only_source_spans_in_context_scope(product_provider_context, repository_fake) -> None:
    user_scope = Scope(user_id="user-a")
    other_scope = Scope(user_id="user-b")
    own_span = _span("span-own", "Atlas uses Qdrant", user_scope)
    other_span = _span("span-other", "Atlas private note", other_scope)
    repository_fake.spans = [own_span, other_span]
    repository_fake.entities = [
        (
            EntityRecord(
                entity_id="entity-atlas",
                scope=user_scope,
                name="Atlas",
                entity_type="project",
                aliases=[],
                source_span_ids=["span-own", "span-other"],
                observed_count=1,
            ),
            {"entity_overlap": 1.0, "score": 1.0},
        )
    ]

    outcome = EntityProvider(repository_fake).recall(product_provider_context("Atlas", ProviderKind.ENTITY))

    assert [candidate.id for candidate in outcome.candidates] == ["span-own"]
    assert outcome.candidates[0].source == "product_entity"
    assert outcome.candidates[0].metadata["entity_name"] == "Atlas"
    assert outcome.candidates[0].metadata["entity_id"] == "entity-atlas"
    assert all(call["scope"] == user_scope for name, call in repository_fake.calls if name == "get_span")


def test_entity_provider_passes_true_session_visibility_to_repository(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    source_span = _span("span-1", "Atlas uses Qdrant", scope)
    repository_fake.spans = [source_span]
    repository_fake.entities = [
        (
            EntityRecord(
                entity_id="entity-atlas",
                scope=scope,
                name="Atlas",
                entity_type="project",
                aliases=[],
                source_span_ids=[source_span.span_id],
                observed_count=1,
            ),
            {"entity_overlap": 1.0, "score": 1.0},
        )
    ]
    context = _with_session(product_provider_context("Atlas", ProviderKind.ENTITY))

    EntityProvider(repository_fake).recall(context)

    assert repository_fake.calls == [
        (
            "search_entities",
            {
                "query": "Atlas",
                "scope": context.runtime.scope,
                "limit": context.limit,
                "include_session": True,
            },
        ),
        (
            "get_span",
            {
                "span_id": source_span.span_id,
                "scope": context.runtime.scope,
                "include_session": True,
            },
        ),
    ]


def test_lexical_provider_sorts_exact_phrase_then_score_timestamp_and_id(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    now = datetime.now(timezone.utc)
    repository_fake.spans = [
        _span("later", "Atlas Qdrant detail", scope, timestamp=now),
        _span("earlier", "Atlas Qdrant detail", scope, timestamp=now - timedelta(seconds=1)),
        _span("higher-score", "Atlas uses Qdrant", scope, timestamp=now),
    ]

    outcome = LexicalProvider(repository_fake).recall(product_provider_context("Atlas Qdrant", ProviderKind.LEXICAL))

    assert [candidate.id for candidate in outcome.candidates] == ["later", "earlier", "higher-score"]


def test_lexical_provider_breaks_full_ties_by_stable_id(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    timestamp = datetime.now(timezone.utc)
    repository_fake.spans = [
        _span("span-z", "Atlas Qdrant detail", scope, timestamp=timestamp),
        _span("span-a", "Atlas Qdrant detail", scope, timestamp=timestamp),
    ]

    outcome = LexicalProvider(repository_fake).recall(product_provider_context("Atlas Qdrant", ProviderKind.LEXICAL))

    assert [candidate.id for candidate in outcome.candidates] == ["span-a", "span-z"]


def test_lexical_provider_passes_true_session_visibility_to_repository(product_provider_context, repository_fake) -> None:
    context = _with_session(product_provider_context("Atlas", ProviderKind.LEXICAL))

    LexicalProvider(repository_fake).recall(context)

    assert repository_fake.calls == [
        ("list_spans", {"scope": context.runtime.scope, "include_session": True}),
        ("list_facts", {"scope": context.runtime.scope, "category": None, "include_session": True}),
        ("list_events", {"scope": context.runtime.scope, "include_session": True}),
        ("list_current_views", {"scope": context.runtime.scope, "view_type": None, "include_session": True}),
        (
            "search_entity_profiles",
            {
                "query": "Atlas",
                "scope": context.runtime.scope,
                "limit": context.limit,
                "include_session": True,
            },
        ),
    ]


def test_provider_outcome_has_only_product_provider_fields(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    repository_fake.spans = [_span("span-1", "Atlas uses Qdrant", scope)]

    outcome = LexicalProvider(repository_fake).recall(product_provider_context("Atlas", ProviderKind.LEXICAL))

    assert set(vars(outcome)) == {"provider", "candidates", "elapsed_ms", "failure"}


def test_temporal_provider_honors_explicit_time_range(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    now = datetime.now(timezone.utc)
    repository_fake.spans = [
        _span("before-range", "Atlas old detail", scope, timestamp=now - timedelta(days=3)),
        _span("inside-range", "Atlas current detail", scope, timestamp=now - timedelta(days=1)),
        _span("after-range", "Atlas future detail", scope, timestamp=now + timedelta(days=1)),
    ]
    context = product_provider_context("Atlas", ProviderKind.TEMPORAL)
    context = replace(
        context,
        request=replace(
            context.request,
            time_range=TimeRange(start=now - timedelta(days=2), end=now),
        ),
    )

    outcome = TemporalProvider(repository_fake).recall(context)

    assert [candidate.id for candidate in outcome.candidates] == ["inside-range"]


def test_chronology_provider_uses_persisted_graph_without_preservation_metadata(
    product_provider_context, repository_fake, monkeypatch
) -> None:
    graph_candidate = Candidate(
        id="graph-event",
        type="event",
        text="Atlas deployed the first version.",
        source="event_ordering_graph",
        scores={"score": 1.0},
        source_span_ids=["span-graph"],
        metadata={
            "must_preserve_reason": "chronology",
            "graph_topic_id": "topic-atlas",
            "graph_phase_id": "phase-deployment",
            "timeline_index": 1,
        },
    )
    monkeypatch.setattr(
        "fusion_memory.retrieval.providers.chronology.select_persisted_graph_event_ordering_candidates",
        lambda *_args, **_kwargs: ([graph_candidate], {}),
    )

    outcome = ChronologyProvider(repository_fake).recall(
        product_provider_context("Atlas timeline", ProviderKind.CHRONOLOGY)
    )

    assert [candidate.source for candidate in outcome.candidates] == ["product_chronology"]
    assert "must_preserve_reason" not in outcome.candidates[0].metadata
    assert outcome.candidates[0].metadata["graph_topic_id"] == "topic-atlas"


def test_chronology_provider_falls_back_to_events_sorted_by_time_then_id(
    product_provider_context, repository_fake
) -> None:
    scope = Scope(user_id="user-a")
    earlier = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = earlier + timedelta(hours=1)
    repository_fake.events = [
        MemoryEvent("event-z", scope, "change", "Atlas changed configuration", [], [], time_start=earlier),
        MemoryEvent("event-a", scope, "change", "Atlas deployed service", [], [], time_start=earlier),
        MemoryEvent("event-later", scope, "change", "Atlas verified service", [], [], time_start=later),
    ]

    outcome = ChronologyProvider(repository_fake).recall(
        product_provider_context("Atlas timeline", ProviderKind.CHRONOLOGY, limit=2)
    )

    assert [candidate.id for candidate in outcome.candidates] == ["event-a", "event-z"]
    assert all(candidate.source == "product_chronology" for candidate in outcome.candidates)


def test_temporal_provider_keeps_query_relevant_records_in_descending_recency(
    product_provider_context, repository_fake
) -> None:
    scope = Scope(user_id="user-a")
    now = datetime.now(timezone.utc)
    repository_fake.spans = [
        _span("old-match", "Atlas used the old index.", scope, timestamp=now - timedelta(days=2)),
        _span("new-match", "Atlas uses the current index.", scope, timestamp=now - timedelta(days=1)),
        _span("irrelevant", "Borealis scheduled a different deployment.", scope, timestamp=now),
    ]

    outcome = TemporalProvider(repository_fake).recall(
        product_provider_context("Atlas index", ProviderKind.TEMPORAL)
    )

    assert [candidate.id for candidate in outcome.candidates] == ["new-match", "old-match"]
    assert all("temporal_score" in candidate.scores for candidate in outcome.candidates)


def test_registry_records_model_provider_failure_and_keeps_lexical_result(
    product_provider_context, repository_fake
) -> None:
    class FailingVectorProvider:
        kind = ProviderKind.VECTOR

        def __init__(self) -> None:
            self.repository = repository_fake

        def recall(self, context: ProviderContext) -> ProviderOutcome:
            raise ProviderUnavailable("model_unavailable")

    class StaticLexicalProvider:
        kind = ProviderKind.LEXICAL

        def __init__(self) -> None:
            self.repository = repository_fake

        def recall(self, context: ProviderContext) -> ProviderOutcome:
            return ProviderOutcome(
                provider=context.provider,
                candidates=(
                    Candidate(
                        id="lexical-1",
                        type="span",
                        text="Atlas uses Qdrant.",
                        source="product_lexical",
                        scores={"score": 1.0},
                        source_span_ids=["lexical-1"],
                        metadata={},
                    ),
                ),
                elapsed_ms=0.0,
            )

    context = product_provider_context("Atlas", ProviderKind.VECTOR)
    plan = replace(
        context.plan,
        provider_requests=(
            ProviderRequest(ProviderKind.VECTOR, 4),
            ProviderRequest(ProviderKind.LEXICAL, 4),
        ),
    )

    outcomes = ProductProviderRegistry([FailingVectorProvider(), StaticLexicalProvider()]).run(
        context.runtime, context.request, plan
    )

    assert outcomes[0].failure is not None
    assert outcomes[0].failure.error_code == "model_unavailable"
    assert [candidate.id for candidate in outcomes[1].candidates] == ["lexical-1"]


def test_registry_skips_disabled_provider(product_provider_context, repository_fake) -> None:
    class VectorProviderStub:
        kind = ProviderKind.VECTOR

        def __init__(self) -> None:
            self.repository = repository_fake

        def recall(self, context: ProviderContext) -> ProviderOutcome:
            raise AssertionError("disabled provider must not run")

    context = product_provider_context("Atlas", ProviderKind.VECTOR)
    plan = replace(context.plan, provider_requests=(ProviderRequest(ProviderKind.VECTOR, 4),))
    request = replace(context.request, enabled_providers=frozenset({ProviderKind.LEXICAL}))

    assert ProductProviderRegistry([VectorProviderStub()]).run(context.runtime, request, plan) == ()


def test_registry_propagates_non_provider_unavailable_errors(product_provider_context, repository_fake) -> None:
    storage_error = sqlite3.OperationalError("database is locked")

    class BrokenProvider:
        kind = ProviderKind.VECTOR

        def __init__(self) -> None:
            self.repository = repository_fake

        def recall(self, context: ProviderContext) -> ProviderOutcome:
            raise storage_error

    context = product_provider_context("Atlas", ProviderKind.VECTOR)
    plan = replace(context.plan, provider_requests=(ProviderRequest(ProviderKind.VECTOR, 4),))

    with pytest.raises(sqlite3.OperationalError) as error:
        ProductProviderRegistry([BrokenProvider()]).run(context.runtime, context.request, plan)

    assert error.value is storage_error


def test_registry_accepts_product_provider_instances(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    vector_span = _span("vector-1", "Atlas uses Qdrant.", scope)
    repository_fake.span_scores = [(vector_span, {"semantic_score": 0.9, "bm25_score": 0.8, "score": 0.9})]
    repository_fake.spans = [vector_span]
    context = product_provider_context("Atlas", ProviderKind.VECTOR)
    plan = replace(
        context.plan,
        provider_requests=(
            ProviderRequest(ProviderKind.VECTOR, 4),
            ProviderRequest(ProviderKind.LEXICAL, 4),
        ),
    )

    outcomes = ProductProviderRegistry([VectorProvider(repository_fake), LexicalProvider(repository_fake)]).run(
        context.runtime, context.request, plan
    )

    assert [outcome.provider for outcome in outcomes] == [ProviderKind.VECTOR, ProviderKind.LEXICAL]
