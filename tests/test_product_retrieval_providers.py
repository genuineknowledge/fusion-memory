from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fusion_memory.core.models import (
    CurrentView,
    EntityProfile,
    EntityRecord,
    EvidenceSpan,
    MemoryEvent,
    MemoryFact,
    Scope,
)
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderKind,
    ProviderRequest,
    RetrievalContext,
    SearchRequest,
)
from fusion_memory.retrieval.providers.entity import EntityProvider
from fusion_memory.retrieval.providers.lexical import LexicalProvider
from fusion_memory.retrieval.providers.product_base import ProviderContext, ProviderUnavailable
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
        self.raise_endpoint_unavailable = False
        self.calls: list[tuple[str, object]] = []

    def search_spans(self, query, scope, limit=20, speaker=None, *, include_session=False):
        self.calls.append(("search_spans", scope))
        if self.raise_endpoint_unavailable:
            raise EndpointUnavailable("all endpoints unavailable")
        return self.span_scores[:limit]

    def list_spans(self, scope, *, include_session=False):
        self.calls.append(("list_spans", scope))
        return list(self.spans)

    def list_facts(self, scope, category=None, *, include_session=False):
        self.calls.append(("list_facts", scope))
        return list(self.facts)

    def list_events(self, scope, *, include_session=False):
        self.calls.append(("list_events", scope))
        return list(self.events)

    def list_current_views(self, scope, view_type=None, *, include_session=False):
        self.calls.append(("list_current_views", scope))
        return list(self.views)

    def search_entity_profiles(self, query, scope, limit=20, *, include_session=False):
        self.calls.append(("search_entity_profiles", scope))
        return self.profiles[:limit]

    def search_entities(self, query, scope, limit=20, *, include_session=False):
        self.calls.append(("search_entities", scope))
        return self.entities[:limit]

    def get_span(self, span_id, scope=None, *, include_session=False):
        self.calls.append(("get_span", scope))
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
    assert all(scope == user_scope for name, scope in repository_fake.calls if name == "get_span")


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


def test_provider_outcome_has_only_product_provider_fields(product_provider_context, repository_fake) -> None:
    scope = Scope(user_id="user-a")
    repository_fake.spans = [_span("span-1", "Atlas uses Qdrant", scope)]

    outcome = LexicalProvider(repository_fake).recall(product_provider_context("Atlas", ProviderKind.LEXICAL))

    assert set(vars(outcome)) == {"provider", "candidates", "elapsed_ms", "failure"}
