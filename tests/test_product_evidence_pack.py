from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from fusion_memory.core.config import MemoryConfig
from fusion_memory.core.models import (
    Candidate,
    CurrentView,
    EvidenceSpan,
    MemoryFact,
    Scope,
)
from fusion_memory.ingestion.candidate_records import candidate_to_fact
from fusion_memory.ingestion.extractors import RuleBasedExtractor
from fusion_memory.ingestion.views import ViewBuilder
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.product_evidence_pack import ProductEvidencePackBuilder
from fusion_memory.retrieval.product_engine import ProductRetrievalEngine
from fusion_memory.retrieval.product_planner import ProductQueryPlanner


class SpanRepository:
    def __init__(
        self,
        spans: list[EvidenceSpan],
        *,
        facts: list[MemoryFact] | None = None,
        views: list[CurrentView] | None = None,
    ) -> None:
        self.spans = {span.span_id: span for span in spans}
        self.facts = {fact.fact_id: fact for fact in facts or []}
        self.views = list(views or [])
        self.calls: list[tuple[str, Scope, bool]] = []
        self.fact_calls: list[tuple[str, Scope, bool]] = []
        self.view_calls: list[tuple[Scope, str | None, bool]] = []

    def get_span(
        self,
        span_id: str,
        scope: Scope,
        *,
        include_session: bool = False,
    ) -> EvidenceSpan | None:
        self.calls.append((span_id, scope, include_session))
        span = self.spans.get(span_id)
        if span is None:
            return None
        for field in ("workspace_id", "user_id", "agent_id", "run_id"):
            expected = getattr(scope, field)
            if expected is not None and getattr(span.scope, field) != expected:
                return None
        if include_session and span.scope.session_id != scope.session_id:
            return None
        return span

    def get_fact(
        self,
        fact_id: str,
        scope: Scope,
        *,
        include_session: bool = False,
    ) -> MemoryFact | None:
        self.fact_calls.append((fact_id, scope, include_session))
        fact = self.facts.get(fact_id)
        if fact is None or not _scope_matches(fact.scope, scope, include_session):
            return None
        return fact

    def list_current_views(
        self,
        scope: Scope,
        view_type: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[CurrentView]:
        self.view_calls.append((scope, view_type, include_session))
        return [
            view
            for view in self.views
            if (view_type is None or view.view_type == view_type)
            and _scope_matches(view.scope, scope, include_session)
        ]


def _scope_matches(record_scope: Scope, scope: Scope, include_session: bool) -> bool:
    for field in ("workspace_id", "user_id", "agent_id", "run_id"):
        expected = getattr(scope, field)
        if expected is not None and getattr(record_scope, field) != expected:
            return False
    return not include_session or record_scope.session_id == scope.session_id


def _span(
    span_id: str,
    *,
    timestamp: datetime,
    content: str = "Atlas source evidence",
    scope: Scope | None = None,
) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        scope=scope or Scope(user_id="user-a", session_id="session-a"),
        turn_id=f"turn-{span_id}",
        speaker="user",
        span_type="turn",
        content=content,
        content_hash=f"hash-{span_id}",
        timestamp=timestamp,
        source_uri=f"memory://{span_id}",
    )


def _candidate(
    candidate_id: str,
    span_id: str | None = None,
    *,
    candidate_type: str = "span",
    text: str = "selected candidate",
    source: str = "product_lexical",
    timeline_index: int | None = None,
    source_span_ids: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> Candidate:
    candidate_metadata = dict(metadata or {})
    if timeline_index is not None:
        candidate_metadata["timeline_index"] = timeline_index
    return Candidate(
        id=candidate_id,
        type=candidate_type,
        text=text,
        source=source,
        scores={"bm25_score": 0.8},
        source_span_ids=source_span_ids if source_span_ids is not None else [str(span_id)],
        metadata=candidate_metadata,
    )


def _plan(
    ordering: OrderingMode = OrderingMode.RELEVANCE,
    *,
    intent: str = "factual",
    entities: tuple[str, ...] = (),
    query_intent: dict[str, object] | None = None,
) -> ProductQueryPlan:
    return ProductQueryPlan(
        intent=intent,
        provider_requests=(),
        time_range=None,
        entities=entities,
        speaker=None,
        ordering=ordering,
        use_reranker=False,
        query_intent={"target": "Atlas"} if query_intent is None else query_intent,
    )


@dataclass
class PackFixture:
    builder: ProductEvidencePackBuilder
    context: RetrievalContext
    request: SearchRequest
    result: RetrievalResult
    repository: SpanRepository


@pytest.fixture
def pack_fixture() -> PackFixture:
    scope = Scope(user_id="user-a", session_id="session-a")
    repository = SpanRepository([_span("span-1", timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc))])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-1",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(query="Where is Atlas?", limit=4)
    result = RetrievalResult(
        candidates=(_candidate("candidate-1", "span-1"),),
        coverage={"degraded": False, "provider_counts": {"lexical": 1}},
        trace={"stages": ["selection"]},
        plan=_plan(),
    )
    return PackFixture(
        builder=ProductEvidencePackBuilder(repository, MemoryConfig(evidence_span_summary_chars=80)),
        context=context,
        request=request,
        result=result,
        repository=repository,
    )


def test_product_pack_preserves_source_provenance(pack_fixture: PackFixture) -> None:
    pack = pack_fixture.builder.build(
        pack_fixture.context,
        pack_fixture.request,
        pack_fixture.result,
        token_budget=1200,
    )

    assert pack.source_spans[0] == {
        "id": "span-1",
        "session_id": "session-a",
        "turn_id": "turn-span-1",
        "speaker": "user",
        "timestamp": "2026-07-01T00:00:00+00:00",
        "source_uri": "memory://span-1",
        "content": "Atlas source evidence",
        "candidate_source": "product_lexical",
        "source_span_ids": ["span-1"],
    }
    assert pack_fixture.repository.calls == [("span-1", pack_fixture.context.scope, True)]


def test_product_pack_hydrates_selected_fact_and_view_from_scoped_repository() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    span = _span(
        "span-structured",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Trusted source evidence",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-1",
        scope=scope,
        subject="user",
        predicate="prefers",
        object="Qdrant",
        text="Repository-backed fact text",
        category="preference",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[span.span_id],
    )
    view = CurrentView(
        view_id="view-1",
        scope=scope,
        view_type="preference",
        subject="user",
        text="Repository-backed current view",
        state_json={"value": "Qdrant"},
        source_fact_ids=[fact.fact_id],
        source_event_ids=[],
        source_span_ids=[span.span_id],
        confidence=0.95,
    )
    repository = SpanRepository([span], facts=[fact], views=[view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-structured",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                fact.fact_id,
                candidate_type="fact",
                text="forged candidate fact text",
                source_span_ids=[span.span_id],
            ),
            _candidate(
                view.view_id,
                candidate_type="view",
                text="forged candidate view text",
                source_span_ids=[span.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What do I prefer?", limit=2),
        result,
        token_budget=1200,
    )

    assert pack.facts == [
        {
            "id": "fact-1",
            "text": "Repository-backed fact text",
            "candidate_source": "product_lexical",
            "source_span_ids": ["span-structured"],
        }
    ]
    assert pack.current_views == [
        {
            "id": "view-1",
            "text": "Repository-backed current view",
            "candidate_source": "product_lexical",
            "source_span_ids": ["span-structured"],
        }
    ]
    assert "forged candidate" not in repr(pack)
    assert repository.fact_calls == [("fact-1", scope, True)]
    assert repository.view_calls == [(scope, None, True)]


def test_product_pack_requires_selected_hydrated_provenance_for_structured_records() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    span = _span(
        "span-supported",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-unsupported",
        scope=scope,
        subject="user",
        predicate="prefers",
        object="Qdrant",
        text="Repository fact without selected span support",
        category="preference",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[span.span_id],
    )
    repository = SpanRepository([span], facts=[fact])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-unsupported",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                fact.fact_id,
                candidate_type="fact",
                source_span_ids=["missing-selected-span"],
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What do I prefer?", limit=1),
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.facts == []


def test_product_pack_applies_token_budget_to_structured_records() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    span = _span(
        "span-budget",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="support",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-budget",
        scope=scope,
        subject="user",
        predicate="prefers",
        object="Qdrant",
        text="structured record",
        category="preference",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[span.span_id],
    )
    repository = SpanRepository([span], facts=[fact])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-structured-budget",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                fact.fact_id,
                candidate_type="fact",
                source_span_ids=[span.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What do I prefer?", limit=1),
        result,
        token_budget=1,
    )

    assert [record["id"] for record in pack.source_spans] == [span.span_id]
    assert pack.facts == []


def test_product_pack_prioritizes_current_view_bundle_before_stale_evidence() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-budget-stale",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="old Qdrant history",
        scope=scope,
    )
    current = _span(
        "span-budget-current",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Postgres current evidence",
        scope=scope,
    )
    view = CurrentView(
        view_id="view-budget-current",
        scope=scope,
        view_type="active_projects",
        subject="Project Atlas",
        text="Postgres current",
        state_json={"object": "Postgres"},
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[current.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([stale, current], views=[view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-current-view-budget",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-budget-stale", stale.span_id, text=stale.content),
            _candidate(
                view.view_id,
                candidate_type="view",
                text=view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            OrderingMode.RECENCY,
            intent="current_state",
            query_intent={"needs_current_state": True},
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What does Project Atlas currently use?", limit=2),
        result,
        token_budget=4,
    )

    assert [record["id"] for record in pack.source_spans] == [current.span_id]
    assert pack.current_views == []


def test_product_pack_reserves_recency_budget_for_authoritative_current_view() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-budget-history",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="old Qdrant historical evidence",
        scope=scope,
    )
    current = _span(
        "span-budget-authoritative",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Postgres current evidence",
        scope=scope,
    )
    view = CurrentView(
        view_id="view-budget-authoritative",
        scope=scope,
        view_type="active_projects",
        subject="Project Atlas",
        text="Postgres current",
        state_json={"object": "Postgres"},
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[current.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([stale, current], views=[view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-authoritative-view-budget",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-budget-history", stale.span_id, text=stale.content),
            _candidate(
                view.view_id,
                candidate_type="view",
                text=view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            OrderingMode.RECENCY,
            intent="current_state",
            query_intent={"needs_current_state": True},
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What does Project Atlas currently use?", limit=2),
        result,
        token_budget=8,
    )

    assert [record["id"] for record in pack.source_spans] == [current.span_id]
    assert [record["id"] for record in pack.current_views] == [view.view_id]


def test_product_pack_abstains_when_selected_sources_do_not_support_query_targets() -> None:
    scope = Scope(user_id="user-a")
    span = _span(
        "span-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Remember that my database is PostgreSQL.",
        scope=scope,
    )
    repository = SpanRepository([span])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-irrelevant",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-database",
                span.span_id,
                text=span.content,
                source="product_lexical+product_vector",
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            query_intent={"target_terms": ["kubernetes", "cluster", "name"]},
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What is my Kubernetes cluster name?", limit=1),
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.answer_policy == "abstain_if_not_supported"
    assert pack.coverage["coverage_insufficient"] is True


def test_product_pack_does_not_treat_arbitrary_selected_span_as_aggregate_item() -> None:
    scope = Scope(user_id="user-a")
    span = _span(
        "span-database-count",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Remember that my database is PostgreSQL.",
        scope=scope,
    )
    repository = SpanRepository([span])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-irrelevant-count",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-database-count",
                span.span_id,
                text=span.content,
                source="product_entity",
                metadata={"speaker": "user"},
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            query_intent={
                "evidence_scope": "local_or_best_match",
                "target_terms": ["kubernetes", "clusters"],
                "aggregation": {
                    "operation": "count",
                    "distinct": False,
                },
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="How many Kubernetes clusters do I have?", limit=1),
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.answer_policy == "abstain_if_not_supported"
    assert pack.coverage["coverage_insufficient"] is True


def test_product_pack_does_not_infer_untyped_multi_session_aggregate_items() -> None:
    scope = Scope(user_id="user-a")
    resume = _span(
        "span-resume",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I focused on adapting my resume to international standards.",
        scope=scope,
    )
    portfolio = _span(
        "span-portfolio",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I also wanted to improve my portfolio project selection.",
        scope=scope,
    )
    coffee = _span(
        "span-morning-coffee",
        timestamp=datetime(2026, 7, 3, tzinfo=timezone.utc),
        content="I drink coffee every morning.",
        scope=scope,
    )
    repository = SpanRepository([resume, portfolio, coffee])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-distinct-aggregate",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=tuple(
            _candidate(
                f"candidate-{span.span_id}",
                span.span_id,
                text=span.content,
                source="product_lexical+product_vector",
                metadata={"speaker": "user"},
            )
            for span in (resume, portfolio, coffee)
        ),
        coverage={},
        trace={},
        plan=_plan(
            query_intent={
                "evidence_scope": "multi_session",
                "target_terms": ["planning", "areas", "sessions"],
                "aggregation": {
                    "operation": "count_distinct",
                    "distinct": True,
                },
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(
            query="How many different planning areas did I mention across my sessions?",
            limit=2,
        ),
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.answer_policy == "abstain_if_not_supported"


def test_product_pack_uses_active_current_view_provenance_for_recency() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-stale",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="For Project Atlas, I initially prefer Qdrant for retrieval experiments.",
        scope=scope,
    )
    current = _span(
        "span-current",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I switched Project Atlas retrieval to Postgres pgvector for production.",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-current",
        scope=scope,
        subject="user",
        predicate="switched",
        object="Postgres pgvector",
        text="User switched Project Atlas retrieval to Postgres pgvector for production.",
        category="preference",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[current.span_id],
    )
    view = CurrentView(
        view_id="view-current",
        scope=scope,
        view_type="current_preferences",
        subject="user",
        text=fact.text,
        state_json={"value": "Postgres pgvector"},
        source_fact_ids=[fact.fact_id],
        source_event_ids=[],
        source_span_ids=[current.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([stale, current], facts=[fact], views=[view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-current",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-stale", stale.span_id, text=stale.content),
            _candidate("candidate-current", current.span_id, text=current.content),
            _candidate(
                fact.fact_id,
                candidate_type="fact",
                text=fact.text,
                source_span_ids=[current.span_id],
            ),
            _candidate(
                view.view_id,
                candidate_type="view",
                text=view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            OrderingMode.RECENCY,
            intent="current_state",
            entities=("Project", "Atlas"),
            query_intent={
                "target_terms": ["retrieval", "backend", "project", "atlas", "use"],
                "needs_current_state": True,
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(
            query="What retrieval backend does Project Atlas currently use?",
            limit=4,
        ),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert pack.current_views == [
        {
            "id": view.view_id,
            "text": fact.text,
            "candidate_source": "product_lexical",
            "source_span_ids": [current.span_id],
        }
    ]
    assert [record["id"] for record in pack.facts] == [fact.fact_id]


def test_product_pack_ignores_unselected_current_view_for_recency_gating() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    deadline = _span(
        "span-atlas-deadline",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Project Atlas deadline is August 1.",
        scope=scope,
    )
    backend = _span(
        "span-atlas-backend",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Project Atlas currently uses Postgres.",
        scope=scope,
    )
    backend_view = CurrentView(
        view_id="view-atlas-backend",
        scope=scope,
        view_type="active_projects",
        subject="Project Atlas",
        text=backend.content,
        state_json={"backend": "Postgres"},
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[backend.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([deadline, backend], views=[backend_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-unselected-view",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-atlas-deadline",
                deadline.span_id,
                text=deadline.content,
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            OrderingMode.RECENCY,
            intent="current_state",
            query_intent={
                "target_terms": ["project", "atlas", "deadline"],
                "needs_current_state": True,
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What is the current Project Atlas deadline?", limit=1),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [deadline.span_id]


def test_product_pack_preserves_target_evidence_with_unrelated_current_view() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    deadline = _span(
        "span-selected-deadline",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Project Atlas deadline is August 1.",
        scope=scope,
    )
    backend = _span(
        "span-selected-backend",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Project Atlas currently uses Postgres.",
        scope=scope,
    )
    backend_view = CurrentView(
        view_id="view-selected-backend",
        scope=scope,
        view_type="active_projects",
        subject="Project Atlas",
        text=backend.content,
        state_json={"backend": "Postgres"},
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[backend.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([deadline, backend], views=[backend_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-selected-wrong-slot-view",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-selected-deadline", deadline.span_id, text=deadline.content),
            _candidate(
                backend_view.view_id,
                candidate_type="view",
                text=backend_view.text,
                source_span_ids=[backend.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            OrderingMode.RECENCY,
            intent="current_state",
            entities=("Project", "Atlas"),
            query_intent={
                "target_terms": ["project", "atlas", "deadline"],
                "needs_current_state": True,
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What is the current Project Atlas deadline?", limit=2),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        deadline.span_id,
        backend.span_id,
    ]


def test_product_pack_preserves_target_evidence_without_planner_entities() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    deadline = _span(
        "span-generic-project-deadline",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="The current project deadline is August 1.",
        scope=scope,
    )
    backend = _span(
        "span-generic-project-backend",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="The project currently uses Postgres.",
        scope=scope,
    )
    backend_view = CurrentView(
        view_id="view-generic-project-backend",
        scope=scope,
        view_type="active_projects",
        subject="project",
        text=backend.content,
        state_json={"backend": "Postgres"},
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[backend.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([deadline, backend], views=[backend_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-generic-project-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(query="What is the current project deadline?", limit=2)
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ()
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-generic-project-deadline",
                deadline.span_id,
                text=deadline.content,
            ),
            _candidate(
                backend_view.view_id,
                candidate_type="view",
                text=backend_view.text,
                source_span_ids=[backend.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        deadline.span_id,
        backend.span_id,
    ]


def test_product_pack_does_not_infer_untyped_slot_without_entities() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    owner = _span(
        "span-generic-project-owner",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="The current project owner is Priya.",
        scope=scope,
    )
    backend = _span(
        "span-generic-project-backend-owner-query",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I switched the project database from Qdrant to PostgreSQL.",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-generic-project-backend-owner-query",
        scope=scope,
        subject="user",
        predicate="switched_to",
        object="PostgreSQL",
        text="User switched the project database to PostgreSQL.",
        category="project_state",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[backend.span_id],
        metadata={
            "topic_terms": [
                "switched",
                "project",
                "database",
                "qdrant",
                "postgresql",
            ]
        },
    )
    backend_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([owner, backend], views=[backend_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-generic-project-owner-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(query="What is the current project owner?", limit=2)
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ()
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-generic-project-owner",
                owner.span_id,
                text=owner.content,
            ),
            _candidate(
                backend_view.view_id,
                candidate_type="view",
                text=backend_view.text,
                source_span_ids=[backend.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        owner.span_id,
        backend.span_id,
    ]


def test_product_pack_orders_current_view_source_before_history_without_entities() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-generic-project-stale-backend",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="The project database initially used Qdrant.",
        scope=scope,
    )
    current = _span(
        "span-generic-project-current-backend",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I switched the database from Qdrant to PostgreSQL.",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-generic-project-current-database",
        scope=scope,
        subject="user",
        predicate="switched_to",
        object="PostgreSQL",
        text="User switched the database to PostgreSQL.",
        category="project_state",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[current.span_id],
    )
    current_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([stale, current], views=[current_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-generic-project-backend-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(query="What database do I currently use?", limit=2)
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ()
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-generic-project-stale-backend",
                stale.span_id,
                text=stale.content,
            ),
            _candidate(
                current_view.view_id,
                candidate_type="view",
                text=current_view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert [record["id"] for record in pack.current_views] == [
        current_view.view_id
    ]
    assert pack.current_views[0]["source_span_ids"] == [current.span_id]


def test_product_pack_orders_current_view_source_before_history_with_entities() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-atlas-stale-structured-backend",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Project Atlas initially used the Qdrant backend.",
        scope=scope,
    )
    current = _span(
        "span-atlas-current-structured-backend",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Project Atlas currently uses PostgreSQL.",
        scope=scope,
    )
    current_view = CurrentView(
        view_id="view-atlas-current-structured-backend",
        scope=scope,
        view_type="active_projects",
        subject="Project Atlas",
        text=current.content,
        state_json={"backend": "PostgreSQL"},
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[current.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([stale, current], views=[current_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-atlas-structured-backend-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="What is the current Project Atlas backend?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("Project", "Atlas")
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-atlas-stale-structured-backend",
                stale.span_id,
                text=stale.content,
            ),
            _candidate(
                current_view.view_id,
                candidate_type="view",
                text=current_view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert [record["id"] for record in pack.current_views] == [
        current_view.view_id
    ]
    assert pack.current_views[0]["source_span_ids"] == [current.span_id]


def test_product_pack_preserves_entity_evidence_with_selected_current_view() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    owner = _span(
        "span-postgresql-project-owner",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Priya is the current owner of the PostgreSQL project.",
        scope=scope,
    )
    database = _span(
        "span-postgresql-project-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I switched the PostgreSQL project database from Qdrant to Postgres.",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-postgresql-project-database",
        scope=scope,
        subject="user",
        predicate="switched_to",
        object="Postgres",
        text="User switched the PostgreSQL project database to Postgres.",
        category="project_state",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[database.span_id],
        metadata={
            "topic_terms": [
                "switched",
                "postgresql",
                "project",
                "database",
                "qdrant",
                "postgres",
            ]
        },
    )
    database_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([owner, database], views=[database_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-postgresql-owner-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="Who is the current owner of the PostgreSQL project?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("PostgreSQL",)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-postgresql-project-owner",
                owner.span_id,
                text=owner.content,
            ),
            _candidate(
                database_view.view_id,
                candidate_type="view",
                text=database_view.text,
                source_span_ids=[database.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        owner.span_id,
        database.span_id,
    ]


def test_product_pack_does_not_gate_acme_service_owner_with_database_view() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    owner = _span(
        "span-acme-service-owner",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Priya is the current owner of the Acme service.",
        scope=scope,
    )
    database = _span(
        "span-acme-service-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I switched the Acme service database from Qdrant to Postgres.",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-acme-service-database",
        scope=scope,
        subject="user",
        predicate="switched_to",
        object="Postgres",
        text="User switched the Acme service database to Postgres.",
        category="project_state",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[database.span_id],
        metadata={
            "topic_terms": [
                "switched",
                "acme",
                "service",
                "database",
                "qdrant",
                "postgres",
            ]
        },
    )
    database_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([owner, database], views=[database_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-acme-service-owner-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="Who is the current owner of the Acme service?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("Acme",)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-acme-service-owner",
                owner.span_id,
                text=owner.content,
            ),
            _candidate(
                database_view.view_id,
                candidate_type="view",
                text=database_view.text,
                source_span_ids=[database.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        owner.span_id,
        database.span_id,
    ]


def test_product_pack_orders_acme_service_current_database_before_history() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-acme-service-stale-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="The Acme service database previously used Qdrant.",
        scope=scope,
    )
    current = _span(
        "span-acme-service-current-database",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I switched the Acme service database to Postgres.",
        scope=scope,
    )
    fact = MemoryFact(
        fact_id="fact-acme-service-current-database",
        scope=scope,
        subject="user",
        predicate="switched_to",
        object="Postgres",
        text="User switched the Acme service database to Postgres.",
        category="project_state",
        confidence=0.9,
        salience=0.8,
        source_span_ids=[current.span_id],
        metadata={
            "topic_terms": [
                "switched",
                "acme",
                "service",
                "database",
                "qdrant",
                "postgres",
            ]
        },
    )
    current_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([stale, current], views=[current_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-acme-service-database-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="What database does the Acme service currently use?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("Acme",)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-acme-service-stale-database",
                stale.span_id,
                text=stale.content,
            ),
            _candidate(
                current_view.view_id,
                candidate_type="view",
                text=current_view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert [record["id"] for record in pack.current_views] == [
        current_view.view_id
    ]
    assert pack.current_views[0]["source_span_ids"] == [current.span_id]


def test_product_pack_orders_latest_single_entity_backend_before_history() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-atlas-stale-latest-backend",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="The Atlas backend previously used Qdrant.",
        scope=scope,
    )
    current = _span(
        "span-atlas-current-latest-backend",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I switched the Atlas backend from Qdrant to Postgres.",
        scope=scope,
    )
    extracted = RuleBasedExtractor().extract([current], [], current.timestamp)
    fact = candidate_to_fact(
        scope,
        next(item for item in extracted if item.candidate_type == "fact"),
        current.timestamp,
    )
    current_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([stale, current], views=[current_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-atlas-latest-backend",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(query="What is the latest Atlas backend?", limit=2)
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("Atlas",)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-atlas-stale-latest-backend",
                stale.span_id,
                text=stale.content,
            ),
            _candidate(
                current_view.view_id,
                candidate_type="view",
                text=current_view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert [record["id"] for record in pack.current_views] == [
        current_view.view_id
    ]
    assert pack.current_views[0]["source_span_ids"] == [current.span_id]


def test_product_pack_does_not_gate_acme_cloud_owner_with_database_view() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    owner = _span(
        "span-acme-cloud-service-owner",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Priya is the current owner of the Acme Cloud service.",
        scope=scope,
    )
    database = _span(
        "span-acme-cloud-service-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I switched the Acme Cloud service database from Qdrant to Postgres.",
        scope=scope,
    )
    extracted = RuleBasedExtractor().extract([database], [], database.timestamp)
    fact = candidate_to_fact(
        scope,
        next(item for item in extracted if item.candidate_type == "fact"),
        database.timestamp,
    )
    database_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([owner, database], views=[database_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-acme-cloud-service-owner",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="Who is the current owner of the Acme Cloud service?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("Acme", "Cloud")
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-acme-cloud-service-owner",
                owner.span_id,
                text=owner.content,
            ),
            _candidate(
                database_view.view_id,
                candidate_type="view",
                text=database_view.text,
                source_span_ids=[database.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        owner.span_id,
        database.span_id,
    ]


def test_product_pack_orders_acme_cloud_current_database_before_history() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-acme-cloud-stale-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="The Acme Cloud service database previously used Qdrant.",
        scope=scope,
    )
    current = _span(
        "span-acme-cloud-current-database",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I switched the Acme Cloud service database to Postgres.",
        scope=scope,
    )
    extracted = RuleBasedExtractor().extract([current], [], current.timestamp)
    fact = candidate_to_fact(
        scope,
        next(item for item in extracted if item.candidate_type == "fact"),
        current.timestamp,
    )
    current_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([stale, current], views=[current_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-acme-cloud-service-database",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="What database does the Acme Cloud service currently use?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("Acme", "Cloud")
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-acme-cloud-stale-database",
                stale.span_id,
                text=stale.content,
            ),
            _candidate(
                current_view.view_id,
                candidate_type="view",
                text=current_view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert [record["id"] for record in pack.current_views] == [
        current_view.view_id
    ]
    assert pack.current_views[0]["source_span_ids"] == [current.span_id]


def test_product_pack_does_not_use_entity_type_as_current_property() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    ownership = _span(
        "span-acme-owned-service",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Acme currently owns the Billing service.",
        scope=scope,
    )
    database = _span(
        "span-acme-service-database-owner-query",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I switched the Acme service database from Qdrant to Postgres.",
        scope=scope,
    )
    extracted = RuleBasedExtractor().extract([database], [], database.timestamp)
    fact = candidate_to_fact(
        scope,
        next(item for item in extracted if item.candidate_type == "fact"),
        database.timestamp,
    )
    database_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([ownership, database], views=[database_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-acme-owned-service",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="What service does Acme currently own?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-acme-owned-service",
                ownership.span_id,
                text=ownership.content,
            ),
            _candidate(
                database_view.view_id,
                candidate_type="view",
                text=database_view.text,
                source_span_ids=[database.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        ownership.span_id,
        database.span_id,
    ]


def test_product_pack_does_not_use_property_modifier_as_current_property() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    backend = _span(
        "span-atlas-retrieval-backend",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content=(
            "I switched Project Atlas retrieval from Qdrant to Postgres pgvector "
            "for production."
        ),
        scope=scope,
    )
    timeout = _span(
        "span-atlas-retrieval-timeout",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content=(
            "I switched Project Atlas retrieval timeout from 30 seconds to 60 seconds."
        ),
        scope=scope,
    )
    extracted = RuleBasedExtractor().extract([timeout], [], timeout.timestamp)
    fact = candidate_to_fact(
        scope,
        next(item for item in extracted if item.candidate_type == "fact"),
        timeout.timestamp,
    )
    timeout_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([backend, timeout], views=[timeout_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-atlas-retrieval-timeout",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="What retrieval backend does Project Atlas currently use?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-atlas-retrieval-backend",
                backend.span_id,
                text=backend.content,
            ),
            _candidate(
                timeout_view.view_id,
                candidate_type="view",
                text=timeout_view.text,
                source_span_ids=[timeout.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        backend.span_id,
        timeout.span_id,
    ]


def test_product_pack_orders_extracted_current_preference_before_history() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-stale-preference-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I previously used Qdrant as my database.",
        scope=scope,
    )
    current = _span(
        "span-current-preference-database",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I use PostgreSQL as my database.",
        scope=scope,
    )
    extracted = RuleBasedExtractor().extract(
        [current],
        [],
        current.timestamp,
    )
    candidate = next(
        item for item in extracted if item.candidate_type == "fact"
    )
    fact = candidate_to_fact(scope, candidate, current.timestamp)
    assert fact.metadata["topic_terms"] == ["postgresql", "database"]
    current_view = ViewBuilder().build_current_views(scope, [fact], set())[0]
    repository = SpanRepository([stale, current], views=[current_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-preference-database-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(query="What database do I currently use?", limit=2)
    plan = ProductQueryPlanner().plan(request)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-stale-preference-database",
                stale.span_id,
                text=stale.content,
            ),
            _candidate(
                current_view.view_id,
                candidate_type="view",
                text=current_view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert [record["id"] for record in pack.current_views] == [
        current_view.view_id
    ]
    assert pack.current_views[0]["source_span_ids"] == [current.span_id]


def test_product_pack_orders_legacy_current_view_source_before_history() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    stale = _span(
        "span-stale-legacy-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I previously used Qdrant as my database.",
        scope=scope,
    )
    current = _span(
        "span-current-legacy-database",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I use PostgreSQL as my database.",
        scope=scope,
    )
    legacy_view = CurrentView(
        view_id="view-current-legacy-database",
        scope=scope,
        view_type="current_preferences",
        subject="user",
        text="User prefers PostgreSQL as my database.",
        state_json={
            "category": "preference",
            "object": "PostgreSQL as my database",
        },
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[current.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([stale, current], views=[legacy_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-legacy-database-slot",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(query="What database do I currently use?", limit=2)
    plan = ProductQueryPlanner().plan(request)
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-stale-legacy-database",
                stale.span_id,
                text=stale.content,
            ),
            _candidate(
                legacy_view.view_id,
                candidate_type="view",
                text=legacy_view.text,
                source_span_ids=[current.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        current.span_id,
        stale.span_id,
    ]
    assert [record["id"] for record in pack.current_views] == [
        legacy_view.view_id
    ]
    assert pack.current_views[0]["source_span_ids"] == [current.span_id]


def test_product_pack_does_not_gate_recency_with_wrong_entity_view() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    atlas = _span(
        "span-atlas-deadline",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Project Atlas deadline is August 1.",
        scope=scope,
    )
    borealis = _span(
        "span-borealis-deadline",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Project Borealis deadline is September 1.",
        scope=scope,
    )
    borealis_view = CurrentView(
        view_id="view-borealis-deadline",
        scope=scope,
        view_type="active_projects",
        subject="Project Borealis",
        text=borealis.content,
        state_json={"deadline": "September 1"},
        source_fact_ids=[],
        source_event_ids=[],
        source_span_ids=[borealis.span_id],
        confidence=0.9,
    )
    repository = SpanRepository([atlas, borealis], views=[borealis_view])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-wrong-entity-view",
        deadline=None,
        include_session=True,
    )
    request = SearchRequest(
        query="What is the current Project Atlas deadline?",
        limit=2,
    )
    plan = ProductQueryPlanner().plan(request)
    assert plan.entities == ("Project", "Atlas")
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-atlas-deadline",
                atlas.span_id,
                text=atlas.content,
            ),
            _candidate(
                borealis_view.view_id,
                candidate_type="view",
                text=borealis_view.text,
                source_span_ids=[borealis.span_id],
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert atlas.span_id in [span["id"] for span in pack.source_spans]


def test_product_pack_drops_redundant_temporal_endpoint_topic() -> None:
    scope = Scope(user_id="user-a")
    feature = _span(
        "span-feature",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="The transaction management features are completed by January 15, 2024.",
        scope=scope,
    )
    deadline = _span(
        "span-deadline",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="The final deployment deadline for the budget tracker is March 15, 2024.",
        scope=scope,
    )
    other_topic = _span(
        "span-screenplay",
        timestamp=datetime(2026, 7, 3, tzinfo=timezone.utc),
        content="The screenplay launch preparation has a final deployment deadline of March 20, 2024.",
        scope=scope,
    )
    repository = SpanRepository([feature, deadline, other_topic])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-temporal-topic",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-feature", feature.span_id, text=feature.content),
            _candidate("candidate-deadline", deadline.span_id, text=deadline.content),
            _candidate("candidate-screenplay", other_topic.span_id, text=other_topic.content),
        ),
        coverage={},
        trace={},
        plan=_plan(
            intent="temporal",
            query_intent={
                "target_terms": [
                    "weeks",
                    "finishing",
                    "transaction",
                    "management",
                    "features",
                    "final",
                    "deployment",
                    "deadline",
                ],
                "temporal": {"requires_duration": True},
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(
            query=(
                "How many weeks do I have between finishing the transaction management "
                "features and the final deployment deadline?"
            ),
            limit=6,
        ),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        feature.span_id,
        deadline.span_id,
    ]


def test_product_pack_keeps_distinct_duration_endpoint_roles() -> None:
    scope = Scope(user_id="user-a")
    started = _span(
        "span-atlas-start",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Project Atlas began on January 1, 2026.",
        scope=scope,
    )
    ended = _span(
        "span-atlas-end",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Project Atlas ended on February 1, 2026.",
        scope=scope,
    )
    repository = SpanRepository([started, ended])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-duration-endpoints",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-atlas-start", started.span_id, text=started.content),
            _candidate("candidate-atlas-end", ended.span_id, text=ended.content),
        ),
        coverage={},
        trace={},
        plan=_plan(
            intent="temporal",
            query_intent={
                "target_terms": ["weeks", "project", "atlas", "take"],
                "temporal": {
                    "requires_duration": True,
                    "endpoint_roles": ["start", "end"],
                },
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="How many weeks did Project Atlas take?", limit=2),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        started.span_id,
        ended.span_id,
    ]


def test_product_pack_keeps_unknown_duration_roles_by_distinct_provenance() -> None:
    scope = Scope(user_id="user-a")
    announced = _span(
        "span-atlas-announced",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Project Atlas was announced on January 1, 2026.",
        scope=scope,
    )
    shipped = _span(
        "span-atlas-shipped",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="Project Atlas shipped on February 1, 2026.",
        scope=scope,
    )
    repository = SpanRepository([announced, shipped])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-duration-unknown-endpoints",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-atlas-announced", announced.span_id, text=announced.content),
            _candidate("candidate-atlas-shipped", shipped.span_id, text=shipped.content),
        ),
        coverage={},
        trace={},
        plan=_plan(
            intent="temporal",
            entities=("Project", "Atlas"),
            query_intent={
                "target_terms": ["weeks", "project", "atlas", "take"],
                "temporal": {
                    "requires_duration": True,
                    "endpoint_roles": ["start", "end"],
                },
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="How many weeks did Project Atlas take?", limit=2),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        announced.span_id,
        shipped.span_id,
    ]


def test_product_pack_does_not_use_cjk_pronoun_as_query_support() -> None:
    scope = Scope(user_id="user-a")
    span = _span(
        "span-coffee",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="我目前使用咖啡机。",
        scope=scope,
    )
    repository = SpanRepository([span])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-cjk-pronoun",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-coffee",
                span.span_id,
                text=span.content,
                source="product_lexical+product_vector",
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            query_intent={"target_terms": ["我目前使用什么数据库"]},
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="我目前使用什么数据库？", limit=1),
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.answer_policy == "abstain_if_not_supported"


def test_product_pack_retains_meaningful_cjk_query_support() -> None:
    scope = Scope(user_id="user-a")
    span = _span(
        "span-postgres-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="我目前使用 PostgreSQL 数据库。",
        scope=scope,
    )
    repository = SpanRepository([span])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-cjk-database",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-postgres-database",
                span.span_id,
                text=span.content,
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            query_intent={"target_terms": ["我目前使用什么数据库"]},
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="我目前使用什么数据库？", limit=1),
        result,
        token_budget=1200,
    )

    assert [record["id"] for record in pack.source_spans] == [span.span_id]
    assert pack.answer_policy == "answer_with_evidence_or_abstain"


def test_product_pack_does_not_use_generic_action_term_as_query_support() -> None:
    scope = Scope(user_id="user-a")
    span = _span(
        "span-coffee-machine",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I use a coffee machine every day.",
        scope=scope,
    )
    repository = SpanRepository([span])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-generic-use",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-coffee-machine",
                span.span_id,
                text=span.content,
                source="product_lexical+product_vector",
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            query_intent={"target_terms": ["use", "database"]},
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What database do I currently use?", limit=1),
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.answer_policy == "abstain_if_not_supported"


def test_product_pack_abstains_when_all_query_support_terms_are_generic() -> None:
    scope = Scope(user_id="user-a")
    span = _span(
        "span-generic-current-use",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I currently use a coffee machine.",
        scope=scope,
    )
    repository = SpanRepository([span])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-all-generic-support",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-generic-current-use",
                span.span_id,
                text=span.content,
            ),
        ),
        coverage={},
        trace={},
        plan=_plan(
            query_intent={"target_terms": ["currently", "use"]},
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="What do I currently use?", limit=1),
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.answer_policy == "abstain_if_not_supported"


def test_product_pack_abstains_for_explicit_empty_planner_targets() -> None:
    scope = Scope(user_id="user-a")
    span = _span(
        "span-empty-target-database",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="Remember that my database is PostgreSQL.",
        scope=scope,
    )
    repository = SpanRepository([span])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-empty-planner-targets",
        deadline=None,
        include_session=False,
    )
    request = SearchRequest(query="What is it?", limit=1)
    plan = ProductQueryPlanner().plan(request)
    assert plan.query_intent["target_terms"] == []
    result = RetrievalResult(
        candidates=(
            _candidate(
                "candidate-empty-target-database",
                span.span_id,
                text=span.content,
            ),
        ),
        coverage={},
        trace={},
        plan=plan,
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        request,
        result,
        token_budget=1200,
    )

    assert pack.source_spans == []
    assert pack.answer_policy == "abstain_if_not_supported"


def test_product_pack_keeps_only_supported_chronology_topics() -> None:
    scope = Scope(user_id="user-a")
    transaction = _span(
        "span-transaction",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content="I'm implementing transaction CRUD response handling for my budget tracker.",
        scope=scope,
    )
    deployment = _span(
        "span-deployment",
        timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content="I'm configuring Render deployment with Gunicorn workers for the budget tracker.",
        scope=scope,
    )
    stress = _span(
        "span-stress",
        timestamp=datetime(2026, 7, 3, tzinfo=timezone.utc),
        content="I started managing stress by setting no-work Sundays and reducing burnout.",
        scope=scope,
    )
    finance = _span(
        "span-finance",
        timestamp=datetime(2026, 7, 4, tzinfo=timezone.utc),
        content="I handled financial concerns by tracking rent, groceries, and emergency savings.",
        scope=scope,
    )
    repository = SpanRepository([transaction, deployment, stress, finance])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-chronology-topic",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=tuple(
            _candidate(
                f"candidate-{span.span_id}",
                span.span_id,
                text=span.content,
                timeline_index=index,
            )
            for index, span in enumerate(
                (transaction, deployment, stress, finance),
                start=1,
            )
        ),
        coverage={},
        trace={},
        plan=_plan(
            OrderingMode.CHRONOLOGICAL,
            intent="chronology",
            query_intent={
                "target_terms": ["managing", "stress", "financial", "concerns"],
                "temporal": {"requires_order": True},
            },
        ),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(
            query=(
                "Walk me through the order in which I brought up ways of managing stress "
                "and financial concerns."
            ),
            limit=8,
        ),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == [
        stress.span_id,
        finance.span_id,
    ]


def test_product_pack_omits_cross_user_and_cross_session_spans() -> None:
    scope = Scope(user_id="user-a", session_id="session-a")
    repository = SpanRepository(
        [
            _span("allowed", timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc)),
            _span(
                "cross-user",
                timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc),
                scope=Scope(user_id="user-b", session_id="session-a"),
            ),
            _span(
                "cross-session",
                timestamp=datetime(2026, 7, 3, tzinfo=timezone.utc),
                scope=Scope(user_id="user-a", session_id="session-b"),
            ),
        ]
    )
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-scope",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("allowed", "allowed"),
            _candidate("cross-user", "cross-user"),
            _candidate("cross-session", "cross-session"),
        ),
        coverage={},
        trace={},
        plan=_plan(),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="Atlas", limit=3),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == ["allowed"]


@pytest.fixture
def chronology_pack_fixture() -> PackFixture:
    scope = Scope(user_id="user-a", session_id="session-a")
    repository = SpanRepository(
        [
            _span("span-late", timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc)),
            _span("span-early", timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc)),
        ]
    )
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-chronology",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-late", "span-late", timeline_index=2),
            _candidate("candidate-early", "span-early", timeline_index=1),
        ),
        coverage={},
        trace={},
        plan=_plan(OrderingMode.CHRONOLOGICAL),
    )
    return PackFixture(
        builder=ProductEvidencePackBuilder(repository),
        context=context,
        request=SearchRequest(query="Order Atlas changes", limit=4),
        result=result,
        repository=repository,
    )


def test_product_pack_orders_chronology_by_timeline_index(
    chronology_pack_fixture: PackFixture,
) -> None:
    pack = chronology_pack_fixture.builder.build(
        chronology_pack_fixture.context,
        chronology_pack_fixture.request,
        chronology_pack_fixture.result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == ["span-early", "span-late"]


@pytest.mark.parametrize(
    ("ordering", "expected_ids"),
    [
        (OrderingMode.RECENCY, ["span-new", "span-old"]),
        (OrderingMode.RELEVANCE, ["span-old", "span-new"]),
    ],
)
def test_product_pack_applies_product_ordering(
    ordering: OrderingMode,
    expected_ids: list[str],
) -> None:
    repository = SpanRepository(
        [
            _span("span-old", timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc)),
            _span("span-new", timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc)),
        ]
    )
    context = RetrievalContext(
        scope=Scope(user_id="user-a", session_id="session-a"),
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-ordering",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-old", "span-old"),
            _candidate("candidate-new", "span-new"),
        ),
        coverage={},
        trace={},
        plan=_plan(ordering),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="Atlas", limit=2),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == expected_ids


def test_product_pack_deduplicates_multiple_source_ids_in_engine_rank_order() -> None:
    repository = SpanRepository(
        [
            _span(f"span-{index}", timestamp=datetime(2026, 7, index, tzinfo=timezone.utc))
            for index in range(1, 4)
        ]
    )
    context = RetrievalContext(
        scope=Scope(user_id="user-a", session_id="session-a"),
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-dedupe",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("candidate-a", source_span_ids=["span-1", "span-2"]),
            _candidate("candidate-b", source_span_ids=["span-2", "span-3"]),
        ),
        coverage={},
        trace={},
        plan=_plan(),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="Atlas", limit=2),
        result,
        token_budget=1200,
    )

    assert [span["id"] for span in pack.source_spans] == ["span-1", "span-2", "span-3"]
    assert [call[0] for call in repository.calls] == ["span-1", "span-2", "span-3"]


@pytest.fixture
def empty_pack_fixture() -> PackFixture:
    scope = Scope(user_id="user-a")
    repository = SpanRepository([])
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-empty",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(_candidate("missing", "missing-span"),),
        coverage={"query_type": "legacy", "category": "benchmark"},
        trace={"query_type": "legacy"},
        plan=_plan(),
    )
    return PackFixture(
        builder=ProductEvidencePackBuilder(repository),
        context=context,
        request=SearchRequest(query="Unsupported source", limit=1),
        result=result,
        repository=repository,
    )


def test_product_pack_abstains_without_supported_source_evidence(
    empty_pack_fixture: PackFixture,
) -> None:
    pack = empty_pack_fixture.builder.build(
        empty_pack_fixture.context,
        empty_pack_fixture.request,
        empty_pack_fixture.result,
        token_budget=1200,
    )

    assert pack.answer_policy == "abstain_if_not_supported"
    assert "query_type" not in pack.coverage
    assert "category" not in pack.coverage
    assert pack.debug_trace == []


def test_product_pack_observability_uses_only_product_schema(
    pack_fixture: PackFixture,
) -> None:
    result = RetrievalResult(
        candidates=pack_fixture.result.candidates,
        coverage={
            "degraded": False,
            "provider_failures": ["model_unavailable"],
            "provider_counts": {
                "lexical": 1,
                "query_type": "nested-leak",
                "unknown_provider": 99,
            },
            "reranker_unavailable": True,
            "planner_fallback": "invalid_plan",
            "benchmark": {"category": "abstention"},
            "rescue": {"preservation": "legacy"},
            "unknown": "injected",
        },
        trace={
            "stages": ["plan", "recall", "fusion", "selection", "rescue"],
            "mode": "fast",
            "intent": "factual",
            "providers": [
                {
                    "kind": "lexical",
                    "count": 1,
                    "elapsed_ms": 0.5,
                    "failure_code": None,
                    "category": "nested-leak",
                }
            ],
            "filtered_count": 0,
            "selected_ids": ["hashed-id"],
            "stage_durations_ms": {
                "plan": 0.1,
                "recall": 0.2,
                "fusion": 0.3,
                "selection": 0.4,
                "preservation": 100.0,
            },
            "reranker_failure": "reranker_unavailable",
            "planner_fallback": "invalid_plan",
            "query_type": "legacy",
            "unknown": {"benchmark": True},
        },
        plan=ProductQueryPlan(
            **{
                **pack_fixture.result.plan.__dict__,
                "query_intent": {
                    "answer_shape": "short_answer",
                    "temporal": {
                        "requires_time": False,
                        "requires_order": False,
                        "query_type": "nested-leak",
                    },
                    "aggregation": {
                        "operation": "none",
                        "distinct": False,
                        "rescue": "nested-leak",
                    },
                    "benchmark": {"category": "abstention"},
                },
            }
        ),
    )

    pack = pack_fixture.builder.build(
        pack_fixture.context,
        pack_fixture.request,
        result,
        token_budget=1200,
    )

    assert pack.coverage == {
        "degraded": False,
        "provider_failures": ["model_unavailable"],
        "provider_counts": {"lexical": 1},
        "reranker_unavailable": True,
        "planner_fallback": "invalid_plan",
        "intent": "factual",
        "query_intent": {
            "answer_shape": "short_answer",
            "temporal": {"requires_time": False, "requires_order": False},
            "aggregation": {"operation": "none", "distinct": False},
        },
        "source_span_count": 1,
        "token_budget": 1200,
        "estimated_source_tokens": 3,
        "coverage_insufficient": False,
    }
    assert isinstance(pack.debug_trace, list)
    assert pack.debug_trace
    for entry in pack.debug_trace:
        assert isinstance(entry, dict)
        assert set(entry) <= {
            "stages",
            "mode",
            "intent",
            "providers",
            "filtered_count",
            "selected_ids",
            "stage_durations_ms",
            "reranker_failure",
            "planner_fallback",
        }
    assert pack.debug_trace == [
        {
            "stages": ["plan", "recall", "fusion", "selection"],
            "mode": "fast",
            "intent": "factual",
            "providers": [
                {
                    "kind": "lexical",
                    "count": 1,
                    "elapsed_ms": 0.5,
                    "failure_code": None,
                }
            ],
            "filtered_count": 0,
            "selected_ids": ["hashed-id"],
            "stage_durations_ms": {
                "plan": 0.1,
                "recall": 0.2,
                "fusion": 0.3,
                "selection": 0.4,
            },
            "reranker_failure": "reranker_unavailable",
            "planner_fallback": "invalid_plan",
        }
    ]


def test_product_pack_drops_nested_values_from_product_observability(
    pack_fixture: PackFixture,
) -> None:
    result = RetrievalResult(
        candidates=pack_fixture.result.candidates,
        coverage={
            "degraded": {"query_type": "nested-leak"},
            "provider_failures": ["model_unavailable", {"category": "nested-leak"}],
            "reranker_unavailable": {"rescue": "nested-leak"},
            "planner_fallback": {"benchmark": "nested-leak"},
        },
        trace={
            "stages": ["plan", "rescue"],
            "providers": [
                {
                    "kind": "lexical",
                    "count": {"category": "nested-leak"},
                    "elapsed_ms": {"query_type": "nested-leak"},
                    "failure_code": {"rescue": "nested-leak"},
                }
            ],
        },
        plan=pack_fixture.result.plan,
    )

    pack = pack_fixture.builder.build(
        pack_fixture.context,
        pack_fixture.request,
        result,
        token_budget=1200,
    )

    assert pack.coverage == {
        "provider_failures": ["model_unavailable"],
        "intent": "factual",
        "query_intent": {},
        "source_span_count": 1,
        "token_budget": 1200,
        "estimated_source_tokens": 3,
        "coverage_insufficient": False,
    }
    assert pack.debug_trace == [
        {
            "stages": ["plan"],
            "providers": [{"kind": "lexical"}],
        }
    ]


def test_product_pack_sanitizes_malformed_plan_intent(
    pack_fixture: PackFixture,
) -> None:
    malformed_intent = {
        "benchmark": {"category": "abstention"},
        "query_type": "legacy",
        "rescue": {"preservation": "legacy"},
        "secrets": "Bearer intent-secret",
    }
    result = RetrievalResult(
        candidates=pack_fixture.result.candidates,
        coverage={},
        trace={},
        plan=ProductQueryPlan(
            **{**pack_fixture.result.plan.__dict__, "intent": malformed_intent}
        ),
    )

    pack = pack_fixture.builder.build(
        pack_fixture.context,
        pack_fixture.request,
        result,
        token_budget=1200,
    )

    rendered = repr(pack.coverage["intent"])
    assert isinstance(pack.coverage["intent"], str)
    assert pack.coverage["intent"].startswith("hashed_")
    for leaked_value in (
        "benchmark",
        "category",
        "query_type",
        "rescue",
        "preservation",
        "secrets",
        "intent-secret",
    ):
        assert leaked_value not in rendered


def test_product_pack_stops_before_exceeding_its_token_budget(pack_fixture: PackFixture) -> None:
    pack = pack_fixture.builder.build(
        pack_fixture.context,
        pack_fixture.request,
        pack_fixture.result,
        token_budget=1,
    )

    assert pack.source_spans == []
    assert pack.coverage["estimated_source_tokens"] == 0


def test_product_pack_hydrates_before_stopping_at_first_over_budget_record() -> None:
    scope = Scope(user_id="user-a")
    repository = SpanRepository(
        [
            _span(
                "too-large",
                timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
                content="one two three",
            ),
            _span("would-fit", timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc)),
        ]
    )
    context = RetrievalContext(
        scope=scope,
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-budget",
        deadline=None,
        include_session=False,
    )
    result = RetrievalResult(
        candidates=(
            _candidate("too-large", "too-large"),
            _candidate("would-fit", "would-fit"),
        ),
        coverage={},
        trace={},
        plan=_plan(),
    )

    pack = ProductEvidencePackBuilder(repository).build(
        context,
        SearchRequest(query="Atlas", limit=2),
        result,
        token_budget=2,
    )

    assert pack.source_spans == []
    assert [call[0] for call in repository.calls] == ["too-large", "would-fit"]


def test_product_pack_compacts_over_limit_source_content() -> None:
    original = "Atlas evidence contains a deliberately long explanation that must be compacted."
    repository = SpanRepository(
        [_span("long-span", timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc), content=original)]
    )
    context = RetrievalContext(
        scope=Scope(user_id="user-a", session_id="session-a"),
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-compaction",
        deadline=None,
        include_session=True,
    )
    result = RetrievalResult(
        candidates=(_candidate("candidate-long", "long-span"),),
        coverage={},
        trace={},
        plan=_plan(),
    )

    pack = ProductEvidencePackBuilder(
        repository,
        MemoryConfig(evidence_span_summary_chars=24),
    ).build(
        context,
        SearchRequest(query="Atlas", limit=1),
        result,
        token_budget=1200,
    )

    assert pack.source_spans[0]["content"] == "Atlas evidence contai..."
    assert pack.source_spans[0]["content"] != original


def test_engine_delegates_product_pack_building(pack_fixture: PackFixture) -> None:
    class RecordingPackBuilder:
        def __init__(self) -> None:
            self.calls: list[tuple[RetrievalContext, SearchRequest, RetrievalResult, int]] = []

        def build(
            self,
            context: RetrievalContext,
            request: SearchRequest,
            result: RetrievalResult,
            token_budget: int,
        ) -> object:
            self.calls.append((context, request, result, token_budget))
            return "delegated-pack"

    class NoopRegistry:
        def run(self, *args: object) -> tuple[object, ...]:
            del args
            return ()

    pack_builder = RecordingPackBuilder()
    engine = ProductRetrievalEngine(None, NoopRegistry(), pack_builder=pack_builder)

    pack = engine.build_evidence_pack(
        pack_fixture.context,
        pack_fixture.request,
        pack_fixture.result,
        token_budget=400,
    )

    assert pack == "delegated-pack"
    assert pack_builder.calls == [
        (pack_fixture.context, pack_fixture.request, pack_fixture.result, 400)
    ]


def test_engine_requires_product_pack_builder() -> None:
    class NoopRegistry:
        def run(self, *args: object) -> tuple[object, ...]:
            del args
            return ()

    with pytest.raises(TypeError, match="pack_builder"):
        ProductRetrievalEngine(None, NoopRegistry())
