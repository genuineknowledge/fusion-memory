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
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.product_evidence_pack import ProductEvidencePackBuilder
from fusion_memory.retrieval.product_engine import ProductRetrievalEngine


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
) -> Candidate:
    metadata = {} if timeline_index is None else {"timeline_index": timeline_index}
    return Candidate(
        id=candidate_id,
        type=candidate_type,
        text=text,
        source=source,
        scores={"bm25_score": 0.8},
        source_span_ids=source_span_ids if source_span_ids is not None else [str(span_id)],
        metadata=metadata,
    )


def _plan(ordering: OrderingMode = OrderingMode.RELEVANCE) -> ProductQueryPlan:
    return ProductQueryPlan(
        intent="factual",
        provider_requests=(),
        time_range=None,
        entities=(),
        speaker=None,
        ordering=ordering,
        use_reranker=False,
        query_intent={"target": "Atlas"},
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


def test_product_pack_stops_after_the_first_record_that_exceeds_its_budget() -> None:
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
    assert [call[0] for call in repository.calls] == ["too-large"]


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
