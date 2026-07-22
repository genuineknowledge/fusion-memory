from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from fusion_memory.core.config import MemoryConfig
from fusion_memory.core.models import Candidate, EvidenceSpan, Scope
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
    def __init__(self, spans: list[EvidenceSpan]) -> None:
        self.spans = {span.span_id: span for span in spans}
        self.calls: list[tuple[str, Scope, bool]] = []

    def get_span(
        self,
        span_id: str,
        scope: Scope,
        *,
        include_session: bool = False,
    ) -> EvidenceSpan | None:
        self.calls.append((span_id, scope, include_session))
        return self.spans.get(span_id)


def _span(
    span_id: str,
    *,
    timestamp: datetime,
    content: str = "Atlas source evidence",
) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        scope=Scope(user_id="user-a", session_id="session-a"),
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
    span_id: str,
    *,
    source: str = "product_lexical",
    timeline_index: int | None = None,
) -> Candidate:
    metadata = {} if timeline_index is None else {"timeline_index": timeline_index}
    return Candidate(
        id=candidate_id,
        type="span",
        text="selected candidate",
        source=source,
        scores={"bm25_score": 0.8},
        source_span_ids=[span_id],
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

    assert pack.source_spans[0]["id"] == "span-1"
    assert pack.source_spans[0]["session_id"] == "session-a"
    assert pack.source_spans[0]["candidate_source"] == "product_lexical"
    assert pack.source_spans[0]["source_span_ids"] == ["span-1"]
    assert pack_fixture.repository.calls == [("span-1", pack_fixture.context.scope, True)]


@pytest.fixture
def chronology_pack_fixture() -> PackFixture:
    scope = Scope(user_id="user-a", session_id="session-a")
    repository = SpanRepository(
        [
            _span("span-late", timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc)),
            _span("span-early", timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc)),
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
    assert pack.debug_trace == {}


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
