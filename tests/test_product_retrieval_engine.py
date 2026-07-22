from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from fusion_memory.core.models import Candidate, Scope
from fusion_memory.core.text import stable_hash
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderFailure,
    ProviderKind,
    ProviderRequest,
    RetrievalContext,
    SearchRequest,
)
from fusion_memory.retrieval.engine import RetrievalUnavailable
from fusion_memory.retrieval.product_engine import ProductRetrievalEngine
from fusion_memory.retrieval.product_planner import ProductQueryPlanner
from fusion_memory.retrieval.providers.product_base import ProviderOutcome


def _candidate(
    candidate_id: str,
    text: str,
    *,
    source: str,
    scores: dict[str, float],
) -> Candidate:
    return Candidate(
        id=candidate_id,
        type="span",
        text=text,
        source=source,
        scores=scores,
        source_span_ids=[candidate_id],
        metadata={},
    )


class StaticPlanner:
    def __init__(self, plan: object) -> None:
        self.planned = plan

    def plan(self, request: SearchRequest) -> object:
        return self.planned

    def safe_default(self, request: SearchRequest) -> ProductQueryPlan:
        return ProductQueryPlanner().safe_default(request)


class RaisingPlanner:
    def plan(self, request: SearchRequest) -> ProductQueryPlan:
        raise RuntimeError("planner programming error")

    def safe_default(self, request: SearchRequest) -> ProductQueryPlan:
        raise AssertionError("safe_default must not hide planner errors")


class StaticRegistry:
    def __init__(self, outcomes: tuple[ProviderOutcome, ...]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def run(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan,
    ) -> tuple[ProviderOutcome, ...]:
        self.calls += 1
        return self.outcomes


@dataclass
class EngineFixture:
    engine: ProductRetrievalEngine
    context: RetrievalContext
    request: SearchRequest
    plan: ProductQueryPlan


@pytest.fixture
def engine_fixture() -> EngineFixture:
    plan = ProductQueryPlan(
        intent="factual",
        provider_requests=(
            ProviderRequest(ProviderKind.VECTOR, 4),
            ProviderRequest(ProviderKind.LEXICAL, 4),
        ),
        time_range=None,
        entities=(),
        speaker=None,
        ordering=OrderingMode.RELEVANCE,
        use_reranker=False,
    )
    outcomes = (
        ProviderOutcome(
            provider=ProviderKind.VECTOR,
            candidates=(
                _candidate(
                    "semantic",
                    "private-memory-body semantic",
                    source="product_vector",
                    scores={"semantic_score": 0.8},
                ),
                _candidate(
                    "exact",
                    "private-memory-body exact",
                    source="product_vector",
                    scores={"semantic_score": 0.5},
                ),
            ),
            elapsed_ms=1.25,
        ),
        ProviderOutcome(
            provider=ProviderKind.LEXICAL,
            candidates=(
                _candidate(
                    "exact",
                    "private-memory-body exact",
                    source="product_lexical",
                    scores={"exact_signal": 1.0, "bm25_score": 0.9},
                ),
            ),
            elapsed_ms=0.75,
        ),
    )
    request = SearchRequest(query="private retrieval query", limit=2)
    context = RetrievalContext(
        scope=Scope(user_id="user-a"),
        user_id="user-a",
        now=datetime.now(timezone.utc),
        trace_id="trace-1",
        deadline=None,
        include_session=False,
    )
    return EngineFixture(
        engine=ProductRetrievalEngine(StaticPlanner(plan), StaticRegistry(outcomes)),
        context=context,
        request=request,
        plan=plan,
    )


def _failed_outcome(provider: ProviderKind) -> ProviderOutcome:
    return ProviderOutcome(
        provider=provider,
        candidates=(),
        elapsed_ms=2.0,
        failure=ProviderFailure(provider, "model_unavailable", True),
    )


def test_engine_runs_one_pass_without_post_selection_rescue(engine_fixture: EngineFixture) -> None:
    result = engine_fixture.engine.search(engine_fixture.context, engine_fixture.request)

    assert [candidate.id for candidate in result.candidates] == ["exact", "semantic"]
    assert result.trace["stages"] == ["plan", "recall", "fusion", "selection"]
    assert "rescue" not in repr(result.trace).lower()


def test_engine_degrades_when_one_provider_is_unavailable(engine_fixture: EngineFixture) -> None:
    lexical = _candidate(
        "lexical",
        "private-memory-body lexical",
        source="product_lexical",
        scores={"bm25_score": 0.9},
    )
    engine = ProductRetrievalEngine(
        StaticPlanner(engine_fixture.plan),
        StaticRegistry(
            (
                _failed_outcome(ProviderKind.VECTOR),
                ProviderOutcome(ProviderKind.LEXICAL, (lexical,), 0.5),
            )
        ),
    )

    result = engine.search(engine_fixture.context, engine_fixture.request)

    assert [candidate.id for candidate in result.candidates] == ["lexical"]
    assert result.coverage["degraded"] is True
    assert result.coverage["provider_failures"] == ["model_unavailable"]


def test_engine_raises_when_all_planned_providers_fail(engine_fixture: EngineFixture) -> None:
    engine = ProductRetrievalEngine(
        StaticPlanner(engine_fixture.plan),
        StaticRegistry(
            (
                _failed_outcome(ProviderKind.VECTOR),
                _failed_outcome(ProviderKind.LEXICAL),
            )
        ),
    )

    with pytest.raises(RetrievalUnavailable, match="all planned providers failed"):
        engine.search(engine_fixture.context, engine_fixture.request)


def test_engine_uses_safe_default_for_invalid_plan(engine_fixture: EngineFixture) -> None:
    engine = ProductRetrievalEngine(
        StaticPlanner(object()),
        StaticRegistry(
            (
                ProviderOutcome(
                    ProviderKind.LEXICAL,
                    (
                        _candidate(
                            "lexical",
                            "private-memory-body lexical",
                            source="product_lexical",
                            scores={"bm25_score": 0.9},
                        ),
                    ),
                    0.5,
                ),
            )
        ),
    )

    result = engine.search(engine_fixture.context, engine_fixture.request)

    assert {request.kind for request in result.plan.provider_requests} == {
        ProviderKind.VECTOR,
        ProviderKind.LEXICAL,
    }
    assert result.plan.ordering is OrderingMode.RELEVANCE
    assert result.plan.intent == "factual"
    assert result.plan.use_reranker is False
    assert result.coverage["planner_fallback"] == "invalid_plan"


def test_trace_never_contains_query_memory_text_or_sensitive_configuration(
    engine_fixture: EngineFixture,
) -> None:
    result = engine_fixture.engine.search(engine_fixture.context, engine_fixture.request)

    rendered = repr(result.trace)
    assert engine_fixture.request.query not in rendered
    assert "private-memory-body" not in rendered
    assert "Bearer secret-token" not in rendered
    assert "https://model.internal/v1/rerank" not in rendered
    assert "model-api-key" not in rendered
    assert result.trace["selected_ids"] == [stable_hash("exact"), stable_hash("semantic")]
    assert result.trace["providers"] == [
        {"kind": "vector", "count": 2, "elapsed_ms": 1.25, "failure_code": None},
        {"kind": "lexical", "count": 1, "elapsed_ms": 0.75, "failure_code": None},
    ]


def test_engine_propagates_planner_exceptions(engine_fixture: EngineFixture) -> None:
    engine = ProductRetrievalEngine(RaisingPlanner(), StaticRegistry(()))

    with pytest.raises(RuntimeError, match="planner programming error"):
        engine.search(engine_fixture.context, engine_fixture.request)


def test_engine_keeps_pre_rerank_selection_when_endpoint_is_unavailable(
    engine_fixture: EngineFixture,
) -> None:
    class UnavailableReranker:
        def score(self, query: str, documents: list[str]) -> list[float]:
            raise EndpointUnavailable("https://model.internal Bearer secret-token")

    plan = ProductQueryPlan(**{**engine_fixture.plan.__dict__, "use_reranker": True})
    engine = ProductRetrievalEngine(
        StaticPlanner(plan),
        StaticRegistry(engine_fixture.engine.registry.outcomes),
        reranker=UnavailableReranker(),
    )

    result = engine.search(engine_fixture.context, engine_fixture.request)

    assert [candidate.id for candidate in result.candidates] == ["exact", "semantic"]
    assert result.coverage["reranker_unavailable"] is True
    assert result.trace["reranker_failure"] == "reranker_unavailable"
    assert "secret-token" not in repr(result.trace)
    assert "model.internal" not in repr(result.trace)


def test_engine_propagates_non_endpoint_reranker_errors(engine_fixture: EngineFixture) -> None:
    class BrokenReranker:
        def score(self, query: str, documents: list[str]) -> list[float]:
            raise TypeError("reranker contract violation")

    plan = ProductQueryPlan(**{**engine_fixture.plan.__dict__, "use_reranker": True})
    engine = ProductRetrievalEngine(
        StaticPlanner(plan),
        StaticRegistry(engine_fixture.engine.registry.outcomes),
        reranker=BrokenReranker(),
    )

    with pytest.raises(TypeError, match="reranker contract violation"):
        engine.search(engine_fixture.context, engine_fixture.request)
