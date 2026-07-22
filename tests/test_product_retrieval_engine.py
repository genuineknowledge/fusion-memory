from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

import fusion_memory.retrieval.product_engine as product_engine_module
import fusion_memory.retrieval.selection as selection_module
from fusion_memory.core.models import Candidate, EvidencePack, Scope
from fusion_memory.core.text import stable_hash
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderFailure,
    ProviderKind,
    ProviderRequest,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.engine import RetrievalUnavailable
from fusion_memory.retrieval.product_engine import ProductRetrievalEngine
from fusion_memory.retrieval.product_planner import ProductQueryPlanner
from fusion_memory.retrieval.providers.product_base import (
    ProviderContext,
    ProviderOutcome,
    ProviderUnavailable,
)
from fusion_memory.retrieval.providers.product_registry import ProductProviderRegistry


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


class StubPackBuilder:
    def build(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack:
        del context, result
        return EvidencePack(
            query=request.query,
            answer_policy="abstain_if_not_supported",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={"token_budget": token_budget},
            debug_trace=[],
        )


PACK_BUILDER = StubPackBuilder()


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
        engine=ProductRetrievalEngine(
            StaticPlanner(plan),
            StaticRegistry(outcomes),
            pack_builder=PACK_BUILDER,
        ),
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


def _hashed_dimension(value: str) -> str:
    return f"hashed_{stable_hash(value)[:16]}"


def test_engine_runs_one_pass_without_post_selection_rescue(engine_fixture: EngineFixture) -> None:
    result = engine_fixture.engine.search(engine_fixture.context, engine_fixture.request)

    assert [candidate.id for candidate in result.candidates] == ["exact", "semantic"]
    assert result.coverage["intent"] == "factual"
    assert result.trace["intent"] == "factual"
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
        pack_builder=PACK_BUILDER,
    )

    result = engine.search(engine_fixture.context, engine_fixture.request)

    assert [candidate.id for candidate in result.candidates] == ["lexical"]
    assert result.coverage["degraded"] is True
    assert result.coverage["provider_failures"] == ["model_unavailable"]
    assert result.trace["providers"][0]["failure_code"] == "model_unavailable"


def test_engine_raises_when_all_planned_providers_fail(engine_fixture: EngineFixture) -> None:
    engine = ProductRetrievalEngine(
        StaticPlanner(engine_fixture.plan),
        StaticRegistry(
            (
                _failed_outcome(ProviderKind.VECTOR),
                _failed_outcome(ProviderKind.LEXICAL),
            )
        ),
        pack_builder=PACK_BUILDER,
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
        pack_builder=PACK_BUILDER,
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


def test_engine_sanitizes_caller_supplied_intent_in_coverage_and_trace(
    engine_fixture: EngineFixture,
) -> None:
    malicious_intent = (
        "Bearer intent-secret https://intent.internal/v1 "
        "username=admin password=intent-password"
    )
    plan = ProductQueryPlan(
        **{**engine_fixture.plan.__dict__, "intent": malicious_intent}
    )

    result = engine_fixture.engine.search_with_plan(
        engine_fixture.context,
        engine_fixture.request,
        plan,
    )

    rendered = f"{result.coverage!r} {result.trace!r}"
    assert malicious_intent not in rendered
    assert "intent-secret" not in rendered
    assert "intent.internal" not in rendered
    assert "intent-password" not in rendered
    assert result.coverage["intent"] == _hashed_dimension(malicious_intent)
    assert result.trace["intent"] == _hashed_dimension(malicious_intent)


def test_engine_sanitizes_provider_unavailable_code_in_coverage_and_trace(
    engine_fixture: EngineFixture,
) -> None:
    malicious_code = (
        "Bearer provider-secret https://provider.internal/v1 "
        "api_key=provider-api-key"
    )

    class FailingVectorProvider:
        kind = ProviderKind.VECTOR
        repository = object()

        def recall(self, context: ProviderContext) -> ProviderOutcome:
            raise ProviderUnavailable(malicious_code)

    class StaticLexicalProvider:
        kind = ProviderKind.LEXICAL
        repository = object()

        def recall(self, context: ProviderContext) -> ProviderOutcome:
            return ProviderOutcome(
                provider=context.provider,
                candidates=(
                    _candidate(
                        "lexical",
                        "private-memory-body lexical",
                        source="product_lexical",
                        scores={"bm25_score": 0.9},
                    ),
                ),
                elapsed_ms=0.5,
            )

    engine = ProductRetrievalEngine(
        StaticPlanner(engine_fixture.plan),
        ProductProviderRegistry([FailingVectorProvider(), StaticLexicalProvider()]),
        pack_builder=PACK_BUILDER,
    )

    result = engine.search(engine_fixture.context, engine_fixture.request)

    rendered = f"{result.coverage!r} {result.trace!r}"
    assert malicious_code not in rendered
    assert "provider-secret" not in rendered
    assert "provider.internal" not in rendered
    assert "provider-api-key" not in rendered
    assert result.coverage["provider_failures"] == [_hashed_dimension(malicious_code)]
    assert result.trace["providers"][0]["failure_code"] == _hashed_dimension(
        malicious_code
    )


def test_engine_sanitizes_reranker_failure_status_in_trace(
    engine_fixture: EngineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malicious_failure = (
        "Bearer reranker-secret https://reranker.internal/v1 "
        "token=reranker-token"
    )

    def select_with_malicious_status(
        query: str,
        candidate_lists: list[list[Candidate]],
        *,
        limit: int,
        use_reranker: bool,
        reranker: object,
        mmr_lambda: float,
        status: dict[str, object],
    ) -> list[Candidate]:
        status.update(
            fused_count=1,
            fusion_elapsed_ms=0.0,
            selection_elapsed_ms=0.0,
            reranker_failure=malicious_failure,
        )
        return candidate_lists[0][:limit]

    monkeypatch.setattr(
        product_engine_module,
        "select_candidates",
        select_with_malicious_status,
    )

    result = engine_fixture.engine.search(engine_fixture.context, engine_fixture.request)

    rendered = f"{result.coverage!r} {result.trace!r}"
    assert malicious_failure not in rendered
    assert "reranker-secret" not in rendered
    assert "reranker.internal" not in rendered
    assert "reranker-token" not in rendered
    assert result.trace["reranker_failure"] == _hashed_dimension(malicious_failure)


def test_balanced_engine_runs_rrf_reranker_and_mmr_exactly_once(
    engine_fixture: EngineFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"rrf": 0, "reranker": 0, "mmr": 0, "score": 0}
    original_rrf = selection_module.reciprocal_rank_fusion
    original_reranker = selection_module.rerank_candidates
    original_mmr = selection_module.mmr

    def rrf_spy(candidate_lists: list[list[Candidate]]) -> list[Candidate]:
        calls["rrf"] += 1
        return original_rrf(candidate_lists)

    def reranker_spy(
        query: str,
        candidates: list[Candidate],
        reranker: object,
    ) -> list[Candidate]:
        calls["reranker"] += 1
        return original_reranker(query, candidates, reranker)

    def mmr_spy(
        candidates: list[Candidate],
        limit: int,
        lambda_: float = 0.72,
    ) -> list[Candidate]:
        calls["mmr"] += 1
        return original_mmr(candidates, limit=limit, lambda_=lambda_)

    class TrackingReranker:
        def score(self, query: str, documents: list[str]) -> list[float]:
            calls["score"] += 1
            return [float(len(documents) - index) for index in range(len(documents))]

    monkeypatch.setattr(selection_module, "reciprocal_rank_fusion", rrf_spy)
    monkeypatch.setattr(selection_module, "rerank_candidates", reranker_spy)
    monkeypatch.setattr(selection_module, "mmr", mmr_spy)
    plan = ProductQueryPlan(
        **{**engine_fixture.plan.__dict__, "use_reranker": True}
    )
    request = SearchRequest(
        query=engine_fixture.request.query,
        limit=engine_fixture.request.limit,
        mode="balanced",
    )
    engine = ProductRetrievalEngine(
        StaticPlanner(plan),
        engine_fixture.engine.registry,
        reranker=TrackingReranker(),
        pack_builder=PACK_BUILDER,
    )

    engine.search(engine_fixture.context, request)

    assert calls == {"rrf": 1, "reranker": 1, "mmr": 1, "score": 1}


def test_selection_orders_exact_rrf_and_utility_ties_by_candidate_id() -> None:
    candidate_z = _candidate(
        "z-candidate",
        "zulu memory",
        source="product_vector",
        scores={"semantic_score": 0.1},
    )
    candidate_a = _candidate(
        "a-candidate",
        "alpha memory",
        source="product_lexical",
        scores={"semantic_score": 0.85},
    )

    selected = selection_module.select_candidates(
        "memory",
        [[candidate_z, candidate_a], [candidate_a, candidate_z]],
        limit=2,
        use_reranker=False,
        reranker=None,
        mmr_lambda=0.72,
    )

    assert selected[0].scores["rrf_score"] == selected[1].scores["rrf_score"]
    assert selected[0].scores["utility_score"] == selected[1].scores["utility_score"]
    assert [candidate.id for candidate in selected] == ["a-candidate", "z-candidate"]


def test_engine_propagates_planner_exceptions(engine_fixture: EngineFixture) -> None:
    engine = ProductRetrievalEngine(
        RaisingPlanner(),
        StaticRegistry(()),
        pack_builder=PACK_BUILDER,
    )

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
        pack_builder=PACK_BUILDER,
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
        pack_builder=PACK_BUILDER,
    )

    with pytest.raises(TypeError, match="reranker contract violation"):
        engine.search(engine_fixture.context, engine_fixture.request)
