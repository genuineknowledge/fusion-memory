import pytest

from fusion_memory.core.llm import StaticLLMClient
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderKind,
    SearchRequest,
)
from fusion_memory.retrieval.query_planner import ProductQueryPlanner


def _refined_response(*, confidence: float = 0.88) -> dict[str, object]:
    return {
        "intent": {
            "language": "zh",
            "answer_shape": "unordered_list",
            "evidence_scope": "multi_session",
            "speaker_scope": "user",
            "target_terms": ["权限控制", "登录保护"],
            "object_types": ["security_feature"],
            "temporal": {
                "requires_time": False,
                "requires_order": False,
                "requires_duration": False,
                "order_direction": "unknown",
                "endpoint_roles": [],
                "time_expressions": [],
            },
            "aggregation": {
                "operation": "count_distinct",
                "distinct": True,
                "target_terms": ["security_feature"],
                "unit_terms": [],
            },
            "needs_current_state": False,
            "needs_conflict_check": False,
            "confidence": confidence,
            "route_reasons": ["llm_multilingual_normalization"],
        }
    }


def _providers(plan):
    return {request.kind for request in plan.provider_requests}


def test_planner_uses_generic_sources_for_fact_query() -> None:
    plan = ProductQueryPlanner().plan(SearchRequest("What database does Atlas use?", 8))
    assert _providers(plan) == {ProviderKind.VECTOR, ProviderKind.LEXICAL, ProviderKind.ENTITY}
    assert plan.intent == "factual"
    assert plan.ordering is OrderingMode.RELEVANCE


def test_planner_routes_explicit_timeline_without_beam_category() -> None:
    plan = ProductQueryPlanner().plan(SearchRequest("按时间顺序列出 Atlas 的部署过程", 10, mode="balanced"))
    assert ProviderKind.CHRONOLOGY in _providers(plan)
    assert ProviderKind.TEMPORAL in _providers(plan)
    assert plan.intent == "chronology"
    assert plan.ordering is OrderingMode.CHRONOLOGICAL
    assert "event_ordering" not in repr(plan)


def test_planner_uses_recency_for_current_state() -> None:
    plan = ProductQueryPlanner().plan(SearchRequest("What is the latest Atlas database?", 6))
    assert ProviderKind.TEMPORAL in _providers(plan)
    assert plan.intent == "current_state"
    assert plan.ordering is OrderingMode.RECENCY


def test_planner_does_not_emit_semantic_current_state_whitelists() -> None:
    owner_plan = ProductQueryPlanner().plan(
        SearchRequest("Who is the current owner of the Acme service?", 6)
    )
    database_plan = ProductQueryPlanner().plan(
        SearchRequest("What database does the Acme service currently use?", 6)
    )
    retrieval_plan = ProductQueryPlanner().plan(
        SearchRequest("What retrieval backend does Project Atlas currently use?", 6)
    )
    latest_plan = ProductQueryPlanner().plan(
        SearchRequest("What is the latest Atlas backend?", 6)
    )

    for plan in (owner_plan, database_plan, retrieval_plan, latest_plan):
        assert "entity_context_terms" not in plan.query_intent
        assert "current_state_slot_groups" not in plan.query_intent


def test_safe_default_carries_generic_query_targets_and_entities() -> None:
    plan = ProductQueryPlanner().safe_default(
        SearchRequest("What is the current Project Atlas deadline?", 6)
    )

    assert set(plan.query_intent["target_terms"]) == {"project", "atlas", "deadline"}
    assert plan.entities == ("Project", "Atlas")
    assert plan.query_intent["entities"] == ["Project", "Atlas"]
    assert plan.query_intent["aggregation"] == {
        "operation": "none",
        "distinct": False,
        "target_terms": [],
        "unit_terms": [],
    }


def test_safe_default_keeps_ambiguous_query_support_empty() -> None:
    plan = ProductQueryPlanner().safe_default(SearchRequest("What is it?", 2))

    assert plan.entities == ()
    assert plan.query_intent["target_terms"] == []
    assert plan.query_intent["entities"] == []


def test_planner_default_and_off_modes_do_not_call_configured_refiner() -> None:
    default_client = StaticLLMClient(_refined_response())
    off_client = StaticLLMClient(_refined_response())
    default_planner = ProductQueryPlanner(intent_refiner=default_client)
    off_planner = ProductQueryPlanner(
        intent_refiner=off_client,
        intent_refiner_mode="off",
    )

    default_plan, default_telemetry = default_planner.plan_with_telemetry(
        SearchRequest("我之前提过哪些权限控制能力？", 6)
    )
    off_plan, off_telemetry = off_planner.plan_with_telemetry(
        SearchRequest("我之前提过哪些权限控制能力？", 6)
    )

    assert default_client.calls == []
    assert off_client.calls == []
    assert "llm_refined" not in default_plan.query_intent["route_reasons"]
    assert "llm_refined" not in off_plan.query_intent["route_reasons"]
    assert default_telemetry == {}
    assert off_telemetry == {}
    assert not hasattr(default_planner, "last_intent_telemetry")
    assert not hasattr(off_planner, "last_intent_telemetry")


def test_planner_always_mode_uses_strict_refinement() -> None:
    client = StaticLLMClient(_refined_response())
    planner = ProductQueryPlanner(intent_refiner=client, intent_refiner_mode="always")

    plan, telemetry = planner.plan_with_telemetry(
        SearchRequest("我之前提过哪些权限控制能力？", 6)
    )

    assert len(client.calls) == 1
    assert plan.query_intent["answer_shape"] == "unordered_list"
    assert plan.query_intent["aggregation"]["operation"] == "count_distinct"
    assert "llm_refined" in plan.query_intent["route_reasons"]
    assert set(plan.query_intent) == {
        "schema_version",
        "language",
        "answer_shape",
        "evidence_scope",
        "speaker_scope",
        "entities",
        "target_terms",
        "object_types",
        "temporal",
        "aggregation",
        "needs_current_state",
        "needs_conflict_check",
        "confidence",
        "route_reasons",
    }
    assert telemetry == {
        "source": "llm_query_intent",
        "prompt_version": "query-intent-refiner-v0",
        "fallback": False,
        "accepted": True,
        "deterministic_confidence": 0.73,
        "confidence": 0.88,
    }
    assert not hasattr(planner, "last_intent_telemetry")

    compatible_plan = planner.plan(SearchRequest("我之前提过哪些权限控制能力？", 6))
    assert isinstance(compatible_plan, ProductQueryPlan)


def test_planner_auto_mode_follows_deterministic_confidence_predicate() -> None:
    client = StaticLLMClient(_refined_response())
    planner = ProductQueryPlanner(intent_refiner=client, intent_refiner_mode="auto")

    high_confidence = planner.plan(
        SearchRequest(
            "What is my current updated Atlas security budget across all sessions?",
            6,
        )
    )
    low_confidence = planner.plan(
        SearchRequest("我之前提过哪些权限控制能力？", 6)
    )

    assert "llm_refined" not in high_confidence.query_intent["route_reasons"]
    assert "llm_refined" in low_confidence.query_intent["route_reasons"]
    assert len(client.calls) == 1


def test_planner_invalid_refinement_falls_back_to_deterministic_plan() -> None:
    query = "我之前提过哪些权限控制能力？"
    deterministic = ProductQueryPlanner(intent_refiner_mode="off").plan(
        SearchRequest(query, 6)
    )
    client = StaticLLMClient(_refined_response(confidence=0.2))
    planner = ProductQueryPlanner(intent_refiner=client, intent_refiner_mode="always")

    plan, telemetry = planner.plan_with_telemetry(SearchRequest(query, 6))

    assert plan == deterministic
    assert telemetry["fallback"] is True
    assert telemetry["reason"] == "invalid_or_low_confidence_output"


def test_planner_model_failure_falls_back_without_exposing_error_in_plan() -> None:
    class FailingClient:
        def structured(self, prompt, schema, input):
            raise RuntimeError("Bearer intent-secret https://intent.internal")

    query = "我之前提过哪些权限控制能力？"
    deterministic = ProductQueryPlanner(intent_refiner_mode="off").plan(
        SearchRequest(query, 6)
    )
    planner = ProductQueryPlanner(
        intent_refiner=FailingClient(),
        intent_refiner_mode="always",
    )

    plan, telemetry = planner.plan_with_telemetry(SearchRequest(query, 6))

    assert plan == deterministic
    assert "intent-secret" not in repr(plan)
    assert telemetry["fallback"] is True
    assert telemetry["reason"] == "llm_call_failed"


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True, "0.7"],
)
def test_planner_rejects_invalid_refiner_min_confidence_before_model_use(
    value: object,
) -> None:
    client = StaticLLMClient(_refined_response())

    with pytest.raises(
        ValueError,
        match="intent_refiner_min_confidence must be a finite number between 0.0 and 1.0",
    ):
        ProductQueryPlanner(
            intent_refiner=client,
            intent_refiner_min_confidence=value,
            intent_refiner_mode="always",
        )

    assert client.calls == []
