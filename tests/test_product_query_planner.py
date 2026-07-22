from fusion_memory.retrieval.context import OrderingMode, ProviderKind, SearchRequest
from fusion_memory.retrieval.query_planner import ProductQueryPlanner


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
