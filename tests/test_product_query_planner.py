from fusion_memory.retrieval.context import OrderingMode, ProviderKind, SearchRequest
from fusion_memory.retrieval.product_planner import ProductQueryPlanner


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
