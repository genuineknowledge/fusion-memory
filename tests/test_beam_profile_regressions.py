from __future__ import annotations

import pytest

from fusion_memory import MemoryService, Scope
from fusion_memory.eval.beam.engine import BeamRetrievalEngine


@pytest.mark.parametrize(
    ("category", "memory", "query"),
    [
        ("event_ordering", "Atlas started, then deployment completed.", "List Atlas events in order."),
        ("temporal_reasoning", "Atlas deployment is July 30.", "When is Atlas deployment?"),
        (
            "contradiction_resolution",
            "Atlas first used SQLite, then switched to Qdrant.",
            "Did Atlas change databases?",
        ),
        (
            "multi_session_reasoning",
            "Atlas databases use Qdrant and reports use PostgreSQL.",
            "List the Atlas databases mentioned.",
        ),
    ],
)
def test_beam_profile_runs_without_category_in_product_plan(
    category: str,
    memory: str,
    query: str,
) -> None:
    service = MemoryService()
    scope = Scope(user_id="beam-user", workspace_id="beam", session_id="beam-session")
    try:
        service.add(memory, scope)
        engine = BeamRetrievalEngine.from_service(service)
        pack = engine.answer_context(query, scope, category, {"limit": 12})
        plan = engine.planner.plan(query, category, 12)
        assert pack.coverage["benchmark"] == "BEAM"
        assert pack.coverage["benchmark_category"] == category
        assert pack.source_spans
        assert not hasattr(plan, "category")
    finally:
        service.close()
