from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.core.models import MemoryEvent
from fusion_memory.retrieval.event_graph_selection import (
    build_event_chronology_graph,
    select_graph_first_event_ordering_candidates,
)


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class EventOrderingGraphTests(unittest.TestCase):
    def test_build_event_chronology_graph_emits_nodes_and_edges(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I first prepared the initial workspace foundation.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Then I implemented the second workflow step.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("Afterward I verified the final handoff.", scope, ts("2026-06-03T10:00:00+00:00"))

        spans = memory.store.list_spans(scope)
        events = memory.store.list_events(scope)
        graph = build_event_chronology_graph(
            "Can you walk me through the order of the workspace work across our conversations?",
            spans,
            events,
        )

        self.assertTrue(graph.nodes)
        self.assertTrue(graph.edges)
        self.assertTrue(any(edge.kind in {"then", "before", "after", "updates", "replaces"} for edge in graph.edges))

    def test_graph_first_event_selection_prefers_causal_chain_over_label_noise(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I first prepared the initial workspace foundation.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Then I implemented the second workflow step.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("Afterward I verified the final handoff.", scope, ts("2026-06-03T10:00:00+00:00"))

        spans = memory.store.list_spans(scope)
        events = memory.store.list_events(scope)
        candidates = select_graph_first_event_ordering_candidates(
            "Can you walk me through the order of the workspace work across our conversations?",
            spans,
            events,
            limit=4,
        )

        self.assertTrue(candidates)
        self.assertTrue(candidates[0].source.startswith("event_ordering_graph"))
        self.assertNotIn("event_ordering_graph_fallback", candidates[0].source)

    def test_graph_candidates_include_temporal_relation_shadow_metadata(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w-rel", user_id="u", agent_id="a", session_id="s")
        memory.add("I first prepared the initial workspace foundation.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Then I implemented the second workflow step.", scope, ts("2026-06-02T10:00:00+00:00"))

        spans = memory.store.list_spans(scope)
        events = memory.store.list_events(scope)
        candidates = select_graph_first_event_ordering_candidates(
            "Can you walk me through the order of the workspace work across our conversations?",
            spans,
            events,
            limit=4,
        )

        self.assertTrue(candidates)
        relation_candidates = [candidate for candidate in candidates if candidate.metadata.get("temporal_relations")]
        self.assertTrue(relation_candidates)
        self.assertIn("temporal_relation_summary", relation_candidates[0].metadata)
        self.assertNotIn("text", relation_candidates[0].metadata["temporal_relations"][0])

    def test_sparse_graph_uses_legacy_fallback_without_high_confidence_edges(self) -> None:
        scope = Scope(workspace_id="w-sparse", user_id="u", agent_id="a", session_id="s")
        spans: list[object] = []
        events = [
            MemoryEvent(
                event_id="event-a",
                scope=scope,
                event_type="generic",
                description="We discussed palette ideas for the dashboard.",
                participants=[],
                source_span_ids=[],
                time_start=ts("2026-06-01T10:00:00+00:00"),
            ),
            MemoryEvent(
                event_id="event-b",
                scope=scope,
                event_type="generic",
                description="We also talked about copy tone for onboarding.",
                participants=[],
                source_span_ids=[],
                time_start=ts("2026-06-02T10:00:00+00:00"),
            ),
        ]
        graph = build_event_chronology_graph(
            "Can you walk me through the order of dashboard decisions?",
            spans,
            events,
        )
        candidates = select_graph_first_event_ordering_candidates(
            "Can you walk me through the order of dashboard decisions?",
            spans,
            events,
            limit=4,
        )

        self.assertEqual(graph.edges, [])
        self.assertTrue(candidates)
        self.assertTrue(candidates[0].source.startswith("event_ordering_graph_fallback_"))

    def test_event_ordering_search_exposes_shadow_graph_coverage(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I first prepared the initial workspace foundation.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Then I implemented the second workflow step.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("Afterward I verified the final handoff.", scope, ts("2026-06-03T10:00:00+00:00"))

        pack = memory.answer_context(
            "List the workspace work in chronological order, first to last.",
            scope,
            budget={"limit": 6, "mode": "benchmark"},
        )

        self.assertTrue("event_ordering_graph" in pack.coverage or "event_ordering_shadow" in pack.coverage)
