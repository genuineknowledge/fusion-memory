from __future__ import annotations

import unittest
from datetime import datetime, timezone, timedelta

from fusion_memory import MemoryService
from fusion_memory.core.models import Scope
from fusion_memory.retrieval.chronology_selector import select_persisted_graph_event_ordering_candidates


class ChronologySelectorTests(unittest.TestCase):
    def test_persisted_graph_selector_returns_topic_scoped_ordered_candidates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-select", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        memory.add("I first set up the budget tracker schema.", scope, base, {"source_uri": "m1"})
        memory.add("Then I implemented transaction CRUD validation.", scope, base + timedelta(minutes=5), {"source_uri": "m2"})
        memory.add("Unrelated: I changed my lunch plan.", scope, base + timedelta(minutes=10), {"source_uri": "m3"})

        candidates, telemetry = select_persisted_graph_event_ordering_candidates(
            "List the budget tracker work in order.",
            scope,
            memory.store,
            limit=5,
            include_session=True,
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0].source, "event_ordering_persisted_graph")
        self.assertIn("schema", candidates[0].text.lower())
        self.assertIn("crud", candidates[1].text.lower())
        self.assertTrue(all("lunch" not in candidate.text.lower() for candidate in candidates))
        self.assertEqual(telemetry["selected_driver"], "persisted_graph")
