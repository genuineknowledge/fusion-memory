from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.chronology_backfill import backfill_chronology_graph


class ChronologyBackfillTests(unittest.TestCase):
    def test_backfill_builds_chronology_graph_from_existing_events_and_spans(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="backfill-ws", user_id="u", agent_id="a", run_id="r", session_id="s")
        memory.add(
            {
                "role": "user",
                "content": "First I set up the budget tracker schema. Then I implemented category filters.",
                "turn_id": "turn-1",
                "timestamp": "2026-06-19T00:00:00+00:00",
            },
            scope,
            datetime(2026, 6, 19, tzinfo=timezone.utc),
            {"source_uri": "test:backfill"},
        )
        for table in (
            "chronology_event_edges",
            "chronology_phases",
            "chronology_event_nodes",
            "chronology_topics",
        ):
            memory.store.conn.execute(f"delete from {table}")
        memory.store.conn.commit()
        self.assertEqual(memory.store.list_chronology_topics(scope, include_session=True), [])
        self.assertTrue(memory.store.list_events(scope, include_session=True))
        self.assertTrue(memory.store.list_spans(scope, include_session=True))

        report = backfill_chronology_graph(memory.store, scope, include_session=True)

        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(report["events"], 1)
        self.assertGreaterEqual(report["topics"], 1)
        self.assertGreaterEqual(report["nodes"], 1)
        self.assertGreaterEqual(len(memory.store.list_chronology_topics(scope, include_session=True)), 1)

    def test_backfill_preserves_event_session_scopes(self) -> None:
        memory = MemoryService()
        base_scope = Scope(workspace_id="backfill-session-ws", user_id="u", agent_id="a", run_id="r")
        session_scope = Scope(workspace_id="backfill-session-ws", user_id="u", agent_id="a", run_id="r", session_id="s1")
        memory.add(
            {
                "role": "user",
                "content": "First I set up the budget tracker schema. Then I implemented category filters.",
                "turn_id": "turn-1",
                "timestamp": "2026-06-19T00:00:00+00:00",
            },
            session_scope,
            datetime(2026, 6, 19, tzinfo=timezone.utc),
            {"source_uri": "test:backfill-session"},
        )
        for table in (
            "chronology_event_edges",
            "chronology_phases",
            "chronology_event_nodes",
            "chronology_topics",
        ):
            memory.store.conn.execute(f"delete from {table}")
        memory.store.conn.commit()

        report = backfill_chronology_graph(memory.store, base_scope, include_session=True)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["session_count"], 1)
        topics = memory.store.list_chronology_topics(session_scope, include_session=True)
        self.assertGreaterEqual(len(topics), 1)
        self.assertEqual({topic.scope.session_id for topic in topics}, {"s1"})


if __name__ == "__main__":
    unittest.main()
