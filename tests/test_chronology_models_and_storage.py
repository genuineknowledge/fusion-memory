from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory.core.chronology import (
    ChronologyEventEdge,
    ChronologyEventNode,
    ChronologyPhase,
    ChronologyTopic,
)
from fusion_memory.core.models import Scope
from fusion_memory.storage.postgres_store import PostgresMemoryStore
from fusion_memory.storage.sqlite_store import SQLiteMemoryStore


class ChronologyStorageTests(unittest.TestCase):
    def test_sqlite_chronology_graph_round_trips_topic_phase_node_and_edge(self) -> None:
        store = SQLiteMemoryStore()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        now = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        topic = ChronologyTopic(
            topic_id="topic_budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget app"],
            language="en",
            taxonomy_tags=["software"],
            source_span_ids=["s1"],
            confidence=0.9,
            created_at=now,
        )
        phase = ChronologyPhase(
            phase_id="phase_setup",
            topic_id=topic.topic_id,
            phase_type="setup",
            order_hint=1,
            source_span_ids=["s1"],
            confidence=0.8,
            created_at=now,
        )
        first = ChronologyEventNode(
            node_id="node_1",
            scope=scope,
            actor="user",
            action="set up",
            object="schema",
            topic_id=topic.topic_id,
            phase_id=phase.phase_id,
            timestamp=now,
            source_span_id="s1",
            source_turn_id="t1",
            text="I first set up the schema.",
            language="en",
            confidence=0.88,
            explicit_order_marker="first",
            created_at=now,
        )
        second = ChronologyEventNode(
            node_id="node_2",
            scope=scope,
            actor="user",
            action="implemented",
            object="transaction CRUD",
            topic_id=topic.topic_id,
            phase_id=phase.phase_id,
            timestamp=now,
            source_span_id="s2",
            source_turn_id="t2",
            text="Then I implemented transaction CRUD.",
            language="en",
            confidence=0.86,
            explicit_order_marker="then",
            created_at=now,
        )
        edge = ChronologyEventEdge(
            edge_id="edge_1",
            from_node_id=first.node_id,
            to_node_id=second.node_id,
            edge_type="before",
            evidence_type="explicit_marker",
            source_span_ids=["s1", "s2"],
            confidence=0.92,
            created_at=now,
        )

        store.upsert_chronology_topic(topic)
        store.upsert_chronology_phase(phase)
        store.upsert_chronology_event_node(first)
        store.upsert_chronology_event_node(second)
        inserted = store.insert_chronology_event_edge(edge)

        self.assertTrue(inserted)
        self.assertEqual(store.list_chronology_topics(scope, include_session=True)[0].canonical_label, "budget tracker")
        self.assertEqual(store.list_chronology_phases([topic.topic_id])[0].phase_type, "setup")
        nodes = store.list_chronology_event_nodes(scope, include_session=True, topic_ids=[topic.topic_id])
        self.assertEqual([node.node_id for node in nodes], ["node_1", "node_2"])
        edges = store.list_chronology_event_edges(["node_1", "node_2"])
        self.assertEqual(edges[0].edge_type, "before")
        self.assertEqual(edges[0].evidence_type, "explicit_marker")

    def test_sqlite_clear_scope_removes_chronology_graph_for_scope_only(self) -> None:
        store = SQLiteMemoryStore()
        cleared_scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="cleared")
        retained_scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="retained")
        cleared_ids = self._insert_chronology_graph(store, cleared_scope, "cleared")
        retained_ids = self._insert_chronology_graph(store, retained_scope, "retained")

        result = store.clear_scope(cleared_scope, include_session=True)

        self.assertEqual(store.list_chronology_topics(cleared_scope, include_session=True), [])
        self.assertEqual(store.list_chronology_phases([cleared_ids["topic_id"]]), [])
        self.assertEqual(store.list_chronology_event_nodes(cleared_scope, include_session=True), [])
        self.assertEqual(store.list_chronology_event_edges(cleared_ids["node_ids"]), [])
        self.assertEqual([topic.topic_id for topic in store.list_chronology_topics(retained_scope, include_session=True)], [retained_ids["topic_id"]])
        self.assertEqual([phase.phase_id for phase in store.list_chronology_phases([retained_ids["topic_id"]])], [retained_ids["phase_id"]])
        self.assertEqual(
            [node.node_id for node in store.list_chronology_event_nodes(retained_scope, include_session=True)],
            retained_ids["node_ids"],
        )
        self.assertEqual([edge.edge_id for edge in store.list_chronology_event_edges(retained_ids["node_ids"])], [retained_ids["edge_id"]])
        self.assertEqual(result["deleted"]["chronology_topics"], 1)
        self.assertEqual(result["deleted"]["chronology_phases"], 1)
        self.assertEqual(result["deleted"]["chronology_event_nodes"], 2)
        self.assertEqual(result["deleted"]["chronology_event_edges"], 1)

    def test_postgres_clear_scope_deletes_chronology_graph_tables_by_scoped_ids(self) -> None:
        conn = RecordingConnection()
        store = PostgresMemoryStore("postgresql://example/fusion", connect=lambda _dsn: conn)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")

        result = store.clear_scope(scope, include_session=True)

        statements = [sql.lower() for sql, _params in conn.cursor_instance.executions]
        self.assertTrue(any("select topic_id from chronology_topics" in sql for sql in statements))
        self.assertTrue(any("select node_id from chronology_event_nodes" in sql for sql in statements))
        self.assertTrue(any("delete from chronology_event_edges" in sql and "from_node_id" in sql for sql in statements))
        self.assertTrue(any("delete from chronology_phases" in sql and "topic_id in" in sql for sql in statements))
        self.assertTrue(any("delete from chronology_event_nodes" in sql for sql in statements))
        self.assertTrue(any("delete from chronology_topics" in sql for sql in statements))
        self.assertEqual(result["deleted"]["chronology_event_edges"], 1)
        self.assertEqual(result["deleted"]["chronology_phases"], 1)
        self.assertEqual(result["deleted"]["chronology_event_nodes"], 2)
        self.assertEqual(result["deleted"]["chronology_topics"], 1)

    def _insert_chronology_graph(
        self,
        store: SQLiteMemoryStore,
        scope: Scope,
        prefix: str,
    ) -> dict[str, object]:
        now = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        topic = ChronologyTopic(
            topic_id=f"topic_{prefix}",
            scope=scope,
            canonical_label=f"{prefix} tracker",
            aliases=[],
            language="en",
            taxonomy_tags=[],
            source_span_ids=[],
            confidence=0.9,
            created_at=now,
        )
        phase = ChronologyPhase(
            phase_id=f"phase_{prefix}",
            topic_id=topic.topic_id,
            phase_type="setup",
            order_hint=1,
            source_span_ids=[],
            confidence=0.8,
            created_at=now,
        )
        first = ChronologyEventNode(
            node_id=f"node_{prefix}_1",
            scope=scope,
            actor="user",
            action="started",
            object="work",
            topic_id=topic.topic_id,
            phase_id=phase.phase_id,
            timestamp=now,
            source_span_id=None,
            source_turn_id=None,
            text="Started work.",
            language="en",
            confidence=0.8,
            explicit_order_marker="first",
            created_at=now,
        )
        second = ChronologyEventNode(
            node_id=f"node_{prefix}_2",
            scope=scope,
            actor="user",
            action="finished",
            object="work",
            topic_id=topic.topic_id,
            phase_id=phase.phase_id,
            timestamp=now,
            source_span_id=None,
            source_turn_id=None,
            text="Finished work.",
            language="en",
            confidence=0.8,
            explicit_order_marker="then",
            created_at=now,
        )
        edge = ChronologyEventEdge(
            edge_id=f"edge_{prefix}",
            from_node_id=first.node_id,
            to_node_id=second.node_id,
            edge_type="before",
            evidence_type="explicit_marker",
            source_span_ids=[],
            confidence=0.8,
            created_at=now,
        )
        store.upsert_chronology_topic(topic)
        store.upsert_chronology_phase(phase)
        store.upsert_chronology_event_node(first)
        store.upsert_chronology_event_node(second)
        store.insert_chronology_event_edge(edge)
        return {
            "topic_id": topic.topic_id,
            "phase_id": phase.phase_id,
            "node_ids": [first.node_id, second.node_id],
            "edge_id": edge.edge_id,
        }


class RecordingConnection:
    def __init__(self) -> None:
        self.cursor_instance = RecordingCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self) -> "RecordingCursor":
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class RecordingCursor:
    def __init__(self) -> None:
        self.executions: list[tuple[str, object]] = []
        self._last_sql = ""
        self.rowcount = 0

    def execute(self, sql: str, params: object = None) -> None:
        self._last_sql = sql.lower()
        self.executions.append((sql, params))
        if "delete from chronology_event_edges" in self._last_sql:
            self.rowcount = 1
        elif "delete from chronology_phases" in self._last_sql:
            self.rowcount = 1
        elif "delete from chronology_event_nodes" in self._last_sql:
            self.rowcount = 2
        elif "delete from chronology_topics" in self._last_sql:
            self.rowcount = 1
        else:
            self.rowcount = 0

    def fetchall(self) -> list[tuple[str]]:
        if "select topic_id from chronology_topics" in self._last_sql:
            return [("topic_cleared",)]
        if "select node_id from chronology_event_nodes" in self._last_sql:
            return [("node_cleared_1",), ("node_cleared_2",)]
        return []

    def close(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
