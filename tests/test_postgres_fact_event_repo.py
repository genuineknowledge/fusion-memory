from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any

from fusion_memory.core.models import EventEdge, FactRelation, MemoryEvent, MemoryFact, Scope
from fusion_memory.storage.postgres_store import PostgresEventRepository, PostgresFactRepository


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class PostgresFactRepositoryTests(unittest.TestCase):
    def test_fact_crud_search_relations_and_superseded_ids(self) -> None:
        fake = FakePostgresConnection()
        repo = PostgresFactRepository("postgresql://example/fusion", connect=lambda dsn: fake)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        old_fact = make_fact("fact_old", scope, "Atlas used Chroma for retrieval.", "Chroma")
        new_fact = make_fact("fact_new", scope, "Atlas now uses Qdrant for retrieval.", "Qdrant")

        self.assertTrue(repo.insert_fact(old_fact))
        self.assertTrue(repo.insert_fact(new_fact))
        self.assertFalse(repo.insert_fact(new_fact))

        insert_sql, insert_params = fake.executed[0]
        self.assertIn("cast(%s as vector)", insert_sql)
        self.assertEqual(insert_params[0], "fact_old")
        self.assertEqual(insert_params[18], '["span_1"]')
        self.assertTrue(str(insert_params[20]).startswith("["))

        loaded = repo.get_fact("fact_new", scope, include_session=True)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.object, "Qdrant")
        self.assertEqual(loaded.scope.session_id, "s")

        listed = repo.list_facts(scope, category="project_state", include_session=True)
        self.assertEqual([fact.fact_id for fact in listed], ["fact_old", "fact_new"])

        relation = FactRelation(
            relation_id="rel_1",
            from_fact_id="fact_new",
            to_fact_id="fact_old",
            relation_type="supersedes",
            source_span_ids=["span_2"],
            confidence=0.9,
        )
        self.assertTrue(repo.insert_fact_relation(relation))
        self.assertFalse(repo.insert_fact_relation(relation))
        self.assertEqual(repo.superseded_fact_ids(), {"fact_old"})
        relations = repo.list_fact_relations(fact_id="fact_old", relation_type="supersedes")
        self.assertEqual([item.relation_id for item in relations], ["rel_1"])

        results = repo.search_facts("Qdrant Atlas", scope, limit=5, include_session=True)
        self.assertEqual(results[0][0].fact_id, "fact_new")
        self.assertGreater(results[0][1]["score"], 0)

        search_sql, search_params = fake.executed[-1]
        normalized = " ".join(search_sql.lower().split())
        self.assertIn("from memory_facts f", normalized)
        self.assertIn("embedding_dense <=> cast(%s as vector)", normalized)
        self.assertIn("exists ( select 1 from fact_relations", normalized)
        self.assertIn("f.session_id = %s", normalized)
        self.assertEqual(search_params[0], "Qdrant Atlas")
        self.assertEqual(search_params[-1], 5)

    def test_fact_insert_rolls_back_on_error(self) -> None:
        fake = FakePostgresConnection(fail_on_insert_table="memory_facts")
        repo = PostgresFactRepository("postgresql://example/fusion", connect=lambda dsn: fake)

        with self.assertRaises(RuntimeError):
            repo.insert_fact(make_fact("fact_1", Scope(workspace_id="w"), "Atlas uses Qdrant.", "Qdrant"))

        self.assertEqual(fake.rolled_back, 1)
        self.assertEqual(fake.committed, 0)


class PostgresEventRepositoryTests(unittest.TestCase):
    def test_event_crud_search_edges_and_temporal_edge_lookup(self) -> None:
        fake = FakePostgresConnection()
        repo = PostgresEventRepository("postgresql://example/fusion", connect=lambda dsn: fake)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        bm25 = make_event("event_bm25", scope, "user_action", "I tested BM25 retrieval.", "2026-06-02T10:00:00+00:00")
        dense = make_event("event_dense", scope, "user_action", "I added dense retrieval.", "2026-06-05T10:00:00+00:00")

        self.assertTrue(repo.insert_event(bm25))
        self.assertTrue(repo.insert_event(dense))
        self.assertFalse(repo.insert_event(dense))

        loaded = repo.get_event("event_dense", scope, include_session=True)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.description, "I added dense retrieval.")
        self.assertEqual(loaded.participants, ["u"])

        timeline = repo.list_events(scope, include_session=True)
        self.assertEqual([event.event_id for event in timeline], ["event_bm25", "event_dense"])

        results = repo.search_events("dense retrieval", scope, limit=5, include_session=True)
        self.assertEqual(results[0][0].event_id, "event_dense")
        self.assertGreater(results[0][1]["temporal_fit"], 0)

        edge = EventEdge(
            edge_id="edge_1",
            from_event_id="event_bm25",
            to_event_id="event_dense",
            edge_type="before",
            source_span_ids=["span_bm25", "span_dense"],
            confidence=0.8,
        )
        self.assertTrue(repo.insert_event_edge(edge))
        self.assertFalse(repo.insert_event_edge(edge))
        self.assertTrue(repo.has_event_edge("event_bm25", "event_dense", edge_type="before"))
        loaded_edge = repo.get_event_edge("event_bm25", "event_dense")
        self.assertIsNotNone(loaded_edge)
        self.assertEqual(loaded_edge["edge_type"], "before")
        self.assertEqual(loaded_edge["source_span_ids"], ["span_bm25", "span_dense"])

        search_sql, search_params = next(item for item in fake.executed if "with scored" in item[0].lower() and "from events e" in item[0].lower())
        normalized = " ".join(search_sql.lower().split())
        self.assertIn("plainto_tsquery('simple', %s)", normalized)
        self.assertIn("e.session_id = %s", normalized)
        self.assertEqual(search_params[0], "dense retrieval")
        self.assertEqual(search_params[-1], 5)


def make_fact(fact_id: str, scope: Scope, text: str, obj: str) -> MemoryFact:
    return MemoryFact(
        fact_id=fact_id,
        scope=scope,
        subject="Atlas",
        predicate="uses",
        object=obj,
        text=text,
        category="project_state",
        polarity="positive",
        confidence=0.86,
        salience=0.7,
        observed_at=ts("2026-06-01T10:00:00+00:00"),
        valid_from=ts("2026-06-01T10:00:00+00:00"),
        source_span_ids=["span_1"],
        linked_fact_ids=[],
        metadata={"hash": fact_id + "_hash"},
        created_at=ts("2026-06-01T10:00:00+00:00"),
    )


def make_event(event_id: str, scope: Scope, event_type: str, description: str, time_start: str) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        scope=scope,
        event_type=event_type,
        description=description,
        participants=["u"],
        source_span_ids=[event_id.replace("event", "span")],
        fact_ids=[],
        time_start=ts(time_start),
        time_granularity="day",
        time_source="explicit",
        confidence=0.82,
    )


class FakePostgresCursor:
    def __init__(self, conn: "FakePostgresConnection") -> None:
        self.conn = conn
        self.description: list[tuple[str]] = []
        self.rowcount = -1
        self._results: list[dict[str, Any]] = []
        self.closed = False

    def execute(self, statement: str, params: Any = None) -> None:
        params = list(params or [])
        self.conn.executed.append((statement, params))
        normalized = " ".join(statement.lower().split())
        if "insert into memory_facts" in normalized:
            self._insert("memory_facts", params[0], _fact_row(params))
        elif "insert into fact_relations" in normalized:
            self._insert("fact_relations", params[0], _fact_relation_row(params))
        elif "insert into events" in normalized:
            self._insert("events", params[0], _event_row(params))
        elif "insert into event_edges" in normalized:
            self._insert("event_edges", params[0], _event_edge_row(params))
        elif normalized.startswith("with scored") and "from memory_facts f" in normalized:
            self._search_facts(params)
        elif normalized.startswith("with scored") and "from events e" in normalized:
            self._search_events(params)
        elif "select to_fact_id from fact_relations" in normalized:
            self._set_results([{"to_fact_id": row["to_fact_id"]} for row in self.conn.tables["fact_relations"].values() if row["relation_type"] == "supersedes"])
        elif "from fact_relations" in normalized:
            self._select_fact_relations(params)
        elif "from event_edges" in normalized and "select 1 as present" in normalized:
            self._has_event_edge(params)
        elif "from event_edges" in normalized:
            self._get_event_edge(params)
        elif "from memory_facts" in normalized and "fact_id = %s" in normalized:
            self._get_by_id("memory_facts", params[0])
        elif "from memory_facts" in normalized:
            self._list_table("memory_facts")
        elif "from events" in normalized and "event_id = %s" in normalized:
            self._get_by_id("events", params[0])
        elif "from events" in normalized:
            self._list_events()
        else:
            self._set_results([])

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._results)

    def close(self) -> None:
        self.closed = True

    def _insert(self, table: str, object_id: str, row: dict[str, Any]) -> None:
        if self.conn.fail_on_insert_table == table:
            raise RuntimeError(f"insert failed: {table}")
        if object_id in self.conn.tables[table]:
            self.rowcount = 0
            self._set_results([])
            return
        self.conn.tables[table][object_id] = row
        self.rowcount = 1
        self._set_results([])

    def _get_by_id(self, table: str, object_id: str) -> None:
        row = self.conn.tables[table].get(object_id)
        self._set_results([row] if row else [])

    def _list_table(self, table: str) -> None:
        self._set_results(sorted(self.conn.tables[table].values(), key=lambda row: row.get("created_at") or ""))

    def _list_events(self) -> None:
        self._set_results(sorted(self.conn.tables["events"].values(), key=lambda row: (row.get("time_start") or "", row.get("created_at") or "")))

    def _select_fact_relations(self, params: list[Any]) -> None:
        rows = list(self.conn.tables["fact_relations"].values())
        if params:
            fact_id = params[0]
            rows = [row for row in rows if row["from_fact_id"] == fact_id or row["to_fact_id"] == fact_id]
            if len(params) > 2:
                rows = [row for row in rows if row["relation_type"] == params[2]]
        self._set_results(rows)

    def _has_event_edge(self, params: list[Any]) -> None:
        rows = [
            row
            for row in self.conn.tables["event_edges"].values()
            if row["from_event_id"] == params[0] and row["to_event_id"] == params[1] and (len(params) == 2 or row["edge_type"] == params[2])
        ]
        self._set_results([{"present": 1}] if rows else [])

    def _get_event_edge(self, params: list[Any]) -> None:
        rows = [
            row
            for row in self.conn.tables["event_edges"].values()
            if row["from_event_id"] == params[0] and row["to_event_id"] == params[1]
        ]
        rows.sort(key=lambda row: row["confidence"], reverse=True)
        self._set_results(rows[:1])

    def _search_facts(self, params: list[Any]) -> None:
        query_terms = {term.lower() for term in params[0].split() if term}
        superseded = {row["to_fact_id"] for row in self.conn.tables["fact_relations"].values() if row["relation_type"] == "supersedes"}
        rows: list[dict[str, Any]] = []
        for row in self.conn.tables["memory_facts"].values():
            hits = sum(1 for term in query_terms if term in row["text"].lower())
            if not hits:
                continue
            scored = dict(row)
            scored["bm25_score"] = hits / max(1, len(query_terms))
            scored["semantic_score"] = 0.6
            active_prior = -0.15 if row["fact_id"] in superseded else 0.0
            scored["score"] = 0.50 * scored["semantic_score"] + 0.35 * scored["bm25_score"] + 0.10 * row["confidence"] + 0.05 * row["salience"] + active_prior
            rows.append(scored)
        rows.sort(key=lambda row: row["score"], reverse=True)
        self._set_results(rows[: int(params[-1])])

    def _search_events(self, params: list[Any]) -> None:
        query_terms = {term.lower() for term in params[0].split() if term}
        rows: list[dict[str, Any]] = []
        for row in self.conn.tables["events"].values():
            hits = sum(1 for term in query_terms if term in row["description"].lower())
            if not hits:
                continue
            scored = dict(row)
            scored["bm25_score"] = hits / max(1, len(query_terms))
            scored["temporal_fit"] = 0.2 if row["time_start"] else 0.0
            scored["score"] = 0.75 * scored["bm25_score"] + scored["temporal_fit"] + 0.05 * row["confidence"]
            rows.append(scored)
        rows.sort(key=lambda row: row["score"], reverse=True)
        self._set_results(rows[: int(params[-1])])

    def _set_results(self, rows: list[dict[str, Any]]) -> None:
        self._results = rows
        keys = list(rows[0].keys()) if rows else []
        self.description = [(key,) for key in keys]


class FakePostgresConnection:
    def __init__(self, *, fail_on_insert_table: str | None = None) -> None:
        self.fail_on_insert_table = fail_on_insert_table
        self.tables: dict[str, dict[str, dict[str, Any]]] = {
            "memory_facts": {},
            "fact_relations": {},
            "events": {},
            "event_edges": {},
        }
        self.executed: list[tuple[str, list[Any]]] = []
        self.committed = 0
        self.rolled_back = 0
        self.closed = False

    def cursor(self) -> FakePostgresCursor:
        return FakePostgresCursor(self)

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1

    def close(self) -> None:
        self.closed = True


def _fact_row(params: list[Any]) -> dict[str, Any]:
    return {
        "fact_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "subject": params[7],
        "predicate": params[8],
        "object": params[9],
        "text": params[10],
        "category": params[11],
        "polarity": params[12],
        "confidence": params[13],
        "salience": params[14],
        "observed_at": params[15],
        "valid_from": params[16],
        "valid_to": params[17],
        "source_span_ids": params[18],
        "linked_fact_ids": params[19],
        "embedding_dense": params[20],
        "hash": params[21],
        "metadata": params[22],
        "created_at": params[23],
    }


def _fact_relation_row(params: list[Any]) -> dict[str, Any]:
    return {
        "relation_id": params[0],
        "from_fact_id": params[1],
        "to_fact_id": params[2],
        "relation_type": params[3],
        "source_span_ids": params[4],
        "confidence": params[5],
        "created_at": "2026-06-01T10:00:00+00:00",
    }


def _event_row(params: list[Any]) -> dict[str, Any]:
    return {
        "event_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "event_type": params[7],
        "participants": params[8],
        "description": params[9],
        "time_start": params[10],
        "time_end": params[11],
        "time_granularity": params[12],
        "time_source": params[13],
        "source_span_ids": params[14],
        "fact_ids": params[15],
        "confidence": params[16],
        "created_at": "2026-06-01T10:00:00+00:00",
    }


def _event_edge_row(params: list[Any]) -> dict[str, Any]:
    return {
        "edge_id": params[0],
        "from_event_id": params[1],
        "to_event_id": params[2],
        "edge_type": params[3],
        "source_span_ids": params[4],
        "confidence": params[5],
        "created_at": "2026-06-01T10:00:00+00:00",
    }


if __name__ == "__main__":
    unittest.main()
