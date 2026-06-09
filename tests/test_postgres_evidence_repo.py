from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any

from fusion_memory.core.models import EvidenceSpan, Scope
from fusion_memory.core.text import stable_hash
from fusion_memory.storage.postgres_store import PostgresEvidenceRepository


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class PostgresEvidenceRepositoryTests(unittest.TestCase):
    def test_insert_get_list_and_duplicate_span(self) -> None:
        fake = FakePostgresConnection()
        repo = PostgresEvidenceRepository("postgresql://example/fusion", connect=lambda dsn: fake)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        span = make_span("span_1", scope, "Atlas uses Qdrant for retrieval.")

        self.assertTrue(repo.insert_span(span))
        self.assertFalse(repo.insert_span(span))

        insert_sql, insert_params = fake.executed[0]
        self.assertIn("cast(%s as vector)", insert_sql)
        self.assertIn("on conflict (span_id) do nothing", " ".join(insert_sql.lower().split()))
        self.assertEqual(insert_params[14], 10)
        self.assertEqual(insert_params[15], 12)
        self.assertTrue(str(insert_params[19]).startswith("["))
        self.assertGreaterEqual(fake.committed, 2)

        loaded = repo.get_span(span.span_id, scope, include_session=True)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, span.content)
        self.assertEqual(loaded.scope.session_id, "s")
        self.assertEqual(loaded.entities, ["Atlas", "Qdrant"])
        self.assertEqual(loaded.metadata["line_start"], 10)

        listed = repo.list_spans(scope, include_session=True)
        self.assertEqual([item.span_id for item in listed], [span.span_id])

        duplicate = repo.find_duplicate_span(span.content_hash, scope)
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate.span_id, span.span_id)

    def test_search_spans_uses_fts_pgvector_and_scope_filters(self) -> None:
        fake = FakePostgresConnection()
        repo = PostgresEvidenceRepository("postgresql://example/fusion", connect=lambda dsn: fake)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        repo.insert_span(make_span("span_1", scope, "Atlas uses Qdrant for retrieval."))
        repo.insert_span(make_span("span_2", scope, "Reports use PostgreSQL."))

        results = repo.search_spans("Qdrant Atlas", scope, limit=5, include_session=True)

        self.assertEqual(len(results), 1)
        span, scores = results[0]
        self.assertEqual(span.span_id, "span_1")
        self.assertGreater(scores["bm25_score"], 0)
        self.assertGreater(scores["semantic_score"], 0)
        self.assertGreater(scores["score"], 0)

        search_sql, search_params = fake.executed[-1]
        normalized = " ".join(search_sql.lower().split())
        self.assertIn("plainto_tsquery('simple', %s)", normalized)
        self.assertIn("embedding_dense <=> cast(%s as vector)", normalized)
        self.assertIn("e.session_id = %s", normalized)
        self.assertEqual(search_params[0], "Qdrant Atlas")
        self.assertEqual(search_params[-1], 5)

    def test_insert_span_rolls_back_on_error(self) -> None:
        fake = FakePostgresConnection(fail_on_insert=True)
        repo = PostgresEvidenceRepository("postgresql://example/fusion", connect=lambda dsn: fake)

        with self.assertRaises(RuntimeError):
            repo.insert_span(make_span("span_1", Scope(workspace_id="w"), "Atlas uses Qdrant."))

        self.assertEqual(fake.rolled_back, 1)
        self.assertEqual(fake.committed, 0)


def make_span(span_id: str, scope: Scope, content: str) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        scope=scope,
        turn_id="turn_1",
        speaker="user",
        span_type="turn",
        content=content,
        content_hash=stable_hash(f"user:{content}"),
        timestamp=ts("2026-06-01T10:00:00+00:00"),
        source_uri="chat://s/1",
        parent_span_id=None,
        entities=["Atlas", "Qdrant"] if "Qdrant" in content else ["PostgreSQL"],
        topics=[],
        metadata={"line_start": 10, "line_end": 12},
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
        if "insert into evidence_spans" in normalized:
            if self.conn.fail_on_insert:
                raise RuntimeError("insert failed")
            self._insert_evidence(params)
        elif normalized.startswith("with scored"):
            self._search_evidence(params)
        elif "from evidence_spans" in normalized and "content_hash = %s" in normalized:
            content_hash = params[-1]
            self._set_results([row for row in self.conn.rows.values() if row["content_hash"] == content_hash][:1])
        elif "from evidence_spans" in normalized and "span_id = %s" in normalized:
            row = self.conn.rows.get(params[0])
            self._set_results([row] if row else [])
        elif "from evidence_spans" in normalized:
            self._set_results(sorted(self.conn.rows.values(), key=lambda row: row["timestamp"]))
        else:
            self._set_results([])

    def fetchone(self) -> dict[str, Any] | None:
        return self._results[0] if self._results else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._results)

    def close(self) -> None:
        self.closed = True

    def _insert_evidence(self, params: list[Any]) -> None:
        span_id = params[0]
        if span_id in self.conn.rows:
            self.rowcount = 0
            self._set_results([])
            return
        row = {
            "span_id": span_id,
            "workspace_id": params[1],
            "user_id": params[2],
            "agent_id": params[3],
            "run_id": params[4],
            "session_id": params[5],
            "app_id": params[6],
            "turn_id": params[7],
            "speaker": params[8],
            "span_type": params[9],
            "content": params[10],
            "content_hash": params[11],
            "timestamp": params[12],
            "source_uri": params[13],
            "line_start": params[14],
            "line_end": params[15],
            "parent_span_id": params[16],
            "entities": params[17],
            "topics": params[18],
            "embedding_dense": params[19],
            "metadata": params[20],
            "created_at": "2026-06-01T10:00:00+00:00",
        }
        self.conn.rows[span_id] = row
        self.rowcount = 1
        self._set_results([])

    def _search_evidence(self, params: list[Any]) -> None:
        query_terms = {term.lower() for term in params[0].split() if term}
        rows: list[dict[str, Any]] = []
        for row in self.conn.rows.values():
            content = row["content"].lower()
            hits = sum(1 for term in query_terms if term in content)
            if not hits:
                continue
            scored = dict(row)
            scored["bm25_score"] = hits / max(1, len(query_terms))
            scored["semantic_score"] = 0.5
            scored["score"] = 0.55 * scored["semantic_score"] + 0.45 * scored["bm25_score"]
            rows.append(scored)
        rows.sort(key=lambda row: row["score"], reverse=True)
        self.rowcount = len(rows)
        self._set_results(rows[: int(params[-1])])

    def _set_results(self, rows: list[dict[str, Any]]) -> None:
        self._results = rows
        keys = list(rows[0].keys()) if rows else []
        self.description = [(key,) for key in keys]


class FakePostgresConnection:
    def __init__(self, *, fail_on_insert: bool = False) -> None:
        self.fail_on_insert = fail_on_insert
        self.rows: dict[str, dict[str, Any]] = {}
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


if __name__ == "__main__":
    unittest.main()
