from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fusion_memory import MemoryService, Scope
from fusion_memory.retrieval.utility_model import LogisticUtilityScorer
from fusion_memory.storage.postgres_store import POSTGRES_TABLES, PostgresMigrationRunner
from fusion_memory.storage.token_store import PostgresTokenStore


class UtilityAndMigrationTests(unittest.TestCase):
    def test_utility_scorer_trains_saves_loads_and_writes_shadow_trace(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I prefer Qdrant for Atlas retrieval.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
        for _ in range(3):
            memory.search("What do I prefer for Atlas?", scope)
            memory.search("What is my unknown cluster name?", scope)
        report = memory.train_utility_scorer()
        self.assertGreater(report.used_examples, 0)
        self.assertGreaterEqual(report.ndcg_at_10, 0.0)
        self.assertGreaterEqual(report.mrr, 0.0)
        self.assertTrue(memory.utility_scorer.trained)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utility.json"
            memory.save_utility_scorer(path)
            loaded = LogisticUtilityScorer.load(path)
            self.assertTrue(loaded.trained)
            self.assertEqual(set(loaded.feature_names), set(memory.utility_scorer.feature_names))

        result = memory.search("What do I prefer for Atlas?", scope)
        trace = memory.debug_trace(result.trace_id)
        self.assertIsNotNone(trace)
        self.assertTrue(trace["utility_shadow"]["enabled"])
        self.assertTrue(trace["utility_shadow"]["ranking"])

    def test_postgres_migration_contains_required_tables_and_indexes(self) -> None:
        migration = Path("fusion_memory/storage/migrations/postgres/001_init.sql").read_text(encoding="utf-8")
        for table in [
            "evidence_spans",
            "memory_facts",
            "fact_relations",
            "events",
            "event_edges",
            "current_views",
            "entity_profiles",
            "entities",
            "encoding_decisions",
            "retrieval_utility_examples",
            "debug_traces",
            "audit_events",
            "background_tasks",
        ]:
            self.assertIn(f"create table if not exists {table}", migration)
        self.assertIn("create extension if not exists vector", migration)
        self.assertIn("using hnsw", migration)
        self.assertIn("vector(1024)", migration)
        self.assertNotIn("vector(96)", migration)
        self.assertIn("debug_traces_scope_idx", migration)
        self.assertIn("background_tasks_dedupe_idx", migration)
        self.assertNotIn(" uuid", migration)

    def test_postgres_migration_runner_applies_schema_with_injected_connection(self) -> None:
        fake = FakePostgresConnection()
        runner = PostgresMigrationRunner("postgresql://example/fusion", connect=lambda dsn: fake)

        report = runner.migrate()

        self.assertEqual(report.backend, "postgres")
        self.assertEqual(report.tables, POSTGRES_TABLES)
        self.assertGreater(report.applied_statements, 10)
        self.assertTrue(fake.committed)
        self.assertFalse(fake.rolled_back)
        self.assertTrue(any("create extension if not exists vector" in statement for statement in fake.executed))
        self.assertTrue(any("create table if not exists evidence_spans" in statement for statement in fake.executed))


class FakePostgresCursor:
    def __init__(self, conn: "FakePostgresConnection") -> None:
        self.conn = conn
        self.closed = False

    def execute(self, statement: str) -> None:
        self.conn.executed.append(statement)

    def close(self) -> None:
        self.closed = True


class FakePostgresConnection:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakePostgresCursor:
        return FakePostgresCursor(self)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()


def test_migration_runner_serializes_duplicate_startup_with_transaction_lock(tmp_path: Path) -> None:
    migration_dir = tmp_path / "postgres"
    migration_dir.mkdir()
    migration = migration_dir / "001_test.sql"
    migration.write_text("create table migration_test (id text);", encoding="utf-8")
    state = SharedMigrationState()

    first = PostgresMigrationRunner("postgresql://example/fusion", connect=lambda _dsn: StatefulMigrationConnection(state), migration_path=migration)
    second = PostgresMigrationRunner("postgresql://example/fusion", connect=lambda _dsn: StatefulMigrationConnection(state), migration_path=migration)

    first.migrate()
    second.migrate()

    assert state.executed_migration_bodies == 1
    assert state.lock_count == 2
    assert state.version_queries_after_lock


def test_token_store_verification_updates_last_used_in_a_short_transaction() -> None:
    connection = TokenStoreConnection()
    store = PostgresTokenStore(lambda: connection, pepper="pepper")

    record = store.verify_digest("digest")

    assert record is not None
    assert record.token_id == "token-1"
    assert connection.committed is True
    assert connection.rolled_back is False
    assert connection.updated_last_used is True
    assert connection.closed is True


class SharedMigrationState:
    def __init__(self) -> None:
        self.versions: set[str] = set()
        self.executed_migration_bodies = 0
        self.lock_count = 0
        self.version_queries_after_lock = True


class StatefulMigrationConnection:
    def __init__(self, state: SharedMigrationState) -> None:
        self.state = state

    def cursor(self):
        return StatefulMigrationCursor(self.state)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class StatefulMigrationCursor:
    def __init__(self, state: SharedMigrationState) -> None:
        self.state = state
        self._row = None
        self._locked = False

    def execute(self, statement: str, params=None) -> None:
        normalized = " ".join(statement.split()).lower()
        if "pg_advisory_xact_lock" in normalized:
            self._locked = True
            self.state.lock_count += 1
        elif normalized.startswith("select 1 from fusion_memory_schema_migrations"):
            self.state.version_queries_after_lock &= self._locked
            version = params[0] if params else statement.split("'")[-2]
            self._row = (1,) if version in self.state.versions else None
        elif normalized.startswith("insert into fusion_memory_schema_migrations"):
            version = params[0] if params else statement.split("'")[-2]
            self.state.versions.add(version)
        elif normalized.startswith("create table migration_test"):
            self.state.executed_migration_bodies += 1

    def fetchone(self):
        return self._row

    def close(self) -> None:
        pass


class TokenStoreConnection:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.updated_last_used = False

    def cursor(self):
        return TokenStoreCursor(self)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class TokenStoreCursor:
    def __init__(self, conn: TokenStoreConnection) -> None:
        self.conn = conn

    def execute(self, statement: str, params=None) -> None:
        if statement.strip().lower().startswith("update memory_api_tokens set last_used_at"):
            self.conn.updated_last_used = True

    def fetchone(self):
        return (
            "token-1",
            "digest",
            "user-a",
            ["memory:read"],
            None,
            None,
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            None,
        )

    def close(self) -> None:
        pass
