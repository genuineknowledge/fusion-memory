from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fusion_memory import MemoryService, Scope
from fusion_memory.retrieval.utility_model import LogisticUtilityScorer
from fusion_memory.storage.postgres_store import POSTGRES_TABLES, PostgresMigrationRunner


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
