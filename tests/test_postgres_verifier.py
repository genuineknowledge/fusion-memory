from __future__ import annotations

import os
import unittest

from fusion_memory.core.models import AddResult, Candidate, EvidencePack, SearchResult
from fusion_memory.storage.postgres_store import PostgresMigrationReport
from fusion_memory.storage.postgres_verifier import verify_postgres_backend


class PostgresVerifierTests(unittest.TestCase):
    def test_verify_postgres_backend_runs_migration_and_service_smoke_with_masked_dsn(self) -> None:
        runner = FakeRunner()
        service = FakeService()

        report = verify_postgres_backend(
            "postgresql://user:secret@example:5432/fusion",
            runner_factory=lambda dsn: runner,
            service_factory=lambda dsn: service,
        )

        self.assertTrue(runner.migrated)
        self.assertTrue(runner.closed)
        self.assertTrue(service.closed)
        self.assertEqual(report.dsn, "postgresql://user:***@example:5432/fusion")
        self.assertTrue(all(report.checks.values()))
        self.assertEqual(report.migration["applied_statements"], 3)
        self.assertEqual(report.add["span_count"], 1)
        self.assertEqual(report.search["candidate_count"], 1)
        self.assertEqual(report.answer_context["source_span_count"], 1)
        self.assertEqual(report.tasks["processed_count"], 1)
        self.assertTrue(report.scope["workspace_id"].startswith("fm_verify_"))

    @unittest.skipUnless(os.environ.get("FUSION_MEMORY_POSTGRES_DSN"), "set FUSION_MEMORY_POSTGRES_DSN to run live Postgres verification")
    def test_live_postgres_backend_smoke_when_dsn_is_provided(self) -> None:
        report = verify_postgres_backend(os.environ["FUSION_MEMORY_POSTGRES_DSN"])

        self.assertTrue(all(report.checks.values()), report.checks)


class FakeRunner:
    def __init__(self) -> None:
        self.migrated = False
        self.closed = False

    def migrate(self) -> PostgresMigrationReport:
        self.migrated = True
        return PostgresMigrationReport(
            backend="postgres",
            migration_path="/fake/001_init.sql",
            applied_statements=3,
            tables=["evidence_spans", "memory_facts"],
        )

    def close(self) -> None:
        self.closed = True


class FakeService:
    def __init__(self) -> None:
        self.closed = False

    def add(self, input_data, scope, session_time, metadata=None) -> AddResult:
        return AddResult(
            span_ids=["span_1"],
            accepted_fact_ids=["fact_1"],
            accepted_event_ids=[],
            updated_view_ids=["view_1"],
            updated_profile_ids=[],
            quarantined_candidate_ids=[],
            trace_id="trace_add",
        )

    def search(self, query, scope, options=None) -> SearchResult:
        return SearchResult(
            candidates=[
                Candidate(
                    id="fact_1",
                    type="fact",
                    text="Atlas uses Qdrant.",
                    source="l1_fact_hybrid",
                    scores={"score": 1.0},
                    source_span_ids=["span_1"],
                )
            ],
            trace_id="trace_search",
            coverage={},
        )

    def answer_context(self, query, scope, budget=None) -> EvidencePack:
        return EvidencePack(
            query=query,
            answer_policy="answer_if_supported",
            current_views=[],
            entity_profiles=[],
            facts=[{"fact_id": "fact_1"}],
            events=[],
            source_spans=[{"span_id": "span_1"}],
            conflicts=[],
            coverage={},
            debug_trace=[],
        )

    def list_background_tasks(self, scope, status=None) -> list[dict]:
        return [{"task_id": "task_1", "status": "pending"}]

    def process_background_tasks(self, scope, limit=10) -> dict:
        return {"processed_count": 1, "status_counts": {"succeeded": 1}, "tasks": []}

    def audit_events(self, scope, limit=20) -> list[dict]:
        return [{"audit_id": "audit_1"}]

    def close(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()
