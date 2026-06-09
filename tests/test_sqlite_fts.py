from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fusion_memory import MemoryService, Scope


class SQLiteFTSTests(unittest.TestCase):
    def test_fts_sparse_source_is_used_for_span_fact_event_and_profile(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I prefer Qdrant for Atlas retrieval.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
        memory.add("Always give me concise technical answers.", scope, datetime(2026, 6, 2, tzinfo=timezone.utc))
        memory.add("Please keep responses concise but include implementation tradeoffs.", scope, datetime(2026, 6, 3, tzinfo=timezone.utc))
        memory.add("I tested BM25 yesterday.", scope, datetime(2026, 6, 4, tzinfo=timezone.utc))

        self.assertTrue(memory.store.fts_enabled)
        span_results = memory.store.search_spans("Qdrant", scope)
        fact_results = memory.store.search_facts("Qdrant", scope)
        event_results = memory.store.search_events("BM25", scope)
        profile_results = memory.store.search_entity_profiles("concise technical", scope)

        self.assertTrue(any(scores.get("sparse_source") == 1.0 for _, scores in span_results))
        self.assertTrue(any(scores.get("sparse_source") == 1.0 for _, scores in fact_results))
        self.assertTrue(any(scores.get("sparse_source") == 1.0 for _, scores in event_results))
        self.assertTrue(any(scores.get("sparse_source") == 1.0 for _, scores in profile_results))

    def test_fts_backfills_existing_sqlite_database_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "fm.sqlite3"
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            first = MemoryService(db_path)
            first.add("I prefer Qdrant for Atlas retrieval.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
            first.close()

            reopened = MemoryService(db_path)
            try:
                results = reopened.store.search_spans("Qdrant", scope)
                self.assertTrue(results)
                self.assertEqual(results[0][1].get("sparse_source"), 1.0)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()

