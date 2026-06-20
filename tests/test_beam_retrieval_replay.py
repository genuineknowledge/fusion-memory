from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.beam_retrieval_replay as replay


class BeamRetrievalReplayTests(unittest.TestCase):
    def test_category_filter_parses_current_multi_and_zh_aliases(self) -> None:
        self.assertEqual(
            replay._parse_categories("current_value,multi_condition,zh_recall"),
            {"current_value", "multi_condition", "zh_recall"},
        )

    def test_record_summary_counts_coverage_and_source_spans(self) -> None:
        records = [
            {"category": "current_value", "source_span_count": 2, "coverage_insufficient": False},
            {"category": "current_value", "source_span_count": 0, "coverage_insufficient": True},
            {"category": "zh_recall", "source_span_count": 3, "coverage_insufficient": False},
        ]

        summary = replay._summarize_records(records)

        self.assertEqual(summary["categories"]["current_value"]["query_count"], 2)
        self.assertEqual(summary["categories"]["current_value"]["coverage_insufficient_rate"], 0.5)
        self.assertEqual(summary["categories"]["current_value"]["mean_source_span_count"], 1.0)
        self.assertEqual(summary["categories"]["zh_recall"]["query_count"], 1)

    def test_run_replay_writes_records_with_pipeline_trace(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={"coverage_insufficient": False},
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            report = replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["summary"]["categories"]["current_value"]["query_count"], 1)
        self.assertIn("pipeline_trace", payload["records"][0])


if __name__ == "__main__":
    unittest.main()
