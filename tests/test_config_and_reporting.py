from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.core.config import MemoryConfig
from fusion_memory.eval.adapter import BenchmarkAdapter, EvalDocument, EvalQuery


class ConfigAndReportingTests(unittest.TestCase):
    def test_config_controls_chunking_quota_and_trace_snapshot(self) -> None:
        config = MemoryConfig(chunk_size_tokens=5, chunk_overlap_tokens=1, raw_evidence_quotas={"factual_exact": 1})
        memory = MemoryService(config=config)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        result = memory.add(
            {
                "role": "document",
                "content": "one two three four five six seven eight nine",
                "source_uri": "doc://config",
            },
            scope,
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        )

        spans = [memory.get(span_id, "span") for span_id in result.span_ids]
        chunks = [span for span in spans if span and span.span_type == "document_chunk"]
        self.assertGreaterEqual(len(chunks), 2)
        memory.add("I prefer Qdrant for Atlas retrieval.", scope, datetime(2026, 6, 2, tzinfo=timezone.utc))
        trace = memory.debug_trace(result.trace_id)
        self.assertEqual(trace["config"]["chunk_size_tokens"], 5)

        pack = memory.answer_context("one seven", scope)
        self.assertEqual(pack.coverage["source_span_quota_required"], 1)

        audit_events = memory.audit_events(scope)
        self.assertTrue(any(event["event_type"] == "memory.add" for event in audit_events))
        self.assertTrue(any(event["event_type"] == "memory.search" for event in audit_events))

        encoding_report = memory.encoding_report(scope)
        self.assertGreater(encoding_report["total"], 0)
        self.assertEqual(encoding_report["accept_source_coverage"], 1.0)

    def test_benchmark_report_includes_config_snapshot(self) -> None:
        config = MemoryConfig(retrieval_output_n=3, raw_evidence_quotas={"factual_exact": 1})
        service = MemoryService(config=config)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        adapter = BenchmarkAdapter(service, scope)
        adapter.ingest_documents(
            [
                EvalDocument(
                    id="doc1",
                    content="Atlas retrieval uses Qdrant.",
                    timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
                    speaker="user",
                )
            ]
        )
        report = adapter.report(
            adapter.run_queries(
                [EvalQuery(id="q1", query="What does Atlas retrieval use?", gold_answers=["Qdrant"], category="factual_exact")]
            )
        )

        self.assertEqual(report["config"]["retrieval_output_n"], 3)
        self.assertEqual(report["config"]["raw_evidence_quotas"]["factual_exact"], 1)
        self.assertIn("encoding_report", report)
        self.assertIn("profile_report", report)


if __name__ == "__main__":
    unittest.main()
