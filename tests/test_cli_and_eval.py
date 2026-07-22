from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.eval.adapter import BenchmarkAdapter, EvalDocument, EvalQuery


class EvalAdapterTests(unittest.TestCase):
    def test_benchmark_adapter_reports_retrieval_match(self) -> None:
        service = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        adapter = BenchmarkAdapter(service, scope)
        adapter.ingest_documents(
            [
                EvalDocument(
                    id="doc1",
                    content="User said Atlas retrieval now uses Qdrant.",
                    timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
                    speaker="user",
                ),
                EvalDocument(
                    id="doc2",
                    content="Atlas retrieval backend is Qdrant for the benchmark fixture.",
                    timestamp=datetime(2026, 6, 2, tzinfo=timezone.utc),
                    speaker="user",
                ),
            ]
        )
        results = adapter.run_queries(
            [EvalQuery(id="q1", query="What does Atlas retrieval use?", gold_answers=["Qdrant"], category="factual_exact")]
        )
        report = adapter.report(results)
        self.assertEqual(results[0].answer_model, "local_extractive_v0")
        self.assertEqual(results[0].judge_model, "lexical_contains_v0")
        self.assertIn("Qdrant", results[0].answer)
        self.assertEqual(report["retrieval_match_rate"], 1.0)
        self.assertEqual(report["answer_match_rate"], 1.0)
        self.assertEqual(report["answer_model"], "local_extractive_v0")
        self.assertEqual(report["judge_model"], "lexical_contains_v0")
        self.assertGreater(report["avg_tokens_query"], 0)
        self.assertIn("p95", report["latency_ms"])
        ablation = adapter.run_ablation(
            [EvalQuery(id="q1", query="What does Atlas retrieval use?", gold_answers=["Qdrant"], category="factual_exact")]
        )
        self.assertEqual(set(ablation), {"fast", "balanced"})
        self.assertTrue(all(item["retrieval_match_rate"] == 1.0 for item in ablation.values()))
        component_ablation = adapter.run_component_ablation(
            [EvalQuery(id="q1", query="What does Atlas retrieval use?", gold_answers=["Qdrant"], category="factual_exact")]
        )
        self.assertEqual(set(component_ablation), {"L0", "L0+L1", "L0+L1+L2", "Full"})
        self.assertTrue(all("config" in item for item in component_ablation.values()))


if __name__ == "__main__":
    unittest.main()
