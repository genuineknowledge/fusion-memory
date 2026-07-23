from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.core.text import tokenize


class RerankAndBudgetTests(unittest.TestCase):
    def test_balanced_mode_applies_reranker_with_canonical_trace(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            [
                {"role": "user", "content": "I tested BM25 yesterday."},
                {"role": "user", "content": "After BM25, I added dense retrieval."},
                {"role": "user", "content": "After dense retrieval, I added graph expansion."},
                {"role": "user", "content": "After graph expansion, I added reranking."},
                {"role": "user", "content": "After reranking, I added evidence packing."},
                {"role": "user", "content": "After evidence packing, I added evaluation reports."},
            ],
            scope,
            datetime(2026, 6, 5, tzinfo=timezone.utc),
        )
        result = memory.search("Which happened before reranking?", scope, options={"mode": "balanced", "limit": 6})
        trace = memory.debug_trace(result.trace_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["mode"], "balanced")
        self.assertEqual(trace["stages"], ["plan", "recall", "fusion", "selection"])
        self.assertLessEqual(len(result.candidates), 6)
        self.assertTrue(result.candidates)
        self.assertTrue(all("rerank_score" in candidate.scores for candidate in result.candidates))

    def test_evidence_pack_respects_token_budget_for_source_spans(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        long_doc = " ".join(f"AtlasToken{i}" for i in range(120))
        memory.add(
            {"role": "document", "content": long_doc, "source_uri": "doc://long", "chunk_size_tokens": 80, "chunk_overlap_tokens": 10},
            scope,
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        pack = memory.answer_context("AtlasToken50", scope, budget={"token_budget": 12, "limit": 6})
        self.assertLessEqual(pack.coverage["estimated_source_tokens"], 12)
        self.assertLessEqual(sum(len(tokenize(span["content"])) for span in pack.source_spans), 12)


if __name__ == "__main__":
    unittest.main()
