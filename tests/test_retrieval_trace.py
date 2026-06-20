from __future__ import annotations

import unittest

from fusion_memory.retrieval.retrieval_trace import RetrievalTraceBuilder


class RetrievalTraceBuilderTests(unittest.TestCase):
    def test_builds_pipeline_sections_without_raw_text(self) -> None:
        builder = RetrievalTraceBuilder(query_type="event_ordering", mode="benchmark")
        builder.query_understanding(language="en", intent="event_ordering", features=["temporal", "multi_condition"])
        builder.candidate_recall(source_counts={"event_ordering_episode_recall": 3, "event_ordering_persisted_graph": 2})
        builder.candidate_fusion(selected_sources=["event_ordering_episode_recall"], dropped_count=1)
        builder.evidence_output(source_span_count=3, coverage_insufficient=False)

        trace = builder.to_dict()

        self.assertEqual(trace["query_understanding"]["intent"], "event_ordering")
        self.assertEqual(trace["candidate_recall"]["source_counts"]["event_ordering_episode_recall"], 3)
        self.assertEqual(trace["candidate_fusion"]["dropped_count"], 1)
        self.assertFalse(trace["evidence_output"]["coverage_insufficient"])
        self.assertNotIn("query", trace)

    def test_pipeline_layers_expose_stable_boundaries(self) -> None:
        builder = RetrievalTraceBuilder(query_type="event_ordering", mode="benchmark")
        builder.query_understanding(language="zh", intent="event_ordering", features=["temporal"])
        builder.candidate_recall(source_counts={"graph": 2})
        builder.candidate_fusion(selected_sources=["graph"], dropped_count=0)
        builder.evidence_output(source_span_count=2, coverage_insufficient=False)

        layers = builder.pipeline_layers()

        self.assertEqual(list(layers), ["QueryUnderstanding", "CandidateRecall", "CandidateFusion", "EvidencePackBuilder"])
        self.assertEqual(layers["QueryUnderstanding"]["language"], "zh")
        self.assertEqual(layers["CandidateRecall"]["source_counts"], {"graph": 2})
