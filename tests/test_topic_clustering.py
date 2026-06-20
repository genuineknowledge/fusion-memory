from __future__ import annotations

import unittest

from fusion_memory.retrieval.topic_clustering import cluster_topic_label, cluster_topic_telemetry


class TopicClusteringTests(unittest.TestCase):
    def test_cluster_uses_session_hint_for_related_fragment(self) -> None:
        decision = cluster_topic_label(
            "Then I compared median formulas and altitude methods.",
            session_hint="triangle geometry",
            previous_label="triangle classification",
        )

        self.assertEqual(decision.label, "triangle geometry")
        self.assertGreaterEqual(decision.confidence, 0.70)
        self.assertIn("session_hint", decision.reasons)

    def test_cluster_keeps_strong_taxonomy_label(self) -> None:
        decision = cluster_topic_label(
            "I need the OpenClaw memory adapter to use beginner friendly errors.",
            session_hint="triangle geometry",
            previous_label="triangle geometry",
        )

        self.assertNotEqual(decision.label, "triangle geometry")
        self.assertIn("taxonomy", decision.reasons)

    def test_cluster_telemetry_counts_merged_and_taxonomy_decisions(self) -> None:
        decisions = [
            cluster_topic_label("Then I compared median formulas.", session_hint="triangle geometry"),
            cluster_topic_label("I need the OpenClaw adapter.", session_hint="triangle geometry"),
        ]

        telemetry = cluster_topic_telemetry(decisions)

        self.assertEqual(telemetry["decision_count"], 2)
        self.assertGreaterEqual(telemetry["merged_by_session_hint"], 1)
        self.assertGreaterEqual(telemetry["taxonomy_count"], 1)
