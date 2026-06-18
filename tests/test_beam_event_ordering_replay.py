from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from tools.beam_event_ordering_replay import _aggregate, _graph_items, _record_diagnostics, evaluate_gate, score_ordering_candidates


class BeamEventOrderingReplayTests(unittest.TestCase):
    def test_score_ordering_candidates_reports_precision_recall_f1_and_tau(self) -> None:
        score = score_ordering_candidates(
            ["1st: Core functionality", "2nd: Transaction error handling", "3rd: Security and deployment"],
            ["Core functionality setup", "Security and deployment", "Transaction error handling"],
        )

        self.assertEqual(score["matched"], 3)
        self.assertAlmostEqual(score["precision"], 1.0)
        self.assertAlmostEqual(score["recall"], 1.0)
        self.assertAlmostEqual(score["f1"], 1.0)
        self.assertLess(score["kendall_tau"], 1.0)
        self.assertGreaterEqual(score["kendall_tau_norm"], 0.0)

    def test_score_ordering_candidates_penalizes_missing_and_extra_items(self) -> None:
        score = score_ordering_candidates(
            ["schema setup", "crud implementation", "deployment"],
            ["schema setup", "unrelated billing notes"],
        )

        self.assertEqual(score["matched"], 1)
        self.assertAlmostEqual(score["precision"], 0.5)
        self.assertAlmostEqual(score["recall"], 1 / 3)
        self.assertAlmostEqual(score["f1"], 0.4)


class BeamEventOrderingGateTests(unittest.TestCase):
    def test_evaluate_gate_requires_graph_to_match_legacy_f1_and_tau(self) -> None:
        summary = {
            "graph": {"f1": 0.10, "kendall_tau_norm": 0.20, "empty_rate": 0.0},
            "legacy": {"f1": 0.20, "kendall_tau_norm": 0.25, "empty_rate": 0.0},
            "hybrid": {"f1": 0.18, "kendall_tau_norm": 0.24, "empty_rate": 0.0},
        }

        gate = evaluate_gate(summary)

        self.assertFalse(gate["passed"])
        self.assertIn("graph_f1_below_legacy", gate["failures"])
        self.assertIn("graph_tau_below_legacy", gate["failures"])

    def test_aggregate_reports_gate_fields_and_path_wins(self) -> None:
        records = [
            {
                "paths": {
                    "graph": {"metrics": {"precision": 1.0, "recall": 1.0, "f1": 0.8, "kendall_tau": 0.4, "kendall_tau_norm": 0.7, "system_count": 2, "matched": 2}},
                    "legacy": {"metrics": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "kendall_tau": 0.0, "kendall_tau_norm": 0.5, "system_count": 2, "matched": 1}},
                    "hybrid": {"metrics": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "kendall_tau": 0.2, "kendall_tau_norm": 0.6, "system_count": 2, "matched": 1}},
                },
            },
            {
                "paths": {
                    "graph": {"metrics": {"precision": 0.4, "recall": 0.4, "f1": 0.4, "kendall_tau": -0.2, "kendall_tau_norm": 0.4, "system_count": 1, "matched": 1}},
                    "legacy": {"metrics": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "kendall_tau": 0.6, "kendall_tau_norm": 0.8, "system_count": 1, "matched": 1}},
                    "hybrid": {"metrics": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "kendall_tau": 0.4, "kendall_tau_norm": 0.7, "system_count": 1, "matched": 1}},
                },
            },
        ]

        summary = _aggregate(records)

        self.assertFalse(summary["graph_vs_legacy_passed"])
        self.assertIn("graph_f1_below_legacy", summary["gate_failures"])
        self.assertEqual(summary["path_wins"]["f1"], {"graph": 1, "legacy": 1, "hybrid": 0})
        self.assertEqual(summary["path_wins"]["kendall_tau_norm"], {"graph": 1, "legacy": 1, "hybrid": 0})

    def test_record_diagnostics_reports_topic_drift_duplicate_labels_and_empty_graph(self) -> None:
        record = {
            "reference": ["Alpha build", "Beta launch"],
            "paths": {
                "graph": {
                    "items": ["Alpha build", "Alpha build", "Unrelated billing note"],
                    "metrics": {"system_count": 3},
                },
            },
        }

        diagnostics = _record_diagnostics(record)

        self.assertEqual(diagnostics["topic_drift_count"], 1)
        self.assertEqual(diagnostics["duplicate_label_count"], 1)
        self.assertFalse(diagnostics["graph_empty"])

    def test_graph_items_only_count_persisted_graph_candidates(self) -> None:
        service = SimpleNamespace(
            _event_ordering_graph_selector_candidates=MagicMock(
                return_value=[
                    SimpleNamespace(
                        source="event_ordering_graph_selector",
                        text="query-time fallback graph candidate",
                        metadata={},
                    ),
                    SimpleNamespace(
                        source="event_ordering_persisted_graph",
                        text="persisted graph candidate",
                        metadata={},
                    ),
                ]
            )
        )

        items, sources = _graph_items(service, "rank the work", SimpleNamespace(), 5)

        self.assertEqual(items, ["persisted graph candidate"])
        self.assertEqual(sources, ["event_ordering_persisted_graph"])
        service._event_ordering_graph_selector_candidates.assert_called_once_with(
            "rank the work",
            unittest.mock.ANY,
            limit=5,
            include_session=True,
        )


if __name__ == "__main__":
    unittest.main()
