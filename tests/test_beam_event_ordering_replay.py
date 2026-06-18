from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.beam_event_ordering_replay as replay
from tools.beam_event_ordering_replay import (
    _aggregate,
    _hybrid_items,
    _graph_items,
    _record_diagnostics,
    evaluate_gate,
    main,
    preflight_replay_environment_from_store,
    score_ordering_candidates,
)


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


class BeamReplayPreflightTests(unittest.TestCase):
    def test_preflight_reports_postgres_chronology_migration_status(self) -> None:
        class Store:
            def list_chronology_topics(self, scope, include_session=False):
                raise RuntimeError('relation "chronology_topics" does not exist')

        report = preflight_replay_environment_from_store(Store())

        self.assertFalse(report["chronology_tables_ready"])
        self.assertEqual(report["chronology_error"], "missing_chronology_tables")

    def test_hybrid_source_spans_skips_pack_for_model(self) -> None:
        pack = SimpleNamespace(
            source_spans=[
                {"content": "first source span", "candidate_source": "source_a"},
                {"conversation_content": "second source span", "selector": "source_b"},
            ],
            coverage={"event_ordering_shadow": {"selected_driver": "graph"}},
        )
        service = SimpleNamespace(answer_context=MagicMock(return_value=pack))

        with patch.object(replay, "_pack_for_model", side_effect=AssertionError("_pack_for_model should not be called")):
            items, sources, coverage = _hybrid_items(
                service,
                "rank the work",
                SimpleNamespace(),
                5,
                "event_ordering",
                hybrid_source="source_spans",
            )

        self.assertEqual(items, ["first source span", "second source span"])
        self.assertEqual(sources, ["source_a", "source_b"])
        self.assertEqual(coverage, {"event_ordering_shadow": {"selected_driver": "graph"}})

    def test_main_preflight_only_writes_preflight_report(self) -> None:
        output_path = "/tmp/beam-replay-preflight-only.json"
        args = SimpleNamespace(
            dataset="/unused",
            split="100k",
            workspace="ws",
            user_id="beam_user",
            agent_id="fusion_memory",
            run_id=None,
            session_id=None,
            db="postgresql://example",
            limit=8,
            query_ids=None,
            max_queries=None,
            gate=False,
            output=output_path,
            preflight_only=True,
            hybrid_source="model_pack",
        )

        with patch.object(replay.argparse.ArgumentParser, "parse_args", return_value=args), patch.object(
            replay, "preflight_replay_environment", return_value={"chronology_tables_ready": False, "chronology_error": "missing_chronology_tables"}
        ), patch.object(replay, "run_replay", side_effect=AssertionError("run_replay should not be called in preflight-only mode")), patch.object(
            replay, "print"
        ) as print_mock, patch.object(
            replay.Path, "write_text"
        ) as write_text_mock, patch.object(
            replay.Path, "mkdir"
        ) as mkdir_mock:
            main()

        mkdir_mock.assert_called_once()
        write_text_mock.assert_called_once()
        written = write_text_mock.call_args.args[0]
        self.assertEqual(
            replay.json.loads(written),
            {
                "preflight": {
                    "chronology_tables_ready": False,
                    "chronology_error": "missing_chronology_tables",
                }
            },
        )
        print_mock.assert_called_once_with(replay.json.dumps({"preflight": {"chronology_tables_ready": False, "chronology_error": "missing_chronology_tables"}, "output": "written"}, ensure_ascii=False))


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
                "coverage": {
                    "event_ordering_shadow": {"selected_driver": "graph"},
                    "dropped_high_signal_candidates": [{"candidate_id": "g1"}],
                },
                "paths": {
                    "graph": {"items": ["Implementation summary", "Schema setup"], "metrics": {"precision": 1.0, "recall": 1.0, "f1": 0.8, "kendall_tau": 0.4, "kendall_tau_norm": 0.7, "system_count": 2, "matched": 2}},
                    "legacy": {"metrics": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "kendall_tau": 0.0, "kendall_tau_norm": 0.5, "system_count": 2, "matched": 1}},
                    "hybrid": {"metrics": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "kendall_tau": 0.2, "kendall_tau_norm": 0.6, "system_count": 2, "matched": 1}},
                },
            },
            {
                "coverage": {
                    "event_ordering_shadow": {"selected_driver": "legacy_fallback"},
                    "dropped_high_signal_candidates": [{"candidate_id": "g2"}, {"candidate_id": "g3"}],
                },
                "paths": {
                    "graph": {"items": ["Implementation summary"], "metrics": {"precision": 0.4, "recall": 0.4, "f1": 0.4, "kendall_tau": -0.2, "kendall_tau_norm": 0.4, "system_count": 1, "matched": 1}},
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
        self.assertAlmostEqual(summary["graph_fallback_rate"], 0.5)
        self.assertEqual(summary["dropped_high_signal_candidate_count"], 3)
        self.assertEqual(summary["over_abstract_label_count"], 2)

    def test_record_diagnostics_reports_topic_drift_duplicate_labels_empty_graph_and_new_counters(self) -> None:
        record = {
            "reference": ["Alpha build", "Beta launch"],
            "coverage": {
                "event_ordering_shadow": {"selected_driver": "legacy_fallback"},
                "dropped_high_signal_candidates": [{"candidate_id": "g1"}, {"candidate_id": "g2"}],
            },
            "paths": {
                "graph": {
                    "items": ["Alpha build", "Alpha build", "Implementation summary", "Unrelated billing note"],
                    "sources": ["event_ordering_graph_selector"],
                    "metrics": {"system_count": 3},
                },
            },
        }

        diagnostics = _record_diagnostics(record)

        self.assertEqual(diagnostics["topic_drift_count"], 1)
        self.assertEqual(diagnostics["duplicate_label_count"], 1)
        self.assertFalse(diagnostics["graph_empty"])
        self.assertTrue(diagnostics["graph_fallback"])
        self.assertEqual(diagnostics["dropped_high_signal_candidate_count"], 2)
        self.assertEqual(diagnostics["over_abstract_label_count"], 1)

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
                        metadata={
                            "graph_selector_telemetry": {"selected_driver": "graph"},
                        },
                    ),
                ]
            )
        )

        items, sources, graph_fallback = _graph_items(service, "rank the work", SimpleNamespace(), 5)

        self.assertEqual(items, ["persisted graph candidate"])
        self.assertEqual(sources, ["event_ordering_persisted_graph"])
        self.assertTrue(graph_fallback)
        service._event_ordering_graph_selector_candidates.assert_called_once_with(
            "rank the work",
            unittest.mock.ANY,
            limit=5,
            include_session=True,
        )

    def test_graph_items_marks_fallback_false_when_persisted_graph_telemetry_stays_on_graph(self) -> None:
        service = SimpleNamespace(
            _event_ordering_graph_selector_candidates=MagicMock(
                return_value=[
                    SimpleNamespace(
                        source="event_ordering_persisted_graph",
                        text="persisted graph candidate",
                        metadata={
                            "persisted_graph_telemetry": {"selected_driver": "persisted_graph"},
                        },
                    )
                ]
            )
        )

        items, sources, graph_fallback = _graph_items(service, "rank the work", SimpleNamespace(), 5)

        self.assertEqual(items, ["persisted graph candidate"])
        self.assertEqual(sources, ["event_ordering_persisted_graph"])
        self.assertFalse(graph_fallback)

    def test_record_diagnostics_uses_graph_sources_when_present_for_fallback(self) -> None:
        record = {
            "coverage": {"event_ordering_shadow": {"selected_driver": "graph"}},
            "paths": {
                "graph": {
                    "items": ["persisted graph candidate"],
                    "sources": ["event_ordering_graph_selector"],
                    "metrics": {"system_count": 1},
                }
            },
        }

        diagnostics = _record_diagnostics(record)

        self.assertTrue(diagnostics["graph_fallback"])


if __name__ == "__main__":
    unittest.main()
