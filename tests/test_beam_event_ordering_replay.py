from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.beam_event_ordering_replay as replay
from tools.beam_event_ordering_replay import (
    _aggregate,
    _compact_coverage,
    _dual_graph_legacy_items,
    _hybrid_items,
    _summary_for_stdout,
    _graph_items,
    _record_diagnostics,
    evaluate_gate,
    main,
    preflight_replay_query_scopes_from_store,
    preflight_replay_environment_from_store,
    run_replay,
    score_ordering_candidates,
)


class BeamReplaySummaryTests(unittest.TestCase):
    def test_summary_for_stdout_includes_dual_and_cluster_fields(self) -> None:
        report = {
            "workspace": "w",
            "split": "100k",
            "query_count": 2,
            "summary": {
                "dual_vs_legacy_passed": True,
                "dual_lift_over_legacy_f1": 0.03,
                "cluster_diagnostics": {"expanded_query_count": 1},
            },
        }

        summary = _summary_for_stdout(report)

        self.assertTrue(summary["dual_vs_legacy_passed"])
        self.assertEqual(summary["dual_lift_over_legacy_f1"], 0.03)
        self.assertEqual(summary["cluster_expanded_query_count"], 1)


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

    def test_aggregate_reports_dual_lift_and_cluster_diagnostics(self) -> None:
        records = [
            {
                "paths": {
                    "legacy": {"active": True, "metrics": {"f1": 0.25, "kendall_tau_norm": 0.40, "system_count": 2, "matched": 1}},
                    "dual": {"active": True, "metrics": {"f1": 0.50, "kendall_tau_norm": 0.60, "system_count": 2, "matched": 2}},
                    "graph": {"active": True, "metrics": {"f1": 0.10, "kendall_tau_norm": 0.30, "system_count": 1, "matched": 0}},
                    "hybrid": {"active": False},
                },
                "coverage": {"event_ordering_graph": {"selected_topic_count": 2, "cluster_expanded_topic_ids": ["topic-b"]}},
            }
        ]

        summary = _aggregate(records)

        self.assertAlmostEqual(summary["dual_lift_over_legacy_f1"], 0.25)
        self.assertAlmostEqual(summary["dual_lift_over_legacy_tau"], 0.20)
        self.assertEqual(summary["cluster_diagnostics"]["expanded_query_count"], 1)

    def test_aggregate_reports_dual_lift_none_when_dual_or_legacy_inactive(self) -> None:
        records = [
            {
                "paths": {
                    "legacy": {"active": True, "metrics": {"f1": 0.25, "kendall_tau_norm": 0.40, "system_count": 2, "matched": 1}},
                    "dual": {"active": False},
                    "graph": {"active": False},
                    "hybrid": {"active": False},
                },
                "coverage": {},
            }
        ]

        summary = _aggregate(records)

        self.assertIsNone(summary["dual_lift_over_legacy_f1"])
        self.assertIsNone(summary["dual_lift_over_legacy_tau"])


class BeamReplayPreflightTests(unittest.TestCase):
    def test_preflight_reports_postgres_chronology_migration_status(self) -> None:
        class Store:
            def list_chronology_topics(self, scope, include_session=False):
                raise RuntimeError('relation "chronology_topics" does not exist')

        report = preflight_replay_environment_from_store(Store())

        self.assertEqual(report["status"], "failure")
        self.assertEqual(report["error"], "missing_chronology_tables")
        self.assertFalse(report["chronology_tables_ready"])
        self.assertEqual(report["chronology_error"], "missing_chronology_tables")

    def test_preflight_accepts_memory_service_wrapper(self) -> None:
        class Store:
            def list_chronology_topics(self, scope, include_session=False):
                return []

        report = preflight_replay_environment_from_store(SimpleNamespace(store=Store()))

        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["chronology_tables_ready"])

    def test_preflight_reports_workspace_not_backfilled_when_graph_counts_are_empty(self) -> None:
        class Store:
            def list_chronology_topics(self, scope, include_session=False):
                return []

            def list_chronology_event_nodes(self, scope, include_session=False, topic_ids=None):
                return []

            def list_chronology_event_edges(self, node_ids):
                return []

        report = preflight_replay_environment_from_store(
            Store(),
            scope=SimpleNamespace(workspace_id="beam-ws"),
            min_topics=1,
            min_nodes=2,
            min_edges=1,
        )

        self.assertEqual(report["status"], "failure")
        self.assertEqual(report["error"], "persisted_graph_not_backfilled")
        self.assertTrue(report["chronology_tables_ready"])
        self.assertFalse(report["persisted_graph_ready"])
        self.assertEqual(report["chronology_counts"], {"topics": 0, "nodes": 0, "edges": 0})

    def test_preflight_reports_workspace_backfilled_when_counts_meet_threshold(self) -> None:
        class Store:
            def list_chronology_topics(self, scope, include_session=False):
                return [SimpleNamespace(topic_id="topic-1")]

            def list_chronology_event_nodes(self, scope, include_session=False, topic_ids=None):
                return [SimpleNamespace(node_id="node-1"), SimpleNamespace(node_id="node-2")]

            def list_chronology_event_edges(self, node_ids):
                return [SimpleNamespace(edge_id="edge-1")]

        report = preflight_replay_environment_from_store(
            Store(),
            scope=SimpleNamespace(workspace_id="beam-ws"),
            min_topics=1,
            min_nodes=2,
            min_edges=1,
        )

        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["chronology_tables_ready"])
        self.assertTrue(report["persisted_graph_ready"])
        self.assertEqual(report["chronology_counts"], {"topics": 1, "nodes": 2, "edges": 1})

    def test_query_scope_preflight_reports_empty_session_scopes(self) -> None:
        class Store:
            def list_chronology_topics(self, scope, include_session=False):
                return [] if scope.session_id == "s-empty" else [SimpleNamespace(topic_id="topic-1")]

            def list_chronology_event_nodes(self, scope, include_session=False, topic_ids=None):
                return [] if scope.session_id == "s-empty" else [SimpleNamespace(node_id="n1"), SimpleNamespace(node_id="n2")]

            def list_chronology_event_edges(self, node_ids):
                return [] if not node_ids else [SimpleNamespace(edge_id="edge-1")]

        report = preflight_replay_query_scopes_from_store(
            Store(),
            [
                SimpleNamespace(workspace_id="ws", run_id="r", session_id="s-ready"),
                SimpleNamespace(workspace_id="ws", run_id="r", session_id="s-empty"),
            ],
            min_topics=1,
            min_nodes=2,
            min_edges=1,
        )

        self.assertEqual(report["checked"], 2)
        self.assertEqual(report["ready"], 1)
        self.assertEqual(report["not_ready"], 1)
        self.assertEqual(report["empty"], 1)
        self.assertEqual(report["failure_samples"][0]["session_id"], "s-empty")

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

    def test_compact_coverage_preserves_rule_hits_for_audit(self) -> None:
        coverage = _compact_coverage(
            {
                "query_type": "event_ordering",
                "rule_hits": [{"rule_id": "event_ordering.legacy_rescue", "text_hash": "abc123"}],
                "unrelated": "drop",
            }
        )

        self.assertEqual(
            coverage["rule_hits"],
            [{"rule_id": "event_ordering.legacy_rescue", "text_hash": "abc123"}],
        )
        self.assertNotIn("unrelated", coverage)

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
            replay,
            "preflight_replay_environment",
            return_value={
                "status": "failure",
                "error": "missing_chronology_tables",
                "chronology_tables_ready": False,
                "chronology_error": "missing_chronology_tables",
            },
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
                    "status": "failure",
                    "error": "missing_chronology_tables",
                    "chronology_tables_ready": False,
                    "chronology_error": "missing_chronology_tables",
                }
            },
        )
        print_mock.assert_called_once_with(
            replay.json.dumps(
                {
                    "preflight": {
                        "status": "failure",
                        "error": "missing_chronology_tables",
                        "chronology_tables_ready": False,
                        "chronology_error": "missing_chronology_tables",
                    },
                    "output": "written",
                },
                ensure_ascii=False,
            )
        )

    def test_main_preflight_only_writes_json_when_initialization_fails(self) -> None:
        output_path = "/tmp/beam-replay-preflight-init-failure.json"
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
            replay, "memory_service_from_env", side_effect=RuntimeError("connect failed")
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
                    "status": "failure",
                    "error": "RuntimeError",
                    "chronology_tables_ready": False,
                    "chronology_error": "RuntimeError",
                }
            },
        )
        print_mock.assert_called_once_with(
            replay.json.dumps(
                {
                    "preflight": {
                        "status": "failure",
                        "error": "RuntimeError",
                        "chronology_tables_ready": False,
                        "chronology_error": "RuntimeError",
                    },
                    "output": "written",
                },
                ensure_ascii=False,
            )
        )


class BeamReplayBucketTests(unittest.TestCase):
    def test_bucket_summary_groups_event_ordering_cases(self) -> None:
        records = [
            {"bucket": "explicit_order", "paths": {"graph": {"metrics": {"f1": 1.0, "kendall_tau_norm": 1.0}}}},
            {"bucket": "explicit_order", "paths": {"graph": {"metrics": {"f1": 0.0, "kendall_tau_norm": 0.5}}}},
            {"bucket": "long_mixed_topic", "paths": {"graph": {"metrics": {"f1": 0.5, "kendall_tau_norm": 0.75}}}},
        ]

        summary = replay._bucket_summary(records, path="graph")

        self.assertEqual(summary["explicit_order"]["count"], 2)
        self.assertAlmostEqual(summary["explicit_order"]["f1"], 0.5)
        self.assertEqual(summary["long_mixed_topic"]["count"], 1)


class BeamReplayModeTests(unittest.TestCase):
    def test_dual_graph_legacy_items_orders_graph_first_then_legacy_fill(self) -> None:
        service = SimpleNamespace(
            _event_ordering_graph_selector_candidates=MagicMock(
                return_value=[
                    SimpleNamespace(
                        id="g1",
                        source="event_ordering_persisted_graph",
                        text="schema setup",
                        source_span_ids=["span-1"],
                        metadata={},
                    ),
                    SimpleNamespace(
                        id="g2",
                        source="event_ordering_persisted_graph",
                        text="crud implementation",
                        source_span_ids=["span-2"],
                        metadata={},
                    ),
                ]
            ),
            planner=SimpleNamespace(plan=MagicMock(return_value=SimpleNamespace())),
            _event_ordering_episode_recall_candidates=MagicMock(
                return_value=[
                    SimpleNamespace(
                        id="legacy-dup",
                        source="event_ordering_episode_recall",
                        text="schema setup",
                        source_span_ids=["span-1"],
                        metadata={},
                    ),
                    SimpleNamespace(
                        id="legacy-fill",
                        source="event_ordering_episode_recall",
                        text="deployment checks",
                        source_span_ids=["span-3"],
                        metadata={},
                    ),
                ]
            ),
            _event_ordering_timeline_candidates=MagicMock(return_value=[]),
            _event_ordering_event_candidates=MagicMock(return_value=[]),
        )

        items, sources = _dual_graph_legacy_items(service, "rank the work", SimpleNamespace(), 3)

        self.assertEqual(items, ["schema setup", "crud implementation", "deployment checks"])
        self.assertEqual(
            sources,
            [
                "event_ordering_episode_recall",
                "event_ordering_persisted_graph",
                "event_ordering_episode_recall",
            ],
        )

    def test_dual_graph_legacy_items_uses_graph_order_over_legacy_text_when_aligned(self) -> None:
        service = SimpleNamespace(
            _event_ordering_graph_selector_candidates=MagicMock(
                return_value=[
                    SimpleNamespace(
                        id="g-crud",
                        source="event_ordering_persisted_graph",
                        text="crud implementation",
                        source_span_ids=["span-crud"],
                        metadata={},
                    ),
                    SimpleNamespace(
                        id="g-schema",
                        source="event_ordering_persisted_graph",
                        text="schema setup",
                        source_span_ids=["span-schema"],
                        metadata={},
                    ),
                ]
            ),
            planner=SimpleNamespace(plan=MagicMock(return_value=SimpleNamespace())),
            _event_ordering_episode_recall_candidates=MagicMock(
                return_value=[
                    SimpleNamespace(
                        id="legacy-schema",
                        source="event_ordering_episode_recall",
                        text="long user request about initial schema setup and models",
                        source_span_ids=["span-schema"],
                        metadata={},
                    ),
                    SimpleNamespace(
                        id="legacy-crud",
                        source="event_ordering_episode_recall",
                        text="long user request about transaction CRUD implementation details",
                        source_span_ids=["span-crud"],
                        metadata={},
                    ),
                    SimpleNamespace(
                        id="legacy-deploy",
                        source="event_ordering_episode_recall",
                        text="long user request about deployment checks",
                        source_span_ids=["span-deploy"],
                        metadata={},
                    ),
                ]
            ),
            _event_ordering_timeline_candidates=MagicMock(return_value=[]),
            _event_ordering_event_candidates=MagicMock(return_value=[]),
        )

        items, sources = _dual_graph_legacy_items(service, "rank the work", SimpleNamespace(), 3)

        self.assertEqual(
            items,
            [
                "long user request about transaction CRUD implementation details",
                "long user request about initial schema setup and models",
                "long user request about deployment checks",
            ],
        )
        self.assertEqual(
            sources,
            [
                "event_ordering_episode_recall",
                "event_ordering_episode_recall",
                "event_ordering_episode_recall",
            ],
        )

    def test_run_replay_graph_dual_legacy_mode_skips_hybrid_path(self) -> None:
        query = SimpleNamespace(
            id="q1",
            query="rank the work",
            category="event_ordering",
            metadata={"ordering_tested": ["first step", "second step"]},
        )
        args = SimpleNamespace(
            dataset="/unused",
            split="100k",
            workspace="ws",
            user_id="beam_user",
            agent_id="fusion_memory",
            run_id=None,
            session_id=None,
            db="postgresql://example",
            limit=3,
            query_ids=None,
            max_queries=None,
            gate=False,
            mode="graph_dual_legacy",
            hybrid_source="source_spans",
            preflight_min_topics=0,
            preflight_min_nodes=0,
            preflight_min_edges=0,
        )
        service = SimpleNamespace(close=MagicMock())

        with patch.object(replay, "_load_official_beam_dataset", return_value=(None, [query])), patch.object(
            replay, "memory_service_from_env", return_value=service
        ), patch.object(
            replay, "BeamAdapter", return_value=SimpleNamespace(_beam_scope=MagicMock(return_value=SimpleNamespace()))
        ), patch.object(
            replay,
            "preflight_replay_environment_from_store",
            return_value={"status": "ok", "error": None, "chronology_tables_ready": True, "chronology_error": None},
        ), patch.object(
            replay,
            "_graph_items",
            return_value=(["first step"], ["event_ordering_persisted_graph"], False),
        ), patch.object(
            replay,
            "_legacy_items",
            return_value=(["first step", "second step"], ["event_ordering_timeline"]),
        ), patch.object(
            replay,
            "_dual_graph_legacy_items",
            return_value=(["first step", "second step"], ["event_ordering_persisted_graph", "event_ordering_timeline"]),
        ), patch.object(
            replay, "_hybrid_items", side_effect=AssertionError("hybrid path should not run in graph_dual_legacy mode")
        ):
            report = run_replay(args)

        self.assertTrue(report["records"][0]["paths"]["graph"]["active"])
        self.assertTrue(report["records"][0]["paths"]["legacy"]["active"])
        self.assertTrue(report["records"][0]["paths"]["dual"]["active"])
        self.assertFalse(report["records"][0]["paths"]["hybrid"]["active"])
        self.assertEqual(report["records"][0]["paths"]["dual"]["items"], ["first step", "second step"])
        self.assertEqual(report["summary"]["dual"]["count"], 1)
        self.assertEqual(report["summary"]["hybrid"]["count"], 0)

    def test_run_replay_graph_legacy_mode_skips_hybrid_path(self) -> None:
        query = SimpleNamespace(
            id="q1",
            query="rank the work",
            category="event_ordering",
            metadata={"ordering_tested": ["first step", "second step"]},
        )
        args = SimpleNamespace(
            dataset="/unused",
            split="100k",
            workspace="ws",
            user_id="beam_user",
            agent_id="fusion_memory",
            run_id=None,
            session_id=None,
            db="postgresql://example",
            limit=3,
            query_ids=None,
            max_queries=None,
            gate=False,
            mode="graph_legacy",
            hybrid_source="source_spans",
            preflight_min_topics=0,
            preflight_min_nodes=0,
            preflight_min_edges=0,
        )
        service = SimpleNamespace(close=MagicMock())

        with patch.object(replay, "_load_official_beam_dataset", return_value=(None, [query])), patch.object(
            replay, "memory_service_from_env", return_value=service
        ), patch.object(
            replay, "BeamAdapter", return_value=SimpleNamespace(_beam_scope=MagicMock(return_value=SimpleNamespace()))
        ), patch.object(
            replay,
            "preflight_replay_environment_from_store",
            return_value={"status": "ok", "error": None, "chronology_tables_ready": True, "chronology_error": None},
        ), patch.object(
            replay,
            "_graph_items",
            return_value=(["first step"], ["event_ordering_persisted_graph"], False),
        ), patch.object(
            replay,
            "_legacy_items",
            return_value=(["first step", "second step"], ["event_ordering_timeline"]),
        ), patch.object(
            replay, "_hybrid_items", side_effect=AssertionError("hybrid path should not run in graph_legacy mode")
        ):
            report = run_replay(args)

        self.assertTrue(report["records"][0]["paths"]["graph"]["active"])
        self.assertTrue(report["records"][0]["paths"]["legacy"]["active"])
        self.assertFalse(report["records"][0]["paths"]["hybrid"]["active"])
        self.assertEqual(report["records"][0]["paths"]["graph"]["items"], ["first step"])
        self.assertEqual(report["records"][0]["paths"]["legacy"]["items"], ["first step", "second step"])
        self.assertEqual(report["summary"]["hybrid"]["count"], 0)

    def test_run_replay_graph_only_excludes_inactive_paths_from_summaries(self) -> None:
        query = SimpleNamespace(
            id="q1",
            query="rank the work",
            category="event_ordering",
            metadata={"ordering_tested": ["first step", "second step"]},
        )
        args = SimpleNamespace(
            dataset="/unused",
            split="100k",
            workspace="ws",
            user_id="beam_user",
            agent_id="fusion_memory",
            run_id=None,
            session_id=None,
            db="postgresql://example",
            limit=3,
            query_ids=None,
            max_queries=None,
            gate=False,
            mode="graph_only",
            hybrid_source="source_spans",
        )
        service = SimpleNamespace(close=MagicMock())

        with patch.object(replay, "_load_official_beam_dataset", return_value=(None, [query])), patch.object(
            replay, "memory_service_from_env", return_value=service
        ), patch.object(
            replay, "BeamAdapter", return_value=SimpleNamespace(_beam_scope=MagicMock(return_value=SimpleNamespace()))
        ), patch.object(
            replay,
            "preflight_replay_environment_from_store",
            return_value={"status": "ok", "error": None, "chronology_tables_ready": True, "chronology_error": None},
        ), patch.object(
            replay,
            "_graph_items",
            return_value=(["first step", "second step"], ["event_ordering_persisted_graph"], False),
        ), patch.object(
            replay, "_legacy_items", side_effect=AssertionError("legacy path should not run in graph_only mode")
        ), patch.object(
            replay, "_hybrid_items", side_effect=AssertionError("hybrid path should not run in graph_only mode")
        ):
            report = run_replay(args)

        self.assertAlmostEqual(report["summary"]["graph"]["f1"], 1.0)
        self.assertEqual(report["summary"]["legacy"]["count"], 0)
        self.assertEqual(report["summary"]["hybrid"]["count"], 0)
        self.assertEqual(report["bucket_summary"]["legacy"], {})
        self.assertEqual(report["bucket_summary"]["hybrid"], {})
        self.assertEqual(report["summary"]["path_wins"]["f1"], {"graph": 1})
        self.assertEqual(report["summary"]["path_wins"]["kendall_tau_norm"], {"graph": 1})
        self.assertEqual(report["route_summary"], {})
        self.assertEqual(
            report["replay_config"]["artifact_commands"],
            {
                "rule_audit_json": "python3 tools/rule_audit.py --input replay.json --output artifacts/rule-audit.json",
                "rule_audit_csv": "python3 tools/rule_audit.py --input replay.json --output artifacts/rule-audit.json --csv artifacts/rule-audit.csv",
            },
        )
        self.assertTrue(report["records"][0]["paths"]["graph"]["active"])
        self.assertFalse(report["records"][0]["paths"]["legacy"]["active"])
        self.assertFalse(report["records"][0]["paths"]["hybrid"]["active"])
        serialized_report = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("rank the work", serialized_report)
        self.assertNotIn("query", report["records"][0])
        self.assertEqual(report["records"][0]["query_hash"], replay.stable_hash("rank the work"))
        self.assertEqual(report["records"][0]["query_length"], len("rank the work"))

    def test_run_replay_graph_only_gate_fails_with_insufficient_active_paths(self) -> None:
        query = SimpleNamespace(
            id="q1",
            query="rank the work",
            category="event_ordering",
            metadata={"ordering_tested": ["first step"]},
        )
        args = SimpleNamespace(
            dataset="/unused",
            split="100k",
            workspace="ws",
            user_id="beam_user",
            agent_id="fusion_memory",
            run_id=None,
            session_id=None,
            db="postgresql://example",
            limit=3,
            query_ids=None,
            max_queries=None,
            gate=True,
            mode="graph_only",
            hybrid_source="source_spans",
        )
        service = SimpleNamespace(close=MagicMock())

        with patch.object(replay, "_load_official_beam_dataset", return_value=(None, [query])), patch.object(
            replay, "memory_service_from_env", return_value=service
        ), patch.object(
            replay, "BeamAdapter", return_value=SimpleNamespace(_beam_scope=MagicMock(return_value=SimpleNamespace()))
        ), patch.object(
            replay,
            "preflight_replay_environment_from_store",
            return_value={"status": "ok", "error": None, "chronology_tables_ready": True, "chronology_error": None},
        ), patch.object(
            replay,
            "_graph_items",
            return_value=(["first step"], ["event_ordering_persisted_graph"], False),
        ):
            report = run_replay(args)

        self.assertFalse(report["gate"]["passed"])
        self.assertIn("insufficient_active_paths", report["gate"]["failures"])


class BeamEventOrderingGateTests(unittest.TestCase):
    def test_evaluate_gate_requires_graph_to_match_legacy_f1_and_tau(self) -> None:
        summary = {
            "graph": {"f1": 0.10, "kendall_tau_norm": 0.20, "empty_rate": 0.0},
            "legacy": {"f1": 0.20, "kendall_tau_norm": 0.25, "empty_rate": 0.0},
            "dual": {"f1": 0.30, "kendall_tau_norm": 0.30, "empty_rate": 0.0},
            "hybrid": {"f1": 0.18, "kendall_tau_norm": 0.24, "empty_rate": 0.0},
        }

        gate = evaluate_gate(summary)

        self.assertFalse(gate["passed"])
        self.assertIn("graph_f1_below_legacy", gate["failures"])
        self.assertIn("graph_tau_below_legacy", gate["failures"])

    def test_evaluate_gate_requires_dual_to_match_legacy_when_active(self) -> None:
        summary = {
            "graph": {"f1": 0.20, "kendall_tau_norm": 0.30, "empty_rate": 0.0},
            "legacy": {"f1": 0.20, "kendall_tau_norm": 0.30, "empty_rate": 0.0},
            "dual": {"active": True, "count": 1, "f1": 0.10, "kendall_tau_norm": 0.20, "empty_rate": 0.0},
            "hybrid": {"active": False, "count": 0},
        }

        gate = evaluate_gate(summary)

        self.assertFalse(gate["passed"])
        self.assertIn("dual_f1_below_legacy", gate["failures"])
        self.assertIn("dual_tau_below_legacy", gate["failures"])

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
        self.assertFalse(summary["dual_vs_legacy_passed"])
        self.assertIn("graph_f1_below_legacy", summary["gate_failures"])
        self.assertEqual(summary["path_wins"]["f1"], {"graph": 1, "legacy": 1, "hybrid": 0})
        self.assertEqual(summary["path_wins"]["kendall_tau_norm"], {"graph": 1, "legacy": 1, "hybrid": 0})
        self.assertAlmostEqual(summary["graph_fallback_rate"], 0.5)
        self.assertEqual(summary["dropped_high_signal_candidate_count"], 3)
        self.assertEqual(summary["over_abstract_label_count"], 2)

    def test_aggregate_reports_dual_vs_legacy_passed_independently_from_graph(self) -> None:
        records = [
            {
                "paths": {
                    "graph": {
                        "active": True,
                        "items": [],
                        "metrics": {
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "kendall_tau": 0.0,
                            "kendall_tau_norm": 0.0,
                            "system_count": 0,
                            "matched": 0,
                        },
                    },
                    "legacy": {
                        "active": True,
                        "items": ["schema setup"],
                        "metrics": {
                            "precision": 0.5,
                            "recall": 0.5,
                            "f1": 0.5,
                            "kendall_tau": 0.0,
                            "kendall_tau_norm": 0.5,
                            "system_count": 1,
                            "matched": 1,
                        },
                    },
                    "dual": {
                        "active": True,
                        "items": ["schema setup", "crud implementation"],
                        "metrics": {
                            "precision": 0.7,
                            "recall": 0.7,
                            "f1": 0.7,
                            "kendall_tau": 0.2,
                            "kendall_tau_norm": 0.6,
                            "system_count": 2,
                            "matched": 2,
                        },
                    },
                    "hybrid": {"active": False, "items": [], "sources": [], "inactive": True},
                }
            }
        ]

        summary = _aggregate(records)

        self.assertFalse(summary["graph_vs_legacy_passed"])
        self.assertTrue(summary["dual_vs_legacy_passed"])

    def test_aggregate_reports_graph_vs_legacy_passed_independently_from_dual(self) -> None:
        records = [
            {
                "paths": {
                    "graph": {
                        "active": True,
                        "items": ["schema setup", "crud implementation"],
                        "metrics": {
                            "precision": 0.8,
                            "recall": 0.8,
                            "f1": 0.8,
                            "kendall_tau": 0.4,
                            "kendall_tau_norm": 0.7,
                            "system_count": 2,
                            "matched": 2,
                        },
                    },
                    "legacy": {
                        "active": True,
                        "items": ["schema setup"],
                        "metrics": {
                            "precision": 0.5,
                            "recall": 0.5,
                            "f1": 0.5,
                            "kendall_tau": 0.0,
                            "kendall_tau_norm": 0.5,
                            "system_count": 1,
                            "matched": 1,
                        },
                    },
                    "dual": {
                        "active": True,
                        "items": [],
                        "metrics": {
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "kendall_tau": 0.0,
                            "kendall_tau_norm": 0.0,
                            "system_count": 0,
                            "matched": 0,
                        },
                    },
                    "hybrid": {"active": False, "items": [], "sources": [], "inactive": True},
                }
            }
        ]

        summary = _aggregate(records)

        self.assertTrue(summary["graph_vs_legacy_passed"])
        self.assertFalse(summary["dual_vs_legacy_passed"])
        self.assertIn("dual_f1_below_legacy", summary["gate_failures"])

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
