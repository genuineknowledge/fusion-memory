from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.beam_retrieval_replay as replay
from tools.rule_audit import build_provider_audit, build_rule_audit


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

    def test_run_replay_prefers_coverage_pipeline_trace_from_answer_context(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={
                "coverage_insufficient": False,
                "pipeline_trace": {
                    "query_type": "current_value",
                    "mode": "benchmark",
                    "pipeline_layers": {
                        "CandidateRecall": {"source_counts": {"l0_raw": 2}},
                        "CandidateFusion": {"selected_sources": ["l0_raw"], "dropped_count": 0},
                        "EvidencePackBuilder": {"source_span_count": 1, "coverage_insufficient": False},
                    },
                },
            },
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["records"][0]["pipeline_trace"],
            [
                {
                    "layer": "retrieval",
                    "query_type": "current_value",
                    "mode": "benchmark",
                    "source_counts": {"l0_raw": 2},
                    "selected_sources": [{"source": "l0_raw"}],
                    "source_span_count": 1,
                    "coverage_insufficient": False,
                }
            ],
        )

    def test_run_replay_preserves_provider_summary_for_provider_audit(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={
                "coverage_insufficient": False,
                "pipeline_trace": {
                    "query_type": "current_value",
                    "mode": "benchmark",
                    "pipeline_layers": {
                        "CandidateRecall": {
                            "source_counts": {"l0_raw_hybrid": 2},
                            "provider_summary": [
                                {
                                    "provider_id": "raw_provider",
                                    "source_family": "raw_provider",
                                    "output_count": 2,
                                    "output_source_counts": {"l0_raw_hybrid": 2},
                                    "production_default": True,
                                    "shadow_only": False,
                                    "graph_related": False,
                                }
                            ],
                        },
                        "CandidateFusion": {"selected_sources": ["l0_raw_hybrid"], "dropped_count": 0},
                        "EvidencePackBuilder": {"source_span_count": 1, "coverage_insufficient": False},
                    },
                },
            },
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        trace = payload["records"][0]["pipeline_trace"]
        provider_summary = trace[0]["pipeline_layers"]["CandidateRecall"]["provider_summary"]
        audit = build_provider_audit(payload["records"])

        self.assertEqual(trace[0]["source_counts"], {"l0_raw_hybrid": 2})
        self.assertEqual(
            provider_summary,
            [
                {
                    "provider_id": "raw_provider",
                    "source_family": "raw_provider",
                    "output_count": 2,
                    "output_source_counts": {"l0_raw_hybrid": 2},
                    "production_default": True,
                    "shadow_only": False,
                    "graph_related": False,
                }
            ],
        )
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["provider_id"], "raw_provider")
        self.assertEqual(audit[0]["output_count"], 2)
        self.assertEqual(audit[0]["output_source_counts"], {"l0_raw_hybrid": 2})

    def test_run_replay_hashes_unsafe_provider_summary_dimensions_without_raw_text(self) -> None:
        secret = "zinc-sparrow-17"
        fake_query = SimpleNamespace(id="q1", query=f"What mentions {secret}?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={
                "coverage_insufficient": False,
                "pipeline_trace": {
                    "query_type": "current_value",
                    "mode": "benchmark",
                    "pipeline_layers": {
                        "CandidateRecall": {
                            "source_counts": {"l0_raw_hybrid": 2},
                            "provider_summary": [
                                {
                                    "provider_id": f"private provider {secret}",
                                    "source_family": f"private family {secret}",
                                    "output_count": float("inf"),
                                    "output_source_counts": {f"candidate text {secret}": "invalid"},
                                    "production_default": True,
                                    "shadow_only": False,
                                    "graph_related": False,
                                    "sample_text": f"raw candidate text {secret}",
                                    "raw_query": f"What mentions {secret}?",
                                }
                            ],
                        },
                        "CandidateFusion": {"selected_sources": ["l0_raw_hybrid"], "dropped_count": 0},
                        "EvidencePackBuilder": {"source_span_count": 1, "coverage_insufficient": False},
                    },
                },
            },
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        provider_summary = payload["records"][0]["pipeline_trace"][0]["pipeline_layers"]["CandidateRecall"]["provider_summary"]
        provider_record = provider_summary[0]
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)

        self.assertEqual(len(provider_record["provider_id"]), 12)
        self.assertEqual(len(provider_record["source_family"]), 12)
        self.assertEqual(provider_record["output_count"], 0)
        self.assertEqual(list(provider_record["output_source_counts"].values()), [0])
        self.assertTrue(all(len(source) == 12 for source in provider_record["output_source_counts"]))
        self.assertNotIn(secret, serialized)
        self.assertNotIn("private provider", serialized)
        self.assertNotIn("private family", serialized)
        self.assertNotIn("candidate text", serialized)
        self.assertNotIn("raw candidate text", serialized)
        self.assertNotIn("sample_text", serialized)
        self.assertNotIn("raw_query", serialized)

    def test_run_replay_sanitizes_pipeline_trace_before_writing(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={"coverage_insufficient": True},
            debug_trace=[
                {
                    "layer": "retrieval",
                    "query_type": "current_value",
                    "mode": "benchmark",
                    "query": "What is my current IDE?",
                    "content": "I use VS Code.",
                    "source_counts": {"candidate": 3, "selected": 1},
                    "selected_sources": [
                        {
                            "source_id": "s1",
                            "selected_text": "selected raw source text",
                            "content": "selected raw content",
                        }
                    ],
                    "source_span_count": 1,
                    "coverage_insufficient": True,
                    "rule_hits": [
                        {"rule_id": "current_value.keep_latest", "content": "candidate raw text"},
                        {"rule_id": "current_value.prefer_recent"},
                    ],
                    "prompt": {"message": "raw prompt text"},
                }
            ],
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

        trace = payload["records"][0]["pipeline_trace"]
        self.assertTrue(trace)
        self.assertNotIn("What is my current IDE?", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("I use VS Code.", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("selected raw source text", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("selected raw content", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("candidate raw text", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("raw prompt text", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(
            trace,
            [
                {
                    "layer": "retrieval",
                    "query_type": "current_value",
                    "mode": "benchmark",
                    "source_counts": {"candidate": 3, "selected": 1},
                    "selected_sources": [{"source_id": "s1"}],
                    "source_span_count": 1,
                    "coverage_insufficient": True,
                    "rule_hit_count": 2,
                }
            ],
        )
        self.assertEqual(report["records"][0]["pipeline_trace"], trace)

    def test_run_replay_keeps_structural_selected_trace_without_text(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="中文原始查询不应落盘", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={"coverage_insufficient": False},
            debug_trace=[
                {
                    "id": "cand_1",
                    "type": "span",
                    "source": "l0_raw",
                    "scores": {"utility_score": 0.75},
                    "source_span_ids": ["span_1"],
                    "text": "中文原始候选内容不应落盘",
                }
            ],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        trace = payload["records"][0]["pipeline_trace"]
        self.assertEqual(
            trace,
            [
                {
                    "id": "cand_1",
                    "type": "span",
                    "source": "l0_raw",
                    "scores": {"utility_score": 0.75},
                    "source_span_ids": ["span_1"],
                }
            ],
        )
        self.assertNotIn("中文原始查询不应落盘", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("中文原始候选内容不应落盘", json.dumps(payload, ensure_ascii=False))

    def test_run_replay_does_not_persist_raw_query_text(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[],
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

        self.assertNotIn("query", report["records"][0])
        self.assertNotIn("query", payload["records"][0])
        self.assertEqual(service.answer_context.call_args.args[0], "What is my current IDE?")

    def test_run_replay_writes_sanitized_candidate_lifecycle(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my private token?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={
                "coverage_insufficient": False,
                "candidate_lifecycle": {
                    "record_count": 2,
                    "stage_counts": {"recalled": 1, "selected": 1},
                    "source_counts": {"l0_raw_hybrid": 2},
                    "reason_counts": {"candidate_provider": 1, "final_selection": 1},
                    "raw_text": "do not persist",
                },
            },
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        lifecycle = payload["records"][0]["candidate_lifecycle"]
        self.assertEqual(lifecycle["stage_counts"]["recalled"], 1)
        self.assertEqual(lifecycle["source_counts"]["l0_raw_hybrid"], 2)
        self.assertNotIn("raw_text", lifecycle)
        self.assertNotIn("do not persist", json.dumps(payload, ensure_ascii=False))

    def test_run_replay_preserves_safe_temporal_relation_telemetry(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="When is the deployment due?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={
                "coverage_insufficient": False,
                "temporal_relation_summary": {
                    "relation_count": 1,
                    "relation_types": ["deadline"],
                    "source_span_count": 1,
                    "query": "When is the deployment due?",
                },
                "candidate_lifecycle": {
                    "record_count": 1,
                    "records": [
                        {
                            "candidate_id": "candidate-1",
                            "candidate_source": "l3_current_view",
                            "candidate_type": "span",
                            "stage": "selected",
                            "reason_code": "views",
                            "source_span_ids": ["span_1"],
                            "temporal_relations": [
                                {
                                    "relation_type": "deadline",
                                    "confidence": 0.82,
                                    "reason_code": "deadline_marker",
                                    "source_span_id": "span_1",
                                    "value_type": "date",
                                    "normalized_date": "2026-07-01",
                                    "text": "Deployment deadline is July 1, 2026.",
                                    "query": "When is the deployment due?",
                                    "context": "raw context",
                                    "prompt": "raw prompt",
                                    "content": "raw content",
                                }
                            ],
                        }
                    ],
                },
            },
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

        record = payload["records"][0]
        self.assertEqual(
            record["temporal_relation_summary"],
            {
                "relation_count": 1,
                "relation_types": ["deadline"],
                "source_span_count": 1,
            },
        )
        self.assertEqual(
            record["candidate_lifecycle"]["records"][0]["temporal_relations"],
            [
                {
                    "relation_type": "deadline",
                    "confidence": 0.82,
                    "reason_code": "deadline_marker",
                    "source_span_id": "span_1",
                    "value_type": "date",
                    "normalized_date": "2026-07-01",
                }
            ],
        )
        payload_text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("When is the deployment due?", payload_text)
        self.assertNotIn("Deployment deadline is July 1, 2026.", payload_text)
        self.assertNotIn("raw context", payload_text)
        self.assertNotIn("raw prompt", payload_text)
        self.assertNotIn("raw content", payload_text)
        self.assertEqual(report["records"][0]["temporal_relation_summary"], record["temporal_relation_summary"])

    def test_run_replay_derives_temporal_relation_summary_from_pipeline_trace_layer(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What happened before the launch?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "span-graph-trace"}],
            coverage={
                "coverage_insufficient": False,
                "pipeline_trace": {
                    "query_type": "event_ordering",
                    "mode": "benchmark",
                    "pipeline_layers": {
                        "CandidateRecall": {"source_counts": {"l2_event_graph": 1}},
                        "CandidateFusion": {"selected_sources": ["l2_event_graph"]},
                        "EvidencePackBuilder": {
                            "source_span_count": 1,
                            "coverage_insufficient": False,
                        },
                        "TemporalRelations": {
                            "relation_count": 1,
                            "relation_types": ["before"],
                            "role_labels": ["earlier_event"],
                            "reason_codes": ["explicit_order_marker"],
                            "source_span_count": 1,
                            "source_span_ids": ["span-graph-trace"],
                            "query": "What happened before the launch?",
                            "content": "The migration happened before the launch.",
                        },
                    },
                },
            },
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

        record = payload["records"][0]
        self.assertEqual(
            record["temporal_relation_summary"],
            {
                "relation_count": 1,
                "relation_types": ["before"],
                "role_labels": ["earlier_event"],
                "reason_codes": ["explicit_order_marker"],
                "source_span_count": 1,
                "source_span_ids": ["span-graph-trace"],
            },
        )
        self.assertEqual(
            record["pipeline_trace"],
            [
                {
                    "layer": "retrieval",
                    "query_type": "event_ordering",
                    "mode": "benchmark",
                    "source_counts": {"l2_event_graph": 1},
                    "selected_sources": [{"source": "l2_event_graph"}],
                    "source_span_count": 1,
                    "coverage_insufficient": False,
                    "temporal_relation_summary": {
                        "relation_count": 1,
                        "relation_types": ["before"],
                        "role_labels": ["earlier_event"],
                        "reason_codes": ["explicit_order_marker"],
                        "source_span_count": 1,
                        "source_span_ids": ["span-graph-trace"],
                    },
                }
            ],
        )
        payload_text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("What happened before the launch?", payload_text)
        self.assertNotIn("The migration happened before the launch.", payload_text)
        self.assertEqual(report["records"][0]["temporal_relation_summary"], record["temporal_relation_summary"])

    def test_run_replay_output_preserves_rule_audit_dimensions(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={
                "coverage_insufficient": False,
                "candidate_lifecycle": {
                    "record_count": 1,
                    "records": [
                        {
                            "candidate_id": "candidate-1",
                            "candidate_source": "l3_current_view",
                            "candidate_type": "span",
                            "stage": "selected",
                            "reason_code": "views",
                            "source_span_ids": ["span_1"],
                            "scores": {"utility_score": 0.9},
                            "contributed": True,
                            "text": "raw candidate text must not persist",
                        }
                    ],
                },
                "rule_hits": [
                    {
                        "rule_id": "rule.audit_dimensions",
                        "contributed_candidate_id": "candidate-1",
                        "provider_id": "views",
                        "lifecycle_stage": "selected",
                        "lifecycle_reason": "views",
                        "protected": True,
                        "protected_reason": "high_precision_current_value",
                        "impact": "selected",
                        "text": "raw hit text must not persist",
                    }
                ],
            },
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        record = payload["records"][0]
        hit = record["rule_hits"][0]
        lifecycle_record = record["candidate_lifecycle"]["records"][0]
        audit = build_rule_audit(payload["records"])
        row = next(item for item in audit if item["rule_id"] == "rule.audit_dimensions")

        self.assertEqual(hit["provider_id"], "views")
        self.assertEqual(hit["lifecycle_stage"], "selected")
        self.assertEqual(hit["lifecycle_reason"], "views")
        self.assertTrue(hit["protected"])
        self.assertEqual(hit["protected_reason"], "high_precision_current_value")
        self.assertEqual(lifecycle_record["candidate_id"], "candidate-1")
        self.assertEqual(lifecycle_record["stage"], "selected")
        self.assertEqual(lifecycle_record["reason_code"], "views")
        self.assertEqual(row["provider_ids"], ["views"])
        self.assertEqual(row["lifecycle_stages"], ["selected"])
        self.assertEqual(row["lifecycle_reasons"], ["views"])
        self.assertTrue(row["protected"])
        self.assertEqual(row["protected_reason"], "high_precision_current_value")
        self.assertFalse(row["safe_to_delete"])
        self.assertNotIn("raw candidate text", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("raw hit text", json.dumps(payload, ensure_ascii=False))

    def test_run_replay_sanitizes_rule_hits_before_writing(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        raw_rule_hits = [
            {
                "rule_id": "current_value.keep_latest",
                "query": "What is my current IDE?",
                "text_hash": "abc123",
                "contributed_candidate_id": "candidate-1",
                "stage": "filter",
                "contributed": True,
                "impact": "selected",
                "metadata": {
                    "decision": "kept",
                    "query": "What is my current IDE?",
                    "raw_text": "I use VS Code.",
                    "message_content": "assistant prompt text",
                    "source_span": {"text": "span content"},
                    "safe": {"category": "current_value", "count": 1},
                    "neutral": {"note": "I use VS Code"},
                    "text_hash": "def456",
                },
            }
        ]
        fake_pack = SimpleNamespace(
            source_spans=[],
            coverage={"coverage_insufficient": False, "rule_hits": raw_rule_hits},
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

        hit = payload["records"][0]["rule_hits"][0]
        self.assertEqual(hit["rule_id"], "current_value.keep_latest")
        self.assertEqual(hit["text_hash"], "abc123")
        self.assertEqual(hit["contributed_candidate_id"], "candidate-1")
        self.assertEqual(hit["stage"], "filter")
        self.assertTrue(hit["contributed"])
        self.assertEqual(hit["impact"], "selected")
        self.assertNotIn("query", hit)
        self.assertEqual(
            hit["metadata"],
            {
                "decision": "kept",
                "safe": {"category": "current_value", "count": 1},
                "neutral": {"note": {"hash": "0a39a96ea3d1"}},
                "text_hash": "def456",
            },
        )
        self.assertNotIn("What is my current IDE?", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("I use VS Code.", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(report["records"][0]["rule_hits"], payload["records"][0]["rule_hits"])

    def test_run_replay_hashes_identifier_like_raw_rule_metadata(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my private token?", category="knowledge_update")
        raw_rule_hits = [
            {
                "rule_id": "current_value.keep_latest",
                "text_hash": "abc123",
                "contributed_candidate_id": "candidate-1",
                "stage": "filter",
                "metadata": {
                    "note": "zinc-sparrow-17",
                    "safe": {"decision": "kept", "source": "l0_raw_hybrid", "category": "current_value"},
                    "text_hash": "def456",
                },
            }
        ]
        fake_pack = SimpleNamespace(
            source_spans=[],
            coverage={"coverage_insufficient": False, "rule_hits": raw_rule_hits},
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))

        metadata = payload["records"][0]["rule_hits"][0]["metadata"]
        self.assertEqual(metadata["note"], {"hash": "8fed895e0dca"})
        self.assertEqual(metadata["safe"], {"decision": "kept", "source": "l0_raw_hybrid", "category": "current_value"})
        self.assertEqual(metadata["text_hash"], "def456")
        self.assertNotIn("zinc-sparrow-17", json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
