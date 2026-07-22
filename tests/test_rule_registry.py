from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from fusion_memory.retrieval.rule_audit import build_rule_audit
from fusion_memory.retrieval.rule_registry import (
    RuleDefinition,
    RuleHit,
    collect_rule_hits,
    drain_rule_hits,
    record_rule_hit,
    register_rule,
    registered_rules,
)


class RuleRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        drain_rule_hits()

    def test_record_rule_hit_uses_sha1_prefix_without_raw_text(self) -> None:
        rule = register_rule(
            RuleDefinition(
                rule_id="current_value.stale_history_marker",
                module="fusion_memory.retrieval.evidence_pack",
                purpose="avoid stale current-value evidence",
                category="generic",
                pattern="initially|previously",
            )
        )

        hit = record_rule_hit(
            rule.rule_id,
            query="What is current?",
            text="I initially used SQLite.",
            stage="evidence_pack_filter",
            contributed_candidate_id="span_1",
        )

        self.assertEqual(hit.rule_id, rule.rule_id)
        self.assertEqual(
            hit.text_hash,
            hashlib.sha1("I initially used SQLite.".encode("utf-8")).hexdigest()[:12],
        )
        self.assertNotIn("SQLite", hit.text_hash)
        self.assertNotEqual(hit.query, "What is current?")
        self.assertRegex(hit.query, r"^[0-9a-f]{12}$")
        self.assertEqual(hit.contributed_candidate_id, "span_1")

    def test_drain_rule_hits_returns_and_clears_queue(self) -> None:
        record_rule_hit(
            rule_id="rule.one",
            query="What is current?",
            text="First hit",
            stage="evidence_pack_filter",
        )
        record_rule_hit(
            rule_id="rule.two",
            query="What changed?",
            text="Second hit",
            stage="answer_requirements",
        )

        drained_hits = drain_rule_hits()

        self.assertEqual([hit.rule_id for hit in drained_hits], ["rule.one", "rule.two"])
        self.assertEqual(drain_rule_hits(), [])

    def test_registered_rules_includes_new_definition(self) -> None:
        rule = RuleDefinition(
            rule_id="current_value.current_marker",
            module="fusion_memory.retrieval.current_value",
            purpose="prefer current state evidence",
            category="generic",
            pattern="currently|now",
        )

        registered = register_rule(rule)

        self.assertIn(registered, registered_rules())

    def test_record_rule_hit_copies_metadata_without_storing_raw_text(self) -> None:
        metadata = {
            "confidence": 0.75,
            "details": {"source": "candidate_1"},
            "raw_text": "I initially used SQLite.",
            "decision": "suppress",
            "span_message": "I initially used SQLite.",
            "label": "history-marker",
        }

        hit = record_rule_hit(
            rule_id="current_value.stale_history_marker",
            query="What is current?",
            text="I initially used SQLite.",
            stage="evidence_pack_filter",
            metadata=metadata,
        )

        metadata["confidence"] = 0.10
        metadata["details"] = {"source": "candidate_2"}
        metadata["raw_text"] = "mutated"
        metadata["span_message"] = "mutated"

        self.assertEqual(hit.metadata["confidence"], 0.75)
        self.assertEqual(hit.metadata["details"], {"source": "candidate_1"})
        self.assertEqual(hit.metadata["decision"], "suppress")
        self.assertEqual(hit.metadata["label"], "history-marker")
        self.assertNotEqual(hit.metadata["raw_text"], "I initially used SQLite.")
        self.assertNotEqual(hit.metadata["span_message"], "I initially used SQLite.")
        self.assertRegex(str(hit.metadata["raw_text"]), r"^[0-9a-f]{12}$")
        self.assertRegex(str(hit.metadata["span_message"]), r"^[0-9a-f]{12}$")
        self.assertNotIn("I initially used SQLite.", hit.text_hash)

    def test_record_rule_hit_hashes_raw_text_in_neutral_metadata_keys(self) -> None:
        hit = record_rule_hit(
            rule_id="current_value.neutral_metadata",
            query="What is my preference?",
            text="I prefer PostgreSQL for memory.",
            stage="test",
            metadata={
                "note": "I prefer PostgreSQL for memory.",
                "nested": {"note": "我的默认数据库是 PostgreSQL"},
                "safe": {"decision": "selected", "candidate": "candidate_1"},
            },
        )

        self.assertRegex(str(hit.metadata["note"]), r"^[0-9a-f]{12}$")
        self.assertRegex(str(hit.metadata["nested"]["note"]), r"^[0-9a-f]{12}$")
        self.assertEqual(hit.metadata["safe"], {"decision": "selected", "candidate": "candidate_1"})
        self.assertNotIn("PostgreSQL for memory", str(hit.metadata))
        self.assertNotIn("默认数据库", str(hit.metadata))

    def test_record_rule_hit_hashes_identifier_like_raw_metadata_under_neutral_keys(self) -> None:
        hit = record_rule_hit(
            rule_id="current_value.neutral_metadata",
            query="What is my private token?",
            text="My private token is zinc-sparrow-17.",
            stage="test",
            metadata={
                "note": "zinc-sparrow-17",
                "safe": {"decision": "selected", "source": "l0_raw_hybrid", "category": "current_value"},
                "stage": "search_filter",
            },
        )

        self.assertRegex(str(hit.metadata["note"]), r"^[0-9a-f]{12}$")
        self.assertEqual(hit.metadata["safe"], {"decision": "selected", "source": "l0_raw_hybrid", "category": "current_value"})
        self.assertEqual(hit.metadata["stage"], "search_filter")
        self.assertNotIn("zinc-sparrow-17", repr(hit.metadata))

    def test_record_rule_hit_keeps_raw_and_graph_structural_dimensions(self) -> None:
        hit = record_rule_hit(
            "current_value.stale_history_marker",
            query="user asks current value",
            text="candidate text",
            stage="filter",
            provider_id="raw_span",
            lifecycle_stage="recalled",
            lifecycle_reason="topic_scope_raw",
            metadata={"source_family": "raw", "graph_policy": "graph"},
        )

        self.assertEqual(hit.provider_id, "raw_span")
        self.assertEqual(hit.lifecycle_stage, "recalled")
        self.assertEqual(hit.lifecycle_reason, "topic_scope_raw")
        self.assertEqual(hit.metadata["source_family"], "raw")
        self.assertEqual(hit.metadata["graph_policy"], "graph")

    def test_record_rule_hit_preserves_positional_metadata(self) -> None:
        metadata = {"decision": "drop_stale_history", "source": "candidate_1"}

        hit = record_rule_hit(
            "current_value.stale_history_marker",
            "What is current?",
            "I initially used SQLite.",
            "evidence_pack_filter",
            "span_1",
            metadata,
        )

        self.assertEqual(hit.contributed_candidate_id, "span_1")
        self.assertEqual(hit.metadata, metadata)
        self.assertIsNone(hit.contributed)
        self.assertEqual(hit.impact, "observed")

    def test_rule_hit_positional_constructor_preserves_metadata_and_defaults(self) -> None:
        metadata = {"decision": "selected", "source": "candidate_1"}

        hit = RuleHit(
            "current_value.stale_history_marker",
            "What is current?",
            "deadbeefcafe",
            "span_1",
            "evidence_pack_filter",
            metadata,
        )

        self.assertEqual(hit.metadata, metadata)
        self.assertIsNone(hit.contributed)
        self.assertEqual(hit.impact, "observed")

    def test_record_rule_hit_accepts_keyword_contributed_and_impact(self) -> None:
        hit = record_rule_hit(
            "current_value.stale_history_marker",
            "What is current?",
            "I initially used SQLite.",
            "evidence_pack_filter",
            contributed_candidate_id="span_1",
            metadata={"decision": "selected"},
            contributed=True,
            impact="selected",
        )

        self.assertEqual(hit.contributed_candidate_id, "span_1")
        self.assertTrue(hit.contributed)
        self.assertEqual(hit.impact, "selected")
        self.assertEqual(hit.metadata, {"decision": "selected"})

    def test_rule_definition_declares_protection_and_duplicates(self) -> None:
        protected = RuleDefinition(
            rule_id="current_value.stale_history_marker",
            module="m",
            purpose="drop stale current-value history",
            category="high_risk",
            ability="current_value",
            protected=True,
            protected_reason="high_precision_current_value",
        )
        duplicate = RuleDefinition(
            rule_id="current_value.stale_history_marker.cn_alias",
            module="m",
            purpose="duplicate Chinese alias",
            category="current_value",
            duplicate_of="current_value.stale_history_marker",
        )

        self.assertTrue(protected.protected)
        self.assertEqual(protected.protected_reason, "high_precision_current_value")
        self.assertEqual(duplicate.duplicate_of, "current_value.stale_history_marker")

    def test_record_rule_hit_accepts_sanitized_provider_and_lifecycle_dimensions(self) -> None:
        hit = record_rule_hit(
            "current_value.stale_history_marker",
            query="What is my current database?",
            text="I now use PostgreSQL.",
            stage="evidence_pack_filter",
            provider_id="l3_current_view",
            lifecycle_stage="selected",
            lifecycle_reason="views",
            metadata={"note": "I now use PostgreSQL."},
        )

        self.assertEqual(hit.provider_id, "l3_current_view")
        self.assertEqual(hit.lifecycle_stage, "selected")
        self.assertEqual(hit.lifecycle_reason, "views")
        self.assertRegex(str(hit.metadata["note"]), r"^[0-9a-f]{12}$")
        self.assertNotIn("PostgreSQL", repr(hit.metadata))

    def test_record_rule_hit_hashes_raw_provider_and_lifecycle_dimensions(self) -> None:
        hit = record_rule_hit(
            "current_value.stale_history_marker",
            query="What is my current database?",
            text="I now use PostgreSQL.",
            stage="evidence_pack_filter",
            provider_id="private provider PostgreSQL",
            lifecycle_stage="数据库 selected",
            lifecycle_reason="current database is PostgreSQL",
        )

        self.assertRegex(str(hit.provider_id), r"^[0-9a-f]{12}$")
        self.assertRegex(str(hit.lifecycle_stage), r"^[0-9a-f]{12}$")
        self.assertRegex(str(hit.lifecycle_reason), r"^[0-9a-f]{12}$")
        self.assertNotIn("PostgreSQL", repr(hit))
        self.assertNotIn("数据库", repr(hit))

    def test_collect_rule_hits_isolates_and_clears_on_exception(self) -> None:
        record_rule_hit(
            rule_id="outer.rule",
            query="outer",
            text="outer text",
            stage="setup",
        )

        with self.assertRaises(RuntimeError):
            with collect_rule_hits() as collector:
                record_rule_hit(
                    rule_id="inner.rule",
                    query="inner",
                    text="inner text",
                    stage="search",
                )
                self.assertEqual([hit.rule_id for hit in collector.drain()], ["inner.rule"])
                record_rule_hit(
                    rule_id="inner.leaked",
                    query="inner",
                    text="inner leaked text",
                    stage="search",
                )
                raise RuntimeError("boom")

        self.assertEqual([hit.rule_id for hit in drain_rule_hits()], ["outer.rule"])

    def test_collect_rule_hits_nested_context_restores_parent_collector(self) -> None:
        with collect_rule_hits() as outer:
            record_rule_hit("outer.one", query="", text="outer one", stage="outer")
            with collect_rule_hits() as inner:
                record_rule_hit("inner.one", query="", text="inner one", stage="inner")
                self.assertEqual([hit.rule_id for hit in inner.drain()], ["inner.one"])
            record_rule_hit("outer.two", query="", text="outer two", stage="outer")

            self.assertEqual([hit.rule_id for hit in outer.drain()], ["outer.one", "outer.two"])

        self.assertEqual(drain_rule_hits(), [])

    def test_rule_audit_reports_hits_contributions_and_zero_hit_rules(self) -> None:
        rules = [
            RuleDefinition(
                rule_id="event.order",
                module="m",
                purpose="event order",
                category="event_ordering",
                ability="event_ordering",
            ),
            RuleDefinition(
                rule_id="zh.recall",
                module="m",
                purpose="Chinese recall",
                category="retrieval",
                ability="chinese_recall",
            ),
        ]
        hits = [
            {"rule_id": "event.order", "contributed": True, "impact": "selected"},
            {"rule_id": "event.order", "contributed": False, "impact": "filtered"},
        ]

        audit = build_rule_audit(rules, hits)

        self.assertEqual(audit[0]["rule_id"], "event.order")
        self.assertEqual(audit[0]["hit_count"], 2)
        self.assertEqual(audit[0]["contribution_count"], 1)
        self.assertEqual(audit[0]["negative_impact_count"], 1)
        self.assertEqual(audit[1]["rule_id"], "zh.recall")
        self.assertEqual(audit[1]["hit_count"], 0)

    def test_registered_rule_audit_reports_provider_and_lifecycle_dimensions(self) -> None:
        rules = [
            RuleDefinition(
                rule_id="event.order",
                module="m",
                purpose="event order",
                category="event_ordering",
                ability="event_ordering",
            ),
        ]
        hits = [
            {
                "rule_id": "event.order",
                "provider_id": "views",
                "lifecycle_stage": "selected",
                "lifecycle_reason": "views",
            },
            {
                "rule_id": "event.order",
                "provider_id": "l3_current_view",
                "lifecycle_stage": "selected",
                "lifecycle_reason": "event_ordering_coverage",
            },
            {
                "rule_id": "event.order",
                "provider_id": ["raw provider value"],
                "lifecycle_stage": None,
                "lifecycle_reason": {"raw": "value"},
            },
        ]

        audit = build_rule_audit(rules, hits)
        row = audit[0]

        self.assertEqual(row["provider_ids"], ["l3_current_view", "views"])
        self.assertEqual(row["lifecycle_stages"], ["selected"])
        self.assertEqual(row["lifecycle_reasons"], ["event_ordering_coverage", "views"])

    def test_registered_rule_audit_hashes_raw_provider_and_lifecycle_dimensions(self) -> None:
        rules = [
            RuleDefinition(
                rule_id="event.order",
                module="m",
                purpose="event order",
                category="event_ordering",
                ability="event_ordering",
            ),
        ]
        hits = [
            {
                "rule_id": "event.order",
                "provider_id": "private provider PostgreSQL",
                "lifecycle_stage": "数据库 selected",
                "lifecycle_reason": "source_private_project",
            }
        ]

        audit = build_rule_audit(rules, hits)
        row = audit[0]

        self.assertTrue(all(len(item) == 12 for item in row["provider_ids"]))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_stages"]))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_reasons"]))
        self.assertNotIn("private provider PostgreSQL", repr(row))
        self.assertNotIn("数据库 selected", repr(row))
        self.assertNotIn("source_private_project", repr(row))

    def test_registered_rule_audit_marks_zero_hit_rules_for_first_pass_cleanup(self) -> None:
        rules = [
            RuleDefinition(
                rule_id="event.order",
                module="m",
                purpose="event order",
                category="event_ordering",
                ability="event_ordering",
            ),
            RuleDefinition(
                rule_id="zh.recall",
                module="m",
                purpose="Chinese recall",
                category="retrieval",
                ability="chinese_recall",
            ),
        ]
        hits = [{"rule_id": "event.order", "contributed": True, "impact": "selected"}]

        audit = build_rule_audit(rules, hits)
        zero_hit = next(row for row in audit if row["rule_id"] == "zh.recall")

        self.assertIsNone(zero_hit["duplicate_of"])
        self.assertEqual(zero_hit["provider_ids"], [])
        self.assertEqual(zero_hit["lifecycle_stages"], [])
        self.assertEqual(zero_hit["lifecycle_reasons"], [])
        self.assertEqual(zero_hit["cleanup_phase"], "first_pass")
        self.assertEqual(zero_hit["cleanup_action"], "delete_no_hits")
        self.assertTrue(zero_hit["safe_to_delete"])

    def test_registered_rule_audit_keeps_observation_only_rules(self) -> None:
        rules = [
            RuleDefinition(
                rule_id="multi_condition.query_token_match",
                module="fusion_memory.api.service_helpers",
                purpose="observe multi-condition matching",
                category="multi_condition",
                ability="multi_condition",
            ),
            RuleDefinition(
                rule_id="zh_recall.cjk_exact_match",
                module="fusion_memory.api.service_helpers",
                purpose="observe CJK exact matching",
                category="zh_recall",
                ability="zh_recall",
            ),
            RuleDefinition(
                rule_id="taxonomy.alias_match",
                module="fusion_memory.api.service_helpers",
                purpose="observe taxonomy alias matching",
                category="taxonomy_candidate",
                ability="zh_recall",
            ),
        ]
        hits = [
            {"rule_id": "multi_condition.query_token_match", "impact": "observed"},
            {"rule_id": "zh_recall.cjk_exact_match", "impact": "observed"},
            {"rule_id": "taxonomy.alias_match", "impact": "observed"},
        ]

        audit = build_rule_audit(rules, hits)

        for row in audit:
            self.assertEqual(row["cleanup_action"], "keep_observation")
            self.assertEqual(row["cleanup_blockers"], ["observation_only_rule"])
            self.assertFalse(row["safe_to_delete"])

    def test_registered_rule_audit_keeps_zero_hit_legacy_shadow_rules(self) -> None:
        rules = [
            RuleDefinition(
                rule_id="event_ordering.legacy.tie_breaker",
                module="m",
                purpose="legacy event ordering shadow",
                category="event_ordering",
                ability="event_ordering",
            )
        ]

        audit = build_rule_audit(rules, [])
        legacy = audit[0]

        self.assertFalse(legacy["candidate_for_deletion"])
        self.assertEqual(legacy["cleanup_action"], "keep_shadow")
        self.assertFalse(legacy["safe_to_delete"])

    def test_registered_rule_audit_keeps_protected_zero_hit_rules(self) -> None:
        rules = [
            RuleDefinition(
                rule_id="current_value.stale_history_marker",
                module="m",
                purpose="drop stale current-value history",
                category="high_risk",
                ability="current_value",
                protected=True,
                protected_reason="high_precision_current_value",
            ),
            RuleDefinition(
                rule_id="current_value.stale_history_marker.cn_alias",
                module="m",
                purpose="duplicate Chinese alias",
                category="current_value",
                duplicate_of="current_value.stale_history_marker",
            ),
        ]

        audit = build_rule_audit(rules, [])
        protected = next(row for row in audit if row["rule_id"] == "current_value.stale_history_marker")
        duplicate = next(row for row in audit if row["rule_id"] == "current_value.stale_history_marker.cn_alias")

        self.assertTrue(protected["protected"])
        self.assertEqual(protected["protected_reason"], "high_precision_current_value")
        self.assertIsNone(protected["duplicate_of"])
        self.assertFalse(protected["candidate_for_deletion"])
        self.assertEqual(protected["cleanup_phase"], "")
        self.assertEqual(protected["cleanup_action"], "keep_protected")
        self.assertFalse(protected["safe_to_delete"])
        self.assertEqual(duplicate["duplicate_of"], "current_value.stale_history_marker")
        self.assertEqual(duplicate["cleanup_action"], "delete_duplicate")
        self.assertTrue(duplicate["safe_to_delete"])


class RuleInstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        drain_rule_hits()

    def test_taxonomy_alias_match_emits_observation_only_rule_hit_without_service_import_side_effects(self) -> None:
        script = """
import importlib
import json
import pathlib
import types
import sys

repo_root = pathlib.Path.cwd()
package = types.ModuleType("fusion_memory")
package.__path__ = [str(repo_root / "fusion_memory")]
sys.modules["fusion_memory"] = package

from fusion_memory.retrieval.rule_registry import collect_rule_hits
from fusion_memory.core.models import MemoryEvent, Scope

event_graph_selection = importlib.import_module("fusion_memory.retrieval.event_graph_selection")

with collect_rule_hits() as collector:
    score = event_graph_selection._event_ordering_event_relevance(
        "walk me through the deployment work",
        MemoryEvent(
            event_id="evt-1",
            scope=Scope(workspace_id="ws"),
            event_type="milestone",
            description="Configured Render deployment and Gunicorn server settings.",
            participants=[],
            source_span_ids=[],
        ),
    )
    hits = [hit.__dict__ for hit in collector.drain()]

print(json.dumps({"score": score, "hits": hits}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        payload = json.loads(result.stdout)

        self.assertGreater(payload["score"], 0.0)
        taxonomy_hits = [hit for hit in payload["hits"] if hit["rule_id"] == "taxonomy.alias_match"]
        self.assertEqual(len(taxonomy_hits), 1)
        taxonomy_hit = taxonomy_hits[0]
        self.assertRegex(taxonomy_hit["query"], r"^[0-9a-f]{12}$")
        self.assertNotEqual(taxonomy_hit["query"], "")
        self.assertEqual(taxonomy_hit["impact"], "observed")
        self.assertEqual(taxonomy_hit["metadata"]["decision"], "observed")
        self.assertEqual(taxonomy_hit["metadata"]["source"], "taxonomy")
        self.assertFalse(any("deployment" in json.dumps(hit, ensure_ascii=False) for hit in payload["hits"]))
        self.assertFalse(any("Render" in json.dumps(hit, ensure_ascii=False) for hit in payload["hits"]))
