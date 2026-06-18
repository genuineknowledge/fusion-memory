from __future__ import annotations

import hashlib
import unittest

from fusion_memory.retrieval.rule_registry import (
    RuleDefinition,
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
