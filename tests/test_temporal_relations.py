from __future__ import annotations

import unittest

from fusion_memory.retrieval.temporal_relations import (
    safe_temporal_relation_records,
    temporal_relation_summary,
    temporal_relations_for_text,
)


class TemporalRelationTests(unittest.TestCase):
    def test_detects_current_value_supersession_without_raw_text(self) -> None:
        relations = temporal_relations_for_text(
            "I updated the budget from $20 to $35 today.",
            query="what is my current budget?",
            value_text="$35",
            value_type="money",
            source_span_id="span-1",
        )

        relation_types = {relation.relation_type for relation in relations}
        self.assertIn("changed_to", relation_types)
        self.assertIn("supersedes", relation_types)

        records = safe_temporal_relation_records(relations)
        self.assertTrue(records)
        for record in records:
            self.assertNotIn("text", record)
            self.assertNotIn("query", record)
            self.assertNotIn("context", record)

    def test_detects_deadline_and_decision_roles(self) -> None:
        relations = temporal_relations_for_text(
            "We decided on June 3, 2026 and the deployment deadline is July 1, 2026.",
            query="when was the decision and deadline?",
            normalized_date="2026-07-01",
            source_span_id="span-2",
        )

        relation_by_type = {relation.relation_type: relation for relation in relations}

        self.assertIn("deadline", relation_by_type)
        self.assertIn("decision_at", relation_by_type)
        self.assertEqual(relation_by_type["observed_at"].normalized_date, "2026-07-01")
        self.assertIsNone(relation_by_type["deadline"].normalized_date)
        self.assertIsNone(relation_by_type["decision_at"].normalized_date)

    def test_query_only_current_phrasing_does_not_emit_update_relations(self) -> None:
        relations = temporal_relations_for_text(
            "My budget is $35.",
            query="what is my current budget?",
            value_text="$35",
            value_type="money",
            source_span_id="span-4",
        )

        relation_types = {relation.relation_type for relation in relations}
        self.assertNotIn("changed_to", relation_types)
        self.assertNotIn("supersedes", relation_types)

    def test_does_not_infer_range_endpoints_from_previous_marker_and_date_alone(self) -> None:
        relations = temporal_relations_for_text(
            "The previous budget was $20 on June 1, 2026.",
            query="what changed?",
            value_text="$20",
            value_type="money",
            source_span_id="span-5",
        )

        relation_types = {relation.relation_type for relation in relations}
        self.assertNotIn("valid_from", relation_types)
        self.assertNotIn("valid_to", relation_types)

    def test_summary_counts_relation_types_and_sources(self) -> None:
        relations = temporal_relations_for_text(
            "First I set the target to 20, then I changed it to 30.",
            query="what changed?",
            value_text="30",
            value_type="count",
            source_span_id="span-3",
        )

        summary = temporal_relation_summary(relations)

        self.assertGreaterEqual(summary["relation_count"], 1)
        self.assertIn("changed_to", summary["relation_types"])
        self.assertEqual(summary["source_span_count"], 1)
