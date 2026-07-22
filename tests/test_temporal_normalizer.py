from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.core.text import extract_entities
from fusion_memory.ingestion.temporal_normalizer import TemporalNormalizer
from fusion_memory.retrieval.context import OrderingMode, ProviderKind, SearchRequest
from fusion_memory.retrieval.query_planner import ProductQueryPlanner


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class TemporalNormalizerTests(unittest.TestCase):
    def test_relative_days_weeks_months_and_weekdays(self) -> None:
        normalizer = TemporalNormalizer()
        session_time = ts("2026-06-09T15:30:00+00:00")

        tomorrow = normalizer.normalize("deploy tomorrow", session_time)
        self.assertEqual(tomorrow.time_start.date().isoformat(), "2026-06-10")
        self.assertEqual(tomorrow.source, "relative_resolved")

        next_month = normalizer.normalize("start rollout next month", session_time)
        self.assertEqual(next_month.time_start.date().isoformat(), "2026-07-01")
        self.assertEqual(next_month.time_end.date().isoformat(), "2026-08-01")
        self.assertEqual(next_month.granularity, "month")

        this_friday = normalizer.normalize("fixed reports this Friday", session_time)
        self.assertEqual(this_friday.time_start.date().isoformat(), "2026-06-12")

    def test_explicit_dates_and_unknown_fallback(self) -> None:
        normalizer = TemporalNormalizer()
        session_time = ts("2026-06-09T15:30:00+00:00")

        iso_date = normalizer.normalize("deployed on 2026-06-15", session_time)
        self.assertEqual(iso_date.time_start.date().isoformat(), "2026-06-15")
        self.assertEqual(iso_date.source, "explicit")

        month_name = normalizer.normalize("deployed on June 16, 2026", session_time)
        self.assertEqual(month_name.time_start.date().isoformat(), "2026-06-16")

        invalid = normalizer.normalize("deployed on June 31, 2026", session_time)
        self.assertIsNone(invalid.time_start)
        self.assertEqual(invalid.source, "unknown")

        unknown = normalizer.normalize("deployed Atlas", session_time)
        self.assertIsNone(unknown.time_start)
        self.assertEqual(unknown.granularity, "unknown")
        self.assertEqual(unknown.source, "unknown")

    def test_product_planner_exposes_temporal_and_duration_intent(self) -> None:
        planner = ProductQueryPlanner()
        temporal = planner.plan(
            SearchRequest("What did we deploy this Friday?", 6)
        )
        duration = planner.plan(
            SearchRequest(
                "How many weeks are between starting the feature and the final deployment deadline?",
                6,
            )
        )

        self.assertEqual(temporal.intent, "temporal")
        self.assertTrue(temporal.query_intent["temporal"]["requires_time"])
        self.assertIn("friday", temporal.query_intent["temporal"]["time_expressions"])
        self.assertEqual(duration.query_intent["answer_shape"], "duration")
        self.assertTrue(duration.query_intent["temporal"]["requires_duration"])
        self.assertEqual(
            duration.query_intent["temporal"]["endpoint_roles"],
            ["start", "end", "deadline"],
        )
        self.assertEqual(duration.query_intent["aggregation"]["operation"], "none")

    def test_product_planner_routes_explicit_memory_ordering(self) -> None:
        plan = ProductQueryPlanner().plan(
            SearchRequest(
                "Can you list the order in which I brought up Flask, SQLite, and Bootstrap?",
                6,
            )
        )

        self.assertEqual(plan.intent, "chronology")
        self.assertEqual(plan.ordering, OrderingMode.CHRONOLOGICAL)
        self.assertEqual(plan.speaker, "user")
        self.assertIn(
            ProviderKind.CHRONOLOGY,
            {request.kind for request in plan.provider_requests},
        )
        self.assertTrue(plan.query_intent["temporal"]["requires_order"])

    def test_product_planner_rejects_procedural_and_social_false_temporal_routes(self) -> None:
        planner = ProductQueryPlanner()
        procedural = planner.plan(
            SearchRequest(
                "If I draw a card and then draw another, how do I calculate both chances?",
                6,
            )
        )
        social = planner.plan(
            SearchRequest(
                "What are common expectations when meeting someone for the first time?",
                6,
            )
        )

        self.assertFalse(procedural.query_intent["temporal"]["requires_order"])
        self.assertNotEqual(procedural.ordering, OrderingMode.CHRONOLOGICAL)
        self.assertFalse(social.query_intent["temporal"]["requires_time"])

    def test_product_planner_exposes_multilingual_query_intent(self) -> None:
        plan = ProductQueryPlanner().plan(
            SearchRequest("请按顺序列出我在所有对话中提到的不同安全功能。", 6)
        )

        self.assertEqual(plan.query_intent["language"], "zh")
        self.assertEqual(plan.query_intent["answer_shape"], "ordered_list")
        self.assertEqual(plan.query_intent["evidence_scope"], "multi_session")
        self.assertIn("security_feature", plan.query_intent["object_types"])

    def test_extract_entities_filters_question_boilerplate(self) -> None:
        entities = extract_entities(
            "Can you list the order in which I brought up Flask, SQLite, and Bootstrap? Mention ONLY three items."
        )
        self.assertIn("Flask", entities)
        self.assertIn("SQLite", entities)
        self.assertIn("Bootstrap", entities)
        self.assertNotIn("Can", entities)
        self.assertNotIn("Mention", entities)
        self.assertNotIn("ONLY", entities)

    def test_service_events_use_extended_temporal_rules(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        try:
            memory.add("I fixed reports this Friday.", scope, ts("2026-06-09T15:30:00+00:00"))
            memory.add("I deployed Atlas on June 16, 2026.", scope, ts("2026-06-09T15:30:00+00:00"))
            memory.add("I started rollout next month.", scope, ts("2026-06-09T15:30:00+00:00"))
            memory.add("I deployed Atlas.", scope, ts("2026-06-09T15:30:00+00:00"))

            events = memory.store.list_events(scope)
            friday = next(event for event in events if "I fixed reports this Friday." in event.description)
            june_16 = next(event for event in events if "I deployed Atlas on June 16, 2026." in event.description)
            rollout = next(event for event in events if "I started rollout next month." in event.description)
            atlas = next(event for event in events if event.description.endswith("I deployed Atlas."))
            self.assertEqual(friday.time_start.date().isoformat(), "2026-06-12")
            self.assertEqual(june_16.time_start.date().isoformat(), "2026-06-16")
            self.assertEqual(rollout.time_start.date().isoformat(), "2026-07-01")
            self.assertIsNone(atlas.time_start)
            self.assertEqual(atlas.time_source, "unknown")
        finally:
            memory.close()

    def test_explicit_after_statement_writes_event_edge(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        try:
            memory.add("I tested BM25 yesterday.", scope, ts("2026-06-03T12:00:00+00:00"))
            memory.add(
                "After the BM25 test, I added dense retrieval.",
                scope,
                ts("2026-06-05T12:00:00+00:00"),
            )

            events = memory.store.list_events(scope)
            bm25 = next(event for event in events if "BM25" in event.description and "After" not in event.description)
            dense = next(event for event in events if "dense retrieval" in event.description)
            comparison = memory.compare_events(bm25.event_id, dense.event_id)
            self.assertEqual(comparison["relation"], "before")
            self.assertEqual(comparison["basis"], "event_edge")
            self.assertGreaterEqual(comparison["confidence"], 0.82)
        finally:
            memory.close()


if __name__ == "__main__":
    unittest.main()
