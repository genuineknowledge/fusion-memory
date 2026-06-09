from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.ingestion.temporal_normalizer import TemporalNormalizer
from fusion_memory.retrieval.query_planner import QueryPlanner


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

    def test_query_planner_extracts_new_time_constraints(self) -> None:
        plan = QueryPlanner().plan("What did we deploy next month and this Friday?")
        self.assertEqual(plan.query_type, "temporal_lookup")
        constraints = {item["text"] for item in plan.time_constraints}
        self.assertIn("next month", constraints)
        self.assertIn("this friday", constraints)

    def test_service_events_use_extended_temporal_rules(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I fixed reports this Friday.", scope, ts("2026-06-09T15:30:00+00:00"))
        memory.add("I deployed Atlas on June 16, 2026.", scope, ts("2026-06-09T15:30:00+00:00"))
        memory.add("I started rollout next month.", scope, ts("2026-06-09T15:30:00+00:00"))
        memory.add("I deployed Atlas.", scope, ts("2026-06-09T15:30:00+00:00"))

        events = memory.store.list_events(scope)
        by_text = {event.description: event for event in events}
        self.assertEqual(by_text["I fixed reports this Friday."].time_start.date().isoformat(), "2026-06-12")
        self.assertEqual(by_text["I deployed Atlas on June 16, 2026."].time_start.date().isoformat(), "2026-06-16")
        self.assertEqual(by_text["I started rollout next month."].time_start.date().isoformat(), "2026-07-01")
        self.assertIsNone(by_text["I deployed Atlas."].time_start)
        self.assertEqual(by_text["I deployed Atlas."].time_source, "unknown")

    def test_explicit_after_statement_writes_event_edge(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I tested BM25 yesterday.", scope, ts("2026-06-03T12:00:00+00:00"))
        memory.add("After the BM25 test, I added dense retrieval.", scope, ts("2026-06-05T12:00:00+00:00"))

        events = memory.store.list_events(scope)
        bm25 = next(event for event in events if "BM25" in event.description and "After" not in event.description)
        dense = next(event for event in events if "dense retrieval" in event.description)
        comparison = memory.compare_events(bm25.event_id, dense.event_id)
        self.assertEqual(comparison["relation"], "before")
        self.assertEqual(comparison["basis"], "event_edge")
        self.assertGreaterEqual(comparison["confidence"], 0.82)


if __name__ == "__main__":
    unittest.main()
