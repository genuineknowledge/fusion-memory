from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fusion_memory import MemoryService
from fusion_memory.core.chronology import ChronologyEventNode, ChronologyPhase, ChronologyTopic
from fusion_memory.core.models import Candidate
from fusion_memory.core.models import Scope
from fusion_memory.retrieval.chronology_selector import select_persisted_graph_event_ordering_candidates
from fusion_memory.retrieval.taxonomy import load_default_taxonomy, taxonomy_alias_hits


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class ChronologySelectorTests(unittest.TestCase):
    def test_persisted_graph_selector_returns_topic_scoped_ordered_candidates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-select", user_id="u", agent_id="a", session_id="s")
        created_at = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        budget_topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget tracker work"],
            language="en",
            taxonomy_tags=[],
            source_span_ids=[],
            confidence=0.95,
            created_at=created_at,
        )
        lunch_topic = ChronologyTopic(
            topic_id="topic-lunch",
            scope=scope,
            canonical_label="lunch plans",
            aliases=[],
            language="en",
            taxonomy_tags=[],
            source_span_ids=[],
            confidence=0.5,
            created_at=created_at,
        )
        for topic in (budget_topic, lunch_topic):
            memory.store.upsert_chronology_topic(topic)
            memory.store.upsert_chronology_phase(
                ChronologyPhase(
                    phase_id=f"phase-{topic.topic_id}",
                    topic_id=topic.topic_id,
                    phase_type="implementation",
                    order_hint=10,
                    source_span_ids=[],
                    confidence=0.9,
                    created_at=created_at,
                )
            )
        for node_id, topic_id, text, timestamp, marker in (
            ("node-budget-1", "topic-budget", "I first set up the budget tracker schema.", "2026-06-18T10:00:00+00:00", "first"),
            ("node-budget-2", "topic-budget", "Then I implemented transaction CRUD validation.", "2026-06-18T10:05:00+00:00", "then"),
            ("node-lunch", "topic-lunch", "Unrelated: I changed my lunch plan.", "2026-06-18T10:10:00+00:00", None),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id=node_id,
                    scope=scope,
                    actor="user",
                    action="implemented",
                    object="budget tracker",
                    topic_id=topic_id,
                    phase_id=f"phase-{topic_id}",
                    timestamp=ts(timestamp),
                    source_span_id=None,
                    source_turn_id=None,
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=marker,
                    created_at=created_at,
                )
            )

        candidates, telemetry = select_persisted_graph_event_ordering_candidates(
            "List the budget tracker work in order.",
            scope,
            memory.store,
            limit=5,
            include_session=True,
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0].source, "event_ordering_persisted_graph")
        self.assertIn("schema", candidates[0].text.lower())
        self.assertIn("crud", candidates[1].text.lower())
        self.assertTrue(all("lunch" not in candidate.text.lower() for candidate in candidates))
        self.assertEqual(telemetry["selected_driver"], "persisted_graph")

    def test_persisted_graph_selector_does_not_expand_outside_selected_topics(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-select-leak", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        budget_topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget tracker work"],
            language="en",
            taxonomy_tags=[],
            source_span_ids=[],
            confidence=0.95,
            created_at=created_at,
        )
        lunch_topic = ChronologyTopic(
            topic_id="topic-lunch",
            scope=scope,
            canonical_label="lunch plans",
            aliases=["meal break"],
            language="en",
            taxonomy_tags=[],
            source_span_ids=[],
            confidence=0.40,
            created_at=created_at,
        )
        for topic in (budget_topic, lunch_topic):
            memory.store.upsert_chronology_topic(topic)
            memory.store.upsert_chronology_phase(
                ChronologyPhase(
                    phase_id=f"phase-{topic.topic_id}",
                    topic_id=topic.topic_id,
                    phase_type="implementation",
                    order_hint=10,
                    source_span_ids=[],
                    confidence=0.9,
                    created_at=created_at,
                )
            )
        for node_id, topic_id, text, timestamp, marker in (
            ("node-budget-1", "topic-budget", "I first set up the budget tracker schema.", "2026-06-18T10:00:00+00:00", "first"),
            ("node-budget-2", "topic-budget", "Then I implemented budget tracker transaction CRUD.", "2026-06-18T10:05:00+00:00", "then"),
            ("node-lunch", "topic-lunch", "Then I ordered budget lunch bowls.", "2026-06-18T10:10:00+00:00", "then"),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id=node_id,
                    scope=scope,
                    actor="user",
                    action="implemented",
                    object="budget",
                    topic_id=topic_id,
                    phase_id=f"phase-{topic_id}",
                    timestamp=ts(timestamp),
                    source_span_id=None,
                    source_turn_id=None,
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=marker,
                    created_at=created_at,
                )
            )

        candidates, telemetry = select_persisted_graph_event_ordering_candidates(
            "List the budget tracker work in order.",
            scope,
            memory.store,
            limit=5,
            include_session=True,
        )

        self.assertEqual(telemetry["selected_driver"], "persisted_graph")
        self.assertEqual({candidate.metadata["graph_topic_id"] for candidate in candidates}, {"topic-budget"})
        self.assertTrue(all("lunch" not in candidate.text.lower() for candidate in candidates))

    def test_single_topic_anchor_does_not_chain_complete_from_other_topic(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-select-single-anchor", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        budget_topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget tracker work"],
            language="en",
            taxonomy_tags=[],
            source_span_ids=[],
            confidence=0.95,
            created_at=created_at,
        )
        lunch_topic = ChronologyTopic(
            topic_id="topic-lunch",
            scope=scope,
            canonical_label="lunch plans",
            aliases=["meal break"],
            language="en",
            taxonomy_tags=[],
            source_span_ids=[],
            confidence=0.40,
            created_at=created_at,
        )
        for topic in (budget_topic, lunch_topic):
            memory.store.upsert_chronology_topic(topic)
            memory.store.upsert_chronology_phase(
                ChronologyPhase(
                    phase_id=f"phase-{topic.topic_id}",
                    topic_id=topic.topic_id,
                    phase_type="implementation",
                    order_hint=10,
                    source_span_ids=[],
                    confidence=0.9,
                    created_at=created_at,
                )
            )
        for node_id, topic_id, text, timestamp, marker in (
            ("node-budget-1", "topic-budget", "I first set up the budget tracker schema.", "2026-06-18T10:00:00+00:00", "first"),
            ("node-lunch", "topic-lunch", "Then I chose a budget-friendly lunch bowl.", "2026-06-18T10:10:00+00:00", "then"),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id=node_id,
                    scope=scope,
                    actor="user",
                    action="planned",
                    object="budget",
                    topic_id=topic_id,
                    phase_id=f"phase-{topic_id}",
                    timestamp=ts(timestamp),
                    source_span_id=None,
                    source_turn_id=None,
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=marker,
                    created_at=created_at,
                )
            )

        candidates, telemetry = select_persisted_graph_event_ordering_candidates(
            "List the budget tracker work in order.",
            scope,
            memory.store,
            limit=5,
            include_session=True,
        )

        self.assertEqual(telemetry["selected_driver"], "persisted_graph")
        self.assertEqual([candidate.metadata["graph_topic_id"] for candidate in candidates], ["topic-budget"])
        self.assertTrue(all("lunch" not in candidate.text.lower() for candidate in candidates))

    def test_service_uses_query_time_graph_selector_when_persisted_graph_is_unavailable(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-fallback-query-time", user_id="u", agent_id="a", session_id="s")
        memory.add("I set up the schema first.", scope, ts("2026-06-18T10:00:00+00:00"))

        def fake_graph_selector(query, spans, events, limit):
            return [
                Candidate(
                    id="graph-candidate",
                    type="event",
                    text="query-time graph candidate",
                    source="event_ordering_graph_selector",
                    scores={"score": 1.0},
                    source_span_ids=[],
                    metadata={"timeline_index": 1},
                )
            ]

        with patch("fusion_memory.api.service.select_graph_first_event_ordering_candidates", fake_graph_selector):
            candidates = memory._event_ordering_graph_selector_candidates(
                "List the budget tracker work in order.",
                scope,
                limit=5,
                include_session=True,
            )

        self.assertEqual(candidates[0].source, "event_ordering_graph_selector")
        self.assertEqual(candidates[0].metadata["persisted_graph_telemetry"]["fallback_reason"], "no_topic")

    def test_service_uses_legacy_timeline_when_no_graph_selector_is_available(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-fallback-legacy", user_id="u", agent_id="a", session_id="s")
        memory.add("I set up the schema first.", scope, ts("2026-06-18T10:00:00+00:00"))
        memory.add("Then I implemented CRUD validation.", scope, ts("2026-06-18T10:05:00+00:00"))

        with patch("fusion_memory.api.service.select_graph_first_event_ordering_candidates", None):
            candidates = memory._event_ordering_graph_selector_candidates(
                "List the budget tracker work in order.",
                scope,
                limit=5,
                include_session=True,
            )

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.source != "event_ordering_graph_selector" for candidate in candidates))
        self.assertTrue(any(candidate.source.startswith("event_ordering_coverage") for candidate in candidates))
        self.assertEqual(candidates[0].metadata["persisted_graph_telemetry"]["fallback_reason"], "no_topic")


class TaxonomyTests(unittest.TestCase):
    def test_default_taxonomy_matches_aliases_without_private_regex_branches(self) -> None:
        entries = load_default_taxonomy()
        hits = taxonomy_alias_hits("I deployed the Flask app on Render with CRUD endpoints.", entries)

        self.assertIn("flask", hits)
        self.assertIn("render", hits)
        self.assertIn("crud", hits)
