from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fusion_memory import MemoryService
from fusion_memory.core.chronology import ChronologyEventEdge, ChronologyEventNode, ChronologyPhase, ChronologyTopic
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
                    object="schema setup" if node_id == "node-budget-1" else "transaction CRUD validation",
                    topic_id=topic_id,
                    phase_id=f"phase-{topic_id}",
                    timestamp=ts(timestamp),
                    source_span_id=f"span-{node_id}",
                    source_turn_id=f"turn-{node_id}",
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=marker,
                    created_at=created_at,
                )
            )
        memory.store.insert_chronology_event_edge(
            ChronologyEventEdge(
                edge_id="edge-budget",
                from_node_id="node-budget-1",
                to_node_id="node-budget-2",
                edge_type="before",
                evidence_type="explicit_marker",
                source_span_ids=["span-node-budget-1", "span-node-budget-2"],
                confidence=0.9,
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
                    source_span_id=f"span-{node_id}",
                    source_turn_id=f"turn-{node_id}",
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=marker,
                    created_at=created_at,
                )
            )
        memory.store.insert_chronology_event_edge(
            ChronologyEventEdge(
                edge_id="edge-budget",
                from_node_id="node-budget-1",
                to_node_id="node-budget-2",
                edge_type="before",
                evidence_type="explicit_marker",
                source_span_ids=["span-node-budget-1", "span-node-budget-2"],
                confidence=0.9,
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

        self.assertEqual(candidates, [])
        self.assertEqual(telemetry["fallback_reason"], "too_few_nodes")

    def test_selector_expands_from_single_matching_node_to_same_topic_timeline(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-select-topic-expand", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget tracker work"],
            language="en",
            taxonomy_tags=[],
            source_span_ids=["s1", "s2", "s3"],
            confidence=0.95,
            created_at=created_at,
        )
        memory.store.upsert_chronology_topic(topic)
        memory.store.upsert_chronology_phase(
            ChronologyPhase(
                phase_id="phase-budget",
                topic_id=topic.topic_id,
                phase_type="implementation",
                order_hint=20,
                source_span_ids=["s1", "s2", "s3"],
                confidence=0.9,
                created_at=created_at,
            )
        )
        for node_id, span_id, text, minute, marker in (
            ("node-budget-1", "s1", "I first set up the budget tracker schema.", 0, "first"),
            ("node-budget-2", "s2", "Then I implemented category filters.", 5, "then"),
            ("node-budget-3", "s3", "Finally I tested monthly reports.", 10, "finally"),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id=node_id,
                    scope=scope,
                    actor="user",
                    action="implemented",
                    object="budget tracker",
                    topic_id=topic.topic_id,
                    phase_id="phase-budget",
                    timestamp=ts(f"2026-06-18T10:{minute:02d}:00+00:00"),
                    source_span_id=span_id,
                    source_turn_id=f"turn-{span_id}",
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=marker,
                    created_at=created_at,
                )
            )
        for edge_id, left, right, spans in (
            ("edge-1", "node-budget-1", "node-budget-2", ["s1", "s2"]),
            ("edge-2", "node-budget-2", "node-budget-3", ["s2", "s3"]),
        ):
            memory.store.insert_chronology_event_edge(
                ChronologyEventEdge(
                    edge_id=edge_id,
                    from_node_id=left,
                    to_node_id=right,
                    edge_type="before",
                    evidence_type="explicit_marker",
                    source_span_ids=spans,
                    confidence=0.9,
                    created_at=created_at,
                )
            )

        candidates, telemetry = select_persisted_graph_event_ordering_candidates(
            "What order did I mention category filters?",
            scope,
            memory.store,
            limit=5,
            include_session=True,
        )

        self.assertEqual(telemetry["selected_driver"], "persisted_graph")
        self.assertEqual([candidate.id for candidate in candidates], ["node-budget-1", "node-budget-2", "node-budget-3"])

    def test_persisted_graph_candidate_text_uses_short_object_label(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-select-short-label", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget tracker work"],
            language="en",
            taxonomy_tags=[],
            source_span_ids=["s1", "s2"],
            confidence=0.95,
            created_at=created_at,
        )
        memory.store.upsert_chronology_topic(topic)
        memory.store.upsert_chronology_phase(
            ChronologyPhase(
                phase_id="phase-budget",
                topic_id=topic.topic_id,
                phase_type="implementation",
                order_hint=20,
                source_span_ids=["s1", "s2"],
                confidence=0.9,
                created_at=created_at,
            )
        )
        for node_id, span_id, obj, text, minute in (
            ("node-core", "s1", "core functionality", "I'm building a budget tracker and need user auth, expenses, and charts.", 0),
            ("node-crud", "s2", "transaction CRUD implementation", "Then I worked on transaction CRUD implementation.", 5),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id=node_id,
                    scope=scope,
                    actor="user",
                    action="implemented",
                    object=obj,
                    topic_id=topic.topic_id,
                    phase_id="phase-budget",
                    timestamp=ts(f"2026-06-18T10:{minute:02d}:00+00:00"),
                    source_span_id=span_id,
                    source_turn_id=f"turn-{span_id}",
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker="then" if node_id == "node-crud" else "first",
                    created_at=created_at,
                )
            )
        memory.store.insert_chronology_event_edge(
            ChronologyEventEdge(
                edge_id="edge-budget",
                from_node_id="node-core",
                to_node_id="node-crud",
                edge_type="before",
                evidence_type="explicit_marker",
                source_span_ids=["s1", "s2"],
                confidence=0.9,
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
        self.assertEqual([candidate.text for candidate in candidates], ["core functionality", "transaction CRUD implementation"])

    def test_selector_falls_back_to_edge_connected_topic_when_topical_hits_are_isolated(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-select-edge-topic", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        isolated_topics = [
            ChronologyTopic(f"topic-isolated-{idx}", scope, f"isolated match {idx}", [f"autocomplete isolated {idx}"], "en", [], [f"iso-{idx}"], 0.9, created_at)
            for idx in range(3)
        ]
        connected_topic = ChronologyTopic(
            "topic-weather",
            scope,
            "weather app",
            ["city autocomplete", "weather app work"],
            "en",
            [],
            ["s1", "s2"],
            0.8,
            created_at,
        )
        for topic in [*isolated_topics, connected_topic]:
            memory.store.upsert_chronology_topic(topic)
            memory.store.upsert_chronology_phase(
                ChronologyPhase(f"phase-{topic.topic_id}", topic.topic_id, "implementation", 20, topic.source_span_ids, 0.8, created_at)
            )
        for idx, topic in enumerate(isolated_topics):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    f"node-isolated-{idx}",
                    scope,
                    "user",
                    "asked",
                    f"autocomplete isolated {idx}",
                    topic.topic_id,
                    f"phase-{topic.topic_id}",
                    ts(f"2026-06-18T10:0{idx}:00+00:00"),
                    f"iso-{idx}",
                    f"turn-iso-{idx}",
                    f"Autocomplete isolated note {idx}",
                    "en",
                    0.8,
                    None,
                    created_at,
                )
            )
        for node_id, obj, text, minute, marker in (
            ("node-weather-1", "debounce implementation", "I first implemented debounce for city autocomplete.", 10, "first"),
            ("node-weather-2", "dropdown error handling", "Then I handled dropdown and invalid city errors.", 15, "then"),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id,
                    scope,
                    "user",
                    "implemented",
                    obj,
                    connected_topic.topic_id,
                    f"phase-{connected_topic.topic_id}",
                    ts(f"2026-06-18T10:{minute}:00+00:00"),
                    f"s-{node_id}",
                    f"turn-{node_id}",
                    text,
                    "en",
                    0.9,
                    marker,
                    created_at,
                )
            )
        memory.store.insert_chronology_event_edge(
            ChronologyEventEdge(
                "edge-weather",
                "node-weather-1",
                "node-weather-2",
                "before",
                "explicit_marker",
                ["s-node-weather-1", "s-node-weather-2"],
                0.9,
                created_at,
            )
        )

        candidates, telemetry = select_persisted_graph_event_ordering_candidates(
            "List the city autocomplete feature work in order.",
            scope,
            memory.store,
            limit=5,
            include_session=True,
        )

        self.assertEqual(telemetry["selected_driver"], "persisted_graph")
        self.assertEqual([candidate.text for candidate in candidates], ["debounce implementation", "dropdown error handling"])


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
        self.assertIn(candidates[0].metadata["persisted_graph_telemetry"]["fallback_reason"], {"no_topic", "too_few_nodes"})

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
        self.assertIn(candidates[0].metadata["persisted_graph_telemetry"]["fallback_reason"], {"no_topic", "too_few_nodes"})

    def test_service_falls_back_when_chronology_tables_are_missing(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-fallback-missing-tables", user_id="u", agent_id="a", session_id="s")
        memory.add("I set up the schema first.", scope, ts("2026-06-18T10:00:00+00:00"))

        class MissingChronologyStore:
            def __init__(self, wrapped) -> None:
                self.wrapped = wrapped
                self.rollback_count = 0

            def __getattr__(self, name: str):
                return getattr(self.wrapped, name)

            def connect(self):
                return self

            def rollback(self):
                self.rollback_count += 1

            def list_chronology_topics(self, *args, **kwargs):
                raise RuntimeError('relation "chronology_topics" does not exist')

        missing_store = MissingChronologyStore(memory.store)
        memory.store = missing_store

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
                "List the schema work in order.",
                scope,
                limit=5,
                include_session=True,
            )

        self.assertEqual(candidates[0].source, "event_ordering_graph_selector")
        self.assertEqual(candidates[0].metadata["persisted_graph_telemetry"]["fallback_reason"], "graph_unavailable")
        self.assertEqual(candidates[0].metadata["persisted_graph_telemetry"]["error"], "RuntimeError")
        self.assertEqual(missing_store.rollback_count, 1)

    def test_selector_returns_no_candidates_for_single_sparse_node(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-sparse-single-node", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget app"],
            language="en",
            taxonomy_tags=["software"],
            source_span_ids=["s1"],
            confidence=0.95,
            created_at=created_at,
        )
        memory.store.upsert_chronology_topic(topic)
        memory.store.upsert_chronology_phase(
            ChronologyPhase(
                phase_id="phase-budget",
                topic_id=topic.topic_id,
                phase_type="setup",
                order_hint=10,
                source_span_ids=["s1"],
                confidence=0.9,
                created_at=created_at,
            )
        )
        memory.store.upsert_chronology_event_node(
            ChronologyEventNode(
                node_id="node-budget-1",
                scope=scope,
                actor="user",
                action="set up",
                object="budget tracker",
                topic_id=topic.topic_id,
                phase_id="phase-budget",
                timestamp=created_at,
                source_span_id="s1",
                source_turn_id="t1",
                text="I first set up the budget tracker schema.",
                language="en",
                confidence=0.9,
                explicit_order_marker="first",
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

        self.assertEqual(candidates, [])
        self.assertEqual(telemetry["fallback_reason"], "too_few_nodes")

    def test_selector_returns_no_candidates_when_graph_has_no_edges(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-sparse-no-edges", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget app"],
            language="en",
            taxonomy_tags=["software"],
            source_span_ids=["s1", "s2"],
            confidence=0.95,
            created_at=created_at,
        )
        memory.store.upsert_chronology_topic(topic)
        memory.store.upsert_chronology_phase(
            ChronologyPhase(
                phase_id="phase-budget",
                topic_id=topic.topic_id,
                phase_type="unknown",
                order_hint=None,
                source_span_ids=["s1", "s2"],
                confidence=0.9,
                created_at=created_at,
            )
        )
        for node_id, span_id, text in (
            ("node-budget-1", "s1", "Budget tracker category notes."),
            ("node-budget-2", "s2", "Budget tracker report notes."),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id=node_id,
                    scope=scope,
                    actor="user",
                    action="noted",
                    object="budget tracker",
                    topic_id=topic.topic_id,
                    phase_id="phase-budget",
                    timestamp=None,
                    source_span_id=span_id,
                    source_turn_id=f"t-{span_id}",
                    text=text,
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=None,
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

        self.assertEqual(candidates, [])
        self.assertEqual(telemetry["fallback_reason"], "no_edges")

    def test_selector_returns_no_candidates_for_weak_source_span_coverage(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-sparse-weak-coverage", user_id="u", agent_id="a", session_id="s")
        created_at = ts("2026-06-18T10:00:00+00:00")
        topic = ChronologyTopic(
            topic_id="topic-budget",
            scope=scope,
            canonical_label="budget tracker",
            aliases=["budget app"],
            language="en",
            taxonomy_tags=["software"],
            source_span_ids=["s1"],
            confidence=0.95,
            created_at=created_at,
        )
        memory.store.upsert_chronology_topic(topic)
        memory.store.upsert_chronology_phase(
            ChronologyPhase(
                phase_id="phase-budget",
                topic_id=topic.topic_id,
                phase_type="implementation",
                order_hint=20,
                source_span_ids=["s1"],
                confidence=0.9,
                created_at=created_at,
            )
        )
        for node_id, marker, offset in (
            ("node-budget-1", "first", 0),
            ("node-budget-2", "then", 5),
        ):
            memory.store.upsert_chronology_event_node(
                ChronologyEventNode(
                    node_id=node_id,
                    scope=scope,
                    actor="user",
                    action="implemented",
                    object="budget tracker",
                    topic_id=topic.topic_id,
                    phase_id="phase-budget",
                    timestamp=ts(f"2026-06-18T10:{offset:02d}:00+00:00"),
                    source_span_id="s1",
                    source_turn_id=f"t-{node_id}",
                    text=f"{marker.title()} I updated the budget tracker.",
                    language="en",
                    confidence=0.9,
                    explicit_order_marker=marker,
                    created_at=created_at,
                )
            )
        memory.store.insert_chronology_event_edge(
            type("Edge", (), {
                "edge_id": "edge-budget",
                "from_node_id": "node-budget-1",
                "to_node_id": "node-budget-2",
                "edge_type": "before",
                "evidence_type": "explicit_marker",
                "source_span_ids": ["s1"],
                "confidence": 0.9,
                "created_at": created_at,
            })()
        )

        candidates, telemetry = select_persisted_graph_event_ordering_candidates(
            "List the budget tracker work in order.",
            scope,
            memory.store,
            limit=5,
            include_session=True,
        )

        self.assertEqual(candidates, [])
        self.assertEqual(telemetry["fallback_reason"], "weak_coverage")


class TaxonomyTests(unittest.TestCase):
    def test_default_taxonomy_matches_aliases_without_private_regex_branches(self) -> None:
        entries = load_default_taxonomy()
        hits = taxonomy_alias_hits("I deployed the Flask app on Render with CRUD endpoints.", entries)

        self.assertIn("flask", hits)
        self.assertIn("render", hits)
        self.assertIn("crud", hits)


class TaxonomyMigrationTests(unittest.TestCase):
    def test_taxonomy_covers_domain_labels_used_by_event_ordering_rules(self) -> None:
        entries = load_default_taxonomy()
        hits = taxonomy_alias_hits("Gunicorn worker ports and SQLite schema migrations", entries)

        self.assertIn("gunicorn", hits)
        self.assertIn("sqlite", hits)
        self.assertIn("schema migration", hits)
