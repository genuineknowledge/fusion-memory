from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from fusion_memory import MemoryService
from fusion_memory.core.models import EvidenceSpan, MemoryEvent, Scope
from fusion_memory.core.text import stable_hash
from fusion_memory.retrieval.chronology_normalizer import build_chronology_write_batch


def _derived_written_chronology_graph(memory: MemoryService, result, scope: Scope) -> dict:
    trace = memory.debug_trace(result.trace_id, scope)
    assert trace is not None
    for step in trace["steps"]:
        if step.get("step") == "derived_written":
            return step["chronology_graph"]
    raise AssertionError("derived_written chronology_graph trace step not found")


class FailingChronologyReadStore:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.fail_chronology_read = False

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)

    def insert_event(self, event) -> None:
        self.wrapped.insert_event(event)
        self.fail_chronology_read = True

    def list_events(self, *args, **kwargs):
        if self.fail_chronology_read and kwargs.get("include_session") is True:
            raise RuntimeError("chronology list failed")
        return self.wrapped.list_events(*args, **kwargs)


class ChronologyNormalizerTests(unittest.TestCase):
    def test_node_object_uses_short_aspect_label_instead_of_full_topic(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        span = EvidenceSpan(
            span_id="s1",
            scope=scope,
            turn_id="t1",
            speaker="user",
            span_type="turn",
            content=(
                "I'm building a personal budget tracker and need help implementing the core functionality, "
                "including user authentication, expense tracking, and data visualization."
            ),
            content_hash=stable_hash("aspect-label-core"),
            timestamp=base,
        )
        event = MemoryEvent(
            event_id="e1",
            scope=scope,
            event_type="user_action",
            description=span.content,
            participants=["user"],
            source_span_ids=["s1"],
            time_start=base,
            confidence=0.8,
        )

        batch = build_chronology_write_batch(scope, [span], [event])

        self.assertEqual(batch.nodes[0].object, "core functionality")
        self.assertEqual(batch.topics[0].canonical_label, "budget tracker")

    def test_build_chronology_batch_skips_assistant_answer_nodes(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                span_id="s-user",
                scope=scope,
                turn_id="t1",
                speaker="user",
                span_type="turn",
                content="First I asked about transaction CRUD implementation.",
                content_hash=stable_hash("user-crud"),
                timestamp=base,
            ),
            EvidenceSpan(
                span_id="s-assistant",
                scope=scope,
                turn_id="t2",
                speaker="assistant",
                span_type="turn",
                content="Certainly! Let's enhance your BudgetTracker class with validation and error handling.",
                content_hash=stable_hash("assistant-answer"),
                timestamp=base + timedelta(minutes=1),
            ),
        ]
        events = [
            MemoryEvent("e-user", scope, "user_action", spans[0].content, ["user"], ["s-user"], time_start=base, confidence=0.8),
            MemoryEvent(
                "e-assistant",
                scope,
                "assistant_action",
                spans[1].content,
                ["assistant"],
                ["s-assistant"],
                time_start=base + timedelta(minutes=1),
                confidence=0.8,
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, events)

        self.assertEqual(len(batch.nodes), 1)
        self.assertEqual(batch.nodes[0].source_span_id, "s-user")

    def test_build_chronology_batch_adds_user_span_nodes_when_events_are_assistant_only(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                span_id="s-user-1",
                scope=scope,
                turn_id="t1",
                speaker="user",
                span_type="turn",
                content="First I'm trying to master classifying triangles by sides and angles, starting with equilateral, isosceles, and scalene types.",
                content_hash=stable_hash("user-triangle-classification"),
                timestamp=base,
            ),
            EvidenceSpan(
                span_id="s-user-2",
                scope=scope,
                turn_id="t2",
                speaker="user",
                span_type="turn",
                content="Then I moved on to calculating triangle areas and comparing altitude and median formulas.",
                content_hash=stable_hash("user-triangle-area"),
                timestamp=base + timedelta(minutes=5),
            ),
            EvidenceSpan(
                span_id="s-assistant",
                scope=scope,
                turn_id="t3",
                speaker="assistant",
                span_type="turn",
                content="Certainly, let's go through the Pythagorean theorem step by step.",
                content_hash=stable_hash("assistant-triangle-answer"),
                timestamp=base + timedelta(minutes=6),
            ),
        ]
        events = [
            MemoryEvent(
                "e-assistant",
                scope,
                "assistant_action",
                spans[2].content,
                ["assistant"],
                ["s-assistant"],
                time_start=base + timedelta(minutes=6),
                confidence=0.8,
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, events)

        self.assertEqual([node.source_span_id for node in batch.nodes], ["s-user-1", "s-user-2"])
        self.assertEqual(batch.nodes[0].object, "triangle classification")
        self.assertEqual(batch.nodes[1].object, "triangle area methods")
        self.assertTrue(batch.edges)

    def test_user_span_topic_clustering_merges_fragmented_session_topics(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                span_id="tri-1",
                scope=scope,
                turn_id="t1",
                speaker="user",
                span_type="turn",
                content="First I'm trying to master classifying triangles by sides and angles, starting with equilateral, isosceles, and scalene types.",
                content_hash=stable_hash("tri-topic-1"),
                timestamp=base,
            ),
            EvidenceSpan(
                span_id="tri-2",
                scope=scope,
                turn_id="t2",
                speaker="user",
                span_type="turn",
                content="I want to compare triangle area methods using altitude, medians, and Heron's formula.",
                content_hash=stable_hash("tri-topic-2"),
                timestamp=base + timedelta(minutes=5),
            ),
            EvidenceSpan(
                span_id="tri-3",
                scope=scope,
                turn_id="t3",
                speaker="user",
                span_type="turn",
                content="Then I applied triangle geometry to Law of Cosines examples for non-right triangles.",
                content_hash=stable_hash("tri-topic-3"),
                timestamp=base + timedelta(minutes=10),
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, [])

        self.assertEqual({topic.canonical_label for topic in batch.topics}, {"triangle geometry"})
        self.assertEqual({node.topic_id for node in batch.nodes}, {batch.topics[0].topic_id})
        self.assertGreaterEqual(len(batch.edges), 2)

    def test_write_batch_telemetry_reports_topic_cluster_merges(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                "s1",
                scope,
                "t1",
                "user",
                "turn",
                "First I studied triangle classification with equilateral and scalene examples.",
                stable_hash("tc1"),
                base,
            ),
            EvidenceSpan(
                "s2",
                scope,
                "t2",
                "user",
                "turn",
                "Then I compared median formulas and altitude methods.",
                stable_hash("tc2"),
                base + timedelta(minutes=5),
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, [])

        self.assertEqual({topic.canonical_label for topic in batch.topics}, {"triangle geometry"})
        self.assertGreaterEqual(batch.telemetry["topic_cluster"]["merged_by_session_hint"], 1)

    def test_build_chronology_batch_extracts_action_object_phase_topic_and_order_edges(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                span_id="s1",
                scope=scope,
                turn_id="t1",
                speaker="user",
                span_type="turn",
                content="I first set up the budget tracker schema.",
                content_hash=stable_hash("s1"),
                timestamp=base,
            ),
            EvidenceSpan(
                span_id="s2",
                scope=scope,
                turn_id="t2",
                speaker="user",
                span_type="turn",
                content="Then I implemented transaction CRUD validation.",
                content_hash=stable_hash("s2"),
                timestamp=base + timedelta(minutes=5),
            ),
        ]
        events = [
            MemoryEvent(
                event_id="e1",
                scope=scope,
                event_type="user_action",
                description=spans[0].content,
                participants=["user"],
                source_span_ids=["s1"],
                time_start=base,
                confidence=0.8,
            ),
            MemoryEvent(
                event_id="e2",
                scope=scope,
                event_type="user_action",
                description=spans[1].content,
                participants=["user"],
                source_span_ids=["s2"],
                time_start=base + timedelta(minutes=5),
                confidence=0.8,
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, events)

        self.assertEqual(len(batch.nodes), 2)
        self.assertEqual(batch.nodes[0].actor, "user")
        self.assertEqual(batch.nodes[0].explicit_order_marker, "first")
        self.assertEqual(batch.nodes[1].explicit_order_marker, "then")
        self.assertTrue(any(topic.canonical_label == "budget tracker" for topic in batch.topics))
        self.assertTrue(any(phase.phase_type == "setup" for phase in batch.phases))
        self.assertTrue(
            any(edge.edge_type == "before" and edge.evidence_type == "explicit_marker" for edge in batch.edges)
        )
        budget_topic = next(topic for topic in batch.topics if topic.canonical_label == "budget tracker")
        self.assertIn("budget app", budget_topic.aliases)
        self.assertIn("software", budget_topic.taxonomy_tags)

    def test_chinese_order_markers_are_supported_without_llm(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                span_id="cn1",
                scope=scope,
                turn_id="t1",
                speaker="user",
                span_type="turn",
                content="我先完成了记忆系统的初始化配置。",
                content_hash=stable_hash("cn1"),
                timestamp=base,
            ),
            EvidenceSpan(
                span_id="cn2",
                scope=scope,
                turn_id="t2",
                speaker="user",
                span_type="turn",
                content="然后我开始测试中文召回。",
                content_hash=stable_hash("cn2"),
                timestamp=base + timedelta(minutes=5),
            ),
        ]
        events = [
            MemoryEvent("e1", scope, "user_action", spans[0].content, ["user"], ["cn1"], time_start=base, confidence=0.8),
            MemoryEvent(
                "e2",
                scope,
                "user_action",
                spans[1].content,
                ["user"],
                ["cn2"],
                time_start=base + timedelta(minutes=5),
                confidence=0.8,
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, events)

        self.assertEqual([node.language for node in batch.nodes], ["zh", "zh"])
        self.assertEqual(batch.nodes[0].explicit_order_marker, "first")
        self.assertEqual(batch.nodes[1].explicit_order_marker, "then")
        self.assertTrue(batch.edges)
        memory_topic = next(topic for topic in batch.topics if topic.canonical_label == "memory system")
        self.assertIn("记忆系统", memory_topic.aliases)
        self.assertIn("memory", memory_topic.taxonomy_tags)

    def test_taxonomy_alias_match_persists_aliases_and_tags_for_non_hardcoded_topic_labels(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                span_id="s1",
                scope=scope,
                turn_id="t1",
                speaker="user",
                span_type="turn",
                content="I first set up the budget app schema.",
                content_hash=stable_hash("s1-budget-app"),
                timestamp=base,
            ),
        ]
        events = [
            MemoryEvent(
                event_id="e1",
                scope=scope,
                event_type="user_action",
                description=spans[0].content,
                participants=["user"],
                source_span_ids=["s1"],
                time_start=base,
                confidence=0.8,
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, events)

        self.assertEqual([topic.canonical_label for topic in batch.topics], ["budget tracker"])
        self.assertIn("budget app", batch.topics[0].aliases)
        self.assertIn("software", batch.topics[0].taxonomy_tags)

    def test_timestamp_only_unknown_phase_adjacency_is_suppressed(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
        spans = [
            EvidenceSpan(
                span_id="s1",
                scope=scope,
                turn_id="t1",
                speaker="user",
                span_type="turn",
                content="Budget tracker notes about categories.",
                content_hash=stable_hash("s1"),
                timestamp=base,
            ),
            EvidenceSpan(
                span_id="s2",
                scope=scope,
                turn_id="t2",
                speaker="user",
                span_type="turn",
                content="Budget tracker notes about reports.",
                content_hash=stable_hash("s2"),
                timestamp=base + timedelta(minutes=5),
            ),
        ]
        events = [
            MemoryEvent(
                event_id="e1",
                scope=scope,
                event_type="user_action",
                description=spans[0].content,
                participants=["user"],
                source_span_ids=["s1"],
                time_start=base,
                confidence=0.8,
            ),
            MemoryEvent(
                event_id="e2",
                scope=scope,
                event_type="user_action",
                description=spans[1].content,
                participants=["user"],
                source_span_ids=["s2"],
                time_start=base + timedelta(minutes=5),
                confidence=0.8,
            ),
        ]

        batch = build_chronology_write_batch(scope, spans, events)

        self.assertEqual([phase.phase_type for phase in batch.phases], ["unknown"])
        self.assertEqual(batch.edges, [])

    def test_timestamp_less_inputs_use_deterministic_created_at(self) -> None:
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        spans: list[EvidenceSpan] = []
        events = [
            MemoryEvent(
                event_id="e1",
                scope=scope,
                event_type="user_action",
                description="Budget tracker notes about categories.",
                participants=["user"],
                source_span_ids=[],
                confidence=0.8,
            ),
        ]

        first_batch = build_chronology_write_batch(scope, spans, events)
        second_batch = build_chronology_write_batch(scope, spans, events)

        self.assertEqual(first_batch.nodes[0].created_at, second_batch.nodes[0].created_at)


class ChronologyServiceIntegrationTests(unittest.TestCase):
    def test_add_writes_chronology_graph_without_changing_add_contract(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-write", user_id="u", agent_id="a", session_id="s")
        timestamp = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)

        result = memory.add(
            "I first set up the graph schema. Then I implemented the selector.",
            scope,
            timestamp,
            {"source_uri": "test:graph-write"},
        )

        self.assertTrue(result.span_ids)
        nodes = memory.store.list_chronology_event_nodes(scope, include_session=True)
        self.assertGreaterEqual(len(nodes), 1)
        self.assertTrue(any(node.source_span_id in result.span_ids for node in nodes))

    def test_add_trace_includes_chronology_graph_counts(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="graph-trace-success", user_id="u", agent_id="a", session_id="s")
        result = memory.add(
            "I first set up the graph schema. Then I implemented the selector.",
            scope,
            datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc),
            {"source_uri": "test:graph-trace-success"},
        )

        chronology_graph = _derived_written_chronology_graph(memory, result, scope)

        self.assertTrue(chronology_graph["enabled"])
        self.assertGreaterEqual(chronology_graph["node_count"], 1)
        self.assertGreaterEqual(chronology_graph["topic_count"], 1)
        self.assertGreaterEqual(chronology_graph["phase_count"], 1)
        self.assertIn("edge_count", chronology_graph)
        self.assertNotIn("error", chronology_graph)

    def test_add_trace_records_chronology_graph_error_non_fatally(self) -> None:
        memory = MemoryService()
        memory.store = FailingChronologyReadStore(memory.store)
        scope = Scope(workspace_id="graph-trace-error", user_id="u", agent_id="a", session_id="s")

        result = memory.add(
            "I first set up the graph schema. Then I implemented the selector.",
            scope,
            datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc),
            {"source_uri": "test:graph-trace-error"},
        )

        self.assertTrue(result.span_ids)
        chronology_graph = _derived_written_chronology_graph(memory, result, scope)
        self.assertTrue(chronology_graph["enabled"])
        self.assertEqual(chronology_graph["error"], "RuntimeError")
        self.assertEqual(chronology_graph["node_count"], 0)
        self.assertEqual(chronology_graph["edge_count"], 0)
