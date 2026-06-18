from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from fusion_memory.core.models import EvidenceSpan, MemoryEvent, Scope
from fusion_memory.core.text import stable_hash
from fusion_memory.retrieval.chronology_normalizer import build_chronology_write_batch


class ChronologyNormalizerTests(unittest.TestCase):
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
