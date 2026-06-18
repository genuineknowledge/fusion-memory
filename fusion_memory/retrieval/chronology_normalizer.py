from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from fusion_memory.core.chronology import ChronologyEventEdge, ChronologyEventNode, ChronologyPhase, ChronologyTopic
from fusion_memory.core.models import EvidenceSpan, MemoryEvent, Scope
from fusion_memory.core.text import compact_summary, stable_hash, tokenize


@dataclass
class ChronologyWriteBatch:
    topics: list[ChronologyTopic]
    phases: list[ChronologyPhase]
    nodes: list[ChronologyEventNode]
    edges: list[ChronologyEventEdge]
    telemetry: dict[str, object]


ORDER_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("first", ("first", "initially", "started", "start", "首先", "一开始", "先")),
    ("then", ("after that", "then", "next", "later", "然后", "接着", "随后")),
    ("finally", ("finally", "最后")),
    ("before", ("before", "之前")),
    ("after", ("after", "之后")),
)

PHASE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("setup", ("set up", "setup", "initialize", "初始化", "schema", "配置")),
    ("implementation", ("implement", "build", "add", "实现", "开发", "完成")),
    ("debug", ("debug", "fix", "error", "修复", "报错")),
    ("validation", ("test", "verify", "coverage", "测试", "验证")),
    ("release", ("deploy", "release", "上线", "部署")),
)

STOPWORDS = {
    "a",
    "after",
    "an",
    "and",
    "first",
    "i",
    "implemented",
    "next",
    "later",
    "set",
    "the",
    "then",
    "up",
}

PHASE_ORDER_HINTS = {
    "setup": 10,
    "implementation": 20,
    "debug": 30,
    "validation": 40,
    "release": 50,
    "unknown": None,
}


def build_chronology_write_batch(
    scope: Scope, spans: list[EvidenceSpan], events: list[MemoryEvent]
) -> ChronologyWriteBatch:
    span_by_id = {span.span_id: span for span in spans}
    created_at = _created_at(events, spans)
    topics_by_label: dict[str, ChronologyTopic] = {}
    phases_by_key: dict[tuple[str, str], ChronologyPhase] = {}
    nodes: list[ChronologyEventNode] = []
    last_topic_label: str | None = None

    for index, event in enumerate(events):
        span = _first_source_span(event, span_by_id)
        text = span.content if span is not None else event.description
        language = _infer_language(text)
        topic_label, topic_is_strong = _infer_topic_label(text)
        if not topic_is_strong and last_topic_label is not None and _infer_order_marker(text) is not None:
            topic_label = last_topic_label
        last_topic_label = topic_label
        topic = _get_topic(scope, topics_by_label, topic_label, language, event, span, created_at)
        phase_type = _infer_phase_type(text)
        phase = _get_phase(phases_by_key, topic, phase_type, event, span, created_at)
        actor = event.participants[0] if event.participants else "unknown"
        node = ChronologyEventNode(
            node_id=_id("chron_node", scope, event.event_id, span.span_id if span is not None else "", index),
            scope=scope,
            actor=actor,
            action=_infer_action(text),
            object=topic.canonical_label,
            topic_id=topic.topic_id,
            phase_id=phase.phase_id,
            timestamp=event.time_start or (span.timestamp if span is not None else None),
            source_span_id=span.span_id if span is not None else None,
            source_turn_id=span.turn_id if span is not None else None,
            text=compact_summary(text),
            language=language,
            confidence=event.confidence,
            explicit_order_marker=_infer_order_marker(text),
            created_at=created_at,
        )
        nodes.append(node)

    edges = _build_edges(nodes, created_at)
    telemetry = {
        "input_span_count": len(spans),
        "input_event_count": len(events),
        "topic_count": len(topics_by_label),
        "phase_count": len(phases_by_key),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    return ChronologyWriteBatch(
        topics=list(topics_by_label.values()),
        phases=list(phases_by_key.values()),
        nodes=nodes,
        edges=edges,
        telemetry=telemetry,
    )


def _created_at(events: list[MemoryEvent], spans: list[EvidenceSpan]) -> datetime:
    timestamps = [event.time_start for event in events if event.time_start is not None]
    timestamps.extend(span.timestamp for span in spans)
    return min(timestamps) if timestamps else datetime.now(timezone.utc)


def _first_source_span(event: MemoryEvent, span_by_id: dict[str, EvidenceSpan]) -> EvidenceSpan | None:
    for span_id in event.source_span_ids:
        span = span_by_id.get(span_id)
        if span is not None:
            return span
    return None


def _infer_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _infer_order_marker(text: str) -> str | None:
    lowered = text.lower()
    for marker, phrases in ORDER_MARKERS:
        if any(_contains_phrase(lowered, phrase) for phrase in phrases):
            return marker
    return None


def _infer_phase_type(text: str) -> str:
    lowered = text.lower()
    for phase_type, phrases in PHASE_RULES:
        if any(_contains_phrase(lowered, phrase) for phrase in phrases):
            return phase_type
    return "unknown"


def _contains_phrase(lowered_text: str, phrase: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", phrase):
        return phrase in lowered_text
    return re.search(rf"\b{re.escape(phrase)}\b", lowered_text) is not None


def _infer_topic_label(text: str) -> tuple[str, bool]:
    lowered = text.lower()
    if "budget" in lowered and "tracker" in lowered:
        return "budget tracker", True
    if "memory" in lowered or "记忆系统" in lowered:
        return "memory system", True

    meaningful_tokens = [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) > 1 and not re.search(r"[\u4e00-\u9fff]", token)
    ]
    if meaningful_tokens:
        return " ".join(meaningful_tokens[:4]), False
    return compact_summary(text, limit=40).lower() or "unknown", False


def _get_topic(
    scope: Scope,
    topics_by_label: dict[str, ChronologyTopic],
    topic_label: str,
    language: str,
    event: MemoryEvent,
    span: EvidenceSpan | None,
    created_at: datetime,
) -> ChronologyTopic:
    topic = topics_by_label.get(topic_label)
    if topic is not None:
        if span is not None and span.span_id not in topic.source_span_ids:
            topic.source_span_ids.append(span.span_id)
        return topic

    source_span_ids = [span.span_id] if span is not None else list(event.source_span_ids)
    topic = ChronologyTopic(
        topic_id=_id("chron_topic", scope, topic_label),
        scope=scope,
        canonical_label=topic_label,
        aliases=[],
        language=language,
        taxonomy_tags=[],
        source_span_ids=source_span_ids,
        confidence=event.confidence,
        created_at=created_at,
    )
    topics_by_label[topic_label] = topic
    return topic


def _get_phase(
    phases_by_key: dict[tuple[str, str], ChronologyPhase],
    topic: ChronologyTopic,
    phase_type: str,
    event: MemoryEvent,
    span: EvidenceSpan | None,
    created_at: datetime,
) -> ChronologyPhase:
    key = (topic.topic_id, phase_type)
    phase = phases_by_key.get(key)
    if phase is not None:
        if span is not None and span.span_id not in phase.source_span_ids:
            phase.source_span_ids.append(span.span_id)
        return phase

    source_span_ids = [span.span_id] if span is not None else list(event.source_span_ids)
    phase = ChronologyPhase(
        phase_id=_id("chron_phase", topic.topic_id, phase_type),
        topic_id=topic.topic_id,
        phase_type=phase_type,
        order_hint=PHASE_ORDER_HINTS[phase_type],
        source_span_ids=source_span_ids,
        confidence=event.confidence,
        created_at=created_at,
    )
    phases_by_key[key] = phase
    return phase


def _infer_action(text: str) -> str:
    lowered = text.lower()
    action_patterns = (
        "set up",
        "implemented",
        "implement",
        "started",
        "start",
        "initialize",
        "test",
        "verify",
        "完成",
        "开始",
        "测试",
    )
    for pattern in action_patterns:
        if _contains_phrase(lowered, pattern):
            return pattern
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    return tokens[0] if tokens else "unknown"


def _build_edges(nodes: list[ChronologyEventNode], created_at: datetime) -> list[ChronologyEventEdge]:
    edges: list[ChronologyEventEdge] = []
    for from_node, to_node in zip(nodes, nodes[1:]):
        if from_node.topic_id != to_node.topic_id:
            continue
        evidence_type = _edge_evidence_type(from_node, to_node)
        if evidence_type is None:
            continue
        source_span_ids = [
            span_id for span_id in (from_node.source_span_id, to_node.source_span_id) if span_id is not None
        ]
        edges.append(
            ChronologyEventEdge(
                edge_id=_id("chron_edge", from_node.node_id, to_node.node_id, evidence_type),
                from_node_id=from_node.node_id,
                to_node_id=to_node.node_id,
                edge_type="before",
                evidence_type=evidence_type,
                source_span_ids=source_span_ids,
                confidence=min(from_node.confidence, to_node.confidence),
                created_at=created_at,
            )
        )
    return edges


def _edge_evidence_type(from_node: ChronologyEventNode, to_node: ChronologyEventNode) -> str | None:
    if from_node.explicit_order_marker or to_node.explicit_order_marker:
        return "explicit_marker"
    if from_node.timestamp is not None and to_node.timestamp is not None and from_node.timestamp <= to_node.timestamp:
        return "timestamp_phase"
    return None


def _id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{stable_hash('|'.join(str(part) for part in parts))[:16]}"
