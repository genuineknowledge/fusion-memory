from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from typing import Any

from fusion_memory.core.models import Candidate, MemoryEvent
from fusion_memory.core.text import compact_summary, keyword_score, tokenize
from fusion_memory.retrieval.structured_annotations import select_event_ordering_timeline
from fusion_memory.retrieval.temporal_relations import (
    safe_temporal_relation_records,
    temporal_relation_summary_from_safe_records,
    temporal_relations_for_text,
)


@dataclass(frozen=True)
class ChronologyNode:
    node_id: str
    kind: str
    label: str
    timestamp: datetime | None
    source_span_id: str | None
    topic: str | None
    confidence: float


@dataclass(frozen=True)
class ChronologyEdge:
    source_id: str
    target_id: str
    kind: str
    confidence: float


@dataclass(frozen=True)
class ChronologyGraph:
    nodes: list[ChronologyNode]
    edges: list[ChronologyEdge]
    phases: list[str]
    topics: list[str]


_PHASE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("setup", ("setup", "set up", "initialize", "initial", "bootstrap", "foundation")),
    ("decision", ("decide", "decided", "choose", "chose", "picked", "opted", "settled on")),
    ("implementation", ("implement", "implemented", "build", "built", "add", "added", "configure", "configured")),
    ("debug", ("debug", "debugged", "fix", "fixed", "issue", "problem", "error", "troubleshoot")),
    ("validation", ("validate", "validated", "test", "tested", "verify", "verified", "coverage", "review")),
    ("release", ("release", "released", "deploy", "deployed", "ship", "shipped", "production", "launch")),
]


def build_event_chronology_graph(query: str, spans: list[Any], events: list[MemoryEvent]) -> ChronologyGraph:
    span_by_id = {str(getattr(span, "span_id", "") or ""): span for span in spans if getattr(span, "span_id", None)}
    ordered_events = sorted(events, key=lambda event: _event_sort_key(event, span_by_id))
    if not ordered_events:
        return ChronologyGraph(nodes=[], edges=[], phases=[], topics=[])

    nodes: list[ChronologyNode] = []
    for index, event in enumerate(ordered_events, start=1):
        node = _build_node(query, event, span_by_id, index)
        nodes.append(node)

    edges = _build_edges(nodes, ordered_events, span_by_id)
    phases = _unique_preserve_order(_phase_for_text(node.label) for node in nodes if _phase_for_text(node.label))
    topics = _unique_preserve_order(node.topic for node in nodes if node.topic)
    return ChronologyGraph(nodes=nodes, edges=edges, phases=phases, topics=topics)


def select_graph_first_event_ordering_candidates(
    query: str,
    spans: list[Any],
    events: list[MemoryEvent],
    limit: int,
) -> list[Candidate]:
    graph = build_event_chronology_graph(query, spans, events)
    graph_candidates = _graph_candidates(query, graph, events)
    if len(graph.nodes) < 2 or not graph.edges or not graph_candidates:
        return _wrap_legacy_candidates(select_event_ordering_timeline(query, spans, events, limit=limit), limit)

    if len(graph_candidates) >= limit:
        return graph_candidates[:limit]

    seen = {candidate.id for candidate in graph_candidates}
    fallback = _wrap_legacy_candidates(select_event_ordering_timeline(query, spans, events, limit=limit), limit)
    for candidate in fallback:
        if candidate.id in seen:
            continue
        graph_candidates.append(candidate)
        seen.add(candidate.id)
        if len(graph_candidates) >= limit:
            break
    return graph_candidates[:limit]


def _build_node(query: str, event: MemoryEvent, span_by_id: dict[str, Any], timeline_index: int) -> ChronologyNode:
    description = str(event.description or "")
    span = next((span_by_id.get(str(span_id)) for span_id in event.source_span_ids if span_id and str(span_id) in span_by_id), None)
    timestamp = event.time_start or getattr(span, "timestamp", None)
    source_span_id = next((str(span_id) for span_id in event.source_span_ids if span_id), None)
    topic = _topic_for_event(query, event, span)
    confidence = 0.55
    if timestamp is not None:
        confidence += 0.15
    if source_span_id:
        confidence += 0.1
    if topic:
        confidence += 0.1
    if _phase_for_text(description):
        confidence += 0.05
    if _has_explicit_sequence_marker(description):
        confidence += 0.05
    return ChronologyNode(
        node_id=event.event_id,
        kind="event",
        label=compact_summary(description, 180),
        timestamp=timestamp,
        source_span_id=source_span_id,
        topic=topic,
        confidence=min(1.0, confidence),
    )


def _build_edges(nodes: list[ChronologyNode], events: list[MemoryEvent], span_by_id: dict[str, Any]) -> list[ChronologyEdge]:
    edges: list[ChronologyEdge] = []
    seen: set[tuple[str, str, str]] = set()
    node_by_id = {node.node_id: node for node in nodes}
    for previous, current, event in _adjacent_event_pairs(events, span_by_id, node_by_id):
        current_text = current.label.lower()
        previous_text = previous.label.lower()
        if any(marker in current_text for marker in (" then ", " next ", " afterward ", " afterwards ", " subsequently ", " later ", " following that ")):
            _add_edge(edges, seen, previous.node_id, current.node_id, "then", 0.9)
        if "before" in current_text:
            _add_edge(edges, seen, previous.node_id, current.node_id, "before", 0.88)
        if any(marker in current_text for marker in ("update", "updated", "replac", "revise", "refine")):
            _add_edge(edges, seen, previous.node_id, current.node_id, "updates", 0.92)
        if any(marker in current_text for marker in ("replace", "replaced", "swap", "swapped")):
            _add_edge(edges, seen, previous.node_id, current.node_id, "replaces", 0.92)
        if current.topic and previous.topic and current.topic == previous.topic and current.node_id != previous.node_id:
            _add_edge(edges, seen, previous.node_id, current.node_id, "related", 0.72)
        if _high_confidence_timestamp_order(previous, current):
            _add_edge(edges, seen, previous.node_id, current.node_id, "before", 0.84)
        if event.time_start and previous.timestamp and current.timestamp and current.timestamp < previous.timestamp:
            _add_edge(edges, seen, current.node_id, previous.node_id, "before", 0.78)
    return edges


def _high_confidence_timestamp_order(previous: ChronologyNode, current: ChronologyNode) -> bool:
    if previous.timestamp is None or current.timestamp is None:
        return False
    if previous.timestamp >= current.timestamp:
        return False
    if not previous.source_span_id or not current.source_span_id:
        return False
    if _has_explicit_sequence_marker(current.label) or _has_explicit_sequence_marker(previous.label):
        return True
    if _phase_for_text(previous.label) and _phase_for_text(current.label):
        return True
    return False


def _adjacent_event_pairs(
    events: list[MemoryEvent],
    span_by_id: dict[str, Any],
    node_by_id: dict[str, ChronologyNode],
) -> list[tuple[ChronologyNode, ChronologyNode, MemoryEvent]]:
    ordered: list[tuple[MemoryEvent, ChronologyNode]] = []
    for event in events:
        node = node_by_id.get(event.event_id)
        if node is not None:
            ordered.append((event, node))
    ordered.sort(key=lambda item: _event_sort_key(item[0], span_by_id))
    out: list[tuple[ChronologyNode, ChronologyNode, MemoryEvent]] = []
    for (previous_event, previous_node), (current_event, current_node) in zip(ordered, ordered[1:]):
        out.append((previous_node, current_node, current_event))
    return out


def _graph_candidates(query: str, graph: ChronologyGraph, events: list[MemoryEvent]) -> list[Candidate]:
    if not graph.nodes:
        return []
    event_by_id = {event.event_id: event for event in events}
    edge_count_by_node: dict[str, int] = {node.node_id: 0 for node in graph.nodes}
    for edge in graph.edges:
        edge_count_by_node[edge.source_id] = edge_count_by_node.get(edge.source_id, 0) + 1
        edge_count_by_node[edge.target_id] = edge_count_by_node.get(edge.target_id, 0) + 1
    ranked: list[tuple[float, int, Candidate]] = []
    for index, node in enumerate(graph.nodes, start=1):
        event = event_by_id.get(node.node_id)
        if event is None:
            continue
        text = event.description
        temporal_relations = safe_temporal_relation_records(
            temporal_relations_for_text(
                node.label,
                query=query,
                source_span_id=node.source_span_id,
            )
        )
        graph_score = node.confidence + keyword_score(query, f"{node.label} {node.topic or ''}")
        graph_score += min(0.25, 0.05 * edge_count_by_node.get(node.node_id, 0))
        if node.timestamp is not None:
            graph_score += 0.1
        if _phase_for_text(text):
            graph_score += 0.05
        candidate = Candidate(
            id=node.node_id,
            type="event",
            text=text,
            source="event_ordering_graph_selector",
            scores={
                "score": round(graph_score, 4),
                "graph_proximity": round(min(1.0, 0.5 + 0.1 * edge_count_by_node.get(node.node_id, 0) + 0.1 * node.confidence), 4),
                "temporal_fit": 0.95 if node.timestamp is not None else 0.55,
                "bm25_score": keyword_score(query, text),
            },
            source_span_ids=list(event.source_span_ids),
            metadata={
                "graph_node_id": node.node_id,
                "graph_edge_count": edge_count_by_node.get(node.node_id, 0),
                "graph_phase": _phase_for_text(text),
                "graph_topic": node.topic,
                "timeline_index": index,
                "event_id": event.event_id,
                "temporal_relations": temporal_relations,
                "temporal_relation_summary": temporal_relation_summary_from_safe_records(temporal_relations),
            },
        )
        ranked.append((graph_score, index, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2].id))
    return [candidate for _score, _index, candidate in ranked]


def _wrap_legacy_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    out: list[Candidate] = []
    for index, candidate in enumerate(candidates, start=1):
        metadata = dict(candidate.metadata)
        metadata.setdefault("timeline_index", index)
        metadata["graph_fallback"] = True
        out.append(
            Candidate(
                id=candidate.id,
                type=candidate.type,
                text=candidate.text,
                source=f"event_ordering_graph_fallback_{candidate.source}",
                scores=dict(candidate.scores),
                source_span_ids=list(candidate.source_span_ids),
                metadata=metadata,
            )
        )
        if len(out) >= limit:
            break
    return out


def _event_sort_key(event: MemoryEvent, span_by_id: dict[str, Any]) -> tuple[int, str, str]:
    timestamp = event.time_start or _best_span_timestamp(event, span_by_id)
    return (
        0 if timestamp is not None else 1,
        timestamp.isoformat() if timestamp is not None else "",
        event.event_id,
    )


def _best_span_timestamp(event: MemoryEvent, span_by_id: dict[str, Any]) -> datetime | None:
    for span_id in event.source_span_ids:
        span = span_by_id.get(str(span_id))
        if span is not None and getattr(span, "timestamp", None) is not None:
            return getattr(span, "timestamp")
    return None


def _topic_for_event(query: str, event: MemoryEvent, span: Any | None) -> str | None:
    topic_tokens = [token for token in tokenize(" ".join(getattr(span, "topics", []) or [])) if len(token) > 3]
    if topic_tokens:
        return Counter(topic_tokens).most_common(1)[0][0]

    event_tokens = [token for token in tokenize(event.description or "") if len(token) > 3]
    if not event_tokens:
        return None

    query_tokens = {token for token in tokenize(query) if len(token) > 3}
    for token in event_tokens:
        if token in query_tokens:
            return token
    return Counter(event_tokens).most_common(1)[0][0]


def _phase_for_text(text: str) -> str:
    lower = text.lower()
    for phase, phrases in _PHASE_PATTERNS:
        if any(phrase in lower for phrase in phrases):
            return phase
    return ""


def _has_explicit_sequence_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in (" before ", " after ", " then ", " next ", " afterward ", " afterwards ", " subsequently "))


def _add_edge(edges: list[ChronologyEdge], seen: set[tuple[str, str, str]], source_id: str, target_id: str, kind: str, confidence: float) -> None:
    key = (source_id, target_id, kind)
    if source_id == target_id or key in seen:
        return
    seen.add(key)
    edges.append(ChronologyEdge(source_id=source_id, target_id=target_id, kind=kind, confidence=confidence))


def _unique_preserve_order(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
