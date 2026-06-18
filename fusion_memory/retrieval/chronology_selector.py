from __future__ import annotations

from typing import Any

from fusion_memory.core.models import Candidate, Scope
from fusion_memory.core.text import keyword_score


def select_persisted_graph_event_ordering_candidates(
    query: str,
    scope: Scope,
    store: Any,
    limit: int,
    *,
    include_session: bool = False,
) -> tuple[list[Candidate], dict[str, object]]:
    topics = store.list_chronology_topics(scope, include_session=include_session)
    scored_topics = [
        (keyword_score(query, topic.canonical_label + " " + " ".join(topic.aliases)), topic)
        for topic in topics
    ]
    scored_topics = [(score, topic) for score, topic in scored_topics if score > 0]
    scored_topics.sort(key=lambda item: (-item[0], item[1].canonical_label))
    topic_ids = [topic.topic_id for _score, topic in scored_topics[:3]]
    if not topic_ids:
        return [], {"selected_driver": "none", "fallback_reason": "no_topic"}
    phases = {phase.phase_id: phase for phase in store.list_chronology_phases(topic_ids)}
    nodes = store.list_chronology_event_nodes(scope, include_session=include_session, topic_ids=topic_ids)
    nodes = _expand_relevant_nodes(query, scope, store, nodes, topic_ids=topic_ids, include_session=include_session)
    if not nodes:
        return [], {"selected_driver": "none", "fallback_reason": "no_nodes", "topic_ids": topic_ids}
    node_ids = [node.node_id for node in nodes]
    phase_ids = [node.phase_id for node in nodes if node.phase_id and node.phase_id not in phases]
    if phase_ids:
        phases.update({phase.phase_id: phase for phase in store.list_chronology_phases(list({node.topic_id for node in nodes if node.topic_id}))})
    edge_count_by_node: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in store.list_chronology_event_edges(node_ids):
        edge_count_by_node[edge.from_node_id] = edge_count_by_node.get(edge.from_node_id, 0) + 1
        edge_count_by_node[edge.to_node_id] = edge_count_by_node.get(edge.to_node_id, 0) + 1
    nodes.sort(
        key=lambda node: (
            node.timestamp is None,
            node.timestamp.isoformat() if node.timestamp else "",
            _phase_order(phases.get(node.phase_id)),
            node.node_id,
        )
    )
    candidates: list[Candidate] = []
    for index, node in enumerate(_dedupe_nodes(nodes), start=1):
        edge_count = edge_count_by_node.get(node.node_id, 0)
        score = 0.55 + keyword_score(query, f"{node.text} {node.action} {node.object}") + min(0.2, edge_count * 0.05)
        candidates.append(
            Candidate(
                id=node.node_id,
                type="event",
                text=_candidate_text(node, phases.get(node.phase_id)),
                source="event_ordering_persisted_graph",
                scores={
                    "score": score,
                    "graph_proximity": min(1.0, 0.5 + edge_count * 0.1),
                    "temporal_fit": 0.95 if node.timestamp else 0.55,
                },
                source_span_ids=[node.source_span_id] if node.source_span_id else [],
                metadata={
                    "graph_node_id": node.node_id,
                    "graph_topic_id": node.topic_id,
                    "graph_phase_id": node.phase_id,
                    "timeline_index": index,
                    "must_preserve_reason": ["graph_chronology_anchor"],
                    "evidence_role": "answer",
                },
            )
        )
    return candidates[:limit], {
        "selected_driver": "persisted_graph",
        "topic_ids": topic_ids,
        "node_count": len(nodes),
        "candidate_count": min(len(candidates), limit),
    }


def _expand_relevant_nodes(
    query: str,
    scope: Scope,
    store: Any,
    selected_nodes: list[Any],
    topic_ids: list[str],
    *,
    include_session: bool,
) -> list[Any]:
    if not selected_nodes:
        return selected_nodes
    selected_ids = {node.node_id for node in selected_nodes}
    expanded = list(selected_nodes)
    for node in store.list_chronology_event_nodes(scope, include_session=include_session, topic_ids=topic_ids):
        if node.node_id in selected_ids:
            continue
        relevance = keyword_score(query, f"{node.text} {node.action} {node.object}")
        if relevance <= 0 and not node.explicit_order_marker:
            continue
        expanded.append(node)
        selected_ids.add(node.node_id)
    return expanded


def _dedupe_nodes(nodes: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[tuple[str | None, str, str | None]] = set()
    for node in nodes:
        key = (node.source_span_id, node.text, node.topic_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def _phase_order(phase: Any | None) -> int:
    if phase is None:
        return 999
    order_hint = getattr(phase, "order_hint", None)
    if order_hint is not None:
        return int(order_hint)
    phase_type = str(getattr(phase, "phase_type", "") or "")
    return {
        "setup": 10,
        "decision": 20,
        "implementation": 30,
        "debug": 40,
        "validation": 50,
        "release": 60,
        "unknown": 900,
    }.get(phase_type, 500)


def _candidate_text(node: Any, phase: Any | None = None) -> str:
    action = str(getattr(node, "action", "") or "").strip()
    obj = str(getattr(node, "object", "") or "").strip()
    phase_type = str(getattr(phase, "phase_type", "") or "").strip()
    text = str(getattr(node, "text", "") or "")
    if action and obj and action != "unknown" and obj != "unknown":
        if text and (action.lower() in text.lower() or obj.lower() in text.lower()):
            return text
        prefix = f"{phase_type}: " if phase_type and phase_type != "unknown" else ""
        return f"{prefix}{action} {obj}"
    return text
