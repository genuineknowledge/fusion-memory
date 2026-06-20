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
    try:
        topics = store.list_chronology_topics(scope, include_session=include_session)
        scored_topics = [
            (keyword_score(query, topic.canonical_label + " " + " ".join(topic.aliases)), topic)
            for topic in topics
        ]
        scored_topics = [(score, topic) for score, topic in scored_topics if score > 0]
        scored_topics.sort(key=lambda item: (-item[0], item[1].canonical_label))
        topic_ids = [topic.topic_id for _score, topic in scored_topics[:3]]
        if not topic_ids and topics:
            topic_ids = _topic_ids_from_node_relevance(
                query,
                scope,
                store,
                [topic.topic_id for topic in topics],
                include_session=include_session,
            )
        topic_ids, cluster_expanded_topic_ids = _expand_topic_ids_by_cluster_alias(topics, topic_ids)
        if not topic_ids:
            return [], {
                "selected_driver": "none",
                "fallback_reason": "no_topic",
                "cluster_expanded_topic_ids": [],
                "selected_topic_count": 0,
                "graph_ordered_legacy_recall_count": 0,
            }
        phases = {phase.phase_id: phase for phase in store.list_chronology_phases(topic_ids)}
        nodes = store.list_chronology_event_nodes(scope, include_session=include_session, topic_ids=topic_ids)
        nodes = _expand_relevant_nodes(query, scope, store, nodes, topic_ids=topic_ids, include_session=include_session)
        if not nodes:
            return [], {
                "selected_driver": "none",
                "fallback_reason": "no_nodes",
                "topic_ids": topic_ids,
                "cluster_expanded_topic_ids": cluster_expanded_topic_ids,
                "selected_topic_count": len(topic_ids),
                "graph_ordered_legacy_recall_count": 0,
            }
        node_ids = [node.node_id for node in nodes]
        if any(node.phase_id and node.phase_id not in phases for node in nodes):
            phases.update(
                {
                    phase.phase_id: phase
                    for phase in store.list_chronology_phases(list({node.topic_id for node in nodes if node.topic_id}))
                }
            )
        edges = list(store.list_chronology_event_edges(node_ids))
    except Exception as exc:
        if _is_missing_chronology_table_error(exc):
            _rollback_after_missing_chronology_table(store)
            return [], {
                "selected_driver": "none",
                "fallback_reason": "graph_unavailable",
                "error": type(exc).__name__,
                "cluster_expanded_topic_ids": [],
                "selected_topic_count": 0,
                "graph_ordered_legacy_recall_count": 0,
            }
        raise

    deduped_nodes = _dedupe_nodes(nodes)
    if len(deduped_nodes) < 2:
        return [], {
            "selected_driver": "none",
            "fallback_reason": "too_few_nodes",
            "topic_ids": topic_ids,
            "cluster_expanded_topic_ids": cluster_expanded_topic_ids,
            "selected_topic_count": len(topic_ids),
            "graph_ordered_legacy_recall_count": 0,
            "node_count": len(deduped_nodes),
        }

    edge_count_by_node: dict[str, int] = {node_id: 0 for node_id in node_ids}
    usable_edges = []
    for edge in edges:
        if edge.from_node_id not in edge_count_by_node or edge.to_node_id not in edge_count_by_node:
            continue
        usable_edges.append(edge)
        edge_count_by_node[edge.from_node_id] = edge_count_by_node.get(edge.from_node_id, 0) + 1
        edge_count_by_node[edge.to_node_id] = edge_count_by_node.get(edge.to_node_id, 0) + 1
    if not usable_edges:
        return [], {
            "selected_driver": "none",
            "fallback_reason": "no_edges",
            "topic_ids": topic_ids,
            "cluster_expanded_topic_ids": cluster_expanded_topic_ids,
            "selected_topic_count": len(topic_ids),
            "graph_ordered_legacy_recall_count": 0,
            "node_count": len(deduped_nodes),
        }
    edge_connected_ids = {node_id for node_id, count in edge_count_by_node.items() if count > 0}
    edge_connected_nodes = [node for node in deduped_nodes if node.node_id in edge_connected_ids]
    if len(edge_connected_nodes) >= 2:
        deduped_nodes = edge_connected_nodes

    source_span_ids = {node.source_span_id for node in deduped_nodes if node.source_span_id}
    for edge in usable_edges:
        source_span_ids.update(span_id for span_id in edge.source_span_ids if span_id)
    if len(source_span_ids) < 2:
        return [], {
            "selected_driver": "none",
            "fallback_reason": "weak_coverage",
            "topic_ids": topic_ids,
            "cluster_expanded_topic_ids": cluster_expanded_topic_ids,
            "selected_topic_count": len(topic_ids),
            "graph_ordered_legacy_recall_count": 0,
            "node_count": len(deduped_nodes),
            "edge_count": len(usable_edges),
            "source_span_count": len(source_span_ids),
        }

    deduped_nodes.sort(
        key=lambda node: (
            node.timestamp is None,
            node.timestamp.isoformat() if node.timestamp else "",
            _phase_order(phases.get(node.phase_id)),
            node.node_id,
        )
    )
    candidates: list[Candidate] = []
    for index, node in enumerate(deduped_nodes, start=1):
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
        "cluster_expanded_topic_ids": cluster_expanded_topic_ids,
        "selected_topic_count": len(topic_ids),
        "graph_ordered_legacy_recall_count": 0,
        "node_count": len(deduped_nodes),
        "edge_count": len(usable_edges),
        "source_span_count": len(source_span_ids),
        "candidate_count": min(len(candidates), limit),
    }


def _expand_topic_ids_by_cluster_alias(topics: list[Any], topic_ids: list[str]) -> tuple[list[str], list[str]]:
    selected_topic_ids = set(topic_ids)
    selected = {topic.topic_id for topic in topics if topic.topic_id in selected_topic_ids}
    selected_aliases = {
        str(alias).lower()
        for topic in topics
        if topic.topic_id in selected
        for alias in getattr(topic, "aliases", []) or []
    }
    expanded: list[str] = list(topic_ids)
    added: list[str] = []
    for topic in topics:
        if topic.topic_id in selected:
            continue
        aliases = {str(alias).lower() for alias in getattr(topic, "aliases", []) or []}
        if selected_aliases and selected_aliases & aliases:
            expanded.append(topic.topic_id)
            added.append(topic.topic_id)
    return list(dict.fromkeys(expanded)), added


def _topic_ids_from_node_relevance(
    query: str,
    scope: Scope,
    store: Any,
    topic_ids: list[str],
    *,
    include_session: bool,
) -> list[str]:
    nodes = store.list_chronology_event_nodes(scope, include_session=include_session, topic_ids=topic_ids)
    scored_by_topic: dict[str, float] = {}
    for node in nodes:
        topic_id = str(getattr(node, "topic_id", "") or "")
        if not topic_id:
            continue
        score = keyword_score(query, f"{node.text} {node.action} {node.object}")
        if score <= 0:
            continue
        scored_by_topic[topic_id] = max(scored_by_topic.get(topic_id, 0.0), score)
    return [
        topic_id
        for topic_id, _score in sorted(scored_by_topic.items(), key=lambda item: (-item[1], item[0]))[:3]
    ]


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
    eligible_nodes = store.list_chronology_event_nodes(scope, include_session=include_session, topic_ids=topic_ids)
    eligible_topic_ids = set(topic_ids)
    node_by_id = {node.node_id: node for node in eligible_nodes}
    for edge in store.list_chronology_event_edges(list(selected_ids)):
        if edge.from_node_id in selected_ids:
            node = node_by_id.get(edge.to_node_id)
        elif edge.to_node_id in selected_ids:
            node = node_by_id.get(edge.from_node_id)
        else:
            node = None
        if node is None or node.node_id in selected_ids or node.topic_id not in eligible_topic_ids:
            continue
        expanded.append(node)
        selected_ids.add(node.node_id)
    if len(_dedupe_nodes(expanded)) < 2:
        for node in eligible_nodes:
            if node.node_id in selected_ids:
                continue
            if not _continues_selected_timeline(node, expanded):
                continue
            expanded.append(node)
            selected_ids.add(node.node_id)
            if len(_dedupe_nodes(expanded)) >= 2:
                break
    for node in eligible_nodes:
        if node.node_id in selected_ids:
            continue
        relevance = keyword_score(query, f"{node.text} {node.action} {node.object}")
        if relevance <= 0 and not node.explicit_order_marker:
            continue
        expanded.append(node)
        selected_ids.add(node.node_id)
    if len(_dedupe_nodes(expanded)) < 2:
        _append_same_topic_timeline(expanded, selected_ids, eligible_nodes, topic_ids)
    elif _has_order_edges(store, list(selected_ids)):
        _append_same_topic_timeline(expanded, selected_ids, eligible_nodes, topic_ids)
    return expanded


def _append_same_topic_timeline(
    expanded: list[Any],
    selected_ids: set[str],
    eligible_nodes: list[Any],
    topic_ids: list[str],
) -> None:
    if not expanded:
        return
    selected_topic_ids = {node.topic_id for node in expanded if node.topic_id in set(topic_ids)}
    if not selected_topic_ids:
        return
    for node in eligible_nodes:
        if node.node_id in selected_ids or node.topic_id not in selected_topic_ids:
            continue
        if not _same_topic_timeline_node(node):
            continue
        expanded.append(node)
        selected_ids.add(node.node_id)


def _same_topic_timeline_node(node: Any) -> bool:
    return (
        getattr(node, "timestamp", None) is not None
        or bool(getattr(node, "explicit_order_marker", None))
        or bool(getattr(node, "source_span_id", None))
    )


def _has_order_edges(store: Any, node_ids: list[str]) -> bool:
    if not node_ids:
        return False
    try:
        return bool(store.list_chronology_event_edges(node_ids))
    except Exception:
        return False


def _continues_selected_timeline(node: Any, selected_nodes: list[Any]) -> bool:
    marker = str(getattr(node, "explicit_order_marker", "") or "").strip().lower()
    if marker not in {"then", "next", "after", "afterward", "afterwards", "subsequently", "later"}:
        return False
    timestamp = getattr(node, "timestamp", None)
    if timestamp is None:
        return False
    selected_timestamps = [getattr(selected, "timestamp", None) for selected in selected_nodes]
    selected_timestamps = [selected_timestamp for selected_timestamp in selected_timestamps if selected_timestamp is not None]
    return bool(selected_timestamps) and timestamp >= min(selected_timestamps)


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
    text = str(getattr(node, "text", "") or "")
    if _usable_object_label(obj):
        return obj
    if action and obj and action != "unknown" and obj != "unknown":
        return f"{action} {obj}"
    return text


def _usable_object_label(value: str) -> bool:
    if not value or value == "unknown":
        return False
    if len(value.split()) > 8:
        return False
    return True


def _is_missing_chronology_table_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "chronology_" not in message:
        return False
    return any(
        token in message
        for token in (
            "does not exist",
            "undefinedtable",
            "no such table",
            "missing table",
        )
    )


def _rollback_after_missing_chronology_table(store: Any) -> None:
    connect = getattr(store, "connect", None)
    if not callable(connect):
        return
    try:
        conn = connect()
        rollback = getattr(conn, "rollback", None)
        if callable(rollback):
            rollback()
    except Exception:
        return
