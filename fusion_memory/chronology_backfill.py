from __future__ import annotations

from typing import Any

from fusion_memory.core.models import Scope
from fusion_memory.retrieval.chronology_normalizer import build_chronology_write_batch


def backfill_chronology_graph(
    store: Any,
    scope: Scope,
    *,
    include_session: bool = True,
) -> dict[str, Any]:
    spans = store.list_spans(scope, include_session=include_session)
    events = store.list_events(scope, include_session=include_session)
    grouped_spans = _spans_by_session(spans)
    grouped_events = _events_by_session(events)
    totals = {"topics": 0, "phases": 0, "nodes": 0, "edges": 0}
    telemetry: dict[str, Any] = {"groups": []}
    for session_key in sorted(grouped_events):
        session_events = grouped_events[session_key]
        session_spans = grouped_spans.get(session_key, [])
        session_scope = _scope_for_session(scope, session_events[0].scope)
        batch = build_chronology_write_batch(session_scope, session_spans, session_events)
        for topic in batch.topics:
            store.upsert_chronology_topic(topic)
        for phase in batch.phases:
            store.upsert_chronology_phase(phase)
        for node in batch.nodes:
            store.upsert_chronology_event_node(node)
        inserted_edges = 0
        for edge in batch.edges:
            inserted_edges += int(store.insert_chronology_event_edge(edge))
        totals["topics"] += len(batch.topics)
        totals["phases"] += len(batch.phases)
        totals["nodes"] += len(batch.nodes)
        totals["edges"] += inserted_edges
        telemetry["groups"].append(
            {
                "session_id": session_scope.session_id,
                "spans": len(session_spans),
                "events": len(session_events),
                "topics": len(batch.topics),
                "nodes": len(batch.nodes),
                "edges": inserted_edges,
            }
        )
    return {
        "status": "ok",
        "spans": len(spans),
        "events": len(events),
        "session_count": len(grouped_events),
        **totals,
        "telemetry": telemetry,
    }


def _scope_for_session(request_scope: Scope, event_scope: Scope) -> Scope:
    return Scope(
        workspace_id=request_scope.workspace_id,
        user_id=request_scope.user_id,
        agent_id=request_scope.agent_id,
        run_id=request_scope.run_id,
        session_id=event_scope.session_id,
        app_id=request_scope.app_id,
    )


def _events_by_session(events: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for event in events:
        key = str(event.scope.session_id or "")
        out.setdefault(key, []).append(event)
    return out


def _spans_by_session(spans: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for span in spans:
        key = str(span.scope.session_id or "")
        out.setdefault(key, []).append(span)
    return out
