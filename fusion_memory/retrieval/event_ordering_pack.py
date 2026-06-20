from __future__ import annotations

"""Event-ordering model-view construction.

This module is a temporary home for event-ordering sequence heuristics that were
previously embedded in the eval adapter. Keep new product semantics upstream in
event extraction, event edges, or timeline graph selection. This module should
primarily serialize a topic-scoped timeline graph and use text heuristics only
as fallback while graph coverage is incomplete.
"""

# Legacy fallback: domain-specific event ordering rescue. Do not extend; migrate to taxonomy after graph parity.

from typing import Any

from fusion_memory.core.text import compact_summary
from fusion_memory.retrieval.event_ordering_common import _event_ordering_record_sort_key
from fusion_memory.retrieval.rule_registry import RuleDefinition, register_rule
from fusion_memory.retrieval.event_ordering_sequence import (
    _event_ordering_assistant_plan_text,
    _event_ordering_choose_sequence_items,
    _event_ordering_cluster_label,
    _event_ordering_compact_aspect_label,
    _event_ordering_phase_clusters,
    _event_ordering_raw_chronology_sequence_items,
    _event_ordering_select_milestones,
    _event_ordering_sequence_label,
    _event_ordering_sequence_output_sort_key,
    _event_ordering_structured_sequence_items,
)
from fusion_memory.retrieval.event_ordering_episodes import event_ordering_referenceable_episodes


register_rule(
    RuleDefinition(
        rule_id="event_ordering.legacy_rescue",
        module=__name__,
        purpose="track legacy event-ordering rescue path",
        category="high_risk",
        pattern="graph_fallback|legacy_fallback",
    )
)


def build_event_ordering_model_pack(
    *,
    query: str,
    source_spans: list[dict[str, Any]],
    events: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    contract_version: str,
    query_intent: dict[str, Any] | None = None,
    graph_coverage: dict[str, Any] | None = None,
    graph_shadow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compact_source_spans = _compact_event_ordering_records(source_spans, preferred_text_key="content")
    compact_events = _compact_event_ordering_records(events, preferred_text_key="description")
    timeline = _event_ordering_timeline(compact_events, compact_source_spans)
    anchor_timeline = [item for item in timeline if item.get("kind") == "user_introduced_aspect"]
    graph_timeline = _event_ordering_graph_timeline(anchor_timeline)
    phase_clusters = _event_ordering_phase_clusters(query, anchor_timeline)
    raw_sequence_items = _event_ordering_raw_chronology_sequence_items(query, compact_source_spans, anchor_timeline)
    referenceable_episodes = event_ordering_referenceable_episodes(query, compact_source_spans, anchor_timeline)
    structured_sequence_items, sequence_source = _event_ordering_structured_sequence_items(
        query,
        source_spans,
        compact_source_spans,
        anchor_timeline,
        phase_clusters,
    )
    sequence_items = _event_ordering_choose_sequence_items(
        query,
        structured_sequence_items,
        raw_sequence_items,
        sequence_source=sequence_source,
        anchor_timeline=anchor_timeline,
    )
    return {
        "pack_contract_version": contract_version,
        **({"sequence_items": sequence_items} if sequence_items else {}),
        **({"raw_chronology_items": raw_sequence_items} if raw_sequence_items else {}),
        **({"referenceable_episodes": referenceable_episodes} if referenceable_episodes else {}),
        **({"query_intent": query_intent} if query_intent else {}),
        **({"graph_coverage": graph_coverage} if graph_coverage else {}),
        **({"graph_shadow": graph_shadow} if graph_shadow else {}),
        **({"graph_timeline": graph_timeline} if graph_timeline else {}),
        "timeline": timeline,
        "anchor_timeline": anchor_timeline,
        "phase_clusters": phase_clusters,
        "context_turns": _event_ordering_context_turns(compact_source_spans),
        "event_hints": _event_ordering_event_hints(compact_events),
        "conflicts": conflicts[:10],
    }


def _event_ordering_graph_timeline(anchor_timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in anchor_timeline:
        copied = dict(item)
        copied["kind"] = "graph_candidate"
        out.append(copied)
    return out



def _compact_event_ordering_records(records: list[dict[str, Any]], *, preferred_text_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    limit = 30 if any(record.get("broad_raw_recall") or "broad_raw_recall" in str(record.get("candidate_source") or "") for record in records) else 20
    for record in records[:limit]:
        compacted: dict[str, Any] = {}
        for key in [
            "id",
            "timeline_index",
            "candidate_source",
            "source_span_ids",
            "event_type",
            "milestone_group",
            "speaker",
            "timeline_label",
            "timeline_role",
            "selector",
            "original_span_id",
            "source_uri",
            "turn_id",
            "timestamp",
            "aspect_key",
            "coverage_terms",
            "supports_timeline_index",
            "conversation_content",
            "broad_raw_recall",
            "recall_query",
        ]:
            if key in record:
                compacted[key] = record[key]
        text = str(record.get(preferred_text_key) or record.get("text") or record.get("content") or "")
        if text:
            compacted[preferred_text_key] = compact_summary(text, 1200)
        out.append(compacted)
    return out
def _event_ordering_timeline(events: list[dict[str, Any]], source_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    span_content_by_id = {
        str(span.get("id")): span.get("content")
        for span in source_spans
        if span.get("id") and span.get("content")
    }
    coverage_spans = [
        span
        for span in source_spans
        if span.get("selector") == "event_ordering_coverage"
        and span.get("timeline_role") in {"user_aspect_anchor", "user_introduced_topic"}
        and str(span.get("speaker") or "").lower() not in {"assistant", "agent"}
        and not _event_ordering_assistant_plan_text(str(span.get("conversation_content") or span.get("content") or ""))
        and span.get("timeline_index") is not None
    ]
    if coverage_spans:
        coverage_spans.sort(key=_event_ordering_record_sort_key)
        timeline = []
        support_by_index: dict[int, list[dict[str, Any]]] = {}
        for span in source_spans:
            support_index = span.get("supports_timeline_index")
            if support_index is None:
                continue
            try:
                support_by_index.setdefault(int(support_index), []).append(span)
            except (TypeError, ValueError):
                continue
        for span in coverage_spans:
            index = len(timeline) + 1
            item: dict[str, Any] = {
                "timeline_index": index,
                "kind": "user_introduced_aspect",
                "speaker": span.get("speaker"),
                "label": span.get("timeline_label"),
                "content": span.get("content"),
                "source_span_ids": [str(span.get("original_span_id") or span.get("id"))],
                "source_uri": span.get("source_uri"),
                "turn_id": span.get("turn_id"),
            }
            if span.get("candidate_source"):
                item["candidate_source"] = span.get("candidate_source")
            if span.get("broad_raw_recall"):
                item["broad_raw_recall"] = True
            if span.get("aspect_key"):
                item["aspect_key"] = span.get("aspect_key")
            if span.get("coverage_terms"):
                item["coverage_terms"] = span.get("coverage_terms")
            if span.get("conversation_content"):
                item["conversation_content"] = span.get("conversation_content")
            supports = [support.get("content") for support in support_by_index.get(int(span.get("timeline_index") or index), []) if support.get("content")]
            if supports:
                item["supporting_content"] = supports[:2]
            timeline.append(item)
            if len(timeline) >= 30:
                break
        return timeline
    if events:
        timeline: list[dict[str, Any]] = []
        ordered_events = [event for event in events if event.get("timeline_index") is not None]
        ordered_events.sort(key=_event_ordering_record_sort_key)
        for event in ordered_events:
            source_span_ids = [str(span_id) for span_id in event.get("source_span_ids", []) if span_id]
            item: dict[str, Any] = {
                "timeline_index": len(timeline) + 1,
                "kind": "event",
                "description": event.get("description") or event.get("text"),
                "event_type": event.get("event_type"),
                "milestone_group": event.get("milestone_group"),
                "source_span_ids": source_span_ids,
            }
            supporting_content = next((span_content_by_id[span_id] for span_id in source_span_ids if span_id in span_content_by_id), None)
            if supporting_content:
                item["supporting_content"] = supporting_content
            supporting_span = next((span for span in source_spans if str(span.get("id")) in source_span_ids), None)
            if supporting_span and supporting_span.get("candidate_source"):
                item["candidate_source"] = supporting_span.get("candidate_source")
            if supporting_span and supporting_span.get("broad_raw_recall"):
                item["broad_raw_recall"] = True
            timeline.append(item)
            if len(timeline) >= 30:
                break
        return timeline

    events_by_span: dict[str, list[dict[str, Any]]] = {}
    unmatched_events: list[dict[str, Any]] = []
    for event in events:
        event_record = {
            "description": event.get("description") or event.get("text"),
            "event_type": event.get("event_type"),
            "milestone_group": event.get("milestone_group"),
            "source_span_ids": event.get("source_span_ids", []),
        }
        span_ids = [str(span_id) for span_id in event.get("source_span_ids", []) if span_id]
        if not span_ids:
            unmatched_events.append({**event_record, "_sort_index": event.get("timeline_index")})
            continue
        for span_id in span_ids:
            events_by_span.setdefault(span_id, []).append(event_record)

    timeline: list[dict[str, Any]] = []
    spans = [span for span in source_spans if span.get("timeline_index") is not None]
    spans.sort(key=_event_ordering_record_sort_key)
    for span in spans:
        span_id = span.get("id")
        source_span_ids = [str(span_id)] if span_id else []
        turn_events = events_by_span.get(str(span_id), []) if span_id else []
        item: dict[str, Any] = {
            "timeline_index": len(timeline) + 1,
            "kind": "conversation_turn",
            "speaker": span.get("speaker"),
            "content": span.get("content"),
            "source_span_ids": source_span_ids,
        }
        if span.get("candidate_source"):
            item["candidate_source"] = span.get("candidate_source")
        if span.get("broad_raw_recall"):
            item["broad_raw_recall"] = True
        if turn_events:
            item["events"] = turn_events[:5]
        timeline.append(item)

    matched_span_ids = {str(span.get("id")) for span in spans if span.get("id")}
    for event in events:
        span_ids = [str(span_id) for span_id in event.get("source_span_ids", []) if span_id]
        if span_ids and any(span_id in matched_span_ids for span_id in span_ids):
            continue
        unmatched_events.append(
            {
                "description": event.get("description") or event.get("text"),
                "event_type": event.get("event_type"),
                "milestone_group": event.get("milestone_group"),
                "source_span_ids": event.get("source_span_ids", []),
                "_sort_index": event.get("timeline_index"),
            }
        )
    unmatched_events.sort(key=_event_ordering_record_sort_key)
    for event in unmatched_events:
        event.pop("_sort_index", None)
        event["timeline_index"] = len(timeline) + 1
        event["kind"] = "event"
        timeline.append(event)
        if len(timeline) >= 30:
            break
    return timeline[:30]


def _event_ordering_context_turns(source_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    ordered = [span for span in source_spans if span.get("timeline_index") is not None and span.get("content")]
    ordered.sort(key=lambda span: int(span.get("timeline_index") or 0))
    for span in ordered:
        if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor":
            continue
        item: dict[str, Any] = {
            "timeline_index": span.get("timeline_index"),
            "speaker": span.get("speaker"),
            "content": span.get("content"),
            "source_span_ids": span.get("source_span_ids") or [span.get("id")],
        }
        for key in ("selector", "timeline_role", "supports_timeline_index", "supports_aspect_key"):
            if span.get(key) is not None:
                item[key] = span.get(key)
        turns.append(item)
        if len(turns) >= 18:
            break
    return turns


def _event_ordering_event_hints(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    ordered = [event for event in events if event.get("description") or event.get("text")]
    ordered.sort(key=_event_ordering_record_sort_key)
    for event in ordered:
        hints.append(
            {
                "timeline_index": event.get("timeline_index"),
                "description": event.get("description") or event.get("text"),
                "event_type": event.get("event_type"),
                "milestone_group": event.get("milestone_group"),
                "source_span_ids": event.get("source_span_ids") or [],
            }
        )
        if len(hints) >= 18:
            break
    return hints
