from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fusion_memory.core.models import Candidate, MemoryEvent
from fusion_memory.core.text import compact_summary, keyword_score


GENERIC_EVENT_FACETS = {
    "user_introduced_aspect",
    "preference_change",
    "plan_step",
    "concern",
    "decision",
    "activity",
    "constraint",
    "request_for_comparison",
    "count_list_mention",
}


@dataclass(frozen=True)
class StructuredAnnotation:
    """Runtime annotation derived from raw evidence; no benchmark labels."""

    kind: str
    subject_key: str
    source_span_ids: list[str]
    label: str
    topic_key: str = ""
    role: str = ""
    polarity: str = "uncertain"
    value_mentions: list[str] = field(default_factory=list)
    date_roles: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimelineAspect:
    span: Any
    aspect_index: int
    label: str
    snippet: str
    aspect_key: str
    topic_key: str
    query_fit: float
    coverage_terms: list[str]
    origin: str = "raw"
    event_id: str = ""


def event_annotation_from_span(span: Any, *, query: str = "") -> StructuredAnnotation:
    text = str(getattr(span, "content", "") or "")
    topic_key = span_topic_key(span)
    subject = _subject_key(query, text, topic_key)
    return StructuredAnnotation(
        kind="timeline_event",
        subject_key=subject,
        source_span_ids=[str(getattr(span, "span_id", ""))],
        label=_event_label(text),
        topic_key=topic_key,
        role="user_introduced_topic" if getattr(span, "speaker", "") == "user" else "supporting_context",
        value_mentions=_value_mentions(text),
        date_roles=_date_roles(query, text),
        attributes={
            "speaker": getattr(span, "speaker", ""),
            "turn_id": getattr(span, "turn_id", ""),
            "source_uri": getattr(span, "source_uri", ""),
            "timestamp": getattr(getattr(span, "timestamp", None), "isoformat", lambda: None)(),
            "topic_score": keyword_score(query, text) if query else 0.0,
        },
    )


def event_annotation_from_event(event: MemoryEvent, *, source_span: Any | None = None, query: str = "") -> StructuredAnnotation:
    text = event.description
    topic_key = span_topic_key(source_span) if source_span is not None else ""
    return StructuredAnnotation(
        kind="timeline_event",
        subject_key=_subject_key(query, text, topic_key),
        source_span_ids=list(event.source_span_ids),
        label=_event_label(text),
        topic_key=topic_key,
        role="graph_event",
        value_mentions=_value_mentions(text),
        date_roles=_date_roles(query, text),
        attributes={
            "event_id": event.event_id,
            "event_type": event.event_type,
            "time_start": event.time_start.isoformat() if event.time_start else None,
            "source_uri": getattr(source_span, "source_uri", "") if source_span is not None else "",
            "turn_id": getattr(source_span, "turn_id", "") if source_span is not None else "",
        },
    )


def annotation_to_metadata(annotation: StructuredAnnotation) -> dict[str, Any]:
    return {
        "structured_annotation": {
            "kind": annotation.kind,
            "subject_key": annotation.subject_key,
            "source_span_ids": annotation.source_span_ids,
            "label": annotation.label,
            "topic_key": annotation.topic_key,
            "role": annotation.role,
            "polarity": annotation.polarity,
            "value_mentions": annotation.value_mentions,
            "date_roles": annotation.date_roles,
            "attributes": annotation.attributes,
        }
    }


def select_event_ordering_timeline(
    query: str,
    spans: list[Any],
    events: list[MemoryEvent] | None = None,
    *,
    limit: int = 20,
) -> list[Candidate]:
    """Select a canonical topic-scoped conversation chronology for ordering queries.

    This selector deliberately favors user-introduced turns. Assistant/event records
    are supporting context and should not define the main chronology unless user
    turns are absent.
    """

    events = events or []
    topic_keys = _topic_keys_for_query(query, spans, max_keys=1)
    scoped = [span for span in spans if not topic_keys or span_topic_key(span) in topic_keys]
    segment_span_ids = _topic_segment_span_ids(query, scoped)
    if segment_span_ids:
        scoped = [span for span in scoped if str(getattr(span, "span_id", "")) in segment_span_ids]
    answer_item_count = _query_item_limit(query)
    coverage_target = _timeline_coverage_target(query, limit)
    chronology_aspects = _chronology_user_aspects(query, scoped, limit=limit)
    if not chronology_aspects and topic_keys:
        broad_scoped = [span for span in spans if span_topic_key(span) in topic_keys]
        chronology_aspects = _chronology_user_aspects(query, broad_scoped, limit=limit)
        if chronology_aspects:
            scoped = broad_scoped
    primary_aspects = _merge_timeline_aspects(
        chronology_aspects,
        _coverage_user_aspects(query, scoped, limit=limit),
        limit=limit,
        desired=coverage_target,
        answer_item_count=answer_item_count,
        prefer_primary=True,
    )
    event_aspects = _coverage_event_aspects(query, scoped, events, limit=limit)
    if event_aspects and not chronology_aspects:
        primary_aspects = _merge_timeline_aspects(
            primary_aspects,
            event_aspects,
            limit=limit,
            desired=coverage_target,
            answer_item_count=answer_item_count,
            prefer_primary=True,
        )
    elif event_aspects and len(primary_aspects) < max(1, coverage_target):
        primary_aspects = _merge_timeline_aspects(
            primary_aspects,
            event_aspects,
            limit=limit,
            desired=coverage_target,
            answer_item_count=answer_item_count,
            prefer_primary=True,
        )
    primary_aspects = _ensure_query_required_event_facets(query, primary_aspects, scoped, limit=limit)
    candidates: list[Candidate] = []
    selected_span_ids: set[str] = set()
    for index, aspect in enumerate(primary_aspects, start=1):
        span = aspect.span
        selected_span_ids.add(str(getattr(span, "span_id", "")))
        annotation = event_annotation_from_span(span, query=query)
        topic_score = aspect.query_fit
        milestone_score = _timeline_signal(query, aspect.snippet)
        metadata = {
            "speaker": getattr(span, "speaker", ""),
            "span_type": getattr(span, "span_type", ""),
            "timestamp": getattr(getattr(span, "timestamp", None), "isoformat", lambda: None)(),
            "source_uri": getattr(span, "source_uri", ""),
            "turn_id": getattr(span, "turn_id", ""),
            "topic_group": aspect.topic_key,
            "topic_key": aspect.topic_key,
            "timeline_index": index,
            "timeline_role": "user_aspect_anchor",
            "timeline_label": aspect.label,
            "selector": "event_ordering_coverage",
            "original_span_id": str(getattr(span, "span_id", "")),
            "aspect_index": aspect.aspect_index,
            "aspect_key": aspect.aspect_key,
            "coverage_terms": aspect.coverage_terms,
            "coverage_origin": aspect.origin,
            "event_id": aspect.event_id,
            "conversation_content": str(getattr(span, "content", "") or ""),
        }
        metadata.update(annotation_to_metadata(annotation))
        metadata["structured_annotation"]["role"] = "user_aspect_anchor"
        metadata["structured_annotation"]["label"] = aspect.label
        metadata["structured_annotation"]["attributes"] = {
            **metadata["structured_annotation"].get("attributes", {}),
            "aspect_key": aspect.aspect_key,
            "coverage_terms": aspect.coverage_terms,
            "query_fit": aspect.query_fit,
            "coverage_origin": aspect.origin,
        }
        candidates.append(
            Candidate(
                id=(
                    f"{getattr(span, 'span_id')}#event{aspect.event_id}"
                    if aspect.event_id
                    else f"{getattr(span, 'span_id')}#aspect{aspect.aspect_index}"
                ),
                type="span",
                text=aspect.snippet,
                source="event_ordering_coverage",
                scores={
                    "semantic_score": topic_score,
                    "bm25_score": topic_score,
                    "temporal_fit": 1.0,
                    "graph_proximity": 1.0,
                    "topic_scope_score": topic_score,
                    "timeline_signal": milestone_score,
                    "speaker_prior": 1.0,
                    "coverage_score": 1.0 + topic_score,
                    "score": 2.0 + topic_score + milestone_score,
                },
                source_span_ids=[str(getattr(span, "span_id"))],
                metadata=metadata,
            )
        )

    support_limit = max(0, min(limit - len(candidates), max(2, len(primary_aspects) // 2)))
    candidates.extend(_assistant_support_candidates(query, scoped, selected_span_ids, primary_aspects, limit=support_limit))
    anchor_span_ids = {
        str(getattr(aspect.span, "span_id", ""))
        for aspect in primary_aspects
        if getattr(aspect.span, "span_id", "")
    }
    event_candidates = _supporting_event_candidates(
        query,
        events,
        topic_keys,
        allowed_source_span_ids=anchor_span_ids,
        limit=max(0, limit - len(candidates)),
    )
    return candidates + event_candidates


def _chronology_user_aspects(query: str, spans: list[Any], *, limit: int) -> list[TimelineAspect]:
    user_spans = [span for span in spans if getattr(span, "speaker", "") == "user"]
    user_spans.sort(key=span_sort_key)
    if not user_spans:
        return []
    pool: list[TimelineAspect] = []
    for span in user_spans:
        text = str(getattr(span, "content", "") or "")
        snippets = _span_timeline_aspect_snippets(query, text)
        if not snippets:
            if not _is_timeline_aspect(text):
                continue
            snippets = [compact_summary(text, max(160, 240))]
        elif not _text_has_explicit_aspect_list(text):
            snippets = snippets[:1]
        topic_key = span_topic_key(span)
        for aspect_index, snippet in enumerate(snippets, start=1):
            label = _aspect_label(snippet) or _aspect_label(text) or compact_summary(text, 80)
            aspect_key = _aspect_key(label) or _aspect_key(snippet) or _aspect_key(text)
            if not aspect_key:
                continue
            terms = sorted(_important_query_terms(query) & _important_query_terms(snippet))[:10]
            query_fit = keyword_score(query, snippet) + _timeline_signal(query, snippet)
            pool.append(
                TimelineAspect(
                    span=span,
                    aspect_index=aspect_index,
                    label=label,
                    snippet=snippet,
                    aspect_key=aspect_key,
                    topic_key=topic_key,
                    query_fit=query_fit,
                    coverage_terms=terms,
                    origin="raw_chronology",
                )
            )
    if not pool:
        return []
    desired = _timeline_coverage_target(query, limit, available=len(pool))
    if desired is None:
        desired = min(len(pool), max(10, min(18, limit)))
    desired = max(1, min(desired, limit, len(pool)))

    selected: list[TimelineAspect] = []
    seen_keys: set[str] = set()
    seen_span_ids: set[str] = set()
    for aspect in pool:
        span_id = str(getattr(aspect.span, "span_id", ""))
        if aspect.aspect_key in seen_keys or span_id in seen_span_ids:
            continue
        selected.append(aspect)
        seen_keys.add(aspect.aspect_key)
        seen_span_ids.add(span_id)
        if len(selected) >= desired:
            break

    if len(selected) < desired:
        for aspect in pool:
            span_id = str(getattr(aspect.span, "span_id", ""))
            if aspect.aspect_key in seen_keys or span_id in seen_span_ids:
                continue
            selected.append(aspect)
            seen_keys.add(aspect.aspect_key)
            seen_span_ids.add(span_id)
            if len(selected) >= desired:
                break

    selected.sort(key=lambda aspect: span_sort_key(aspect.span))
    return selected


def span_topic_key(span: Any) -> str:
    for value in (getattr(span, "source_uri", None), getattr(span, "turn_id", None)):
        if not value:
            continue
        text = str(value)
        match = re.match(r"^(beam:[^:]+:\d+):", text)
        if match:
            return match.group(1)
        if "#" in text:
            return text.split("#", 1)[0]
    return ""


def span_sort_key(span: Any) -> tuple[int, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...], str, str]:
    return (
        0 if getattr(span, "source_uri", None) or getattr(span, "turn_id", None) else 1,
        _natural_key(getattr(span, "source_uri", "")),
        _natural_key(getattr(span, "turn_id", "")),
        getattr(getattr(span, "timestamp", None), "isoformat", lambda: "")(),
        str(getattr(span, "span_id", "")),
    )


def _coverage_user_aspects(query: str, spans: list[Any], *, limit: int) -> list[TimelineAspect]:
    user_spans = [span for span in spans if getattr(span, "speaker", "") == "user"]
    user_spans.sort(key=span_sort_key)
    pool: list[TimelineAspect] = []
    for span in user_spans:
        topic_key = span_topic_key(span)
        for aspect_index, snippet in enumerate(
            _span_timeline_aspect_snippets(query, str(getattr(span, "content", "") or "")),
            start=1,
        ):
            if not _is_timeline_aspect(snippet):
                continue
            label = _aspect_label(snippet)
            aspect_key = _aspect_key(label)
            if not aspect_key:
                continue
            terms = sorted(_important_query_terms(query) & _important_query_terms(snippet))
            query_fit = keyword_score(query, snippet) + _timeline_signal(query, snippet)
            pool.append(
                TimelineAspect(
                    span=span,
                    aspect_index=aspect_index,
                    label=label,
                    snippet=snippet,
                    aspect_key=aspect_key,
                    topic_key=topic_key,
                    query_fit=query_fit,
                    coverage_terms=terms[:10],
                )
            )
    if not pool:
        return []
    desired = _timeline_coverage_target(query, limit, available=len(pool))
    if desired is None:
        desired = min(len(pool), max(10, min(18, limit)))
    desired = max(1, min(desired, limit, len(pool)))

    query_terms = _important_query_terms(query)
    selected: list[TimelineAspect] = []
    seen_keys: set[str] = set()
    seen_phases: set[str] = set()

    # First pass: topic scope is already fixed, so keep conversation order and
    # take distinct user-introduced aspect phases. This is coverage retrieval,
    # not top-k semantic retrieval.
    for aspect in pool:
        if aspect.aspect_key in seen_keys:
            continue
        phase = _aspect_phase_key(aspect.label, aspect.snippet)
        if phase and phase in seen_phases and len(pool) > desired:
            continue
        selected.append(aspect)
        seen_keys.add(aspect.aspect_key)
        if phase:
            seen_phases.add(phase)
        if len(selected) >= desired:
            break

    if len(selected) < desired:
        for aspect in pool:
            if aspect in selected or aspect.aspect_key in seen_keys:
                continue
            selected.append(aspect)
            seen_keys.add(aspect.aspect_key)
            phase = _aspect_phase_key(aspect.label, aspect.snippet)
            if phase:
                seen_phases.add(phase)
            if len(selected) >= desired:
                break
    if len(selected) < desired:
        for aspect in pool:
            if aspect in selected:
                continue
            selected.append(aspect)
            if len(selected) >= desired:
                break
    return selected


def _ensure_query_required_event_facets(
    query: str,
    aspects: list[TimelineAspect],
    spans: list[Any],
    *,
    limit: int,
) -> list[TimelineAspect]:
    return _ensure_generic_event_facets(query, aspects, spans, limit=limit)


def _ensure_generic_event_facets(
    query: str,
    aspects: list[TimelineAspect],
    spans: list[Any],
    *,
    limit: int,
) -> list[TimelineAspect]:
    selected = list(aspects)
    selected_keys = {aspect.aspect_key for aspect in selected}
    selected_span_ids = {str(getattr(aspect.span, "span_id", "")) for aspect in selected}

    answer_item_count = _query_item_limit(query)
    desired = _timeline_coverage_target(query, limit)
    if desired is None:
        desired = answer_item_count or len(selected)
    desired = max(len(selected), min(limit, max(1, desired)))
    if len(selected) >= desired:
        return _dedupe_near_duplicate_timeline_aspects(selected)[:limit]

    selected_phases = {_timeline_aspect_phase(aspect) for aspect in selected}
    candidates = _generic_event_facet_candidates(query, spans, selected_span_ids, selected_keys)
    candidates.sort(
        key=lambda aspect: (
            _generic_event_facet_score(query, aspect, selected_phases),
            _reverse_key(span_sort_key(aspect.span)),
        ),
        reverse=True,
    )

    for aspect in candidates:
        span_id = str(getattr(aspect.span, "span_id", ""))
        if span_id in selected_span_ids or aspect.aspect_key in selected_keys:
            continue
        selected.append(aspect)
        selected_keys.add(aspect.aspect_key)
        selected_span_ids.add(span_id)
        selected_phases.add(_timeline_aspect_phase(aspect))
        if len(selected) >= desired:
            break

    selected.sort(key=lambda aspect: span_sort_key(aspect.span))
    return _dedupe_near_duplicate_timeline_aspects(selected)[:limit]


def _dedupe_near_duplicate_timeline_aspects(aspects: list[TimelineAspect]) -> list[TimelineAspect]:
    out: list[TimelineAspect] = []
    for aspect in aspects:
        span_id = str(getattr(aspect.span, "span_id", ""))
        if any(
            span_id
            and span_id == str(getattr(existing.span, "span_id", ""))
            and _timeline_aspects_overlap(aspect, existing)
            for existing in out
        ):
            continue
        out.append(aspect)
    return out


def _timeline_aspects_overlap(left: TimelineAspect, right: TimelineAspect) -> bool:
    left_terms = _important_query_terms(f"{left.label} {left.snippet}") - _ASPECT_KEY_STOPWORDS
    right_terms = _important_query_terms(f"{right.label} {right.snippet}") - _ASPECT_KEY_STOPWORDS
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))
    return overlap >= 0.60


def _generic_event_facet_candidates(
    query: str,
    spans: list[Any],
    selected_span_ids: set[str],
    selected_keys: set[str],
) -> list[TimelineAspect]:
    candidates: list[TimelineAspect] = []
    user_spans = [span for span in spans if getattr(span, "speaker", "") == "user"]
    user_spans.sort(key=span_sort_key)
    for span in user_spans:
        span_id = str(getattr(span, "span_id", ""))
        if span_id in selected_span_ids:
            continue
        text = str(getattr(span, "content", "") or "")
        for aspect_index, snippet in enumerate(_span_timeline_aspect_snippets(query, text), start=1):
            if not _is_timeline_aspect(snippet):
                continue
            label = _aspect_label(snippet)
            label_key = _aspect_key(label)
            if not label_key:
                continue
            aspect_key = f"coverage_gap:{label_key}"
            if aspect_key in selected_keys:
                continue
            query_fit = keyword_score(query, snippet) + _timeline_signal(query, snippet)
            candidates.append(
                TimelineAspect(
                    span=span,
                    aspect_index=aspect_index,
                    label=label,
                    snippet=snippet,
                    aspect_key=aspect_key,
                    topic_key=span_topic_key(span),
                    query_fit=query_fit,
                    coverage_terms=sorted(_important_query_terms(query) & _important_query_terms(snippet))[:10],
                    origin="query_required_facet",
                )
            )
    return candidates


def _generic_event_facet_score(query: str, aspect: TimelineAspect, selected_phases: set[str]) -> float:
    text = f"{aspect.label} {aspect.snippet}"
    lower = text.lower()
    phase = _timeline_aspect_phase(aspect)
    score = aspect.query_fit + keyword_score(query, text)
    if phase and phase not in selected_phases:
        score += 0.35
    if re.search(r"\b(?:started|finished|completed|implemented|configured|created|added|fixed|reviewed|worked|changed|updated|decided|chose|asked|mentioned|focused)\b", lower):
        score += 0.25
    if re.search(r"\b(?:error|issue|problem|blocked|trouble|concern|question|compare|decision|result|progress|next)\b", lower):
        score += 0.12
    if _is_outline_or_schedule(lower):
        score -= 0.25
    return score


def _coverage_event_aspects(query: str, spans: list[Any], events: list[MemoryEvent], *, limit: int) -> list[TimelineAspect]:
    if not events:
        return []
    span_by_id = {str(getattr(span, "span_id", "")): span for span in spans}
    topic_keys = {span_topic_key(span) for span in spans if span_topic_key(span)}
    answer_item_count = _query_item_limit(query)
    desired = _timeline_coverage_target(query, limit)
    desired = max(1, min(desired, limit))
    group_support: dict[str, int] = {}
    for event in events:
        group = _event_group_label(event)
        if group:
            group_support[group] = group_support.get(group, 0) + 1

    scored: list[tuple[float, str, TimelineAspect]] = []
    for event in events:
        group = _event_group_label(event)
        if not group:
            continue
        source_span = _event_user_source_span(event, span_by_id)
        if source_span is None:
            continue
        topic_key = span_topic_key(source_span)
        if topic_keys and topic_key not in topic_keys:
            continue
        evidence = _event_evidence_text(event.description) or str(getattr(source_span, "content", "") or "")
        if not _is_timeline_aspect(evidence):
            continue
        label = _event_display_label(event, group)
        coverage_terms = sorted(_important_query_terms(query) & (_important_query_terms(label) | _important_query_terms(evidence)))[:10]
        query_fit = keyword_score(query, f"{label} {evidence}") + _timeline_signal(query, evidence)
        score = _event_aspect_quality(query, label, evidence, source_span, group_support.get(group, 1))
        aspect = TimelineAspect(
            span=source_span,
            aspect_index=1,
            label=label,
            snippet=f"{label}. Evidence: {evidence}",
            aspect_key=f"event:{_aspect_key(label) or group}",
            topic_key=topic_key,
            query_fit=query_fit,
            coverage_terms=coverage_terms,
            origin="event_graph",
            event_id=event.event_id,
        )
        scored.append((score, group, aspect))
    if not scored:
        return []

    selected_by_group: dict[str, tuple[float, TimelineAspect]] = {}
    for score, group, aspect in scored:
        current = selected_by_group.get(group)
        if current is None or score > current[0]:
            selected_by_group[group] = (score, aspect)
    selected = _select_phase_coverage_aspects(selected_by_group, desired, answer_item_count=answer_item_count)
    selected.sort(key=lambda aspect: span_sort_key(aspect.span))
    return selected


def _select_phase_coverage_aspects(
    group_best: dict[str, tuple[float, TimelineAspect]],
    desired: int,
    *,
    answer_item_count: int | None = None,
) -> list[TimelineAspect]:
    by_phase: dict[str, list[tuple[str, float, TimelineAspect]]] = {}
    for group, (score, aspect) in group_best.items():
        by_phase.setdefault(_event_phase_family(group), []).append((group, score, aspect))

    selected: list[TimelineAspect] = []
    used_groups: set[str] = set()
    used_labels: set[str] = set()
    used_span_ids: set[str] = set()
    for phase in _event_phase_order(answer_item_count or desired):
        options = by_phase.get(phase, [])
        if not options:
            continue
        ranked_options = sorted(
            options,
            key=lambda item: (
                _phase_group_preference(item[0], desired),
                item[1],
                _reverse_key(span_sort_key(item[2].span)),
            ),
            reverse=True,
        )
        if phase == "foundation" and desired > 3:
            ranked_options = sorted(
                options,
                key=lambda item: (
                    _foundation_coverage_preference(item[0], item[2]),
                    item[1],
                    _reverse_key(span_sort_key(item[2].span)),
                ),
                reverse=True,
            )
        choice = next(
            (
                item
                for item in ranked_options
                if _label_identity(item[2].label) not in used_labels
                and str(getattr(item[2].span, "span_id", "")) not in used_span_ids
            ),
            None,
        )
        if choice is None:
            continue
        group, _score, aspect = choice
        selected.append(aspect)
        used_groups.add(group)
        used_labels.add(_label_identity(aspect.label))
        used_span_ids.add(str(getattr(aspect.span, "span_id", "")))
        if len(selected) >= desired:
            return selected

    remaining = [
        (group, score, aspect)
        for group, (score, aspect) in group_best.items()
        if group not in used_groups
    ]
    remaining.sort(key=lambda item: (span_sort_key(item[2].span), -item[1]))
    for group, _score, aspect in remaining:
        span_id = str(getattr(aspect.span, "span_id", ""))
        if _label_identity(aspect.label) in used_labels or span_id in used_span_ids:
            continue
        selected.append(aspect)
        used_groups.add(group)
        used_labels.add(_label_identity(aspect.label))
        used_span_ids.add(span_id)
        if len(selected) >= desired:
            break
    return selected


def _label_identity(label: str) -> str:
    terms = sorted(_important_query_terms(label) - _ASPECT_KEY_STOPWORDS)
    return "-".join(terms[:8]) if terms else label.lower()[:80]


def _label_overlaps_existing(label: str, seen_labels: set[str]) -> bool:
    terms = set(_label_identity(label).split("-"))
    if not terms:
        return False
    for seen in seen_labels:
        seen_terms = set(seen.split("-"))
        if len(terms & seen_terms) / max(1, min(len(terms), len(seen_terms))) >= 0.60:
            return True
    return False


def _event_phase_order(desired: int) -> list[str]:
    if desired <= 3:
        return [
            "foundation",
            "activity",
            "problem_solving",
            "decision",
            "validation",
            "delivery",
            "risk_controls",
            "preference_change",
            "concern",
            "constraint",
            "plan_step",
            "user_introduced_aspect",
        ]
    return [
        "foundation",
        "activity",
        "problem_solving",
        "validation",
        "delivery",
        "risk_controls",
        "preference_change",
        "decision",
        "concern",
        "constraint",
        "plan_step",
        "user_introduced_aspect",
    ]


def _event_phase_family(group: str) -> str:
    facet = group.split(":", 1)[0] if ":" in group else group
    if facet in GENERIC_EVENT_FACETS:
        return facet
    return _generic_phase_from_text(group.replace("_", " ")) or group


def _phase_group_preference(group: str, desired: int) -> float:
    if group == "transaction_crud_implementation":
        return 0.96
    facet = group.split(":", 1)[0] if ":" in group else group
    if facet in GENERIC_EVENT_FACETS:
        return 0.72
    phase = _generic_phase_from_text(group.replace("_", " "))
    if phase == "foundation":
        return 0.92 if desired > 3 else 0.95
    if phase in {"problem_solving", "decision", "validation", "delivery"}:
        return 0.86
    if phase in {"risk_controls", "completion"}:
        return 0.80
    return 0.70


def _foundation_coverage_preference(group: str, aspect: TimelineAspect) -> float:
    text = f"{aspect.label} {aspect.snippet}".lower()
    phase = _generic_phase_from_text(f"{group} {text}")
    if phase == "foundation":
        return 1.0
    if phase == "problem_solving":
        return 0.94
    if re.search(r"\b(?:setup|set up|initialize|initial|plan|design|outline|requirements?|architecture|schema|configuration)\b", text):
        return 0.90
    return 0.70


def _generic_phase_from_text(text: str) -> str:
    lower = text.lower()
    patterns = [
        ("foundation", r"\b(?:setup|set up|initial|initialize|started|start|plan|planning|design|architecture|requirements?|schema|structure|baseline|foundation)\b"),
        ("problem_solving", r"\b(?:error|issue|problem|bug|fix|fixed|debug|blocked|failure|failed|exception|warning|trouble|troubleshoot)\b"),
        ("validation", r"\b(?:test|tests|testing|validate|validation|verify|verified|check|review|coverage|quality|accuracy|evaluation)\b"),
        ("delivery", r"\b(?:deploy|deployment|release|launch|ship|publish|production|rollout|finalize|finalizing)\b"),
        ("risk_controls", r"\b(?:security|privacy|access|permission|auth|authentication|authorization|password|compliance|risk|safe|safety)\b"),
        ("comparison", r"\b(?:compare|compared|comparison|versus|vs|between|difference|tradeoff)\b"),
        ("decision", r"\b(?:decided|decision|chose|choose|picked|selected|settled|opted)\b"),
        ("concern", r"\b(?:worried|worry|concern|concerns|uncertain|not sure|question)\b"),
        ("completion", r"\b(?:finished|completed|done|resolved|closed|wrapped)\b"),
    ]
    for phase, pattern in patterns:
        if re.search(pattern, lower):
            return phase
    return ""


def _merge_timeline_aspects(
    primary: list[TimelineAspect],
    fallback: list[TimelineAspect],
    *,
    limit: int,
    desired: int | None,
    answer_item_count: int | None = None,
    prefer_primary: bool = False,
) -> list[TimelineAspect]:
    target = desired or min(limit, max(len(primary), 10))
    target = max(1, min(target, limit))
    if not fallback:
        return primary[:target]
    if prefer_primary:
        out = list(primary[:target])
        used_span_ids = {str(getattr(aspect.span, "span_id", "")) for aspect in out}
        used_families = {_aspect_family(aspect.aspect_key, aspect.label) for aspect in out}
        used_labels = {_label_identity(aspect.label) for aspect in out}
        if len(out) < target and answer_item_count:
            for aspect in fallback:
                span_id = str(getattr(aspect.span, "span_id", ""))
                label_id = _label_identity(aspect.label)
                if span_id in used_span_ids or label_id in used_labels:
                    continue
                out.append(aspect)
                used_span_ids.add(span_id)
                used_families.add(_aspect_family(aspect.aspect_key, aspect.label))
                used_labels.add(label_id)
                if len(out) >= target:
                    break
        for aspect in fallback:
            span_id = str(getattr(aspect.span, "span_id", ""))
            family = _aspect_family(aspect.aspect_key, aspect.label)
            label_id = _label_identity(aspect.label)
            if span_id in used_span_ids or family in used_families or label_id in used_labels:
                continue
            out.append(aspect)
            used_span_ids.add(span_id)
            used_families.add(family)
            used_labels.add(label_id)
            if len(out) >= target:
                break
        out.sort(key=lambda aspect: span_sort_key(aspect.span))
        return out[:target]
    combined = list(primary) + list(fallback)
    if desired:
        fallback_phases = {_timeline_aspect_phase(aspect) for aspect in fallback}
        phase_floor = max(1, min(answer_item_count or target, len(fallback)))
        if len(fallback) >= (answer_item_count or target) and len(fallback_phases) >= phase_floor:
            out = list(fallback)
            if len(out) < target:
                used_ids = {str(getattr(aspect.span, "span_id", "")) for aspect in out}
                for aspect in primary:
                    span_id = str(getattr(aspect.span, "span_id", ""))
                    if span_id in used_ids:
                        continue
                    out.append(aspect)
                    used_ids.add(span_id)
                    if len(out) >= target:
                        break
            out.sort(key=lambda aspect: span_sort_key(aspect.span))
            return out[:target]
        ranked_by_phase: dict[str, TimelineAspect] = {}
        phase_order: list[str] = []
        for aspect in combined:
            phase = _timeline_aspect_phase(aspect)
            if phase not in ranked_by_phase:
                ranked_by_phase[phase] = aspect
                phase_order.append(phase)
                continue
            current = ranked_by_phase[phase]
            if _timeline_aspect_quality(aspect) > _timeline_aspect_quality(current):
                ranked_by_phase[phase] = aspect
        out = [ranked_by_phase[phase] for phase in phase_order]
        if len(out) < target:
            used_ids = {str(getattr(aspect.span, "span_id", "")) for aspect in out}
            used_keys = {_aspect_family(aspect.aspect_key, aspect.label) for aspect in out}
            for aspect in combined:
                span_id = str(getattr(aspect.span, "span_id", ""))
                family = _aspect_family(aspect.aspect_key, aspect.label)
                if span_id in used_ids or family in used_keys:
                    continue
                out.append(aspect)
                used_ids.add(span_id)
                used_keys.add(family)
                if len(out) >= target:
                    break
        out.sort(key=lambda aspect: span_sort_key(aspect.span))
        return out[:target]
    out = list(primary)
    seen = {_aspect_family(aspect.aspect_key, aspect.label) for aspect in out}
    seen_labels = {_label_identity(aspect.label) for aspect in out}
    seen_span_ids = {str(getattr(aspect.span, "span_id", "")) for aspect in out}
    for aspect in fallback:
        family = _aspect_family(aspect.aspect_key, aspect.label)
        label_id = _label_identity(aspect.label)
        span_id = str(getattr(aspect.span, "span_id", ""))
        if span_id in seen_span_ids or family in seen or label_id in seen_labels or _label_overlaps_existing(aspect.label, seen_labels):
            continue
        out.append(aspect)
        seen.add(family)
        seen_labels.add(label_id)
        seen_span_ids.add(span_id)
        if len(out) >= target:
            break
    out.sort(key=lambda aspect: span_sort_key(aspect.span))
    return out[:target]


def _event_user_source_span(event: MemoryEvent, span_by_id: dict[str, Any]) -> Any | None:
    for span_id in event.source_span_ids:
        span = span_by_id.get(str(span_id))
        if span is not None and getattr(span, "speaker", "") == "user":
            return span
    return None


def _timeline_aspect_phase(aspect: TimelineAspect) -> str:
    phase = _aspect_phase_key(aspect.label, aspect.snippet)
    if phase:
        return phase
    if aspect.origin == "event_graph" and aspect.aspect_key.startswith("event:"):
        return _event_phase_family(aspect.aspect_key.removeprefix("event:"))
    return _aspect_family(aspect.aspect_key, aspect.label)


def _timeline_aspect_quality(aspect: TimelineAspect) -> float:
    quality = aspect.query_fit
    if aspect.origin == "event_graph":
        quality += 0.45
    label = f"{aspect.label} {aspect.snippet}".lower()
    if _generic_phase_from_text(label) in {"foundation", "problem_solving", "validation", "delivery", "risk_controls", "decision"}:
        quality += 0.22
    if re.search(r"\b(?:review my code|virtual environment|not sure how|schedule|deadline)\b", label):
        quality -= 0.18
    return quality


def _event_group_label(event: MemoryEvent) -> str:
    match = re.search(r"\bMilestone\s+\[([^\]]+)\]", event.description or "")
    if match:
        return match.group(1).strip().lower()
    match = re.search(r"\bFacet\s+\[([^\]]+)\]", event.description or "")
    if match:
        facet = match.group(1).strip().lower()
        label = _event_label_field(event.description)
        label_key = _aspect_key(label) if label else ""
        return f"{facet}:{label_key}" if label_key else facet
    if event.event_type in GENERIC_EVENT_FACETS:
        label = _event_label(event.description)
        label_key = _aspect_key(label)
        return f"{event.event_type}:{label_key}" if label_key else event.event_type
    if event.event_type and event.event_type != "unknown":
        label = _event_label(event.description)
        return _aspect_key(label) or event.event_type
    return ""


def _humanize_event_group(group: str) -> str:
    if ":" in group:
        facet, label = group.split(":", 1)
        human = re.sub(r"\s+", " ", label.replace("-", " ")).strip().title()
        return human or facet.replace("_", " ").title()
    return re.sub(r"\s+", " ", group.replace("_", " ")).strip().title()


def _event_display_label(event: MemoryEvent, group: str) -> str:
    label = _event_label_field(event.description)
    if label:
        return label[:140]
    return _humanize_event_group(group)


def _event_label_field(description: str) -> str:
    match = re.search(r"\bLabel:\s*(.+?)(?:\.\s*Evidence:|\s*Evidence:|$)", description or "", flags=re.I | re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" .:-")


def _event_evidence_text(description: str) -> str:
    match = re.search(r"\bEvidence:\s*(.+)$", description or "", flags=re.I | re.S)
    text = match.group(1) if match else description
    return re.sub(r"\s+", " ", text).strip()[:600]


def _event_aspect_quality(query: str, label: str, evidence: str, span: Any, support_count: int) -> float:
    lower = evidence.lower()
    score = 0.35 + keyword_score(query, f"{label} {evidence}") + min(0.25, 0.04 * support_count)
    if _action_signal(lower):
        score += 0.25
    if _is_outline_or_schedule(lower):
        score -= 0.35
    if re.search(r"\b(?:finalizing|completed|currently working|working on|trying to implement|trying to fix|having trouble|error|exception|deploy|deployment|security|test|coverage)\b", lower):
        score += 0.18
    if getattr(span, "speaker", "") == "user":
        score += 0.12
    return score


def _action_signal(lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:i[' ]?m|i am|i've|i have|i need|i want|i'm currently|trying|working|implemented|completed|finalizing|configuring|fixing|reviewing|switching|testing)\b",
            lower,
        )
    )


def _is_outline_or_schedule(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:break it down|components:|nov\s+\d+|dec\s+\d+|jan\s+\d+|feb\s+\d+|schedule|timeline|sprint plan|time anchor)\b", lower)
        or lower.count(" - ") >= 3
    )


def _aspect_family(aspect_key: str, label: str) -> str:
    key = aspect_key
    if key.startswith("event:"):
        key = key[6:]
    facet = key.split(":", 1)[0] if ":" in key else key
    if facet in GENERIC_EVENT_FACETS:
        return facet
    phase = _generic_phase_from_text(f"{key} {label}".replace("-", " "))
    if phase:
        return phase
    terms = set(key.split("-")) | (_important_query_terms(label) - _ASPECT_KEY_STOPWORDS)
    return "-".join(sorted(terms)[:4]) if terms else key


def _assistant_support_candidates(
    query: str,
    spans: list[Any],
    selected_span_ids: set[str],
    primary_aspects: list[TimelineAspect],
    *,
    limit: int,
) -> list[Candidate]:
    if limit <= 0 or not primary_aspects:
        return []
    support_spans = [
        span
        for span in spans
        if getattr(span, "speaker", "") in {"assistant", "agent"}
        and str(getattr(span, "span_id", "")) not in selected_span_ids
    ]
    support_spans.sort(key=span_sort_key)
    selected: list[Candidate] = []
    seen: set[str] = set()
    for aspect_index, aspect in enumerate(primary_aspects, start=1):
        support = _nearest_following_support(aspect.span, support_spans, seen)
        if support is None:
            continue
        seen.add(str(getattr(support, "span_id", "")))
        annotation = event_annotation_from_span(support, query=query)
        metadata = {
            "speaker": getattr(support, "speaker", ""),
            "span_type": getattr(support, "span_type", ""),
            "timestamp": getattr(getattr(support, "timestamp", None), "isoformat", lambda: None)(),
            "source_uri": getattr(support, "source_uri", ""),
            "turn_id": getattr(support, "turn_id", ""),
            "topic_group": span_topic_key(support),
            "topic_key": span_topic_key(support),
            "timeline_role": "supporting_context",
            "timeline_label": _event_label(str(getattr(support, "content", "") or "")),
            "selector": "event_ordering_coverage",
            "supports_timeline_index": aspect_index,
            "supports_aspect_key": aspect.aspect_key,
        }
        metadata.update(annotation_to_metadata(annotation))
        metadata["structured_annotation"]["role"] = "supporting_context"
        selected.append(
            Candidate(
                id=str(getattr(support, "span_id")),
                type="span",
                text=str(getattr(support, "content", "")),
                source="event_ordering_coverage_support",
                scores={
                    "semantic_score": keyword_score(query, str(getattr(support, "content", "") or "")),
                    "bm25_score": keyword_score(query, str(getattr(support, "content", "") or "")),
                    "temporal_fit": 0.2,
                    "graph_proximity": 0.65,
                    "speaker_prior": 0.25,
                    "score": 0.7,
                },
                source_span_ids=[str(getattr(support, "span_id"))],
                metadata=metadata,
            )
        )
        if len(selected) >= limit:
            break
    return selected


def _nearest_following_support(anchor: Any, support_spans: list[Any], seen: set[str]) -> Any | None:
    anchor_key = span_sort_key(anchor)
    for span in support_spans:
        span_id = str(getattr(span, "span_id", ""))
        if span_id in seen:
            continue
        if span_topic_key(span) != span_topic_key(anchor):
            continue
        if span_sort_key(span) >= anchor_key:
            return span
    return None


def _topic_keys_for_query(query: str, spans: list[Any], *, max_keys: int) -> set[str]:
    query_anchors = _topic_anchor_phrases(query)
    query_terms = _important_query_terms(query) - _ASPECT_KEY_STOPWORDS
    scored: dict[str, float] = {}
    group_best_anchor: dict[str, float] = {}
    group_best_single: dict[str, float] = {}
    for span in spans:
        key = span_topic_key(span)
        if not key:
            continue
        text = str(getattr(span, "content", "") or "")
        anchor_score = _topic_anchor_score(query, text, query_anchors=query_anchors, query_terms=query_terms)
        lexical_score = keyword_score(query, text) + _timeline_signal(query, text)
        score = (1.8 * anchor_score) + min(0.35, lexical_score)
        if getattr(span, "speaker", "") == "user":
            score += 0.18
        if score <= 0:
            continue
        scored[key] = scored.get(key, 0.0) + score
        group_best_anchor[key] = max(group_best_anchor.get(key, 0.0), anchor_score)
        group_best_single[key] = max(group_best_single.get(key, 0.0), score)
    ranked = sorted(
        scored,
        key=lambda key: (group_best_anchor.get(key, 0.0), group_best_single.get(key, 0.0), scored[key]),
        reverse=True,
    )
    return set(ranked[:max_keys])


def _topic_segment_span_ids(query: str, spans: list[Any]) -> set[str]:
    """Return the contiguous topic segment anchored by the query inside a chat.

    BEAM chats often contain several unrelated tasks in one conversation file.
    Event ordering should use conversation chronology inside the queried topic,
    not the entire chat chronology.
    """

    user_spans = [span for span in spans if getattr(span, "speaker", "") == "user"]
    user_spans.sort(key=span_sort_key)
    if len(user_spans) <= 1:
        return set()
    anchors = _topic_anchor_phrases(query)
    query_terms = _important_query_terms(query) - _ASPECT_KEY_STOPWORDS
    scored: list[tuple[int, Any, float]] = []
    for index, span in enumerate(user_spans):
        score = _topic_anchor_score(
            query,
            str(getattr(span, "content", "") or ""),
            query_anchors=anchors,
            query_terms=query_terms,
        )
        if score > 0:
            scored.append((index, span, score))
    if not scored:
        return set()
    best_index, _best_span, best_score = max(scored, key=lambda item: item[2])
    if best_score < 0.30:
        return set()

    keep_user_ids: set[str] = set()
    low_gap = 0
    started = False
    broad_chronology = _query_needs_broad_topic_chronology(query, anchors)
    if broad_chronology:
        anchor_threshold = max(0.18, min(0.45, best_score * 0.45))
        for _index, span in enumerate(user_spans):
            text = str(getattr(span, "content", "") or "")
            score = _topic_anchor_score(query, text, query_anchors=anchors, query_terms=query_terms)
            if not started:
                if score >= anchor_threshold:
                    started = True
                    keep_user_ids.add(str(getattr(span, "span_id", "")))
                continue
            if score >= 0.08 or _broad_topic_continuation_signal(text):
                keep_user_ids.add(str(getattr(span, "span_id", "")))
                low_gap = 0
                continue
            low_gap += 1
            if low_gap > 2:
                break
        if len(keep_user_ids) >= 2:
            support_ids = set(keep_user_ids)
            user_positions = {str(getattr(span, "span_id", "")): index for index, span in enumerate(user_spans)}
            for span in spans:
                if getattr(span, "speaker", "") == "user":
                    continue
                turn_key = span_sort_key(span)
                previous = [
                    user_span
                    for user_span in user_spans
                    if span_sort_key(user_span) <= turn_key
                    and str(getattr(user_span, "span_id", "")) in keep_user_ids
                ]
                if previous:
                    last_user = previous[-1]
                    if user_positions.get(str(getattr(last_user, "span_id", "")), -1) >= 0:
                        support_ids.add(str(getattr(span, "span_id", "")))
            return support_ids
    pre_anchor_threshold = 0.08 if broad_chronology else max(0.30, best_score * 0.45)
    for index, span in enumerate(user_spans):
        score = _topic_anchor_score(
            query,
            str(getattr(span, "content", "") or ""),
            query_anchors=anchors,
            query_terms=query_terms,
        )
        if index < best_index and score < pre_anchor_threshold:
            continue
        if score >= 0.18:
            started = True
            low_gap = 0
            keep_user_ids.add(str(getattr(span, "span_id", "")))
            continue
        if started and index > best_index:
            if _continuation_timeline_signal(str(getattr(span, "content", "") or "")):
                keep_user_ids.add(str(getattr(span, "span_id", "")))
                low_gap = 0
                continue
            low_gap += 1
            if low_gap > 2:
                break

    if len(keep_user_ids) < 2:
        keep_user_ids = {str(getattr(span, "span_id", "")) for _index, span, score in scored if score >= max(0.18, best_score * 0.45)}
    if not keep_user_ids:
        return set()

    user_positions = {str(getattr(span, "span_id", "")): index for index, span in enumerate(user_spans)}
    support_ids = set(keep_user_ids)
    for span in spans:
        if getattr(span, "speaker", "") == "user":
            continue
        turn_key = span_sort_key(span)
        previous = [
            user_span
            for user_span in user_spans
            if span_sort_key(user_span) <= turn_key
            and str(getattr(user_span, "span_id", "")) in keep_user_ids
        ]
        if previous:
            last_user = previous[-1]
            if user_positions.get(str(getattr(last_user, "span_id", "")), -1) >= best_index - 1:
                support_ids.add(str(getattr(span, "span_id", "")))
    return support_ids


def _split_aspect_mentions(text: str) -> list[str]:
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    items: list[str] = []
    for line in lines:
        line = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
        bullet = re.match(r"^(?:[-*+]|\d+[.)]|[A-Za-z][.)])\s+(.{4,})$", line)
        if bullet:
            items.append(bullet.group(1).strip())
            continue
        if re.match(r"^[A-Z][A-Za-z0-9 /&+-]{2,48}:\s+.{4,}$", line):
            items.append(line)
    if items:
        return _clean_aspect_items(items)
    sentence_items = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(sentence_items) > 1:
        expanded: list[str] = []
        for item in sentence_items:
            expanded.extend(_split_compound_aspect(item))
        return _clean_aspect_items(expanded)
    return _clean_aspect_items(_split_compound_aspect(cleaned))


def _span_timeline_aspect_snippets(query: str, text: str) -> list[str]:
    snippets = [snippet for snippet in _split_aspect_mentions(text) if _is_timeline_aspect(snippet)]
    if len(snippets) <= 1 or _text_has_explicit_aspect_list(text):
        return snippets
    scored = [
        (
            keyword_score(query, snippet)
            + _timeline_signal(query, snippet)
            + _single_turn_aspect_signal(snippet),
            index,
            snippet,
        )
        for index, snippet in enumerate(snippets)
        if not _is_low_information_conversation_turn(snippet)
    ]
    if not scored:
        return []
    best = max(scored, key=lambda item: (item[0], -item[1]))
    return [best[2]]


def _text_has_explicit_aspect_list(text: str) -> bool:
    if re.search(r"(?:^|\n)\s*(?:[-*+]|\d+[.)]|[A-Za-z][.)])\s+.{4,}", text):
        return True
    if re.search(r"(?:^|\n)\s*[A-Z][A-Za-z0-9 /&+-]{2,48}:\s+.{4,}", text):
        return True
    return False


def _single_turn_aspect_signal(text: str) -> float:
    lower = text.lower()
    score = 0.0
    if re.search(r"\b(?:started|using|created|built|implemented|configured|fixed|reviewed|decided|chose|selected|set up|setup|asked about|compared|planned)\b", lower):
        score += 0.25
    if re.search(r"\b(?:i|we)\s+(?:want|wanted|need|needed|would like)\b", lower):
        score += 0.18
    if re.search(r"\b(?:thanks|thank you|sounds good|makes sense|good idea)\b", lower):
        score -= 0.35
    if re.search(r"\b(?:question|problem|issue|concern|goal|budget|deadline|error|test|deploy|security|preference|dropdown|debounce|limit|constraint)\b", lower):
        score += 0.10
    return score


def _split_compound_aspect(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        return [text]
    parts = re.split(
        r"\s+(?:and then|then|also|plus|as well as|along with|, and|;)\s+",
        text,
        flags=re.I,
    )
    cleaned = [part.strip(" ,;:-") for part in parts if len(part.strip(" ,;:-")) >= 18]
    if len(cleaned) <= 1:
        return [text]
    prefix_match = re.match(r"^((?:i|we)\s+(?:brought up|mentioned|discussed|asked about|wanted|needed|planned|decided|worked on)\s+)", text, flags=re.I)
    prefix = prefix_match.group(1) if prefix_match else ""
    out: list[str] = []
    for index, part in enumerate(cleaned):
        if index > 0 and prefix and not re.match(r"^(?:i|we)\b", part, flags=re.I):
            out.append(prefix + part)
        else:
            out.append(part)
    return out


def _aspect_label(text: str) -> str:
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)]|[A-Za-z][.)])\s+", "", cleaned)
    cleaned = re.sub(
        r"^\s*(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b|20\d{2}[-/]\d{1,2}[-/]\d{1,2})\s*(?:-|to|–|—)\s*(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b|20\d{2}[-/]\d{1,2}[-/]\d{1,2})\s*:\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -;")
    if not cleaned:
        return ""
    return cleaned[:140]


def _is_timeline_aspect(text: str) -> bool:
    lower = text.lower()
    if len(_important_query_terms(text) - _ASPECT_KEY_STOPWORDS) < 1:
        return False
    if _is_low_information_conversation_turn(text):
        return False
    meta_patterns = [
        r"\btime anchor\b",
        r"\bcan you help me (?:create|make|plan)\b",
        r"\bneed to plan my tasks\b",
        r"\bcreate a schedule\b",
        r"\bensure i meet my deadline\b",
    ]
    if any(re.search(pattern, lower) for pattern in meta_patterns):
        return False
    if re.search(r"\b(?:schedule|timeline|deadline)\b", lower) and not re.search(
        r"\b(?:implement|setup|set up|design|connect|auth|authentication|crud|deploy|deployment|test|security|schema|database|ui|ux|feature|component|library|version|error|fix|configure|optimize|integrat)\b",
        lower,
    ):
        return False
    return True


def _clean_aspect_items(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", item).strip(" -;")
        if len(text) < 8:
            continue
        if re.fullmatch(r"(?:sure|thanks|okay|ok|yes|no|great)[.!]?", text, flags=re.I):
            continue
        if _is_low_information_conversation_turn(text):
            continue
        out.append(text[:600])
    return list(dict.fromkeys(out))


def _is_low_information_conversation_turn(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip(" .!?,-;:").lower()
    if not normalized:
        return True
    if len(normalized.split()) <= 3 and re.fullmatch(
        r"(?:ok(?:ay)?|cool|great|sure|thanks|thank you|sounds good|sounds great|good idea|nice)",
        normalized,
    ):
        return True
    if re.fullmatch(
        r"(?:ok(?:ay)?|cool|great|sure|thanks|thank you|thanks for (?:the )?[^.!?]{2,40})",
        normalized,
    ):
        return True
    if re.fullmatch(r"(?:that|this|it) sounds (?:good|great|fine|reasonable|solid|like (?:a )?(?:good|great|solid)? ?(?:plan|idea|breakdown|approach))", normalized):
        return True
    if re.fullmatch(r"(?:let'?s|lets) see how it goes", normalized):
        return True
    if re.fullmatch(r"what do you think (?:about|of) that", normalized):
        return True
    return False


def _aspect_key(label: str) -> str:
    terms = sorted(_important_query_terms(label) - _ASPECT_KEY_STOPWORDS)
    return "-".join(terms[:6]) if terms else ""


def _aspect_phase_key(label: str, snippet: str) -> str:
    lower = f"{label} {snippet}".lower()
    phase_patterns = [
        ("foundation", r"\b(?:setup|set up|schema|structure|initialize|initial|plan|design|architecture|requirements?)\b"),
        ("problem_solving", r"\b(?:error|issue|problem|bug|fix|fixed|debug|blocked|failure|failed|exception|warning|trouble|troubleshoot)\b"),
        ("validation", r"\b(?:test|tests|testing|coverage|suite|validate|verify|review|accuracy|quality)\b"),
        ("risk_controls", r"\b(?:security|privacy|access|permission|auth|authentication|authorization|password|safe|safety|risk)\b"),
        ("delivery", r"\b(?:deploy|deployment|release|launch|ship|publish|production|rollout|finalize|finalizing)\b"),
        ("comparison", r"\b(?:compare|compared|comparison|versus|vs|between)\b"),
        ("count_list", r"\b(?:how many|count|total|unique|list|listed|options|items)\b"),
        ("decision", r"\b(?:decided|chose|picked|settled on|went with|opted for)\b"),
        ("concern", r"\b(?:worried|concern|issue|problem|blocked|struggling|trouble)\b"),
        ("constraint", r"\b(?:must|need to|required|deadline|budget|limit|only|format)\b"),
        ("activity", r"\b(?:started|finished|completed|implemented|configured|created|added|fixed|reviewed|worked on)\b"),
    ]
    for phase, pattern in phase_patterns:
        if re.search(pattern, lower):
            return phase
    terms = sorted(_important_query_terms(lower) - _ASPECT_KEY_STOPWORDS)
    return "-".join(terms[:2]) if terms else ""


def _supporting_event_candidates(
    query: str,
    events: list[MemoryEvent],
    topic_keys: set[str],
    *,
    allowed_source_span_ids: set[str] | None = None,
    limit: int,
) -> list[Candidate]:
    if limit <= 0:
        return []
    allowed_source_span_ids = set(allowed_source_span_ids or set())
    out: list[Candidate] = []
    for event in sorted(events, key=lambda event: (event.time_start is None, event.time_start.isoformat() if event.time_start else "", event.event_id)):
        source_span_ids = {str(span_id) for span_id in event.source_span_ids if span_id}
        if allowed_source_span_ids and not source_span_ids & allowed_source_span_ids:
            continue
        if keyword_score(query, event.description) <= 0 and _timeline_signal(query, event.description) <= 0:
            continue
        annotation = event_annotation_from_event(event, query=query)
        metadata = {
            "event_type": event.event_type,
            "time_start": event.time_start.isoformat() if event.time_start else None,
            "topic_key": annotation.topic_key,
            "timeline_role": "supporting_graph_event",
            "timeline_label": annotation.label,
            "selector": "structured_event_graph",
        }
        metadata.update(annotation_to_metadata(annotation))
        out.append(
            Candidate(
                id=event.event_id,
                type="event",
                text=event.description,
                source="event_ordering_graph_selector_event",
                scores={"score": 0.9, "graph_proximity": 0.9, "temporal_fit": 0.7},
                source_span_ids=event.source_span_ids,
                metadata=metadata,
            )
        )
        if len(out) >= limit:
            break
    return out


def _timeline_signal(query: str, text: str) -> float:
    lower = text.lower()
    query_lower = query.lower()
    signal = 0.0
    for token in _important_query_terms(query_lower):
        if token in lower:
            signal += 0.12
    if re.search(r"\b(?:trying|started|implemented|completed|configured|planned|decided|asked|mentioned|brought up|added|fixed|reviewed|prepared)\b", lower):
        signal += 0.18
    if re.search(r"\b(?:i|we)\s+(?:want|wanted|need|needed|would like)\b", lower):
        signal += 0.14
    if re.search(r"\b(?:feature|project|task|topic|meeting|deadline|schedule|milestone|issue|decision|preference|constraint|requirement|result|progress)\b", lower):
        signal += 0.10
    return min(1.0, signal)


def _event_label(text: str) -> str:
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    for pattern in (
        r"\b(?:trying to|want to|need to|started|implemented|completed|configured|planned|decided|asked about|mentioned)\s+([^,.?;]{8,120})",
        r"\b(?:what about|how about)\s+([^,.?;]{8,120})",
    ):
        match = re.search(pattern, cleaned, flags=re.I)
        if match:
            return match.group(1).strip()
    return cleaned[:140]


def _subject_key(query: str, text: str, topic_key: str) -> str:
    terms = sorted(_important_query_terms(query) & _important_query_terms(text))
    if terms:
        return f"{topic_key}:{'-'.join(terms[:6])}" if topic_key else "-".join(terms[:6])
    return topic_key or "timeline"


def _important_query_terms(text: str) -> set[str]:
    stop = {
        "about",
        "across",
        "after",
        "before",
        "brought",
        "can",
        "different",
        "during",
        "give",
        "have",
        "into",
        "list",
        "mention",
        "mentioned",
        "only",
        "order",
        "our",
        "over",
        "throughout",
        "walk",
        "what",
        "when",
        "which",
        "with",
        "you",
    }
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in stop}


_ASPECT_KEY_STOPWORDS = {
    "also",
    "app",
    "apps",
    "before",
    "break",
    "build",
    "building",
    "chat",
    "conversation",
    "development",
    "different",
    "discussed",
    "project",
    "sure",
    "throughout",
    "tracker",
}

_TOPIC_ANCHOR_STOPWORDS = _ASPECT_KEY_STOPWORDS | {
    "across",
    "and",
    "aspect",
    "aspects",
    "brought",
    "conversation",
    "conversations",
    "developing",
    "development",
    "different",
    "feature",
    "implementing",
    "item",
    "items",
    "list",
    "mentioned",
    "mention",
    "five",
    "four",
    "nine",
    "only",
    "order",
    "personal",
    "project",
    "projects",
    "the",
    "eight",
    "seven",
    "six",
    "ten",
    "three",
    "through",
    "two",
    "walk",
    "which",
}


def _topic_anchor_phrases(query: str) -> list[str]:
    lower = query.lower()
    out: list[str] = []
    patterns = [
        r"\b(?:my|the|this|that)\s+([a-z0-9][a-z0-9 +#./-]{2,80}\s+(?:development|deployment|implementation|setup|workflow))\b",
        r"\b(?:my|the|this|that)\s+([a-z0-9][a-z0-9 +#./-]*(?:app|application|website|tracker|dashboard|feature|project|system|tool|api|bot|portfolio))\b",
        r"\b(?:implementing|developing|building|creating|setting up|working on)\s+(?:my|the|this|that)?\s*([a-z0-9][a-z0-9 +#./-]{4,80})\b",
        r"\baspects of\s+(?:implementing|developing|building|creating|setting up|working on)\s+(?:my|the|this|that)?\s*([a-z0-9][a-z0-9 +#./-]{4,80})\b",
        r"\b(?:aspects of|features of|concerns about)\s+(?:my|the|this|that)?\s*([a-z0-9][a-z0-9 +#./-]{4,80})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            phrase = _clean_topic_anchor(match.group(1))
            if phrase:
                out.append(phrase)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", lower)
        if len(token) >= 4 and token not in _TOPIC_ANCHOR_STOPWORDS
    ]
    for size in (3, 2):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[index : index + size])
            if phrase and phrase not in out:
                out.append(phrase)
    return list(dict.fromkeys(out))[:12]


def _query_needs_broad_topic_chronology(query: str, anchors: list[str]) -> bool:
    lower = query.lower()
    if not re.search(r"\b(?:across|throughout|over|all|during)\s+(?:our\s+)?(?:conversations?|sessions?|chats?)\b", lower):
        return False
    if re.search(r"\bdifferent\s+(?:aspects?|topics?|experiences?|ideas?|contributions?|challenges?|stages?|phases?)\b", lower):
        return True
    return _topic_anchors_are_generic_lifecycle(anchors) or not _topic_anchors_have_specific_entity(anchors)


def _topic_anchors_have_specific_entity(anchors: list[str]) -> bool:
    generic_terms = {
        "aspect",
        "aspects",
        "chat",
        "chats",
        "conversation",
        "conversations",
        "different",
        "experience",
        "experiences",
        "idea",
        "ideas",
        "item",
        "items",
        "order",
        "process",
        "project",
        "projects",
        "related",
        "topic",
        "topics",
        "using",
    }
    for anchor in anchors[:5]:
        terms = {
            term
            for term in re.findall(r"[a-z0-9]+", anchor.lower())
            if len(term) >= 3 and term not in generic_terms
        }
        if len(terms) >= 2:
            return True
    return False


def _clean_topic_anchor(value: str) -> str:
    value = re.sub(
        r"\b(?:throughout|across|in order|only|mention|different|aspects?|features?|conversations?|sessions?)\b.*$",
        "",
        value,
        flags=re.I,
    )
    anchor_stopwords = _TOPIC_ANCHOR_STOPWORDS - {"app", "apps", "development", "feature", "features", "project", "projects", "tracker"}
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", value.lower())
        if len(term) >= 3 and term not in anchor_stopwords
    ]
    if len(terms) < 2:
        return ""
    return " ".join(terms[:5])


def _topic_anchor_score(query: str, text: str, *, query_anchors: list[str], query_terms: set[str]) -> float:
    lower = text.lower()
    text_terms = _important_query_terms(text) - _ASPECT_KEY_STOPWORDS
    expanded_query_terms = _expand_topic_query_terms(query_terms)
    score = 0.0
    for phrase in query_anchors:
        phrase_terms = set(phrase.split())
        if not phrase_terms:
            continue
        if phrase in lower:
            score += 0.70 + 0.08 * min(len(phrase_terms), 4)
            continue
        overlap = len(phrase_terms & text_terms) / max(1, len(phrase_terms))
        if overlap >= 0.67 and len(phrase_terms & text_terms) >= 2:
            score += 0.28 + 0.28 * overlap
    if expanded_query_terms:
        distinctive = expanded_query_terms - _TOPIC_ANCHOR_STOPWORDS
        overlap = len(distinctive & text_terms)
        score += min(0.34, 0.10 * overlap)
    if re.search(r"\b(?:i(?:'m| am)?|we(?:'re| are)?)\s+(?:building|developing|implementing|creating|trying to|working on|setting up)\b", lower):
        score += 0.12
    if _timeline_signal(query, text) > 0:
        score += 0.08
    return min(1.0, score)


def _expand_topic_query_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    equivalents = {
        "financial": {"finance", "finances", "money", "saving", "savings", "budget", "budgeting", "investment", "investing", "workshop", "literacy"},
        "planning": {"plan", "plans", "goal", "goals", "budget", "budgeting", "saving", "savings"},
        "topics": {"topic", "matter", "issue", "concern"},
        "experiences": {"experience", "purchase", "shopping", "return", "collection"},
        "challenges": {"challenge", "concern", "stress", "workload", "burnout", "conflict"},
        "ideas": {"idea", "suggestion", "plan", "contribution"},
    }
    for term in list(terms):
        expanded.update(equivalents.get(term, set()))
    return expanded


def _continuation_timeline_signal(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(?:also|then|next|later|finally|after that|now|still|got it|what about|how about|another|following up|continue|deploy|test|fix|configure|integrate|add|implement|setup|set up|want|wanted|need|needed|would like|dropdown|debounce|limit|constraint)\b",
            lower,
        )
    )


def _broad_topic_continuation_signal(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(?:then|next|later|finally|after that|following up|continue|continued|now|still)\b",
            lower,
        )
    )


def _topic_anchors_are_generic_lifecycle(anchors: list[str]) -> bool:
    if not anchors:
        return False
    generic_terms = {
        "app",
        "application",
        "code",
        "deployment",
        "develop",
        "development",
        "implementation",
        "project",
        "setup",
        "system",
        "tool",
        "workflow",
    }
    for anchor in anchors[:3]:
        terms = set(re.findall(r"[a-z0-9]+", anchor.lower()))
        if terms and terms - generic_terms:
            return False
    return True


def _value_mentions(text: str) -> list[str]:
    patterns = [
        r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
        r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|hours?|minutes?|ms|seconds?|%)\b",
        r"\bv?\d+\.\d+(?:\.\d+)?\b",
    ]
    out: list[str] = []
    for pattern in patterns:
        out.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.I))
    return list(dict.fromkeys(out))[:12]


def _date_roles(query: str, text: str) -> list[str]:
    lower = f"{query} {text}".lower()
    roles: list[str] = []
    if "deadline" in lower:
        roles.append("deadline_date")
    if re.search(r"\b(?:start|started|begin|began)\b", lower):
        roles.append("start_date")
    if re.search(r"\b(?:finish|finished|complete|completed)\b", lower):
        roles.append("completion_date")
    if "reschedul" in lower:
        roles.append("rescheduled_date")
    return roles


def _query_item_limit(query: str) -> int | None:
    lower = query.lower()
    digit = re.search(r"\b(?:only\s+and\s+only\s+)?([2-9])\s+items?\b", lower)
    if digit:
        return int(digit.group(1))
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    for word, value in words.items():
        if re.search(rf"\b(?:only\s+and\s+only\s+)?{word}\s+items?\b", lower):
            return value
    return None


def _timeline_coverage_target(query: str, limit: int, *, available: int | None = None) -> int:
    answer_count = _query_item_limit(query)
    if answer_count is None:
        target = max(10, min(18, limit))
    else:
        target = min(limit, max(answer_count + 5, answer_count * 3, 8))
    if available is not None:
        target = min(target, available)
    return max(1, target)


def _label_family(label: str) -> str:
    terms = sorted(_important_query_terms(label))
    return "-".join(terms[:3]) if terms else label[:40].lower()


def _natural_key(value: object) -> tuple[tuple[int, int | str], ...]:
    text = "" if value is None else str(value)
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(parts)


def _reverse_key(value: tuple[Any, ...]) -> tuple[int, ...]:
    encoded = "|".join(str(part) for part in value)
    return tuple(-ord(char) for char in encoded)
