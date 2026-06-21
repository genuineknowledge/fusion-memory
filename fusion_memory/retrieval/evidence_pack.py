from __future__ import annotations

import re
from typing import Any

from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import Candidate, EvidencePack, QueryPlan
from fusion_memory.core.text import compact_summary, tokenize
from fusion_memory.retrieval.aggregation_keys import (
    aggregation_keys_for_query as _aggregation_keys,
    combinatorics_aggregation_keys,
    generic_aggregation_keys,
    generic_list_candidate_keys,
    is_combinatorics_aggregation_query as _is_combinatorics_aggregation_query,
    is_generic_count_or_list_query,
    is_stress_break_aggregation_query as _is_stress_break_aggregation_query,
    vendor_tool_aggregation_keys,
    stress_break_aggregation_keys,
)
from fusion_memory.retrieval.aggregation_preferences import _preference_constraint_items
from fusion_memory.retrieval.exact_answer_operators import exact_answer_operator_fields
from fusion_memory.retrieval.pack_contract import active_pack_sections_for, pack_contract_metadata
from fusion_memory.retrieval.slot_update_recall import build_slot_update_recall_rows as _slot_update_recall_rows
from fusion_memory.retrieval.rule_registry import RuleDefinition, record_rule_hit, register_rule
from fusion_memory.retrieval.temporal_pack import (
    date_signal as _date_signal,
    temporal_candidate_table as _temporal_candidate_table,
    temporal_mentions as _temporal_mentions,
    temporal_range_pairs as _temporal_range_pairs,
    temporal_roles_in_text as _temporal_roles_in_text,
    temporal_summary as _temporal_summary,
    temporal_target_roles as _temporal_target_roles,
)
from fusion_memory.retrieval.value_history_pack import (
    build_value_history_table as _value_history_table,
    dedupe_value_mentions as _dedupe_value_mentions,
    query_targeted_value_mentions as _query_targeted_value_mentions,
    value_context_is_target_goal as _value_context_is_target_goal,
    value_history_target_type_priority as _value_history_target_type_priority,
    value_history_topic_mismatch_rank as _value_history_topic_mismatch_rank,
    value_mentions as _value_mentions,
    value_update_marker_strength as _value_update_marker_strength,
)
from fusion_memory.storage.sqlite_store import SQLiteMemoryStore


register_rule(
    RuleDefinition(
        rule_id="current_value.stale_history_marker",
        module=__name__,
        purpose="drop stale historical spans when resolving current value",
        category="high_risk",
        pattern="initially|previously|formerly|原来|曾经",
        ability="current_value",
        protected=True,
        protected_reason="high_precision_current_value",
    )
)


class EvidencePackBuilder:
    def __init__(self, store: SQLiteMemoryStore, config: MemoryConfig | None = None) -> None:
        self.store = store
        self.config = config or DEFAULT_CONFIG

    def build(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        coverage: dict,
        trace: list[dict],
        token_budget: int | None = None,
    ) -> EvidencePack:
        token_budget = token_budget or self.config.answer_context_budget_tokens
        intent = getattr(plan, "intent", {}) or {}
        skip_stale_current_value_history = _query_needs_current_value_resolution(query, intent)
        current_views: list[dict] = []
        profiles: list[dict] = []
        facts: list[dict] = []
        events: list[dict] = []
        spans: list[dict] = []
        conflicts: list[dict] = []
        seen_spans: set[str] = set()
        estimated_tokens = 0
        selected_scope = None
        for candidate in candidates:
            if candidate.type == "view":
                current_views.append({"id": candidate.id, "text": candidate.text, "source_span_ids": candidate.source_span_ids})
            elif candidate.type == "profile":
                profiles.append({"id": candidate.id, "text": candidate.text, "source_span_ids": candidate.source_span_ids})
            elif candidate.type == "fact":
                facts.append({"id": candidate.id, "text": candidate.text, "source_span_ids": candidate.source_span_ids})
            elif candidate.type == "event":
                event_record = {
                    "id": candidate.id,
                    "description": candidate.text,
                    "text": candidate.text,
                    "event_type": candidate.metadata.get("event_type"),
                    "milestone_group": candidate.metadata.get("milestone_group") or _milestone_group_from_text(candidate.text),
                    "time_start": candidate.metadata.get("time_start"),
                    "source_span_ids": candidate.source_span_ids,
                }
                self._apply_candidate_metadata(event_record, candidate)
                events.append(event_record)
            for span_id in candidate.source_span_ids:
                is_coverage_anchor = (
                    candidate.metadata.get("selector") == "event_ordering_coverage"
                    and candidate.metadata.get("timeline_role") == "user_aspect_anchor"
                )
                seen_key = candidate.id if is_coverage_anchor else span_id
                if seen_key in seen_spans:
                    continue
                span = self.store.get_span(span_id)
                if not span:
                    continue
                if skip_stale_current_value_history and _is_stale_historical_current_value_span(span.content):
                    continue
                selected_scope = selected_scope or span.scope
                seen_spans.add(seen_key)
                record, estimated_tokens = self._span_record(query, plan, span, estimated_tokens, token_budget)
                if record:
                    record["candidate_source"] = candidate.source
                    if candidate.metadata.get("aggregation_coverage") or (
                        plan.query_type == "multi_session_reasoning" and candidate.metadata.get("aggregation_keys")
                    ):
                        aggregation_content = _aggregation_content_summary(query, span.content, max(self.config.evidence_span_summary_chars, 1800))
                        if aggregation_content:
                            record["content"] = aggregation_content
                    if is_coverage_anchor:
                        record["id"] = candidate.id
                        record["conversation_content"] = compact_summary(span.content, max(self.config.evidence_span_summary_chars, 1200))
                        record["content"] = compact_summary(candidate.text, self.config.evidence_span_summary_chars)
                        record["source_span_ids"] = [span_id]
                    self._apply_candidate_metadata(record, candidate)
                    if plan.query_type == "multi_session_reasoning" and not record.get("aggregation_keys"):
                        keys = _aggregation_keys(query, span.content, speaker=span.speaker)
                        if keys:
                            record["aggregation_keys"] = keys
                            record["aggregation_signal"] = max(
                                record.get("aggregation_signal", 0.0),
                                _aggregation_pack_signal(query, span.content),
                            )
                    spans.append(record)
        if selected_scope:
            expanded, estimated_tokens = self._expand_category_spans(
                query,
                plan,
                selected_scope,
                spans,
                seen_spans,
                estimated_tokens,
                token_budget,
            )
            spans.extend(expanded)
            exact_scope_spans = self.store.list_spans(
                selected_scope,
                include_session=bool(getattr(selected_scope, "session_id", None)),
            )
            exact_answer_candidates = _exact_answer_candidates(query, plan, exact_scope_spans, seen_spans)
            if exact_answer_candidates:
                coverage["exact_answer_candidates"] = exact_answer_candidates
            if plan.query_type in {"preference", "instruction"}:
                preference_constraints = _preference_constraint_items(
                    query,
                    [_span_record_for_model_view(span) for span in exact_scope_spans],
                )
                if preference_constraints:
                    coverage["preference_constraints"] = preference_constraints
        if plan.query_type == "contradiction_resolution":
            conflicts = _contradiction_claim_buckets(query, spans, facts)
        elif plan.query_type == "knowledge_update":
            conflicts = [
                {"fact_id": fact["id"], "source_span_ids": fact["source_span_ids"]}
                for fact in facts[:4]
            ]
        if plan.query_type == "event_ordering":
            graph_spans = [
                span
                for span in spans
                if span.get("selector") in {"structured_event_graph", "event_ordering_coverage"}
                and span.get("timeline_index") is not None
                and span.get("timeline_role") != "supporting_context"
            ]
            if graph_spans:
                graph_ids = {span.get("id") for span in graph_spans}
                graph_raw_ids = {
                    str(value)
                    for span in graph_spans
                    for value in [span.get("original_span_id"), *(span.get("source_span_ids") or [])]
                    if value
                }
                graph_spans.sort(key=_event_ordering_span_sort_key)
                supporting = [
                    span
                    for span in spans
                    if span.get("id") not in graph_ids
                    and str(span.get("id") or "") not in graph_raw_ids
                    and str(span.get("original_span_id") or "") not in graph_raw_ids
                ]
                supporting.sort(key=_timeline_sort_key)
                spans = graph_spans + supporting
            else:
                spans.sort(key=_timeline_sort_key)
            for index, span in enumerate(spans, start=1):
                span["timeline_index"] = index
            events.sort(key=self._event_timeline_sort_key)
            for index, event in enumerate(events, start=1):
                event["timeline_index"] = index
        needs_value_history = _plan_needs_value_history(plan.query_type, query, intent)
        needs_recency_order = plan.query_type == "knowledge_update" or (
            needs_value_history and _query_needs_current_value_resolution(query, intent)
        )
        if plan.query_type in {"knowledge_update", "multi_session_reasoning"} or needs_value_history:
            spans.sort(key=_timeline_sort_key)
            for index, span in enumerate(spans, start=1):
                span["history_index"] = index
            if needs_recency_order:
                for recency_rank, span in enumerate(reversed(spans), start=1):
                    span["recency_rank"] = recency_rank
                    values = _value_mentions(span.get("content", ""))
                    if values:
                        span["value_mentions"] = values
                spans.sort(key=lambda span: int(span.get("recency_rank") or 10**9))
        if plan.query_type == "temporal_lookup":
            temporal_role_counts: dict[str, int] = {}
            for span in spans:
                mentions = _temporal_mentions(query, span.get("content", ""), span.get("timestamp"))
                if mentions:
                    span["temporal_mentions"] = mentions
                    span["temporal_roles"] = list(dict.fromkeys(mention["role"] for mention in mentions))
                    for mention in mentions:
                        role = str(mention["role"])
                        temporal_role_counts[role] = temporal_role_counts.get(role, 0) + 1
            temporal_candidates = _temporal_candidate_table(query, spans)
            if temporal_candidates:
                coverage["temporal_candidates"] = temporal_candidates
            temporal_range_pairs = _temporal_range_pairs(query, spans)
            if temporal_range_pairs:
                coverage["temporal_range_pairs"] = temporal_range_pairs
        if plan.query_type == "knowledge_update" or needs_value_history:
            value_history = _value_history_table(query, spans, facts)
            if plan.query_type == "knowledge_update" and selected_scope:
                recall_rows = _slot_update_recall_rows(query, exact_scope_spans)
                if recall_rows:
                    coverage["slot_update_recall"] = recall_rows
                    value_history = _merge_value_history_rows(value_history, recall_rows, limit=24)
            if value_history:
                coverage["value_history"] = value_history
        if plan.query_type == "summarization":
            resolution_pairs = _summary_resolution_pairs(query, spans)
            if resolution_pairs:
                coverage["resolution_pairs"] = resolution_pairs
            summary_clusters = _summary_clusters(query, spans)
            if summary_clusters:
                coverage["summary_clusters"] = summary_clusters
        if plan.query_type == "instruction":
            instruction_constraints = _instruction_constraints(query)
            if instruction_constraints:
                coverage["instruction_constraints"] = instruction_constraints
        answer_policy = "answer_with_evidence_or_abstain"
        if plan.query_type == "abstention" or coverage.get("coverage_insufficient"):
            answer_policy = "abstain_if_not_supported"
        coverage = {
            **coverage,
            "token_budget": token_budget,
            "estimated_source_tokens": estimated_tokens,
            "timeline_span_count": len(spans) if plan.query_type == "event_ordering" else 0,
            "query_intent": intent,
            "dropped_high_signal_candidates": coverage.get("dropped_high_signal_candidates", []),
        }
        coverage["pack_contract"] = pack_contract_metadata(
            active_sections=active_pack_sections_for(plan.query_type, coverage)
        )
        if plan.query_type == "event_ordering":
            coverage["timeline_basis"] = "conversation_order"
        if plan.query_type == "temporal_lookup":
            coverage["temporal_role_counts"] = temporal_role_counts
            coverage["temporal_target_roles"] = _temporal_target_roles(query)
        format_requirements = _format_requirements(query)
        if format_requirements:
            coverage["format_requirements"] = format_requirements
        if plan.query_type == "contradiction_resolution" and conflicts:
            coverage["claim_polarity_counts"] = {
                "positive": len(conflicts[0].get("positive_source_span_ids", [])),
                "negative": len(conflicts[0].get("negative_source_span_ids", [])),
                "uncertain": len(conflicts[0].get("uncertain_source_span_ids", [])),
            }
        return EvidencePack(
            query=query,
            answer_policy=answer_policy,
            current_views=current_views,
            entity_profiles=profiles,
            facts=facts,
            events=events,
            source_spans=spans,
            conflicts=conflicts,
            coverage=coverage,
            debug_trace=trace,
        )

    def _span_record(
        self,
        query: str,
        plan: QueryPlan,
        span,
        estimated_tokens: int,
        token_budget: int,
    ) -> tuple[dict | None, int]:
        content_limit = self.config.evidence_span_summary_chars
        content = (
            _temporal_summary(query, span.content, max(content_limit, 1200))
            if plan.query_type == "temporal_lookup"
            else compact_summary(span.content, content_limit)
        )
        content_tokens = len(tokenize(content))
        if estimated_tokens + content_tokens > token_budget:
            remaining = max(0, token_budget - estimated_tokens)
            if remaining <= 0:
                return None, estimated_tokens
            words = content.split()
            content = " ".join(words[:remaining])
            content_tokens = len(tokenize(content))
        estimated_tokens += content_tokens
        return (
            {
                "id": span.span_id,
                "session_id": span.scope.session_id,
                "turn_id": span.turn_id,
                "source_uri": span.source_uri,
                "speaker": span.speaker,
                "timestamp": span.timestamp.isoformat(),
                "content": content,
                "topic_group": _span_group_key(span.source_uri, span.turn_id),
            },
            estimated_tokens,
        )

    def _apply_candidate_metadata(self, record: dict, candidate: Candidate) -> None:
        for key in (
            "structured_annotation",
            "topic_key",
            "timeline_index",
            "timeline_label",
            "timeline_role",
            "selector",
            "original_span_id",
            "aspect_index",
            "aspect_key",
            "coverage_terms",
            "supports_timeline_index",
            "supports_aspect_key",
            "coverage_origin",
            "event_id",
            "conversation_content",
            "broad_raw_recall",
            "recall_query",
            "aggregation_keys",
            "aggregation_signal",
            "aggregation_focus",
            "graph_node_id",
            "graph_edge_count",
            "graph_phase",
            "graph_topic",
            "graph_fallback",
        ):
            value = candidate.metadata.get(key)
            if value is not None:
                record[key] = value

    def _expand_category_spans(
        self,
        query: str,
        plan: QueryPlan,
        scope,
        current_spans: list[dict],
        seen_spans: set[str],
        estimated_tokens: int,
        token_budget: int,
    ) -> tuple[list[dict], int]:
        mode = _pack_expansion_mode(query, plan.query_type)
        if not mode:
            return [], estimated_tokens
        intent = getattr(plan, "intent", {}) or {}
        skip_stale_current_value_history = mode in {"preference", "exact", "broad"} and _query_needs_current_value_resolution(query, intent)
        groups = {str(span.get("topic_group") or "") for span in current_spans if span.get("topic_group")}
        if not groups:
            return [], estimated_tokens
        event_ordering_anchor_ids = {
            str(value)
            for span in current_spans
            if plan.query_type == "event_ordering"
            and span.get("selector") == "event_ordering_coverage"
            and span.get("timeline_role") == "user_aspect_anchor"
            for value in [span.get("original_span_id"), *(span.get("source_span_ids") or [])]
            if value
        }
        event_ordering_support_ids: set[str] = set()
        event_ordering_user_ids: set[str] = set()
        if event_ordering_anchor_ids:
            ordered = [
                span
                for span in self.store.list_spans(scope)
                if span.span_type in {"turn", "tool_result", "document_chunk"}
                and _span_group_key(span.source_uri, span.turn_id) in groups
            ]
            ordered.sort(key=lambda span: _timeline_sort_key(_span_sort_record(span)))
            anchor_positions = {
                index
                for index, span in enumerate(ordered)
                if span.span_id in event_ordering_anchor_ids
            }
            if anchor_positions:
                anchor_min = min(anchor_positions)
                anchor_max = max(anchor_positions)
                event_ordering_user_ids = {
                    span.span_id
                    for index, span in enumerate(ordered)
                    if span.speaker == "user"
                    and anchor_min <= index <= anchor_max
                }
                rescue_candidates = [
                    (
                        _event_ordering_chronology_rescue_score(query, span.content, span.speaker),
                        index,
                        span,
                    )
                    for index, span in enumerate(ordered)
                    if span.speaker == "user"
                    and span.span_id not in event_ordering_user_ids
                    and span.span_id not in event_ordering_anchor_ids
                ]
                rescue_candidates = [
                    item for item in rescue_candidates if item[0] >= 0.26
                ]
                rescue_limit = max(6, min(18, len(event_ordering_user_ids) + 8))
                event_ordering_user_ids.update(
                    span.span_id
                    for _score, _index, span in _event_ordering_diverse_chronology_rescue(rescue_candidates, rescue_limit)
                )
            for index, span in enumerate(ordered):
                if span.span_id in event_ordering_anchor_ids:
                    event_ordering_support_ids.add(span.span_id)
                    continue
                if span.speaker == "user":
                    continue
                if any(index == anchor_index + 1 for anchor_index in anchor_positions):
                    event_ordering_support_ids.add(span.span_id)
        scope_spans = [
            span
            for span in self.store.list_spans(scope)
            if span.span_type in {"turn", "tool_result", "document_chunk"}
            and _span_group_key(span.source_uri, span.turn_id) in groups
            and (
                not event_ordering_anchor_ids
                or span.span_id in event_ordering_support_ids
                or span.span_id in event_ordering_user_ids
            )
        ]
        candidates = [
            span
            for span in scope_spans
            if span.span_id not in seen_spans
            and not (
                skip_stale_current_value_history
                and _is_stale_historical_current_value_span(span.content)
            )
        ]
        if not candidates:
            return [], estimated_tokens
        scored: list[tuple[float, object]] = []
        target_roles = set(_temporal_target_roles(query)) if mode == "temporal" else set()
        for span in candidates:
            score = _topic_scope_score(query, span.content)
            exact = _exact_overlap(query, span.content)
            date_signal = _date_signal(span.content)
            role_signal = 0.0
            if target_roles:
                roles = _temporal_roles_in_text(query, span.content)
                if roles & target_roles:
                    role_signal = 0.55 + 0.15 * min(len(roles & target_roles), 2)
            speaker_signal = 0.12 if span.speaker == "user" else 0.04
            if mode == "summary":
                summary_signal = _summary_pack_signal(query, span.content)
                total = (0.44 * score) + (0.18 * exact) + (0.30 * summary_signal) + speaker_signal
            elif mode == "temporal":
                total = (0.36 * score) + (0.22 * exact) + (0.30 * role_signal) + (0.12 * date_signal)
            elif mode == "ordering":
                total = (0.50 * score) + (0.18 * exact) + (0.20 if span.speaker == "user" else 0.04)
            elif mode == "aggregation":
                aggregation = _aggregation_pack_signal(query, span.content)
                total = (0.34 * score) + (0.24 * exact) + (0.34 * aggregation) + speaker_signal
            elif mode == "preference":
                preference = _preference_constraint_pack_signal(query, span.content, span.speaker)
                total = (0.30 * score) + (0.18 * exact) + (0.44 * preference) + speaker_signal
            else:
                total = (0.48 * score) + (0.30 * exact) + (0.12 * date_signal) + speaker_signal
            if total > 0.08:
                scored.append((total, span))
        if not scored:
            return [], estimated_tokens
        ordered_scope_spans = sorted(scope_spans, key=lambda span: _timeline_sort_key(_span_sort_record(span)))
        max_total = {
            "summary": 64,
            "temporal": 18,
            "ordering": max(20, min(32, len(current_spans) + 8)),
            "aggregation": 24,
            "preference": 28,
            "broad": 16,
            "exact": 14,
        }.get(mode, 14)
        if mode == "summary":
            scored = _summary_diverse_candidates(
                scored,
                current_spans=current_spans,
                ordered_spans=ordered_scope_spans,
                limit=max(0, max_total - len(current_spans)),
            )
            scored.sort(key=lambda item: _timeline_sort_key(_span_sort_record(item[1])))
        elif mode == "ordering":
            scored = _event_ordering_diverse_scored_spans(
                scored,
                limit=max(0, max_total - len(current_spans)),
            )
            scored.sort(key=lambda item: (_timeline_sort_key(_span_sort_record(item[1])), -item[0]))
        elif mode == "aggregation":
            scored.sort(key=lambda item: (item[0], _reverse_timeline_key(_span_sort_record(item[1]))), reverse=True)
            scored = _aggregation_anchor_diverse_candidates(query, scored, limit=max(0, max_total - len(current_spans)))
        elif mode == "preference":
            scored.sort(
                key=lambda item: (
                    item[0],
                    1.0 if getattr(item[1], "speaker", "") == "user" else 0.0,
                    _reverse_timeline_key(_span_sort_record(item[1])),
                ),
                reverse=True,
            )
        else:
            scored.sort(key=lambda item: (item[0], _reverse_timeline_key(_span_sort_record(item[1]))), reverse=True)
        if mode == "aggregation":
            scored = sorted(scored[: max(0, max_total - len(current_spans))], key=lambda item: _timeline_sort_key(_span_sort_record(item[1])))
        out: list[dict] = []
        for _score, span in scored:
            if len(current_spans) + len(out) >= max_total:
                break
            if mode == "ordering" and event_ordering_anchor_ids and span.speaker != "user" and span.span_id not in event_ordering_support_ids:
                continue
            record, estimated_tokens = self._span_record(query, plan, span, estimated_tokens, token_budget)
            if not record:
                break
            if mode == "summary":
                paired = _summary_pair_content(span, ordered_scope_spans, max(self.config.evidence_span_summary_chars, 900))
                if paired:
                    record["content"] = paired
            if mode == "aggregation":
                keys = _aggregation_keys(query, span.content, speaker=span.speaker)
                if keys:
                    record["aggregation_keys"] = keys
                    record["aggregation_signal"] = max(record.get("aggregation_signal", 0.0), _aggregation_pack_signal(query, span.content))
            record["category_expansion"] = mode
            out.append(record)
            seen_spans.add(span.span_id)
        return out, estimated_tokens

    def _event_timeline_sort_key(self, event: dict) -> tuple[int, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...], str, str]:
        span_id = next(iter(event.get("source_span_ids") or []), None)
        span = self.store.get_span(span_id) if span_id else None
        if span:
            return (
                0,
                _natural_turn_key(span.source_uri),
                _natural_turn_key(span.turn_id),
                span.timestamp.isoformat(),
                str(event.get("id") or ""),
            )
        return (
            1,
            (),
            (),
            str(event.get("time_start") or ""),
            str(event.get("id") or ""),
        )


def _timeline_sort_key(span: dict) -> tuple[int, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...], str, str]:
    return (
        0 if span.get("source_uri") or span.get("turn_id") else 1,
        _natural_turn_key(span.get("source_uri")),
        _natural_turn_key(span.get("turn_id")),
        str(span.get("timestamp") or ""),
        str(span.get("id") or ""),
    )


def _event_ordering_span_sort_key(span: dict) -> tuple[Any, ...]:
    if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor":
        return (0, _timeline_sort_key(span))
    return (1, int(span.get("timeline_index") or 10**9), _timeline_sort_key(span))


def _natural_turn_key(value: object) -> tuple[tuple[int, int | str], ...]:
    text = "" if value is None else str(value)
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(parts)


def _span_sort_record(span) -> dict:
    return {
        "source_uri": span.source_uri,
        "turn_id": span.turn_id,
        "timestamp": span.timestamp.isoformat(),
        "id": span.span_id,
    }


def _span_record_for_model_view(span: Any) -> dict[str, Any]:
    return {
        "id": getattr(span, "span_id", None),
        "source_span_id": getattr(span, "span_id", None),
        "speaker": getattr(span, "speaker", None),
        "timestamp": getattr(getattr(span, "timestamp", None), "isoformat", lambda: None)(),
        "turn_id": getattr(span, "turn_id", None),
        "source_uri": getattr(span, "source_uri", None),
        "content": str(getattr(span, "content", "") or ""),
        "topic_group": _span_group_key(getattr(span, "source_uri", None), getattr(span, "turn_id", None)),
    }


def _event_ordering_chronology_rescue_score(query: str, text: str, speaker: str | None) -> float:
    if str(speaker or "").lower() != "user":
        return 0.0
    lower = text.lower()
    if re.search(r"\b(?:sure|here(?:'s| is))\b.{0,60}\b(?:break it down|components?|milestones?)\b", lower):
        return 0.0
    topic = _topic_scope_score(query, text)
    exact = _exact_overlap(query, text)
    if max(topic, exact) < 0.05:
        return 0.0
    score = (0.44 * topic) + (0.26 * exact)
    if re.search(
        r"\b(?:i|we)\s+(?:am|are|was|were|have|had|need|needed|want|wanted|started|finished|completed|implemented|configured|created|added|fixed|reviewed|worked|tried|decided|chose|asked|mentioned|focused|used|collaborated|declined|reduced|increased|planned|scheduled)\b",
        lower,
    ):
        score += 0.14
    if re.search(r"\b(?:worried|concerned|conflicted|nervous|stress|stressed|burnout|challenge|problem|issue|risk)\b", lower):
        score += 0.12
    if re.search(r"\b(?:updated?|revised?|changed?|improved?|increased|reduced|finished|completed|started|decided|declined|accepted|suggested|recommended)\b", lower):
        score += 0.10
    if re.search(r"\b(?:\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+\s*(?:days?|weeks?|months?|hours?))\b", text):
        score += 0.05
    if re.search(r"\b[A-Z][A-Za-z0-9'&.-]{2,}(?:\s+[A-Z][A-Za-z0-9'&.-]{2,}){0,2}\b", text):
        score += 0.05
    if re.search(r"\b(?:over\s+time|throughout|order|chronolog|sequence|different aspects?|challenges?)\b", query.lower()):
        score += 0.04
    return min(1.0, score)


def _event_ordering_diverse_chronology_rescue(
    scored: list[tuple[float, int, Any]],
    limit: int,
) -> list[tuple[float, int, Any]]:
    if limit <= 0 or len(scored) <= limit:
        return sorted(scored, key=lambda item: item[1])
    ordered = sorted(scored, key=lambda item: item[1])
    selected: list[tuple[float, int, Any]] = []
    seen_positions: set[int] = set()
    for bucket in range(limit):
        start = round(bucket * len(ordered) / limit)
        end = round((bucket + 1) * len(ordered) / limit)
        if end <= start:
            end = min(len(ordered), start + 1)
        window = [item for item in ordered[start:end] if item[1] not in seen_positions]
        if not window:
            continue
        choice = max(window, key=lambda item: (item[0], -abs(item[1] - ((start + end) // 2))))
        selected.append(choice)
        seen_positions.add(choice[1])
    if len(selected) < limit:
        for item in sorted(scored, key=lambda value: (value[0], -value[1]), reverse=True):
            if item[1] in seen_positions:
                continue
            selected.append(item)
            seen_positions.add(item[1])
            if len(selected) >= limit:
                break
    return sorted(selected, key=lambda item: item[1])


def _event_ordering_diverse_scored_spans(
    scored: list[tuple[float, Any]],
    *,
    limit: int,
) -> list[tuple[float, Any]]:
    if limit <= 0 or len(scored) <= limit:
        return scored
    ordered = sorted(scored, key=lambda item: _timeline_sort_key(_span_sort_record(item[1])))
    selected: list[tuple[float, Any]] = []
    seen_ids: set[str] = set()
    for bucket in range(limit):
        start = round(bucket * len(ordered) / limit)
        end = round((bucket + 1) * len(ordered) / limit)
        if end <= start:
            end = min(len(ordered), start + 1)
        window = [
            item
            for item in ordered[start:end]
            if str(getattr(item[1], "span_id", "")) not in seen_ids
        ]
        if not window:
            continue
        choice = max(window, key=lambda item: item[0])
        selected.append(choice)
        seen_ids.add(str(getattr(choice[1], "span_id", "")))
    if len(selected) < limit:
        for item in sorted(scored, key=lambda value: value[0], reverse=True):
            span_id = str(getattr(item[1], "span_id", ""))
            if span_id in seen_ids:
                continue
            selected.append(item)
            seen_ids.add(span_id)
            if len(selected) >= limit:
                break
    return selected


def _reverse_timeline_key(span: dict) -> tuple[int, ...]:
    encoded = "|".join(str(part) for part in _timeline_sort_key(span))
    return tuple(-ord(char) for char in encoded)


def _span_group_key(source_uri: object, turn_id: object) -> str:
    for value in (source_uri, turn_id):
        if not value:
            continue
        text = str(value)
        match = re.match(r"^(beam:[^:]+:\d+):", text)
        if match:
            return match.group(1)
        if "#" in text:
            return text.split("#", 1)[0]
    return ""


def _pack_expansion_mode(query: str, query_type: str) -> str | None:
    lower = query.lower()
    if query_type == "event_ordering":
        return "ordering"
    if query_type == "temporal_lookup":
        return "temporal"
    if query_type == "summarization":
        return "summary"
    if query_type in {"contradiction_resolution", "knowledge_update"}:
        return "broad"
    if query_type == "multi_session_reasoning":
        return "aggregation"
    if query_type in {"preference", "instruction"}:
        return "preference"
    if query_type == "factual_exact" and re.search(r"\b(?:across|throughout|over time|different|total|how many|between)\b", lower):
        return "broad"
    if query_type == "factual_exact":
        return "exact"
    if query_type == "assistant_reference":
        return "exact"
    return None


def _preference_constraint_pack_signal(query: str, text: str, speaker: str | None = None) -> float:
    lower = text.lower()
    query_lower = query.lower()
    if not text.strip():
        return 0.0
    score = 0.0
    if str(speaker or "").lower() == "user":
        score += 0.10
    if re.search(r"\b(?:i|we)\s+(?:prefer|like|want|need|usually|always|try to|avoid|would rather)\b", lower):
        score += 0.24
    if re.search(r"\b(?:prefer|preference|rather than|instead of|avoid|important to me|must|should)\b", lower):
        score += 0.16
    if re.search(r"\b(?:schedule|sessions?|routine|timing|pace|pacing|breaks?|burnout|fatigue|marathon|focused)\b", query_lower):
        if re.search(r"\b(?:short bursts?|shorter intervals?|30\s*-?\s*minutes?|minutes?\s+at\s+a\s+time|marathon sessions?|burnout|breaks?)\b", lower):
            score += 0.34
        if re.search(r"\b(?:session|sessions|schedule|routine|pace|pacing)\b", lower):
            score += 0.10
    if re.search(r"\b(?:edit|editing|draft|revision|writing)\b", query_lower) and re.search(r"\b(?:edit|editing|draft|revision|writing)\b", lower):
        score += 0.10
    if re.search(r"\b(?:choose|candidate|responsibilities|executor|appoint)\b", query_lower) and re.search(r"\b(?:organized|organizational|reliable|responsible|best fit|candidate)\b", lower):
        score += 0.24
    if re.search(r"\b(?:spreadsheet|manual tracking|digital|electronic|detailed drawings?|video demos?|neutral colors?|recycled materials?|family reviews?)\b", lower):
        score += 0.12
    return min(1.0, score)


def _exact_answer_candidates(query: str, plan: QueryPlan, scope_spans: list[Any], seen_spans: set[str]) -> list[dict[str, Any]]:
    if plan.query_type not in {"factual_exact", "assistant_reference", "knowledge_update", "temporal_lookup", "multi_session_reasoning"}:
        return []
    lower = query.lower()
    asks_assistant_recommendation = bool(
        re.search(
            r"\b(?:you recommend|you recommended|you suggest|you suggested|did you recommend|"
            r"what steps did you recommend|how did you recommend)\b",
            lower,
        )
    )
    asks_user_fact = bool(re.search(r"\b(?:did i say|i say|i|my|me|i'm|i am)\b", lower))
    asks_process_plan = asks_assistant_recommendation and bool(
        re.search(r"\b(?:process|timeline|schedule|sequence|structure|plan|writing|draft|review|revision|submission|deadline|cutoff)\b", lower)
    )
    target_terms = _topic_scope_tokens(query)
    scored: list[tuple[float, Any]] = []
    for span in scope_spans:
        if getattr(span, "span_type", "") not in {"turn", "tool_result", "document_chunk"}:
            continue
        speaker = getattr(span, "speaker", "")
        content = str(getattr(span, "content", "") or "")
        if not content.strip():
            continue
        content_lower = content.lower()
        topic_score = _topic_scope_score(query, content)
        exact_score = _exact_overlap(query, content)
        score = (0.45 * _topic_scope_score(query, content)) + (0.35 * _exact_overlap(query, content))
        score += 0.30 * _value_update_marker_strength(lower, content_lower, "")
        if asks_assistant_recommendation:
            score += 0.35 if speaker == "assistant" else -0.10
            if re.search(r"\b(?:recommend|suggest|steps?|prepare|plan|strategy|research|talk to|clarify|expectations?)\b", content_lower):
                score += 0.22
            score += _assistant_recommendation_domain_signal(lower, content_lower)
            if asks_process_plan:
                score += _process_plan_signal(lower, content_lower)
            if re.search(r"\bwork\s+environment\b", lower) and re.search(
                r"\b(?:work\s+environment|company(?:'s)?\s+(?:mission|values|financial health)|financial health|"
                r"current employees?|company culture|workload|hours|performance metrics?|expectations?|"
                r"preparing for the transition|due diligence|research the company|understand expectations)\b",
                content_lower,
            ):
                score += 0.62
            if asks_process_plan and topic_score < 0.10 and exact_score < 0.10:
                score -= 0.20
        elif asks_user_fact:
            score += 0.28 if speaker == "user" else 0.04
            if re.search(r"\b(?:how many|total|count|number of)\b", lower):
                targeted_values = _query_targeted_value_mentions(query, content)
                if targeted_values:
                    score += 0.34 if speaker == "user" else 0.12
                elif speaker == "user" and _value_mentions(content):
                    score += 0.12
            if _location_event_query(lower, content):
                score += 0.52 if speaker == "user" else -0.08
                if re.search(r"\b(?:i(?:'m| am)\s+planning|upcoming|weekend getaway|anniversary dinner)\b", content_lower):
                    score += 0.22
                if re.search(r"\bam\s+i\s+planning\b", lower) and re.search(r"\b[A-Z][a-z]{2,}\s+planned\b", content):
                    score -= 0.62
            if _historical_said_query(lower) and re.search(r"\b(?:new|now|current|currently|latest|updated|revised)\b", content_lower):
                score -= 0.22
            if re.search(r"\b(?:how far|distance|town|where)\b", lower):
                if re.search(r"\b\d+(?:\.\d+)?\s*miles?\s+away\b", content_lower):
                    score += 0.45
                if re.search(r"\b(?:in|near|from)\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b", content):
                    score += 0.18
                if re.search(r"\b(?:parents?|mother|father|mom|dad|family)\b", lower) and re.search(
                    r"\b(?:parents?|mother|father|mom|dad|family)\b",
                    content_lower,
                ):
                    score += 0.15
                if not re.search(r"\b\d+(?:\.\d+)?\s*miles?\b", content_lower):
                    score -= 0.25
        else:
            score += 0.12 if speaker == "user" else 0.06
        if target_terms and len(target_terms & _topic_scope_tokens(content)) >= 2:
            score += 0.15
        proper_names = _query_proper_names(query)
        if proper_names:
            matched_names = {name for name in proper_names if re.search(rf"\b{re.escape(name)}\b", content)}
            if matched_names:
                score += min(0.30, 0.15 * len(matched_names))
            elif _location_event_query(lower, content):
                score -= 0.28
        target_value_types = _value_history_target_type_priority(query)
        if target_value_types:
            topic_rank = min(_value_history_topic_mismatch_rank(query, content, value_type) for value_type in target_value_types)
            if topic_rank == 2:
                score -= 0.55
            elif topic_rank == 1:
                score -= 0.25
        if span.span_id in seen_spans:
            score -= 0.12
        operator_fields = exact_answer_operator_fields(query, content, speaker=speaker)
        if operator_fields:
            score += 0.72 * float(operator_fields.get("confidence") or 0.0)
        if score >= 0.20:
            scored.append((score, span))
    if not scored:
        return []
    scored.sort(key=lambda item: (item[0], _reverse_timeline_key(_span_sort_record(item[1]))), reverse=True)
    out: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    ordered_spans = [
        span
        for span in sorted(scope_spans, key=lambda span: _timeline_sort_key(_span_sort_record(span)))
        if getattr(span, "span_type", "") in {"turn", "tool_result", "document_chunk"}
    ]
    history_indices = {str(getattr(span, "span_id", "") or ""): index for index, span in enumerate(ordered_spans, start=1)}
    for score, span in scored:
        span_content = str(span.content)
        label = compact_summary(span_content, 180).lower()
        if label in seen_content:
            continue
        seen_content.add(label)
        out.append(_exact_candidate_record(query, lower, span, score, history_indices))
        for support in _adjacent_exact_answer_support_spans(query, span, ordered_spans):
            support_content = str(getattr(support, "content", "") or "")
            support_label = compact_summary(support_content, 180).lower()
            if support_label in seen_content:
                continue
            seen_content.add(support_label)
            out.append(_exact_candidate_record(query, lower, support, max(0.20, score - 0.05), history_indices, adjacent_support_for=span))
        if len(out) >= 12:
            break
    return out[:12]


def _exact_candidate_record(
    query: str,
    query_lower: str,
    span: Any,
    score: float,
    history_indices: dict[str, int],
    *,
    adjacent_support_for: Any | None = None,
) -> dict[str, Any]:
    span_content = str(getattr(span, "content", "") or "")
    record = {
        "source_span_id": span.span_id,
        "speaker": span.speaker,
        "score": round(float(score), 4),
        "update_marker_strength": _value_update_marker_strength(query_lower, span_content.lower(), ""),
        "value_mentions": _dedupe_value_mentions(_query_targeted_value_mentions(query, span_content) + _value_mentions(span_content)),
        "content": compact_summary(span_content, 2600),
        "source_uri": span.source_uri,
        "turn_id": span.turn_id,
    }
    operator_fields = exact_answer_operator_fields(query, span_content, speaker=str(getattr(span, "speaker", "") or ""))
    if operator_fields:
        record.update(operator_fields)
    history_index = history_indices.get(str(getattr(span, "span_id", "") or ""))
    if history_index is not None:
        record["history_index"] = history_index
    if adjacent_support_for is not None:
        record["candidate_source"] = "adjacent_exact_answer_support"
        record["supports_source_span_id"] = getattr(adjacent_support_for, "span_id", None)
    return record


def _adjacent_exact_answer_support_spans(query: str, span: Any, ordered_spans: list[Any]) -> list[Any]:
    if getattr(span, "speaker", "") not in {"user", "document"}:
        return []
    content = str(getattr(span, "content", "") or "")
    if not _exact_candidate_user_turn_can_have_assistant_support(query, content):
        return []
    try:
        index = next(i for i, item in enumerate(ordered_spans) if getattr(item, "span_id", None) == getattr(span, "span_id", None))
    except StopIteration:
        return []
    span_group = _span_group_key(getattr(span, "source_uri", None), getattr(span, "turn_id", None))
    out: list[Any] = []
    for candidate in ordered_spans[index + 1 : index + 5]:
        if getattr(candidate, "speaker", "") in {"user", "document"}:
            break
        if span_group and _span_group_key(getattr(candidate, "source_uri", None), getattr(candidate, "turn_id", None)) != span_group:
            continue
        if _exact_assistant_support_matches(query, content, str(getattr(candidate, "content", "") or "")):
            out.append(candidate)
            break
    return out


def _exact_candidate_user_turn_can_have_assistant_support(query: str, content: str) -> bool:
    lower = content.lower()
    query_lower = query.lower()
    if not re.search(r"\b(?:recommend|suggest|help|what|which|how|can you|could you|should)\b", lower):
        return False
    if re.search(r"\b(?:recommend|suggest|options?|ideas?|steps?|plan|schedule|timeline|list|find|available|availability)\b", lower):
        return True
    return bool(
        re.search(r"\b(?:recommend|suggest|options?|steps?|plan|schedule|timeline|how did|what did)\b", query_lower)
    )


def _exact_assistant_support_matches(query: str, request: str, content: str) -> bool:
    lower = content.lower()
    if not content.strip():
        return False
    query_terms = _expand_topic_tokens(_topic_scope_tokens(query))
    request_terms = _expand_topic_tokens(_topic_scope_tokens(request))
    content_terms = _expand_topic_tokens(_topic_scope_tokens(content))
    request_topical = bool(query_terms & request_terms)
    topical = bool((query_terms & content_terms) or (request_terms & content_terms) or request_topical)
    answer_shape = bool(
        re.search(r"\b(?:recommend|suggest|here are|steps?|plan|schedule|timeline|availability|available|options?|list)\b", lower)
        or len(generic_list_candidate_keys(query.lower(), content)) >= 2
        or len(re.findall(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+\S", content)) >= 2
    )
    return topical and answer_shape


def _location_event_query(query_lower: str, content: str) -> bool:
    if not re.search(r"\b(?:where|take place|takes place|planning|planned|event|events?|celebration|dinner|getaway|trip)\b", query_lower):
        return False
    content_lower = content.lower()
    if not re.search(r"\b(?:at|to|in)\s+(?:the\s+)?[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,5}", content):
        return False
    if re.search(r"\b(?:planning|planned|upcoming|scheduled|event|celebration|dinner|getaway|trip|picnic|workshop|meeting)\b", content_lower):
        return True
    return bool(re.search(r"\bwith\s+[A-Z][A-Za-z' -]{2,40}\b", content))


def _query_proper_names(query: str) -> set[str]:
    names = {
        match.group(0)
        for match in re.finditer(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b", query)
        if match.group(0).lower()
        not in {
            "What",
            "Where",
            "When",
            "How",
            "Can",
            "Could",
            "Should",
            "Would",
            "Mention",
            "ONLY",
        }
    }
    return names


def _assistant_recommendation_domain_signal(query_lower: str, content_lower: str) -> float:
    score = 0.0
    if re.search(r"\b(?:work|job|role|company|career|transition|environment)\b", query_lower):
        for pattern in [
            r"\bpreparing for the transition\b",
            r"\bdue diligence\b",
            r"\bresearch (?:the )?company\b",
            r"\bmission\b",
            r"\bvalues\b",
            r"\bfinancial health\b",
            r"\bcurrent employees?\b",
            r"\bworkload\b",
            r"\bhours\b",
            r"\bperformance metrics?\b",
        ]:
            if re.search(pattern, content_lower):
                score += 0.10
    if re.search(r"\b(?:essay|writing|draft|submission|scholarship|application)\b", query_lower):
        for pattern in [
            r"\btimeline\b",
            r"\binitial draft\b",
            r"\bfirst draft\b",
            r"\bsecond draft\b",
            r"\bfinal (?:review|edits?)\b",
            r"\bsubmission\b",
            r"\bdeadline\b",
            r"\bcutoff\b",
            r"\bscholarship\b",
            r"\bapplication\b",
        ]:
            if re.search(pattern, content_lower):
                score += 0.09
    if re.search(r"\b(?:partner|shared interests?|movie|movies?|film|films?|evening|date night)\b", query_lower):
        for pattern in [
            r"\bshared interests?\b",
            r"\bclassic movies?\b",
            r"\btimeless classics?\b",
            r"\bnostalgic\b",
            r"\bmet\b",
            r"\blove for\b",
            r"\bpartner\b",
            r"\bdate night\b",
        ]:
            if re.search(pattern, content_lower):
                score += 0.08
    if re.search(r"\b(?:hiring|candidate|evaluation|fairness|fair|bias|screening|pilot|time-to-hire)\b", query_lower):
        for pattern in [
            r"\bpilot\b",
            r"\btime-?to-?hire\b",
            r"\bfair(?:ness)?\b",
            r"\bbias\b",
            r"\bhuman (?:oversight|review)\b",
            r"\bcandidate evaluation\b",
            r"\bshortlist\b",
            r"\btransparent\b",
        ]:
            if re.search(pattern, content_lower):
                score += 0.08
    return min(score, 0.45)


def _process_plan_signal(query_lower: str, content_lower: str) -> float:
    score = 0.0
    if re.search(r"\b(?:timeline|schedule|plan|process|structure)\b", content_lower):
        score += 0.18
    if re.search(r"\b(?:initial|first|second|final)\s+draft\b", content_lower):
        score += 0.18
    if re.search(r"\binitial draft\b", content_lower) and re.search(r"\b(?:writing|process|organizing|submission)\b", query_lower):
        score += 0.48
    if re.search(r"\bpersonal statement\b", content_lower) and re.search(r"\b(?:writing|essay|application)\b", query_lower):
        score += 0.10
    if re.search(r"\b(?:review|revision|edits?|feedback)\b", content_lower):
        score += 0.12
    if re.search(r"\b(?:submission|submit|deadline|cutoff)\b", content_lower):
        score += 0.12
    date_mentions = re.findall(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
        content_lower,
    )
    if date_mentions:
        score += 0.16
    if len(date_mentions) >= 3:
        score += 0.34
    month_mentions = {
        match[:3]
        for match in re.findall(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", content_lower)
    }
    if len(month_mentions) >= 2:
        score += 0.16
    if len(month_mentions) >= 3:
        score += 0.12
    if (
        re.search(r"\b(?:initial|first|second|final)\s+draft\b", content_lower)
        and re.search(r"\b(?:review|revision|edits?|feedback)\b", content_lower)
        and re.search(r"\b(?:submission|submit|deadline|cutoff)\b", content_lower)
    ):
        score += 0.24
    if re.search(r"\b\d+\.\s+|\n\s*[-*]\s+", content_lower):
        score += 0.08
    if re.search(r"\b(?:cutoff|dates?|deadline|each step|well before)\b", query_lower) and not date_mentions:
        score -= 0.20
    if re.search(r"\b(?:writing|organizing|structure|process)\b", query_lower) and not re.search(
        r"\b(?:draft|personal statement|writing)\b",
        content_lower,
    ):
        score -= 0.22
    if re.search(r"\bsubmitting?\s+(?:a\s+)?(?:scholarship|application)\b", content_lower) and not date_mentions:
        score -= 0.18
    if re.search(r"\b(?:uploading documents|gather (?:necessary|required )?documents|double-check all documents)\b", content_lower) and not re.search(
        r"\binitial draft\b",
        content_lower,
    ):
        score -= 0.28
    if re.search(r"\b(?:essay|writing|draft|submission|scholarship|application)\b", query_lower) and not re.search(
        r"\b(?:essay|writing|draft|submission|scholarship|application|personal statement)\b",
        content_lower,
    ):
        score -= 0.14
    return max(-0.34, min(score, 1.30))


def _milestone_group_from_text(text: str) -> str | None:
    match = re.search(r"\bMilestone\s+\[([^\]]+)\]", text)
    return match.group(1) if match else None


def _contradiction_claim_buckets(query: str, spans: list[dict], facts: list[dict]) -> list[dict]:
    positive: list[str] = []
    negative: list[str] = []
    uncertain: list[str] = []
    for span in spans:
        polarity = _claim_polarity(query, span.get("content", ""))
        span_id = span.get("id")
        if not span_id:
            continue
        span["claim_polarity"] = polarity
        if polarity == "positive":
            positive.append(str(span_id))
        elif polarity == "negative":
            negative.append(str(span_id))
        else:
            uncertain.append(str(span_id))
    return [
        {
            "type": "claim_polarity_buckets",
            "positive_source_span_ids": positive[:8],
            "negative_source_span_ids": negative[:8],
            "uncertain_source_span_ids": uncertain[:8],
            "fact_source_span_ids": [span_id for fact in facts[:6] for span_id in fact.get("source_span_ids", [])],
            "note": "Buckets organize retrieved raw claims by surface polarity; they do not decide the answer.",
        }
    ]


def _claim_polarity(query: str, content: str) -> str:
    lower = content.lower()
    query_tokens = _topic_scope_tokens(query)
    text_tokens = _topic_scope_tokens(content)
    if query_tokens and len(query_tokens & text_tokens) == 0:
        return "uncertain"
    negative_patterns = [
        r"\bnever\b",
        r"\bnot\s+(?:yet\s+)?(?:used|integrated|worked|read|listened|met|downloaded|placed|attended|completed|tested|made|drafted|managed)\b",
        r"\bhaven['’]?t\b",
        r"\bhave\s+not\b",
        r"\bno\s+experience\b",
        r"\bwithout\s+(?:using|having|integrating|testing)\b",
    ]
    positive_patterns = [
        r"\b(?:used|integrated|worked|read|listened|met|downloaded|placed|attended|completed|tested|made|drafted|managed)\b",
        r"\bstarted\s+(?:using|listening|reading|working|testing)\b",
        r"\b(?:have|has|had|i['’]?ve|we['’]?ve)\s+been\s+(?:using|tracking|reading|listening|working|testing|attending|meeting|drafting|managing)\b",
        r"\b(?:am|are|is|was|were)\s+(?:using|tracking|reading|listening|working|testing|attending|meeting|drafting|managing)\b",
        r"\bhas\s+been\s+(?:used|integrated|tested|completed)\b",
        r"\balready\s+(?:used|integrated|tested|completed|started|drafted)\b",
    ]
    neg = any(re.search(pattern, lower) for pattern in negative_patterns)
    pos = any(re.search(pattern, lower) for pattern in positive_patterns)
    if neg:
        return "negative"
    if pos:
        return "positive"
    return "uncertain"


def _plan_needs_value_history(query_type: str, query: str, intent: dict[str, Any]) -> bool:
    if query_type == "knowledge_update":
        return True
    lower = query.lower()
    if _historical_said_query(lower):
        return False
    if bool(intent.get("needs_current_state")):
        return True
    aggregation = intent.get("aggregation") if isinstance(intent.get("aggregation"), dict) else {}
    if aggregation.get("operation") not in {None, "", "none"}:
        return True
    if query_type in {"factual_exact", "temporal_lookup", "instruction"} and re.search(
        r"\b(?:what\s+is|what\s+are|which|how\s+many|how\s+much|current|currently|latest|"
        r"what\s+time|quota|coverage|percentage|percent|version|libraries?|dependencies?|scheduled|per\s+week|per\s+day)\b",
        lower,
    ):
        return True
    return False


def _merge_value_history_rows(
    rows: list[dict[str, Any]],
    recall_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in [*recall_rows, *rows]:
        key = (
            str(row.get("source_span_id") or row.get("subject_key") or ""),
            str(row.get("value_type") or ""),
            str(row.get("value") or "").lower(),
        )
        if not key[2] or key in seen:
            continue
        seen.add(key)
        merged.append(row)
    merged.sort(
        key=lambda item: (
            0 if item.get("candidate_source") == "slot_update_recall" else 1,
            -float(item.get("recall_score") or item.get("update_marker_strength") or 0.0),
            0 if item.get("speaker") in {"user", "document", "fact"} else 1,
            int(item.get("recency_rank") or 10**9),
            -int(item.get("history_index") or -1),
        )
    )
    return merged[:limit]


def _query_needs_current_value_resolution(query: str, intent: dict[str, Any]) -> bool:
    if bool(intent.get("needs_current_state")):
        return True
    return bool(re.search(r"\b(?:recent|current|currently|latest|now|updated|newest|final|finally|revised|rescheduled)\b", query.lower()))


def _is_stale_historical_current_value_span(text: str) -> bool:
    lower = text.lower()
    if not lower.strip():
        return False
    if re.search(
        r"\b(?:switched\b.*\bfrom\b.*\bto\b|switch(?:ed)?\b.*\bto\b|migrat(?:e|ed|ing)\b|"
        r"changed\b.*\bto\b|no\s+longer|not\s+anymore|historical context|current(?:ly)?|now|latest|updated)\b",
        lower,
    ):
        return False
    if re.search(
        r"(?:不是之前说的|不再是|改(?:成|为)|更新(?:为|成)|变(?:成|为)|切换(?:为|成))",
        text,
    ):
        return False
    if re.search(r"(?:从[^。！？\n]{0,40}(?:改(?:成|为)|更新(?:为|成)|变(?:成|为)|切换(?:为|成)))", text):
        return False
    if re.search(r"(?:\b(?:updated?|revised?|changed?|switched|migrat(?:e|ed|ing))\b.*\b(?:from|to)\b)", lower):
        return False
    is_stale = bool(
        re.search(
            r"\b(?:initially|previously|formerly|originally|used to|before the switch|at first)\b",
            lower,
        )
        or re.search(r"(?:最初|以前|之前|原来|曾经)", text)
    )
    if is_stale:
        record_rule_hit(
            "current_value.stale_history_marker",
            query="",
            text=text,
            stage="evidence_pack_filter",
            metadata={"decision": "drop_stale_history"},
        )
    return is_stale


def _historical_said_query(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:did\s+i\s+say|what\s+did\s+i\s+say|i\s+said|i\s+mentioned|i\s+told\s+you)\b", lower)
        or re.search(r"我.*(?:说过|提到过|告诉过)", lower)
    ) and not bool(re.search(r"\b(?:current|currently|latest|recent|recently|now|updated|newest)\b", lower))


def _format_requirements(query: str) -> list[str]:
    lower = query.lower()
    requirements: list[str] = []
    if "only and only" in lower or re.search(r"\bmention only\b", lower):
        requirements.append("exact_item_count_or_only_constraint")
    if re.search(r"\b(?:code|function|snippet|program)\b", lower):
        requirements.append("code_or_snippet_expected")
    if "```" in query or "fenced" in lower:
        requirements.append("fenced_code_block")
    if re.search(r"\b(?:tree drawing|diagram|table|bullet|list)\b", lower):
        requirements.append("specific_visual_or_list_format")
    if re.search(r"\b(?:version|libraries|dependencies)\b", lower):
        requirements.append("include_exact_versions_if_supported")
    return requirements


TOPIC_SCOPE_STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "and",
    "answer",
    "approach",
    "approached",
    "approaches",
    "aspect",
    "aspects",
    "before",
    "been",
    "between",
    "brought",
    "can",
    "challenge",
    "challenges",
    "chat",
    "chats",
    "comprehensive",
    "conversation",
    "conversations",
    "deadline",
    "deadlines",
    "developed",
    "development",
    "different",
    "does",
    "during",
    "each",
    "for",
    "feature",
    "features",
    "final",
    "finish",
    "finished",
    "finishing",
    "from",
    "give",
    "have",
    "have",
    "help",
    "how",
    "include",
    "including",
    "into",
    "item",
    "items",
    "key",
    "list",
    "many",
    "management",
    "mention",
    "mentioned",
    "need",
    "only",
    "order",
    "our",
    "over",
    "project",
    "projects",
    "request",
    "requests",
    "resolve",
    "resolved",
    "resolves",
    "should",
    "so",
    "summary",
    "summarize",
    "target",
    "targets",
    "the",
    "through",
    "throughout",
    "time",
    "used",
    "using",
    "various",
    "walk",
    "want",
    "wanted",
    "way",
    "ways",
    "week",
    "weeks",
    "what",
    "which",
    "with",
    "work",
    "worked",
    "you",
}

TOPIC_SCOPE_EQUIVALENTS = {
    "auth": {"auth", "authentication", "login", "logout", "session"},
    "authentication": {"auth", "authentication", "login", "logout", "session"},
    "columns": {"column", "columns", "field", "fields"},
    "deadline": {"deadline", "deadlines", "due", "target"},
    "deployment": {"deployment", "deploy", "deployed", "launch", "production", "render", "gunicorn"},
    "features": {"feature", "features", "module", "modules", "functionality"},
    "estate": {"estate", "will", "probate", "assets", "asset", "property", "trust"},
    "assets": {"assets", "asset", "items", "property", "home", "vehicle", "equipment", "safe", "will"},
    "items": {"items", "assets", "asset", "property", "home", "vehicle", "equipment", "safe", "will"},
    "financial": {"financial", "finance", "budget", "money", "cost", "costs", "income", "expense", "expenses"},
    "finish": {"finish", "finished", "complete", "completed", "completion", "end", "ended"},
    "latency": {"latency", "response", "time", "ms", "milliseconds"},
    "linkedin": {"linkedin", "profile", "resume", "portfolio", "career", "cv"},
    "portfolio": {"portfolio", "profile", "resume", "linkedin", "career", "cv"},
    "profession": {"profession", "job", "career", "role", "work"},
    "profile": {"profile", "resume", "portfolio", "linkedin", "career", "cv"},
    "resume": {"resume", "profile", "portfolio", "linkedin", "career", "cv"},
    "sprint": {"sprint", "sprints", "phase", "milestone"},
    "stress": {"stress", "stressed", "burnout", "overwhelmed", "workload"},
    "transaction": {"transaction", "transactions", "crud", "income", "expense", "expenses"},
}


def _topic_scope_score(query: str, text: str) -> float:
    query_tokens = _topic_scope_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _topic_scope_tokens(text)
    if not text_tokens:
        return 0.0
    direct = len(query_tokens & text_tokens) / max(1, len(query_tokens))
    expanded = len(_expand_topic_tokens(query_tokens) & _expand_topic_tokens(text_tokens)) / max(1, len(_expand_topic_tokens(query_tokens)))
    return min(1.0, (0.70 * direct) + (0.30 * expanded))


def _topic_scope_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", text.lower())
    tokens: set[str] = set()
    for token in raw:
        if re.search(r"[\u4e00-\u9fff]", token):
            tokens.add(token)
            for size in (2, 3, 4):
                tokens.update(token[index : index + size] for index in range(0, max(0, len(token) - size + 1)))
            continue
        token = token.strip("_+-")
        if len(token) < 3 or token in TOPIC_SCOPE_STOPWORDS:
            continue
        tokens.add(token)
        if token.endswith("s") and len(token) > 4:
            tokens.add(token[:-1])
        if token.endswith("ing") and len(token) > 6:
            tokens.add(token[:-3])
        if token.endswith("ed") and len(token) > 5:
            tokens.add(token[:-2])
    return tokens


def _expand_topic_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        equivalents = TOPIC_SCOPE_EQUIVALENTS.get(token)
        if equivalents:
            expanded.update(equivalents)
    return expanded


def _exact_overlap(query: str, text: str) -> float:
    query_tokens = _topic_scope_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _topic_scope_tokens(text)
    return len(query_tokens & text_tokens) / max(1, len(query_tokens))


def _summary_pack_signal(query: str, text: str) -> float:
    query_lower = query.lower()
    lower = text.lower()
    signal = 0.0
    if re.search(r"\b(?:issues?|problems?|challenges?|resolved?|fix(?:ed|es)?|debug|trouble)\b", query_lower):
        if re.search(r"\b(?:issues?|problems?|challenges?|errors?|exceptions?|warnings?|debug|trouble|fix|resolve|broken|failed|failures?)\b", lower):
            signal += 0.40
    if re.search(r"\b(?:progress|develop(?:ed|ment)?|evolved?|over time|throughout|summary|summarize|complete)\b", query_lower):
        if re.search(r"\b(?:started|planned|decided|chose|implemented|added|updated|improved|reduced|increased|completed|finalized|prepared|scheduled|registered|attended)\b", lower):
            signal += 0.24
    if re.search(r"\b(?:plans?|preparations?|decisions?|strategy|strategic|milestones?)\b", query_lower):
        if re.search(r"\b(?:plan|prepare|decision|decided|strategy|milestone|deadline|meeting|interview|session|review|budget)\b", lower):
            signal += 0.24
    if re.search(r"\b(?:\d+(?:\.\d+)?\s*(?:%|kb|mb|s|ms|seconds?|minutes?|hours?|days?|weeks?|months?)|v?\d+\.\d+(?:\.\d+)?)\b", text, flags=re.I):
        signal += 0.16
    if _date_signal(text) > 0:
        signal += 0.14
    if re.search(r"\b[A-Z][A-Za-z0-9.+#-]{2,}(?:\s+v?\d+(?:\.\d+)*)?\b", text):
        signal += 0.08
    return min(1.0, signal)


def _aggregation_anchor_diverse_candidates(
    query: str,
    scored: list[tuple[float, object]],
    *,
    limit: int,
) -> list[tuple[float, object]]:
    if limit <= 0 or not scored:
        return []
    anchor_terms = _aggregation_anchor_terms(query)
    if not anchor_terms:
        return scored[:limit]
    ordered = list(scored)
    selected: list[tuple[float, object]] = []
    covered: set[str] = set()
    while ordered and len(selected) < limit:
        best_index = 0
        best_key: tuple[float, float, float, float, str] | None = None
        for index, (score, span) in enumerate(ordered):
            text = str(getattr(span, "content", "") or "")
            terms = _topic_scope_tokens(text)
            overlap = anchor_terms & terms
            if covered and not overlap and len(selected) < len(anchor_terms):
                anchor_bonus = 0.0
            else:
                anchor_bonus = float(len(overlap - covered))
            has_new_anchor = 1.0 if anchor_bonus > 0 else 0.0
            user_bonus = 1.0 if getattr(span, "speaker", None) == "user" else 0.0
            key = (
                has_new_anchor,
                user_bonus,
                anchor_bonus,
                score,
                _aggregation_pack_signal(query, text),
                str(getattr(span, "id", "") or getattr(span, "span_id", "")),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        _score, chosen = ordered.pop(best_index)
        selected.append((_score, chosen))
        covered.update(_aggregation_anchor_terms(str(getattr(chosen, "content", "") or "")) & anchor_terms)
    if len(selected) < limit:
        seen = {str(getattr(item[1], "id", "") or getattr(item[1], "span_id", "")) for item in selected}
        for item in scored:
            span = item[1]
            span_id = str(getattr(span, "id", "") or getattr(span, "span_id", ""))
            if span_id in seen:
                continue
            selected.append(item)
            seen.add(span_id)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _aggregation_anchor_terms(query: str) -> set[str]:
    terms = _topic_scope_tokens(query)
    return {
        term
        for term in terms
        if term not in {
            "how",
            "what",
            "when",
            "where",
            "why",
            "which",
            "will",
            "would",
            "could",
            "should",
            "can",
            "many",
            "different",
            "total",
            "across",
            "throughout",
            "over",
            "time",
            "sessions",
            "session",
            "requests",
            "request",
            "things",
            "items",
            "features",
            "concerns",
            "topics",
            "areas",
            "aspects",
            "list",
            "count",
            "number",
            "ability",
            "still",
        }
    }


def _summary_diverse_candidates(
    scored: list[tuple[float, object]],
    *,
    current_spans: list[dict],
    ordered_spans: list[object],
    limit: int,
) -> list[tuple[float, object]]:
    if limit <= 0:
        return []
    remaining = list(scored)
    selected: list[tuple[float, object]] = []
    pair_keys = _summary_pair_keys(ordered_spans)
    selected_pair_keys = {
        pair_keys.get(str(span.get("id") or ""), str(span.get("id") or ""))
        for span in current_spans
        if span.get("id")
    }
    selected_token_sets = [
        _topic_scope_tokens(str(span.get("content") or span.get("text") or ""))
        for span in current_spans
    ]
    while remaining and len(selected) < limit:
        best_index = 0
        best_key: tuple[float, float, float, str] | None = None
        for index, (base_score, span) in enumerate(remaining):
            span_pair_key = pair_keys.get(str(getattr(span, "span_id", "") or ""))
            if span_pair_key and span_pair_key in selected_pair_keys:
                continue
            text = str(getattr(span, "content", "") or "")
            tokens = _topic_scope_tokens(text)
            max_similarity = 0.0
            if tokens:
                for selected_tokens in selected_token_sets:
                    if not selected_tokens:
                        continue
                    similarity = len(tokens & selected_tokens) / max(1, len(tokens | selected_tokens))
                    max_similarity = max(max_similarity, similarity)
            novelty = 1.0 - max_similarity if tokens else 0.0
            issue_signal = 0.12 if _summary_issue_signal(text) else 0.0
            detail_signal = _summary_detail_signal(text)
            adjusted = (0.54 * base_score) + (0.24 * novelty) + issue_signal + (0.18 * detail_signal)
            key = (
                adjusted,
                base_score,
                novelty,
                str(getattr(span, "span_id", "")),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_key is None:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        chosen_pair_key = pair_keys.get(str(getattr(chosen[1], "span_id", "") or ""))
        if chosen_pair_key:
            selected_pair_keys.add(chosen_pair_key)
        selected_token_sets.append(_topic_scope_tokens(str(getattr(chosen[1], "content", "") or "")))
    return selected


def _summary_pair_keys(ordered_spans: list[object]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for index, span in enumerate(ordered_spans):
        span_id = str(getattr(span, "span_id", "") or "")
        if not span_id:
            continue
        previous_span = ordered_spans[index - 1] if index > 0 else None
        next_span = ordered_spans[index + 1] if index + 1 < len(ordered_spans) else None
        if getattr(span, "speaker", None) == "user" and next_span is not None and getattr(next_span, "speaker", None) == "assistant":
            keys[span_id] = span_id
        elif getattr(span, "speaker", None) == "assistant" and previous_span is not None and getattr(previous_span, "speaker", None) == "user":
            keys[span_id] = str(getattr(previous_span, "span_id", "") or span_id)
        else:
            keys[span_id] = span_id
    return keys


def _summary_issue_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:issues?|problems?|errors?|exceptions?|warnings?|debug|trouble|fix|resolve|resolved|broken|failed|failures?|not\s+working)\b",
            text,
            flags=re.I,
        )
    )


def _summary_detail_signal(text: str) -> float:
    signal = 0.0
    if _date_signal(text) > 0 or re.search(r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*%", text):
        signal += 0.25
    if re.search(r"\b(?:deadline|offer|interview|workshop|meeting|event|review|feedback|mentor|policy|format|presentation|networking)\b", text, flags=re.I):
        signal += 0.25
    if re.search(r"\b(?:i(?:'m| am)?\s+(?:thinking|trying|planning|worried|excited|deciding|stuck)|should i|how can i|what should i)\b", text, flags=re.I):
        signal += 0.25
    proper_phrases = re.findall(r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*)+\b", text)
    if len(proper_phrases) >= 2:
        signal += 0.25
    return min(1.0, signal)


def _summary_pair_content(span, ordered_spans: list[object], limit: int) -> str:
    try:
        index = next(i for i, item in enumerate(ordered_spans) if item.span_id == span.span_id)
    except StopIteration:
        return ""
    previous_span = ordered_spans[index - 1] if index > 0 else None
    next_span = ordered_spans[index + 1] if index + 1 < len(ordered_spans) else None
    if span.speaker == "user" and next_span is not None and getattr(next_span, "speaker", None) == "assistant":
        return compact_summary(f"User: {span.content}\nAssistant: {next_span.content}", limit)
    if span.speaker == "assistant" and previous_span is not None and getattr(previous_span, "speaker", None) == "user":
        return compact_summary(f"User: {previous_span.content}\nAssistant: {span.content}", limit)
    return ""


def _aggregation_content_summary(query: str, text: str, limit: int) -> str:
    lower_query = query.lower()
    if is_generic_count_or_list_query(lower_query):
        query_terms = _topic_scope_tokens(query)
        include_patterns = [
            r'"[^"\n]{2,80}"',
            r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+[^\n]{3,140}",
            r"\b(?:selected|chose|decided|planned|finalized|mentioned|listed|included|added|tracked|submitted|ordered)\b",
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|months?|weeks?|days?|hours?|monthly|per month|per year|items?|options?|entries?)\b",
        ]
        if query_terms:
            include_patterns.append(r"\b(?:" + "|".join(re.escape(term) for term in sorted(query_terms, key=len, reverse=True)[:16]) + r")\b")
        return _line_focus_summary(text, limit, include_patterns=include_patterns)
    return ""


def _line_focus_summary(text: str, limit: int, *, include_patterns: list[str]) -> str:
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if len(lines) <= 1:
        chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|(?=\s*#{2,}|\s*\d+\.\s+\*\*)", text) if chunk.strip()]
    else:
        chunks = lines
    kept: list[str] = []
    for chunk in chunks:
        if any(re.search(pattern, chunk, flags=re.I) for pattern in include_patterns):
            kept.append(chunk)
    if not kept:
        return ""
    return compact_summary(" ".join(kept), limit)


def _aggregation_pack_signal(query: str, text: str) -> float:
    lower_query = query.lower()
    lower = text.lower()
    signal = 0.0
    generic_keys = generic_aggregation_keys(query, lower, speaker="user" if re.search(r"\b(?:i|my|we|our)\b", lower) else None)
    if generic_keys:
        signal += min(0.70, 0.26 * len(generic_keys))
    synthesis = _synthesis_pack_signal(query, text)
    if synthesis:
        signal += synthesis
    if re.search(r"\b(?:how many|total|unique|count|number of|different|across|throughout)\b", lower_query):
        if re.search(r"\b\d+(?:,\d{3})*\b", lower):
            signal += 0.25
        if re.search(r"\b(?:how many|total|unique|count|number of|different|ways?|items?|options?|topics?|calculations?|movies?|books?|series|genres|days?|columns?|features?)\b", lower):
            signal += 0.35
    if _is_combinatorics_aggregation_query(lower_query):
        keys = combinatorics_aggregation_keys(lower)
        if keys:
            signal += min(0.40, 0.15 * len(keys))
        if re.search(r"\b(?:\d+\s*c\s*\d+|\d+c\d+|n!|\d+!|choose\s+\d+|choosing\s+\d+|draw\s+\d+|arrange\s+\d+|arranging\s+\d+)\b", lower):
            signal += 0.20
        if re.search(r"\b(?:i(?:'m| am)?\s+trying|can you help|would i use|i want|i came across)\b", lower):
            signal += 0.12
    if _is_stress_break_aggregation_query(lower_query):
        keys = stress_break_aggregation_keys(lower)
        if keys:
            signal += min(0.45, 0.20 * len(keys))
        if re.search(r"\b(?:i\s+took|i\s+had\s+to|i(?:'m| am)?\s+feeling|prevent burnout|manage stress)\b", lower):
            signal += 0.18
    if is_generic_count_or_list_query(lower_query):
        keys = generic_list_candidate_keys(lower_query, lower)
        if keys:
            signal += min(0.44, 0.10 * len(keys))
        if re.search(r"\b(?:selected|chose|decided|planned|finalized|mentioned|listed|included|added|tracked|submitted|ordered)\b", lower):
            signal += 0.18
    query_terms = _topic_scope_tokens(query)
    text_terms = _topic_scope_tokens(text)
    if query_terms:
        signal += min(0.35, len(query_terms & text_terms) / max(1, len(query_terms)))
    if re.search(r"\b(?:also|another|across|in another|previously|earlier|later|April|May|June|July|August|September|October|November|December)\b", text):
        signal += 0.15
    return min(1.0, signal)


def _synthesis_pack_signal(query: str, text: str) -> float:
    query_lower = query.lower()
    lower = text.lower()
    if not re.search(r"\b(?:how|considering|given|what|which)\b", query_lower):
        return 0.0
    if not re.search(r"\b(?:i|my|we|our)\b", lower):
        return 0.0
    query_terms = _expand_topic_tokens(_topic_scope_tokens(query))
    text_terms = _expand_topic_tokens(_topic_scope_tokens(text))
    if query_terms and not (query_terms & text_terms):
        return 0.0
    signal = 0.0
    if re.search(r"\$\s?\d|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|months?|weeks?|days?|hours?|monthly|per month|per year)\b", lower):
        signal += 0.24
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}|20\d{2}\b", lower):
        signal += 0.16
    if re.search(r"\b(?:agreed|decided|chose|started|completed|increased|reduced|improved|reported|confirmed|took on|taking on|support|saving|savings|budget|income|expense|goal|deadline)\b", lower):
        signal += 0.24
    if re.search(r"\b(?:also|later|after|before|while|since|now|current|currently)\b", lower):
        signal += 0.10
    return min(0.62, signal)


def _quoted_title_candidates(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r'"([^"\n]{2,80})"', text) if match.group(1).strip()]


def _normalize_title_key(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return normalized[:60] or "untitled"


def _is_non_title_quote(title: str) -> bool:
    lower = title.lower().strip()
    return lower in {
        "netflix",
        "disney+",
        "pg",
        "pg-13",
        "r",
        "audible",
        "libby",
    } or bool(re.fullmatch(r"\d{4}", lower))


def _summary_resolution_pairs(query: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        content = str(span.get("content") or "")
        if not _summary_issue_signal(content):
            continue
        partner = None
        for candidate in spans[index + 1 : index + 4]:
            candidate_text = str(candidate.get("content") or "")
            if candidate.get("speaker") == "assistant" and (_summary_issue_signal(candidate_text) or _summary_detail_signal(candidate_text) > 0.15):
                partner = candidate
                break
        item = {
            "issue_span_id": span.get("id"),
            "issue": compact_summary(content, 240),
            "issue_speaker": span.get("speaker"),
        }
        if partner is not None:
            item["resolution_span_id"] = partner.get("id")
            item["resolution"] = compact_summary(str(partner.get("content") or ""), 240)
        pairs.append(item)
        if len(pairs) >= 12:
            break
    return pairs


def _summary_clusters(query: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not spans:
        return []
    clusters: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        key = str(span.get("topic_group") or span.get("source_uri") or span.get("session_id") or "shared")
        clusters.setdefault(key, []).append(span)
    ranked: list[dict[str, Any]] = []
    for key, items in clusters.items():
        items.sort(key=lambda item: (
            0 if item.get("speaker") == "user" else 1,
            int(item.get("timeline_index") or item.get("history_index") or 10**9),
            str(item.get("timestamp") or ""),
        ))
        representative = items[0]
        user_count = sum(1 for item in items if item.get("speaker") == "user")
        detail_count = sum(1 for item in items if _summary_detail_signal(str(item.get("content") or "")) > 0.15)
        ranked.append(
            {
                "cluster_key": key,
                "representative_span_id": representative.get("id"),
                "representative": compact_summary(str(representative.get("content") or ""), 260),
                "speaker": representative.get("speaker"),
                "span_count": len(items),
                "user_span_count": user_count,
                "detail_span_count": detail_count,
            }
        )
    ranked.sort(
        key=lambda item: (
            item.get("user_span_count", 0),
            item.get("detail_span_count", 0),
            item.get("span_count", 0),
            str(item.get("cluster_key") or ""),
        ),
        reverse=True,
    )
    return ranked[:8]


def _instruction_constraints(query: str) -> list[str]:
    constraints: list[str] = []
    lower = query.lower()
    if re.search(r"\b(?:only|exactly|just)\b", lower):
        constraints.append("exact_count_or_scope")
    if re.search(r"\b(?:bullet|list|table|tree|diagram)\b", lower):
        constraints.append("format_constraint")
    if "```" in query or re.search(r"\bcode\b", lower):
        constraints.append("code_block_expected")
    if re.search(r"\b(?:narrator|author|version|library|dependency)\b", lower):
        constraints.append("include_requested_details")
    return list(dict.fromkeys(constraints))
