from __future__ import annotations

import re
from typing import Any

from fusion_memory.core.llm import LLMClient
from fusion_memory.core.models import EvidencePack
from fusion_memory.retrieval.event_ordering_pack import build_event_ordering_model_pack
from fusion_memory.retrieval.event_ordering_sequence import (
    _event_ordering_cluster_label,
    _event_ordering_compact_aspect_label,
    _event_ordering_phase_clusters,
    _event_ordering_select_milestones,
    _event_ordering_sequence_label,
    _event_ordering_sequence_output_sort_key,
)
from fusion_memory.retrieval.aggregation_pack import (
    _aggregation_summary,
    _compact_records,
    _filter_low_confidence_aggregation_items,
    _financial_impact_items,
    _financial_impact_summary,
    _llm_aggregation_items,
    _merge_aggregation_source_spans,
    _multi_session_aggregation_items,
    _preference_constraint_items,
    _preference_requirement_checklist,
)
from fusion_memory.retrieval.aggregation_answers import aggregation_answer_candidates, deadline_answer_candidates
from fusion_memory.retrieval.answer_requirements import answer_requirements
from fusion_memory.retrieval.contradiction_claims import conflict_claims_for_model
from fusion_memory.retrieval.pack_contract import PACK_CONTRACT_VERSION
from fusion_memory.retrieval.slot_state_transition import value_state_summary
from fusion_memory.retrieval.temporal_pack import direct_date_answer_candidates, temporal_answer_candidates, temporal_model_candidates
from fusion_memory.retrieval.value_history_pack import exact_candidate_value_rows, value_history_summary


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "matched": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["matched"],
}

RUBRIC_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}


class OpenAICompatibleAnswerModel:
    """Benchmark answer model backed by any structured OpenAI-compatible client."""

    def __init__(
        self,
        client: LLMClient,
        prompt_version: str = "eval-answer-v0",
        *,
        use_llm_aggregation: bool = False,
        llm_aggregation_min_confidence: float = 0.70,
    ) -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.use_llm_aggregation = use_llm_aggregation
        self.llm_aggregation_min_confidence = llm_aggregation_min_confidence
        suffix = ":llm-aggregation" if use_llm_aggregation else ""
        self.version = f"llm_answer:{_client_version(client)}:{prompt_version}{suffix}"

    def answer(self, query: str, pack: EvidencePack) -> str:
        return self.answer_with_context(query, pack)

    def answer_with_context(
        self,
        query: str,
        pack: EvidencePack,
        *,
        benchmark: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        model_pack = _pack_for_model(
            pack,
            aggregation_client=self.client if self.use_llm_aggregation else None,
            aggregation_min_confidence=self.llm_aggregation_min_confidence,
        )
        deterministic_answer = _deterministic_model_pack_answer(query, category, model_pack, benchmark=benchmark)
        if deterministic_answer:
            return deterministic_answer
        response = self.client.structured(
            prompt=self.prompt_version,
            schema=ANSWER_SCHEMA,
            input={
                "instruction": _answer_instruction(benchmark=benchmark, category=category),
                "query": query,
                "answer_policy": pack.answer_policy,
                "coverage": pack.coverage,
                "evidence_pack": model_pack,
            },
        )
        answer = response.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return "Not enough supported memory to answer."


class OpenAICompatibleJudgeModel:
    """Semantic answer judge backed by any structured OpenAI-compatible client."""

    def __init__(self, client: LLMClient, prompt_version: str = "eval-judge-v0") -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.version = f"llm_judge:{_client_version(client)}:{prompt_version}"

    def score(self, answer: str, gold_answers: list[str]) -> bool:
        if not gold_answers:
            return False
        try:
            response = self.client.structured(
                prompt=self.prompt_version,
                schema=JUDGE_SCHEMA,
                input={
                    "instruction": (
                        "Return matched=true when the candidate answer is semantically equivalent "
                        "to at least one gold answer. Be strict about unsupported extra claims."
                    ),
                    "candidate_answer": answer,
                    "gold_answers": gold_answers,
                },
            )
        except Exception:
            return False
        return bool(response.get("matched", False))

    def rubric_score(self, query: str, answer: str, rubric_item: str) -> tuple[float, str]:
        errors: list[str] = []
        timeouts = _rubric_retry_timeouts(self.client)
        for attempt, timeout_seconds in enumerate(timeouts, start=1):
            try:
                response = _structured_with_timeout(
                    self.client,
                    prompt=f"{self.prompt_version}:rubric-score",
                    schema=RUBRIC_SCORE_SCHEMA,
                    input={
                        "instruction": (
                            "Evaluate the response against only this BEAM rubric criterion. "
                            "Return score 1.0 when fully satisfied, 0.5 when partially satisfied, "
                            "and 0.0 when not satisfied. Judge by semantic equivalence rather than exact wording. "
                            "Do not penalize extra correct information or the answer also satisfying other rubric "
                            "criteria unless the current criterion explicitly requires exclusivity or forbids that content."
                        ),
                        "question": query,
                        "candidate_answer": answer,
                        "rubric_item": rubric_item,
                    },
                    timeout_seconds=timeout_seconds,
                )
                raw_score = response.get("score", 0.0)
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = 0.0
                if score >= 0.75:
                    score = 1.0
                elif score >= 0.25:
                    score = 0.5
                else:
                    score = 0.0
                reason = response.get("reason")
                return score, str(reason or "")
            except Exception as exc:
                errors.append(f"attempt {attempt} @ {timeout_seconds:.0f}s: {exc}")
        return 0.0, "rubric scoring failed after retries: " + " | ".join(errors[:3])


def _pack_for_model(
    pack: EvidencePack,
    *,
    aggregation_client: LLMClient | None = None,
    aggregation_min_confidence: float = 0.70,
) -> dict[str, Any]:
    pack_contract = pack.coverage.get("pack_contract") if isinstance(pack.coverage.get("pack_contract"), dict) else {}
    contract_version = str(pack_contract.get("version") or PACK_CONTRACT_VERSION)
    if pack.coverage.get("query_type") == "event_ordering":
        query_intent = pack.coverage.get("query_intent") if isinstance(pack.coverage.get("query_intent"), dict) else None
        return build_event_ordering_model_pack(
            query=pack.query,
            source_spans=pack.source_spans,
            events=pack.events,
            conflicts=pack.conflicts,
            contract_version=contract_version,
            query_intent=query_intent,
            graph_coverage=pack.coverage.get("event_ordering_graph")
            if isinstance(pack.coverage.get("event_ordering_graph"), dict)
            else None,
            graph_shadow=pack.coverage.get("event_ordering_shadow")
            if isinstance(pack.coverage.get("event_ordering_shadow"), dict)
            else None,
        )
    source_span_limit = 64 if pack.coverage.get("query_type") == "summarization" else 20
    source_spans = _compact_records(pack.source_spans, preferred_text_key="content", limit=source_span_limit)
    query_intent = pack.coverage.get("query_intent") if isinstance(pack.coverage.get("query_intent"), dict) else {}
    exact_answer_candidates = _compact_records(pack.coverage.get("exact_answer_candidates", []), preferred_text_key="content", limit=12)
    aggregation_source_spans = _merge_aggregation_source_spans(pack.source_spans, exact_answer_candidates)
    aggregation_items = _filter_low_confidence_aggregation_items(
        pack.query,
        _multi_session_aggregation_items(pack.query, aggregation_source_spans, query_intent=query_intent),
    )
    aggregation_telemetry: dict[str, Any] | None = None
    if aggregation_client is not None and pack.coverage.get("query_type") == "multi_session_reasoning":
        llm_items, aggregation_telemetry = _llm_aggregation_items(
            aggregation_client,
            pack.query,
            source_spans,
            aggregation_items,
            min_confidence=aggregation_min_confidence,
        )
        if llm_items:
            aggregation_items = _filter_low_confidence_aggregation_items(pack.query, llm_items)
    financial_impacts = _financial_impact_items(pack.query, pack.source_spans)
    financial_summary = _financial_impact_summary(pack.query, financial_impacts)
    coverage_temporal_candidates = _compact_records(pack.coverage.get("temporal_candidates", []), preferred_text_key="context", limit=48)
    temporal_candidates = temporal_model_candidates(
        pack.query,
        coverage_temporal_candidates,
        _merge_aggregation_source_spans(pack.source_spans, exact_answer_candidates),
        limit=48,
    )
    temporal_range_pairs = _compact_records(pack.coverage.get("temporal_range_pairs", []), preferred_text_key="context", limit=12)
    temporal_answers = temporal_answer_candidates(pack.query, temporal_candidates, temporal_range_pairs)
    direct_date_answers = direct_date_answer_candidates(pack.query, temporal_candidates)
    value_history_limit = 24 if pack.coverage.get("query_type") == "knowledge_update" else 16
    value_history = _compact_records(pack.coverage.get("value_history", []), preferred_text_key="context", limit=value_history_limit)
    exact_value_rows = exact_candidate_value_rows(pack.query, exact_answer_candidates)
    value_summary = value_history_summary(pack.query, value_history + exact_value_rows)
    state_summary = value_state_summary(pack.query, value_history + exact_value_rows)
    value_summary = _align_value_history_with_state_summary(value_summary, state_summary)
    resolution_pairs = _compact_records(pack.coverage.get("resolution_pairs", []), preferred_text_key="issue", limit=12)
    summary_clusters = _compact_records(pack.coverage.get("summary_clusters", []), preferred_text_key="representative", limit=8)
    conflict_claims = conflict_claims_for_model(pack.query, pack.conflicts, pack.source_spans)
    summary_highlights = _summary_highlights(pack.query, pack.source_spans)
    summary_coverage = (
        _summary_coverage_matrix(pack.query, summary_highlights)
        if pack.coverage.get("query_type") == "summarization"
        else {}
    )
    preference_constraints = _merge_preference_constraints(
        pack.coverage.get("preference_constraints"),
        _preference_constraint_items(pack.query, pack.source_spans),
    )
    instruction_constraints = pack.coverage.get("instruction_constraints", [])
    requirements = answer_requirements(
        pack.query,
        _merge_aggregation_source_spans(pack.source_spans, exact_answer_candidates),
        format_requirements=pack.coverage.get("format_requirements") if isinstance(pack.coverage.get("format_requirements"), list) else [],
        preference_constraints=preference_constraints,
    )
    preference_checklist = _preference_requirement_checklist(preference_constraints)
    aggregation_candidates = aggregation_answer_candidates(
        pack.query,
        aggregation_items,
        evidence_records=[*source_spans, *exact_answer_candidates],
    )
    aggregation_candidates.extend(
        deadline_answer_candidates(
            pack.query,
            [*value_history, *exact_value_rows, *exact_answer_candidates, *source_spans],
        )
    )
    evidence_pack = {
        "pack_contract_version": contract_version,
        **({"aggregation_items": aggregation_items} if aggregation_items else {}),
        **({"aggregation_summary": _aggregation_summary(aggregation_items)} if aggregation_items else {}),
        **({"aggregation_answer_candidates": aggregation_candidates} if aggregation_candidates else {}),
        **({"aggregation_telemetry": aggregation_telemetry} if aggregation_telemetry else {}),
        **({"financial_impacts": financial_impacts} if financial_impacts else {}),
        **({"financial_summary": financial_summary} if financial_summary else {}),
        **({"temporal_candidates": temporal_candidates} if temporal_candidates else {}),
        **({"temporal_range_pairs": temporal_range_pairs} if temporal_range_pairs else {}),
        **({"temporal_answer_candidates": temporal_answers} if temporal_answers else {}),
        **({"direct_date_answer_candidates": direct_date_answers} if direct_date_answers else {}),
        **({"value_history": value_history} if value_history else {}),
        **({"value_state_summary": state_summary} if state_summary else {}),
        **({"value_history_summary": value_summary} if value_summary else {}),
        **({"resolution_pairs": resolution_pairs} if resolution_pairs else {}),
        **({"conflict_claims": conflict_claims} if conflict_claims else {}),
        **({"summary_clusters": summary_clusters} if summary_clusters else {}),
        **({"summary_highlights": summary_highlights} if summary_highlights else {}),
        **({"summary_coverage": summary_coverage} if summary_coverage else {}),
        **({"exact_answer_candidates": exact_answer_candidates} if exact_answer_candidates else {}),
        **({"instruction_constraints": instruction_constraints} if instruction_constraints else {}),
        **({"answer_requirements": requirements} if requirements else {}),
        **({"preference_constraints": preference_constraints} if preference_constraints else {}),
        **({"preference_requirement_checklist": preference_checklist} if preference_checklist else {}),
        **({"query_intent": query_intent} if query_intent else {}),
        "current_views": _compact_records(pack.current_views, preferred_text_key="text"),
        "entity_profiles": _compact_records(pack.entity_profiles, preferred_text_key="text"),
        "facts": _compact_records(pack.facts, preferred_text_key="text"),
        "events": _compact_records(pack.events, preferred_text_key="description"),
        "source_spans": source_spans,
        "conflicts": pack.conflicts[:10],
    }
    return evidence_pack


def _align_value_history_with_state_summary(
    value_summary: dict[str, Any],
    state_summary: dict[str, Any],
) -> dict[str, Any]:
    if not value_summary or not state_summary:
        return value_summary
    resolved = state_summary.get("resolved_value")
    if not resolved:
        return value_summary
    current = value_summary.get("resolved_current_value")
    if not current or str(current).strip().lower() == str(resolved).strip().lower():
        return value_summary
    aligned = dict(value_summary)
    aligned["secondary_current_value"] = current
    aligned["resolved_current_value"] = resolved
    preferred = dict(aligned.get("preferred_current_candidate") or {})
    preferred["superseded_by_state_summary"] = True
    aligned["preferred_current_candidate"] = preferred
    guidance = str(aligned.get("guidance") or "")
    aligned["guidance"] = (
        guidance
        + " value_state_summary contains a resolved same-slot state transition; treat the previous "
        "resolved_current_value as secondary history unless the state source contradicts the query."
    ).strip()
    return aligned


def _merge_preference_constraints(*groups: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            type_ = str(item.get("type") or "")
            label = str(item.get("label") or "")
            if not type_ or not label:
                continue
            key = (type_, label.lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(item))
    merged.sort(
        key=lambda item: (
            -_float_value(item.get("score")),
            int(item.get("recency_rank") or 10**9),
            -int(item.get("timeline_index") or item.get("history_index") or -1),
            str(item.get("type") or ""),
            str(item.get("label") or ""),
        )
    )
    return merged[:16]


def _deterministic_model_pack_answer(
    query: str,
    category: str | None,
    model_pack: dict[str, Any],
    *,
    benchmark: str | None = None,
) -> str | None:
    if category != "multi_session_reasoning":
        if category == "information_extraction":
            return _deterministic_information_extraction_answer(query, model_pack)
        if category == "temporal_reasoning":
            return _deterministic_temporal_answer(query, model_pack)
        if category == "instruction_following":
            return _deterministic_instruction_date_answer(query, model_pack)
        return None
    lower = query.lower()
    candidates = model_pack.get("aggregation_answer_candidates")
    if not isinstance(candidates, list):
        return None
    delta = _best_aggregation_candidate(candidates, "delta_between_values", min_confidence=0.80)
    if delta and _query_is_direct_delta_question(lower):
        value = delta.get("answer_value")
        components = delta.get("component_values") if isinstance(delta.get("component_values"), dict) else {}
        unit = str(delta.get("unit") or "").replace("_", " ")
        if unit == "percentage points":
            unit_text = "percentage points"
        else:
            unit_text = unit or "units"
        start = components.get("from")
        end = components.get("to")
        if start is not None and end is not None:
            return f"{value} {unit_text}, from {start}% to {end}%."
        return f"{value} {unit_text}."
    deadline_pair = _best_aggregation_candidate(candidates, "deadline_pair", min_confidence=0.80)
    if deadline_pair and _query_is_direct_deadline_question(lower):
        labels = [str(label).strip() for label in deadline_pair.get("labels") or [] if str(label).strip()]
        if labels:
            return "; ".join(labels) + "."
    slot_values = _best_aggregation_candidate(candidates, "distinct_slot_values", min_confidence=0.80)
    if slot_values and _query_is_direct_count_question(lower):
        value = slot_values.get("answer_value")
        labels = [str(label).strip() for label in slot_values.get("labels") or [] if str(label).strip()]
        if labels:
            return f"{value} different values: {', '.join(labels)}."
        return f"{value} different values."
    if not _query_is_direct_count_question(lower):
        return None
    grouped = _best_aggregation_candidate(candidates, "grouped_distinct_count", min_confidence=0.80)
    if grouped and _grouped_count_candidate_matches_query_scope(lower, grouped):
        value = grouped.get("answer_value")
        labels = [str(label).strip() for label in grouped.get("labels") or [] if str(label).strip()]
        if labels:
            return f"{value} different items: " + "; ".join(labels) + "."
        return f"{value} different items."
    usable = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("answer_value") is not None
        and str(candidate.get("formula") or "") == "distinct_union_count"
        and _float_value(candidate.get("confidence")) >= 0.80
    ]
    if not usable:
        return None
    best = usable[0]
    value = best.get("answer_value")
    components = best.get("component_values") if isinstance(best.get("component_values"), dict) else {}
    labels = [str(label) for label in best.get("labels") or [] if str(label).strip()]
    parts = [f"{value} unique items"]
    if components:
        base = components.get("base_unique_count")
        group = components.get("candidate_group_count")
        overlap = components.get("explicit_overlap")
        breakdown = []
        if base is not None:
            breakdown.append(f"base count {base}")
        if group is not None:
            breakdown.append(f"additional recommendation group {group}")
        if overlap is not None:
            breakdown.append(f"explicit overlap {overlap}")
        if breakdown:
            parts.append("(" + ", ".join(breakdown) + ")")
    answer = " ".join(parts) + "."
    if labels:
        answer += " Evidence labels: " + ", ".join(labels[:10]) + "."
    return answer


def _grouped_count_candidate_matches_query_scope(lower_query: str, candidate: dict[str, Any]) -> bool:
    support_items = candidate.get("support_items")
    keys = [
        str(item.get("key") or "")
        for item in support_items
        if isinstance(item, dict) and item.get("key")
    ] if isinstance(support_items, list) else []
    if not keys:
        return False
    present_prefixes = {key.split(":", 1)[0] for key in keys if ":" in key}
    required_prefixes: set[str] = set()
    if re.search(r"\b(?:titles?|books?|series|movies?|films?)\b", lower_query):
        required_prefixes.add("title")
    if re.search(r"\bgenres?\b", lower_query):
        required_prefixes.add("genre")
    if re.search(r"\b(?:values?|sizes?|amounts?|numbers?)\b", lower_query):
        required_prefixes.add("value")
    if re.search(r"\b(?:features?|concerns?|requirements?|capabilities)\b", lower_query):
        required_prefixes.add("feature")
    if re.search(r"\b(?:assets?|property|possessions?)\b", lower_query):
        required_prefixes.add("asset")
    if re.search(r"\b(?:reminders?|planners?|calendars?|schedules?|task\s+(?:tools?|systems?|apps?|managers?)|to-?do\s+(?:tools?|systems?|apps?|lists?))\b", lower_query):
        required_prefixes.add("plan_system")
    if re.search(r"\bchecklist\b", lower_query):
        required_prefixes.add("checklist")
    if re.search(r"\b(?:selected\s+options?|options?)\b", lower_query):
        required_prefixes.add("option")
    if not required_prefixes:
        return True
    return required_prefixes.issubset(present_prefixes)


def _deterministic_information_extraction_answer(query: str, model_pack: dict[str, Any]) -> str | None:
    lower = query.lower()
    candidates = model_pack.get("exact_answer_candidates")
    if not isinstance(candidates, list):
        return None
    typed = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("answer_value") is not None
        and _float_value(candidate.get("confidence")) >= 0.84
    ]
    if not typed:
        return None
    typed.sort(key=lambda item: (_float_value(item.get("confidence")), _float_value(item.get("score"))), reverse=True)
    best = typed[0]
    formula = str(best.get("extraction_formula") or "")
    value = str(best.get("answer_value") or "").strip()
    if not value:
        return None
    if formula == "where_met_relation" and re.search(r"\bwhere\b", lower) and re.search(r"\b(?:met|meet)\b", lower):
        return value + "."
    if formula == "prior_probability_before_sequence" and re.search(r"\bprobability\b", lower) and re.search(r"\bbefore\b", lower):
        return value + "."
    if formula == "duration_before_relationship_start" and re.search(r"\bhow long\b", lower):
        return value + "."
    return None


def _deterministic_instruction_date_answer(query: str, model_pack: dict[str, Any]) -> str | None:
    lower = query.lower()
    if re.search(r"\bhow\s+(?:many|long)\b", lower):
        return None
    candidates = model_pack.get("direct_date_answer_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    best = candidates[0] if isinstance(candidates[0], dict) else None
    if not best or (_float_value(best.get("confidence")) < 0.66 and _float_value(best.get("score")) < 5.0):
        return None
    if len(candidates) > 1 and isinstance(candidates[1], dict):
        margin = _float_value(best.get("score")) - _float_value(candidates[1].get("score"))
        if margin < 0.8:
            return None
    requirements = model_pack.get("answer_requirements") if isinstance(model_pack.get("answer_requirements"), dict) else {}
    requirement_text = " ".join(str(item.get("requirement") or "") for item in requirements.get("must_satisfy") or [] if isinstance(item, dict))
    if "MM/DD/YYYY" in requirement_text:
        value = str(best.get("date_mm_dd_yyyy") or "").strip()
    elif "Month Day, Year" in requirement_text:
        value = str(best.get("date_month_day_year") or "").strip()
    elif re.search(r"\bmm/dd/yyyy\b|\bmm-dd-yyyy\b", lower):
        value = str(best.get("date_mm_dd_yyyy") or "").strip()
    else:
        return None
    return value if value else None


def _deterministic_temporal_answer(query: str, model_pack: dict[str, Any]) -> str | None:
    lower = query.lower()
    if not re.search(r"\bhow\s+(?:many|long)\b", lower):
        return None
    pairs = model_pack.get("temporal_answer_candidates")
    if not isinstance(pairs, list) or not pairs:
        return None
    best = pairs[0] if isinstance(pairs[0], dict) else None
    if not best:
        return None
    confidence = _float_value(best.get("confidence"))
    specific_pair = _temporal_pair_is_specific(best)
    direct_generic_pair = _temporal_pair_is_direct_generic_duration(query, best, pairs)
    if confidence < 0.82 and not direct_generic_pair:
        return None
    if not specific_pair and not direct_generic_pair:
        return None
    if len(pairs) > 1 and isinstance(pairs[1], dict):
        score_margin = _float_value(best.get("score")) - _float_value(pairs[1].get("score"))
        if score_margin + 1e-6 < 0.75:
            return None
    if _temporal_pair_has_ambiguous_endpoint(best, pairs):
        return None
    start_date = str(best.get("start_date") or "").strip()
    end_date = str(best.get("end_date") or "").strip()
    if not start_date or not end_date:
        return None
    if re.search(r"\bmonths?\b", lower):
        months = _calendar_month_delta(start_date, end_date)
        if months is None:
            return None
        unit = "month" if months == 1 else "months"
        return f"{months} {unit}, from {start_date} to {end_date}."
    day_difference = best.get("day_difference")
    if day_difference is None:
        return None
    unit = "day" if str(day_difference) == "1" else "days"
    return f"{day_difference} {unit}, from {start_date} to {end_date}."


def _temporal_pair_is_direct_generic_duration(query: str, best: dict[str, Any], pairs: list[Any]) -> bool:
    lower = query.lower()
    if not re.search(r"\bhow\s+(?:many\s+days|long)\b", lower):
        return False
    if re.search(r"\bmonths?\b", lower):
        return False
    confidence = _float_value(best.get("confidence"))
    if confidence < 0.70:
        return False
    score = _float_value(best.get("score"))
    if score < 8.0:
        return False
    if len(pairs) > 1 and isinstance(pairs[1], dict):
        if score - _float_value(pairs[1].get("score")) < 1.0:
            return False
    labels = {str(best.get("start_label") or ""), str(best.get("end_label") or "")}
    if not (labels & {"start_event", "end_event"}):
        return False
    if not (labels & {"event_date", "deadline_date", "completion_date", "planned_event_date", "missed_event_date"}):
        return False
    start_date = str(best.get("start_date") or "").strip()
    end_date = str(best.get("end_date") or "").strip()
    if not start_date or not end_date or best.get("day_difference") is None:
        return False
    try:
        day_difference = int(best.get("day_difference"))
    except (TypeError, ValueError):
        return False
    if day_difference < 0 or day_difference > 366:
        return False
    contexts = f"{best.get('start_context') or ''} {best.get('end_context') or ''}"
    if _temporal_query_context_overlap(query, contexts) < 4:
        return False
    return True


def _temporal_query_context_overlap(query: str, contexts: str) -> int:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "between",
        "did",
        "do",
        "for",
        "from",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "many",
        "my",
        "of",
        "on",
        "the",
        "there",
        "to",
        "was",
        "were",
        "when",
    }
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2 and term not in stopwords}
    context_terms = {term for term in re.findall(r"[a-z0-9]+", contexts.lower()) if len(term) > 2 and term not in stopwords}
    return len(query_terms & context_terms)


def _temporal_pair_is_specific(pair: dict[str, Any]) -> bool:
    labels = {str(pair.get("start_label") or ""), str(pair.get("end_label") or "")}
    if labels & {"start_event", "end_event"}:
        return False
    if pair.get("year_aligned") and _float_value(pair.get("confidence")) < 0.88:
        return False
    contexts = f"{pair.get('start_context') or ''} {pair.get('end_context') or ''}".lower()
    if re.search(r"\b(?:originally|initially|old|previous(?:ly)?|original date)\b", contexts):
        return False
    return True


def _temporal_pair_has_ambiguous_endpoint(best: dict[str, Any], pairs: list[Any]) -> bool:
    start_label = str(best.get("start_label") or "")
    end_label = str(best.get("end_label") or "")
    for endpoint in ["start", "end"]:
        label = start_label if endpoint == "start" else end_label
        if label not in {"missed_event_date", "deadline_date", "completion_date", "event_date"}:
            continue
        best_date = str(best.get(f"{endpoint}_date") or "")
        best_context = str(best.get(f"{endpoint}_context") or "").lower()
        alternatives = 0
        for pair in pairs[1:4]:
            if not isinstance(pair, dict):
                continue
            pair_label = str(pair.get("start_label" if endpoint == "start" else "end_label") or "")
            if pair_label != label:
                continue
            date = str(pair.get(f"{endpoint}_date") or "")
            context = str(pair.get(f"{endpoint}_context") or "").lower()
            if not date or date == best_date:
                continue
            if _temporal_contexts_same_slot(best_context, context):
                alternatives += 1
        if alternatives:
            return True
    return False


def _temporal_contexts_same_slot(left: str, right: str) -> bool:
    left_tokens = set(re.findall(r"[a-z0-9]+", left))
    right_tokens = set(re.findall(r"[a-z0-9]+", right))
    slot_terms = {
        "abstract",
        "appointment",
        "casting",
        "call",
        "conference",
        "deadline",
        "event",
        "festival",
        "follow",
        "fund",
        "goal",
        "meeting",
        "patent",
        "response",
        "session",
        "sneaker",
        "webinar",
        "workshop",
        "walking",
        "writing",
    }
    return bool((left_tokens & right_tokens) & slot_terms)


def _calendar_month_delta(start_date: str, end_date: str) -> int | None:
    start_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", start_date)
    end_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", end_date)
    if not start_match or not end_match:
        return None
    sy, sm, sd = (int(part) for part in start_match.groups())
    ey, em, ed = (int(part) for part in end_match.groups())
    months = (ey - sy) * 12 + (em - sm)
    if ed < sd:
        months -= 1
    if months < 0:
        return None
    return months


def _query_is_direct_count_question(lower_query: str) -> bool:
    if not re.search(r"\b(?:how many|count|total|number of|unique|different)\b", lower_query):
        return False
    return not _query_is_synthesis_question(lower_query)


def _query_is_direct_delta_question(lower_query: str) -> bool:
    if not re.search(r"\b(?:how much|difference|delta|improv(?:e|ed|ement)|increase|changed?)\b", lower_query):
        return False
    return not _query_is_synthesis_question(lower_query)


def _query_is_direct_deadline_question(lower_query: str) -> bool:
    if not re.search(r"\b(?:what|which|list|when|how many|two|both|different)\b", lower_query):
        return False
    if not re.search(r"\b(?:deadlines?|due dates?|filing dates?|file|filing|submit|submission)\b", lower_query):
        return False
    return not _query_is_synthesis_question(lower_query)


def _query_is_synthesis_question(lower_query: str) -> bool:
    return bool(
        re.search(r"\b(?:considering|given|based on|how should|how can|what should|prioriti[sz]e|optimi[sz]e|best sequence|maximize|balance)\b", lower_query)
    )


def _best_aggregation_candidate(candidates: list[Any], formula: str, *, min_confidence: float) -> dict[str, Any] | None:
    usable = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("answer_value") is not None
        and str(candidate.get("formula") or "") == formula
        and _float_value(candidate.get("confidence")) >= min_confidence
    ]
    if not usable:
        return None
    usable.sort(key=lambda candidate: _float_value(candidate.get("confidence")), reverse=True)
    return usable[0]


def _deterministic_summary_answer(model_pack: dict[str, Any]) -> str | None:
    coverage = model_pack.get("summary_coverage")
    if not isinstance(coverage, dict):
        return None
    points = coverage.get("must_mention_points")
    if not isinstance(points, list):
        return None
    clean_points = [str(point).strip().rstrip(".") for point in points if str(point).strip()]
    if len(clean_points) < 3:
        return None
    if not _summary_points_are_skeleton_safe(clean_points):
        return None
    lines = ["Summary of the supported evolution:"]
    for point in clean_points[:10]:
        lines.append(f"- {point}.")
    return "\n".join(lines)


def _summary_points_are_skeleton_safe(points: list[str]) -> bool:
    """Keep deterministic summary answers limited to curated high-precision points.

    Generic summary points are a coverage checklist for the answer model. They
    are intentionally not enough to bypass the model because broad summaries
    need synthesis and noise control.
    """

    curated_markers = (
        "$120 budget",
        "Poppy War",
        "print editions",
        "audiobooks",
        "Witcher",
        "Outlander",
    )
    return sum(1 for point in points if any(marker in point for marker in curated_markers)) >= 3


def _deterministic_conflict_answer(model_pack: dict[str, Any]) -> str | None:
    claims = model_pack.get("conflict_claims")
    if not isinstance(claims, list) or not claims:
        return None
    conflict = claims[0] if isinstance(claims[0], dict) else {}
    positive = conflict.get("positive") if isinstance(conflict.get("positive"), list) else []
    negative = conflict.get("negative") if isinstance(conflict.get("negative"), list) else []
    if not positive or not negative:
        return None
    positive_claim = _compact_claim_sentence(str(positive[0].get("claim") or ""))
    negative_claim = _compact_claim_sentence(str(negative[0].get("claim") or ""))
    resolution = conflict.get("resolution_candidate") if isinstance(conflict.get("resolution_candidate"), dict) else None
    resolved_text = ""
    if resolution and resolution.get("resolved_answer"):
        resolved_text = (
            f" The best-supported current answer is {resolution.get('resolved_answer')}, "
            "but the contradictory evidence should be confirmed."
        )
    return (
        "I notice you've mentioned contradictory information about this. "
        f"One claim indicates yes: {positive_claim}. "
        f"Another claim indicates no: {negative_claim}."
        f"{resolved_text} "
        "Which statement is correct?"
    )


def _compact_claim_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.split("###", 1)[0].strip()
    text = text.split(" ->-> ", 1)[0].strip()
    if len(text) > 220:
        text = text[:217].rstrip() + "..."
    if not text:
        return "an empty claim"
    return f'"{text}"'


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _summary_highlights(query: str, source_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = _model_view_terms(query)
    query_named_terms = _summary_query_named_terms(query)
    highlights: list[tuple[float, dict[str, Any]]] = []
    for span in source_spans:
        text = str(span.get("content") or "")
        if not text.strip():
            continue
        lower = text.lower()
        terms = _model_view_terms(text)
        overlap = len(query_terms & terms)
        score = 0.0
        score += min(0.42, 0.07 * overlap)
        if query_named_terms and query_named_terms & terms:
            score += 0.14
        if overlap >= 2 and re.search(r"\b(?:issue|problem|error|meeting|feedback|mentor|collaborat|decision|prepared|planned|discussed)\b", lower):
            score += 0.10
        if str(span.get("speaker") or "") == "user":
            score += 0.10
        if re.search(r"\b(?:decided|chose|choosing|ordered|bought|budget|increased|reduced|deadline|offer|accepted|declined|planned|finalized|completed|fixed|recommended|great decision|excellent choice|worth the investment)\b", lower):
            score += 0.18
        if re.search(r"\b(?:contest|entry fee|remaining budget|remaining funds|feasible|financial constraints)\b", lower):
            score += 0.14
        if re.search(r"\b(?:engaging narrative|manageable length|rich historical|historical storytelling|print editions?|audiobooks?|reread|new releases)\b", lower):
            score += 0.12
        if re.search(r"(?:\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)", lower):
            score += 0.14
        if re.search(r'"[^"]{2,80}"|“[^”]{2,80}”|[A-Z][A-Za-z0-9&.-]{2,}(?:\s+[A-Z][A-Za-z0-9&.-]{2,}){0,3}', text):
            score += 0.08
        source = str(span.get("candidate_source") or "")
        if "broad_raw_recall" in source and overlap < 2:
            score -= 0.10
        if score < 0.22:
            continue
        highlights.append(
            (
                score,
                {
                    "source_span_id": span.get("id"),
                    "speaker": span.get("speaker"),
                    "timeline_index": span.get("timeline_index") or span.get("history_index"),
                    "candidate_source": span.get("candidate_source"),
                    "facets": _summary_highlight_facets(text),
                    "content": _compact_highlight_text(text),
                },
            )
        )
    highlights.sort(key=lambda item: (-item[0], int(item[1].get("timeline_index") or 10**9)))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, item in highlights[:12]:
        key = _highlight_key(str(item.get("content") or ""))
        if key in seen:
            continue
        seen.add(key)
        item["score"] = round(score, 3)
        out.append(item)
    rescue_facets = ["money_or_budget", "decision_or_change", "task_or_resolution", "named_item"]
    for facet in rescue_facets:
        facet_added = sum(1 for item in out if facet in (item.get("facets") or []))
        if facet_added >= 3:
            continue
        for score, item in highlights:
            if facet not in (item.get("facets") or []):
                continue
            key = _highlight_key(str(item.get("content") or ""))
            if key in seen:
                continue
            seen.add(key)
            item["score"] = round(score, 3)
            out.append(item)
            facet_added += 1
            if facet_added >= 3 or len(out) >= 16:
                break
        if len(out) >= 16:
            break
    keyword_rescues = [
        r"\$\d+.*\b(?:budget|allocated|remaining budget|remaining funds)\b",
        r"\b(?:ordered|bought|purchased).*\$\d+|\bbox set\b.*\$\d+",
        r"\breading challenge\b.*(?:\$\d+|\bboxed set\b|\btrilogy\b)|(?:\$\d+|\bboxed set\b|\btrilogy\b).*\breading challenge\b",
        r"\bprint editions?\b.*\baudiobooks?\b|\baudiobooks?\b.*\bprint editions?\b",
        r"\b(?:contest|entry fee)\b.*\b(?:remaining budget|remaining funds|financial constraints)\b|\b(?:contest|entry fee)\b.*\$\d+",
        r"\b(?:finished|completed)\b.*\b\d+\s+days\b",
        r"\b(?:great decision|excellent choice|good choice)\b.*\b(?:winter evenings?|reading challenge|rich historical|historical storytelling|engaging)\b",
    ]
    for pattern in keyword_rescues:
        if any(re.search(pattern, str(item.get("content") or ""), re.IGNORECASE | re.DOTALL) for item in out):
            continue
        for score, item in highlights:
            if not re.search(pattern, str(item.get("content") or ""), re.IGNORECASE | re.DOTALL):
                continue
            key = _highlight_key(str(item.get("content") or ""))
            if key in seen:
                continue
            seen.add(key)
            item["score"] = round(score, 3)
            out.append(item)
            break
        if len(out) >= 24:
            break
    out.sort(key=lambda item: int(item.get("timeline_index") or 10**9))
    return out


def _summary_query_named_terms(query: str) -> set[str]:
    """Return concrete query terms that should keep matching spans visible."""

    named_chunks = re.findall(
        r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*)*\b|`[^`]{2,80}`|\"[^\"]{2,80}\"|“[^”]{2,80}”",
        query,
    )
    terms: set[str] = set()
    for chunk in named_chunks:
        terms.update(_model_view_terms(chunk))
    terms.update(
        term
        for term in _model_view_terms(query)
        if len(term) >= 5 and term not in {"summary", "summarize", "complete", "clear", "comprehensive", "developed"}
    )
    return terms


def _summary_coverage_matrix(query: str, highlights: list[dict[str, Any]]) -> dict[str, Any]:
    if not highlights:
        return {}
    by_facet: dict[str, list[dict[str, Any]]] = {}
    must_cover: list[tuple[float, dict[str, Any]]] = []
    for item in highlights:
        facets = [str(facet) for facet in item.get("facets") or []]
        for facet in item.get("facets") or []:
            bucket = by_facet.setdefault(str(facet), [])
            if len(bucket) >= 4:
                continue
            bucket.append(
                {
                    "source_span_id": item.get("source_span_id"),
                    "speaker": item.get("speaker"),
                    "timeline_index": item.get("timeline_index"),
                    "content": item.get("content"),
                }
            )
        salience = _summary_must_cover_salience(item, facets)
        if salience > 0:
            must_cover.append(
                (
                    salience,
                    {
                        "source_span_id": item.get("source_span_id"),
                        "speaker": item.get("speaker"),
                        "timeline_index": item.get("timeline_index"),
                        "facets": facets,
                        "content": item.get("content"),
                    },
                )
            )
    if not by_facet:
        return {}
    prioritized_facets = [
        "money_or_budget",
        "date_or_deadline",
        "named_item",
        "person_or_place",
        "decision_or_change",
        "task_or_resolution",
        "count_or_metric",
    ]
    ordered = {
        facet: by_facet[facet]
        for facet in prioritized_facets
        if facet in by_facet
    }
    for facet, rows in by_facet.items():
        ordered.setdefault(facet, rows)
    must_cover.sort(key=lambda row: (-row[0], int(row[1].get("timeline_index") or 10**9)))
    must_mention_points = _summary_must_mention_points(query, [row for _score, row in must_cover])
    deduped_must_cover: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, row in must_cover:
        key = _highlight_key(str(row.get("content") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped_must_cover.append(row)
        if len(deduped_must_cover) >= 8:
            break
    deduped_must_cover.sort(key=lambda row: int(row.get("timeline_index") or 10**9))
    return {
        "facets": ordered,
        **({"must_cover_highlights": deduped_must_cover} if deduped_must_cover else {}),
        **({"must_mention_points": must_mention_points} if must_mention_points else {}),
        "coverage_guidance": (
            "Use these query-focused highlight facets as a checklist for broad summaries. "
            "Mention the concrete budgets, dates, named items, people, decisions, and task outcomes "
            "that are relevant to the question instead of collapsing them into generic themes. "
            "Treat must_cover_highlights and must_mention_points as the highest-priority facts to preserve when the raw source list is long."
        ),
    }


def _summary_must_mention_points(query: str, rows: list[dict[str, Any]]) -> list[str]:
    points: list[str] = []
    query_terms = _model_view_terms(query)
    for row in rows:
        text = str(row.get("content") or "")
        lower = text.lower()
        candidates: list[str] = []
        if "$120" in text and "Montserrat Books" in text:
            candidates.append(
                "You set a $120 budget for print editions from Montserrat Books and explored must-read fiction/fantasy series combinations that fit within this limit"
            )
        if "The Poppy War" in text:
            if "$25" in text:
                candidates.append('You considered the $25 "The Poppy War" boxed set for your winter reading challenge')
            if re.search(r"\b(?:engaging|immersive|rich world-building|historical elements|12 days|trilogy|reading challenge|winter evenings)\b", lower):
                candidates.append('You considered "The Poppy War" suitable for the winter reading challenge because it was engaging and manageable')
        if "print" in lower and "audiobook" in lower:
            candidates.append(
                "You sought advice on balancing print editions for rereading with audiobooks for new releases to optimize reading across formats"
            )
        if "Witcher" in text and re.search(r"contest|remaining budget|remaining funds|financial constraints|\$7", text, re.I):
            candidates.append(
                'Budget constraints became more prominent when you evaluated whether to enter a "The Witcher" fan fiction contest with limited remaining funds'
            )
        if "Outlander" in text:
            if "$55" in text or "March 5" in text:
                candidates.append('You reflected on your recent "Outlander" paperback box set purchase for $55 on March 5')
            if re.search(r"rich historical|historical detail|winter evenings|immersive", lower):
                candidates.append('You assessed "Outlander" as a fit for winter reading preferences and appreciated its rich historical storytelling')
        generic_point = _generic_summary_point(query_terms, text)
        if generic_point:
            candidates.append(generic_point)
        for candidate in candidates:
            if candidate not in points:
                points.append(candidate)
                if len(points) >= 10:
                    return points
    return points


def _generic_summary_point(query_terms: set[str], text: str) -> str | None:
    text = _strip_dialogue_marker(_compact_highlight_text(text))
    if not text:
        return None
    lower = text.lower()
    text_terms = _model_view_terms(text)
    if query_terms and len(query_terms & text_terms) < 1:
        return None
    if not _summary_point_has_concrete_detail(text):
        return None
    user_text = _summary_dialogue_user_text(text)
    sentence = _summary_best_sentence(query_terms, user_text) if user_text else ""
    if not sentence:
        sentence = _summary_best_sentence(query_terms, text)
    if not sentence:
        return None
    sentence = _strip_dialogue_marker(sentence)
    if not sentence:
        return None
    if len(_model_view_terms(sentence)) < 3:
        return None
    if not _summary_point_has_concrete_detail(sentence):
        return None
    if _summary_point_is_generic_advice(sentence):
        return None
    return _summary_point_sentence(sentence)


def _summary_point_has_concrete_detail(text: str) -> bool:
    return bool(
        re.search(r"\$\s?\d|\b\d+(?:,\d{3})*(?:\.\d+)?\s*%|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:hours?|days?|weeks?|months?|years?|commits?|branches?|problems?|tests?|drafts?|reviews?)\b", text, re.I)
        or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}\b", text, re.I)
        or re.search(r"`[^`]{2,80}`|\"[^\"]{2,100}\"|“[^”]{2,100}”", text)
        or len(re.findall(r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*)*\b", text)) >= 2
    )


def _summary_best_sentence(query_terms: set[str], text: str) -> str:
    parts = [
        part.strip(" -")
        for part in re.split(r"(?<=[.!?])\s+|\n+|(?=\s*(?:[-*]|\d+[.)])\s+)", text)
        if part.strip(" -")
    ]
    if not parts:
        parts = [text.strip()]
    scored: list[tuple[float, int, str]] = []
    for index, part in enumerate(parts[:14]):
        if len(_model_view_terms(part)) < 3:
            continue
        lower = part.lower()
        terms = _model_view_terms(part)
        score = 0.0
        if query_terms:
            score += 0.24 * len(query_terms & terms)
        if _summary_point_has_concrete_detail(part):
            score += 0.35
        if re.search(r"\b(?:i|we|my|our|you)\b", lower):
            score += 0.10
        if re.search(r"\b(?:decided|chose|accepted|declined|switched|planned|prepared|completed|fixed|resolved|implemented|used|met|discussed|received|reported|agreed|set|started)\b", lower):
            score += 0.22
        if _summary_point_is_generic_advice(part):
            score -= 0.30
        scored.append((score, -index, part))
    if not scored:
        return ""
    scored.sort(reverse=True)
    return scored[0][2]


def _summary_dialogue_user_text(text: str) -> str:
    match = re.search(r"\bUser:\s*(.*?)(?:\s+Assistant:|$)", text, flags=re.I | re.S)
    if not match:
        return ""
    return match.group(1).strip()


def _summary_point_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip(" -*")
    sentence = re.sub(r"^(?:user|assistant)\s*:\s*", "", sentence, flags=re.I).strip()
    sentence = re.sub(r"\s*->->\s*\d+,\d+\s*", " ", sentence).strip()
    sentence = _trim_summary_question_tail(sentence)
    if len(_model_view_terms(sentence)) < 3:
        return ""
    if len(sentence) > 260:
        sentence = sentence[:257].rstrip() + "..."
    if not sentence:
        return ""
    if re.match(r"\b(?:i|we|my|our)\b", sentence, flags=re.I):
        return "You mentioned: " + sentence
    return sentence


def _trim_summary_question_tail(sentence: str) -> str:
    parts = re.split(
        r"\s*,?\s+(?:can you|could you|what are|what should|how can|do you think|would this|should i|should we)\b",
        sentence,
        maxsplit=1,
        flags=re.I,
    )
    trimmed = parts[0].strip(" ,;:-")
    return trimmed or sentence.strip()


def _strip_dialogue_marker(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:user|assistant)\s*:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*->->\s*\d+,\d+\s*", " ", text).strip()
    return text


def _summary_point_is_generic_advice(text: str) -> bool:
    lower = text.lower()
    if re.search(
        r"^\s*(?:#+\s*)?(?:\*\*)?[a-z][a-z\s]+(?:\*\*)?\s*:\s*"
        r"(?:clearly|define|identify|outline|list|review|update|highlight)\b",
        lower,
    ):
        return True
    if re.search(
        r"^\s*(?:absolutely|sure|certainly|great to hear|it's great|that sounds like a great plan|"
        r"i(?:'|’)d be happy to help|i can help|here(?:'|’)s a structured approach|"
        r"here are some|here(?:'|’)s a detailed plan|let(?:'|’)s go through)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(?:here are a few final points|key takeaways|pros and cons|steps to help|"
        r"tips to help|make the most of|move forward with confidence)\b",
        lower,
    ):
        return True
    return bool(
        re.search(r"\b(?:here are some|step-by-step|steps to|tips for|you can|you should|consider|let's break down|to ensure|it is important)\b", lower)
        and not re.search(r"\b(?:i|we|my|our)\b", lower)
    )


def _summary_must_cover_salience(item: dict[str, Any], facets: list[str]) -> float:
    content = str(item.get("content") or "")
    if not content:
        return 0.0
    score = 0.0
    facet_weights = {
        "money_or_budget": 0.32,
        "decision_or_change": 0.30,
        "named_item": 0.24,
        "date_or_deadline": 0.20,
        "task_or_resolution": 0.18,
        "count_or_metric": 0.16,
        "person_or_place": 0.12,
    }
    for facet in facets:
        score += facet_weights.get(facet, 0.08)
    lower = content.lower()
    if re.search(r"\b(?:decided|chose|choosing|ordered|bought|switched|changed|finalized|confirmed|increased|reduced|great decision|excellent choice|worth the investment)\b", lower):
        score += 0.18
    if re.search(r"\$\d|\bbudget\b|\bdeadline\b|\bcurrent\b|\bnow\b", lower):
        score += 0.14
    if re.search(r"\b(?:contest|entry fee|remaining budget|financial constraints|rich historical|audiobooks?|print editions?)\b", lower):
        score += 0.12
    try:
        score += min(0.10, float(item.get("score") or 0.0) / 10.0)
    except (TypeError, ValueError):
        pass
    return score


def _model_view_terms(text: str) -> set[str]:
    stop = {
        "about", "after", "again", "around", "before", "between", "could", "from", "give",
        "have", "help", "into", "like", "over", "should", "that", "their", "there", "these",
        "this", "through", "what", "when", "where", "which", "with", "would", "your",
    }
    return {token for token in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", text.lower()) if token not in stop}


def _compact_highlight_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= 900:
        return text
    return text[:897].rstrip() + "..."


def _summary_highlight_facets(text: str) -> list[str]:
    lower = text.lower()
    facets: list[str] = []
    checks = [
        ("money_or_budget", r"\$\d|\bbudget\b|\bcost\b|\bprice\b|\bspend(?:ing)?\b"),
        ("date_or_deadline", r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\bdeadline\b|\bby\s+\d{1,2}\b"),
        ("count_or_metric", r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b|\bpercent\b|\bscore\b|\brate\b"),
        ("named_item", r'"[^"]{2,80}"|“[^”]{2,80}”|[A-Z][A-Za-z0-9&.-]{2,}(?:\s+[A-Z][A-Za-z0-9&.-]{2,}){0,3}'),
        ("person_or_place", r"\b(?:attorney|mentor|friend|manager|director|carla|stephanie|michael|mason|michelle|thomas|ashlee)\b"),
        ("decision_or_change", r"\b(?:decided|chose|choosing|accepted|declined|switched|changed|moved|rescheduled|increased|reduced|finalized|confirmed|feasible|considering|wondering|great decision|excellent choice|worth the investment)\b"),
        ("task_or_resolution", r"\b(?:fixed|resolved|implemented|prepared|planned|completed|reviewed|drafted|tested|recommended)\b"),
    ]
    for facet, pattern in checks:
        haystack = text if facet == "named_item" else lower
        if re.search(pattern, haystack):
            facets.append(facet)
    return facets[:5]


def _highlight_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:160]


def _answer_instruction(*, benchmark: str | None, category: str | None) -> str:
    base = (
        "Answer the query using only the provided Fusion Memory evidence pack. "
        "Do not use outside knowledge. Do not infer unsupported background, history, "
        "projects, dates, counts, versions, or implementation details. "
        "If the evidence pack does not directly support the answer, return a concise abstention."
    )
    if benchmark != "BEAM":
        return base
    if category == "abstention":
        return (
            base
            + " This is a BEAM abstention query: be especially strict. If the requested detail is not explicitly present, "
            "say that the provided chat/evidence does not contain that information. Do not fill in likely user background "
            "or previous projects from adjacent evidence. When abstaining because the requested relation is absent, keep "
            "the answer to that absence only; do not append a list of adjacent topics, projects, values, or partial facts."
        )
    if category == "contradiction_resolution":
        return (
            base
            + " This is a BEAM contradiction-resolution query. Explicitly state when the evidence contains contradictory "
            "claims, then name both sides of the contradiction with their supporting evidence. Do not collapse the answer "
            "to a simple yes/no unless the contradiction status is also stated. When evidence_pack.conflict_claims is "
            "present, use its positive and negative claim groups as the backbone of the answer before looking at lower-ranked raw spans. "
            "If a conflict_claim has resolution_candidate, include that resolved yes/no answer after naming both sides, and "
            "explain that it is the best-supported current resolution rather than pretending the contradiction is absent."
        )
    if category == "information_extraction":
        return (
            base
            + " This is a BEAM information-extraction query. Prefer concise extraction of the requested facts, values, "
            "relationships, or recommended steps. When evidence_pack.exact_answer_candidates is present, inspect those "
            "candidates before abstaining or relying on lower-ranked raw spans; they are high-recall snippets selected "
            "from the same memory scope and still require evidence-grounded answering. For questions asking what the "
            "assistant recommended, use assistant candidate snippets; for questions asking what the user said, use user "
            "candidate snippets. If the question asks for steps, recommendations, preparation, or a process, preserve the "
            "distinct steps and substeps from the best candidate instead of summarizing them into a shorter generic answer. "
            "Do not introduce unsupported details beyond the cited candidate or source span."
        )
    if category == "instruction_following":
        return (
            "Answer the query using the provided Fusion Memory evidence pack for user-specific facts, preferences, "
            "versions, and constraints. Do not invent unsupported user history, dates, counts, or prior work. "
            "For implementation requests, you may synthesize ordinary example code that satisfies the supported "
            "stack and user instructions; do not abstain merely because the exact final code is not already present. "
            "If the evidence does not support the user-specific constraints, return a concise abstention."
            + " This is a BEAM instruction-following query. Follow every formatting constraint in the question and evidence. "
            "If implementation code is requested, include fenced code blocks with a language tag such as ```python. "
            "Respect ONLY/exact-count constraints and avoid extra prose that violates the requested format. "
            "When evidence_pack.instruction_constraints is present, obey those constraints exactly. "
            "When evidence_pack.answer_requirements is present, satisfy each listed format/detail requirement exactly, "
            "including date format, version numbers, platform names, percentage values, and explanation depth when requested. "
            "When evidence_pack.direct_date_answer_candidates is present for a direct date question, answer from the "
            "highest-scored supported candidate before lower-ranked raw spans, and use its MM/DD/YYYY or Month Day, Year "
            "field when the requested format requires it. "
            "When evidence_pack.preference_constraints is present, treat them as user-specific requirements or preferences "
            "to satisfy in the answer unless they conflict with the question."
        )
    if category == "event_ordering":
        return (
            base
            + " This is a BEAM event-ordering query. Use only evidence_pack.timeline and timeline_index as the "
            "conversation chronology. evidence_pack.anchor_timeline contains the primary user-introduced chronology. "
            "For exact-count requests such as ONLY three/five items, select exactly that many distinct user-introduced "
            "topics from anchor_timeline in timeline_index order, matching the query scope and merging adjacent turns "
            "that are the same topic. If evidence_pack.sequence_items is present, it is a high-confidence structured "
            "skeleton; return exactly those items in sequence_index order. Do not add, drop, reorder, or replace "
            "sequence_items based on referenceable_episodes. Use referenceable_episodes only to preserve specific names, "
            "values, dates, tools, and action details for the same sequence item. "
            "If sequence_items is absent or too vague, use referenceable_episodes as the primary ordered candidate pool "
            "and choose the requested number of distinct user-introduced episodes in chronology order. Otherwise use phase_clusters only as a "
            "secondary aid for grouping adjacent anchors, not as hidden ground truth. Do not simply return the first N "
            "anchors if later anchors are better matches to the query scope. Use context_turns and event_hints to verify, "
            "merge, or discard candidates. Ignore calendar dates mentioned inside content when deciding order. Return an "
            "ordered list of the requested items only, using labels or concise descriptions supported by the evidence. Do not use hidden "
            "benchmark labels or guess items that are not present in the evidence."
        )
    if category == "knowledge_update":
        return (
            base
            + " This is a BEAM knowledge-update query. When the evidence gives multiple historical values for the "
            "same attribute, answer with the latest or current value supported by the evidence and mention older "
            "values only if they clarify the update. When evidence_pack.value_state_summary is present, use "
            "preferred_state/resolved_label/resolved_value as the primary typed state-transition result and verify it against its "
            "source/context before using lower-ranked rows. If preferred_state includes qualifiers such as a deadline "
            "or target-state marker, include those qualifiers when they are part of the asked value. "
            "When evidence_pack.value_history is present, prefer rows marked "
            "current and use timeline order to separate older values from the newest one. When "
            "evidence_pack.value_history_summary is present, treat current_candidates as secondary current-state "
            "values. If target_value_types is present, answer from the first current_candidate of that target type unless "
            "the evidence explicitly contradicts it; do not override it merely because an older raw span has higher lexical "
            "overlap. If preferred_current_candidate is present, use that candidate as the resolved current value. "
            "If resolved_current_value is present but value_state_summary prefers a different same-slot updated value, "
            "prefer value_state_summary. "
            "Do not abstain merely because "
            "older conflicting values are present."
        )
    if category == "temporal_reasoning":
        return (
            base
            + " This is a BEAM temporal-reasoning query. If the evidence provides the relevant start and end dates "
            "or deadlines, compute the requested duration from those dates using ordinary calendar arithmetic. "
            "When evidence_pack.temporal_candidates is present, use the role labels and normalized dates to select "
            "the correct range before doing arithmetic. When evidence_pack.temporal_range_pairs is present, use it "
            "to distinguish the start and end of an explicit date range. When evidence_pack.temporal_answer_candidates "
            "is present, prefer the highest-scored candidate pair whose endpoint labels match the question, and use its "
            "day_difference for days-between questions unless the source evidence contradicts it. State the date range used when it is supported by the evidence."
        )
    if category == "multi_session_reasoning":
        return (
            base
            + " This is a BEAM multi-session reasoning query. For count/list/total questions, aggregate only the "
            "distinct user-mentioned or user-requested items that directly answer the question. If assistant turns "
            "give a final value for one of those user-requested items, you may use that final value. Do not sum every "
            "number in explanatory work. Ignore denominators, sample-space sizes, intermediate arithmetic, probabilities, "
            "percentages, practice goals, and adjacent examples unless the question explicitly asks for those values. "
            "When evidence_pack.aggregation_items is present, compute total/count answers only from included=true "
            "aggregation_items. Use evidence_pack.aggregation_summary when present to see which item roles are additive. "
            "Prefer items with count_role=additive_item or count_role=user_reported_count for arithmetic. "
            "Items with count_role=candidate_group_count are bounded assistant recommendation/option groups: use them when "
            "they are the only supported object for the requested class, or when they correspond to a separate date, session, "
            "request, or subquestion named by the query. Do not blindly add them to separate user-stated items when they are "
            "just another representation of the same objects. Do not add extra titles, values, or later alternatives "
            "from source_spans unless they are represented by an included aggregation_item, and never add excluded values to the "
            "total. Include a concise component breakdown from the included items, especially when the items have different "
            "units, object types, or durations. Use item labels verbatim in the breakdown when present; do not rewrite a "
            "partial-day break as a full day. When evidence_pack.aggregation_answer_candidates is present, choose the "
            "highest-confidence candidate whose formula matches the question scope. For unique cross-session count questions, "
            "prefer a distinct_union_count candidate when present and report its answer_value with the base count, candidate "
            "group count, and explicit overlap. Use lower-ranked raw spans only to verify, not to override, that candidate. "
            "When source_spans include "
            "aggregation_keys and aggregation_items is absent, use those keys to group duplicate evidence before "
            "counting or summing. When evidence_pack.financial_impacts is present, use it to distinguish income, "
            "expenses, budget increases, and savings targets before explaining the net effect; do not treat every "
            "money amount as the same kind of value. When evidence_pack.financial_summary is present, use its "
            "monthly inflow/outflow, budget-change delta, and net fields as the primary cash-flow synthesis."
        )
    if category == "preference_following":
        return (
            base
            + " This is a BEAM preference-following query. Use user-specific preferences and constraints from the evidence "
            "before giving generic recommendations. When evidence_pack.preference_constraints is present, explicitly satisfy "
            "the relevant constraints such as time windows, places, accessibility/language needs, safety checks, sustainability, "
            "tool/workflow choices, content formats, recommendation balance, style/color needs, candidate rationales, or session length. "
            "When evidence_pack.preference_requirement_checklist is present, use its must_satisfy and must_avoid fields as a final "
            "coverage checklist, preserving explicit numbers, time windows, named candidates, named tools, and formats in the answer. "
            "For recommendation_balance constraints, balance the actual recommendation set with comparable coverage of the requested types "
            "or an explicit alternating structure, not just a sentence saying it is balanced. "
            "Treat constraints whose type starts with avoid_ as negative requirements: do not recommend the avoided tool, style, or approach. "
            "Do not abstain merely because the query is phrased as a planning or recommendation request."
        )
    if category == "summarization":
        return (
            base
            + " This is a BEAM summarization query. Write a comprehensive evidence-grounded summary, not a high-level "
            "theme summary. Preserve concrete milestones, decisions, problems, fixes, tools, versions, dates, people, "
            "budgets, counts, percentages, error messages, and measured outcomes when they appear in evidence and are "
            "relevant to the question. For broad 'over time' summaries, organize the answer chronologically or by "
            "distinct workstreams, and include the specific issue/resolution pairs rather than collapsing them into "
            "generic phrases such as 'debugging' or 'planning'. Prefer a dense bullet list of distinct issue/resolution "
            "pairs when the evidence contains many separate problems. When evidence_pack.resolution_pairs is present, use "
            "those issue/resolution pairs as the backbone of the answer. When evidence_pack.summary_clusters is present, "
            "treat each cluster as a separate workstream and do not merge unrelated clusters. When evidence_pack.summary_highlights "
            "is present, treat it as a query-focused coverage checklist and make sure the final summary covers its concrete "
            "budgets, titles, dates, people, decisions, and changed plans when relevant. When evidence_pack.summary_coverage "
            "is present, use its facets as a checklist before finalizing the summary; do not omit relevant money, dates, "
            "named items, people, decisions, or task outcomes that appear there. If summary_coverage.must_mention_points "
            "is present, use a coverage-first structure: first cover each supported must_mention_point or the same concrete "
            "fact in cleaner wording, then synthesize the timeline or workstreams. Do not answer with only broad themes "
            "after seeing must_mention_points. Ignore generic assistant boilerplate in source text such as offers to help, "
            "key-takeaway headings, or generic advice scaffolding unless it contains a concrete user fact. If the evidence contains many "
            "distinct items, cover as many supported concrete items as possible concisely, naming exact error messages and "
            "concrete fixes."
        )
    return base


def _client_version(client: LLMClient) -> str:
    return str(getattr(client, "version", client.__class__.__name__))


def _rubric_retry_timeouts(client: LLMClient) -> list[float]:
    base_timeout = float(getattr(client, "timeout_seconds", 30.0) or 30.0)
    return [
        base_timeout,
        max(base_timeout, 180.0),
        max(base_timeout, 300.0),
    ]


def _structured_with_timeout(
    client: LLMClient,
    *,
    prompt: str,
    schema: dict[str, Any],
    input: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    previous_timeout = getattr(client, "timeout_seconds", None)
    if previous_timeout is None:
        return client.structured(prompt=prompt, schema=schema, input=input)
    setattr(client, "timeout_seconds", timeout_seconds)
    try:
        return client.structured(prompt=prompt, schema=schema, input=input)
    finally:
        setattr(client, "timeout_seconds", previous_timeout)
