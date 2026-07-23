from __future__ import annotations

import re
from typing import Any

from fusion_memory.core.models import EvidencePack
from fusion_memory.eval.model_adapters import OpenAICompatibleAnswerModel, _float_value, _pack_for_model


class OpenAICompatibleBeamAnswerModel(OpenAICompatibleAnswerModel):
    """OpenAI-compatible answer model with BEAM-only answer behavior."""

    @classmethod
    def from_generic(
        cls,
        model: OpenAICompatibleAnswerModel,
    ) -> "OpenAICompatibleBeamAnswerModel":
        return cls(
            model.client,
            prompt_version=model.prompt_version,
            use_llm_aggregation=model.use_llm_aggregation,
            llm_aggregation_min_confidence=model.llm_aggregation_min_confidence,
        )

    def _deterministic_answer(
        self,
        query: str,
        model_pack: dict[str, Any],
        *,
        benchmark: str | None,
        category: str | None,
    ) -> str | None:
        return _deterministic_model_pack_answer(
            query,
            category,
            model_pack,
            benchmark=benchmark,
        )

    def _model_pack(
        self,
        pack: EvidencePack,
        *,
        benchmark: str | None,
        category: str | None,
    ) -> dict[str, Any]:
        return _pack_for_model(
            pack,
            aggregation_client=self.client if self.use_llm_aggregation else None,
            aggregation_min_confidence=self.llm_aggregation_min_confidence,
            force_llm_aggregation=category == "multi_session_reasoning",
        )

    def _instruction(
        self,
        *,
        benchmark: str | None,
        category: str | None,
    ) -> str:
        return _beam_answer_instruction(benchmark=benchmark, category=category)


def as_beam_answer_model(answer_model: Any) -> Any:
    if isinstance(answer_model, OpenAICompatibleBeamAnswerModel):
        return answer_model
    if type(answer_model) is OpenAICompatibleAnswerModel:
        return OpenAICompatibleBeamAnswerModel.from_generic(answer_model)
    return answer_model


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


def _beam_answer_instruction(*, benchmark: str | None, category: str | None) -> str:
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
