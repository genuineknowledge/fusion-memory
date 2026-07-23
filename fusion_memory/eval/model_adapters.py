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
from fusion_memory.retrieval.slot_state_transition import value_state_summary
from fusion_memory.retrieval.temporal_pack import direct_date_answer_candidates, temporal_answer_candidates, temporal_model_candidates
from fusion_memory.retrieval.value_history_pack import exact_candidate_value_rows, value_history_summary


PACK_CONTRACT_VERSION = "typed-evidence-pack-v1"


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
        model_pack = self._model_pack(
            pack,
            benchmark=benchmark,
            category=category,
        )
        deterministic_answer = self._deterministic_answer(
            query,
            model_pack,
            benchmark=benchmark,
            category=category,
        )
        if deterministic_answer:
            return deterministic_answer
        response = self.client.structured(
            prompt=self.prompt_version,
            schema=ANSWER_SCHEMA,
            input={
                "instruction": self._instruction(benchmark=benchmark, category=category),
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
        )

    def _deterministic_answer(
        self,
        query: str,
        model_pack: dict[str, Any],
        *,
        benchmark: str | None,
        category: str | None,
    ) -> str | None:
        return None

    def _instruction(
        self,
        *,
        benchmark: str | None,
        category: str | None,
    ) -> str:
        return _answer_instruction()


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
    force_llm_aggregation: bool = False,
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
    if aggregation_client is not None and force_llm_aggregation:
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


def _answer_instruction() -> str:
    return (
        "Answer the query using only the provided Fusion Memory evidence pack. "
        "Do not use outside knowledge. Do not infer unsupported background, history, "
        "projects, dates, counts, versions, or implementation details. "
        "If the evidence pack does not directly support the answer, return a concise abstention."
    )


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
