from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from fusion_memory.core.text import compact_summary, tokenize
from fusion_memory.retrieval.temporal_relations import safe_temporal_relation_records, temporal_relations_for_text


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
    "was",
    "were",
    "what",
    "when",
    "which",
    "with",
    "work",
    "would",
    "you",
    "your",
}


def topic_scope_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?", text.lower())
    tokens: set[str] = set()
    for token in raw:
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


def date_signal(text: str) -> float:
    lower = text.lower()
    if re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", lower):
        return 1.0
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b", lower):
        return 0.9
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?)\b", lower):
        return 0.55
    return 0.0


def temporal_roles_in_text(query: str, text: str) -> set[str]:
    lower = text.lower()
    roles: set[str] = set()
    query_terms = topic_scope_tokens(query)
    text_terms = topic_scope_tokens(text)
    overlap = len(query_terms & text_terms)
    if ("deployment" in lower or "deploy" in lower or "launch" in lower or "production" in lower) and (
        "deadline" in lower or "by " in lower or "target" in lower or date_signal(lower)
    ):
        roles.add("deployment_deadline")
    if re.search(r"\b(?:decided|decision|chose|reject(?:ed|ing)?|declin(?:ed|e|ing))\b", lower):
        roles.add("decision_date")
    if re.search(r"\b(?:rescheduled|reschedule|moved|pushed|postponed)\b", lower):
        roles.add("reschedule_date")
    if re.search(r"\b(?:downloaded?|installed|borrowed|checked out|acquired)\b", lower):
        roles.add("download_date")
    if re.search(r"\b(?:finish(?:ed|ing)?|complete(?:d|ion)?|done|read)\b", lower) and overlap >= 1:
        roles.add("completion_date")
    if (
        re.search(r"\bfinish|finished|complete|completed|completion|end|ended\b", lower)
        and ("feature" in lower or "features" in lower or overlap >= 2)
    ):
        roles.add("feature_finish_date")
    if "sprint" in lower and re.search(r"\bend|ends|ended|first\b", lower):
        roles.add("sprint_end_date")
    if re.search(r"\bstart|starts|started|begin|begins\b", lower):
        roles.add("start_date")
    return roles


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
TEMPORAL_DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
TEMPORAL_MONTH_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
    re.I,
)
TEMPORAL_CJK_MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日(?!\d)")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
EXPLICIT_MONTH_DAY_YEAR_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))\b",
    re.I,
)
TEMPORAL_STOPWORDS = {
    "about",
    "after",
    "before",
    "between",
    "date",
    "dates",
    "deadline",
    "deadlines",
    "did",
    "feature",
    "features",
    "final",
    "complete",
    "completed",
    "completing",
    "completion",
    "finish",
    "finishing",
    "first",
    "have",
    "how",
    "many",
    "need",
    "second",
    "sprint",
    "sprints",
    "time",
    "weeks",
    "what",
    "when",
    "which",
}
DEPLOYMENT_TERMS = {
    "deploy",
    "deployed",
    "deploying",
    "deployment",
    "launch",
    "launched",
    "release",
    "released",
    "ship",
    "shipping",
    "production",
    "rollout",
}
FINAL_TERMS = {"final", "production", "launch", "release", "ship", "deployment"}
MVP_TERMS = {"mvp", "prototype", "scope", "minimum viable"}
DEADLINE_TERMS = {"deadline", "deadlines", "due", "target date", "by "}
SUBMISSION_TERMS = {"submit", "submitted", "submission", "submitting"}
COMPLETION_TERMS = {"finish", "finishing", "finished", "complete", "completed", "completing", "completion", "done", "end", "ends", "ending"}
START_TERMS = {"start", "starts", "started", "starting", "begin", "begins", "began", "from"}
FEATURE_TERMS = {
    "feature",
    "features",
    "implementation",
    "phase",
    "milestone",
    "work",
    "module",
    "component",
    "transaction",
    "management",
}


def temporal_mentions(query: str, content: str, span_timestamp: object = None) -> list[dict[str, object]]:
    if not content:
        return []
    query_lower = query.lower()
    default_year_value = default_year(span_timestamp)
    mentions: list[dict[str, object]] = []
    for match in list(TEMPORAL_DATE_RE.finditer(content)) + list(TEMPORAL_MONTH_RE.finditer(content)) + list(TEMPORAL_CJK_MONTH_RE.finditer(content)):
        text = match.group(0)
        context = mention_context(content, match.start(), match.end())
        role_text = cjk_date_role_context(content, match.start(), match.end()) if TEMPORAL_CJK_MONTH_RE.fullmatch(text) else role_context(content, match.start(), match.end())
        endpoint = range_endpoint(content, match.start(), match.end())
        role, confidence = temporal_role(query_lower, role_text.lower(), endpoint)
        normalized_date = normalize_date_text(text, infer_year_for_match(content, match.start(), match.end(), default_year_value))
        mention: dict[str, object] = {
            "text": text,
            "normalized_date": normalized_date,
            "explicit_year": bool(re.search(r"\b20\d{2}\b", text)),
            "role": role,
            "role_confidence": confidence,
            "context": compact_summary(context, 220),
            "temporal_relations": safe_temporal_relation_records(
                temporal_relations_for_text(
                    role_text,
                    query=query,
                    normalized_date=normalized_date,
                )
            ),
        }
        if endpoint:
            mention["range_endpoint"] = endpoint
        mentions.append(mention)
    return mentions


def temporal_candidate_table(query: str, spans: list[dict[str, Any]], *, limit: int = 24) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ordered_target_roles = temporal_target_roles(query)
    target_roles = set(ordered_target_roles)
    target_role_order = {role: index for index, role in enumerate(ordered_target_roles)}
    query_lower = query.lower()
    for span in spans:
        mentions = span.get("temporal_mentions") or []
        if not mentions:
            continue
        for mention in mentions:
            context = str(mention.get("context") or "")
            role = str(mention.get("role") or "")
            confidence = float(mention.get("role_confidence", 0.0) or 0.0)
            candidates.append(
                {
                    "source_span_id": span.get("id"),
                    "speaker": span.get("speaker"),
                    "timeline_index": span.get("timeline_index"),
                    "history_index": span.get("history_index"),
                    "role": role,
                    "date_text": mention.get("text"),
                    "normalized_date": mention.get("normalized_date"),
                    "explicit_year": mention.get("explicit_year"),
                    "context": context,
                    "confidence": confidence,
                    "target_role_match": role in target_roles,
                    "target_role_order": target_role_order.get(role, 999),
                    "query_overlap": query_context_overlap_score(query_lower, context.lower()),
                    "range_endpoint": mention.get("range_endpoint"),
                    "temporal_relations": [
                        item for item in (mention.get("temporal_relations") or []) if isinstance(item, dict)
                    ],
                }
            )
    candidates.sort(
        key=lambda item: (
            0 if item.get("target_role_match") else 1,
            int(item.get("target_role_order") if item.get("target_role_order") is not None else 999),
            0 if item.get("speaker") == "user" else 1,
            -float(item.get("confidence") or 0.0),
            -int(item.get("query_overlap") or 0),
            int(item.get("timeline_index") or 10**9),
            str(item.get("normalized_date") or ""),
            str(item.get("source_span_id") or ""),
        )
    )
    return dedupe_temporal_candidates(candidates)[:limit]


def temporal_model_candidates(
    query: str,
    coverage_candidates: list[dict[str, Any]],
    source_spans: list[dict[str, Any]],
    *,
    limit: int = 48,
) -> list[dict[str, Any]]:
    """Merge stored temporal candidates with source-span-derived candidates.

    Retrieval-time temporal coverage can be truncated before a later endpoint
    profile is known. For model-view construction, regenerate the same typed
    candidate table from retained source spans and merge it with coverage
    candidates instead of adding answer-layer fallbacks.
    """
    merged: list[dict[str, Any]] = [dict(candidate) for candidate in coverage_candidates if isinstance(candidate, dict)]
    seen = {_temporal_candidate_identity(candidate) for candidate in merged}
    loose_index = {_temporal_candidate_loose_identity(candidate): candidate for candidate in merged}
    temporal_spans: list[dict[str, Any]] = []
    for span in source_spans[:96]:
        content = str(span.get("content") or span.get("context") or span.get("text") or "")
        if not content.strip():
            continue
        mentions = temporal_mentions(query, content, span.get("timestamp"))
        if not mentions:
            continue
        row = dict(span)
        row.setdefault("id", span.get("source_span_id") or span.get("turn_id"))
        row["content"] = content
        row["temporal_mentions"] = mentions
        temporal_spans.append(row)
    for candidate in temporal_candidate_table(query, temporal_spans, limit=limit):
        loose_key = _temporal_candidate_loose_identity(candidate)
        existing = loose_index.get(loose_key)
        if existing is not None:
            merge_temporal_candidate_metadata(existing, candidate)
            continue
        key = _temporal_candidate_identity(candidate)
        if key in seen:
            continue
        seen.add(key)
        loose_index[loose_key] = candidate
        merged.append(candidate)
    merged.sort(
        key=lambda item: (
            0 if item.get("target_role_match") else 1,
            int(item.get("target_role_order") if item.get("target_role_order") is not None else 999),
            0 if item.get("speaker") == "user" else 1,
            -float(item.get("confidence") or 0.0),
            -int(item.get("query_overlap") or 0),
            int(item.get("timeline_index") or 10**9),
            str(item.get("normalized_date") or ""),
            str(item.get("source_span_id") or ""),
        )
    )
    return dedupe_temporal_candidates(merged)[:limit]


def _temporal_candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(candidate.get("source_span_id") or candidate.get("id") or ""),
        str(candidate.get("role") or ""),
        str(candidate.get("normalized_date") or ""),
        compact_summary(str(candidate.get("context") or ""), 80),
    )


def _temporal_candidate_loose_identity(candidate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate.get("source_span_id") or candidate.get("id") or ""),
        str(candidate.get("role") or ""),
        str(candidate.get("normalized_date") or ""),
    )


def merge_temporal_candidate_metadata(existing: dict[str, Any], candidate: dict[str, Any]) -> None:
    if candidate.get("explicit_year") and not existing.get("explicit_year"):
        existing["explicit_year"] = candidate.get("explicit_year")
    for key in ["date_text", "range_endpoint", "history_index"]:
        if candidate.get(key) and not existing.get(key):
            existing[key] = candidate.get(key)
    if len(str(candidate.get("context") or "")) > len(str(existing.get("context") or "")):
        existing["context"] = candidate.get("context")
    if candidate.get("temporal_relations") and not existing.get("temporal_relations"):
        existing["temporal_relations"] = [
            item for item in (candidate.get("temporal_relations") or []) if isinstance(item, dict)
        ]


def temporal_answer_candidates(
    query: str,
    temporal_candidates: list[dict[str, Any]],
    temporal_range_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    endpoint_profiles = temporal_endpoint_profiles(query)
    if not endpoint_profiles:
        return []
    if len(endpoint_profiles) == 1 and temporal_range_pairs:
        return temporal_answer_candidates_from_ranges(query, endpoint_profiles[0], temporal_range_pairs)
    if len(endpoint_profiles) < 2 or not temporal_candidates:
        return []
    start_profile, end_profile = endpoint_profiles[0], endpoint_profiles[1]
    start_ranked = rank_temporal_endpoint_candidates(temporal_candidates, start_profile)
    end_ranked = rank_temporal_endpoint_candidates(temporal_candidates, end_profile)
    rows: list[dict[str, Any]] = []
    for start_item, start_score in start_ranked[:8]:
        original_start_date = parse_iso_date(str(start_item.get("normalized_date") or ""))
        if not original_start_date:
            continue
        for end_item, end_score in end_ranked[:10]:
            original_end_date = parse_iso_date(str(end_item.get("normalized_date") or ""))
            if not original_end_date or original_end_date == original_start_date:
                continue
            start_date = original_start_date
            end_date = original_end_date
            start_date, end_date = align_temporal_pair_years(start_date, start_item, end_date, end_item)
            if not start_date or not end_date or end_date == start_date:
                continue
            if end_date < start_date:
                continue
            day_difference = abs((end_date - start_date).days)
            if day_difference > 3650:
                continue
            year_aligned = start_date != original_start_date or end_date != original_end_date
            pair_score = round(start_score + end_score + temporal_pair_bonus(query, start_item, end_item, year_aligned=year_aligned), 3)
            confidence = temporal_pair_confidence(query, pair_score, start_item, end_item, year_aligned=year_aligned)
            rows.append(
                {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "day_difference": day_difference,
                    "confidence": confidence,
                    "start_label": start_profile["label"],
                    "end_label": end_profile["label"],
                    "start_role": start_item.get("role"),
                    "end_role": end_item.get("role"),
                    "start_range_endpoint": start_item.get("range_endpoint"),
                    "end_range_endpoint": end_item.get("range_endpoint"),
                    "start_source_span_id": start_item.get("source_span_id"),
                    "end_source_span_id": end_item.get("source_span_id"),
                    "start_speaker": start_item.get("speaker"),
                    "end_speaker": end_item.get("speaker"),
                    "score": pair_score,
                    "year_aligned": year_aligned,
                    "start_context": compact_summary(str(start_item.get("context") or ""), 180),
                    "end_context": compact_summary(str(end_item.get("context") or ""), 180),
                }
            )
    rows.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            int(item.get("day_difference") or 10**9),
            str(item.get("start_date") or ""),
            str(item.get("end_date") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("start_label") or ""),
            str(row.get("end_label") or ""),
            str(row.get("start_date") or ""),
            str(row.get("end_date") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= 6:
            break
    return out


def direct_date_answer_candidates(
    query: str,
    temporal_candidates: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Rank single-date answers for direct "when/date/deadline" questions.

    Duration questions use temporal_answer_candidates because they need a pair
    of endpoints. Direct date questions need a different contract: one selected
    date plus the source context and common output formats. Keeping this as a
    typed model-pack section avoids adding category-specific answer templates.
    """

    if not _query_requests_direct_date(query):
        return []
    profile = temporal_endpoint_profile(query, fallback_label="requested_date")
    ranked_by_profile = {
        _temporal_candidate_loose_identity(item): score
        for item, score in rank_temporal_endpoint_candidates(temporal_candidates, profile)
    }
    query_lower = query.lower()
    slot_terms = set(profile.get("slot_terms") or set())
    required_slot_terms = set(profile.get("required_slot_terms") or set())
    rows: list[dict[str, Any]] = []
    for candidate in temporal_candidates:
        normalized = str(candidate.get("normalized_date") or "")
        parsed = parse_iso_date(normalized)
        if not parsed:
            continue
        context = str(candidate.get("context") or "")
        context_lower = context.lower()
        profile_score = ranked_by_profile.get(_temporal_candidate_loose_identity(candidate), 0.0)
        overlap = int(candidate.get("query_overlap") or query_context_overlap_score(query_lower, context_lower))
        slot_score = temporal_keyword_score(context_lower, slot_terms)
        required_slot_match = temporal_required_slot_match(context_lower, context_lower, required_slot_terms)
        direct_score = min(4.0, profile_score)
        direct_score += min(2.4, overlap * 0.55)
        direct_score += min(1.8, slot_score)
        direct_score += min(1.0, float(candidate.get("confidence") or 0.0))
        if candidate.get("speaker") == "user":
            direct_score += 0.55
        if candidate.get("target_role_match"):
            direct_score += 0.5
        if not required_slot_match and required_slot_terms:
            direct_score -= 2.3
        if _query_requests_deadline_date(query_lower):
            direct_score += _direct_deadline_role_bonus(candidate, context_lower, query_lower)
        if _query_requests_meeting_or_event_date(query_lower):
            direct_score += _direct_event_role_bonus(candidate, context_lower, query_lower)
        if overlap <= 0 and slot_score <= 0.0 and profile_score <= 0.0:
            direct_score -= 1.8
        if direct_score < 1.6:
            continue
        rows.append(
            {
                "type": "direct_date",
                "answer_value": normalized,
                "date_text": candidate.get("date_text"),
                "date_mm_dd_yyyy": f"{parsed.month:02d}/{parsed.day:02d}/{parsed.year:04d}",
                "date_month_day_year": f"{MONTH_NAMES[parsed.month]} {parsed.day}, {parsed.year}",
                "role": candidate.get("role"),
                "source_span_id": candidate.get("source_span_id"),
                "speaker": candidate.get("speaker"),
                "confidence": round(min(0.98, max(0.0, direct_score / 8.5)), 3),
                "score": round(direct_score, 3),
                "query_overlap": overlap,
                "context": compact_summary(context, 220),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            0 if item.get("speaker") == "user" else 1,
            str(item.get("answer_value") or ""),
            str(item.get("source_span_id") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("answer_value") or ""), str(row.get("role") or ""), str(row.get("source_span_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _query_requests_direct_date(query: str) -> bool:
    lower = query.lower()
    if re.search(r"\bhow\s+(?:many|long)\b", lower):
        return False
    if re.search(r"\b(?:between|from)\b.+\b(?:and|to|until|through|till)\b", lower):
        return False
    return bool(
        re.search(r"\bwhen\s+(?:is|was|are|were|did|do|does|will|should)\b", lower)
        or re.search(r"\b(?:what|which)\s+date\b", lower)
        or re.search(r"\b(?:due|deadline|scheduled)\b", lower)
    )


def _query_requests_deadline_date(query_lower: str) -> bool:
    return bool(re.search(r"\b(?:due|deadline|submit|submission|file|filing|target date)\b", query_lower))


def _query_requests_meeting_or_event_date(query_lower: str) -> bool:
    return bool(re.search(r"\b(?:meeting|meetings|event|workshop|webinar|conference|festival|appointment|call)\b", query_lower))


def _direct_deadline_role_bonus(candidate: dict[str, Any], context_lower: str, query_lower: str) -> float:
    role = str(candidate.get("role") or "")
    bonus = 0.0
    if role in {"deadline_date", "submission_deadline", "deployment_deadline", "mvp_deadline"}:
        bonus += 0.9
    if re.search(r"\b(?:submit|submission)\b", query_lower):
        if re.search(r"\b(?:submit|submission|ready for submission|final version)\b", context_lower):
            bonus += 1.8
        elif role == "deployment_deadline":
            bonus -= 1.2
    if re.search(r"\b(?:by|before|no later than|due|deadline)\b", context_lower):
        bonus += 0.6
    if candidate.get("speaker") == "assistant" and temporal_context_looks_like_assistant_plan_deadline(context_lower):
        bonus -= 1.2
    return bonus


def _direct_event_role_bonus(candidate: dict[str, Any], context_lower: str, query_lower: str) -> float:
    role = str(candidate.get("role") or "")
    bonus = 0.0
    if role in {"meeting_date", "event_date", "scheduled_date", "mentioned_date", "feature_finish_date"}:
        bonus += 0.35
    if re.search(r"\bmeeting", query_lower) and re.search(r"\b(?:meeting|met|schedule|scheduled|session|sessions|appointment|call)\b", context_lower):
        bonus += 1.2
    if re.search(r"\bmeeting", query_lower) and re.search(r"\b(?:worked with|final edits|complete(?:d)?\s+\d+\s+scenes|productivity)\b", context_lower):
        bonus -= 0.8
    if re.search(r"\bmeetings\b", query_lower) and re.search(r"\b(?:\d+\s+sessions|schedule finalized|scheduled)\b", context_lower):
        bonus += 0.9
    if re.search(r"\b(?:at|in)\s+[a-z]", query_lower) and query_context_overlap_score(query_lower, context_lower) >= 2:
        bonus += 0.7
    return bonus


def temporal_answer_candidates_from_ranges(
    query: str,
    profile: dict[str, Any],
    temporal_range_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not re.search(r"\b(?:between|from|range|period|duration)\b", query.lower()):
        return []
    rows: list[dict[str, Any]] = []
    for pair in temporal_range_pairs[:12]:
        start_date = parse_iso_date(str(pair.get("start_date") or ""))
        end_date = parse_iso_date(str(pair.get("end_date") or ""))
        if not start_date or not end_date or start_date == end_date:
            continue
        day_difference = abs((end_date - start_date).days)
        if day_difference > 3650:
            continue
        context = str(pair.get("context") or "")
        score = (
            float(pair.get("confidence") or 0.0)
            + int(pair.get("query_overlap") or 0) * 0.2
            + temporal_keyword_score(context, set(profile.get("keywords", set())))
        )
        rows.append(
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "day_difference": day_difference,
                "start_label": "range_start",
                "end_label": "range_end",
                "start_role": pair.get("start_role"),
                "end_role": pair.get("end_role"),
                "source_span_id": pair.get("source_span_id"),
                "speaker": pair.get("speaker"),
                "score": round(score, 3),
                "context": compact_summary(context, 220),
            }
        )
    rows.sort(key=lambda item: (-float(item.get("score") or 0.0), int(item.get("day_difference") or 10**9)))
    return rows[:4]


def temporal_endpoint_profiles(query: str) -> list[dict[str, Any]]:
    lower = query.lower()
    took_after = re.search(r"\bhow\s+long\s+did\s+it\s+take\b(.+?)\bafter\b(.+?)(?:\?|$)", lower)
    if took_after:
        return [
            temporal_endpoint_profile(took_after.group(2), fallback_label="start_event"),
            temporal_endpoint_profile(took_after.group(1), fallback_label="end_event"),
        ]
    between = re.search(r"\bbetween\b(.+?)\band\b(.+?)(?:\?|$)", lower)
    if between:
        return [
            temporal_endpoint_profile(between.group(1), fallback_label="start_event"),
            temporal_endpoint_profile(between.group(2), fallback_label="end_event"),
        ]
    from_to = re.search(r"\bfrom\b(.+?)\b(?:to|until|through|till)\b(.+?)(?:\?|$)", lower)
    if from_to:
        return [
            temporal_endpoint_profile(from_to.group(1), fallback_label="start_event"),
            temporal_endpoint_profile(from_to.group(2), fallback_label="end_event"),
        ]
    after = re.search(r"\bafter\b(.+?)\b(?:did\s+i|did\s+we|do\s+i|do\s+we|had\s+i|have\s+i|was\s+i|were\s+we|could\s+i|can\s+i)\b(.+?)(?:\?|$)", lower)
    if after:
        return [
            temporal_endpoint_profile(after.group(1), fallback_label="start_event"),
            temporal_endpoint_profile(after.group(2), fallback_label="end_event"),
        ]
    if re.search(r"\b(?:range|period|duration)\b", lower):
        return [temporal_endpoint_profile(lower, fallback_label="date_range")]
    return []


def temporal_endpoint_profile(text: str, *, fallback_label: str) -> dict[str, Any]:
    lower = text.lower()
    key_terms = temporal_profile_terms(lower)
    slot_terms = temporal_slot_terms(key_terms)
    required_slot_terms = temporal_required_slot_terms(slot_terms)
    if re.search(r"\b(?:planned|planning|plan)\b.{0,80}\b(?:reach|reached|goal|target|save|saved|fund)\b", lower) or (
        re.search(r"\b(?:reach|reached|saved)\b", lower) and re.search(r"\b(?:goal|fund|target)\b", lower)
    ):
        return {
            "label": "goal_state_date",
            "roles": {"goal_state_date", "mentioned_date", "completion_date"},
            "keywords": _expand_temporal_keywords({"saved", "save", "reached", "reach", "goal", "fund", "target", "full"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:reach|reached|save|saved)\b", r"\b(?:goal|target|fund)\b"],
            "require_slot_overlap": True,
        }
    if re.search(r"\b(?:planned|planning|plan)\b", lower):
        return {
            "label": "planned_event_date",
            "roles": {"planned_event_date", "mentioned_date", "start_date", "completion_date"},
            "keywords": _expand_temporal_keywords({"planned", "planning", "plan", "scheduled", "schedule"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:planned|planning|plan)\b"],
            "weak_action_patterns": [r"\b(?:scheduled|schedule|scheduling)\b"],
        }
    if re.search(r"\b(?:missed|missing|skipped)\b", lower):
        return {
            "label": "missed_event_date",
            "roles": {"missed_event_date", "mentioned_date", "start_date"},
            "keywords": _expand_temporal_keywords({"missed", "missing", "skipped"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:missed|missing|skipped|skip)\b"],
            "negative_action_patterns": [r"\b(?:never|not|no|without|haven['’]?t|hadn['’]?t)\b[^.;!?]{0,50}\b(?:missed|missing|skipped|skip)\b"],
            "require_action": True,
        }
    if re.search(r"\b(?:submitted|submit|submission)\b", lower) and not re.search(r"\b(?:deadline|due)\b", lower):
        return {
            "label": "submission_date",
            "roles": {"submission_date", "mentioned_date", "completion_date"},
            "keywords": _expand_temporal_keywords({"submitted", "submit", "submission"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:submitted|submit|submission)\b"],
        }
    if re.search(r"\b(?:follow[-\s]?up|check[-\s]?in)\b", lower):
        return {
            "label": "followup_date",
            "roles": {"followup_date", "meeting_date", "mentioned_date", "scheduled_date"},
            "keywords": _expand_temporal_keywords({"followup", "follow-up", "checkin", "check-in", "meeting", "scheduled"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:follow[-\s]?up|check[-\s]?in)\b"],
            "weak_action_patterns": [r"\b(?:scheduled|schedule|scheduling)\b"],
            "require_action": True,
        }
    if re.search(r"\b(?:submit|submitted|submission|submitting)\b", lower) and re.search(r"\b(?:deadline|due|by|before|no later|final)\b", lower):
        return {
            "label": "submission_deadline",
            "roles": {"submission_deadline", "deadline_date", "completion_date", "mentioned_date"},
            "keywords": _expand_temporal_keywords({"submit", "submitted", "submission", "submitting", "final", "due", "deadline", "ready"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": temporal_required_slot_terms(slot_terms | {"submission"}),
            "action_patterns": [r"\b(?:submit|submitted|submission|submitting|ready\s+for\s+submission|final\s+version)\b", r"\b(?:deadline|due|by|before|no later)\b"],
            "weak_action_patterns": [r"\b(?:complete|completed|finish|finished|ready|final)\b"],
            "require_slot_overlap": True,
        }
    if re.search(r"\b(?:deadline|due|file|filing|submit|submission)\b", lower):
        return {
            "label": "deadline_date",
            "roles": {"deadline_date", "deployment_deadline", "mvp_deadline", "submission_deadline", "mentioned_date"},
            "keywords": _expand_temporal_keywords({"deadline", "due", "file", "filing", "submit", "submission", "target"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:deadline|due|target date|no later|by)\b"],
            "require_slot_overlap": bool(slot_terms),
        }
    if re.search(r"\b(?:webinar|workshop|conference|festival|event)\b", lower):
        event_terms = {
            term
            for term in ["webinar", "workshop", "conference", "festival", "event"]
            if re.search(rf"\b{term}\b", lower)
        }
        return {
            "label": "event_date",
            "roles": {"event_date", "mentioned_date", "scheduled_date", "start_date"},
            "keywords": _expand_temporal_keywords({"webinar", "workshop", "conference", "festival", "event", "scheduled", "upcoming"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [rf"\b{re.escape(term)}\b" for term in (event_terms or {"webinar", "workshop", "conference", "festival", "event"})],
            "event_terms": event_terms,
            "require_slot_overlap": bool(slot_terms),
        }
    if re.search(r"\b(?:downloaded?|installed|borrowed|checked out|acquired)\b", lower):
        return {
            "label": "download_date",
            "roles": {"download_date", "mentioned_date"},
            "keywords": _expand_temporal_keywords({"downloaded", "download", "installed", "borrowed", "checked", "acquired"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:downloaded?|installed|borrowed|checked out|acquired)\b"],
        }
    if re.search(r"\b(?:rescheduled|reschedule|moved|pushed|postponed)\b", lower):
        return {
            "label": "reschedule_date",
            "roles": {"reschedule_date", "mentioned_date"},
            "keywords": _expand_temporal_keywords({"rescheduled", "reschedule", "moved", "pushed", "postponed"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:rescheduled|reschedule|moved|pushed|postponed)\b"],
        }
    if re.search(r"\b(?:decided|decision|reject|rejected|decline|declined|chose|chosen)\b", lower):
        return {
            "label": "decision_date",
            "roles": {"decision_date", "mentioned_date"},
            "keywords": _expand_temporal_keywords({"decided", "decision", "reject", "rejected", "decline", "declined", "chose", "chosen"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:decided|decision|reject|rejected|decline|declined|chose|chosen)\b"],
        }
    if re.search(r"\b(?:meeting|met|call|appointment)\b", lower):
        return {
            "label": "meeting_date",
            "roles": {"meeting_date", "mentioned_date", "scheduled_date", "start_date"},
            "keywords": _expand_temporal_keywords({"meeting", "met", "call", "appointment", "schedule", "scheduled", "scheduling"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:meeting|met|call|appointment)\b"],
        }
    if re.search(r"\b(?:testing|test|deployment|deploy)\b", lower) and re.search(r"\b(?:start|begin|period|phase)\b", lower):
        return {
            "label": "testing_or_deployment_start",
            "roles": {"testing_start_date", "deployment_deadline", "mvp_deadline", "completion_date", "feature_finish_date", "start_date"},
            "keywords": _expand_temporal_keywords({"testing", "test", "deployment", "deploy", "mvp", "deadline", "completion", "complete", "project"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:testing|test|deployment|deploy|mvp)\b"],
        }
    if re.search(r"\b(?:improved?|improvement|score|quiz|practice|practicing|practiced|accuracy)\b", lower):
        return {
            "label": "progress_improvement_date",
            "roles": {"completion_date", "feature_finish_date", "mentioned_date"},
            "keywords": _expand_temporal_keywords({"improved", "improvement", "score", "quiz", "practice", "practicing", "practiced", "accuracy", "problems"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:improved?|improvement|score|practice|practicing|practiced|accuracy)\b"],
        }
    if re.search(r"\b(?:completed|complete|completion|finished|finish|done|read|reading)\b", lower):
        return {
            "label": "completion_date",
            "roles": {"completion_date", "feature_finish_date", "mentioned_date", "deadline_date"},
            "keywords": _expand_temporal_keywords({"completed", "complete", "completion", "finished", "finish", "done", "read", "reading"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:completed|complete|completion|finished|finishing|finish|done|read|reading)\b"],
        }
    if re.search(r"\b(?:started|starting|start|begin|began|challenge|focused|focus)\b", lower):
        return {
            "label": "start_date",
            "roles": {"start_date", "mentioned_date"},
            "keywords": _expand_temporal_keywords({"started", "starting", "start", "begin", "began", "challenge", "focused", "focus"} | key_terms),
            "key_terms": key_terms,
            "slot_terms": slot_terms,
            "required_slot_terms": required_slot_terms,
            "action_patterns": [r"\b(?:started|starting|start|begin|began|focused|focus)\b"],
        }
    keywords = {
        token
        for token in re.findall(r"[a-z0-9]+", lower)
        if len(token) > 3 and token not in {"when", "there", "were", "have", "between", "from", "with", "after", "before"}
    }
    return {
        "label": fallback_label,
        "roles": {"mentioned_date"},
        "keywords": _expand_temporal_keywords(keywords),
        "key_terms": key_terms or keywords,
        "slot_terms": temporal_slot_terms(key_terms or keywords),
        "required_slot_terms": temporal_required_slot_terms(temporal_slot_terms(key_terms or keywords)),
    }


def rank_temporal_endpoint_candidates(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[tuple[dict[str, Any], float]]:
    roles = set(profile.get("roles") or set())
    keywords = set(profile.get("keywords") or set())
    key_terms = set(profile.get("key_terms") or set())
    slot_terms = set(profile.get("slot_terms") or set())
    required_slot_terms = set(profile.get("required_slot_terms") or set())
    max_user_history_index = max(
        (
            int(candidate.get("history_index"))
            for candidate in candidates
            if candidate.get("speaker") == "user" and str(candidate.get("history_index") or "").isdigit()
        ),
        default=0,
    )
    ranked: list[tuple[dict[str, Any], float]] = []
    for candidate in candidates:
        normalized = str(candidate.get("normalized_date") or "")
        if not parse_iso_date(normalized):
            continue
        role = str(candidate.get("role") or "")
        context = str(candidate.get("context") or "")
        local_context = temporal_local_context(context, normalized)
        if str(profile.get("label") or "") == "meeting_date" and not temporal_context_implies_profile(local_context, profile):
            continue
        profile_match = temporal_profile_match(local_context, context, profile)
        if profile_match["negative"]:
            continue
        if profile.get("require_action") and profile_match["action"] <= 0.0:
            continue
        if str(profile.get("label") or "") == "missed_event_date" and candidate.get("range_endpoint") == "range_end":
            continue
        if profile.get("require_slot_overlap") and slot_terms and profile_match["slot"] <= 0.0:
            continue
        if required_slot_terms and not temporal_required_slot_match(local_context, context, required_slot_terms):
            score_penalty_for_required_slot = True
        else:
            score_penalty_for_required_slot = False
        score = 0.0
        user_goal_deadline_match = temporal_user_goal_deadline_match(candidate, local_context, profile)
        if role in roles:
            if role != "mentioned_date" or temporal_keyword_score(local_context, keywords) > 0 or temporal_context_implies_profile(local_context, profile):
                score += 2.0
            else:
                score += 0.5
        elif user_goal_deadline_match:
            score += 1.5
        if str(profile.get("label") or "") == "completion_date" and candidate.get("range_endpoint") == "range_start":
            if temporal_completion_query_targets_range_start(local_context, profile):
                score += 3.6
            else:
                score -= 0.8
        local_keyword_score = temporal_keyword_score(local_context, keywords)
        full_keyword_score = temporal_keyword_score(context, keywords)
        local_key_term_score = temporal_keyword_score(local_context, key_terms)
        score += min(2.0, local_keyword_score + full_keyword_score * 0.25)
        score += min(1.4, local_key_term_score)
        score += min(1.0, int(candidate.get("query_overlap") or 0) * 0.25)
        score += min(1.0, float(candidate.get("confidence") or 0.0))
        if candidate.get("speaker") == "user":
            score += 0.35
        if temporal_context_implies_profile(local_context, profile):
            score += 1.0
        score += min(2.2, profile_match["action"])
        score += min(1.7, profile_match["slot"])
        score += min(0.8, profile_match["weak_action"])
        score += temporal_candidate_source_bonus(candidate, local_context, profile)
        if profile_match["slot"] <= 0.0 and slot_terms:
            score -= 0.45
        if score_penalty_for_required_slot:
            score -= 2.4
        if profile_match["action"] <= 0.0 and profile.get("action_patterns"):
            score -= 0.55
        if key_terms and local_key_term_score <= 0.0 and str(profile.get("label") or "") in {"start_event", "end_event", "completion_date", "progress_improvement_date"}:
            score -= 0.65
        if user_goal_deadline_match:
            score += 2.0
            score += temporal_user_goal_recency_bonus(candidate, max_user_history_index)
        if str(profile.get("label") or "") == "missed_event_date" and profile_match["action"] > 0.0:
            if candidate.get("speaker") == "user":
                score += 1.4
                if role not in roles:
                    score += 1.1
            else:
                score -= 2.2
        if score >= 1.75:
            ranked.append((candidate, round(score, 3)))
    ranked.sort(
        key=lambda item: (
            -item[1],
            0 if item[0].get("speaker") == "user" else 1,
            str(item[0].get("normalized_date") or ""),
            str(item[0].get("source_span_id") or ""),
        )
    )
    return ranked


def temporal_candidate_source_bonus(candidate: dict[str, Any], local_context: str, profile: dict[str, Any]) -> float:
    label = str(profile.get("label") or "")
    lower = local_context.lower()
    bonus = 0.0
    if label in {"completion_date", "goal_state_date"}:
        if candidate.get("speaker") == "user" and re.search(r"\b(?:i|we)\b[^.;!?]{0,50}\b(?:finished|completed|reached|saved|just)\b", lower):
            bonus += 0.75
        if candidate.get("speaker") == "assistant" and re.search(
            r"\b(?:timeline|schedule|plan|step|phase|week|day|new timeline|example|target|deadline)\b", lower
        ):
            bonus -= 1.25
        if candidate.get("speaker") == "assistant" and label == "completion_date" and re.search(
            r"\*\*|####|(?:may|june|july|august|september|october|november|december)\s+\d{1,2}\s*[-–—]\s*(?:may|june|july|august|september|october|november|december)?\s*\d{0,2}",
            lower,
        ):
            bonus -= 1.0
    if label in {"deadline_date", "planned_event_date"}:
        if temporal_user_goal_deadline_match(candidate, local_context, profile):
            bonus += 0.9
        if candidate.get("speaker") == "assistant" and temporal_context_looks_like_assistant_plan_deadline(lower):
            bonus -= 1.6
    if label == "missed_event_date":
        if candidate.get("speaker") == "user":
            bonus += 0.45
        if candidate.get("speaker") == "assistant" and re.search(r"\b(?:current situation|assessment|revised plan|example schedule|you missed)\b", lower):
            bonus -= 1.2
    return bonus


def temporal_user_goal_deadline_match(candidate: dict[str, Any], local_context: str, profile: dict[str, Any]) -> bool:
    if candidate.get("speaker") != "user":
        return False
    label = str(profile.get("label") or "")
    if label not in {"deadline_date", "planned_event_date"}:
        return False
    lower = local_context.lower()
    if not re.search(r"\b(?:aim|aimed|plan|planned|goal|target|deadline|due|decision|decided|file|filing|submit|submission|complete|finish)\b", lower):
        return False
    if not re.search(r"\b(?:by|before|no later than|deadline of|deadline for|due)\b", lower):
        return False
    return bool(
        re.search(r"\b(?:i|we|my|our)\b", lower)
        or re.search(r"\b(?:decision|goal|plan|aim|target)\s+to\b", lower)
        or re.search(r"\b(?:file|submit|complete|finish)\s+(?:a|an|the|my|our)?\s*[a-z0-9 -]{0,80}\bby\b", lower)
    )


def temporal_user_goal_recency_bonus(candidate: dict[str, Any], max_user_history_index: int) -> float:
    if max_user_history_index <= 0:
        return 0.0
    try:
        history_index = int(candidate.get("history_index") or 0)
    except (TypeError, ValueError):
        return 0.0
    if history_index <= 0:
        return 0.0
    return min(3.2, max(0.0, history_index / max_user_history_index) * 3.2)


def temporal_context_looks_like_assistant_plan_deadline(context_lower: str) -> bool:
    if not re.search(r"\b(?:deadline|file|filing|submit|submission|target)\b", context_lower):
        return False
    return bool(
        re.search(r"\b(?:example timeline|timeline|schedule|plan|step|phase|week|day|draft|cover sheet|checklist)\b", context_lower)
        or re.search(r"\*\*|####|^[-*]\s", context_lower)
    )


def temporal_completion_query_targets_range_start(local_context: str, profile: dict[str, Any]) -> bool:
    lower = local_context.lower()
    key_terms = set(profile.get("key_terms") or set())
    if not re.search(r"\b(?:complet(?:e|ed|es|ing|ion)?|finished|finishing|finish|done)\b", lower):
        return False
    if not re.search(r"\b(?:from|between)\b", lower):
        return False
    local_terms = _expand_temporal_keywords(set(re.findall(r"[a-z0-9]+", lower)))
    return bool(local_terms & key_terms)


def temporal_profile_match(local_context: str, full_context: str, profile: dict[str, Any]) -> dict[str, float | bool]:
    local_lower = local_context.lower()
    full_lower = full_context.lower()
    negative = any(re.search(pattern, local_lower) for pattern in profile.get("negative_action_patterns") or [])
    action = 0.0
    for pattern in profile.get("action_patterns") or []:
        if re.search(pattern, local_lower):
            action += 1.15
        elif str(profile.get("label") or "") == "missed_event_date" and re.search(
            r"\b(?:after\s+)?missing\b[^.;!?]{0,30}\b(?:the\s+)?(?:[a-z0-9 -]+)?\s+(?:one|session|meeting|call)\b",
            full_lower,
        ):
            action += 0.85
        elif re.search(pattern, full_lower):
            action += 0.35
    event_terms = set(profile.get("event_terms") or set())
    if event_terms and not any(re.search(rf"\b{re.escape(term)}\b", local_lower) for term in event_terms):
        action = min(action, 0.35)
    weak_action = 0.0
    for pattern in profile.get("weak_action_patterns") or []:
        if re.search(pattern, local_lower):
            weak_action += 0.45
        elif re.search(pattern, full_lower):
            weak_action += 0.15
    slot = temporal_keyword_score(local_lower, set(profile.get("slot_terms") or set()))
    if slot <= 0.0:
        slot = min(0.45, temporal_keyword_score(full_lower, set(profile.get("slot_terms") or set())) * 0.35)
    return {
        "negative": negative,
        "action": action,
        "weak_action": weak_action,
        "slot": slot,
    }


def temporal_context_implies_profile(context: str, profile: dict[str, Any]) -> bool:
    lower = context.lower()
    label = str(profile.get("label") or "")
    if label == "meeting_date":
        if re.search(r"\bmeeting\s+(?:the\s+)?(?:deadline|target|goal|requirement|requirements)\b", lower):
            return False
        return bool(
            re.search(r"\b(?:schedule|scheduled|scheduling)\s+(?:a\s+|the\s+)?(?:meeting|call|appointment)\b", lower)
            or re.search(r"\b(?:meeting|call|appointment)\s+(?:with|at|on|for)\b", lower)
            or re.search(r"\bmet\s+(?:with\s+)?(?:him|her|them|[a-z][a-z]+)\b", lower)
        )
    if label == "testing_or_deployment_start":
        return bool(
            re.search(r"\b(?:testing|test|deployment|deploy)\b", lower)
            and re.search(r"\b(?:mvp|deadline|completion|complete|final|project|allow)\b", lower)
        )
    if label == "planned_event_date":
        return bool(re.search(r"\b(?:planned|planning|plan(?:ned)?)\b", lower))
    if label == "missed_event_date":
        return bool(re.search(r"\b(?:missed|missing|skipped)\b", lower)) and not bool(
            re.search(r"\b(?:never|not|no|without|haven['’]?t|hadn['’]?t)\b[^.;!?]{0,50}\b(?:missed|missing|skipped)\b", lower)
        )
    if label == "submission_date":
        return bool(re.search(r"\b(?:submitted|submit|submission)\b", lower))
    if label == "followup_date":
        return bool(re.search(r"\b(?:follow[-\s]?up|check[-\s]?in)\b", lower))
    if label == "event_date":
        return bool(re.search(r"\b(?:webinar|workshop|conference|festival|event)\b", lower))
    if label == "goal_state_date":
        return bool(re.search(r"\b(?:saved|reached|goal|fund|target)\b", lower))
    return False


def temporal_pair_bonus(query: str, start_item: dict[str, Any], end_item: dict[str, Any], *, year_aligned: bool = False) -> float:
    bonus = 0.0
    query_lower = query.lower()
    start_context = str(start_item.get("context") or "").lower()
    end_context = str(end_item.get("context") or "").lower()
    if "between" in query_lower:
        bonus += 0.2
    if start_item.get("speaker") == "user":
        bonus += 0.15
    if end_item.get("speaker") == "user":
        bonus += 0.1
    if re.search(r"\bmeeting\b", query_lower) and re.search(r"\bmeeting\b", start_context):
        bonus += 0.4
    if re.search(r"\btesting|deployment|project\b", query_lower) and re.search(r"\btesting|deployment|mvp|project\b", end_context):
        bonus += 0.4
    if _query_requests_current_or_upcoming(query_lower) and _context_marks_original_or_old(end_context):
        bonus -= 1.1
    if _pair_has_inferred_cross_year(start_item, end_item):
        bonus -= 1.4
    if year_aligned:
        bonus += 0.45
    if (
        start_item.get("source_span_id")
        and start_item.get("source_span_id") == end_item.get("source_span_id")
        and start_item.get("speaker") == "user"
        and end_item.get("speaker") == "user"
    ):
        shared_context = f"{start_context} {end_context}"
        if query_context_overlap_score(query_lower, shared_context) >= 4:
            bonus += 2.4
        else:
            bonus += 0.6
    return bonus


def temporal_pair_confidence(
    query: str,
    pair_score: float,
    start_item: dict[str, Any],
    end_item: dict[str, Any],
    *,
    year_aligned: bool = False,
) -> float:
    confidence = min(0.97, max(0.0, pair_score / 14.0))
    if _pair_has_inferred_cross_year(start_item, end_item):
        confidence = min(confidence, 0.72)
    if year_aligned:
        confidence = min(confidence, 0.80)
    if _query_requests_current_or_upcoming(query.lower()) and _context_marks_original_or_old(str(end_item.get("context") or "").lower()):
        confidence = min(confidence, 0.70)
    return round(confidence, 3)


def _query_requests_current_or_upcoming(query_lower: str) -> bool:
    return bool(re.search(r"\b(?:upcoming|current|currently|latest|rescheduled|new|now)\b", query_lower))


def _context_marks_original_or_old(context_lower: str) -> bool:
    return bool(re.search(r"\b(?:originally|initially|old|previous|previously|was scheduled|original date)\b", context_lower))


def _pair_has_inferred_cross_year(start_item: dict[str, Any], end_item: dict[str, Any]) -> bool:
    start_date = parse_iso_date(str(start_item.get("normalized_date") or ""))
    end_date = parse_iso_date(str(end_item.get("normalized_date") or ""))
    if not start_date or not end_date or start_date.year == end_date.year:
        return False
    return not (bool(start_item.get("explicit_year")) and bool(end_item.get("explicit_year")))


def align_temporal_pair_years(
    start_date: date,
    start_item: dict[str, Any],
    end_date: date,
    end_item: dict[str, Any],
) -> tuple[date | None, date | None]:
    start_explicit = bool(start_item.get("explicit_year"))
    end_explicit = bool(end_item.get("explicit_year"))
    if start_date.year == end_date.year:
        return start_date, end_date
    if start_explicit and not end_explicit:
        aligned_end = safe_date_from_parts(start_date.year, end_date.month, end_date.day)
        if aligned_end and aligned_end >= start_date:
            return start_date, aligned_end
    if end_explicit and not start_explicit:
        aligned_start = safe_date_from_parts(end_date.year, start_date.month, start_date.day)
        if aligned_start and aligned_start <= end_date:
            return aligned_start, end_date
    return start_date, end_date


def safe_date_from_parts(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def temporal_keyword_score(context: str, keywords: set[str]) -> float:
    if not keywords:
        return 0.0
    context_tokens = _expand_temporal_keywords(set(re.findall(r"[a-z0-9]+", context.lower())))
    return float(len(context_tokens & keywords)) * 0.35


def temporal_profile_terms(text: str) -> set[str]:
    return {
        normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 3
        and token
        not in {
            "after",
            "before",
            "between",
            "date",
            "days",
            "during",
            "from",
            "have",
            "many",
            "month",
            "months",
            "passed",
            "there",
            "time",
            "when",
            "with",
        }
    }


TEMPORAL_SLOT_STOPWORDS = {
    "complete",
    "completed",
    "completion",
    "deadline",
    "deadlines",
    "event",
    "festival",
    "follow",
    "followup",
    "meeting",
    "missed",
    "months",
    "planned",
    "planning",
    "reach",
    "reached",
    "scheduled",
    "start",
    "started",
    "submitted",
    "submission",
    "upcoming",
}


def temporal_slot_terms(terms: set[str]) -> set[str]:
    return {
        token
        for token in _expand_temporal_keywords(set(terms))
        if len(token) > 3 and token not in TEMPORAL_STOPWORDS and token not in TEMPORAL_SLOT_STOPWORDS
    }


def temporal_required_slot_terms(slot_terms: set[str]) -> set[str]:
    generic = {
        "abstract",
        "code",
        "daily",
        "draft",
        "emergency",
        "final",
        "full",
        "goal",
        "hiring",
        "letter",
        "patent",
        "project",
        "session",
        "sneaker",
        "walking",
        "writing",
    }
    required = {term for term in slot_terms if term not in generic}
    return required or set(slot_terms)


def temporal_required_slot_match(local_context: str, full_context: str, required_slot_terms: set[str]) -> bool:
    if not required_slot_terms:
        return True
    local_tokens = _expand_temporal_keywords(set(re.findall(r"[a-z0-9]+", local_context.lower())))
    if local_tokens & required_slot_terms:
        return True
    full_tokens = _expand_temporal_keywords(set(re.findall(r"[a-z0-9]+", full_context.lower())))
    return bool(full_tokens & required_slot_terms)


def _expand_temporal_keywords(keywords: set[str]) -> set[str]:
    expanded: set[str] = set()
    for keyword in keywords:
        token = normalize_token(str(keyword).lower())
        if not token:
            continue
        expanded.add(token)
        if token.endswith("e") and len(token) > 4:
            expanded.add(token[:-1])
        if token.endswith("ing") and len(token) > 6:
            expanded.add(token[:-3])
        if token.endswith("ed") and len(token) > 5:
            expanded.add(token[:-2])
        if token in {"submitted", "submitting"}:
            expanded.add("submit")
        if token in {"filing", "filed"}:
            expanded.add("file")
        if token in {"practicing", "practiced"}:
            expanded.add("practice")
        if token in {"improved", "improvement"}:
            expanded.add("improve")
        if token in {"planned", "planning"}:
            expanded.add("plan")
    return expanded


def temporal_local_context(context: str, normalized_date: str) -> str:
    parsed = parse_iso_date(normalized_date)
    if not parsed:
        return context
    month = parsed.strftime("%B").lower()
    short_month = parsed.strftime("%b").lower()
    day = str(parsed.day)
    year = str(parsed.year)
    patterns = [
        rf"\b{month}\s+{day}(?:st|nd|rd|th)?(?:,?\s+{year})?\b",
        rf"\b{short_month}\s+{day}(?:st|nd|rd|th)?(?:,?\s+{year})?\b",
        rf"\b{year}[-/]{parsed.month:02d}[-/]{parsed.day:02d}\b",
        rf"\b{year}[-/]{parsed.month}[-/]{parsed.day}\b",
    ]
    lower = context.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            start = max(0, match.start() - 90)
            end = min(len(context), match.end() + 90)
            local = context[start:end]
            relative_start = match.start() - start
            relative_end = match.end() - start
            return temporal_clause_around_date(local, relative_start, relative_end)
    return context


def temporal_clause_around_date(text: str, start: int, end: int) -> str:
    left = 0
    for pattern in [
        r",\s+(?:and|but|so|then)\s+",
        r"\s+\b(?:after|before|until|through)\b\s+",
        r";",
        r"\.\s+",
        r"\n",
        r"\s+but\s+",
    ]:
        matches = list(re.finditer(pattern, text[:start], flags=re.I))
        if matches:
            left = max(left, matches[-1].end())
    right = len(text)
    suffix = text[end:]
    for pattern in [
        r",\s+(?:and|but|so|then)\s+",
        r"\s+\b(?:after|before|until|through)\b\s+",
        r";",
        r"\.\s+",
        r"\n",
        r"\s+but\s+",
    ]:
        match = re.search(pattern, suffix, flags=re.I)
        if match:
            right = min(right, end + match.start())
    return text[left:right].strip() or text


def parse_iso_date(value: str) -> date | None:
    try:
        parts = [int(part) for part in value.split("-")]
    except ValueError:
        return None
    if len(parts) != 3:
        return None
    try:
        return date(parts[0], parts[1], parts[2])
    except ValueError:
        return None


def temporal_range_pairs(query: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_lower = query.lower()
    pairs: list[dict[str, Any]] = []
    default_year_by_span = {str(span.get("id") or ""): default_year(span.get("timestamp")) for span in spans}
    month_date = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?"
    )
    iso_date = r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}"
    date_expr = rf"(?:{iso_date}|{month_date})"
    range_re = re.compile(rf"({date_expr})\s*(?:[-\u2013\u2014]|\bto\b|\bthrough\b|\buntil\b)\s*({date_expr})", re.I)
    for span in spans:
        content = str(span.get("content") or "")
        if not content:
            continue
        span_default_year = default_year_by_span.get(str(span.get("id") or ""))
        for match in range_re.finditer(content):
            start_text, end_text = match.group(1), match.group(2)
            start_date = normalize_date_text(start_text, infer_year_for_match(content, match.start(1), match.end(1), span_default_year))
            end_date = normalize_date_text(end_text, infer_year_for_match(content, match.start(2), match.end(2), span_default_year))
            if not start_date or not end_date:
                continue
            context = compact_summary(mention_context(content, match.start(), match.end(), radius=180), 320)
            start_role, start_confidence = temporal_role(query_lower, role_context(content, match.start(1), match.end(1)).lower(), "range_start")
            end_role, end_confidence = temporal_role(query_lower, role_context(content, match.start(2), match.end(2)).lower(), "range_end")
            pairs.append(
                {
                    "source_span_id": span.get("id"),
                    "speaker": span.get("speaker"),
                    "timeline_index": span.get("timeline_index"),
                    "start_date": start_date,
                    "end_date": end_date,
                    "start_text": start_text,
                    "end_text": end_text,
                    "start_role": start_role,
                    "end_role": end_role,
                    "confidence": round(max(start_confidence, end_confidence), 3),
                    "query_overlap": query_context_overlap_score(query_lower, context.lower()),
                    "context": context,
                }
            )
    pairs.sort(
        key=lambda item: (
            0 if item.get("speaker") == "user" else 1,
            -int(item.get("query_overlap") or 0),
            -float(item.get("confidence") or 0.0),
            int(item.get("timeline_index") or 10**9),
            str(item.get("start_date") or ""),
            str(item.get("end_date") or ""),
        )
    )
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in pairs:
        key = (str(item.get("source_span_id") or ""), str(item.get("start_date") or ""), str(item.get("end_date") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= 8:
            break
    return out


def dedupe_temporal_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in candidates:
        key = (
            str(item.get("source_span_id") or ""),
            str(item.get("role") or ""),
            str(item.get("normalized_date") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def default_year(span_timestamp: object) -> int | None:
    if isinstance(span_timestamp, str):
        try:
            parsed = datetime.fromisoformat(span_timestamp)
            return parsed.year
        except ValueError:
            return None
    if hasattr(span_timestamp, "year"):
        return int(getattr(span_timestamp, "year"))
    return None


def mention_context(content: str, start: int, end: int, *, radius: int = 140) -> str:
    left = max(0, start - radius)
    right = min(len(content), end + radius)
    return content[left:right].strip()


def role_context(content: str, start: int, end: int) -> str:
    sentence = sentence_window(content, start, end)
    text = str(sentence["text"]).strip()
    if len(list(TEMPORAL_DATE_RE.finditer(text))) + len(list(TEMPORAL_MONTH_RE.finditer(text))) + len(list(TEMPORAL_CJK_MONTH_RE.finditer(text))) > 1:
        return mention_context(content, start, end, radius=60)
    if len(text) <= 260:
        return text
    return mention_context(content, start, end, radius=90)


def cjk_date_role_context(content: str, start: int, end: int) -> str:
    left = max(content.rfind(mark, 0, start) for mark in ("，", "。", "！", "？", "\n", ";", "；"))
    right_candidates = [content.find(mark, end) for mark in ("，", "。", "！", "？", "\n", ";", "；")]
    right_candidates = [value for value in right_candidates if value != -1]
    right = min(right_candidates) if right_candidates else len(content)
    return content[left + 1 : right].strip() or mention_context(content, start, end, radius=40)


def temporal_role(query_lower: str, context_lower: str, range_endpoint: str | None = None) -> tuple[str, float]:
    if re.search(r"(?:上线|发布|生产|投产)", context_lower):
        return "deployment_deadline", 0.94 if re.search(r"(?:上线|发布|生产|投产)", query_lower) else 0.78
    if re.search(r"(?:灰度|预发|测试环境)", context_lower):
        return "staging_date", 0.72
    explicit_deadline = has_any(context_lower, {"deadline", "deadlines", "due", "target date"})
    deadline = explicit_deadline or (
        has_any(context_lower, DEPLOYMENT_TERMS) and has_any(context_lower, {"by ", "before", "no later"})
    )
    completion = has_any(context_lower, COMPLETION_TERMS)
    feature = has_any(context_lower, FEATURE_TERMS)
    query_deployment_target = has_any(query_lower, DEPLOYMENT_TERMS | FINAL_TERMS)
    context_deployment_target = has_any(context_lower, DEPLOYMENT_TERMS)
    context_final_target = has_any(context_lower, FINAL_TERMS)
    context_mvp_target = has_any(context_lower, MVP_TERMS)
    query_submission_target = has_any(query_lower, SUBMISSION_TERMS)
    context_submission_target = has_any(context_lower, SUBMISSION_TERMS)
    query_feature_target = has_any(query_lower, FEATURE_TERMS)
    query_overlap = query_context_overlap(query_lower, context_lower)

    if re.search(r"\b(?:downloaded?|installed|borrowed|checked out|acquired)\b", context_lower):
        return "download_date", 0.86 if query_overlap or re.search(r"\b(?:downloaded?|installed|borrowed|checked out|acquired)\b", query_lower) else 0.68
    if re.search(r"\b(?:rescheduled|reschedule|moved|pushed|postponed)\b", context_lower):
        return "reschedule_date", 0.90 if query_overlap or "reschedul" in query_lower else 0.76
    if re.search(r"\b(?:decided|decision|chose|reject(?:ed|ing)?|declin(?:ed|e|ing))\b", context_lower):
        return "decision_date", 0.88 if query_overlap or re.search(r"\b(?:decided|decision|reject|decline)\b", query_lower) else 0.72
    if range_endpoint == "range_end" and context_deployment_target and query_deployment_target:
        return "deployment_deadline", 0.82
    if range_endpoint == "range_end" and feature and (query_overlap or not query_feature_target):
        return "feature_finish_date", 0.91 if query_overlap else 0.76
    if range_endpoint == "range_start" and "between" in query_lower:
        return "start_date", 0.70
    if (feature or query_overlap) and has_any(context_lower, {"target", "targets", "targeting", "due"}) and has_any(context_lower, {"by "}) and query_overlap:
        return "feature_finish_date", 0.88
    if has_any(context_lower, {"sprint"}) and deadline:
        return "sprint_deadline", 0.84
    if has_any(context_lower, {"sprint"}) and has_any(context_lower, {"end", "ends", "ending"}):
        return "sprint_end_date", 0.82
    if completion and feature and (query_overlap or not query_feature_target):
        return "feature_finish_date", 0.90 if query_overlap else 0.74
    if completion and feature:
        return "phase_end_date", 0.72
    if deadline and context_submission_target:
        return "submission_deadline", 0.92 if query_submission_target or query_overlap else 0.78
    if query_submission_target and context_submission_target and re.search(r"\b(?:by|before|due|deadline|ready for submission)\b", context_lower):
        return "submission_deadline", 0.88
    if completion:
        return "completion_date", 0.78
    if deadline and context_mvp_target and not (context_deployment_target and context_final_target):
        return "mvp_deadline", 0.88
    if deadline and context_deployment_target and (context_final_target or query_deployment_target):
        return "deployment_deadline", 0.94
    if deadline and query_deployment_target and not context_deployment_target:
        return "deployment_deadline", 0.76
    if deadline:
        return "deadline_date", 0.86
    if range_endpoint != "range_end" and "between" in query_lower and any(term in context_lower for term in ["from", "start", "begin", "begins", "starting"]):
        return "start_date", 0.70
    return "mentioned_date", 0.50


def normalize_date_text(text: str, default_year_value: int | None) -> str | None:
    iso = re.fullmatch(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text.strip())
    if iso:
        year, month, day = map(int, iso.groups())
        return safe_iso_date(year, month, day)
    cjk = re.fullmatch(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", text.strip())
    if cjk:
        if not default_year_value:
            return None
        return safe_iso_date(default_year_value, int(cjk.group(1)), int(cjk.group(2)))
    match = re.fullmatch(
        r"(?i)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?",
        text.strip(),
    )
    if not match:
        return None
    month_text, day_text, year_text = match.group(1), match.group(2), match.group(3)
    year = int(year_text) if year_text else default_year_value
    if not year:
        return None
    return safe_iso_date(year, MONTHS[month_text.lower()], int(day_text))


def infer_year_for_match(content: str, start: int, end: int, default_year_value: int | None) -> int | None:
    text = content[start:end]
    if re.search(r"\b20\d{2}\b", text):
        return default_year_value
    month_day = re.fullmatch(
        r"(?i)(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?",
        text.strip().rstrip(","),
    )
    if not month_day:
        return default_year_value
    current_month = MONTHS[month_day.group(1).lower()]
    current_day = int(month_day.group(2))
    previous_date = nearest_explicit_month_date(content, start, direction=-1)
    next_date = nearest_explicit_month_date(content, end, direction=1)
    if previous_date and looks_like_date_range(content[previous_date["end"] : start]):
        previous_month = int(previous_date["month"])
        previous_day = int(previous_date["day"])
        previous_year = int(previous_date["year"])
        if (current_month, current_day) < (previous_month, previous_day):
            return previous_year + 1
        return previous_year
    if next_date and looks_like_date_range(content[end : next_date["start"]]):
        next_month = int(next_date["month"])
        next_day = int(next_date["day"])
        next_year = int(next_date["year"])
        if (current_month, current_day) > (next_month, next_day):
            return next_year - 1
        return next_year
    sentence = sentence_window(content, start, end)
    years = [(abs((start + end) // 2 - (sentence["offset"] + match.start())), int(match.group(1))) for match in YEAR_RE.finditer(sentence["text"])]
    if years:
        years.sort(key=lambda item: item[0])
        return years[0][1]
    wider = nearby_year(content, start, end)
    if wider is not None:
        return wider
    return default_year_value


def nearest_explicit_month_date(content: str, index: int, *, direction: int) -> dict[str, int] | None:
    window_size = 100
    if direction < 0:
        left = max(0, index - window_size)
        matches = list(EXPLICIT_MONTH_DAY_YEAR_RE.finditer(content[left:index]))
        if not matches:
            return None
        match = matches[-1]
        return {
            "start": left + match.start(),
            "end": left + match.end(),
            "month": MONTHS[match.group(1).lower()],
            "day": int(match.group(2)),
            "year": int(match.group(3)),
        }
    right = min(len(content), index + window_size)
    match = next(EXPLICIT_MONTH_DAY_YEAR_RE.finditer(content[index:right]), None)
    if not match:
        return None
    return {
        "start": index + match.start(),
        "end": index + match.end(),
        "month": MONTHS[match.group(1).lower()],
        "day": int(match.group(2)),
        "year": int(match.group(3)),
    }


def nearest_month_date(content: str, index: int, *, direction: int) -> dict[str, int] | None:
    window_size = 100
    if direction < 0:
        left = max(0, index - window_size)
        matches = list(TEMPORAL_MONTH_RE.finditer(content[left:index]))
        if not matches:
            return None
        match = matches[-1]
        return {"start": left + match.start(), "end": left + match.end()}
    right = min(len(content), index + window_size)
    match = next(TEMPORAL_MONTH_RE.finditer(content[index:right]), None)
    if not match:
        return None
    return {"start": index + match.start(), "end": index + match.end()}


def looks_like_date_range(text: str) -> bool:
    return bool(re.fullmatch(r"[\s,;:()]*[-\u2013\u2014]|[\s,;:()]*\b(?:to|through|until|and)\b[\s,;:()]*", text.strip(), re.I))


def range_endpoint(content: str, start: int, end: int) -> str | None:
    previous_date = nearest_month_date(content, start, direction=-1)
    if previous_date and looks_like_date_range(content[previous_date["end"] : start]):
        return "range_end"
    next_date = nearest_month_date(content, end, direction=1)
    if next_date and looks_like_date_range(content[end : next_date["start"]]):
        return "range_start"
    return None


def sentence_window(content: str, start: int, end: int) -> dict[str, object]:
    left_candidates = [content.rfind(mark, 0, start) for mark in [".", "\n", "?", "!"]]
    left = max(left_candidates)
    left = 0 if left < 0 else left + 1
    right_candidates = [pos for pos in [content.find(mark, end) for mark in [".", "\n", "?", "!"]] if pos >= 0]
    right = min(right_candidates) if right_candidates else len(content)
    return {"text": content[left:right], "offset": left}


def has_any(text: str, terms: set[str]) -> bool:
    for term in terms:
        if not term:
            continue
        if not term.replace("_", "").isalnum():
            if term in text:
                return True
            continue
        if re.search(rf"\b{re.escape(term)}\b", text):
            return True
    return False


def query_context_overlap(query_lower: str, context_lower: str) -> bool:
    return query_context_overlap_score(query_lower, context_lower) > 0


def query_context_overlap_score(query_lower: str, context_lower: str) -> int:
    query_tokens = {normalize_token(token) for token in tokenize(query_lower)} | cjk_temporal_tokens(query_lower)
    context_tokens = {normalize_token(token) for token in tokenize(context_lower)} | cjk_temporal_tokens(context_lower)
    query_tokens = {
        token
        for token in query_tokens
        if (len(token) > 3 or token.isdigit()) and token not in TEMPORAL_STOPWORDS
    }
    context_tokens = {
        token
        for token in context_tokens
        if len(token) > 3 or token.isdigit()
    }
    return len(query_tokens & context_tokens)


def normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def cjk_temporal_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.add(chunk)
        for size in (2, 3, 4):
            tokens.update(chunk[index : index + size] for index in range(0, max(0, len(chunk) - size + 1)))
    return {token for token in tokens if token not in {"什么", "时候", "日期", "时间"}}


def temporal_target_roles(query: str) -> list[str]:
    query_lower = query.lower()
    roles: list[str] = []
    if re.search(r"(?:上线|发布|生产|投产)", query):
        roles.append("deployment_deadline")
    if has_any(query_lower, {"sprint"}) and has_any(query_lower, {"end", "ends", "ending"}):
        roles.append("sprint_end_date")
    if has_any(query_lower, FEATURE_TERMS) and has_any(query_lower, COMPLETION_TERMS | {"finish", "finishing"}):
        roles.append("feature_finish_date")
    if has_any(query_lower, SUBMISSION_TERMS) and has_any(query_lower, DEADLINE_TERMS | {"before", "no later", "ready"}):
        roles.append("submission_deadline")
    if has_any(query_lower, DEPLOYMENT_TERMS | FINAL_TERMS) and has_any(query_lower, DEADLINE_TERMS):
        roles.append("deployment_deadline")
    if has_any(query_lower, START_TERMS):
        roles.append("start_date")
    if re.search(r"\b(?:downloaded?|installed|borrowed|checked out|acquired)\b", query_lower):
        roles.append("download_date")
    if re.search(r"\b(?:completed|completion|finished|finish)\b", query_lower):
        roles.append("completion_date")
    if re.search(r"\b(?:decided|decision|reject|decline)\b", query_lower):
        roles.append("decision_date")
    if re.search(r"\b(?:rescheduled|reschedule|moved|pushed|postponed)\b", query_lower):
        roles.append("reschedule_date")
    if not roles and has_any(query_lower, DEADLINE_TERMS):
        roles.append("deadline_date")
    return list(dict.fromkeys(roles))


def mention_target_roles(query_lower: str, context_lower: str, role: str) -> list[str]:
    target_roles = temporal_target_roles(query_lower)
    if role in target_roles:
        return [role]
    if "deployment_deadline" in target_roles and role == "deadline_date" and not has_any(context_lower, MVP_TERMS):
        return ["deployment_deadline"]
    if "submission_deadline" in target_roles and role in {"deadline_date", "completion_date"} and has_any(context_lower, SUBMISSION_TERMS):
        return ["submission_deadline"]
    if (
        "feature_finish_date" in target_roles
        and role in {"completion_date", "phase_end_date"}
        and has_any(context_lower, FEATURE_TERMS)
        and query_context_overlap(query_lower, context_lower)
    ):
        return ["feature_finish_date"]
    return []


def nearby_year(content: str, start: int, end: int, *, radius: int = 600) -> int | None:
    left = max(0, start - radius)
    right = min(len(content), end + radius)
    center = start - left
    years = [(abs(center - match.start()), int(match.group(1))) for match in YEAR_RE.finditer(content[left:right])]
    if not years:
        return None
    years.sort(key=lambda item: item[0])
    return years[0][1]


def temporal_summary(query: str, content: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", content).strip()
    matches = list(TEMPORAL_DATE_RE.finditer(content)) + list(TEMPORAL_MONTH_RE.finditer(content)) + list(TEMPORAL_CJK_MONTH_RE.finditer(content))
    if len(normalized) <= limit and len(matches) <= 1:
        return normalized
    if not matches:
        return compact_summary(content, limit)
    query_lower = query.lower()
    windows: list[tuple[float, int, int]] = []
    for match in matches:
        context = cjk_date_role_context(content, match.start(), match.end()) if TEMPORAL_CJK_MONTH_RE.fullmatch(match.group(0)) else role_context(content, match.start(), match.end())
        role, _ = temporal_role(query_lower, context.lower(), range_endpoint(content, match.start(), match.end()))
        score = 1.0
        if role != "mentioned_date":
            score += 2.0
        if mention_target_roles(query_lower, context.lower(), role):
            score += 3.0
        if query_context_overlap(query_lower, context.lower()):
            score += 1.0
        if TEMPORAL_CJK_MONTH_RE.fullmatch(match.group(0)):
            left = content.find(context)
            right = left + len(context) if left >= 0 else match.end()
            left = max(0, left if left >= 0 else match.start() - 40)
        else:
            left = max(0, match.start() - 170)
            right = min(len(content), match.end() + 170)
        windows.append((score, left, right))
    if any(TEMPORAL_CJK_MONTH_RE.fullmatch(match.group(0)) for match in matches) and temporal_target_roles(query):
        _score, left, right = sorted(windows, key=lambda item: (-item[0], item[1]))[0]
        return compact_summary(content[left:right].strip(), limit)
    selected: list[tuple[int, int]] = []
    used = 0
    for _, left, right in sorted(windows, key=lambda item: (-item[0], item[1])):
        if any(not (right < old_left or left > old_right) for old_left, old_right in selected):
            continue
        snippet_len = right - left
        if selected and used + snippet_len + 5 > limit:
            continue
        selected.append((left, right))
        used += snippet_len + (5 if len(selected) > 1 else 0)
        if used >= limit:
            break
    if not selected:
        return compact_summary(content, limit)
    snippets = [re.sub(r"\s+", " ", content[left:right]).strip() for left, right in sorted(selected)]
    return compact_summary(" ... ".join(snippet for snippet in snippets if snippet), limit)


def safe_iso_date(year: int, month: int, day: int) -> str | None:
    try:
        value = datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None
    return value.date().isoformat()
