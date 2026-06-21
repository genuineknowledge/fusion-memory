from __future__ import annotations

import re
from typing import Any

from fusion_memory.core.text import compact_summary
from fusion_memory.retrieval.temporal_relations import (
    safe_temporal_relation_records,
    temporal_relation_summary_from_safe_records,
    temporal_relations_for_text,
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
    "times",
    "when",
    "which",
}


def build_value_history_table(query: str, spans: list[dict[str, Any]], facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_lower = query.lower()
    query_terms = value_summary_terms(query)
    for index, span in enumerate(spans):
        content = str(span.get("content") or "")
        values = dedupe_value_mentions(query_targeted_value_mentions(query, content) + value_mentions(content))
        if not values:
            continue
        lower = content.lower()
        if span.get("speaker") not in {"user", "assistant", "document"}:
            continue
        current_markers = bool(
            re.search(
                r"\b(?:current|currently|now|latest|new|updated|finally|final|revised|rescheduled|"
                r"added|spent|reached|achieved|improved|increased|secured|securing|scheduled)\b",
                lower,
            )
        )
        for value in values:
            context = str(value.get("context") or "")
            value_text = str(value.get("text") or "")
            value_type = str(value.get("type") or "")
            marker_strength = value_update_marker_strength(query_lower, lower, value_text)
            value_role = value_history_value_role(query, context, value_text, value_type)
            context_terms = value_summary_terms(context)
            relations = safe_temporal_relation_records(
                temporal_relations_for_text(
                    context or content,
                    query=query,
                    value_text=value_text,
                    value_type=value_type,
                    source_span_id=str(span.get("id") or "") or None,
                )
            )
            rows.append(
                {
                    "source_span_id": span.get("id"),
                    "speaker": span.get("speaker"),
                    "timeline_index": span.get("timeline_index"),
                    "history_index": span.get("history_index"),
                    "recency_rank": span.get("recency_rank"),
                    "value_type": value_type,
                    "value": value_text,
                    "context": context,
                    "subject_key": value_history_subject_key(query, context),
                    "current": value_role_is_current(value_role)
                    or (current_markers and value_role not in {"previous", "target", "example"}),
                    "query_overlap": len(query_terms & context_terms),
                    "span_query_overlap": len(query_terms & value_summary_terms(content)),
                    "slot_overlap": len(value_topic_terms(query) & value_topic_terms(context)),
                    "value_role": value_role,
                    "update_marker_strength": marker_strength,
                    "temporal_relations": relations,
                }
            )
    for fact in facts:
        text = str(fact.get("text") or "")
        values = dedupe_value_mentions(query_targeted_value_mentions(query, text) + value_mentions(text))
        for value in values:
            context = str(value.get("context") or "")
            value_text = str(value.get("text") or "")
            value_type = str(value.get("type") or "")
            value_role = value_history_value_role(query, context or text, value_text, value_type)
            context_terms = value_summary_terms(context or text)
            relations = safe_temporal_relation_records(
                temporal_relations_for_text(
                    context or text,
                    query=query,
                    value_text=value_text,
                    value_type=value_type,
                    source_span_id=str(next(iter(fact.get("source_span_ids") or []), "") or "") or None,
                )
            )
            rows.append(
                {
                    "source_span_id": next(iter(fact.get("source_span_ids") or []), None),
                    "speaker": "fact",
                    "timeline_index": None,
                    "history_index": None,
                    "recency_rank": None,
                    "value_type": value_type,
                    "value": value_text,
                    "context": context,
                    "subject_key": value_history_subject_key(query, context or text),
                    "current": value_role_is_current(value_role) or fact.get("polarity") in {"positive", "current"},
                    "query_overlap": len(query_terms & context_terms),
                    "span_query_overlap": len(query_terms & value_summary_terms(text)),
                    "slot_overlap": len(value_topic_terms(query) & value_topic_terms(context or text)),
                    "value_role": value_role,
                    "update_marker_strength": value_update_marker_strength(query_lower, text.lower(), value_text),
                    "temporal_relations": relations,
                }
            )
    rows.sort(key=value_history_sort_key(query))
    grouped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("subject_key") or ""),
            str(row.get("value_type") or ""),
            str(row.get("value") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        grouped.append(row)
        if len(grouped) >= 16:
            break
    return grouped


def value_history_summary(query: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    target_type_priority = value_history_target_type_priority(query)
    recency_priority = bool(re.search(r"\b(?:recent|current|currently|latest|now|updated|newest)\b", query.lower()))

    current_candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    sorted_rows = sorted(rows, key=value_history_sort_key(query))
    selected_row_relations: list[dict[str, object]] = []
    for row in sorted_rows:
        value = str(row.get("value") or "").strip()
        if not value:
            continue
        value_type = str(row.get("value_type") or "")
        context = str(row.get("context") or "")
        role = str(row.get("value_role") or value_history_value_role(query, context, value, value_type))
        key = (
            str(row.get("subject_key") or ""),
            value_type,
            value.lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        selected_row_relations.extend(
            item for item in (row.get("temporal_relations") or []) if isinstance(item, dict)
        )
        current_candidates.append(
            {
                "value": value,
                "value_type": row.get("value_type"),
                "subject_key": row.get("subject_key"),
                "current": bool(row.get("current")),
                "query_overlap": row.get("query_overlap"),
                "source_span_id": row.get("source_span_id"),
                "speaker": row.get("speaker"),
                "timeline_index": row.get("timeline_index"),
                "recency_rank": row.get("recency_rank"),
                "update_marker_strength": row.get("update_marker_strength"),
                "slot_overlap": row.get("slot_overlap"),
                "value_role": role,
                "context": compact_summary(str(row.get("context") or ""), 180),
                "temporal_relations": [item for item in (row.get("temporal_relations") or []) if isinstance(item, dict)],
            }
        )
        if len(current_candidates) >= 6:
            break
    if not current_candidates:
        return {}
    if recency_priority:
        guidance = (
            "For recent/current/latest-value questions, compare target-value candidates by recency first; "
            "prefer user-sourced newer rows over older or unrelated values."
        )
    else:
        guidance = (
            "For current/latest-value questions, first consider current_candidates with high query_overlap; "
            "prefer user-sourced newer rows over older or unrelated values."
        )
    return {
        "current_candidates": current_candidates,
        "preferred_current_candidate": current_candidates[0],
        "resolved_current_value": current_candidates[0].get("value"),
        **({"target_value_types": target_type_priority} if target_type_priority else {}),
        "recency_priority": recency_priority,
        "guidance": guidance,
        "temporal_relation_summary": temporal_relation_summary_from_safe_records(selected_row_relations),
    }


def exact_candidate_value_rows(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = value_summary_terms(query)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        content = str(candidate.get("content") or "")
        mentions = candidate.get("value_mentions") if isinstance(candidate.get("value_mentions"), list) else []
        if not mentions:
            continue
        content_terms = value_summary_terms(content)
        marker_strength = float(candidate.get("update_marker_strength") or 0.0)
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            value = str(mention.get("text") or "").strip()
            value_type = str(mention.get("type") or "").strip()
            if not value or not value_type:
                continue
            context = str(mention.get("context") or content)
            context_terms = value_summary_terms(context)
            value_role = value_history_value_role(query, context, value, value_type)
            rows.append(
                {
                    "source_span_id": candidate.get("source_span_id"),
                    "speaker": candidate.get("speaker"),
                    "timeline_index": candidate.get("timeline_index"),
                    "history_index": candidate.get("history_index"),
                    "recency_rank": index + 1,
                    "value_type": value_type,
                    "value": value,
                    "context": context,
                    "subject_key": "exact:" + str(candidate.get("source_span_id") or index),
                    "current": value_role_is_current(value_role)
                    or (
                        float(mention.get("update_marker_strength") or marker_strength or 0.0) > 0.0
                        and value_role not in {"previous", "target", "example"}
                    ),
                    "query_overlap": len(query_terms & context_terms),
                    "span_query_overlap": len(query_terms & content_terms),
                    "slot_overlap": len(value_topic_terms(query) & value_topic_terms(context)),
                    "value_role": value_role,
                    "update_marker_strength": float(mention.get("update_marker_strength") or marker_strength or 0.0),
                }
            )
    return rows


def value_mentions(content: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    occupied: list[tuple[int, int]] = []
    patterns = [
        ("money", r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?\b"),
        ("percentage", r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent(?:age)?(?:\s+points?)?)(?=\W|$)"),
        ("date", r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"),
        (
            "date",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
        ),
        ("date", r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日(?!\d)"),
        ("time", r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\b"),
        ("time", r"\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b"),
        (
            "count",
            r"\b\d+(?:,\d{3})*\s+of\s+\d+(?:,\d{3})*\s+"
            r"(?:tasks?|items?|modules?|tests?|problems?|books?|pages?|scenes?|cards?|columns?|features?)\b",
        ),
        (
            "count",
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*"
            r"(?:calls?|requests?|commits?|cards?|columns?|features?|concerns?|interviews?|"
            r"problems?|items?|tests?|modules?|roles?|mentees?|people|women|sources?|series|books?|pages?|words?|scenes?|"
            r"days?\s+a\s+week|days?\s+per\s+week)"
            r"(?:\s+per\s+(?:day|week|month|minute|hour))?\b",
        ),
        (
            "count",
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(?:calls?|requests?|commits?|cards?|columns?|features?|concerns?|interviews?|"
            r"problems?|items?|tests?|modules?|roles?|mentees?|people|women|sources?|series|books?|pages?|words?|scenes?|"
            r"days?\s+a\s+week|days?\s+per\s+week)"
            r"(?:\s+per\s+(?:day|week|month|minute|hour))?\b",
        ),
        (
            "duration",
            r"\b\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|hours?|minutes?)\b",
        ),
        ("duration", r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|hours?|minutes?)\b"),
        (
            "duration",
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(?:days?|weeks?|months?|hours?|minutes?)\b",
        ),
        ("latency", r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?)\b"),
        ("version", r"\bv?\d+\.\d+(?:\.\d+)?\b(?!\s*(?:%|percent))"),
    ]
    for kind, pattern in patterns:
        for match in re.finditer(pattern, content, flags=re.I):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            out.append(
                {
                    "type": kind,
                    "text": match.group(0),
                    "context": compact_summary(mention_context(content, match.start(), match.end()), 180),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            occupied.append((match.start(), match.end()))
            if len(out) >= 12:
                return out
    return out


def query_targeted_value_mentions(query: str, content: str) -> list[dict[str, Any]]:
    query_lower = query.lower()
    content_lower = content.lower()
    units: list[tuple[str, str]] = []
    if "slot_value" in value_history_target_type_priority(query):
        slot_rows = _query_targeted_slot_value_mentions(query, content)
        if slot_rows:
            return dedupe_value_mentions(slot_rows)
    if re.search(r"\bcommits?\b", query_lower):
        units.append(("commits", r"commits?"))
    if re.search(r"\b(?:women|mentees?)\b", query_lower):
        units.append(("women" if "women" in query_lower else "mentees", r"(?:women|mentees?|people)"))
    if re.search(r"\bsources?\b", query_lower):
        units.append(("sources", r"sources?"))
    if re.search(r"\bcupcakes?\b", query_lower):
        units.append(("cupcakes", r"cupcakes?"))
    if re.search(r"\bproblems?\b", query_lower):
        units.append(("problems", r"problems?"))
    if re.search(r"\bwords?\b|\bword\s+count\b", query_lower):
        units.append(("words", r"words?"))
    if re.search(r"\bcolumns?\b", query_lower):
        units.append(("columns", r"columns?"))
    if re.search(r"\bscenes?\b", query_lower):
        units.append(("scenes", r"scenes?"))
    if re.search(r"\bseries\b", query_lower):
        units.append(("series", r"series"))
    if not units:
        return []
    rows: list[dict[str, Any]] = []
    number = r"\d+(?:,\d{3})*(?:\.\d+)?"
    for label, unit_pattern in units:
        patterns = [
            rf"\b{unit_pattern}\b[^.?!]{{0,100}}\b(?:now\s+)?(?:has\s+|have\s+|had\s+)?(?:reached|increased|improved|rose|grown|changed|updated|adjusted|expanded)\s+(?:to\s+)?({number})\b",
            rf"\b(?:reached|increased|improved|rose|grown|changed|updated|adjusted|expanded)\s+(?:to\s+)?({number})\b[^.?!]{{0,80}}\b{unit_pattern}\b",
            rf"\b(?:now|currently|latest|new)\s+(?:at|is|are|stands?\s+at|sits?\s+at)\s+({number})\b[^.?!]{{0,80}}\b{unit_pattern}\b",
            rf"\b({number})\s+{unit_pattern}\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, content_lower, flags=re.I):
                value_number = match.group(1)
                value_text = f"{value_number} {label}"
                context = mention_context(content, match.start(), match.end())
                rows.append(
                    {
                        "type": "count",
                        "text": value_text,
                        "context": compact_summary(context, 180),
                        "start": match.start(),
                        "end": match.end(),
                        "update_marker_strength": value_update_marker_strength(query_lower, context.lower(), value_text),
                    }
                )
                if len(rows) >= 8:
                    return dedupe_value_mentions(rows)
    return dedupe_value_mentions(rows)


def _query_targeted_slot_value_mentions(query: str, content: str) -> list[dict[str, Any]]:
    if not _query_targets_named_slot(query.lower(), query):
        return []
    rows: list[dict[str, Any]] = []
    patterns = [
        r"\b(?:current|currently|latest|updated|new|preferred)?\s*(?:vector\s+database|database|model|provider|tool|engine|framework|library|reranker|embedding)\s+(?:is|are|should\s+be|uses?|using|changed\s+to|switched\s+to|migrated\s+to)\s+([A-Za-z][A-Za-z0-9_.+-]{1,50})\b",
        r"\b(?:changed|switched|migrated|updated)\s+(?:the\s+)?(?:vector\s+database|database|model|provider|tool|engine|framework|library|reranker|embedding)\s+(?:to|from\s+[A-Za-z][A-Za-z0-9_.+-]{1,50}\s+to)\s+([A-Za-z][A-Za-z0-9_.+-]{1,50})\b",
        r"(?:当前|现在|最新|改成|改为|更新为|切换为|使用的?)(?:[^。！？\n]{0,24})(?:向量库|数据库|模型|工具|框架|引擎|服务商)(?:[^。！？\n]{0,12})(?:是|为|成|使用)?\s*([A-Za-z][A-Za-z0-9_.+-]{1,50})",
        r"(?:向量库|数据库|模型|工具|框架|引擎|服务商)(?:[^。！？\n]{0,12})(?:改成|改为|更新为|切换为|是|为|使用)\s*([A-Za-z][A-Za-z0-9_.+-]{1,50})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, content, flags=re.I):
            value = str(match.group(1) or "").strip().strip(".,;:，。；：")
            if not value or value.lower() in {"not", "the", "current", "now", "new", "updated"}:
                continue
            context = mention_context(content, match.start(1), match.end(1))
            rows.append(
                {
                    "type": "slot_value",
                    "text": value,
                    "context": compact_summary(context, 220),
                    "start": match.start(1),
                    "end": match.end(1),
                    "update_marker_strength": value_update_marker_strength(query.lower(), context.lower(), value),
                }
            )
            if len(rows) >= 8:
                return rows
    return rows


def dedupe_value_mentions(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (str(value.get("type") or ""), str(value.get("text") or "").lower())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= 16:
            break
    return out


def value_summary_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "what",
        "how",
        "many",
        "much",
        "have",
        "been",
        "into",
        "from",
        "your",
        "my",
        "you",
        "are",
        "was",
        "were",
    }
    terms: set[str] = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?", text.lower()):
        token = token.strip("_+-")
        if len(token) < 3 or token in stopwords:
            continue
        terms.add(token)
        if token.endswith("s") and len(token) > 4:
            terms.add(token[:-1])
        if token.endswith("ing") and len(token) > 6:
            terms.add(token[:-3])
        if token.endswith("ed") and len(token) > 5:
            terms.add(token[:-2])
    return terms


def value_history_sort_key(query: str):
    lower = query.lower()
    target_type_priority = value_history_target_type_priority(query)
    target_rank = {value_type: index for index, value_type in enumerate(target_type_priority)}
    recency_priority = bool(re.search(r"\b(?:recent|current|currently|latest|now|updated|newest)\b", lower))

    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        value_type = str(item.get("value_type") or "")
        value = str(item.get("value") or "")
        context = str(item.get("context") or "")
        role = str(item.get("value_role") or value_history_value_role(query, context, value, value_type))
        type_rank = target_rank.get(value_type, len(target_rank) if target_rank else 0)
        unit_rank = value_history_unit_mismatch_rank(query, value, value_type)
        context_rank = value_history_context_mismatch_rank(query, context, value_type, value=value)
        slot_value_rank = value_history_value_slot_mismatch_rank(query, context, value_type, value=value)
        topic_rank = value_history_topic_mismatch_rank(query, context, value_type)
        role_rank = value_history_role_rank(query, role)
        current_rank = value_history_current_rank(query, item, role)
        slot_rank = -int(item.get("slot_overlap") or 0)
        local_overlap_rank = -int(item.get("query_overlap") or 0)
        common = (
            type_rank,
            unit_rank,
            context_rank,
            slot_value_rank,
            topic_rank,
            role_rank,
            slot_rank,
            local_overlap_rank,
            current_rank,
            value_history_source_rank(query, item),
            -float(item.get("update_marker_strength") or 0.0),
        )
        recency_rank = int(item.get("recency_rank") or 10**9)
        timeline_rank = -int(item.get("timeline_index") or item.get("history_index") or -1)
        if recency_priority:
            return common + (
                recency_rank,
                -int(item.get("query_overlap") or 0),
                0 if item.get("speaker") == "user" else 1,
                timeline_rank,
                str(item.get("subject_key") or ""),
                str(item.get("value") or ""),
            )
        return common + (
            -int(item.get("query_overlap") or 0),
            0 if item.get("speaker") == "user" else 1,
            recency_rank,
            timeline_rank,
            str(item.get("subject_key") or ""),
            str(item.get("value") or ""),
        )

    return key


def value_history_value_role(query: str, context: str, value_text: str, value_type: str) -> str:
    lower = context.lower()
    value = re.escape(value_text.lower().strip())
    value_window = lower
    if value:
        match = re.search(value, lower)
        if match:
            value_window = lower[max(0, match.start() - 90) : match.end() + 90]
    if value_context_is_example(value_window):
        return "example"
    if value and (
        re.search(rf"\b(?:previously|before|baseline|originally|initially|former|old|last year)\b[^.?!]{{0,100}}{value}", value_window)
        or re.search(rf"{value}[^.?!]{{0,100}}\b(?:previously|before|baseline|originally|initially|former|old|last year)\b", value_window)
        or re.search(rf"\b(?:from|was|were)\s+{value}\b", value_window)
    ):
        return "previous"
    if value and re.search(
        rf"\b(?:new|updated|adjusted|revised|increased|raised|changed)\b[^.?!]{{0,100}}{value}",
        value_window,
    ):
        return "current"
    if value_type == "slot_value" and value and re.search(
        rf"\b(?:current|currently|latest|new|updated|preferred|should\s+be|is|are|changed\s+to|switched\s+to|migrated\s+to)\b[^.?!]{{0,120}}{value}",
        value_window,
    ):
        return "current"
    if value_type == "slot_value" and re.search(r"(?:当前|现在|最新|改成|改为|更新为|切换为|不再用|不再是)", context):
        return "current"
    if value and re.search(
        rf"{value}[^.?!]{{0,100}}\b(?:new|updated|adjusted|revised|increase|quota|budget|target)\b",
        value_window,
    ):
        return "current"
    if value and re.search(
        rf"\b(?:update|adjust|revise|change)\b[^.?!]{{0,120}}\b(?:to\s+reflect|reflect|to)\b[^.?!]{{0,80}}{value}",
        value_window,
    ):
        return "current"
    if value and re.search(
        rf"\b(?:allocated|budgeted|set|agreed|confirmed)\b[^.?!]{{0,100}}{value}",
        value_window,
    ):
        return "current"
    if value and re.search(
        rf"{value}[^.?!]{{0,100}}\b(?:allocated|budgeted|set|agreed|confirmed|current budget allocation)\b",
        value_window,
    ):
        return "current"
    if _query_asks_target_value(query) and re.search(
        r"\b(?:aim(?:ing)?|goal|target|deadline|due|need|needs|finish|complete|submit|by)\b",
        value_window,
    ):
        return "target"
    if value_context_is_target_goal(context, value_text):
        return "target"
    if value and re.search(
        rf"\b(?:now|current(?:ly)?|latest|new|updated|revised|rescheduled|moved|scheduled|set|confirmed|correct|rated|achieved|reached|improved|increased|decreased|reduced)\b[^.?!]{{0,120}}{value}",
        value_window,
    ):
        return "current"
    if value and re.search(
        rf"{value}[^.?!]{{0,120}}\b(?:now|current(?:ly)?|latest|new|updated|revised|rescheduled|scheduled|confirmed|rated|accuracy|deadline|due)\b",
        value_window,
    ):
        return "current"
    if value_type == "date" and re.search(r"\b(?:deadline|due|scheduled|date|takes?\s+place|on)\b", value_window):
        return "current"
    if value_type in {"percentage", "latency", "duration", "count", "money", "version"} and re.search(
        r"\b(?:accuracy|rate|current|now|total|stands?|sits?|is|are|has|have|achieved|reached)\b",
        value_window,
    ):
        return "current"
    return "mentioned"


def value_context_is_example(context: str) -> bool:
    lower = context.lower()
    return bool(
        re.search(r"\b(?:for example|example|sample|hypothetical|would|could|might|try|template|placeholder)\b", lower)
    )


def _query_asks_target_value(query: str) -> bool:
    return bool(re.search(r"\b(?:aim(?:ing)?|goal|target|deadline|due|by what|by when|need to|expected to|supposed to)\b", query.lower()))


def value_role_is_current(role: str) -> bool:
    return role == "current"


def value_history_role_rank(query: str, role: str) -> int:
    lower = query.lower()
    if role == "target":
        return 0 if re.search(r"\b(?:aim(?:ing)?|goal|target|deadline|due|by what date|when)\b", lower) else 2
    if role == "current":
        return 1 if _query_asks_target_value(query) else 0
    if role == "mentioned":
        return 2 if _query_asks_target_value(query) else 1
    if role == "previous":
        return 3
    if role == "example":
        return 4
    return 2


def value_history_current_rank(query: str, item: dict[str, Any], role: str) -> int:
    if _query_asks_target_value(query) and role == "target":
        return 0
    return 0 if item.get("current") or value_role_is_current(role) else 1


def value_history_source_rank(query: str, item: dict[str, Any]) -> int:
    lower = query.lower()
    speaker = str(item.get("speaker") or "").lower()
    context = str(item.get("context") or "").lower()
    if re.search(r"\b(?:recent|current|currently|latest|now|updated|newest|tracked|recorded)\b", lower):
        if speaker in {"user", "document", "fact"}:
            return 0
        if speaker in {"assistant", "agent"}:
            if re.search(r"\b(?:example|hypothetical|would|should|could|goal|target|aim|plan|week\s+\d+)\b", context):
                return 3
            return 1
    return 0 if speaker in {"user", "document", "fact"} else 1


def value_history_unit_mismatch_rank(query: str, value: str, value_type: str) -> int:
    lower = query.lower()
    value_lower = value.lower()
    if value_type == "money":
        return 0 if re.search(r"\$|usd|dollars?", value_lower) else 1
    if value_type == "count":
        unit_groups = [
            (r"\b(?:quota|calls?|requests?|per\s+day|daily)\b", r"\b(?:calls?|requests?)\s+per\s+day\b"),
            (r"\bword(?:\s+count)?s?\b", r"\bwords?\b"),
            (r"\bcommits?\b", r"\bcommits?\b"),
            (r"\bwomen\b", r"\bwomen\b"),
            (r"\bmentees?\b", r"\b(?:mentees?|women|people)\b"),
            (r"\bsources?\b", r"\bsources?\b"),
            (r"\bbooks?\b", r"\bbooks?\b"),
            (r"\bcupcakes?\b", r"\bcupcakes?\b"),
            (r"\bseries\b", r"\bseries\b"),
            (r"\bpages?\b", r"\bpages?\b"),
            (r"\bcards?\b", r"\bcards?\b"),
            (r"\bitems?\b", r"\bitems?\b"),
            (r"\bcolumns?\b", r"\bcolumns?\b"),
            (r"\bproblems?\b", r"\bproblems?\b"),
            (r"\binterviews?\b", r"\binterviews?\b"),
            (r"\bscenes?\b", r"\bscenes?\b"),
            (r"\bdays?\s+(?:a|per)\s+week\b", r"\bdays?\s+(?:a|per)\s+week\b"),
        ]
        for query_pattern, value_pattern in unit_groups:
            if re.search(query_pattern, lower):
                return 0 if re.search(value_pattern, value_lower) else 1
        if re.search(r"\bhow many\b", lower):
            return 0
        return 0
    if value_type != "duration":
        return 0
    unit_groups = [
        (r"\bhours?\b", r"\bhours?\b"),
        (r"\bminutes?\b", r"\bminutes?\b"),
        (r"\bweeks?\b", r"\bweeks?\b"),
        (r"\bmonths?\b", r"\bmonths?\b"),
        (r"\bdays?\b", r"\bdays?\b"),
    ]
    for query_pattern, value_pattern in unit_groups:
        if re.search(query_pattern, lower):
            return 0 if re.search(value_pattern, value_lower) else 1
    return 0


def value_history_context_mismatch_rank(query: str, context: str, value_type: str, *, value: str = "") -> int:
    lower = query.lower()
    context_lower = context.lower()
    rank = 0
    if value_type == "count" and re.search(r"\bweekly\b|\bper\s+week\b|\bweek\b", lower) and re.search(
        r"\b(?:daily|per\s+day|each\s+day|every\s+day)\b",
        context_lower,
    ):
        rank = max(rank, 1)
    if value_type == "count" and re.search(r"\bdaily\b|\bper\s+day\b", lower) and re.search(r"\bweekly\b|\bper\s+week\b", context_lower):
        rank = max(rank, 1)
    if re.search(r"\b(?:recent|current|currently|latest|now|updated|newest|tracked|recorded)\b", lower):
        escaped = re.escape(value.lower().strip())
        value_near_historical_baseline = bool(
            escaped
            and (
                re.search(rf"\b(?:given that|previously|before|baseline|originally|initially)\b[^.?!]{{0,100}}{escaped}", context_lower)
                or re.search(rf"{escaped}[^.?!]{{0,100}}\b(?:previously|before|baseline|originally|initially)\b", context_lower)
            )
        )
        if value_near_historical_baseline:
            rank = max(rank, 2)
        elif re.search(r"\b(?:example|hypothetical|would|should|could|week\s+\d+)\b", context_lower) and not re.search(
            r"\b(?:now|current|currently|latest|recently|managed|reached|tracked|recorded|updated)\b",
            context_lower,
        ):
            rank = max(rank, 1)
    return rank


def value_history_value_slot_mismatch_rank(query: str, context: str, value_type: str, *, value: str = "") -> int:
    if value_type != "percentage":
        return 0
    lower = query.lower()
    if not re.search(r"\b(?:accuracy|rate|evaluation|evaluations|coverage)\b", lower):
        return 0
    context_lower = context.lower()
    escaped = re.escape(value.lower().strip())
    clause = context_lower
    if escaped:
        match = re.search(escaped, context_lower)
        if match:
            clause = context_lower[max(0, match.start() - 90) : match.end() + 90]
    if re.search(r"\b(?:accuracy|accurate|evaluation|evaluations|matching|match\s+rate|success\s+rate|coverage)\b", clause):
        return 0
    if re.search(r"\b(?:screening\s+time|hiring\s+time|time\s+reduction|reduced\s+screening|faster)\b", clause):
        return 2
    return 1


def value_context_is_target_goal(context: str, value_text: str) -> bool:
    lower = context.lower()
    value = re.escape(value_text.lower())
    if value and re.search(rf"\b(?:reached|achieved|currently|now|is|was)\b[^.?!]{{0,60}}{value}", lower):
        return False
    if value and re.search(rf"\b(?:deadline|due|scheduled|confirmed|correct|set)\b[^.?!]{{0,80}}{value}", lower):
        return False
    if value and re.search(rf"{value}[^.?!]{{0,80}}\b(?:deadline|due|scheduled|confirmed|correct)\b", lower):
        return False
    goal_patterns = [
        r"\b(?:trying|aiming|hoping|planning|want|wanted|need|goal|target)\b[^.?!]{0,80}" + value,
        value + r"[^.?!]{0,80}\b(?:goal|target|aim)\b",
        r"\bto\s+(?:reach|achieve|get to|increase to|improve to)\b[^.?!]{0,60}" + value,
    ]
    return any(re.search(pattern, lower) for pattern in goal_patterns)


def value_history_topic_mismatch_rank(query: str, context: str, value_type: str) -> int:
    if value_type not in {"money", "count", "percentage", "latency", "duration", "date", "version"}:
        return 0
    query_terms = value_topic_terms(query)
    if not query_terms:
        return 0
    context_terms = value_summary_terms(context)
    overlap = len(query_terms & context_terms)
    if value_type == "date" and re.search(r"\b(?:when|date|scheduled|deadline|due|takes?\s+place)\b", query.lower()):
        anchors = value_history_date_anchor_terms(query)
        anchor_overlap = len(anchors & context_terms)
        if anchors and anchor_overlap == 0:
            return 3
        if len(anchors) >= 3 and anchor_overlap < 2:
            return 2
    required = 2 if len(query_terms) >= 4 else 1
    if overlap >= required:
        return 0
    if overlap:
        return 1
    return 2


def value_history_date_anchor_terms(query: str) -> set[str]:
    generic = {
        "aim",
        "aiming",
        "complete",
        "completed",
        "deadline",
        "date",
        "due",
        "place",
        "scheduled",
        "take",
        "takes",
        "time",
        "when",
    }
    return {term for term in value_topic_terms(query) if term not in generic}


def value_topic_terms(text: str) -> set[str]:
    generic = {
        "budget",
        "amount",
        "total",
        "year",
        "current",
        "currently",
        "latest",
        "value",
        "count",
        "number",
        "many",
        "much",
        "percentage",
        "percent",
        "rate",
        "target",
        "goal",
    }
    return {term for term in value_summary_terms(text) if term not in generic}


def value_history_target_type_priority(query: str) -> list[str]:
    lower = query.lower()
    out: list[str] = []
    if re.search(r"\b(?:what\s+time|which\s+time|at\s+what\s+time)\b", lower):
        out.append("time")
    if re.search(r"\bhow\s+many\s+(?:days?|weeks?|months?|hours?|minutes?)\b", lower):
        out.append("duration")
    if (
        re.search(r"\b(?:hours?|minutes?|weeks?|months?)\b", lower)
        or re.search(r"\bhow\s+long\b", lower)
        or re.search(r"\b(?:duration|timeline|time\s+required|takes?|take)\b", lower)
    ) and not re.search(r"\b(?:days?\s+a\s+week|days?\s+per\s+week)\b", lower):
        out.append("duration")
    if re.search(r"\b(?:how many|count|number of|quota|requests?|sources?|per\s+day|per\s+week|days?\s+a\s+week)\b", lower) and not re.search(
        r"\bhow\s+many\s+(?:days?|weeks?|months?|hours?|minutes?)\b",
        lower,
    ):
        out.append("count")
    if re.search(r"\b(?:budget|amount|cost|costs|fee|fees|funds?|allocation|allocated|spending|money|\$)\b", lower):
        out.append("money")
    if re.search(r"\b(?:percentage|percent|coverage|accuracy|rate|%)\b", lower):
        out.append("percentage")
    if re.search(r"\b(?:when|date|deadline|due|scheduled)\b", lower) or re.search(
        r"(?:日期|时间|截止|发布目标|目标日|目标时间|什么时候)",
        query,
    ):
        out.append("date")
    if re.search(r"\b(?:version|library|libraries|dependencies?|package)\b", lower):
        out.append("version")
    if re.search(r"\b(?:latency|response\s+time|ms|seconds?)\b", lower):
        out.append("latency")
    if _query_targets_named_slot(lower, query):
        out.append("slot_value")
    return list(dict.fromkeys(out))


def _query_targets_named_slot(lower: str, query: str) -> bool:
    has_current_intent = bool(
        re.search(r"\b(?:current|currently|latest|updated|new|preferred|using|use|what\s+is|which)\b", lower)
        or re.search(r"(?:当前|现在|最新|使用|用的|是什么|哪一个|哪个)", query)
    )
    if not has_current_intent:
        return False
    return bool(
        re.search(
            r"\b(?:vector\s+database|database|model|provider|tool|engine|framework|library|reranker|embedding)\b",
            lower,
        )
        or re.search(r"(?:向量库|数据库|模型|工具|框架|引擎|服务商)", query)
    )


def value_update_marker_strength(query_lower: str, text_lower: str, value_text: str) -> float:
    value = re.escape(value_text.lower().strip())
    scoped_text = text_lower
    if value:
        match = re.search(value, text_lower)
        if match:
            scoped_text = text_lower[max(0, match.start() - 100) : match.end() + 100]
    score = 0.0
    strong_patterns = [
        r"\b(?:recently|now|latest|newly)\b[^.?!]{0,80}\b(?:improved|increased|decreased|reduced|reached|updated|adjusted|revised|rescheduled|raised|lowered)\b",
        r"\b(?:has|have|had)?\s*(?:now|recently)\s+(?:reached|increased|improved|decreased|reduced|changed|grown)\b",
        r"\b(?:increased|improved|reduced|decreased|updated|adjusted|revised|rescheduled|raised|lowered)\s+(?:to|at|by)\b",
        r"\b(?:has|have)\s+(?:increased|improved|reduced|decreased|changed|grown)\s+to\b",
    ]
    medium_patterns = [
        r"\b(?:currently|current)\b[^.?!]{0,80}\b(?:reached|at|is|are|stands?|sits?)\b",
        r"\b(?:managed|able)\s+to\s+(?:reduce|increase|improve|reach|get)\b",
        r"\b(?:capped|set|allocated|budgeted|scheduled)\b[^.?!]{0,80}\b(?:at|to|for)\b",
    ]
    if any(re.search(pattern, scoped_text) for pattern in strong_patterns):
        score = max(score, 2.0)
    if any(re.search(pattern, scoped_text) for pattern in medium_patterns):
        score = max(score, 1.0)
    if re.search(r"\b(?:initially|originally|previously|before|old|older)\b", scoped_text):
        score -= 0.6
    if value and re.search(rf"\b(?:initial|original|previous|old)\b[^.?!]{{0,80}}{value}", scoped_text):
        score -= 0.8
    if re.search(r"\b(?:current|currently|latest|now|updated|newest|recently)\b", query_lower):
        score += 0.2 if score > 0 else 0.0
    return max(-1.0, min(score, 2.4))


def value_history_subject_key(query: str, text: str) -> str:
    query_terms = value_summary_terms(query)
    text_terms = value_summary_terms(text)
    shared = sorted(query_terms & text_terms)
    if shared:
        return "subject:" + "_".join(shared[:4])
    important = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in TEMPORAL_STOPWORDS
    ]
    return "subject:" + "_".join(important[:4]) if important else "subject:unknown"


def mention_context(content: str, start: int, end: int, *, radius: int = 140) -> str:
    left = max(0, start - radius)
    right = min(len(content), end + radius)
    label_start = _nearest_label_start(content, start)
    if label_start is not None and label_start >= left:
        left = label_start
    boundary_start = max(content.rfind("\n", 0, start), content.rfind(". ", 0, start), content.rfind("; ", 0, start))
    if boundary_start >= left and start - boundary_start <= 120:
        left = boundary_start + (2 if content[boundary_start : boundary_start + 2] in {". ", "; "} else 1)
    label_end = _next_label_start(content, end)
    if label_end is not None and label_end <= right and label_end > end:
        right = label_end
    boundary_candidates = [
        pos
        for pos in [content.find("\n", end), content.find(". ", end), content.find("; ", end)]
        if pos != -1 and pos <= right
    ]
    if boundary_candidates:
        right = min(boundary_candidates) + 1
    return content[left:right].strip()


def _nearest_label_start(content: str, position: int) -> int | None:
    label_pattern = re.compile(r"(?:^|[\n.;]\s*)(?:[-*]\s*)?(?:\*\*)?[A-Z][A-Za-z0-9 /&-]{2,80}(?:\*\*)?\s*:")
    best: int | None = None
    for match in label_pattern.finditer(content[:position]):
        label_start = match.start()
        if match.group(0).startswith(("\n", ".", ";")):
            label_start += 1
        if position - label_start <= 180:
            best = label_start
    return best


def _next_label_start(content: str, position: int) -> int | None:
    label_pattern = re.compile(r"(?:^|[\n.;]\s*)(?:[-*]\s*)?(?:\*\*)?[A-Z][A-Za-z0-9 /&-]{2,80}(?:\*\*)?\s*:")
    for match in label_pattern.finditer(content, position):
        label_start = match.start()
        if match.group(0).startswith(("\n", ".", ";")):
            label_start += 1
        return label_start
    return None
