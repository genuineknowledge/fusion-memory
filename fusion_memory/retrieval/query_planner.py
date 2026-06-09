from __future__ import annotations

import re

from fusion_memory.core.models import QueryPlan
from fusion_memory.core.text import extract_entities


class QueryPlanner:
    def plan(self, query: str) -> QueryPlan:
        lower = query.lower()
        query_type = "factual_exact"
        needs_current = False
        must_include = ["raw_evidence"]
        if any(w in lower for w in ["current", "currently", "now", "现在", "当前", "以后按"]):
            query_type = "preference" if any(w in lower for w in ["prefer", "preference", "喜欢", "偏好", "用什么"]) else "instruction"
            needs_current = True
            must_include = ["current_view", "raw_evidence"]
        if any(
            w in lower
            for w in [
                "when",
                "first",
                "before",
                "after",
                "previously",
                "yesterday",
                "today",
                "tomorrow",
                "last week",
                "this week",
                "next week",
                "last month",
                "this month",
                "next month",
                "this monday",
                "this tuesday",
                "this wednesday",
                "this thursday",
                "this friday",
                "this saturday",
                "this sunday",
                "next monday",
                "next tuesday",
                "next wednesday",
                "next thursday",
                "next friday",
                "next saturday",
                "next sunday",
                "什么时候",
                "之前",
                "之后",
                "先",
            ]
        ):
            query_type = "event_ordering" if any(w in lower for w in ["before", "after", "先", "顺序"]) else "temporal_lookup"
            must_include = ["raw_evidence", "events"]
        if any(w in lower for w in ["contradict", "conflict", "changed", "switched", "矛盾", "冲突", "改"]):
            query_type = "contradiction_resolution"
            must_include = ["raw_evidence", "facts"]
        if any(w in lower for w in ["unknown", "not mentioned", "没有提到", "不知道", "cluster name"]):
            query_type = "abstention"
            must_include = ["raw_evidence"]
        if any(w in lower for w in ["summarize", "summary", "总结"]):
            query_type = "summarization"
            must_include = ["raw_evidence"]
        speaker_focus = "any"
        if any(w in lower for w in ["you suggested", "assistant", "你建议", "上次你"]):
            speaker_focus = "assistant"
            query_type = "assistant_reference"
        return QueryPlan(
            query=query,
            query_type=query_type,
            entities=extract_entities(query),
            time_constraints=_time_constraints(query),
            speaker_focus=speaker_focus,
            needs_current_state=needs_current,
            needs_source_evidence=True,
            must_include_sources=must_include,
        )


def _time_constraints(query: str) -> list[dict]:
    lower = query.lower()
    out: list[dict] = []
    for phrase in [
        "last week",
        "this week",
        "next week",
        "last month",
        "this month",
        "next month",
        "yesterday",
        "today",
        "tomorrow",
        "this monday",
        "this tuesday",
        "this wednesday",
        "this thursday",
        "this friday",
        "this saturday",
        "this sunday",
        "next monday",
        "next tuesday",
        "next wednesday",
        "next thursday",
        "next friday",
        "next saturday",
        "next sunday",
    ]:
        if phrase in lower:
            out.append({"type": "relative", "text": phrase})
    explicit = re.findall(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", query)
    out.extend({"type": "explicit", "text": item} for item in explicit)
    month_dates = re.findall(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
        query,
        flags=re.I,
    )
    out.extend({"type": "explicit", "text": item} for item in month_dates)
    return out
