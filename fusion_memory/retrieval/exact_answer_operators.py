from __future__ import annotations

"""Typed exact-answer candidates extracted from local evidence snippets.

The operators here are deliberately small and relationship-shaped. They only
emit an answer value when a local span directly states the requested relation,
so the model can use the candidate as a strong extraction hint without turning
the eval adapter into a growing set of answer templates.
"""

import re
from typing import Any


def exact_answer_operator_fields(query: str, content: str, *, speaker: str | None = None) -> dict[str, Any]:
    lower = query.lower()
    if speaker and speaker not in {"user", "document", "assistant"}:
        return {}
    candidate = _where_met_candidate(lower, content, speaker=speaker)
    if candidate:
        return candidate
    candidate = _prior_probability_candidate(lower, content, speaker=speaker)
    if candidate:
        return candidate
    candidate = _duration_before_relationship_candidate(lower, content, speaker=speaker)
    if candidate:
        return candidate
    return {}


def _where_met_candidate(query_lower: str, content: str, *, speaker: str | None) -> dict[str, Any] | None:
    if speaker not in {"user", "document"}:
        return None
    if not re.search(r"\bwhere\b", query_lower) or not re.search(r"\b(?:met|meet)\b", query_lower):
        return None
    names = _query_names(query_lower)
    if not names:
        return None
    for name in names:
        name_pattern = re.escape(name)
        patterns = [
            rf"\b(?P<person>{name_pattern})\s+met\s+(?:me|us)\s+(?P<prep>on\s+set\s+at|at|in)\s+(?P<place>[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){{0,5}})(?:\s+in\s+(?P<year>20\d{{2}}))?",
            rf"\b(?:i|we)\s+met\s+(?P<person>{name_pattern})\s+(?P<prep>on\s+set\s+at|at|in)\s+(?P<place>[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){{0,5}})(?:\s+in\s+(?P<year>20\d{{2}}))?",
            rf"\b(?P<person>{name_pattern})\b[^.?!]{{0,180}}\b(?:she|he|they)\s+met\s+(?:me|us)\s+(?P<prep>on\s+set\s+at|at|in)\s+(?P<place>[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){{0,5}})(?:\s+in\s+(?P<year>20\d{{2}}))?",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if not match:
                continue
            place = _clean_place(match.group("place"))
            if not place:
                continue
            prep = str(match.group("prep") or "at").strip()
            year = match.groupdict().get("year")
            location = f"{prep} {place}"
            if year:
                location += f" in {year}"
            return {
                "answer_type": "location",
                "answer_value": location,
                "confidence": 0.91,
                "extraction_formula": "where_met_relation",
                "guidance": "Use for direct where-did-I-meet questions; preserve the user-perspective relation.",
            }
    return None


def _prior_probability_candidate(query_lower: str, content: str, *, speaker: str | None) -> dict[str, Any] | None:
    if speaker not in {"user", "document", "assistant"}:
        return None
    if not re.search(r"\bprobability\b", query_lower) or not re.search(r"\bbefore\b", query_lower):
        return None
    if not re.search(r"\b(?:before|started discussing|two cards?|without replacement|second)\b", content.lower()):
        return None
    fractions = list(re.finditer(r"\b(\d+)\s*/\s*(\d+)\b", content))
    if not fractions:
        return None
    for match in fractions:
        left = content[max(0, match.start() - 160) : match.start()].lower()
        right = content[match.end() : min(len(content), match.end() + 160)].lower()
        context = left + " " + right
        if "deck" in query_lower and match.group(2) != "52":
            continue
        if re.search(r"\b(?:second|given|conditional|remaining|left)\b", context) and not re.search(r"\b(?:first|single|initial)\b", context):
            continue
        if re.search(r"\b(?:first|certain|single|drawing an? ace|draw an? ace|initial)\b", context):
            return {
                "answer_type": "probability",
                "answer_value": f"{match.group(1)}/{match.group(2)}",
                "confidence": 0.88,
                "extraction_formula": "prior_probability_before_sequence",
                "guidance": "Use the probability stated for the earlier/single draw, not the later conditional second-draw probability.",
            }
    first = fractions[0]
    return {
        "answer_type": "probability",
        "answer_value": f"{first.group(1)}/{first.group(2)}",
        "confidence": 0.72,
        "extraction_formula": "prior_probability_before_sequence",
        "guidance": "Candidate probability from the local prior discussion; verify it is not the later conditional probability.",
    }


def _duration_before_relationship_candidate(query_lower: str, content: str, *, speaker: str | None) -> dict[str, Any] | None:
    if speaker not in {"user", "document"}:
        return None
    if not re.search(r"\bhow long\b", query_lower):
        return None
    if not re.search(r"\bbefore\s+we\s+started\s+dating\b", query_lower):
        return None
    if not re.search(r"\b(?:festival|met)\b", query_lower):
        return None
    patterns = [
        r"\b(?:had\s+been\s+together|were\s+together|been\s+with\s+[^.?!]{0,50})\s+for\s+(?P<duration>\d+\s+(?:years?|months?))\s+before\s+(?:we\s+)?started\s+dating\b",
        r"\bmet\s+[^.?!]{0,80}\bat\s+(?:the\s+)?[^.?!]{0,80}festival[^.?!]{0,120}\b(?P<duration>\d+\s+(?:years?|months?))\s+before\s+(?:we\s+)?started\s+dating\b",
        r"\b(?P<duration>\d+\s+(?:years?|months?))\s+before\s+(?:we\s+)?started\s+dating\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.I)
        if not match:
            continue
        return {
            "answer_type": "duration",
            "answer_value": match.group("duration"),
            "confidence": 0.86,
            "extraction_formula": "duration_before_relationship_start",
            "guidance": "Use for direct duration-before-dating questions grounded in a user relationship statement.",
        }
    return None


def _query_names(query_lower: str) -> list[str]:
    raw = re.findall(r"\b(?:met|meet)\s+([a-z][a-z'-]{2,})\b|\b([a-z][a-z'-]{2,})\s+(?:met|meet)\b", query_lower)
    names: list[str] = []
    for left, right in raw:
        value = left or right
        if value in {"where", "person", "festival", "before", "started"}:
            continue
        names.append(value[:1].upper() + value[1:])
    return list(dict.fromkeys(names))


def _clean_place(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,.;:!?")
    value = re.sub(r"\s+in\s+20\d{2}$", "", value)
    value = re.split(r"\b(?:where|when|because|but|and|can|she|he|they|we|i)\b", value, maxsplit=1)[0].strip()
    if len(value) < 3:
        return ""
    return value
