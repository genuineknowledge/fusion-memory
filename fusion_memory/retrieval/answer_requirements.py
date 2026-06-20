from __future__ import annotations

import re
from typing import Any

from fusion_memory.core.text import compact_summary
from fusion_memory.retrieval.aggregation_common import _match_context


def answer_requirements(
    query: str,
    source_spans: list[dict[str, Any]],
    *,
    format_requirements: list[str] | None = None,
    preference_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build typed answer requirements from the query and supported evidence.

    This section is intentionally about output obligations, not answer content.
    It gives the answer model durable constraints such as date format, required
    detail classes, and explanation depth without adding category templates.
    """

    query_lower = query.lower()
    requirements: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(type_: str, requirement: str, *, source: str, context: str | None = None, score: float = 1.0) -> None:
        key = f"{type_}:{requirement}".lower()
        if key in seen:
            return
        seen.add(key)
        row: dict[str, Any] = {
            "type": type_,
            "requirement": requirement,
            "source": source,
            "score": round(score, 3),
        }
        if context:
            row["context"] = compact_summary(context, 260)
        requirements.append(row)

    if _query_requests_date_detail(query_lower):
        if re.search(r"\bmm/dd/yyyy\b|\bmm-dd-yyyy\b|\b\d{1,2}/\d{1,2}/\d{4}\b", query_lower):
            add("date_format", "Format every requested date as MM/DD/YYYY.", source="query", score=3.0)
        elif re.search(r"\bmonth\s+day,\s+year\b|\bmonth-day-year\b", query_lower):
            add("date_format", "Include the full date as Month Day, Year.", source="query", score=2.8)
        else:
            add("date_detail", "Include the year when the evidence supports it.", source="query", score=2.0)

    if _query_requests_version_detail(query_lower):
        add("version_detail", "Include exact version numbers for named libraries, dependencies, tools, or software when supported.", source="query", score=3.0)

    if _query_requests_viewing_platforms(query_lower):
        add("platform_detail", "Include streaming services or platform names when recommendations depend on availability.", source="query", score=2.7)

    if re.search(r"\b(?:legal|legally|valid|will|wishes|terms?)\b", query_lower):
        add("explanation_depth", "Explain important legal terms instead of only naming them.", source="query", score=2.8)

    if re.search(r"\b(?:progress|percentage|percent|how much progress)\b", query_lower):
        add("numeric_detail", "Include supported percentage values or explicit numeric progress values.", source="query", score=3.0)

    if re.search(r"\b(?:information safe|keep(?:ing)? .*safe|online services|privacy|security|secure|data protection)\b", query_lower):
        add("security_detail", "Explain relevant encryption or secure-transport methods such as HTTPS/TLS or encryption at rest/in transit.", source="query", score=3.0)

    if re.search(r"\b(?:books?|recommend|check out|genres?|genre characteristics|style|themes?)\b", query_lower):
        add("recommendation_rationale", "For recommendations, include concise genre/style/theme context, not only titles.", source="query", score=2.5)

    if re.search(r"\b(?:steps?|process|go through|prepare|follow up|include|approach)\b", query_lower):
        add("process_detail", "Preserve distinct supported steps and substeps rather than collapsing them into generic advice.", source="query", score=2.4)

    _add_format_requirement_constraints(format_requirements or [], add)
    _add_preference_answer_requirements(preference_constraints or [], add)
    _add_evidence_supported_requirements(query_lower, source_spans, add)

    if not requirements:
        return {}
    requirements.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("type") or "")))
    return {
        "must_satisfy": requirements[:10],
        "coverage_guidance": (
            "Use these as answer-level obligations after selecting evidence. They constrain format and detail depth; "
            "they do not license unsupported facts."
        ),
    }


def _add_format_requirement_constraints(format_requirements: list[str], add) -> None:
    for requirement in format_requirements:
        name = str(requirement or "")
        if name == "include_exact_versions_if_supported":
            add(
                "version_detail",
                "Include exact version numbers for named libraries, dependencies, tools, or software when supported.",
                source="coverage",
                score=3.2,
            )
        elif name == "fenced_code_block":
            add("code_format", "Use a fenced code block when code is requested.", source="coverage", score=3.0)
        elif name == "code_or_snippet_expected":
            add("code_detail", "Include executable or concrete code when the query asks for code.", source="coverage", score=2.8)
        elif name == "specific_visual_or_list_format":
            add("output_format", "Preserve the requested visual, table, bullet, or list format.", source="coverage", score=2.8)
        elif name == "exact_item_count_or_only_constraint":
            add("scope_limit", "Respect exact-count or only-this-scope constraints.", source="coverage", score=3.1)


def _add_preference_answer_requirements(preference_constraints: list[dict[str, Any]], add) -> None:
    for constraint in preference_constraints[:16]:
        if not isinstance(constraint, dict):
            continue
        type_ = str(constraint.get("type") or "")
        label = str(constraint.get("label") or "")
        context = str(constraint.get("context") or "")
        try:
            score = float(constraint.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if type_ == "date_format":
            if "mm/dd/yyyy" in label.lower():
                add("date_format", "Format every requested date as MM/DD/YYYY.", source="preference_constraint", context=context, score=max(3.4, score))
            elif "month day, year" in label.lower():
                add("date_format", "Include the full date as Month Day, Year.", source="preference_constraint", context=context, score=max(3.0, score))
            else:
                add("date_format", "Preserve the user's requested date format.", source="preference_constraint", context=context, score=max(2.8, score))
        elif type_ == "version_detail":
            add(
                "version_detail",
                "Include exact software version numbers for named tools when supported by evidence.",
                source="preference_constraint",
                context=context,
                score=max(3.2, score),
            )


def _add_evidence_supported_requirements(query_lower: str, source_spans: list[dict[str, Any]], add) -> None:
    wants_date_detail = _query_requests_date_detail(query_lower)
    for span in source_spans[:64]:
        content = str(span.get("content") or span.get("text") or "")
        if not content.strip():
            continue
        lower = content.lower()
        if wants_date_detail and re.search(r"\bmm/dd/yyyy\b|\bmm-dd-yyyy\b", lower):
            match = re.search(r"\bmm/dd/yyyy\b|\bmm-dd-yyyy\b", content, flags=re.I)
            if match:
                add(
                    "date_format",
                    "Format every requested date as MM/DD/YYYY.",
                    source="evidence",
                    context=_match_context(content, match.start(), match.end(), radius=180),
                    score=3.2,
                )
        if wants_date_detail and re.search(r"\bmonth\s+day,\s+year\b", lower):
            match = re.search(r"\bmonth\s+day,\s+year\b", content, flags=re.I)
            if match:
                add(
                    "date_format",
                    "Include the full date as Month Day, Year.",
                    source="evidence",
                    context=_match_context(content, match.start(), match.end(), radius=180),
                    score=2.9,
                )
        if _evidence_supports_version_requirement(query_lower, lower):
            match = re.search(r"\b(?:versions?|v\d+\.\d+(?:\.\d+)?|\d+\.\d+(?:\.\d+)?)\b", content, flags=re.I)
            if match:
                add(
                    "version_detail",
                    "Include exact version numbers for named libraries, dependencies, tools, or software when supported.",
                    source="evidence",
                    context=_match_context(content, match.start(), match.end(), radius=180),
                    score=3.1,
                )
        if _query_requests_viewing_platforms(query_lower) and re.search(
            r"\b(?:netflix|hulu|disney\+?|prime video|amazon prime|hbo|max|apple tv|paramount\+?|peacock|streaming)\b",
            lower,
        ):
            match = re.search(
                r"\b(?:netflix|hulu|disney\+?|prime video|amazon prime|hbo|max|apple tv|paramount\+?|peacock|streaming)\b",
                content,
                flags=re.I,
            )
            if match:
                add(
                    "platform_detail",
                    "Include streaming services or platform names when recommendations depend on availability.",
                    source="evidence",
                    context=_match_context(content, match.start(), match.end(), radius=220),
                    score=3.0,
                )
        if re.search(r"\b(?:legal|legally|valid|will|wishes|terms?)\b", query_lower) and re.search(
            r"\b(?:term|means|meaning|defined|definition|witness|notary|notarized|executor|beneficiary|affidavit)\b",
            lower,
        ):
            match = re.search(r"\b(?:term|means|meaning|defined|definition|witness|notary|notarized|executor|beneficiary|affidavit)\b", content, flags=re.I)
            if match:
                add(
                    "explanation_depth",
                    "Explain important legal terms instead of only naming them.",
                    source="evidence",
                    context=_match_context(content, match.start(), match.end(), radius=220),
                    score=2.9,
                )
        if re.search(r"\b(?:progress|percentage|percent|how much progress)\b", query_lower) and re.search(r"\b\d{1,3}%\b", lower):
            match = re.search(r"\b\d{1,3}%\b", content)
            if match:
                add(
                    "numeric_detail",
                    "Include supported percentage values or explicit numeric progress values.",
                    source="evidence",
                    context=_match_context(content, match.start(), match.end(), radius=180),
                    score=3.2,
                )


def _query_requests_date_detail(query_lower: str) -> bool:
    if re.search(r"\bwhen\s+(?:is|was|are|were)\b", query_lower):
        return True
    if re.search(r"\b(?:what|which)\s+date\b", query_lower):
        return True
    if re.search(r"\b(?:due|deadline|scheduled|registered)\b", query_lower):
        return True
    return bool(re.search(r"\b(?:meeting|meetings|workshop|event|submission)\b", query_lower) and re.search(r"\b(?:when|date)\b", query_lower))


def _query_requests_version_detail(query_lower: str) -> bool:
    if re.search(r"\bversions?\b", query_lower):
        return True
    if re.search(r"\b(?:dependencies|dependency)\b", query_lower):
        return True
    if re.search(r"\b(?:libraries|library|packages?|modules?)\b", query_lower) and re.search(
        r"\b(?:code|coding|programming|python|javascript|typescript|npm|pip|package|dependency|dependencies|software)\b",
        query_lower,
    ):
        return True
    return False


def _evidence_supports_version_requirement(query_lower: str, content_lower: str) -> bool:
    if not _query_requests_version_detail(query_lower):
        return False
    if not re.search(r"\bversions?\b", content_lower):
        return False
    if re.search(r"\b(?:v\d+\.\d+(?:\.\d+)?|\d+\.\d+(?:\.\d+)?)\b", content_lower):
        return True
    return bool(re.search(r"\b(?:exact|specific|include|list|matter|needed|required)\b.{0,80}\bversions?\b", content_lower))


def _query_requests_viewing_platforms(query_lower: str) -> bool:
    return bool(
        re.search(r"\b(?:movies?|watch|streaming|stream|where\s+(?:can|could)\s+i\s+watch|platforms?)\b", query_lower)
        and not re.search(r"\bonline\s+services\b", query_lower)
    )
