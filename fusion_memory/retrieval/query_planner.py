from __future__ import annotations

import re

from fusion_memory.core.text import ENTITY_STOPWORDS, extract_entities, tokenize
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderKind,
    ProviderRequest,
    SearchRequest,
)
from fusion_memory.retrieval.query_intent import QueryIntent, analyze_query_intent


_FALLBACK_QUERY_STOPWORDS = ENTITY_STOPWORDS | {
    "about",
    "across",
    "all",
    "any",
    "current",
    "currently",
    "different",
    "has",
    "have",
    "having",
    "its",
    "latest",
    "many",
    "mine",
    "my",
    "now",
    "of",
    "our",
    "ours",
    "this",
    "was",
    "were",
    "your",
}
_FALLBACK_CJK_QUERY_FILLER_RE = re.compile(
    r"(?:请问|我(?:们)?(?:的)?|你(?:们)?(?:的)?|您(?:的)?|"
    r"目前|当前|现在|正在|使用|什么|哪一个|哪个|哪种)"
)


class ProductQueryPlanner:
    """Build product retrieval plans from deterministic query capabilities."""

    def plan(self, request: SearchRequest) -> ProductQueryPlan:
        intent = analyze_query_intent(request.query)
        return ProductQueryPlan(
            intent=_intent_label(intent),
            provider_requests=_provider_requests(intent, request.limit),
            time_range=request.time_range,
            entities=tuple(intent.entities),
            speaker=None if intent.speaker_scope == "any" else intent.speaker_scope,
            ordering=_ordering(intent),
            use_reranker=request.mode == "balanced",
            query_intent=intent.to_dict(),
        )

    def safe_default(self, request: SearchRequest) -> ProductQueryPlan:
        entities = tuple(extract_entities(request.query))
        return ProductQueryPlan(
            intent="factual",
            provider_requests=(
                ProviderRequest(ProviderKind.VECTOR, max(request.limit * 2, 12)),
                ProviderRequest(ProviderKind.LEXICAL, max(request.limit * 2, 12)),
            ),
            time_range=request.time_range,
            entities=entities,
            speaker=None,
            ordering=OrderingMode.RELEVANCE,
            use_reranker=request.mode == "balanced",
            query_intent=_safe_default_query_intent(request.query, entities),
        )


def _intent_label(intent: QueryIntent) -> str:
    if intent.temporal.requires_order:
        return "chronology"
    if intent.needs_current_state:
        return "current_state"
    if intent.needs_conflict_check:
        return "conflict"
    if intent.answer_shape == "summary":
        return "summary"
    if intent.aggregation.operation != "none":
        return "aggregation"
    if intent.answer_shape == "instruction":
        return "instruction"
    if intent.temporal.requires_time:
        return "temporal"
    return "factual"


def _provider_requests(intent: QueryIntent, limit: int) -> tuple[ProviderRequest, ...]:
    kinds = [ProviderKind.VECTOR, ProviderKind.LEXICAL]
    if intent.entities:
        kinds.append(ProviderKind.ENTITY)
    if intent.temporal.requires_time or intent.needs_current_state:
        kinds.append(ProviderKind.TEMPORAL)
    if intent.temporal.requires_order:
        kinds.append(ProviderKind.CHRONOLOGY)
    if not intent.entities:
        kinds.append(ProviderKind.ENTITY)
    return tuple(ProviderRequest(kind, max(limit * 2, 12)) for kind in dict.fromkeys(kinds))


def _ordering(intent: QueryIntent) -> OrderingMode:
    if intent.temporal.requires_order:
        return OrderingMode.CHRONOLOGICAL
    if intent.needs_current_state:
        return OrderingMode.RECENCY
    return OrderingMode.RELEVANCE


def _safe_default_query_intent(
    query: str,
    entities: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": "query-intent-v1",
        "language": _fallback_language(query),
        "answer_shape": "short_answer",
        "evidence_scope": "local_or_best_match",
        "speaker_scope": "any",
        "entities": list(entities),
        "target_terms": _fallback_target_terms(query),
        "object_types": [],
        "temporal": {
            "requires_time": False,
            "requires_order": False,
            "requires_duration": False,
            "order_direction": "unknown",
            "endpoint_roles": [],
            "time_expressions": [],
        },
        "aggregation": {
            "operation": "none",
            "distinct": False,
            "target_terms": [],
            "unit_terms": [],
        },
        "needs_current_state": False,
        "needs_conflict_check": False,
        "confidence": 0.0,
        "route_reasons": ["planner_fallback"],
    }


def _fallback_target_terms(query: str) -> list[str]:
    normalized = _FALLBACK_CJK_QUERY_FILLER_RE.sub(" ", query.lower())
    terms: list[str] = []
    for token in tokenize(normalized):
        has_cjk = any("\u4e00" <= character <= "\u9fff" for character in token)
        if has_cjk:
            if len(token) < 2:
                continue
        elif len(token) < 3 or token in _FALLBACK_QUERY_STOPWORDS:
            continue
        terms.append(token)
    return list(dict.fromkeys(terms[:16]))


def _fallback_language(query: str) -> str:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query))
    has_latin = bool(re.search(r"[A-Za-z]", query))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    return "en"
