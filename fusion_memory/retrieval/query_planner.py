from __future__ import annotations

import re
from typing import Any

from fusion_memory.core.models import QueryPlan
from fusion_memory.core.text import extract_entities
from fusion_memory.retrieval.query_intent import analyze_query_intent, refine_query_intent_with_llm, should_refine_query_intent


EVENT_ORDERING_PHRASES = (
    "order in which",
    "in order",
    "chronological",
    "chronology",
    "sequence",
    "timeline",
)
TEMPORAL_HINTS = (
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
    "deadline",
    "duration",
    "how many weeks",
    "how many days",
    "between",
    "weeks between",
    "days between",
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
)
RETRIEVAL_STOPWORDS = {
    "a",
    "across",
    "all",
    "and",
    "answer",
    "aspect",
    "aspects",
    "brought",
    "can",
    "chats",
    "conversations",
    "could",
    "different",
    "during",
    "each",
    "five",
    "four",
    "how",
    "i",
    "in",
    "into",
    "items",
    "list",
    "me",
    "mention",
    "only",
    "order",
    "our",
    "please",
    "show",
    "tell",
    "the",
    "their",
    "them",
    "these",
    "things",
    "through",
    "throughout",
    "three",
    "up",
    "walk",
    "was",
    "we",
    "which",
    "you",
}


class QueryPlanner:
    def __init__(
        self,
        *,
        intent_refiner: Any | None = None,
        intent_refiner_min_confidence: float = 0.70,
        intent_refiner_mode: str = "auto",
    ) -> None:
        self.intent_refiner = intent_refiner
        self.intent_refiner_min_confidence = intent_refiner_min_confidence
        self.intent_refiner_mode = intent_refiner_mode
        self.last_intent_telemetry: dict[str, Any] | None = None

    def plan(self, query: str, *, query_type_hint: str | None = None) -> QueryPlan:
        lower = query.lower()
        intent = analyze_query_intent(query)
        self.last_intent_telemetry = None
        if self.intent_refiner is not None and self._should_call_intent_refiner(query, intent):
            intent, self.last_intent_telemetry = refine_query_intent_with_llm(
                self.intent_refiner,
                query,
                intent,
                min_confidence=self.intent_refiner_min_confidence,
            )
        intent_requests_multi_session = intent.evidence_scope == "multi_session"
        intent_requests_aggregation = intent.aggregation.operation not in {"", "none"} or intent.answer_shape in {"count", "sum", "unordered_list"}
        intent_requests_event_order = intent.answer_shape == "ordered_list" and intent.temporal.requires_order
        intent_was_refined = "llm_refined" in intent.route_reasons
        query_type = "factual_exact"
        needs_current = False
        must_include = ["raw_evidence"]
        if _is_preference_query(lower):
            query_type = "preference"
            needs_current = _preference_query_needs_current(lower)
            must_include = ["current_view", "raw_evidence"] if needs_current else ["raw_evidence", "facts"]
        elif any(w in lower for w in ["current", "currently", "now", "现在", "当前", "以后按"]) or intent.needs_current_state:
            if any(w in lower for w in ["prefer", "preference", "喜欢", "偏好", "用什么"]):
                query_type = "preference"
                must_include = ["current_view", "raw_evidence"]
            elif _is_current_value_query(lower) or intent.needs_current_state:
                query_type = "knowledge_update"
                must_include = ["raw_evidence", "facts", "events", "current_view"]
            else:
                query_type = "instruction"
                must_include = ["current_view", "raw_evidence"]
            needs_current = True
        if _is_event_ordering_query(lower) or intent_requests_event_order:
            query_type = "event_ordering"
            must_include = ["raw_evidence", "events"]
        elif (
            (any(w in lower for w in TEMPORAL_HINTS) or intent.temporal.requires_time)
            and not _is_historical_yes_no_query(lower)
            and not _is_hypothetical_or_procedural_sequence(lower)
            and not _is_non_temporal_first_time_phrase(lower)
        ):
            query_type = "temporal_lookup"
            must_include = ["raw_evidence", "events"]
        if query_type == "factual_exact" and _is_current_value_query(lower) and not _is_historical_said_query(lower):
            query_type = "knowledge_update"
            must_include = ["raw_evidence", "facts", "events"]
        if query_type in {"factual_exact", "knowledge_update", "summarization"} and (
            _is_multi_session_query(lower) or (intent_requests_multi_session and intent_requests_aggregation)
        ):
            query_type = "multi_session_reasoning"
            must_include = ["raw_evidence", "facts"]
        if (
            query_type == "temporal_lookup"
            and intent_requests_multi_session
            and intent_requests_aggregation
            and not intent.temporal.requires_duration
        ):
            query_type = "multi_session_reasoning"
            must_include = ["raw_evidence", "facts"]
        if query_type == "event_ordering" and (
            _is_multi_session_query(lower) or intent_requests_multi_session
        ) and not (_is_strict_event_ordering_query(lower) or (intent_was_refined and intent_requests_event_order)):
            query_type = "multi_session_reasoning"
            must_include = ["raw_evidence", "facts"]
        if query_type == "temporal_lookup" and _is_non_temporal_first_time_phrase(lower):
            query_type = "factual_exact"
            must_include = ["raw_evidence"]
        if query_type != "multi_session_reasoning" and not (_is_explicit_aggregation_query(lower) or intent_requests_aggregation) and (
            _is_historical_yes_no_query(lower)
            or any(w in lower for w in ["contradict", "conflict", "changed", "switched", "矛盾", "冲突", "改"])
        ):
            query_type = "contradiction_resolution"
            must_include = ["raw_evidence", "facts"]
        if any(w in lower for w in ["unknown", "not mentioned", "没有提到", "不知道", "cluster name"]):
            query_type = "abstention"
            must_include = ["raw_evidence"]
        if any(w in lower for w in ["summarize", "summary", "总结"]):
            query_type = "summarization"
            must_include = ["raw_evidence"]
        speaker_focus = "any"
        if query_type == "event_ordering" and (
            any(phrase in lower for phrase in ["i brought up", "i mentioned", "i asked", "my "])
            or re.search(r"我.*(?:提到|提出|问过|说过|聊过)", lower)
        ):
            speaker_focus = "user"
        if any(
            w in lower
            for w in [
                "you suggested",
                "you recommend",
                "you recommended",
                "did you recommend",
                "how did you recommend",
                "what steps did you recommend",
                "assistant",
                "你建议",
                "上次你",
            ]
        ):
            speaker_focus = "assistant"
            query_type = "assistant_reference"
        query_type, must_include = _apply_query_type_hint(query_type, must_include, query_type_hint)
        if query_type == "event_ordering" and (
            any(phrase in lower for phrase in ["i brought up", "i mentioned", "i asked", "my "])
            or re.search(r"我.*(?:提到|提出|问过|说过|聊过)", lower)
        ):
            speaker_focus = "user"
        return QueryPlan(
            query=query,
            query_type=query_type,
            entities=extract_entities(query),
            time_constraints=_time_constraints(query),
            retrieval_hints=_retrieval_hints(query),
            speaker_focus=speaker_focus,
            needs_current_state=needs_current,
            needs_source_evidence=True,
            must_include_sources=must_include,
            intent=intent.to_dict(),
        )

    def _should_call_intent_refiner(self, query: str, intent: Any) -> bool:
        if self.intent_refiner_mode == "always":
            return True
        if self.intent_refiner_mode == "never":
            return False
        return should_refine_query_intent(query, intent)


def _apply_query_type_hint(
    query_type: str,
    must_include: list[str],
    query_type_hint: str | None,
) -> tuple[str, list[str]]:
    hint = str(query_type_hint or "").strip()
    if not hint:
        return query_type, must_include
    hint_map = {
        "contradiction_resolution": ("contradiction_resolution", ["raw_evidence", "facts"]),
        "multi_session_reasoning": ("multi_session_reasoning", ["raw_evidence", "facts"]),
        "summarization": ("summarization", ["raw_evidence"]),
        "event_ordering": ("event_ordering", ["raw_evidence", "events"]),
        "knowledge_update": ("knowledge_update", ["raw_evidence", "facts", "events", "current_view"]),
        "temporal_reasoning": ("temporal_lookup", ["raw_evidence", "events"]),
        "information_extraction": ("factual_exact", ["raw_evidence"]),
        "instruction_following": ("instruction", ["current_view", "raw_evidence"]),
        "preference_following": ("preference", ["current_view", "raw_evidence"]),
        "abstention": ("abstention", ["raw_evidence"]),
    }
    return hint_map.get(hint, (query_type, must_include))


def _is_preference_query(lower: str) -> bool:
    if re.search(r"\b(?:how many|how much|count|total|number of)\b", lower):
        return False
    if re.search(r"\b(?:prefer|preference|preferences|like|likes)\b", lower):
        return True
    if re.search(
        r"\bwhat\s+(?:do|does|did|should|would)\s+[^?。！？]{0,80}\b(?:prefer|like|use|eat|drink|watch|read|listen)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(?:what|which)\s+(?:drink|food|tool|model|database|framework|library|language|format|style|option|book|movie|music)\b",
        lower,
    ):
        return True
    if re.search(r"(?:喜欢|偏好|倾向|想要|想用|要用|爱吃|爱喝|常吃|常喝)", lower):
        return True
    if re.search(r"(?:喝|吃|用|看|读|听|选择|选|采用|使用)\s*什么", lower):
        return True
    if re.search(r"什么(?:饮料|食物|工具|模型|数据库|框架|语言|格式|风格|选项|书|电影|音乐)", lower):
        return True
    return False


def _preference_query_needs_current(lower: str) -> bool:
    if re.search(r"\b(?:mention|mentioned|history|historical|over time|across|throughout|list|all)\b", lower):
        return False
    if re.search(r"(?:提到|说过|历史|变化|所有|全部|列出|总结)", lower):
        return False
    return True


def _is_event_ordering_query(lower: str) -> bool:
    if _is_hypothetical_or_procedural_sequence(lower):
        return False
    if any(phrase in lower for phrase in EVENT_ORDERING_PHRASES):
        return True
    if re.search(r"按顺序|顺序|时间线|先后|依次|先.*再", lower):
        return True
    if any(marker in lower for marker in ["what order", "first came up", "brought up"]) and "conversation" in lower:
        return True
    if ("first" in lower and "then" in lower) or ("first" in lower and "next" in lower):
        return True
    return False


def _is_hypothetical_or_procedural_sequence(lower: str) -> bool:
    return bool(
        re.search(r"\bif\b.{0,160}\bthen\b", lower)
        or (
            re.search(r"\b(?:chance|probability|calculate|figure out|how do i|how should i)\b", lower)
            and re.search(r"\b(?:first|then|next|after)\b", lower)
        )
    )


def _is_non_temporal_first_time_phrase(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:meet|meeting|met)\s+(?:someone|people|a person|somebody)\s+for\s+the\s+first\s+time\b", lower)
        or re.search(r"\bfor\s+the\s+first\s+time\s+(?:meeting|meet|met)\b", lower)
    )


def _is_strict_event_ordering_query(lower: str) -> bool:
    return bool(
        _is_event_ordering_query(lower)
        and (
            any(phrase in lower for phrase in EVENT_ORDERING_PHRASES)
            or re.search(r"\b(?:what order|first came up|brought up)\b", lower)
            or re.search(r"按顺序|顺序|时间线|先后|依次", lower)
        )
        and not re.search(r"\b(?:how many|total|evolved|progress|compare|balance|optimize|summary|summarize|how well|how should)\b|多少|几个|总共|一共|总结|比较|如何", lower)
    )


def _is_historical_yes_no_query(lower: str) -> bool:
    if re.search(r"\bhow many\b", lower) or re.search(r"\b(?:how much|what (?:count|number))\b", lower):
        return False
    return bool(
        re.search(r"\bhave\s+i\s+(?:ever\s+)?(?:used|worked|read|listened|met|done|tried|mentioned)\b", lower)
        or re.search(r"\bdid\s+i\s+(?:ever\s+)?(?:use|work|read|listen|meet|do|try|mention)\b", lower)
    )


def _is_multi_session_query(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:across|throughout|over time|in total|total after|how many different|between my .+ and my .+|considering .+ and .+)\b", lower)
        or re.search(r"\b(?:how have|how has|how did|how do|how will)\b.{0,120}\b(?:evolved?|progress(?:ed)?|changed|developed|balanced?|align(?:ed)?|affect|optimi[sz]e|prioriti[sz]e)\b", lower)
        or re.search(r"\b(?:considering|given)\b.{0,180}\b(?:and|,)\b.{0,180}\b(?:how|what|which|should|can)\b", lower)
        or re.search(r"\bhow\s+will\b.{0,180}\b(?:while|and|with)\b.{0,180}\b(?:affect|impact|influence|change)\b", lower)
        or re.search(r"\bhow\s+will\b.{0,180}\b(?:affect|impact|influence|change)\b.{0,180}\b(?:while|and|with)\b", lower)
        or (
            re.search(r"\b(?:timeline|actions?|progress|achievements?|choices?|preferences?|goals?|feedback|constraints?)\b", lower)
            and re.search(r"\b(?:how have|how has|how did|how do|how will|how well|how should|evolved?|progress(?:ed)?|changed|developed|balanced?|align(?:ed)?|affect|optimi[sz]e|prioriti[sz]e)\b", lower)
        )
        or (re.search(r"\bhow many\b", lower) and re.search(r"\b(?:requests?|conversations?|sessions?|features?|concerns?|columns?|cards?|sources?|types?|plans?|areas?)\b", lower))
        or re.search(r"跨.*(?:会话|对话)|所有(?:会话|对话)|多次(?:会话|对话)|整个(?:会话|对话)|全部(?:会话|对话)", lower)
        or (re.search(r"多少|几个|几种|一共|总共|不同", lower) and re.search(r"会话|对话|消息|功能|关注点|类型|计划|方面|角色|安全", lower))
    )


def _is_explicit_aggregation_query(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:how many|total|unique|count|number of|different|list)\b", lower)
        and re.search(r"\b(?:across|throughout|over time|sessions?|conversations?|chats?|in total|total)\b", lower)
        or (re.search(r"多少|几个|几种|一共|总共|不同|列出", lower) and re.search(r"跨|所有|全部|多次|会话|对话|消息", lower))
    )


def _is_current_value_query(lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:current|currently|latest|recent|recently|now|final|updated|reached|improved|reduced|deadline|target|target date|budget|amount|count|number|version|status|response time|word count|subscription|snack budget|weekly target|monthly budget)\b",
            lower,
        )
        or re.search(r"\bwhat\s+is\s+(?:the\s+)?(?:average|deadline|count|number|version|status|response time|budget|target|word count)\b", lower)
        or re.search(r"\bby\s+what\s+date\b", lower)
        or re.search(r"\bwhat\s+deadline\b", lower)
    )


def _is_historical_said_query(lower: str) -> bool:
    return bool(
        re.search(r"\b(?:did\s+i\s+say|what\s+did\s+i\s+say|i\s+said|i\s+mentioned|i\s+told\s+you)\b", lower)
        or re.search(r"我.*(?:说过|提到过|告诉过)", lower)
    ) and not bool(re.search(r"\b(?:current|currently|latest|recent|recently|now|updated|newest)\b", lower))


def _retrieval_hints(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", query.lower())
    hints: list[str] = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token in RETRIEVAL_STOPWORDS:
            continue
        hints.append(token)
    return list(dict.fromkeys(hints[:8]))


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
