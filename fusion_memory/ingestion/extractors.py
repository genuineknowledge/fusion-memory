from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from fusion_memory.core.models import EvidenceSpan, ExtractedCandidate, MemoryFact, new_id
from fusion_memory.core.text import compact_summary, extract_entities
from fusion_memory.ingestion.temporal_normalizer import TemporalNormalizer


PREFERENCE_RE = re.compile(r"\b(?:i|we)\s+(?:now\s+)?(?:prefer|like|use|want)\s+(.+?)(?:[.!?]|$)", re.I)
NOW_PREFERRED_RE = re.compile(r"\b([A-Z][A-Za-z0-9_\- ]{1,80}?)\s+is\s+now\s+preferred\b", re.I)
SWITCH_RE = re.compile(r"\b(?:switched|moved|changed)\s+(.+?)\s+(?:from\s+(.+?)\s+)?to\s+(.+?)(?:[.!?]|$)", re.I)
INSTRUCTION_RE = re.compile(r"\b(?:remember|always|please|以后|记住|默认|do not|don't)\b", re.I)
EVENT_RE = re.compile(
    r"\b(?:"
    r"tested|switched|added|removed|decided|deployed|created|fixed|changed|moved|started|finished|"
    r"planned|mentioned|asked|discussed|reviewed|implemented|configured|debugged|launched|"
    r"brought\s+up|came\s+up|worked\s+on"
    r")\b",
    re.I,
)
ZH_PREFERENCE_RE = re.compile(
    r"(?:我|我们|用户)?(?:现在|目前|以后|默认)?(?:更?喜欢|偏好|倾向于|想用|要用|希望用|打算用|准备用|采用|使用)\s*(?:用|使用)?\s*(.+?)(?:[。！？.!?]|$)"
)
ZH_SWITCH_RE = re.compile(
    r"(?:我|我们)?(?:已经|现在|决定|打算|准备)?\s*(?:把\s*)?(.{1,80}?)\s*从\s*(.+?)\s*(?:切换|迁移|换|改|转)(?:到|为|成)\s*(.+?)(?:[。！？.!?]|$)"
)
ZH_INSTRUCTION_RE = re.compile(r"(?:以后|记住|请记住|默认|总是|不要|别|请|务必|必须)")
ZH_EVENT_RE = re.compile(r"(?:测试|切换|迁移|决定|部署|创建|修复|更改|开始|完成|实现|配置|调试|上线|讨论|评审|接入|处理)")


class RuleBasedExtractor:
    def __init__(self) -> None:
        self.temporal = TemporalNormalizer()

    def extract(self, spans: list[EvidenceSpan], existing_facts: list[MemoryFact], session_time: datetime) -> list[ExtractedCandidate]:
        candidates: list[ExtractedCandidate] = []
        for span in spans:
            candidates.extend(self._extract_fact_candidates(span))
            candidates.extend(self._extract_event_candidates(span, session_time))
        candidates.extend(self._extract_relation_candidates(candidates, existing_facts))
        return candidates

    def _extract_fact_candidates(self, span: EvidenceSpan) -> list[ExtractedCandidate]:
        text = span.content.strip()
        lower = text.lower()
        out: list[ExtractedCandidate] = []
        if span.speaker == "tool":
            out.append(
                self._candidate(
                    "fact",
                    f"Tool result: {compact_summary(text, 220)}",
                    span,
                    {
                        "subject": "tool",
                        "predicate": "returned",
                        "object": compact_summary(text, 180),
                        "category": "tool_result",
                        "confidence": 0.82,
                        "salience": 0.72,
                    },
                )
            )
            return out
        if span.speaker in {"assistant", "agent"}:
            if any(word in lower for word in ["recommend", "suggest", "plan", "建议", "方案"]):
                out.append(
                    self._candidate(
                        "fact",
                        f"Assistant/agent stated: {compact_summary(text, 220)}",
                        span,
                        {
                            "subject": span.speaker,
                            "predicate": "stated",
                            "object": compact_summary(text, 180),
                            "category": "assistant_statement" if span.speaker == "assistant" else "agent_action",
                            "confidence": 0.76,
                            "salience": 0.55,
                        },
                    )
                )
            return out
        if "don't remember that as my preference" in lower or "do not remember that as my preference" in lower:
            out.append(
                self._candidate(
                    "fact",
                    "User explicitly said not to store the previous suggestion as a preference.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "rejects_memory",
                        "object": "previous suggestion as preference",
                        "category": "instruction",
                        "polarity": "negative",
                        "topic_terms": _topic_terms(text),
                        "confidence": 0.85,
                        "salience": 0.68,
                    },
                )
            )
            return out
        switch = SWITCH_RE.search(text)
        if switch:
            target = switch.group(3).strip()
            out.append(
                self._candidate(
                    "fact",
                    f"User switched {switch.group(1).strip()} to {target}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "switched_to",
                        "object": target,
                        "category": "project_state",
                        "polarity": _fact_polarity(text),
                        "value_mentions": _value_mentions(text),
                        "topic_terms": _topic_terms(text),
                        "confidence": 0.86,
                        "salience": 0.82,
                    },
                )
            )
        zh_switch = ZH_SWITCH_RE.search(text)
        if zh_switch:
            target = _clean_zh_object(zh_switch.group(3))
            subject = _clean_zh_object(zh_switch.group(1))
            out.append(
                self._candidate(
                    "fact",
                    f"User switched {subject} to {target}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "switched_to",
                        "object": target,
                        "category": "project_state",
                        "polarity": _fact_polarity(text),
                        "value_mentions": _value_mentions(text),
                        "topic_terms": _topic_terms(text),
                        "confidence": 0.86,
                        "salience": 0.82,
                    },
                )
            )
        pref = PREFERENCE_RE.search(text)
        if pref:
            obj = pref.group(1).strip()
            out.append(
                self._candidate(
                    "fact",
                    f"User prefers {obj}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "prefers",
                        "object": obj,
                        "category": "preference",
                        "polarity": _fact_polarity(text),
                        "value_mentions": _value_mentions(text),
                        "topic_terms": _topic_terms(text),
                        "confidence": 0.82,
                        "salience": 0.78,
                    },
                )
            )
        zh_pref = ZH_PREFERENCE_RE.search(text)
        if zh_pref and not zh_switch:
            obj = _clean_zh_object(zh_pref.group(1))
            out.append(
                self._candidate(
                    "fact",
                    f"User prefers {obj}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "prefers",
                        "object": obj,
                        "category": "preference",
                        "polarity": _fact_polarity(text),
                        "value_mentions": _value_mentions(text),
                        "topic_terms": _topic_terms(text),
                        "confidence": 0.82,
                        "salience": 0.78,
                    },
                )
            )
        now_pref = NOW_PREFERRED_RE.search(text)
        if now_pref:
            obj = now_pref.group(1).strip()
            out.append(
                self._candidate(
                    "fact",
                    f"User prefers {obj}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "prefers",
                        "object": obj,
                        "category": "preference",
                        "polarity": _fact_polarity(text),
                        "value_mentions": _value_mentions(text),
                        "topic_terms": _topic_terms(text),
                        "confidence": 0.84,
                        "salience": 0.80,
                    },
                )
            )
        has_instruction = INSTRUCTION_RE.search(text) or ZH_INSTRUCTION_RE.search(text)
        if has_instruction and not pref and not zh_pref:
            category = "instruction" if any(w in lower for w in ["always", "please", "do not", "don't"]) or ZH_INSTRUCTION_RE.search(text) else "general_fact"
            out.append(
                self._candidate(
                    "fact",
                    f"User instruction/fact: {compact_summary(text, 220)}",
                    span,
                    {
                        "subject": "user",
                        "predicate": "said",
                        "object": compact_summary(text, 180),
                        "category": category,
                        "polarity": _fact_polarity(text),
                        "value_mentions": _value_mentions(text),
                        "topic_terms": _topic_terms(text),
                        "confidence": 0.72,
                        "salience": 0.55 if category == "general_fact" else 0.70,
                    },
                )
            )
        elif not out and len(text) > 20 and span.speaker == "user":
            if any(w in lower for w in ["database", "project", "atlas", "qdrant", "postgres", "kubernetes"]):
                out.append(
                    self._candidate(
                        "fact",
                        f"User said: {compact_summary(text, 220)}",
                        span,
                        {
                            "subject": "user",
                            "predicate": "said",
                            "object": compact_summary(text, 180),
                            "category": "general_fact",
                            "polarity": _fact_polarity(text),
                            "value_mentions": _value_mentions(text),
                            "topic_terms": _topic_terms(text),
                            "confidence": 0.64,
                            "salience": 0.45,
                        },
                    )
                )
        return out

    def _extract_event_candidates(self, span: EvidenceSpan, session_time: datetime) -> list[ExtractedCandidate]:
        generic_facets = extract_generic_event_facets(span.content) if span.speaker == "user" else []
        milestone_mentions = extract_milestone_mentions(span.content) if span.speaker == "user" else []
        if not generic_facets and not milestone_mentions and not EVENT_RE.search(span.content) and not ZH_EVENT_RE.search(span.content):
            return []
        normalized = self.temporal.normalize(span.content, session_time)
        participants = list(dict.fromkeys([*(extract_entities(span.content) or ["user"]), *_topic_terms(span.content)[:6]]))
        out: list[ExtractedCandidate] = []
        for facet, label, snippet in generic_facets[:6]:
            out.append(
                self._candidate(
                    "event",
                    _facet_description(facet, label, snippet),
                    span,
                    {
                        "event_type": facet,
                        "participants": list(dict.fromkeys([facet, label.lower(), *participants])),
                        "description": _facet_description(facet, label, snippet),
                        "time_start": normalized.time_start.isoformat() if normalized.time_start else None,
                        "time_end": normalized.time_end.isoformat() if normalized.time_end else None,
                        "time_granularity": normalized.granularity,
                        "time_source": normalized.source,
                        "confidence": 0.80,
                    },
                )
            )
        if milestone_mentions:
            for milestone, snippet in milestone_mentions[:4]:
                out.append(
                    self._candidate(
                        "event",
                        _milestone_description(milestone, snippet),
                        span,
                        {
                            "event_type": "milestone",
                            "participants": list(dict.fromkeys([milestone, *participants])),
                            "description": _milestone_description(milestone, snippet),
                            "time_start": normalized.time_start.isoformat() if normalized.time_start else None,
                            "time_end": normalized.time_end.isoformat() if normalized.time_end else None,
                            "time_granularity": normalized.granularity,
                            "time_source": normalized.source,
                            "confidence": 0.82,
                        },
                    )
                )
            return out
        if out:
            return out
        description = compact_summary(span.content, 240)
        return [
            self._candidate(
                "event",
                description,
                span,
                {
                    "event_type": _event_type(span.content),
                    "participants": participants,
                    "description": description,
                    "time_start": normalized.time_start.isoformat() if normalized.time_start else None,
                    "time_end": normalized.time_end.isoformat() if normalized.time_end else None,
                    "time_granularity": normalized.granularity,
                    "time_source": normalized.source,
                    "confidence": 0.74,
                },
            )
        ]

    def _extract_relation_candidates(
        self, candidates: Iterable[ExtractedCandidate], existing_facts: list[MemoryFact]
    ) -> list[ExtractedCandidate]:
        out: list[ExtractedCandidate] = []
        for candidate in candidates:
            if candidate.candidate_type != "fact":
                continue
            structured = candidate.structured
            if structured.get("category") not in {"preference", "project_state", "instruction"}:
                continue
            for fact in existing_facts:
                if fact.category != structured.get("category"):
                    continue
                if fact.object.lower() == str(structured.get("object", "")).lower():
                    continue
                if fact.subject == structured.get("subject"):
                    out.append(
                        ExtractedCandidate(
                            local_id=new_id("cand"),
                            candidate_type="relation",
                            text=f"{candidate.text} supersedes {fact.text}",
                            structured={
                                "relation_type": "supersedes",
                                "from_local_id": candidate.local_id,
                                "to_fact_id": fact.fact_id,
                                "confidence": 0.78,
                            },
                            confidence=0.78,
                            source_span_ids=candidate.source_span_ids + fact.source_span_ids,
                            extractor_name="relation_detector",
                        )
                    )
        return out

    def _candidate(self, candidate_type: str, text: str, span: EvidenceSpan, structured: dict) -> ExtractedCandidate:
        return ExtractedCandidate(
            local_id=new_id("cand"),
            candidate_type=candidate_type,
            text=text,
            structured=structured,
            confidence=float(structured.get("confidence", 0.5)),
            source_span_ids=[span.span_id],
            extractor_name="rule_based_extractor",
        )


def _event_type(text: str) -> str:
    if re.search(r"\b(?:switched|changed|moved)\b", text, re.I):
        return "state_change"
    if re.search(r"(?:切换|迁移|改为|换成)", text):
        return "state_change"
    if re.search(r"(?:决定|选择|选定|敲定)", text):
        return "decision"
    if re.search(r"\b(?:planned|discussed|mentioned|asked|brought\s+up|came\s+up)\b", text, re.I):
        return "milestone"
    return "user_action"


GENERIC_EVENT_FACETS = {
    "user_introduced_aspect",
    "preference_change",
    "plan_step",
    "concern",
    "decision",
    "activity",
    "constraint",
    "request_for_comparison",
    "count_list_mention",
}


def extract_generic_event_facets(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for segment in _generic_event_segments(text):
        for facet in classify_event_facets(segment):
            label = _facet_label(segment, facet)
            if not label:
                continue
            out.append((facet, label, segment))
    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for facet, label, segment in out:
        key = (facet, _facet_key(label))
        if key in seen:
            continue
        deduped.append((facet, label, segment))
        seen.add(key)
    return deduped[:8]


def classify_event_facet(text: str) -> str | None:
    facets = classify_event_facets(text)
    return facets[0] if facets else None


def classify_event_facets(text: str) -> list[str]:
    lower = text.lower()
    if _low_value_event_segment(lower):
        return []
    out: list[str] = []
    if re.search(r"\b(?:switched|moved|changed|now prefer|now preferred|instead of|rather than)\b", lower):
        out.append("preference_change")
    if re.search(r"(?:切换|迁移|改用|改为|换成|转到|从.+到)", text):
        out.append("preference_change")
    if re.search(r"\b(?:decided|chose|picked|settled on|went with|opted for)\b", lower):
        out.append("decision")
    if re.search(r"(?:决定|选择|选定|敲定|采用)", text):
        out.append("decision")
    if re.search(r"\b(?:worried|concerned|concern|stressed|anxious|trouble|issue|problem|error|blocked|struggling)\b", lower):
        out.append("concern")
    if re.search(r"(?:担心|顾虑|问题|错误|报错|失败|卡住|阻塞|风险)", text) and not re.search(r"(?:完成|修复|解决|处理)", text):
        out.append("concern")
    if re.search(r"\b(?:always|never|must|need to|have to|required|requirement|constraint|deadline|budget|limit|only|format)\b", lower):
        out.append("constraint")
    if re.search(r"(?:总是|永远|不要|不能|必须|需要|只能|默认|以后|要求|限制|截止|预算|格式)", text):
        out.append("constraint")
    if re.search(r"\b(?:compare|compared|comparing|comparison|versus|vs\.?|between|which (?:one|option)|choose between|decide between)\b", lower):
        out.append("request_for_comparison")
    if re.search(r"(?:比较|对比|相比|哪个|哪种|二选一|选择.+还是)", text):
        out.append("request_for_comparison")
    if re.search(r"\b(?:how many|total|unique|count|number of|list(?:ed)?(?: of)?|ways to|different (?:ways|items|options|topics|calculations|movies|books|series|genres)|(?:two|three|four|five|six|seven|eight|nine|\d+)\s+(?:ways|items|options|topics|calculations|movies|books|series|genres))\b", lower):
        out.append("count_list_mention")
    if re.search(r"(?:多少|几个|总数|数量|列出|列表|清单|不同.+(?:项|种|个))", text):
        out.append("count_list_mention")
    if re.search(r"\b(?:plan|planned|planning|schedule|step|sprint|phase|timeline|roadmap)\b", lower):
        out.append("plan_step")
    if re.search(r"(?:计划|规划|步骤|阶段|排期|路线图|时间线)", text):
        out.append("plan_step")
    if re.search(r"\b(?:started|finished|completed|implemented|configured|tested|deployed|launched|created|added|fixed|reviewed|worked on|working on|trying to)\b", lower):
        out.append("activity")
    if re.search(r"(?:开始|完成|实现|配置|测试|部署|上线|创建|添加|修复|评审|处理|接入|推进|调试)", text):
        out.append("activity")
    if re.search(r"\b(?:brought up|mentioned|discussed|asked about|asked|want to|i want|i need|i'm trying|i am trying)\b", lower):
        out.append("user_introduced_aspect")
    if re.search(r"(?:提到|讨论|问到|询问|想要|想用|需要|正在|尝试)", text):
        out.append("user_introduced_aspect")
    return out


def _generic_event_segments(text: str) -> list[str]:
    normalized = re.sub(r"```.*?```", " ", text, flags=re.S)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    segments: list[str] = []
    for line in lines:
        line = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
        bullet = re.match(r"^(?:[-*+]|\d+[.)]|[A-Za-z][.)])\s+(.{8,})$", line)
        if bullet:
            segments.append(bullet.group(1).strip())
            continue
        if re.match(r"^[A-Z][A-Za-z0-9 /&+_.-]{2,60}:\s+.{6,}$", line):
            segments.append(line)
    if not segments:
        segments = [
            part.strip(" -:\n\t")
            for part in re.split(r"(?:\n{2,}|\.\s+|;\s+|\?\s+|!\s+|,\s+(?:and|but|so|because|while|also)\s+)", normalized)
            if len(part.strip(" -:\n\t")) >= 18
        ]
    if not segments and len(normalized.strip()) >= 18:
        segments = [normalized.strip()]
    return [compact_summary(segment, 260) for segment in segments[:40]]


def _low_value_event_segment(lower: str) -> bool:
    return bool(
        re.fullmatch(r"(?:ok|okay|thanks|thank you|sure|great|yes|no)[.!]?", lower.strip())
        or re.search(r"\b(?:can you help me with that|can you review this|what do you think)\??$", lower.strip())
    )


def _facet_label(text: str, facet: str) -> str:
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)]|[A-Za-z][.)])\s+", "", cleaned)
    if ":" in cleaned and len(cleaned.split(":", 1)[0].strip()) <= 70:
        prefix, rest = cleaned.split(":", 1)
        if len(prefix.strip()) >= 3:
            cleaned = prefix.strip() if facet in {"user_introduced_aspect", "plan_step"} else f"{prefix.strip()}: {rest.strip()}"
    cleaned = re.sub(
        r"^\s*(?:i(?:'m| am)?|we(?:'re| are)?)\s+(?:am\s+|are\s+)?(?:trying to|want to|need to|started|finished|completed|implemented|configured|tested|deployed|launched|created|added|fixed|reviewed|worked on|working on|planning to|plan to|decided to|chose to|worried about|concerned about)\s+",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -;.")
    terms = _topic_terms(cleaned)
    if not cleaned or not terms:
        return ""
    return cleaned[:120]


def _facet_key(label: str) -> str:
    terms = [term for term in _topic_terms(label) if term not in {"trying", "want", "need", "help"}]
    return "-".join(terms[:6])


def _facet_description(facet: str, label: str, text: str) -> str:
    facet_label = facet.replace("_", " ")
    return f"Facet [{facet}]: {facet_label}. Label: {compact_summary(label, 90)}. Evidence: {compact_summary(text, 220)}"


def classify_milestone(text: str) -> str | None:
    milestones = classify_milestones(text)
    if "deployment_and_test_improvements" in milestones:
        return "deployment_and_test_improvements"
    return milestones[0] if milestones else None


def extract_milestone_mentions(text: str) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    segments = _milestone_segments(text)
    for segment in segments:
        for milestone in classify_milestones(segment):
            mentions.append((milestone, segment))
    if not mentions and len(text) <= 500:
        mentions = [(milestone, text) for milestone in classify_milestones(text)]
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for milestone, segment in mentions:
        key = (milestone, compact_summary(segment, 80).lower())
        if key in seen:
            continue
        out.append((milestone, segment))
        seen.add(key)
    return out


def _milestone_segments(text: str) -> list[str]:
    normalized = re.sub(r"```.*?```", " ", text, flags=re.S)
    rough_parts = re.split(
        r"(?:\n{2,}|\n[-*]\s+|\.\s+|;\s+|\?\s+|!\s+|,\s+(?:and|but|so|because|while|also)\s+)",
        normalized,
    )
    parts: list[str] = []
    for rough in rough_parts:
        rough = rough.strip(" -:\n\t")
        if len(rough) < 20:
            continue
        parts.extend(_split_long_segment(rough))
    if not parts:
        parts = [text.strip()]
    return [compact_summary(part, 260) for part in parts[:40]]


def _split_long_segment(text: str, *, max_words: int = 45, overlap: int = 8) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out: list[str] = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if len(chunk) >= 20:
            out.append(chunk)
    return out


def classify_milestones(text: str) -> list[str]:
    lower = text.lower()
    out: list[str] = []
    deployment_specific = _has_any(
        lower,
        [
            "render.com",
            "on render",
            "render deployment",
            "render service",
            "gunicorn",
            "port 10000",
            "worker setup",
            "worker configuration",
            "deployment setting",
            "deployment settings",
            "deployment config",
            "deployment configuration",
            "environment variable",
            "environment variables",
            "database_url",
            "secret_key",
            "production server",
            "hosting",
        ],
    )
    deployment_generic = _has_any(lower, ["deployment", "deploy"]) and _has_any(
        lower,
        ["configure", "configured", "configuring", "configuration", "settings", "server", "hosting", "render.com", "gunicorn"],
    )
    test_context = _has_any(lower, ["integration test", "integration tests", "test suite", "coverage", "endpoint coverage", "additional tests"])
    deployment_test_improvement = _has_any(lower, ["deployment", "deploy"]) and test_context and _has_any(
        lower,
        ["improve", "improving", "expanded", "expanding", "additional", "review", "reviewing"],
    )
    deployment_context = deployment_specific or deployment_generic or deployment_test_improvement
    security_context = _has_any(lower, ["security hardening", "security", "authentication", "authorization", "argon2", "password hashing"])
    core_context = _has_any(lower, ["core functionality", "mvp scope", "income/expense tracking", "user login", "basic analytics"])
    setup_debug_context = _has_any(lower, ["templatenotfound", "operationalerror", "no such table", "setup error", "set up my flask app"])
    if deployment_context:
        out.append("deployment_configuration")
    if deployment_context and (test_context or _has_any(lower, ["improve", "improving", "expanded", "expanding", "additional"])):
        out.append("deployment_and_test_improvements")
    if test_context:
        out.append("integration_test_coverage")
    transaction_impl_context = _has_any(
        lower,
        [
            "implement",
            "implemented",
            "working on",
            "create",
            "creating",
            "optimize",
            "optimizing",
            "post /transactions",
            "transaction post",
            "create_transaction",
            "crud endpoint",
            "crud endpoints",
            "crud in my flask app",
            "rest api for transactions",
        ],
    )
    if _has_any(lower, ["transaction crud", "post /transactions", "transaction post", "create_transaction"]) and (
        transaction_impl_context or not test_context
    ):
        out.append("transaction_crud_implementation")
    if setup_debug_context:
        out.append("setup_debugging")
    if not setup_debug_context and "transaction" in lower and _has_any(
        lower,
        ["try-except", "exception", "exceptions", "error", "errors", "error handling", "meaningful error", "validation"],
    ):
        out.append("transaction_error_handling")
    if "transaction creation" in lower:
        out.append("transaction_crud_implementation")
    if "transaction" in lower and _has_any(lower, ["error", "response", "validation", "handling", "201 status"]):
        out.append("transaction_crud_implementation")
    if _has_any(lower, ["initial project", "set up the database", "database schema", "local server"]):
        out.append("initial_project_setup")
    if core_context:
        out.append("core_functionality")
    if security_context and not core_context and not test_context:
        out.append("security_and_deployment" if _has_any(lower, ["deployment", "deploy", "launch"]) else "security_auth")
    return list(dict.fromkeys(out))


def _milestone_description(group: str | None, text: str) -> str:
    if not group:
        return compact_summary(text, 240)
    label = group.replace("_", " ")
    return f"Milestone [{group}]: {label}. Evidence: {compact_summary(text, 220)}"


def _has_any(lower: str, phrases: list[str]) -> bool:
    return any(phrase in lower for phrase in phrases)


FACT_TOPIC_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "before",
    "between",
    "can",
    "could",
    "for",
    "from",
    "have",
    "help",
    "how",
    "into",
    "like",
    "need",
    "now",
    "project",
    "should",
    "that",
    "the",
    "this",
    "want",
    "with",
    "you",
    "以及",
    "以后",
    "使用",
    "修改",
    "决定",
    "可以",
    "因为",
    "如果",
    "我们",
    "我的",
    "现在",
    "用户",
    "这个",
}

ZH_TOPIC_TERMS = (
    "中文",
    "回复",
    "报表",
    "检索",
    "部署",
    "端口",
    "配置",
    "问题",
    "数据库",
    "记忆",
    "偏好",
    "切换",
    "迁移",
    "决定",
    "完成",
    "修复",
    "默认",
)


def _fact_polarity(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(?:never|not yet|haven['’]?t|have not|don['’]?t|do not|without)\b", lower):
        return "negative"
    if re.search(r"(?:不要|别|不能|不再|未|没有)", text):
        return "negative"
    if re.search(r"\b(?:used|integrated|worked|read|listened|met|downloaded|placed|attended|completed|tested|made|drafted|managed|started)\b", lower):
        return "positive"
    if re.search(r"(?:用了|使用|集成|完成|测试|部署|修复|开始|实现)", text):
        return "positive"
    return "unknown"


def _value_mentions(text: str) -> list[str]:
    values: list[str] = []
    for pattern in [
        r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b",
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:ms|s|sec|seconds?|days?|weeks?|months?|commits?|cards?|columns?|features?|%)\b",
        r"\bv?\d+\.\d+(?:\.\d+)?\b",
    ]:
        values.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.I))
        if len(values) >= 12:
            break
    return list(dict.fromkeys(values[:12]))


def _topic_terms(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())
    terms: list[str] = []
    for token in raw:
        token = token.strip("_+-")
        if len(token) < 4 or token in FACT_TOPIC_STOPWORDS:
            continue
        terms.append(token[:-1] if token.endswith("s") and len(token) > 5 else token)
    for term in ZH_TOPIC_TERMS:
        if term in text and term not in FACT_TOPIC_STOPWORDS:
            terms.append(term)
    return list(dict.fromkeys(terms[:12]))


def _clean_zh_object(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" ，,。.!?！？")
    cleaned = re.sub(r"^(?:我|我们|用户)?(?:现在|目前|以后|默认)?(?:更?喜欢|偏好|倾向于|想用|要用|希望用|打算用|准备用|采用|使用|用)\s*", "", cleaned)
    return cleaned.strip(" ，,。.!?！？")
