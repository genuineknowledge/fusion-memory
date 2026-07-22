from __future__ import annotations

import re
from typing import Any

from fusion_memory.core.models import Candidate, EvidenceSpan
from fusion_memory.retrieval.rule_registry import RuleDefinition, record_rule_hit, register_rule
from fusion_memory.retrieval.aggregation_keys import (
    combinatorics_aggregation_keys,
    generic_aggregation_keys,
    generic_list_candidate_keys,
    is_combinatorics_aggregation_query as _is_combinatorics_aggregation_query,
    is_generic_count_or_list_query,
    is_stress_break_aggregation_query as _is_stress_break_aggregation_query,
    stress_break_aggregation_keys,
    vendor_tool_aggregation_keys,
)


register_rule(
    RuleDefinition(
        rule_id="exact_match.cjk_phrase",
        module=__name__,
        purpose="mark Chinese exact phrase preservation hits",
        category="high_risk",
        pattern="cjk_exact_phrase",
        ability="zh_recall",
        protected=True,
        protected_reason="chinese_recall_precision",
    )
)

register_rule(
    RuleDefinition(
        rule_id="zh_recall.cjk_exact_match",
        module=__name__,
        purpose="Observe CJK exact phrase preservation without storing the phrase.",
        category="zh_recall",
        ability="zh_recall",
    )
)

register_rule(
    RuleDefinition(
        rule_id="multi_condition.query_token_match",
        module=__name__,
        purpose="Observe multi-condition query token matching without changing retrieval behavior.",
        category="multi_condition",
        ability="multi_condition",
    )
)

def _broad_recall_candidate_allowed(query: str, plan: Any, candidate: Candidate) -> bool:
    if getattr(plan, "query_type", None) != "summarization":
        return True
    text = candidate.text or ""
    anchor_phrases = _topic_anchor_phrases_for_service(query)
    if not anchor_phrases:
        return True
    text_lower = text.lower()
    text_terms = _topic_scope_tokens(text)
    for phrase in anchor_phrases:
        phrase_terms = {term for term in phrase.split() if term}
        if len(phrase_terms) < 2:
            continue
        if phrase in text_lower:
            return True
        if len(phrase_terms & text_terms) >= 2:
            return True
    return False


def _replaceable_low_synthesis_index(candidates: list[Candidate]) -> int | None:
    best_index: int | None = None
    best_key: tuple[float, float, float] | None = None
    for index, candidate in enumerate(candidates):
        if _aggregation_context_support_candidate(candidate) and _high_value_aggregation_context_support(candidate):
            continue
        synthesis = float(candidate.scores.get("synthesis_signal", 0.0) or 0.0)
        is_user = candidate.metadata.get("speaker") == "user"
        key = (
            1.0 if not is_user else 0.0,
            1.0 - min(1.0, synthesis),
            -float(candidate.scores.get("utility_score", 0.0) or 0.0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    return best_index


def _dedupe_event_ordering_support_events(candidates: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen_groups: set[str] = set()
    seen_primary_spans: set[str] = set()
    for candidate in candidates:
        group = str(candidate.metadata.get("milestone_group") or candidate.metadata.get("event_type") or candidate.id)
        if group in seen_groups:
            continue
        primary_span = str(next(iter(candidate.source_span_ids), ""))
        if primary_span and primary_span in seen_primary_spans:
            continue
        out.append(candidate)
        seen_groups.add(group)
        if primary_span:
            seen_primary_spans.add(primary_span)
    return out


TOPIC_SCOPE_STOPWORDS = {
    "answer",
    "about",
    "across",
    "after",
    "also",
    "and",
    "application",
    "aspect",
    "aspects",
    "based",
    "been",
    "before",
    "between",
    "brought",
    "can",
    "conversation",
    "conversations",
    "before",
    "concern",
    "concerns",
    "challenge",
    "challenges",
    "comprehensive",
    "currently",
    "deadline",
    "deadlines",
    "different",
    "developed",
    "development",
    "does",
    "ever",
    "feature",
    "features",
    "final",
    "finish",
    "finished",
    "finishing",
    "for",
    "from",
    "give",
    "have",
    "happened",
    "help",
    "how",
    "include",
    "including",
    "information",
    "into",
    "item",
    "items",
    "key",
    "list",
    "made",
    "make",
    "management",
    "many",
    "mention",
    "mentioned",
    "need",
    "new",
    "only",
    "order",
    "our",
    "over",
    "personal",
    "previous",
    "project",
    "projects",
    "question",
    "request",
    "requests",
    "say",
    "said",
    "should",
    "so",
    "summary",
    "target",
    "targets",
    "the",
    "three",
    "through",
    "throughout",
    "used",
    "using",
    "want",
    "wanted",
    "walk",
    "way",
    "ways",
    "week",
    "weeks",
    "were",
    "which",
    "with",
    "work",
    "worked",
    "you",
}

TOPIC_SCOPE_EQUIVALENTS = {
    "auth": {"auth", "authentication", "login", "logout", "session"},
    "authentication": {"auth", "authentication", "login", "logout", "session"},
    "columns": {"column", "columns", "field", "fields"},
    "deadline": {"deadline", "deadlines", "due", "target"},
    "deployment": {"deployment", "deploy", "deployed", "launch", "production", "render", "gunicorn"},
    "features": {"feature", "features", "module", "modules", "functionality"},
    "estate": {"estate", "will", "probate", "assets", "asset", "property", "trust"},
    "assets": {"assets", "asset", "items", "property", "home", "vehicle", "equipment", "safe", "will"},
    "items": {"items", "assets", "asset", "property", "home", "vehicle", "equipment", "safe", "will"},
    "financial": {"financial", "finance", "budget", "money", "cost", "costs", "income", "expense", "expenses"},
    "finish": {"finish", "finished", "complete", "completed", "completion", "end", "ended"},
    "latency": {"latency", "response", "time", "ms", "milliseconds"},
    "profession": {"profession", "job", "career", "role", "work"},
    "sprint": {"sprint", "sprints", "phase", "milestone"},
    "stress": {"stress", "stressed", "burnout", "overwhelmed", "workload"},
    "transaction": {"transaction", "transactions", "crud", "income", "expense", "expenses"},
}


def _topic_scope_group_limit(query_type: str) -> int:
    if query_type in {"event_ordering", "temporal_lookup", "summarization"}:
        return 1
    if query_type == "multi_session_reasoning":
        return 1
    if query_type in {"contradiction_resolution", "knowledge_update"}:
        return 2
    return 1


def _topic_scope_groups(query: str, plan: Any, spans: list[EvidenceSpan], *, max_groups: int) -> set[str]:
    query_tokens = _topic_scope_tokens(query)
    if not query_tokens:
        return set()
    anchor_phrases = _topic_anchor_phrases_for_service(query)
    group_scores: dict[str, float] = {}
    group_top_scores: dict[str, list[float]] = {}
    group_tokens: dict[str, set[str]] = {}
    group_hits: dict[str, int] = {}
    group_anchor_scores: dict[str, float] = {}
    for span in spans:
        group = _span_group_key(span)
        if not group:
            continue
        anchor_score = _topic_anchor_score_for_service(query, span.content, anchor_phrases=anchor_phrases)
        score = _topic_scope_score(query, span.content, plan)
        if getattr(plan, "query_type", None) == "event_ordering" and anchor_score > 0:
            score = max(score, anchor_score)
        if score <= 0.04:
            continue
        tokens = _topic_scope_tokens(span.content)
        group_tokens.setdefault(group, set()).update(tokens)
        group_hits[group] = group_hits.get(group, 0) + 1
        group_top_scores.setdefault(group, []).append(min(score, 0.75))
        group_scores[group] = max(group_scores.get(group, 0.0), score)
        group_anchor_scores[group] = max(group_anchor_scores.get(group, 0.0), anchor_score)
    if not group_scores:
        return set()
    ranked: list[tuple[str, float]] = []
    for group, score in group_scores.items():
        coverage = len(query_tokens & group_tokens.get(group, set())) / max(1, len(query_tokens))
        top_scores = sorted(group_top_scores.get(group, []), reverse=True)[:8]
        top_mass = sum(top_scores) / max(1, len(top_scores))
        density = min(0.16, 0.015 * min(group_hits.get(group, 0), 12))
        anchor = group_anchor_scores.get(group, 0.0)
        ranked.append((group, (0.30 * score) + (0.30 * top_mass) + (0.75 * coverage) + (0.95 * anchor) + density))
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < 0.30:
        return set()
    selected = {ranked[0][0]}
    for group, score in ranked[1:max_groups]:
        if score >= max(0.35, ranked[0][1] * 0.72):
            selected.add(group)
    return selected


def _topic_scope_score(query: str, text: str, plan: Any | None = None) -> float:
    query_tokens = _topic_scope_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _topic_scope_tokens(text)
    if not text_tokens:
        return 0.0
    expanded_query = _expand_topic_tokens(query_tokens)
    expanded_text = _expand_topic_tokens(text_tokens)
    direct = len(query_tokens & text_tokens) / max(1, len(query_tokens))
    expanded = len(expanded_query & expanded_text) / max(1, len(expanded_query))
    phrase_bonus = _topic_phrase_bonus(query, text)
    value_bonus = 0.0
    query_lower = query.lower()
    text_lower = text.lower()
    if _has_value_intent(query_lower) and _compatible_value_mention(query_lower, text_lower):
        value_bonus = 0.08
    if getattr(plan, "query_type", None) == "temporal_lookup" and _date_signal(text) > 0:
        value_bonus += 0.08
    return min(1.0, (0.62 * direct) + (0.26 * expanded) + phrase_bonus + value_bonus)


def _exact_overlap_score(query: str, text: str) -> float:
    query_tokens = _topic_scope_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _topic_scope_tokens(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / max(1, len(query_tokens))


def _surface_claim_polarity(query: str, text: str) -> str:
    lower = text.lower()
    query_tokens = _topic_scope_tokens(query)
    if query_tokens and len(query_tokens & _topic_scope_tokens(text)) == 0:
        return "uncertain"
    negative_patterns = [
        r"\bnever\b",
        r"\bnot\s+(?:yet\s+)?(?:used|integrated|worked|read|listened|met|downloaded|placed|attended|completed|tested|made|drafted|managed)\b",
        r"\bhaven['’]?t\b",
        r"\bhave\s+not\b",
        r"\bno\s+experience\b",
        r"\bwithout\s+(?:using|having|integrating|testing)\b",
    ]
    if any(re.search(pattern, lower) for pattern in negative_patterns):
        return "negative"
    positive_patterns = [
        r"\b(?:used|integrated|worked|read|listened|met|downloaded|placed|attended|completed|tested|made|drafted|managed)\b",
        r"\bstarted\s+(?:using|listening|reading|working|testing)\b",
        r"\b(?:have|has|had|i['’]?ve|we['’]?ve)\s+been\s+(?:using|tracking|reading|listening|working|testing|attending|meeting|drafting|managing)\b",
        r"\b(?:am|are|is|was|were)\s+(?:using|tracking|reading|listening|working|testing|attending|meeting|drafting|managing)\b",
        r"\bhas\s+been\s+(?:used|integrated|tested|completed)\b",
        r"\balready\s+(?:used|integrated|tested|completed|started|drafted)\b",
    ]
    if any(re.search(pattern, lower) for pattern in positive_patterns):
        return "positive"
    return "uncertain"


def _topic_phrase_bonus(query: str, text: str) -> float:
    query_lower = query.lower()
    text_lower = text.lower()
    phrases = []
    for match in re.finditer(r"\b([a-z0-9]+(?:\s+[a-z0-9]+){1,3})\b", query_lower):
        phrase = match.group(1)
        terms = [term for term in phrase.split() if term not in TOPIC_SCOPE_STOPWORDS and len(term) >= 3]
        if len(terms) >= 2:
            phrases.append(" ".join(terms))
    hits = sum(1 for phrase in dict.fromkeys(phrases) if phrase in text_lower)
    return min(0.18, 0.06 * hits)


def _topic_anchor_phrases_for_service(query: str) -> list[str]:
    lower = query.lower()
    out: list[str] = []
    patterns = [
        r"\b(?:my|the|this|that)\s+([a-z0-9][a-z0-9 +#./-]*(?:app|application|website|tracker|dashboard|feature|project|system|tool|api|bot|portfolio))\b",
        r"\b(?:implementing|developing|building|creating|setting up|working on)\s+(?:my|the|this|that)?\s*([a-z0-9][a-z0-9 +#./-]{4,80})\b",
        r"\b(?:aspects of|features of|concerns about)\s+(?:my|the|this|that)?\s*([a-z0-9][a-z0-9 +#./-]{4,80})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            phrase = _clean_topic_anchor_for_service(match.group(1))
            if phrase:
                out.append(phrase)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", lower)
        if len(token) >= 4 and token not in TOPIC_SCOPE_STOPWORDS
    ]
    for size in (3, 2):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[index : index + size])
            if phrase and phrase not in out:
                out.append(phrase)
    return list(dict.fromkeys(out))[:12]


def _clean_topic_anchor_for_service(value: str) -> str:
    value = re.sub(
        r"\b(?:throughout|across|in order|only|mention|different|aspects?|features?|conversations?|sessions?)\b.*$",
        "",
        value,
        flags=re.I,
    )
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", value.lower())
        if len(term) >= 3 and term not in TOPIC_SCOPE_STOPWORDS
    ]
    if len(terms) < 2:
        return ""
    return " ".join(terms[:5])


def _topic_anchor_score_for_service(query: str, text: str, *, anchor_phrases: list[str]) -> float:
    lower = text.lower()
    text_terms = _topic_scope_tokens(text)
    score = 0.0
    for phrase in anchor_phrases:
        terms = set(phrase.split())
        if not terms:
            continue
        if phrase in lower:
            score += 0.70 + 0.08 * min(len(terms), 4)
            continue
        overlap = len(terms & text_terms) / max(1, len(terms))
        if overlap >= 0.67 and len(terms & text_terms) >= 2:
            score += 0.28 + 0.28 * overlap
    distinctive = _topic_scope_tokens(query)
    if distinctive:
        score += min(0.34, 0.10 * len(distinctive & text_terms))
    if re.search(r"\b(?:i(?:'m| am)?|we(?:'re| are)?)\s+(?:building|developing|implementing|creating|trying to|working on|setting up)\b", lower):
        score += 0.12
    return min(1.0, score)


def _topic_scope_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", text.lower())
    tokens: set[str] = set()
    for token in raw:
        if re.search(r"[\u4e00-\u9fff]", token):
            tokens.add(token)
            for size in (2, 3, 4):
                tokens.update(token[index : index + size] for index in range(0, max(0, len(token) - size + 1)))
            continue
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


def _expand_topic_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        equivalents = TOPIC_SCOPE_EQUIVALENTS.get(token)
        if equivalents:
            expanded.update(equivalents)
    return expanded


def _span_group_key(span: EvidenceSpan) -> str:
    for value in (span.source_uri, span.turn_id):
        if not value:
            continue
        text = str(value)
        match = re.match(r"^(beam:[^:]+:\d+):", text)
        if match:
            return match.group(1)
        if "#" in text:
            return text.split("#", 1)[0]
    return span.scope.session_id or span.scope.run_id or span.scope.workspace_id or ""


def _date_signal(text: str) -> float:
    lower = text.lower()
    if re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", lower):
        return 1.0
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b", lower):
        return 0.9
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?)\b", lower):
        return 0.55
    return 0.0


def _temporal_target_roles_for_service(query: str) -> set[str]:
    lower = query.lower()
    roles: set[str] = set()
    if "deployment" in lower or "deploy" in lower or "launch" in lower:
        roles.add("deployment_deadline")
    if re.search(r"\bfinish|finishing|complete|completed|completion|features?\b", lower):
        roles.add("feature_finish_date")
    if re.search(r"\b(?:finish|finished|finishing|complete|completed|completion|done|read|reading)\b", lower):
        roles.add("completion_date")
    if re.search(r"\b(?:downloaded?|installed|borrowed|checked out|acquired)\b", lower):
        roles.add("download_date")
    if re.search(r"\b(?:decided|decision|reject|rejected|decline|declined|chose)\b", lower):
        roles.add("decision_date")
    if re.search(r"\b(?:rescheduled|reschedule|moved|pushed|postponed)\b", lower):
        roles.add("reschedule_date")
    if "sprint" in lower and re.search(r"\bend|first\b", lower):
        roles.add("sprint_end_date")
    if re.search(r"\bstart|starts|started|begin|began|from\b", lower):
        roles.add("start_date")
    if re.search(r"\bfinish|finished|complete|completed|completion|done\b", lower):
        roles.add("completion_date")
    if re.search(r"\bdownload(?:ed|ing)?|borrow(?:ed|ing)?|checked out\b", lower):
        roles.add("download_date")
    if re.search(r"\bdecided|decision|chose|reject(?:ed|ing)?|declin(?:ed|e|ing)\b", lower):
        roles.add("decision_date")
    if re.search(r"\brescheduled|reschedule|moved|pushed|postponed\b", lower):
        roles.add("reschedule_date")
    return roles


def _temporal_roles_in_text(query: str, text: str) -> set[str]:
    lower = text.lower()
    roles: set[str] = set()
    query_terms = _topic_scope_tokens(query)
    text_terms = _topic_scope_tokens(text)
    overlap = len(query_terms & text_terms)
    if ("deployment" in lower or "deploy" in lower or "launch" in lower or "production" in lower) and (
        "deadline" in lower or "by " in lower or "target" in lower or _date_signal(lower)
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
    if re.search(r"\bfinish|finished|complete|completed|completion|done\b", lower):
        roles.add("completion_date")
    if re.search(r"\bdownload(?:ed|ing)?|borrow(?:ed|ing)?|checked out\b", lower):
        roles.add("download_date")
    if re.search(r"\bdecided|decision|chose|reject(?:ed|ing)?|declin(?:ed|e|ing)\b", lower):
        roles.add("decision_date")
    if re.search(r"\brescheduled|reschedule|moved|pushed|postponed\b", lower):
        roles.add("reschedule_date")
    return roles


def _temporal_focus_terms_for_service(query: str) -> set[str]:
    role_words = {
        "after",
        "before",
        "between",
        "date",
        "days",
        "decid",
        "decide",
        "decided",
        "decision",
        "decline",
        "declined",
        "download",
        "downloaded",
        "finish",
        "finished",
        "complete",
        "completed",
        "completion",
        "long",
        "many",
        "meet",
        "more",
        "myself",
        "passed",
        "pass",
        "reschedule",
        "rescheduled",
        "reschedul",
        "start",
        "started",
        "take",
        "took",
        "time",
        "when",
    }
    return {
        token
        for token in _topic_scope_tokens(query)
        if token not in role_words and not token.isdigit()
    }


def _exact_query_terms(query: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9_./#:-]+", query.lower())
    terms: list[str] = []
    for token in raw:
        normalized = token.strip(".,;:!?()[]{}")
        if len(normalized) < 3:
            continue
        if normalized in EVENT_ORDERING_STOPWORDS or normalized in {"what", "when", "many", "much", "does", "have", "between"}:
            continue
        terms.append(normalized)
        if "_" in normalized:
            terms.extend(part for part in normalized.split("_") if len(part) >= 3)
    return list(dict.fromkeys(terms[:16]))


def _aggregation_query_terms(query: str) -> set[str]:
    terms = set(_topic_scope_tokens(query))
    expanded = set(terms)
    for token in terms:
        expanded.update(TOPIC_SCOPE_EQUIVALENTS.get(token, set()))
    return expanded


def _broad_raw_recall_queries(query: str, plan: Any, *, limit: int = 8) -> list[str]:
    ordered = _ordered_topic_scope_tokens(query)
    intent = getattr(plan, "intent", {}) or {}
    queries: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(_ordered_topic_scope_tokens(value))
        if len(cleaned.split()) >= 2 and cleaned not in queries:
            queries.append(cleaned)

    add(query)
    target_terms = _intent_string_list(intent.get("target_terms"))
    object_types = _intent_string_list(intent.get("object_types"))
    entities = [str(value) for value in getattr(plan, "entities", []) if str(value).strip()]
    for phrase in [*target_terms[:6], *object_types[:4], *entities[:4]]:
        add(phrase)
    if target_terms and object_types:
        add(" ".join([*target_terms[:4], *object_types[:2]]))
    aggregation = intent.get("aggregation") if isinstance(intent.get("aggregation"), dict) else {}
    aggregation_terms = _intent_string_list(aggregation.get("target_terms")) + _intent_string_list(aggregation.get("unit_terms"))
    if aggregation_terms:
        add(" ".join(aggregation_terms[:6]))
    temporal = intent.get("temporal") if isinstance(intent.get("temporal"), dict) else {}
    temporal_terms = _intent_string_list(temporal.get("endpoint_roles")) + _intent_string_list(temporal.get("time_expressions"))
    if temporal_terms:
        add(" ".join([*ordered[:5], *temporal_terms[:3]]))
    if getattr(plan, "retrieval_hints", None):
        add(" ".join(str(hint) for hint in plan.retrieval_hints if hint))
    if getattr(plan, "query_type", "") in {"preference", "instruction"}:
        for terms in _preference_recall_terms(query):
            add(" ".join([*ordered[:4], *terms]))
    if getattr(plan, "query_type", "") in {"knowledge_update", "contradiction_resolution"}:
        add(" ".join([*ordered[:5], "changed updated current previous"]))
    if getattr(plan, "query_type", "") == "multi_session_reasoning":
        add(" ".join([*ordered[:5], "across sessions total different"]))
    if getattr(plan, "query_type", "") == "summarization":
        add(" ".join([*ordered[:6], "summary overview discussed"]))
    if getattr(plan, "query_type", "") in {"event_ordering", "temporal_lookup"}:
        add(" ".join([*ordered[:5], "first then before after"]))
    for size in (5, 4, 3):
        if len(ordered) >= size:
            add(" ".join(ordered[:size]))
    return queries[:limit]


def _intent_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _intent_recall_signal(query: str, plan: Any, text: str) -> float:
    intent = getattr(plan, "intent", {}) or {}
    text_terms = _expand_topic_tokens(_topic_scope_tokens(text))
    if not text_terms:
        return 0.0
    signal = 0.0
    target_terms = _topic_scope_tokens(" ".join(_intent_string_list(intent.get("target_terms"))))
    object_types = _topic_scope_tokens(" ".join(_intent_string_list(intent.get("object_types"))))
    if target_terms:
        signal += min(0.38, 0.38 * len(_expand_topic_tokens(target_terms) & text_terms) / max(1, len(_expand_topic_tokens(target_terms))))
    if object_types:
        signal += min(0.18, 0.18 * len(_expand_topic_tokens(object_types) & text_terms) / max(1, len(_expand_topic_tokens(object_types))))
    aggregation = intent.get("aggregation") if isinstance(intent.get("aggregation"), dict) else {}
    if aggregation.get("operation") not in {None, "", "none"}:
        aggregation_terms = _topic_scope_tokens(" ".join(_intent_string_list(aggregation.get("target_terms")) + _intent_string_list(aggregation.get("unit_terms"))))
        if aggregation_terms:
            signal += min(0.24, 0.24 * len(_expand_topic_tokens(aggregation_terms) & text_terms) / max(1, len(_expand_topic_tokens(aggregation_terms))))
        signal += 0.12 * _aggregation_signal(query, text, _aggregation_query_terms(query))
    temporal = intent.get("temporal") if isinstance(intent.get("temporal"), dict) else {}
    if temporal.get("requires_time") or temporal.get("requires_order") or temporal.get("requires_duration"):
        if _date_signal(text) > 0:
            signal += 0.14
        if _temporal_roles_in_text(query, text):
            signal += 0.18
    if intent.get("needs_current_state") or getattr(plan, "query_type", "") == "knowledge_update":
        lower = text.lower()
        if re.search(r"\b(?:now|currently|current|latest|updated|changed|switched|instead|previously|used to)\b", lower):
            signal += 0.18
    if intent.get("needs_conflict_check") or getattr(plan, "query_type", "") == "contradiction_resolution":
        if _surface_claim_polarity(query, text) in {"positive", "negative"}:
            signal += 0.14
    if getattr(plan, "query_type", "") in {"preference", "instruction"}:
        signal += _preference_recall_signal(query, text)
    if getattr(plan, "query_type", "") == "summarization":
        signal += min(0.12, 0.02 * len(_topic_scope_tokens(query) & _topic_scope_tokens(text)))
    return max(0.0, min(1.0, signal))


def _preference_recall_terms(query: str) -> list[list[str]]:
    lower = query.lower()
    terms: list[list[str]] = [["prefer", "preference", "constraint", "requirement"]]
    if re.search(r"\b(?:schedule|sessions?|work sessions?|time|timing|routine|breaking up|break up|pace|pacing|burn(?:ing)? out|fatigue)\b", lower):
        terms.append(["prefer", "short", "session", "minute", "burst", "marathon", "burnout"])
        terms.append(["schedule", "pace", "breaks", "focused", "avoid", "burnout"])
    if re.search(r"\b(?:edit|editing|draft|revision|writing)\b", lower):
        terms.append(["editing", "writing", "sessions", "short", "minutes", "tools"])
    if re.search(r"\b(?:choose|candidate|responsibilities|executor|appoint)\b", lower):
        terms.append(["candidate", "responsible", "reliable", "organized", "best", "fit"])
    return terms[:4]


def _preference_recall_signal(query: str, text: str) -> float:
    query_lower = query.lower()
    lower = text.lower()
    signal = 0.0
    if re.search(r"\b(?:prefer|preference|rather than|instead of|avoid|like|want|looking for|important to me)\b", lower):
        signal += 0.12
    if re.search(r"\b(?:schedule|sessions?|routine|timing|pace|pacing|breaks?|burnout|fatigue|marathon|focused)\b", query_lower):
        if re.search(r"\b(?:short bursts?|shorter intervals?|30\s*-?\s*minutes?|minutes?\s+at\s+a\s+time|rather than marathon|marathon sessions?|burnout|breaks?)\b", lower):
            signal += 0.24
        if re.search(r"\b(?:session|sessions|schedule|routine|pace|pacing)\b", lower):
            signal += 0.08
    if re.search(r"\b(?:edit|editing|draft|revision|writing)\b", query_lower) and re.search(r"\b(?:edit|editing|draft|revision|writing)\b", lower):
        signal += 0.08
    if re.search(r"\b(?:choose|candidate|responsibilities|executor|appoint)\b", query_lower) and re.search(r"\b(?:organized|organizational|reliable|responsible|best fit|candidate)\b", lower):
        signal += 0.18
    return min(0.40, signal)


def _scent_trail_queries(query: str, seed_texts: list[str], *, limit: int = 4) -> list[str]:
    tokens: list[str] = []
    for text in [query, *seed_texts[:6]]:
        for token in _ordered_topic_scope_tokens(text):
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= 12:
                break
        if len(tokens) >= 12:
            break
    if not tokens:
        return []
    query_lower = query.lower()
    prefixes: list[list[str]] = [tokens[:4]]
    if len(tokens) > 4:
        prefixes.append(tokens[2:6])
    if len(tokens) > 6:
        prefixes.append(tokens[4:8])
    if len(tokens) > 8:
        prefixes.append(tokens[6:10])
    queries: list[str] = []
    for prefix in prefixes[:limit]:
        phrase = " ".join(prefix).strip()
        if len(prefix) >= 2 and phrase and phrase not in queries:
            queries.append(phrase)
    if "version" in query_lower or "library" in query_lower or "dependency" in query_lower:
        version_terms = [
            token
            for token in tokens
            if re.search(r"\b\d+(?:\.\d+){1,3}\b", token)
            or token in {"version", "versions", "library", "libraries", "dependency", "dependencies"}
        ]
        for prefix in prefixes[:limit]:
            augmented = " ".join([*prefix, "version"]).strip()
            if augmented and augmented not in queries:
                queries.insert(0, augmented)
        if version_terms:
            phrase = " ".join(dict.fromkeys(version_terms[:4]))
            if phrase and phrase not in queries:
                queries.insert(0, phrase)
    return queries[:limit]


def _ordered_topic_scope_tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", text.lower())
    tokens: list[str] = []
    seen: set[str] = set()
    for token in raw:
        if re.search(r"[\u4e00-\u9fff]", token):
            for size in (len(token), 4, 3, 2):
                if size <= 1 or len(token) < size:
                    continue
                for index in range(0, len(token) - size + 1):
                    variant = token[index : index + size]
                    if variant not in seen:
                        tokens.append(variant)
                        seen.add(variant)
            continue
        token = token.strip("_+-")
        if len(token) < 3 or token in TOPIC_SCOPE_STOPWORDS:
            continue
        variants = [token]
        if token.endswith("s") and len(token) > 4:
            variants.append(token[:-1])
        if token.endswith("ing") and len(token) > 6:
            variants.append(token[:-3])
        if token.endswith("ed") and len(token) > 5:
            variants.append(token[:-2])
        for variant in variants:
            if variant not in seen:
                tokens.append(variant)
                seen.add(variant)
    return tokens


def _cjk_exact_match_phrases(query: str, text: str, *, min_len: int = 2) -> list[str]:
    query_tokens = [token for token in _ordered_topic_scope_tokens(query) if re.search(r"[\u4e00-\u9fff]", token) and len(token) >= min_len]
    if not query_tokens:
        return []
    matches = [token for token in query_tokens if token in text]
    unique_matches = list(dict.fromkeys(matches))
    if unique_matches:
        record_rule_hit(
            "exact_match.cjk_phrase",
            query=query,
            text=text,
            stage="exact_filter",
            metadata={"decision": "preserve_language_exact_match", "source": "cjk_exact", "match_count": len(unique_matches)},
        )
        record_rule_hit(
            "zh_recall.cjk_exact_match",
            query=query,
            text=text,
            stage="exact_filter",
            impact="observed",
            metadata={"decision": "observed", "source": "cjk_exact", "match_count": len(unique_matches)},
        )
    return unique_matches


def _matched_query_conditions(query: str, text: str, *, min_len: int = 3) -> list[str]:
    query_tokens = [token for token in _ordered_topic_scope_tokens(query) if len(token) >= min_len]
    if not query_tokens:
        return []
    text_tokens = _expand_topic_tokens(_topic_scope_tokens(text))
    matched = [token for token in query_tokens if token in text_tokens]
    unique_matches = list(dict.fromkeys(matched))
    if unique_matches:
        record_rule_hit(
            "multi_condition.query_token_match",
            query=query,
            text=text,
            stage="search_filter",
            impact="observed",
            metadata={"decision": "observed", "source": "multi_condition", "match_count": len(unique_matches)},
        )
    return unique_matches


def _scent_trail_score(query: str, text: str) -> float:
    query_tokens = _topic_scope_tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _topic_scope_tokens(text)
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
    phrase_bonus = 0.0
    query_lower = query.lower()
    text_lower = text.lower()
    if re.search(r"\b(?:version|library|libraries|dependency|dependencies)\b", query_lower):
        version_hits = len(re.findall(r"\b\d+(?:\.\d+){1,3}\b", text_lower))
        if version_hits:
            phrase_bonus += min(0.25, 0.08 * version_hits)
        if any(term in text_lower for term in ["flask", "sqlalchemy", "gunicorn", "redis", "chart.js", "bootstrap"]):
            phrase_bonus += 0.08
        if any(term in text_lower for term in ["version", "versions", "dependency", "dependencies", "uses", "used", "stack"]):
            phrase_bonus += 0.08
    if re.search(r"\b(?:summary|summarize|comprehensive)\b", query_lower):
        phrase_bonus += min(0.10, 0.02 * len(text_tokens & query_tokens))
    if re.search(r"\b(?:how many|total|count|list|different)\b", query_lower):
        phrase_bonus += min(0.08, 0.02 * len(text_tokens & query_tokens))
    return min(1.0, (0.70 * overlap) + phrase_bonus)


def _quality_fallback_terms(query: str, *, limit: int = 4) -> list[str]:
    tokens = _ordered_topic_scope_tokens(query)
    high_signal = [
        token
        for token in tokens
        if len(token) >= 4
        and token not in {
            "answer",
            "asked",
            "based",
            "between",
            "conversation",
            "conversations",
            "different",
            "include",
            "mentioned",
            "question",
            "summary",
            "summarize",
            "throughout",
        }
    ]
    if not high_signal:
        return []
    phrases: list[str] = []
    action_terms = [
        token
        for token in high_signal
        if token
        in {
            "decided",
            "decision",
            "reject",
            "rejected",
            "decline",
            "declined",
            "rescheduled",
            "reschedule",
            "downloaded",
            "download",
            "finish",
            "finished",
            "complete",
            "completed",
            "started",
            "start",
        }
    ]
    object_terms = [token for token in high_signal if token not in action_terms]
    for action in action_terms:
        for size in (2, 1):
            for index in range(0, max(0, len(object_terms) - size + 1)):
                phrase = " ".join([action, *object_terms[index : index + size]]).strip()
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= limit:
                    return phrases
    for size in (3, 2):
        for index in range(0, max(0, len(high_signal) - size + 1)):
            phrase = " ".join(high_signal[index : index + size]).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    for token in high_signal:
        if token not in phrases:
            phrases.append(token)
        if len(phrases) >= limit:
            break
    return phrases[:limit]


def _fallback_salience_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    lower = stripped.lower()
    score = 0.0
    length = len(stripped)
    if length >= 40:
        score += 0.16
    if length >= 120:
        score += 0.08
    if re.search(r"\b\d+(?:\.\d+)?%?|\$\s?\d|\b20\d{2}\b", stripped):
        score += 0.18
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}\b", lower):
        score += 0.12
    if "\n" in stripped or re.search(r"^\s*(?:[-*]|\d+\.)\s+", stripped, flags=re.M):
        score += 0.12
    if re.search(r"\b(?:decided|changed|updated|completed|finished|started|added|removed|fixed|error|deadline|budget|version|target|result|score|accuracy)\b", lower):
        score += 0.18
    if re.fullmatch(r"(?:ok|okay|thanks?|yes|no|sure|got it|sounds good)[.! ]*", lower):
        score -= 0.30
    return max(0.0, min(1.0, score))


def _aggregation_signal(query: str, text: str, query_terms: set[str]) -> float:
    lower = text.lower()
    text_terms = _topic_scope_tokens(text)
    generic_keys = generic_aggregation_keys(query, lower, speaker="user" if re.search(r"\b(?:i|my|we|our)\b", lower) else None)
    vendor_tool_keys = vendor_tool_aggregation_keys(query.lower(), text, speaker="user" if re.search(r"\b(?:i|my|we|our)\b", lower) else "assistant")
    if query_terms and len((query_terms | _expand_topic_tokens(query_terms)) & _expand_topic_tokens(text_terms)) == 0 and not generic_keys and not vendor_tool_keys:
        return 0.0
    signal = 0.0
    query_lower = query.lower()
    object_patterns = []
    if re.search(r"\b(?:columns?|fields?|schema|table)\b", query_lower):
        object_patterns.extend(
            [
                r"\b(?:add|adding|added|include|including|new|want)\b.{0,80}\b(?:columns?|fields?)\b",
                r"\b(?:columns?|fields?)\b.{0,80}\b(?:category|notes?|status|type|amount|date|user_id)\b",
                r"`[^`]*(?:category|notes?|status|type|amount|date|user_id)[^`]*`",
            ]
        )
    if re.search(r"\b(?:roles?|security features?|features?)\b", query_lower):
        object_patterns.extend(
            [
                r"\b(?:role-based access control|rbac|admin|user role|roles?)\b",
                r"\b(?:password hashing|account lockout|failed login|session validation|authorization|authentication|csrf|security feature)\b",
            ]
        )
    if re.search(r"\b(?:versions?|libraries?|tools?|dependencies?)\b", query_lower):
        object_patterns.extend(
            [
                r"\b(?:flask|sqlite|sqlalchemy|jinja|bootstrap|pytest|werkzeug|gunicorn|render|argon2|bcrypt)[A-Za-z0-9_.=-]*\b",
                r"\bv?\d+\.\d+(?:\.\d+)?\b",
            ]
        )
    if re.search(r"\b(?:features?|components?|cards?|requests?)\b", query_lower):
        object_patterns.extend(
            [
                r"\b(?:feature|component|card|request)s?\b",
                r"\b(?:add|adding|added|implement|implemented|need|want|include|including)\b",
            ]
        )
    if _is_combinatorics_aggregation_query(query_lower):
        object_patterns.extend(
            [
                r"\b(?:arrang(?:e|ing)|permutations?|n!|\d+!)\b.{0,120}\b(?:ways?|balls?|objects?)\b",
                r"\b(?:choos(?:e|ing)|combinations?|combination|c\(|\d+\s*c\s*\d+|\d+c\d+)\b.{0,140}\b(?:ways?|balls?|cards?|deck)\b",
                r"\b(?:balls?|cards?|deck|objects?)\b.{0,140}\b(?:arrang(?:e|ing)|choos(?:e|ing)|permutations?|combinations?|ways?)\b",
                r"\b(?:probability calculations?|calculations?|confirm|verify|checked|tried)\b",
            ]
        )
    if _is_stress_break_aggregation_query(query_lower):
        object_patterns.extend(
            [
                r"\b(?:\d+\s*-?\s*hours?|one-hour|1-hour|full days? off|days? off|yoga break|break)\b.{0,140}\b(?:stress|stressed|burnout|focus|reset|rest)\b",
                r"\b(?:stress|stressed|burnout|focus|reset|rest)\b.{0,140}\b(?:\d+\s*-?\s*hours?|one-hour|1-hour|full days? off|days? off|yoga break|break)\b",
            ]
        )
    if is_generic_count_or_list_query(query_lower):
        object_patterns.extend(
            [
                r'"[^"\n]{2,80}"',
                r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+[^\n]{3,140}",
                r"\b(?:selected|chose|decided|planned|finalized|mentioned|listed|included|added|tracked|saved|ordered|submitted)\b.{0,140}\b(?:[A-Z][A-Za-z0-9&'.-]+|\$?\d)",
                r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|months?|weeks?|days?|hours?|monthly|per month|per year|items?|options?|entries?)\b",
            ]
        )
    if not object_patterns and not re.search(r"\b(?:how many|different|across|throughout|sessions?|requests?|total|count)\b", query_lower):
        return 0.0
    hits = sum(1 for pattern in object_patterns if re.search(pattern, lower))
    if hits:
        signal += min(0.80, 0.28 * hits)
    if generic_keys:
        signal += min(0.70, 0.26 * len(generic_keys))
    if vendor_tool_keys:
        signal += min(0.72, 0.30 * len(vendor_tool_keys))
    synthesis = _synthesis_evidence_signal(query, text)
    if synthesis:
        signal += synthesis
    if re.search(r"\b(?:also|another|additionally|later|next|across|in another|second|third|new)\b", lower):
        signal += 0.12
    if re.search(r"\b(?:want|wanted|asked|requested|trying|need|decided|planned)\b", lower):
        signal += 0.10
    if _is_combinatorics_aggregation_query(query_lower):
        keys = combinatorics_aggregation_keys(lower)
        if keys:
            signal += min(0.36, 0.14 * len(keys))
        if re.search(r"\b(?:i(?:'m| am)?\s+trying|can you help|would i use|i want|i came across)\b", lower):
            signal += 0.12
        if re.search(r"\b(?:\d+\s*c\s*\d+|\d+c\d+|n!|\d+!|choose\s+\d+|draw\s+\d+|arrange\s+\d+)\b", lower):
            signal += 0.16
    if _is_stress_break_aggregation_query(query_lower):
        keys = stress_break_aggregation_keys(lower)
        if keys:
            signal += min(0.44, 0.18 * len(keys))
        if re.search(r"\b(?:i\s+took|i\s+had\s+to|i(?:'m| am)?\s+feeling|prevent burnout|manage stress)\b", lower):
            signal += 0.18
    overlap = len(_expand_topic_tokens(query_terms) & _expand_topic_tokens(text_terms)) / max(1, len(query_terms))
    signal += min(0.20, 0.20 * overlap)
    return min(1.0, signal)


def _adjacent_assistant_recommendation_spans(query: str, spans: list[EvidenceSpan]) -> dict[str, list[EvidenceSpan]]:
    query_lower = query.lower()
    if not is_generic_count_or_list_query(query_lower):
        return {}
    if not re.search(r"\b(?:books?|series|genres?|titles?|movies?|films?|items?|options?|recommendations?)\b", query_lower):
        return {}
    out: dict[str, list[EvidenceSpan]] = {}
    for index, span in enumerate(spans):
        if span.speaker not in {"user", "document"}:
            continue
        if not _aggregation_recommendation_request_signal(query, span.content):
            continue
        support: list[EvidenceSpan] = []
        for next_span in spans[index + 1 : index + 5]:
            if next_span.speaker in {"user", "document"}:
                break
            if _span_group_key(next_span) != _span_group_key(span):
                continue
            if _assistant_recommendation_list_signal(query, next_span.content) > 0:
                support.append(next_span)
                break
        if support:
            out[span.span_id] = support
    return out


def _aggregation_recommendation_request_signal(query: str, text: str) -> bool:
    lower = text.lower()
    if not re.search(r"\b(?:recommend|suggest|give me|list|options?|ideas?|looking for|help me pick|help me choose|find)\b", lower):
        return False
    if not re.search(r"\b(?:books?|series|genres?|titles?|movies?|films?|items?|options?|recommendations?)\b", lower):
        return False
    query_terms = _aggregation_query_terms(query)
    if query_terms and _expand_topic_tokens(query_terms).isdisjoint(_expand_topic_tokens(_topic_scope_tokens(text))):
        return False
    return True


def _recommendation_request_specificity(text: str) -> float:
    lower = text.lower()
    score = 0.0
    if re.search(r"\$\s?\d|\b\d+\s*(?:dollars?|usd)\b|\bbudget\b", lower):
        score += 0.35
    if re.search(r"\b(?:buy|purchase|order|borrow|download|print editions?|audiobooks?|e-?books?)\b", lower):
        score += 0.22
    if re.search(r"\b(?:from|at|on)\s+[A-Z][A-Za-z0-9&' -]{2,60}", text):
        score += 0.18
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}|20\d{2}\b", lower):
        score += 0.12
    if re.search(r"\b(?:fit|fits|criteria|preference|preferences|constraint|constraints|deadline|goal)\b", lower):
        score += 0.13
    return min(1.0, score)


def _assistant_recommendation_list_signal(query: str, text: str) -> float:
    lower = text.lower()
    if not re.search(r"\b(?:recommend|suggest|here are|few|options?|good fit|might fit|could fit|series|books?|titles?|genres?)\b", lower):
        return 0.0
    bullets = len(re.findall(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+\S", text))
    keys = generic_list_candidate_keys(query.lower(), text)
    title_count = len([key for key in keys if key.startswith("title:")])
    genre_count = len([key for key in keys if key.startswith("genre:")])
    item_count = max(bullets, title_count + genre_count)
    if item_count < 2:
        return 0.0
    score = 0.45 + min(0.35, item_count * 0.06)
    if re.search(r"\b(?:budget|fit|fits|preferences?|criteria|looking for|requested|asked)\b", lower):
        score += 0.12
    if re.search(r"\b(?:additional|alternative|more options|also consider)\b", lower):
        score -= 0.08
    return max(0.0, min(1.0, score))


def _synthesis_evidence_signal(query: str, text: str) -> float:
    query_lower = query.lower()
    lower = text.lower()
    if not re.search(r"\b(?:how|considering|given|what|which)\b", query_lower):
        return 0.0
    if not re.search(r"\b(?:i|my|we|our)\b", lower):
        return 0.0
    query_terms = _expand_topic_tokens(_topic_scope_tokens(query))
    text_terms = _expand_topic_tokens(_topic_scope_tokens(text))
    if query_terms and not (query_terms & text_terms):
        return 0.0
    signal = 0.0
    if re.search(r"\$\s?\d|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|months?|weeks?|days?|hours?|monthly|per month|per year)\b", lower):
        signal += 0.24
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}|20\d{2}\b", lower):
        signal += 0.16
    if re.search(r"\b(?:agreed|decided|chose|started|completed|increased|reduced|improved|reported|confirmed|took on|taking on|support|saving|savings|budget|income|expense|goal|deadline)\b", lower):
        signal += 0.24
    if re.search(r"\b(?:also|later|after|before|while|since|now|current|currently)\b", lower):
        signal += 0.10
    if _is_cross_factor_synthesis_query(query_lower):
        overlap = len(query_terms & text_terms) / max(1, len(query_terms))
        signal += min(0.28, 0.28 * overlap)
        if re.search(r"\b(?:affect|impact|support|meet|cover|balance|ability|able|goal|goals|budget|income|expense|expenses?|bills?|savings?)\b", lower):
            signal += 0.18
    return min(0.62, signal)


def _is_cross_factor_synthesis_query(query_lower: str) -> bool:
    return bool(
        re.search(r"\b(?:how|what|which|should|can)\b", query_lower)
        and re.search(r"\b(?:affect|impact|influence|balance|prioriti[sz]e|optimi[sz]e|ability|able|support|meet|cover)\b", query_lower)
        and re.search(r"\b(?:while|with|and|considering|given)\b", query_lower)
    )


def _synthesis_candidate_key(candidate: Candidate) -> str:
    if candidate.metadata.get("speaker") != "user":
        return ""
    if float(candidate.scores.get("synthesis_signal", 0.0) or 0.0) <= 0:
        return ""
    text = candidate.text.lower()
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", text)
        if len(term) >= 3
        and term
        not in {
            "the",
            "and",
            "for",
            "with",
            "how",
            "can",
            "you",
            "help",
            "this",
            "that",
            "what",
            "when",
            "will",
            "have",
            "been",
            "about",
            "because",
            "maybe",
        }
    ]
    value_terms = re.findall(r"\$?\d+(?:,\d{3})*(?:\.\d+)?(?:%|[a-z]*)?", text)
    key_terms = terms[:8] + value_terms[:3]
    return "synthesis:" + "_".join(key_terms[:10]) if key_terms else ""


def _aggregation_focus_priority(query: str, text: str) -> float:
    query_lower = query.lower()
    lower = text.lower()
    focus = 0.0
    if is_generic_count_or_list_query(query_lower):
        if generic_list_candidate_keys(query_lower, lower):
            focus += 0.45
        if vendor_tool_aggregation_keys(query_lower, text, speaker="user" if re.search(r"\b(?:i|my|we|our)\b", lower) else "assistant"):
            focus += 0.50
        if re.search(r"\b(?:selected|chose|decided|planned|finalized|mentioned|listed|included|added|tracked|submitted|ordered)\b", lower):
            focus += 0.25
        if re.search(r"\b(?:also|another|later|next|previously|earlier)\b", lower):
            focus += 0.15
    return max(0.0, min(1.0, focus))


def _is_generic_count_or_list_query(query_lower: str) -> bool:
    return is_generic_count_or_list_query(query_lower)


def _generic_aggregation_keys(query: str, lower: str, *, speaker: str | None = None) -> list[str]:
    return generic_aggregation_keys(query, lower, speaker=speaker)


def _clean_generic_aggregation_key(value: str) -> str:
    value = re.split(r"\b(?:across|throughout|because|while|after|before|and then|,|\.|;)\b", value, maxsplit=1)[0]
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", value)
        if len(term) >= 3
        and term
        not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "also",
            "want",
            "wanted",
            "need",
            "needed",
            "trying",
            "handle",
            "using",
            "into",
        }
    ]
    if not terms:
        return ""
    return "_".join(terms[:5])


def _quoted_title_candidates(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r'"([^"\n]{2,80})"', text) if match.group(1).strip()]


def _normalize_title_key(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return normalized[:60] or "untitled"


def _is_non_title_quote(title: str) -> bool:
    lower = title.lower().strip()
    return lower in {
        "netflix",
        "disney+",
        "pg",
        "pg-13",
        "r",
        "audible",
        "libby",
    } or bool(re.fullmatch(r"\d{4}", lower))


def _has_value_intent(query_lower: str) -> bool:
    return bool(
        re.search(r"\b(?:how many|how much|average|count|number|version|date|deadline|duration|weeks?|days?|time|response time)\b", query_lower)
        or re.search(r"(?:日期|时间|截止|发布目标|目标日|目标时间|什么时候)", query_lower)
    )


def _has_current_intent(query_lower: str) -> bool:
    return bool(
        re.search(r"\b(?:current|currently|now|latest|recent|recently|final|finally|updated|reached|reduced|improved|switched)\b", query_lower)
        or re.search(r"\bwhat\s+is\s+(?:the\s+)?(?:average|status|value|count|number|version|response time)\b", query_lower)
    )


def _compatible_value_mention(query_lower: str, lower: str) -> bool:
    if re.search(r"\b(?:response time|latency|average.*time|time.*average)\b", query_lower):
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?)\b", lower) or "response time" in lower)
    if "version" in query_lower:
        return bool(re.search(r"\bv?\d+\.\d+(?:\.\d+)?\b", lower))
    asks_date = bool(
        re.search(r"\b(?:date|deadline|weeks?|days?|duration|between)\b", query_lower)
        or re.search(r"(?:日期|时间|截止|发布目标|目标日|目标时间|什么时候)", query_lower)
    )
    if asks_date:
        return bool(
            re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?)\b", lower)
            or re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", lower)
            or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b", lower)
            or re.search(r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日(?!\d)", lower)
        )
    if re.search(r"\b(?:how many|count|number)\b", query_lower):
        return bool(re.search(r"\b\d+\b", lower))
    return bool(re.search(r"\b\d", lower))


def _value_signal(query_lower: str, lower: str) -> float:
    signal = 0.0
    asks_date = bool(
        re.search(r"\b(?:date|deadline|weeks?|days?|duration|between)\b", query_lower)
        or re.search(r"(?:日期|时间|截止|发布目标|目标日|目标时间|什么时候)", query_lower)
    )
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds?|minutes?|hours?)\b", lower):
        signal += 0.45 if re.search(r"\b(?:response time|latency|average.*time|time.*average)\b", query_lower) else 0.30
    if asks_date and re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?)\b", lower):
        signal += 0.35
    if re.search(r"\b(?:how many|count|number)\b", query_lower) and re.search(r"\b\d+(?:\.\d+)?\s*(?:%|commits?)\b", lower):
        signal += 0.35
    if "version" in query_lower and re.search(r"\bv?\d+\.\d+(?:\.\d+)?\b", lower):
        signal += 0.35
    if asks_date and re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b", lower):
        signal += 0.25
    if asks_date and re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", lower):
        signal += 0.25
    if asks_date and re.search(r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日(?!\d)", lower):
        signal += 0.35
    return min(0.60, signal)


def _current_state_signal(lower: str) -> float:
    signal = 0.0
    if re.search(r"\b(?:now|currently|latest|recently|final|finalized|updated|current)\b", lower):
        signal += 0.22
    if re.search(r"\b(?:reached|reduced to|improved to|switched to|moved to|is now|has now|now reached)\b", lower):
        signal += 0.28
    if re.search(r"\b(?:initially|previously|before|was originally|used to)\b", lower):
        signal -= 0.10
    return max(0.0, min(0.45, signal))


def _code_identifier_signal(query_lower: str, lower: str) -> float:
    identifiers = [
        token
        for token in re.findall(r"[a-z][a-z0-9_]{2,}", query_lower)
        if "_" in token or token in {"api", "crud", "auth", "pytest", "flask", "sqlalchemy", "dashboard", "transactions"}
    ]
    if not identifiers:
        return 0.0
    hits = sum(1 for token in dict.fromkeys(identifiers) if token in lower)
    return min(0.25, 0.08 * hits)


EVENT_ORDERING_STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "and",
    "application",
    "aspect",
    "aspects",
    "before",
    "brought",
    "can",
    "conversation",
    "conversations",
    "different",
    "for",
    "from",
    "help",
    "how",
    "into",
    "list",
    "mention",
    "mentioned",
    "only",
    "order",
    "our",
    "project",
    "through",
    "throughout",
    "walk",
    "which",
    "with",
    "you",
}


def _candidate_in_timeline_window(position: tuple[Any, ...] | None, start: tuple[Any, ...] | None, end: tuple[Any, ...] | None) -> bool:
    if start is None or end is None or position is None:
        return True
    return start <= position <= end


def _natural_turn_key(value: object) -> tuple[tuple[int, int | str], ...]:
    text = "" if value is None else str(value)
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(parts)


def _key_diverse_aggregation_candidates(
    scored: list[tuple[float, Candidate, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]],
    limit: int,
) -> list[Candidate]:
    if not scored or limit <= 0:
        return []
    keyed: dict[str, list[tuple[float, Candidate, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]]] = {}
    fallback: list[tuple[float, Candidate, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]] = []
    for item in scored:
        _score, candidate, _source_key, _turn_key = item
        keys = [str(key) for key in candidate.metadata.get("aggregation_keys") or [] if key]
        if not keys:
            fallback.append(item)
            continue
        for key in keys:
            keyed.setdefault(key, []).append(item)

    representatives: list[tuple[float, Candidate, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]] = []
    for items in keyed.values():
        items.sort(
            key=lambda item: (
                item[1].metadata.get("speaker") == "user",
                item[0],
                item[2],
                item[3],
            ),
            reverse=True,
        )
        representatives.append(items[0])
    representatives.sort(
        key=lambda item: (
            item[1].metadata.get("speaker") == "user",
            item[0],
            item[2],
            item[3],
        ),
        reverse=True,
    )
    contextual_support = [
        item
        for item in scored
        if _aggregation_context_support_candidate(item[1])
    ]
    contextual_support.sort(
        key=lambda item: (
            item[1].scores.get("request_specificity", item[1].metadata.get("request_specificity", 0.0)),
            item[0],
            item[2],
            item[3],
        ),
        reverse=True,
    )
    scene_representatives = _aggregation_scene_representatives([item[1] for item in scored], limit=max(1, min(6, limit)))
    scene_items_by_id = {(candidate.type, candidate.id): (0.0, candidate, (), ()) for candidate in scene_representatives}

    out: list[Candidate] = []
    seen_ids: set[tuple[str, str]] = set()
    representative_head = representatives[: max(1, min(len(representatives), limit // 2))]
    for _score, candidate, _source_key, _turn_key in list(scene_items_by_id.values()) + representative_head + contextual_support + representatives + fallback + scored:
        identity = (candidate.type, candidate.id)
        if identity in seen_ids:
            continue
        out.append(candidate)
        seen_ids.add(identity)
        if len(out) >= limit:
            break
    return out


def _aggregation_scene_representatives(candidates: list[Candidate], *, limit: int) -> list[Candidate]:
    if limit <= 0:
        return []
    by_scene: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        for key in candidate.metadata.get("aggregation_keys") or []:
            key_text = str(key)
            if key_text.startswith("query_context:"):
                by_scene.setdefault(key_text, []).append(candidate)
    representatives: list[Candidate] = []
    for scene_key, items in by_scene.items():
        items.sort(
            key=lambda candidate: (
                candidate.metadata.get("speaker") == "user",
                candidate.scores.get("aggregation_focus", 0.0),
                candidate.scores.get("aggregation_signal", 0.0),
                candidate.scores.get("score", 0.0),
                candidate.scores.get("topic_scope_score", 0.0),
            ),
            reverse=True,
        )
        representative = items[0]
        representative.scores["scene_diversity_signal"] = max(float(representative.scores.get("scene_diversity_signal", 0.0) or 0.0), 1.0)
        representative.metadata["aggregation_scene_key"] = scene_key
        representatives.append(representative)
    representatives.sort(
        key=lambda candidate: (
            candidate.metadata.get("speaker") == "user",
            candidate.scores.get("scene_diversity_signal", 0.0),
            candidate.scores.get("aggregation_focus", 0.0),
            candidate.scores.get("aggregation_signal", 0.0),
            candidate.scores.get("score", 0.0),
        ),
        reverse=True,
    )
    return representatives[:limit]


def _aggregation_context_support_candidate(candidate: Candidate) -> bool:
    if candidate.metadata.get("aggregation_context_support"):
        return True
    keys = [str(key) for key in candidate.metadata.get("aggregation_keys") or [] if key]
    if not keys:
        return False
    lower = candidate.text.lower()
    has_date_or_time = bool(
        re.search(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}|"
            r"\b20\d{2}\b|\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
            lower,
        )
    )
    has_commitment_context = bool(
        re.search(
            r"\b(?:schedule|agenda|timeline|final list|watchlist|will include|would include|"
            r"included|selected|chosen|planned|morning session|afternoon session)\b",
            lower,
        )
    )
    return has_date_or_time and has_commitment_context


def _aggregation_group_support_specificity(candidate: Candidate) -> float:
    keys = [str(key) for key in candidate.metadata.get("aggregation_keys") or [] if key]
    if not any(key.startswith("group_support:") for key in keys):
        return 0.0
    return float(candidate.scores.get("request_specificity", candidate.metadata.get("request_specificity", 0.0)) or 0.0)


def _high_value_aggregation_context_support(candidate: Candidate) -> bool:
    specificity = _aggregation_group_support_specificity(candidate)
    if specificity <= 0.0:
        return True
    return specificity >= 0.45 or float(candidate.scores.get("score", 0.0) or 0.0) >= 0.82


def _aggregation_query_date_support(query: str, candidate: Candidate) -> bool:
    query_dates = _service_date_scope_labels(query.lower())
    if not query_dates:
        return False
    content_dates = _service_date_scope_labels(candidate.text.lower())
    return bool(content_dates and not query_dates.isdisjoint(content_dates))


def _aggregation_context_specificity(candidate: Candidate) -> float:
    keys = [str(key) for key in candidate.metadata.get("aggregation_keys") or [] if key]
    if not keys:
        return 0.0
    lower = candidate.text.lower()
    score = min(0.45, 1.0 / max(1, len(keys)))
    if "exact_filter" in candidate.source:
        score += 0.35
    if re.search(r"\b(?:schedule|timing|agenda|morning session|afternoon session|final list)\b", lower):
        score += 0.25
    if len(keys) > 8:
        score -= 0.20
    return max(0.0, min(1.0, score))


def _aggregation_query_context_keys(query: str, text: str) -> list[str]:
    query_lower = query.lower()
    lower = text.lower()
    text_features = _aggregation_context_features(lower)
    if not text_features:
        return []
    broad_exploration = _is_broad_exploration_aggregation_query(query_lower)
    if broad_exploration:
        active_features = text_features
    else:
        active_features = text_features & _aggregation_context_features(query_lower)
    return [f"query_context:feature:{feature}" for feature in sorted(active_features)]


def _aggregation_context_features(lower: str) -> set[str]:
    features: set[str] = set()
    if re.search(r"\b(?:live\s+chat|co-?host|host(?:ed|ing)?|moderate|webinar|panel|workshop|meeting|event)\b", lower):
        features.add("event")
    if re.search(r"\b(?:discussion|discuss|book\s+club|chat|talk|conversation)\b", lower):
        features.add("discussion")
    if re.search(r"\b(?:partner|friend|colleague|coworker|family|together|shared|with\s+[A-Z][A-Za-z]+)\b", lower):
        features.add("social")
    if re.search(r"\$\s?\d|\b(?:budget|cost|price|purchase|buy|bought|ordered|spend|spent)\b", lower):
        features.add("budget")
    if re.search(r"\b(?:library|borrow|e-?book|audiobook|print edition|paperback|kindle|download)\b", lower):
        features.add("access_format")
    if re.search(r"\b(?:challenge|goal|deadline|target|due|by\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))\b", lower):
        features.add("goal")
    if re.search(r"\b(?:recommend|suggest|options?|pick|choose|decide|explore|interested|looking for)\b", lower):
        features.add("exploration")
    return features


def _is_broad_exploration_aggregation_query(query_lower: str) -> bool:
    return bool(
        is_generic_count_or_list_query(query_lower)
        and re.search(r"\b(?:across|throughout|conversation|conversations|sessions?|mentioned|want(?:ed|ing)?|explor(?:e|ing)|interested)\b", query_lower)
        and re.search(r"\b(?:books?|series|genres?|titles?|movies?|films?|items?|options?|topics?)\b", query_lower)
    )


def _service_date_scope_labels(text_lower: str) -> set[str]:
    labels: set[str] = set()
    month_names = (
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    )
    month_map = {
        "jan": "january",
        "january": "january",
        "feb": "february",
        "february": "february",
        "mar": "march",
        "march": "march",
        "apr": "april",
        "april": "april",
        "may": "may",
        "jun": "june",
        "june": "june",
        "jul": "july",
        "july": "july",
        "aug": "august",
        "august": "august",
        "sep": "september",
        "sept": "september",
        "september": "september",
        "oct": "october",
        "october": "october",
        "nov": "november",
        "november": "november",
        "dec": "december",
        "december": "december",
    }
    for match in re.finditer(rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*[-–—]\s*(\d{{1,2}})(?:st|nd|rd|th)?\b", text_lower):
        month = month_map.get(match.group(1), match.group(1))
        start_day = int(match.group(2))
        end_day = int(match.group(3))
        if 1 <= start_day <= end_day <= 31 and end_day - start_day <= 14:
            for day in range(start_day, end_day + 1):
                labels.add(f"{month}:{day}")
    for match in re.finditer(rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text_lower):
        month = month_map.get(match.group(1), match.group(1))
        labels.add(f"{month}:{int(match.group(2))}")
    for match in re.finditer(r"\b20\d{2}[-/](\d{1,2})[-/](\d{1,2})\b", text_lower):
        labels.add(f"{int(match.group(1))}:{int(match.group(2))}")
    return labels
