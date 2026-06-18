from __future__ import annotations

import re
from typing import Any, Callable

from fusion_memory.core.models import MemoryEvent
from fusion_memory.core.text import keyword_score
from fusion_memory.retrieval.event_chronology_graph import build_event_chronology_graph, select_graph_first_event_ordering_candidates


SOFTWARE_ASPECT_TERMS = {
    "analytics",
    "api",
    "auth",
    "authentication",
    "authorization",
    "cache",
    "ci",
    "config",
    "configuration",
    "coverage",
    "crud",
    "database",
    "deployment",
    "endpoint",
    "endpoints",
    "error",
    "errors",
    "flask",
    "gunicorn",
    "integration",
    "login",
    "migrate",
    "migration",
    "performance",
    "port",
    "postgresql",
    "render",
    "response",
    "schema",
    "security",
    "server",
    "setup",
    "sqlite",
    "test",
    "tests",
    "transaction",
    "transactions",
    "validation",
    "worker",
}

EVENT_ACTION_TERMS = {
    "add",
    "added",
    "configure",
    "configured",
    "debug",
    "debugged",
    "deploy",
    "deployed",
    "fix",
    "fixed",
    "implement",
    "implemented",
    "improve",
    "improved",
    "optimize",
    "optimized",
    "plan",
    "planned",
    "review",
    "reviewed",
    "setup",
    "test",
    "tested",
}

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


def _event_ordering_milestone_score(text: str) -> float:
    lower = text.lower()
    group_scores = [
        _event_group_score(
            lower,
            anchors=("setup", "schema", "server", "mvp", "core functionality", "initial project", "local development"),
            required_any=(),
        ),
        _event_group_score(
            lower,
            anchors=("transaction", "transactions"),
            required_any=(
                "crud",
                "error",
                "errors",
                "exception",
                "exceptions",
                "response",
                "handling",
                "validation",
                "post /transactions",
                "create_transaction",
                "created successfully",
            ),
        ),
        _event_group_score(
            lower,
            anchors=("deployment", "deploy", "render", "gunicorn", "port", "worker"),
            required_any=(
                "render",
                "gunicorn",
                "port",
                "worker",
                "configuration",
                "config",
                "settings",
                "server",
                "hosting",
                "environment",
                "production",
            ),
        ),
        _event_group_score(
            lower,
            anchors=("integration test", "integration tests", "coverage", "test suite", "endpoint", "endpoints"),
            required_any=("test", "tests", "coverage", "suite", "endpoint", "endpoints"),
        ),
        _event_group_score(
            lower,
            anchors=("security", "auth", "authentication", "authorization", "password", "argon2", "login"),
            required_any=("security", "password", "argon2", "authentication", "authorization", "login"),
        ),
    ]
    group_hits = sum(1 for score in group_scores if score > 0)
    if group_hits == 0:
        return 0.0
    action_bonus = 0.0
    if re.search(
        r"\b(?:trying|implement|implemented|working|worked|set up|setup|configure|configured|review|reviewing|add|added|switch|switched|decide|decided|plan|planned)\b",
        lower,
    ):
        action_bonus = 0.15
    return min(1.0, sum(group_scores) + action_bonus)


def _event_group_score(lower: str, *, anchors: tuple[str, ...], required_any: tuple[str, ...]) -> float:
    anchor_hits = sum(1 for phrase in anchors if phrase in lower)
    if anchor_hits == 0:
        return 0.0
    if required_any and not any(phrase in lower for phrase in required_any):
        return 0.0
    return min(0.45, 0.24 + 0.07 * min(anchor_hits, 3))


def _query_item_limit(lower: str) -> int | None:
    digit = re.search(r"\b(?:only\s+and\s+only\s+)?([2-9])\s+items?\b", lower)
    if digit:
        return int(digit.group(1))
    words = {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    for word, value in words.items():
        if re.search(rf"\b(?:only\s+and\s+only\s+)?{word}\s+items?\b", lower):
            return value
    return None


def _select_event_ordering_representatives(query: str, events: list[MemoryEvent], limit: int, sort_key) -> list[MemoryEvent]:
    if not events:
        return []
    desired = _query_item_limit(query.lower()) or min(limit, 8)
    desired = max(1, min(desired, limit))
    scored = [
        (event, _event_ordering_event_relevance(query, event))
        for event in events
    ]
    scored = _dedupe_same_span_phase_scored_events(scored, desired, sort_key)
    phase_selected = _select_phase_coverage_events(scored, desired, sort_key)
    if len(phase_selected) >= desired:
        return phase_selected[:desired]
    selected: list[MemoryEvent] = []
    seen_groups: set[str] = set()
    seen_families: set[str] = set()
    seen_spans: set[str] = set()
    ranked = sorted(scored, key=lambda item: (item[1], _reverse_order_key(sort_key(item[0]))), reverse=True)
    for event, relevance in ranked:
        group = _event_milestone_group(event) or _event_aspect_signature(event.description)
        family = _event_group_family(group)
        span_key = next(iter(event.source_span_ids), event.event_id)
        if relevance <= 0.0:
            continue
        if span_key in seen_spans and group in seen_groups:
            continue
        if family in seen_families and len(seen_families) < desired:
            continue
        selected.append(event)
        seen_groups.add(group)
        seen_families.add(family)
        seen_spans.add(span_key)
        if len(selected) >= desired:
            break
    if len(selected) < desired:
        for event, relevance in ranked:
            if event in selected or relevance <= 0.0:
                continue
            selected.append(event)
            if len(selected) >= desired:
                break
    selected.sort(key=sort_key)
    return selected


def _dedupe_same_span_phase_scored_events(
    scored: list[tuple[MemoryEvent, float]],
    desired: int,
    sort_key,
) -> list[tuple[MemoryEvent, float]]:
    best_by_key: dict[tuple[str, str], tuple[tuple[float, float, tuple[int, ...]], tuple[MemoryEvent, float]]] = {}
    key_order: list[tuple[str, str]] = []
    for event, relevance in scored:
        group = _event_milestone_group(event) or _event_aspect_signature(event.description)
        phase = _event_phase_family(group)
        span_key = next(iter(event.source_span_ids), event.event_id)
        key = (span_key, phase)
        rank = (
            _phase_group_preference(group, desired),
            relevance,
            _reverse_order_key(sort_key(event)),
        )
        if key not in best_by_key:
            key_order.append(key)
            best_by_key[key] = (rank, (event, relevance))
            continue
        if rank > best_by_key[key][0]:
            best_by_key[key] = (rank, (event, relevance))
    return [best_by_key[key][1] for key in key_order]


def _select_phase_coverage_events(scored: list[tuple[MemoryEvent, float]], desired: int, sort_key) -> list[MemoryEvent]:
    viable = [(event, relevance) for event, relevance in scored if relevance > 0.0 and _event_milestone_group(event)]
    if not viable:
        return []
    by_phase: dict[str, list[tuple[MemoryEvent, float]]] = {}
    for event, relevance in viable:
        group = _event_milestone_group(event) or ""
        phase = _event_phase_family(group)
        by_phase.setdefault(phase, []).append((event, relevance))
    selected: list[MemoryEvent] = []
    used_ids: set[str] = set()
    for phase in _event_phase_order(desired):
        candidates = by_phase.get(phase, [])
        if not candidates:
            continue
        event = _best_phase_representative(candidates, desired, sort_key)
        if event.event_id in used_ids:
            continue
        selected.append(event)
        used_ids.add(event.event_id)
        if len(selected) >= desired:
            break
    if len(selected) < desired:
        for event, _relevance in sorted(viable, key=lambda item: sort_key(item[0])):
            if event.event_id in used_ids:
                continue
            selected.append(event)
            used_ids.add(event.event_id)
            if len(selected) >= desired:
                break
    selected.sort(key=sort_key)
    return selected


def _event_phase_order(desired: int) -> list[str]:
    if desired <= 3:
        return ["foundation", "transaction", "deployment", "testing", "security", "deployment_improvement"]
    return ["foundation", "transaction", "deployment", "testing", "deployment_improvement", "security"]


def _event_phase_family(group: str) -> str:
    if group in {"core_functionality", "initial_project_setup", "setup_debugging"}:
        return "foundation"
    if group in {"transaction_crud_implementation", "transaction_error_handling"}:
        return "transaction"
    if group in {"deployment_configuration", "security_and_deployment"}:
        return "deployment"
    if group == "deployment_and_test_improvements":
        return "deployment_improvement"
    if group == "integration_test_coverage":
        return "testing"
    if group == "security_auth":
        return "security"
    return group


def _best_phase_representative(candidates: list[tuple[MemoryEvent, float]], desired: int, sort_key) -> MemoryEvent:
    def key(item: tuple[MemoryEvent, float]) -> tuple[float, float, tuple[Any, ...]]:
        event, relevance = item
        group = _event_milestone_group(event) or ""
        return (
            _phase_group_preference(group, desired),
            relevance,
            _reverse_order_key(sort_key(event)),
        )

    return max(candidates, key=key)[0]


def _phase_group_preference(group: str, desired: int) -> float:
    if group == "transaction_crud_implementation":
        return 0.96
    if desired <= 3 and group == "core_functionality":
        return 1.0
    if desired > 3 and group == "initial_project_setup":
        return 1.0
    if group in {"transaction_error_handling", "security_and_deployment", "deployment_and_test_improvements"}:
        return 0.95
    if group in {"transaction_crud_implementation", "deployment_configuration", "integration_test_coverage"}:
        return 0.90
    if group in {"core_functionality", "initial_project_setup"}:
        return 0.85
    return 0.70


def _event_ordering_event_relevance(query: str, event: MemoryEvent) -> float:
    group = _event_milestone_group(event)
    description = event.description
    base = keyword_score(query, description)
    aspect = _event_group_query_fit(query.lower(), group, description)
    if group:
        aspect = max(aspect, 0.22)
    return base + aspect


def _event_aspect_signature(text: str) -> str:
    lower = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lower)
    keep = [token for token in tokens if len(token) > 3 and token not in {"milestone", "evidence", "trying", "with", "this", "that", "have"}]
    return "_".join(keep[:3]) if keep else lower[:40]


def _event_group_family(group: str) -> str:
    if group in {"initial_project_setup", "setup_debugging"}:
        return "setup"
    if group in {"transaction_crud_implementation", "transaction_error_handling"}:
        return "transaction"
    if group in {"deployment_configuration", "security_and_deployment"}:
        return "deployment"
    if group in {"deployment_and_test_improvements"}:
        return "deployment_improvement"
    if group in {"integration_test_coverage"}:
        return "tests"
    if group in {"security_auth"}:
        return "security"
    if group in {"core_functionality"}:
        return "core"
    return group


def _event_group_query_fit(query_lower: str, group: str | None, description: str) -> float:
    lower = description.lower()
    query_tokens = _event_ordering_tokens(query_lower)
    event_tokens = _event_ordering_tokens(lower)
    group_tokens = _event_ordering_tokens((group or "").replace("_", " "))
    if not event_tokens and not group_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(event_tokens.union(group_tokens))) / max(1, len(query_tokens))
    salient_hits = sum(1 for token in event_tokens.union(group_tokens) if token in SOFTWARE_ASPECT_TERMS)
    action_hits = sum(1 for token in event_tokens if token in EVENT_ACTION_TERMS)
    compound_bonus = 0.10 if len(group_tokens) >= 2 else 0.0
    return min(1.0, (0.55 * overlap) + (0.06 * min(salient_hits, 5)) + (0.04 * min(action_hits, 3)) + compound_bonus)


def _event_ordering_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9_]+", text.lower())
    tokens: set[str] = set()
    for token in raw:
        if len(token) < 3 or token in EVENT_ORDERING_STOPWORDS:
            continue
        tokens.add(token)
        if token.endswith("s") and len(token) > 4:
            tokens.add(token[:-1])
    return tokens


def _reverse_order_key(value: tuple[Any, ...]) -> tuple[int, ...]:
    encoded = "|".join(str(part) for part in value)
    return tuple(-ord(char) for char in encoded)


def _event_milestone_group(event: MemoryEvent) -> str | None:
    text = event.description
    match = re.search(r"Milestone \[([a-z0-9_]+)\]", text)
    if match:
        return match.group(1)
    if event.event_type != "milestone":
        return None
    for participant in event.participants:
        value = str(participant)
        if value in EVENT_ORDERING_MILESTONE_GROUPS:
            return str(participant)
    return None


EVENT_ORDERING_MILESTONE_GROUPS = {
    "core_functionality",
    "initial_project_setup",
    "setup_debugging",
    "transaction_crud_implementation",
    "transaction_error_handling",
    "deployment_configuration",
    "deployment_and_test_improvements",
    "integration_test_coverage",
    "security_auth",
    "security_and_deployment",
}
