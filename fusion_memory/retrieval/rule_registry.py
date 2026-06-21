from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha1
import re
from typing import Iterator


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    module: str
    purpose: str
    category: str
    pattern: str | None = None
    owner: str = "retrieval"
    ability: str = "general"
    protected: bool = False
    protected_reason: str = ""
    duplicate_of: str | None = None


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    query: str
    text_hash: str
    contributed_candidate_id: str | None
    stage: str
    metadata: dict[str, object] = field(default_factory=dict)
    contributed: bool | None = None
    impact: str = "observed"
    provider_id: str | None = None
    lifecycle_stage: str | None = None
    lifecycle_reason: str | None = None


_RULE_REGISTRY: dict[str, RuleDefinition] = {}
_ACTIVE_RULE_HITS: ContextVar[list[RuleHit] | None] = ContextVar("fusion_memory_rule_hits", default=None)
_SENSITIVE_METADATA_KEY_PARTS = (
    "raw_text",
    "text",
    "content",
    "span",
    "message",
    "query",
    "prompt",
)
_SENSITIVE_METADATA_KEYS = {
    "phrases",
    "conditions",
    "taxonomy_hits",
}
_PLAINTEXT_METADATA_STRINGS = {
    "answer",
    "candidate_1",
    "candidate_2",
    "current_value",
    "delete_no_hits",
    "drop_stale_history",
    "evidence_pack_filter",
    "event_ordering",
    "event_ordering_coverage",
    "event_ordering_episode_recall",
    "event_ordering_graph_selector",
    "event_ordering_timeline",
    "fallback",
    "filter",
    "filtered",
    "first_pass",
    "generic",
    "keep_shadow",
    "kept",
    "l0_raw",
    "l0_raw_hybrid",
    "l1_fact_hybrid",
    "l3_current_view",
    "legacy_fallback",
    "observed",
    "preserve_language_exact_match",
    "retrieval",
    "search_filter",
    "selected",
    "span_1",
    "suppress",
    "test_exception_cleanup",
    "views",
}
_PLAINTEXT_METADATA_KEY_VALUES = {
    "category": {
        "current_value",
        "event_ordering",
        "generic",
        "retrieval",
    },
    "decision": {
        "drop_stale_history",
        "fallback",
        "keep",
        "kept",
        "legacy_fallback",
        "preserve_language_exact_match",
        "selected",
        "suppress",
        "test_exception_cleanup",
    },
    "impact": {
        "filtered",
        "observed",
        "selected",
    },
    "label": {
        "history-marker",
    },
    "source": {
        "candidate_1",
        "candidate_2",
        "event_ordering_coverage",
        "event_ordering_episode_recall",
        "event_ordering_graph_selector",
        "event_ordering_timeline",
        "l0_raw",
        "l0_raw_hybrid",
        "l1_fact_hybrid",
        "l3_current_view",
        "quality_fallback",
        "span_1",
    },
    "stage": {
        "evidence_pack_filter",
        "filter",
        "search_filter",
        "test",
    },
}


def register_rule(rule: RuleDefinition) -> RuleDefinition:
    _RULE_REGISTRY[rule.rule_id] = rule
    return rule


def record_rule_hit(
    rule_id: str,
    query: str,
    text: str,
    stage: str,
    contributed_candidate_id: str | None = None,
    metadata: dict[str, object] | None = None,
    *,
    contributed: bool | None = None,
    impact: str = "observed",
    provider_id: str | None = None,
    lifecycle_stage: str | None = None,
    lifecycle_reason: str | None = None,
) -> RuleHit:
    hit = RuleHit(
        rule_id=rule_id,
        query=sha1(query.encode("utf-8")).hexdigest()[:12],
        text_hash=sha1(text.encode("utf-8")).hexdigest()[:12],
        contributed_candidate_id=contributed_candidate_id,
        stage=stage,
        metadata=_sanitize_metadata(metadata),
        contributed=contributed,
        impact=impact,
        provider_id=_safe_identifier(provider_id),
        lifecycle_stage=_safe_identifier(lifecycle_stage),
        lifecycle_reason=_safe_identifier(lifecycle_reason),
    )
    _current_hits().append(hit)
    return hit


def drain_rule_hits() -> list[RuleHit]:
    current = _current_hits()
    hits = list(current)
    current.clear()
    return hits


def registered_rules() -> list[RuleDefinition]:
    return list(_RULE_REGISTRY.values())


@contextmanager
def collect_rule_hits() -> Iterator[RuleHitCollector]:
    hits: list[RuleHit] = []
    token = _ACTIVE_RULE_HITS.set(hits)
    collector = RuleHitCollector(hits)
    try:
        yield collector
    finally:
        hits.clear()
        _ACTIVE_RULE_HITS.reset(token)


class RuleHitCollector:
    def __init__(self, hits: list[RuleHit]) -> None:
        self._hits = hits

    def drain(self) -> list[RuleHit]:
        hits = list(self._hits)
        self._hits.clear()
        return hits


def _current_hits() -> list[RuleHit]:
    active = _ACTIVE_RULE_HITS.get()
    if active is None:
        active = []
        _ACTIVE_RULE_HITS.set(active)
    return active


def _sanitize_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in (metadata or {}).items():
        if _metadata_key_contains_raw_text(key):
            sanitized[key] = _hash_metadata_value(value)
            continue
        sanitized[key] = _sanitize_metadata_value(value, str(key))
    return sanitized


def _metadata_key_contains_raw_text(key: str) -> bool:
    normalized = key.lower()
    if normalized in _SENSITIVE_METADATA_KEYS:
        return True
    return any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS)


def _hash_metadata_value(value: object) -> str:
    return sha1(repr(value).encode("utf-8")).hexdigest()[:12]


def _sanitize_metadata_value(value: object, key: str | None = None) -> object:
    if isinstance(value, dict):
        return {
            str(item_key): (
                _hash_metadata_value(item)
                if _metadata_key_contains_raw_text(str(item_key))
                else _sanitize_metadata_value(item, str(item_key))
            )
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_metadata_value(item, key) for item in value]
    if isinstance(value, str):
        return value if _is_safe_metadata_string(value, key) else _hash_metadata_value(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _hash_metadata_value(value)


def _safe_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if _is_safe_metadata_string(text) else _hash_metadata_value(text)


def _is_safe_metadata_string(value: str, key: str | None = None) -> bool:
    if len(value) > 128:
        return False
    if re.search(r"\s|[\u4e00-\u9fff]", value):
        return False
    if not re.fullmatch(r"[A-Za-z0-9_.:/@+\-]*", value):
        return False
    normalized_key = (key or "").lower()
    allowed_for_key = _PLAINTEXT_METADATA_KEY_VALUES.get(normalized_key)
    if allowed_for_key is not None:
        return value in allowed_for_key
    return value in _PLAINTEXT_METADATA_STRINGS
