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
        sanitized[key] = _sanitize_metadata_value(value)
    return sanitized


def _metadata_key_contains_raw_text(key: str) -> bool:
    normalized = key.lower()
    if normalized in _SENSITIVE_METADATA_KEYS:
        return True
    return any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS)


def _hash_metadata_value(value: object) -> str:
    return sha1(repr(value).encode("utf-8")).hexdigest()[:12]


def _sanitize_metadata_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): (_hash_metadata_value(item) if _metadata_key_contains_raw_text(str(key)) else _sanitize_metadata_value(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_metadata_value(item) for item in value]
    if isinstance(value, str):
        return value if _is_safe_metadata_string(value) else _hash_metadata_value(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _hash_metadata_value(value)


def _is_safe_metadata_string(value: str) -> bool:
    if len(value) > 128:
        return False
    if re.search(r"\s|[\u4e00-\u9fff]", value):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.:/@+\-]*", value))
