from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    module: str
    purpose: str
    category: str
    pattern: str | None = None
    owner: str = "retrieval"


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    query: str
    text_hash: str
    contributed_candidate_id: str | None
    stage: str
    metadata: dict[str, object] = field(default_factory=dict)


_RULE_REGISTRY: dict[str, RuleDefinition] = {}
_RULE_HITS: list[RuleHit] = []
_SENSITIVE_METADATA_KEY_PARTS = (
    "raw_text",
    "text",
    "content",
    "span",
    "message",
    "query",
    "prompt",
)


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
) -> RuleHit:
    hit = RuleHit(
        rule_id=rule_id,
        query=query,
        text_hash=sha1(text.encode("utf-8")).hexdigest()[:12],
        contributed_candidate_id=contributed_candidate_id,
        stage=stage,
        metadata=_sanitize_metadata(metadata),
    )
    _RULE_HITS.append(hit)
    return hit


def drain_rule_hits() -> list[RuleHit]:
    hits = list(_RULE_HITS)
    _RULE_HITS.clear()
    return hits


def registered_rules() -> list[RuleDefinition]:
    return list(_RULE_REGISTRY.values())


def _sanitize_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in (metadata or {}).items():
        if _metadata_key_contains_raw_text(key):
            sanitized[key] = _hash_metadata_value(value)
            continue
        sanitized[key] = value
    return sanitized


def _metadata_key_contains_raw_text(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS)


def _hash_metadata_value(value: object) -> str:
    return sha1(repr(value).encode("utf-8")).hexdigest()[:12]
