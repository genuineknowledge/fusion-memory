from __future__ import annotations

from typing import Any

from fusion_memory.retrieval.rule_registry import RuleDefinition


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _duplicate_of(rule_hits: list[dict[str, object]]) -> str | None:
    for hit in rule_hits:
        metadata = _as_dict(hit.get("metadata"))
        duplicate_of = metadata.get("duplicate_of")
        if isinstance(duplicate_of, str) and duplicate_of:
            return duplicate_of
    return None


def _cleanup_classification(
    rule_id: str,
    hit_count: int,
    contribution_count: int,
    duplicate_of: str | None,
) -> tuple[str, str, bool]:
    if rule_id.startswith("event_ordering.legacy"):
        cleanup_action = "keep_shadow"
    elif duplicate_of is not None:
        cleanup_action = "delete_duplicate"
    elif hit_count == 0:
        cleanup_action = "delete_no_hits"
    elif contribution_count == 0:
        cleanup_action = "delete_no_contribution"
    else:
        cleanup_action = "keep"

    cleanup_phase = "first_pass" if cleanup_action.startswith("delete_") else ""
    safe_to_delete = cleanup_action.startswith("delete_")
    return cleanup_phase, cleanup_action, safe_to_delete


def build_rule_audit(
    rule_definitions: list[RuleDefinition],
    hits: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_rule: dict[str, list[dict[str, object]]] = {}
    for hit in hits:
        by_rule.setdefault(str(hit.get("rule_id")), []).append(hit)

    rows: list[dict[str, Any]] = []
    for rule in sorted(rule_definitions, key=lambda item: item.rule_id):
        rule_hits = by_rule.get(rule.rule_id, [])
        contribution_count = sum(
            1
            for hit in rule_hits
            if hit.get("contributed") is True or hit.get("impact") == "selected"
        )
        negative_impact_count = sum(
            1 for hit in rule_hits if hit.get("impact") in {"filtered", "dropped", "misranked"}
        )
        evidence_inputs = sorted(
            {
                audit_input
                for hit in rule_hits
                if isinstance((audit_input := hit.get("_audit_input")), str) and audit_input
            }
        )
        duplicate_of = _duplicate_of(rule_hits)
        cleanup_phase, cleanup_action, safe_to_delete = _cleanup_classification(
            rule.rule_id,
            len(rule_hits),
            contribution_count,
            duplicate_of,
        )
        rows.append(
            {
                "rule_id": rule.rule_id,
                "ability": rule.ability,
                "category": rule.category,
                "module": rule.module,
                "hit_count": len(rule_hits),
                "contribution_count": contribution_count,
                "negative_impact_count": negative_impact_count,
                "candidate_for_deletion": safe_to_delete,
                "evidence_inputs": evidence_inputs,
                "duplicate_of": duplicate_of,
                "cleanup_phase": cleanup_phase,
                "cleanup_action": cleanup_action,
                "safe_to_delete": safe_to_delete,
            }
        )
    return rows
