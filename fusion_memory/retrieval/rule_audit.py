from __future__ import annotations

import hashlib
from typing import Any

from fusion_memory.retrieval.rule_registry import RuleDefinition

_SAFE_DIMENSION_IDENTIFIERS = {
    "aggregation_context_support",
    "aggregation_coverage",
    "aggregation_coverage_raw",
    "broad_raw",
    "broad_raw_recall",
    "contradiction_claim",
    "contradiction_claim_negative",
    "contradiction_claim_positive",
    "contradiction_claim_uncertain",
    "dropped",
    "entities",
    "entity_graph",
    "event_ordering_coverage",
    "event_ordering_coverage_support",
    "event_ordering_episode",
    "event_ordering_episode_recall",
    "event_ordering_timeline",
    "event_timeline_graph",
    "events",
    "exact",
    "exact_answer",
    "facts",
    "filtered",
    "final_selection",
    "hybrid",
    "l0_raw_hybrid",
    "l1_fact_hybrid",
    "l2_event_graph",
    "l3_current_view",
    "l3_entity_profile",
    "legacy_fallback",
    "misranked",
    "packed",
    "profiles",
    "quality_fallback",
    "raw_provider",
    "raw_scent_trail",
    "raw_span",
    "recalled",
    "rescued",
    "scent_trail",
    "scored",
    "selected",
    "taxonomy",
    "temporal_coverage",
    "temporal_coverage_raw",
    "timeline",
    "topic_scope",
    "topic_scoped_raw",
    "views",
}


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _duplicate_of(rule_hits: list[dict[str, object]]) -> str | None:
    for hit in rule_hits:
        metadata = _as_dict(hit.get("metadata"))
        duplicate_of = metadata.get("duplicate_of")
        if isinstance(duplicate_of, str) and duplicate_of:
            return duplicate_of
    return None


def _string_values(rule_hits: list[dict[str, object]], key: str) -> list[str]:
    return sorted(
        {
            safe_value
            for hit in rule_hits
            if (safe_value := _safe_string(hit.get(key))) is not None
        }
    )


def _safe_string(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if _is_safe_identifier(value):
        return value
    return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()[:12]


def _is_safe_identifier(value: str) -> bool:
    if len(value) > 128:
        return False
    if value != value.strip():
        return False
    if any(char.isspace() or "\u4e00" <= char <= "\u9fff" for char in value):
        return False
    return value in _SAFE_DIMENSION_IDENTIFIERS


def _cleanup_classification(
    rule_id: str,
    category: str,
    hit_count: int,
    contribution_count: int,
    duplicate_of: str | None,
    protected: bool,
    protected_reason: str,
) -> tuple[str, str, bool, list[str]]:
    cleanup_blockers: list[str] = []
    domain_label_or_taxonomy = ".domain_label" in rule_id or category == "taxonomy_candidate"

    if protected:
        cleanup_action = "keep_protected"
        cleanup_blockers.append(f"protected:{protected_reason or 'unspecified'}")
    elif rule_id.startswith("event_ordering.legacy"):
        cleanup_action = "keep_shadow"
    elif domain_label_or_taxonomy:
        cleanup_action = "migrate_to_taxonomy"
        cleanup_blockers.append("domain_label_taxonomy_migration_required")
    elif duplicate_of is not None:
        cleanup_action = "delete_duplicate"
    elif hit_count == 0:
        cleanup_action = "delete_no_hits"
    elif contribution_count == 0:
        cleanup_action = "delete_no_contribution"
    else:
        cleanup_action = "keep"

    cleanup_phase = "first_pass" if cleanup_action.startswith("delete_") or cleanup_action == "migrate_to_taxonomy" else ""
    safe_to_delete = cleanup_action.startswith("delete_")
    return cleanup_phase, cleanup_action, safe_to_delete, cleanup_blockers


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
        provider_ids = _string_values(rule_hits, "provider_id")
        lifecycle_stages = _string_values(rule_hits, "lifecycle_stage")
        lifecycle_reasons = _string_values(rule_hits, "lifecycle_reason")
        duplicate_of = _duplicate_of(rule_hits) or rule.duplicate_of
        cleanup_phase, cleanup_action, safe_to_delete, cleanup_blockers = _cleanup_classification(
            rule.rule_id,
            rule.category,
            len(rule_hits),
            contribution_count,
            duplicate_of,
            rule.protected,
            rule.protected_reason,
        )
        rows.append(
            {
                "rule_id": rule.rule_id,
                "ability": rule.ability,
                "category": rule.category,
                "module": rule.module,
                "protected": rule.protected,
                "protected_reason": rule.protected_reason,
                "hit_count": len(rule_hits),
                "contribution_count": contribution_count,
                "negative_impact_count": negative_impact_count,
                "candidate_for_deletion": safe_to_delete,
                "evidence_inputs": evidence_inputs,
                "provider_ids": provider_ids,
                "lifecycle_stages": lifecycle_stages,
                "lifecycle_reasons": lifecycle_reasons,
                "duplicate_of": duplicate_of,
                "cleanup_phase": cleanup_phase,
                "cleanup_action": cleanup_action,
                "safe_to_delete": safe_to_delete,
                "cleanup_blockers": cleanup_blockers,
            }
        )
    return rows
