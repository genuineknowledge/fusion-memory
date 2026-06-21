from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


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
    "high_precision_current_value",
    "hybrid",
    "l0_raw_hybrid",
    "l1_fact_hybrid",
    "l2_event_graph",
    "l3_current_view",
    "l3_entity_profile",
    "legacy_event_ordering_fallback",
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
    "unspecified",
    "views",
}

_PROTECTED_RULE_REASONS = {
    "current_value.stale_history_marker": "high_precision_current_value",
    "exact_match.cjk_phrase": "chinese_recall_precision",
    "event_ordering.legacy_rescue": "legacy_event_ordering_fallback",
}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rule_hits_for_record(record: dict[str, object]) -> list[dict[str, Any]]:
    direct_hits = _as_list(record.get("rule_hits"))
    coverage = _as_dict(record.get("coverage"))
    nested_hits = _as_list(coverage.get("rule_hits"))
    combined_hits: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for item in [*direct_hits, *nested_hits]:
        if not isinstance(item, dict):
            continue
        signature = json.dumps(item, sort_keys=True, default=str)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        combined_hits.append(item)

    return combined_hits


def _safe_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if _is_safe_identifier(value):
        return value
    return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()[:12]


def _safe_int(value: Any) -> int:
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return 0
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _is_safe_identifier(value: str) -> bool:
    if len(value) > 128:
        return False
    if value != value.strip():
        return False
    if any(char.isspace() or "\u4e00" <= char <= "\u9fff" for char in value):
        return False
    return value in _SAFE_DIMENSION_IDENTIFIERS


def _protected_governance_for_rule(rule_id: str) -> tuple[bool, str]:
    protected_reason = _PROTECTED_RULE_REASONS.get(rule_id)
    if protected_reason is not None:
        return True, protected_reason
    if rule_id.startswith("event_ordering.legacy"):
        return True, "legacy_event_ordering_fallback"
    return False, ""


def _explicit_protected_governance(row: dict[str, Any]) -> tuple[bool, str | None]:
    if row.get("explicit_protected") is not True:
        return False, None
    explicit_reason = row.get("explicit_protected_reason")
    if isinstance(explicit_reason, str) and explicit_reason:
        return True, explicit_reason
    return True, None


def _lifecycle_records_for_record(record: dict[str, object]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for container in (
        _as_dict(record.get("candidate_lifecycle")),
        _as_dict(_as_dict(record.get("coverage")).get("candidate_lifecycle")),
        _as_dict(_as_dict(record.get("pipeline_trace")).get("candidate_lifecycle")),
    ):
        for item in _as_list(container.get("records")):
            if isinstance(item, dict):
                records.append(item)
    return records


def _candidate_sources_for_record(record: dict[str, object]) -> list[str]:
    paths = _as_dict(record.get("paths"))
    hybrid = _as_dict(paths.get("hybrid"))
    sources = _as_list(hybrid.get("sources"))
    return sorted({safe_source for source in sources if (safe_source := _safe_string(source)) is not None})


def _provider_summary_for_record(record: dict[str, object]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for trace in (
        _as_dict(record.get("pipeline_trace")),
        _as_dict(_as_dict(record.get("coverage")).get("pipeline_trace")),
    ):
        layers = _as_dict(trace.get("pipeline_layers"))
        candidate_recall = _as_dict(layers.get("CandidateRecall"))
        for item in _as_list(candidate_recall.get("provider_summary")):
            if not isinstance(item, dict):
                continue
            signature = json.dumps(item, sort_keys=True, default=str)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            summaries.append(item)
    return summaries


def _dropped_candidate_ids(record: dict[str, object]) -> set[str]:
    coverage = _as_dict(record.get("coverage"))
    dropped = _as_list(coverage.get("dropped_high_signal_candidates"))
    candidate_ids: set[str] = set()
    for item in dropped:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            candidate_ids.add(candidate_id)
    return candidate_ids


def _ability_for_hit(hit: dict[str, Any]) -> str | None:
    ability = hit.get("ability")
    if isinstance(ability, str) and ability:
        return ability
    metadata = _as_dict(hit.get("metadata"))
    metadata_ability = metadata.get("ability")
    if isinstance(metadata_ability, str) and metadata_ability:
        return metadata_ability
    return None


def _recommendation_for_rule(rule_id: str, hit_count: int, contribution_count: int, categories: set[str]) -> str:
    if rule_id.startswith("event_ordering.legacy"):
        return "legacy_shadow"
    if ".domain_label" in rule_id or "taxonomy_candidate" in categories:
        return "migrate_to_taxonomy"
    if hit_count == 0 or contribution_count == 0:
        return "delete_candidate"
    return "keep"


def _cleanup_classification(
    rule_id: str,
    hit_count: int,
    contribution_count: int,
    recommendation: str,
    duplicate_of: str | None,
    protected: bool,
    protected_reason: str,
) -> tuple[str, str, bool, list[str]]:
    cleanup_blockers: list[str] = []
    domain_label_or_taxonomy = recommendation == "migrate_to_taxonomy"

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


def build_rule_audit(records: list[dict[str, object]]) -> list[dict[str, object]]:
    stats: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        query_id = record.get("query_id")
        query_key = query_id if isinstance(query_id, str) else None
        candidate_sources = _candidate_sources_for_record(record)
        dropped_candidate_ids = _dropped_candidate_ids(record)
        lifecycle_by_candidate_id = {
            candidate_id: lifecycle_record
            for lifecycle_record in _lifecycle_records_for_record(record)
            if isinstance((candidate_id := lifecycle_record.get("candidate_id")), str) and candidate_id
        }

        for hit in _rule_hits_for_record(record):
            rule_id = hit.get("rule_id")
            if not isinstance(rule_id, str) or not rule_id:
                continue
            row = stats.setdefault(
                rule_id,
                {
                    "hit_count": 0,
                    "query_ids": set(),
                    "ability": None,
                    "contribution_count": 0,
                    "negative_impact_count": 0,
                    "dropped_count": 0,
                    "candidate_sources": set(),
                    "evidence_inputs": set(),
                    "categories": set(),
                    "duplicate_of": None,
                    "provider_ids": set(),
                    "lifecycle_stages": set(),
                    "lifecycle_reasons": set(),
                    "explicit_protected": False,
                    "explicit_protected_reason": None,
                },
            )
            row["hit_count"] += 1
            ability = _ability_for_hit(hit)
            if ability is not None and row["ability"] is None:
                row["ability"] = ability
            if query_key is not None:
                row["query_ids"].add(query_key)
            row["candidate_sources"].update(candidate_sources)
            audit_input = record.get("_audit_input")
            if isinstance(audit_input, str) and audit_input:
                row["evidence_inputs"].add(audit_input)

            contributed_candidate_id = hit.get("contributed_candidate_id")
            contributed = (
                isinstance(contributed_candidate_id, str)
                and bool(contributed_candidate_id)
                or hit.get("contributed") is True
                or hit.get("impact") == "selected"
            )
            if contributed:
                row["contribution_count"] += 1
            if hit.get("impact") in {"filtered", "dropped", "misranked"}:
                row["negative_impact_count"] += 1
            if isinstance(contributed_candidate_id, str) and contributed_candidate_id:
                if contributed_candidate_id in dropped_candidate_ids:
                    row["dropped_count"] += 1
                lifecycle_record = lifecycle_by_candidate_id.get(contributed_candidate_id)
                if lifecycle_record is not None:
                    lifecycle_stage = _safe_string(lifecycle_record.get("stage"))
                    if lifecycle_stage is not None:
                        row["lifecycle_stages"].add(lifecycle_stage)
                    lifecycle_reason = _safe_string(lifecycle_record.get("reason_code"))
                    if lifecycle_reason is not None:
                        row["lifecycle_reasons"].add(lifecycle_reason)

            provider_id = _safe_string(hit.get("provider_id"))
            if provider_id is not None:
                row["provider_ids"].add(provider_id)
            lifecycle_stage = _safe_string(hit.get("lifecycle_stage"))
            if lifecycle_stage is not None:
                row["lifecycle_stages"].add(lifecycle_stage)
            lifecycle_reason = _safe_string(hit.get("lifecycle_reason"))
            if lifecycle_reason is not None:
                row["lifecycle_reasons"].add(lifecycle_reason)
            if hit.get("protected") is True:
                row["explicit_protected"] = True
                explicit_reason = _safe_string(hit.get("protected_reason"))
                if explicit_reason is not None and row["explicit_protected_reason"] is None:
                    row["explicit_protected_reason"] = explicit_reason

            metadata = _as_dict(hit.get("metadata"))
            category = metadata.get("category")
            if isinstance(category, str) and category:
                row["categories"].add(category)
            duplicate_of = metadata.get("duplicate_of")
            if isinstance(duplicate_of, str) and duplicate_of and row["duplicate_of"] is None:
                row["duplicate_of"] = duplicate_of

    audit_rows: list[dict[str, object]] = []
    for rule_id in sorted(stats):
        row = stats[rule_id]
        hit_count = int(row["hit_count"])
        contribution_count = int(row["contribution_count"])
        categories = set(row["categories"])
        recommendation = _recommendation_for_rule(rule_id, hit_count, contribution_count, categories)
        duplicate_of = row["duplicate_of"] if isinstance(row["duplicate_of"], str) else None
        explicit_protected, explicit_protected_reason = _explicit_protected_governance(row)
        fallback_protected, fallback_protected_reason = _protected_governance_for_rule(rule_id)
        protected = explicit_protected or fallback_protected
        protected_reason = explicit_protected_reason or fallback_protected_reason or ("unspecified" if protected else "")
        cleanup_phase, cleanup_action, safe_to_delete, cleanup_blockers = _cleanup_classification(
            rule_id,
            hit_count,
            contribution_count,
            recommendation,
            duplicate_of,
            protected,
            protected_reason,
        )
        audit_rows.append(
            {
                "rule_id": rule_id,
                "ability": row["ability"] if isinstance(row["ability"], str) else "",
                "hit_count": hit_count,
                "query_count": len(row["query_ids"]),
                "contribution_count": contribution_count,
                "negative_impact_count": int(row["negative_impact_count"]),
                "dropped_count": int(row["dropped_count"]),
                "candidate_sources": sorted(row["candidate_sources"]),
                "evidence_inputs": sorted(row["evidence_inputs"]),
                "provider_ids": sorted(row["provider_ids"]),
                "lifecycle_stages": sorted(row["lifecycle_stages"]),
                "lifecycle_reasons": sorted(row["lifecycle_reasons"]),
                "recommendation": recommendation,
                "protected": protected,
                "protected_reason": protected_reason,
                "duplicate_of": duplicate_of,
                "cleanup_phase": cleanup_phase,
                "cleanup_action": cleanup_action,
                "safe_to_delete": safe_to_delete,
                "cleanup_blockers": cleanup_blockers,
            }
        )
    return audit_rows


def build_provider_audit(records: list[dict[str, object]]) -> list[dict[str, object]]:
    stats: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        query_id = record.get("query_id")
        query_key = query_id if isinstance(query_id, str) else None
        audit_input = record.get("_audit_input")
        evidence_input = audit_input if isinstance(audit_input, str) and audit_input else None

        for summary in _provider_summary_for_record(record):
            provider_id = _safe_string(summary.get("provider_id"))
            if provider_id is None:
                continue
            row = stats.setdefault(
                provider_id,
                {
                    "source_family": "",
                    "hit_count": 0,
                    "query_ids": set(),
                    "output_count": 0,
                    "output_source_counts": {},
                    "evidence_inputs": set(),
                    "production_default": False,
                    "shadow_only": False,
                    "graph_related": False,
                },
            )
            row["hit_count"] += 1
            if query_key is not None:
                row["query_ids"].add(query_key)
            if evidence_input is not None:
                row["evidence_inputs"].add(evidence_input)

            source_family = _safe_string(summary.get("source_family"))
            if source_family is not None and not row["source_family"]:
                row["source_family"] = source_family

            row["output_count"] += _safe_int(summary.get("output_count"))
            output_source_counts = row["output_source_counts"]
            for source, count in _as_dict(summary.get("output_source_counts")).items():
                safe_source = _safe_string(source)
                if safe_source is None:
                    continue
                output_source_counts[safe_source] = output_source_counts.get(safe_source, 0) + _safe_int(count)

            row["production_default"] = bool(row["production_default"] or summary.get("production_default"))
            row["shadow_only"] = bool(row["shadow_only"] or summary.get("shadow_only"))
            row["graph_related"] = bool(row["graph_related"] or summary.get("graph_related"))

    audit_rows: list[dict[str, object]] = []
    for provider_id in sorted(stats):
        row = stats[provider_id]
        output_source_counts = _as_dict(row["output_source_counts"])
        audit_rows.append(
            {
                "provider_id": provider_id,
                "source_family": row["source_family"],
                "hit_count": int(row["hit_count"]),
                "query_count": len(row["query_ids"]),
                "output_count": int(row["output_count"]),
                "output_source_counts": {source: int(output_source_counts[source]) for source in sorted(output_source_counts)},
                "evidence_inputs": sorted(row["evidence_inputs"]),
                "production_default": bool(row["production_default"]),
                "shadow_only": bool(row["shadow_only"]),
                "graph_related": bool(row["graph_related"]),
            }
        )
    return audit_rows


def _load_records(input_path: Path) -> list[dict[str, object]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    raise ValueError("input JSON must be a top-level list or an object containing a 'records' list")


def _write_json(output_path: Path, audit_rows: list[dict[str, object]]) -> None:
    output_path.write_text(json.dumps(audit_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(csv_path: Path, audit_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "rule_id",
        "ability",
        "hit_count",
        "query_count",
        "contribution_count",
        "negative_impact_count",
        "dropped_count",
        "candidate_sources",
        "evidence_inputs",
        "provider_ids",
        "lifecycle_stages",
        "lifecycle_reasons",
        "recommendation",
        "protected",
        "protected_reason",
        "duplicate_of",
        "cleanup_phase",
        "cleanup_action",
        "safe_to_delete",
        "cleanup_blockers",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            csv_row = dict(row)
            csv_row["candidate_sources"] = ";".join(row["candidate_sources"])
            csv_row["evidence_inputs"] = ";".join(row["evidence_inputs"])
            csv_row["provider_ids"] = ";".join(row["provider_ids"])
            csv_row["lifecycle_stages"] = ";".join(row["lifecycle_stages"])
            csv_row["lifecycle_reasons"] = ";".join(row["lifecycle_reasons"])
            csv_row["cleanup_blockers"] = ";".join(row["cleanup_blockers"])
            writer.writerow(csv_row)


def _write_provider_csv(csv_path: Path, audit_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "provider_id",
        "source_family",
        "hit_count",
        "query_count",
        "output_count",
        "output_source_counts",
        "evidence_inputs",
        "production_default",
        "shadow_only",
        "graph_related",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            csv_row = dict(row)
            output_source_counts = _as_dict(row["output_source_counts"])
            csv_row["output_source_counts"] = ";".join(
                f"{source}:{output_source_counts[source]}" for source in sorted(output_source_counts)
            )
            csv_row["evidence_inputs"] = ";".join(row["evidence_inputs"])
            writer.writerow(csv_row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a retrieval rule audit report from replay records.")
    parser.add_argument("--input", action="append", required=True, help="Replay JSON input. May be repeated.")
    parser.add_argument("--output", required=True, help="Path to audit JSON output.")
    parser.add_argument("--csv", default=None, help="Optional path to audit CSV output.")
    parser.add_argument("--provider-output", default=None, help="Optional path to provider audit JSON output.")
    parser.add_argument("--provider-csv", default=None, help="Optional path to provider audit CSV output.")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for raw_input in args.input:
        input_path = Path(raw_input)
        try:
            loaded_records = _load_records(input_path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            print(
                f"Error: failed to load input {input_path}: {exc}. "
                "Fix the JSON syntax or provide a readable replay JSON file.",
                file=sys.stderr,
            )
            return 1
        for record in loaded_records:
            record["_audit_input"] = str(input_path)
            records.append(record)

    try:
        audit_rows = build_rule_audit(records)
        _write_json(Path(args.output), audit_rows)
        if args.csv:
            _write_csv(Path(args.csv), audit_rows)
        if args.provider_output or args.provider_csv:
            provider_rows = build_provider_audit(records)
            if args.provider_output:
                _write_json(Path(args.provider_output), provider_rows)
            if args.provider_csv:
                _write_provider_csv(Path(args.provider_csv), provider_rows)
    except (OSError, ValueError) as exc:
        print(
            f"Error: failed to write audit output: {exc}. "
            "Check that the output path is writable and its parent directory exists.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
