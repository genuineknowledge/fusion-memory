from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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


def _candidate_sources_for_record(record: dict[str, object]) -> list[str]:
    paths = _as_dict(record.get("paths"))
    hybrid = _as_dict(paths.get("hybrid"))
    sources = _as_list(hybrid.get("sources"))
    return sorted({source for source in sources if isinstance(source, str)})


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
    if hit_count == 0 or contribution_count == 0:
        return "delete_candidate"
    if ".domain_label" in rule_id or "taxonomy_candidate" in categories:
        return "migrate_to_taxonomy"
    return "keep"


def _cleanup_classification(
    rule_id: str,
    hit_count: int,
    contribution_count: int,
    recommendation: str,
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
    elif recommendation == "migrate_to_taxonomy":
        cleanup_action = "migrate_to_taxonomy"
    else:
        cleanup_action = "keep"

    cleanup_phase = "first_pass" if cleanup_action.startswith("delete_") or cleanup_action == "migrate_to_taxonomy" else ""
    safe_to_delete = cleanup_action.startswith("delete_")
    return cleanup_phase, cleanup_action, safe_to_delete


def build_rule_audit(records: list[dict[str, object]]) -> list[dict[str, object]]:
    stats: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        query_id = record.get("query_id")
        query_key = query_id if isinstance(query_id, str) else None
        candidate_sources = _candidate_sources_for_record(record)
        dropped_candidate_ids = _dropped_candidate_ids(record)

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
        cleanup_phase, cleanup_action, safe_to_delete = _cleanup_classification(
            rule_id,
            hit_count,
            contribution_count,
            recommendation,
            duplicate_of,
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
                "recommendation": recommendation,
                "duplicate_of": duplicate_of,
                "cleanup_phase": cleanup_phase,
                "cleanup_action": cleanup_action,
                "safe_to_delete": safe_to_delete,
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
        "recommendation",
        "duplicate_of",
        "cleanup_phase",
        "cleanup_action",
        "safe_to_delete",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit_rows:
            csv_row = dict(row)
            csv_row["candidate_sources"] = ";".join(row["candidate_sources"])
            csv_row["evidence_inputs"] = ";".join(row["evidence_inputs"])
            writer.writerow(csv_row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a retrieval rule audit report from replay records.")
    parser.add_argument("--input", action="append", required=True, help="Replay JSON input. May be repeated.")
    parser.add_argument("--output", required=True, help="Path to audit JSON output.")
    parser.add_argument("--csv", default=None, help="Optional path to audit CSV output.")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for raw_input in args.input:
        input_path = Path(raw_input)
        for record in _load_records(input_path):
            record["_audit_input"] = str(input_path)
            records.append(record)
    audit_rows = build_rule_audit(records)
    _write_json(Path(args.output), audit_rows)
    if args.csv:
        _write_csv(Path(args.csv), audit_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
