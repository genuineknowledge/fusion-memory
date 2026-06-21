from __future__ import annotations

import argparse
import json
import math
import os
import sys
from hashlib import sha1
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_memory import Scope  # noqa: E402
from fusion_memory.core.runtime_config import memory_service_from_env  # noqa: E402
from fusion_memory.eval.beam_adapter import _load_official_beam_dataset  # noqa: E402


CATEGORY_ALIASES = {
    "current_value": {"knowledge_update", "preference_following", "instruction_following"},
    "multi_condition": {"multi_session_reasoning", "temporal_reasoning"},
    "zh_recall": {"zh_recall"},
}

_RULE_HIT_SAFE_KEYS = {
    "rule_id",
    "text_hash",
    "contributed_candidate_id",
    "stage",
    "contributed",
    "impact",
}

_RULE_HIT_DIMENSION_KEYS = {
    "provider_id",
    "lifecycle_stage",
    "lifecycle_reason",
    "protected_reason",
}

_TEMPORAL_RELATION_SAFE_KEYS = (
    "relation_type",
    "confidence",
    "reason_code",
    "source_span_id",
    "value_type",
    "value_hash",
    "normalized_date",
    "role_labels",
    "source_span_ids",
)

_SAFE_DIMENSION_IDENTIFIERS = {
    "aggregation_context_support",
    "aggregation_coverage",
    "aggregation_coverage_raw",
    "broad_raw",
    "broad_raw_recall",
    "chinese_recall_precision",
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

_METADATA_KEY_EXCEPTIONS = {
    "text_hash",
}
_PLAINTEXT_METADATA_STRINGS = {
    "answer",
    "candidate_1",
    "candidate_2",
    "current_value",
    "drop_stale_history",
    "event_ordering_coverage",
    "event_ordering_episode_recall",
    "event_ordering_graph_selector",
    "event_ordering_timeline",
    "filter",
    "filtered",
    "kept",
    "l0_raw",
    "l0_raw_hybrid",
    "l1_fact_hybrid",
    "l3_current_view",
    "observed",
    "selected",
    "span_1",
}
_PLAINTEXT_METADATA_KEY_VALUES = {
    "category": {"current_value", "event_ordering", "retrieval"},
    "decision": {"drop_stale_history", "fallback", "kept", "selected"},
    "impact": {"filtered", "observed", "selected"},
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
    "stage": {"evidence_pack_filter", "filter", "search_filter"},
    "text_hash": None,
}

ZH_PROBES = [
    SimpleNamespace(id="zh:1", category="zh_recall", query="我现在使用的数据库是什么？"),
    SimpleNamespace(id="zh:2", category="zh_recall", query="我之前说过偏好的模型是什么？"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay BEAM retrieval categories through Fusion Memory.")
    parser.add_argument("--dataset", default="/public/home/wwb/datasets/BEAM")
    parser.add_argument("--split", default="100k")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--categories", default="current_value,multi_condition,zh_recall")
    parser.add_argument("--user-id", default="beam_user")
    parser.add_argument("--agent-id", default="fusion_memory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--db", default=os.getenv("FUSION_MEMORY_DB", "postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory"))
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    backend = "postgres" if str(args.db).startswith(("postgresql://", "postgres://")) else None
    service = memory_service_from_env(args.db, storage_backend=backend)
    try:
        report = run_replay(
            service,
            base_scope=Scope(
                workspace_id=args.workspace,
                user_id=args.user_id,
                agent_id=args.agent_id,
                run_id=args.run_id or args.workspace,
                session_id=args.session_id,
            ),
            categories=_parse_categories(args.categories),
            output_path=Path(args.output),
            query_limit=args.max_queries,
            dataset=args.dataset,
            split=args.split,
        )
    finally:
        service.close()
    print(json.dumps(_summary_for_stdout(report), ensure_ascii=False))


def run_replay(
    service: Any,
    *,
    base_scope: Scope,
    categories: set[str],
    output_path: Path,
    query_limit: int | None,
    dataset: str | Path = "/public/home/wwb/datasets/BEAM",
    split: str = "100k",
) -> dict[str, Any]:
    selected_categories = set(categories)
    queries = _select_queries(_load_queries(dataset, split), selected_categories)
    if query_limit is not None:
        queries = queries[: max(0, query_limit)]

    records: list[dict[str, Any]] = []
    started = perf_counter()
    for query in queries:
        canonical_category = _canonical_category(query.category, selected_categories) or str(query.category)
        query_scope = _query_scope(base_scope, query.id)
        pack = service.answer_context(
            query.query,
            query_scope,
            budget={"mode": "benchmark", "query_type_hint": canonical_category},
        )
        coverage = _coverage_dict(getattr(pack, "coverage", {}))
        pipeline_trace = _pipeline_trace_from_pack(coverage, getattr(pack, "debug_trace", []) or [])
        record = {
            "query_id": query.id,
            "category": canonical_category,
            "beam_category": query.category,
            "source_span_count": len(getattr(pack, "source_spans", []) or []),
            "coverage_insufficient": bool(coverage.get("coverage_insufficient", False)),
            "pipeline_trace": pipeline_trace,
        }
        provider_audit_coverage = _provider_audit_coverage_from_pipeline_trace(pipeline_trace)
        if provider_audit_coverage:
            record["coverage"] = provider_audit_coverage
        temporal_relation_summary = _temporal_relation_summary_from_coverage(coverage)
        if temporal_relation_summary:
            record["temporal_relation_summary"] = temporal_relation_summary
        lifecycle = _sanitize_candidate_lifecycle(coverage.get("candidate_lifecycle"))
        if lifecycle:
            record["candidate_lifecycle"] = lifecycle
        if "rule_hits" in coverage:
            record["rule_hits"] = _sanitize_rule_hits(coverage["rule_hits"])
        records.append(record)

    report: dict[str, Any] = {
        "benchmark": "BEAM",
        "split": split,
        "workspace": base_scope.workspace_id,
        "query_count": len(records),
        "elapsed_ms": (perf_counter() - started) * 1000,
        "summary": _summarize_records(records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _parse_categories(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_queries(dataset: str | Path, split: str) -> list[Any]:
    loaded = _load_official_beam_dataset(dataset, split)
    if not loaded:
        raise ValueError("official BEAM dataset layout is required")
    _, queries = loaded
    return list(queries)


def _select_queries(queries: list[Any], categories: set[str]) -> list[Any]:
    selected = [query for query in queries if _canonical_category(query.category, categories)]
    if "zh_recall" in categories and not any(str(query.category) == "zh_recall" for query in selected):
        selected.extend(ZH_PROBES)
    return selected


def _canonical_category(beam_category: str, categories: set[str]) -> str | None:
    for category in sorted(categories):
        aliases = CATEGORY_ALIASES.get(category, {category})
        if beam_category == category or beam_category in aliases:
            return category
    return None


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    for record in records:
        category = str(record["category"])
        entry = categories.setdefault(
            category,
            {
                "query_count": 0,
                "coverage_insufficient_count": 0,
                "source_span_count": 0,
            },
        )
        entry["query_count"] += 1
        entry["coverage_insufficient_count"] += 1 if record.get("coverage_insufficient") else 0
        entry["source_span_count"] += int(record.get("source_span_count") or 0)

    for entry in categories.values():
        query_count = int(entry["query_count"])
        entry["coverage_insufficient_rate"] = entry["coverage_insufficient_count"] / query_count if query_count else 0.0
        entry["mean_source_span_count"] = entry["source_span_count"] / query_count if query_count else 0.0
        del entry["source_span_count"]
    return {"categories": categories}


def _query_scope(base_scope: Scope, query_id: str) -> Scope:
    return Scope(
        workspace_id=base_scope.workspace_id,
        user_id=base_scope.user_id,
        agent_id=base_scope.agent_id,
        run_id=base_scope.run_id,
        session_id=_beam_session_id_from_id(query_id) or base_scope.session_id,
        app_id=base_scope.app_id,
    )


def _beam_session_id_from_id(item_id: str | None) -> str | None:
    parts = str(item_id or "").split(":")
    if len(parts) >= 3 and parts[0] == "beam":
        return ":".join(parts[:3])
    return None


def _coverage_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sanitize_rule_hits(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized_hits: list[dict[str, Any]] = []
    for hit in value:
        hit_dict = _object_dict(hit)
        if not hit_dict:
            continue
        sanitized = {key: hit_dict[key] for key in _RULE_HIT_SAFE_KEYS if key in hit_dict}
        for key in _RULE_HIT_DIMENSION_KEYS:
            safe_value = _sanitize_dimension_string(hit_dict.get(key))
            if safe_value is not None:
                sanitized[key] = safe_value
        if isinstance(hit_dict.get("protected"), bool):
            sanitized["protected"] = hit_dict["protected"]
        metadata = _sanitize_metadata(hit_dict.get("metadata"))
        if metadata:
            sanitized["metadata"] = metadata
        sanitized_hits.append(sanitized)
    return sanitized_hits


def _sanitize_pipeline_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized_trace: list[dict[str, Any]] = []
    for entry in value:
        entry_dict = _object_dict(entry)
        if not entry_dict:
            continue
        sanitized = _sanitize_pipeline_trace_entry(entry_dict)
        if sanitized:
            sanitized_trace.append(sanitized)
    return sanitized_trace


def _pipeline_trace_from_pack(coverage: dict[str, Any], debug_trace: Any) -> list[dict[str, Any]]:
    coverage_trace = _sanitize_pipeline_record(coverage.get("pipeline_trace"))
    if coverage_trace:
        return [coverage_trace]
    return _sanitize_pipeline_trace(debug_trace)


def _provider_audit_coverage_from_pipeline_trace(pipeline_trace: list[dict[str, Any]]) -> dict[str, Any]:
    for trace in pipeline_trace:
        layers = _object_dict(trace.get("pipeline_layers"))
        recall = _object_dict(layers.get("CandidateRecall"))
        if recall.get("provider_summary"):
            return {"pipeline_trace": {"pipeline_layers": {"CandidateRecall": recall}}}
    return {}


def _temporal_relation_summary_from_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    summary = _sanitize_temporal_relation_summary(coverage.get("temporal_relation_summary"))
    if summary:
        return summary
    pipeline_trace = _object_dict(coverage.get("pipeline_trace"))
    layers = _object_dict(pipeline_trace.get("pipeline_layers"))
    return _sanitize_temporal_relation_summary(layers.get("TemporalRelations"))


def _sanitize_candidate_lifecycle(value: Any) -> dict[str, Any]:
    data = _object_dict(value)
    if not data:
        return {}
    out: dict[str, Any] = {}
    for key in ("record_count", "contributed_count", "packed_source_span_count"):
        count = _sanitize_count_value(data.get(key))
        if count is not None:
            out[key] = count
    for key in ("stage_counts", "source_counts", "reason_counts"):
        mapping = _sanitize_count_mapping(data.get(key))
        if mapping:
            out[key] = mapping
    records = _sanitize_candidate_lifecycle_records(data.get("records"))
    if records:
        out["records"] = records
    return out


def _sanitize_candidate_lifecycle_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized_records: list[dict[str, Any]] = []
    for record in value:
        record_dict = _object_dict(record)
        if not record_dict:
            continue
        sanitized: dict[str, Any] = {}
        for key in ("candidate_id", "candidate_source", "candidate_type"):
            identifier = _sanitize_identifier_string(record_dict.get(key))
            if identifier is not None:
                sanitized[key] = identifier
        for source_key, target_key in (("stage", "stage"), ("reason_code", "reason_code")):
            dimension = _sanitize_dimension_string(record_dict.get(source_key))
            if dimension is not None:
                sanitized[target_key] = dimension
        source_span_ids = _sanitize_identifier_list(record_dict.get("source_span_ids"))
        if source_span_ids:
            sanitized["source_span_ids"] = source_span_ids
        scores = _sanitize_count_mapping(record_dict.get("scores"))
        if scores:
            sanitized["scores"] = scores
        if isinstance(record_dict.get("contributed"), bool):
            sanitized["contributed"] = record_dict["contributed"]
        temporal_relations = _sanitize_temporal_relations(record_dict.get("temporal_relations"))
        if temporal_relations:
            sanitized["temporal_relations"] = temporal_relations
        if sanitized:
            sanitized_records.append(sanitized)
    return sanitized_records


def _sanitize_pipeline_record(value: Any) -> dict[str, Any]:
    record = _object_dict(value)
    if not record:
        return {}
    layers = _object_dict(record.get("pipeline_layers"))
    recall = _object_dict(layers.get("CandidateRecall"))
    fusion = _object_dict(layers.get("CandidateFusion"))
    evidence = _object_dict(layers.get("EvidencePackBuilder"))
    temporal_relations = _sanitize_temporal_relation_summary(layers.get("TemporalRelations"))
    entry: dict[str, Any] = {"layer": "retrieval"}
    for key in ("query_type", "mode"):
        if key in record:
            sanitized = _sanitize_identifier_string(record[key])
            if sanitized is not None:
                entry[key] = sanitized
    source_counts = _sanitize_count_mapping(recall.get("source_counts"))
    if source_counts:
        entry["source_counts"] = source_counts
    provider_summary = _sanitize_provider_summary_list(recall.get("provider_summary"))
    if provider_summary:
        candidate_recall: dict[str, Any] = {"provider_summary": provider_summary}
        if source_counts:
            candidate_recall["source_counts"] = source_counts
        entry["pipeline_layers"] = {"CandidateRecall": candidate_recall}
    selected_sources = _sanitize_selected_source_names(fusion.get("selected_sources"))
    if selected_sources:
        entry["selected_sources"] = selected_sources
    if "source_span_count" in evidence:
        value = _sanitize_count_value(evidence["source_span_count"])
        if value is not None:
            entry["source_span_count"] = value
    if "coverage_insufficient" in evidence:
        entry["coverage_insufficient"] = bool(evidence["coverage_insufficient"])
    if temporal_relations:
        entry["temporal_relation_summary"] = temporal_relations
    return entry if len(entry) > 1 else {}


def _sanitize_pipeline_trace_entry(entry: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in ("layer", "query_type", "mode", "id", "type", "source"):
        if key in entry:
            value = _sanitize_identifier_string(entry[key])
            if value is not None:
                sanitized[key] = value
    if "scores" in entry:
        scores = _sanitize_count_mapping(entry["scores"])
        if scores:
            sanitized["scores"] = scores
    if "source_span_ids" in entry:
        source_span_ids = _sanitize_identifier_list(entry["source_span_ids"])
        if source_span_ids:
            sanitized["source_span_ids"] = source_span_ids
    if "source_counts" in entry:
        source_counts = _sanitize_count_mapping(entry["source_counts"])
        if source_counts:
            sanitized["source_counts"] = source_counts
    if "selected_sources" in entry:
        selected_sources = _sanitize_selected_sources(entry["selected_sources"])
        if selected_sources:
            sanitized["selected_sources"] = selected_sources
    if "source_span_count" in entry:
        value = _sanitize_count_value(entry["source_span_count"])
        if value is not None:
            sanitized["source_span_count"] = value
    if "coverage_insufficient" in entry:
        sanitized["coverage_insufficient"] = bool(entry["coverage_insufficient"])
    if "temporal_relation_summary" in entry:
        temporal_relation_summary = _sanitize_temporal_relation_summary(entry["temporal_relation_summary"])
        if temporal_relation_summary:
            sanitized["temporal_relation_summary"] = temporal_relation_summary
    rule_hit_count = _rule_hit_count(entry)
    if rule_hit_count is not None:
        sanitized["rule_hit_count"] = rule_hit_count
    return sanitized


def _sanitize_count_mapping(value: Any) -> dict[str, int | float | bool]:
    mapping = _object_dict(value)
    if not mapping:
        return {}
    sanitized: dict[str, int | float | bool] = {}
    for key, item in mapping.items():
        key_text = str(key)
        if _metadata_key_contains_raw_text(key_text):
            continue
        safe_key = _sanitize_identifier_string(key_text)
        if safe_key is None:
            continue
        count = _sanitize_count_value(item)
        if count is not None:
            sanitized[safe_key] = count
    return sanitized


def _sanitize_provider_summary_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized_summary: list[dict[str, Any]] = []
    for item in value:
        summary = _object_dict(item)
        if not summary:
            continue
        sanitized = _sanitize_provider_summary(summary)
        if sanitized:
            sanitized_summary.append(sanitized)
    return sanitized_summary


def _sanitize_provider_summary(summary: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in ("provider_id", "source_family"):
        value = _sanitize_dimension_string(summary.get(key))
        if value is not None:
            sanitized[key] = value
    if "output_count" in summary:
        sanitized["output_count"] = _safe_int_count(summary.get("output_count"))
    output_source_counts = _sanitize_provider_output_source_counts(summary.get("output_source_counts"))
    if output_source_counts:
        sanitized["output_source_counts"] = output_source_counts
    for key in ("production_default", "shadow_only", "graph_related"):
        if key in summary:
            sanitized[key] = bool(summary[key])
    return sanitized


def _sanitize_provider_output_source_counts(value: Any) -> dict[str, int]:
    mapping = _object_dict(value)
    if not mapping:
        return {}
    sanitized: dict[str, int] = {}
    for key, item in mapping.items():
        safe_key = _sanitize_dimension_string(key)
        if safe_key is not None:
            sanitized[safe_key] = _safe_int_count(item)
    return sanitized


def _safe_int_count(value: Any) -> int:
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return 0
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _sanitize_temporal_relation_summary(value: Any) -> dict[str, Any]:
    summary = _object_dict(value)
    if not summary:
        return {}
    sanitized: dict[str, Any] = {}
    relation_count = _sanitize_count_value(summary.get("relation_count"))
    if isinstance(relation_count, int):
        sanitized["relation_count"] = relation_count
    relation_types = _sanitize_identifier_list(summary.get("relation_types"))
    if relation_types:
        sanitized["relation_types"] = relation_types
    source_span_count = _sanitize_count_value(summary.get("source_span_count"))
    if isinstance(source_span_count, int):
        sanitized["source_span_count"] = source_span_count
    reason_codes = _sanitize_identifier_list(summary.get("reason_codes"))
    if reason_codes:
        sanitized["reason_codes"] = reason_codes
    role_labels = _sanitize_identifier_list(summary.get("role_labels"))
    if role_labels:
        sanitized["role_labels"] = role_labels
    source_span_ids = _sanitize_identifier_list(summary.get("source_span_ids"))
    if source_span_ids:
        sanitized["source_span_ids"] = source_span_ids
    return sanitized


def _sanitize_temporal_relations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized_relations: list[dict[str, Any]] = []
    for relation in value:
        relation_dict = _object_dict(relation)
        if not relation_dict:
            continue
        sanitized: dict[str, Any] = {}
        for key in _TEMPORAL_RELATION_SAFE_KEYS:
            item = relation_dict.get(key)
            if item is None:
                continue
            if key in {"role_labels", "source_span_ids"}:
                identifiers = _sanitize_identifier_list(item)
                if identifiers:
                    sanitized[key] = identifiers
                continue
            if key == "confidence":
                confidence = _sanitize_count_value(item)
                if isinstance(confidence, bool):
                    continue
                if isinstance(confidence, int | float):
                    sanitized[key] = confidence
                continue
            value_text = _sanitize_identifier_string(item)
            if value_text is not None:
                sanitized[key] = value_text
        if sanitized:
            sanitized_relations.append(sanitized)
    return sanitized_relations


def _sanitize_selected_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized_sources: list[dict[str, Any]] = []
    for source in value:
        source_dict = _object_dict(source)
        if not source_dict:
            continue
        sanitized: dict[str, Any] = {}
        for key, item in source_dict.items():
            key_text = str(key)
            if _metadata_key_contains_raw_text(key_text):
                continue
            sanitized_value = _sanitize_trace_scalar(item)
            if sanitized_value is not None:
                sanitized[key_text] = sanitized_value
        if sanitized:
            sanitized_sources.append(sanitized)
    return sanitized_sources


def _sanitize_selected_source_names(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized_sources: list[dict[str, Any]] = []
    for source in value:
        sanitized = _sanitize_identifier_string(source)
        if sanitized is not None:
            sanitized_sources.append({"source": sanitized})
    return sanitized_sources


def _rule_hit_count(entry: dict[str, Any]) -> int | None:
    for key in ("rule_hit_count", "rule_hits_count"):
        if key in entry:
            value = _sanitize_count_value(entry[key])
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
    rule_hits = entry.get("rule_hits")
    if isinstance(rule_hits, list):
        return len(rule_hits)
    return None


def _sanitize_trace_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    return _sanitize_identifier_string(value)


def _sanitize_count_value(value: Any) -> int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    return None


def _sanitize_identifier_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if any(character.isspace() for character in value):
        return None
    if any("\u4e00" <= character <= "\u9fff" for character in value):
        return None
    if not all(character.isalnum() or character in "._:+/@=-" for character in value):
        return None
    return value


def _sanitize_identifier_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    sanitized: list[str] = []
    for item in value:
        identifier = _sanitize_identifier_string(item)
        if identifier is not None:
            sanitized.append(identifier)
    return sanitized


def _sanitize_dimension_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if _is_safe_dimension_identifier(value):
        return value
    return sha1(repr(value).encode("utf-8")).hexdigest()[:12]


def _is_safe_dimension_identifier(value: str) -> bool:
    if len(value) > 128:
        return False
    if value != value.strip():
        return False
    if any(character.isspace() or "\u4e00" <= character <= "\u9fff" for character in value):
        return False
    return value in _SAFE_DIMENSION_IDENTIFIERS


def _object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    attrs = getattr(value, "__dict__", None)
    return attrs if isinstance(attrs, dict) else {}


def _sanitize_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if _metadata_key_contains_raw_text(key_text):
            continue
        sanitized_value = _sanitize_metadata_value(item, key_text)
        if sanitized_value is not None:
            sanitized[key_text] = sanitized_value
    return sanitized


def _sanitize_metadata_value(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return _sanitize_metadata(value)
    if isinstance(value, list):
        sanitized_items = [_sanitize_metadata_value(item, key) for item in value]
        return [item for item in sanitized_items if item is not None]
    if isinstance(value, tuple):
        sanitized_items = [_sanitize_metadata_value(item, key) for item in value]
        return [item for item in sanitized_items if item is not None]
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        identifier = _sanitize_identifier_string(value)
        if identifier is not None and len(value) <= 64 and _is_plaintext_metadata_string(value, key):
            return identifier
        return {"hash": sha1(value.encode("utf-8")).hexdigest()[:12]}
    return None


def _is_plaintext_metadata_string(value: str, key: str | None = None) -> bool:
    normalized_key = (key or "").lower()
    if normalized_key in _PLAINTEXT_METADATA_KEY_VALUES and _PLAINTEXT_METADATA_KEY_VALUES[normalized_key] is None:
        return True
    allowed_for_key = _PLAINTEXT_METADATA_KEY_VALUES.get(normalized_key)
    if allowed_for_key is not None:
        return value in allowed_for_key
    return value in _PLAINTEXT_METADATA_STRINGS


def _metadata_key_contains_raw_text(key: str) -> bool:
    normalized = key.lower()
    if normalized in _METADATA_KEY_EXCEPTIONS:
        return False
    if normalized in _SENSITIVE_METADATA_KEYS:
        return True
    return any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS)


def _summary_for_stdout(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace": report.get("workspace"),
        "split": report.get("split"),
        "query_count": report.get("query_count", 0),
        "summary": report.get("summary", {}),
    }


if __name__ == "__main__":
    main()
