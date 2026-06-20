from __future__ import annotations

import argparse
import json
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
        record = {
            "query_id": query.id,
            "category": canonical_category,
            "beam_category": query.category,
            "source_span_count": len(getattr(pack, "source_spans", []) or []),
            "coverage_insufficient": bool(coverage.get("coverage_insufficient", False)),
            "pipeline_trace": _sanitize_pipeline_trace(getattr(pack, "debug_trace", []) or []),
        }
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
        sanitized_value = _sanitize_metadata_value(item)
        if sanitized_value is not None:
            sanitized[key_text] = sanitized_value
    return sanitized


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _sanitize_metadata(value)
    if isinstance(value, list):
        sanitized_items = [_sanitize_metadata_value(item) for item in value]
        return [item for item in sanitized_items if item is not None]
    if isinstance(value, tuple):
        sanitized_items = [_sanitize_metadata_value(item) for item in value]
        return [item for item in sanitized_items if item is not None]
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        identifier = _sanitize_identifier_string(value)
        if identifier is not None and len(value) <= 64:
            return identifier
        return {"hash": sha1(value.encode("utf-8")).hexdigest()[:12]}
    return None


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
