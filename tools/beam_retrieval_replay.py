from __future__ import annotations

import argparse
import json
import os
import sys
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
            "query": query.query,
            "source_span_count": len(getattr(pack, "source_spans", []) or []),
            "coverage_insufficient": bool(coverage.get("coverage_insufficient", False)),
            "pipeline_trace": list(getattr(pack, "debug_trace", []) or []),
        }
        if "rule_hits" in coverage:
            record["rule_hits"] = coverage["rule_hits"]
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


def _summary_for_stdout(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace": report.get("workspace"),
        "split": report.get("split"),
        "query_count": report.get("query_count", 0),
        "summary": report.get("summary", {}),
    }


if __name__ == "__main__":
    main()
