from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_memory import Scope  # noqa: E402
from fusion_memory.core.runtime_config import memory_service_from_env  # noqa: E402
from fusion_memory.eval.beam_adapter import BeamAdapter, _load_official_beam_dataset  # noqa: E402
from fusion_memory.eval.model_adapters import _pack_for_model  # noqa: E402


def _default_beam_dataset() -> str:
    return os.getenv("BEAM_DATASET", "datasets/BEAM")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect current Fusion Memory evidence/model packs for BEAM query ids.")
    parser.add_argument("--dataset", default=_default_beam_dataset())
    parser.add_argument("--split", default="100k")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--user-id", default="beam_user")
    parser.add_argument("--agent-id", default="fusion_memory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--db", default=os.getenv("FUSION_MEMORY_DB", "postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory"))
    parser.add_argument("--query-ids", default=None, help="Comma-separated BEAM query ids")
    parser.add_argument("--query-ids-file", default=None, help="File containing BEAM query ids")
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-full-pack", action="store_true")
    args = parser.parse_args()

    report = build_probe(args)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


def build_probe(args: argparse.Namespace | SimpleNamespace) -> dict[str, Any]:
    loaded = _load_official_beam_dataset(args.dataset, args.split)
    if not loaded:
        raise ValueError("official BEAM dataset layout is required")
    _, queries = loaded
    query_ids = _selected_ids(args.query_ids, args.query_ids_file)
    if not query_ids:
        raise ValueError("provide --query-ids or --query-ids-file")
    by_id = {query.id: query for query in queries}
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"unknown BEAM query ids: {missing[:5]}")

    backend = "postgres" if str(args.db).startswith(("postgresql://", "postgres://")) else None
    service = memory_service_from_env(args.db, storage_backend=backend)
    scope = Scope(
        workspace_id=args.workspace,
        user_id=args.user_id,
        agent_id=args.agent_id,
        run_id=args.run_id or args.workspace,
        session_id=args.session_id,
    )
    adapter = BeamAdapter(service, scope, split=args.split)
    try:
        records = []
        for query_id in query_ids:
            query = by_id[query_id]
            query_scope = adapter._beam_scope(query.id)
            pack = adapter.retrieval_engine.answer_context(
                query.query,
                query_scope,
                query.category,
                budget={"mode": "balanced"},
            )
            model_pack = _pack_for_model(pack)
            records.append(_probe_record(query, pack, model_pack, include_full_pack=bool(args.include_full_pack)))
        return {
            "workspace": args.workspace,
            "split": args.split,
            "query_count": len(records),
            "records": records,
        }
    finally:
        service.close()


def _probe_record(query: Any, pack: Any, model_pack: dict[str, Any], *, include_full_pack: bool) -> dict[str, Any]:
    record = {
        "query_id": query.id,
        "category": query.category,
        "query": query.query,
        "pack_counts": {
            "facts": len(pack.facts),
            "events": len(pack.events),
            "source_spans": len(pack.source_spans),
            "current_views": len(pack.current_views),
            "entity_profiles": len(pack.entity_profiles),
        },
        "coverage": _compact_mapping(pack.coverage, max_items=40),
        "model_keys": sorted(model_pack.keys()),
        "sequence_items": model_pack.get("sequence_items", []),
        "aggregation_items": model_pack.get("aggregation_items", []),
        "aggregation_summary": model_pack.get("aggregation_summary"),
        "aggregation_answer_candidates": model_pack.get("aggregation_answer_candidates", []),
        "conflict_claims": model_pack.get("conflict_claims", []),
        "summary_highlights": model_pack.get("summary_highlights", []),
        "value_history_summary": model_pack.get("value_history_summary"),
        "temporal_answer_candidates": model_pack.get("temporal_answer_candidates", []),
        "source_span_ids": [span.get("id") for span in pack.source_spans[:50]],
    }
    if query.category == "event_ordering":
        record["anchor_timeline"] = model_pack.get("anchor_timeline", [])
        record["phase_clusters"] = model_pack.get("phase_clusters", [])
    if include_full_pack:
        record["model_pack"] = model_pack
    return record


def _compact_mapping(value: Any, *, max_items: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= max_items:
            out["_truncated"] = True
            break
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
        elif isinstance(item, list):
            out[str(key)] = item[:12]
        elif isinstance(item, dict):
            out[str(key)] = {str(k): v for k, v in list(item.items())[:12]}
        else:
            out[str(key)] = str(item)
    return out


def _selected_ids(query_ids: str | None, query_ids_file: str | None) -> list[str]:
    selected: list[str] = []
    selected.extend(_split_csv(query_ids))
    if query_ids_file:
        raw = Path(query_ids_file).read_text(encoding="utf-8")
        for line in raw.splitlines():
            selected.extend(_split_csv(line.split("#", 1)[0]))
    return list(dict.fromkeys(selected))


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
