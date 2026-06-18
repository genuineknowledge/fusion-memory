from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_memory import Scope  # noqa: E402
from fusion_memory.core.models import Candidate  # noqa: E402
from fusion_memory.core.runtime_config import memory_service_from_env  # noqa: E402
from fusion_memory.eval.beam_adapter import (  # noqa: E402
    BeamAdapter,
    _align_order_items,
    _event_order_reference,
    _kendall_tau_b,
    _load_official_beam_dataset,
    _normalize_order_item,
)
from fusion_memory.eval.model_adapters import _pack_for_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay BEAM event_ordering queries through graph, legacy, and hybrid retrieval paths.")
    parser.add_argument("--dataset", default="/public/home/wwb/datasets/BEAM")
    parser.add_argument("--split", default="100k")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--user-id", default="beam_user")
    parser.add_argument("--agent-id", default="fusion_memory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--db", default=os.getenv("FUSION_MEMORY_DB", "postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--query-ids", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--hybrid-source", choices=("model_pack", "source_spans"), default="model_pack")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.preflight_only:
        report = {"preflight": preflight_replay_environment(args)}
    else:
        report = run_replay(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_summary_for_stdout(report), ensure_ascii=False))


def preflight_replay_environment_from_store(store: Any) -> dict[str, object]:
    try:
        store.list_chronology_topics(
            Scope(workspace_id="preflight", user_id="preflight", agent_id="preflight"),
            include_session=True,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "chronology_" in message and ("does not exist" in message or "no such table" in message):
            return {"chronology_tables_ready": False, "chronology_error": "missing_chronology_tables"}
        return {"chronology_tables_ready": False, "chronology_error": type(exc).__name__}
    return {"chronology_tables_ready": True, "chronology_error": None}


def preflight_replay_environment(args: argparse.Namespace) -> dict[str, object]:
    backend = "postgres" if str(args.db).startswith(("postgresql://", "postgres://")) else None
    service = memory_service_from_env(args.db, storage_backend=backend)
    try:
        return preflight_replay_environment_from_store(service)
    finally:
        service.close()


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    loaded = _load_official_beam_dataset(args.dataset, args.split)
    if not loaded:
        raise ValueError("official BEAM dataset layout is required")
    _, queries = loaded
    selected_ids = set(_split_csv(args.query_ids))
    event_queries = [query for query in queries if query.category == "event_ordering"]
    if selected_ids:
        event_queries = [query for query in event_queries if query.id in selected_ids]
    if args.max_queries:
        event_queries = event_queries[: max(0, args.max_queries)]

    backend = "postgres" if str(args.db).startswith(("postgresql://", "postgres://")) else None
    service = memory_service_from_env(args.db, storage_backend=backend)
    base_scope = Scope(
        workspace_id=args.workspace,
        user_id=args.user_id,
        agent_id=args.agent_id,
        run_id=args.run_id or args.workspace,
        session_id=args.session_id,
    )
    adapter = BeamAdapter(service, base_scope, split=args.split)
    records: list[dict[str, Any]] = []
    started = perf_counter()
    try:
        preflight = preflight_replay_environment_from_store(service)
        for query in event_queries:
            query_scope = adapter._beam_scope(query.id)
            reference = _event_order_reference(query)
            graph_items, graph_sources, graph_fallback = _graph_items(service, query.query, query_scope, args.limit)
            legacy_items, legacy_sources = _legacy_items(service, query.query, query_scope, args.limit)
            hybrid_items, hybrid_sources, hybrid_coverage = _hybrid_items(
                service,
                query.query,
                query_scope,
                args.limit,
                query.category,
                hybrid_source=args.hybrid_source,
            )
            records.append(
                _with_record_diagnostics(
                    {
                        "query_id": query.id,
                        "query": query.query,
                        "reference": reference,
                        "graph_fallback": graph_fallback,
                        "coverage": hybrid_coverage,
                        "paths": {
                            "graph": {
                                "items": graph_items,
                                "sources": graph_sources,
                                "metrics": score_ordering_candidates(reference, graph_items),
                            },
                            "legacy": {
                                "items": legacy_items,
                                "sources": legacy_sources,
                                "metrics": score_ordering_candidates(reference, legacy_items),
                            },
                            "hybrid": {
                                "items": hybrid_items,
                                "sources": hybrid_sources,
                                "coverage": hybrid_coverage,
                                "metrics": score_ordering_candidates(reference, hybrid_items),
                            },
                        },
                    },
                )
            )
        report = {
            "workspace": args.workspace,
            "split": args.split,
            "query_count": len(records),
            "limit": args.limit,
            "elapsed_seconds": perf_counter() - started,
            "preflight": preflight,
            "summary": _aggregate(records),
            "records": records,
        }
        if getattr(args, "gate", False):
            report["gate"] = evaluate_gate(report["summary"])
        return report
    finally:
        service.close()


def score_ordering_candidates(reference: list[str], system: list[str]) -> dict[str, Any]:
    reference_norm = [_normalize_order_item(item) for item in reference if str(item).strip()]
    system_norm = [_normalize_order_item(item) for item in system if str(item).strip()]
    aligned = _align_order_items(reference_norm, system_norm)
    reference_set = set(reference_norm)
    matched_items = [item for item in aligned if item in reference_set]
    matched = len(set(matched_items))
    precision = matched / len(system_norm) if system_norm else 0.0
    recall = matched / len(reference_norm) if reference_norm else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    tau = _kendall_tau(reference_norm, aligned)
    return {
        "reference_count": len(reference_norm),
        "system_count": len(system_norm),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kendall_tau": tau,
        "kendall_tau_norm": (tau + 1.0) / 2.0,
        "aligned": aligned,
    }


def _kendall_tau(reference_norm: list[str], aligned_system_norm: list[str]) -> float:
    if not reference_norm or not aligned_system_norm:
        return 0.0
    union = list(dict.fromkeys(reference_norm + aligned_system_norm))
    tie_rank = len(union) + 1

    def to_rank(sequence: list[str]) -> list[int]:
        ranks = {item: index + 1 for index, item in enumerate(sequence)}
        return [ranks.get(item, tie_rank) for item in union]

    return _kendall_tau_b(to_rank(reference_norm), to_rank(aligned_system_norm))


def _graph_items(service: Any, query: str, scope: Scope, limit: int) -> tuple[list[str], list[str], bool]:
    candidates = list(
        service._event_ordering_graph_selector_candidates(query, scope, limit=limit, include_session=True)
    )
    graph_candidates = [
        candidate
        for candidate in candidates
        if candidate.source == "event_ordering_persisted_graph"
    ]
    return _candidate_texts(graph_candidates, limit), [candidate.source for candidate in graph_candidates[:limit]], _graph_fallback(candidates)


def _legacy_items(service: Any, query: str, scope: Scope, limit: int) -> tuple[list[str], list[str]]:
    plan = service.planner.plan(query, query_type_hint="event_ordering")
    candidates: list[Candidate] = []
    candidates.extend(
        service._event_ordering_episode_recall_candidates(
            query,
            scope,
            plan,
            limit=max(limit * 4, limit + 24),
            include_session=True,
        )
    )
    candidates.extend(
        service._event_ordering_timeline_candidates(
            query,
            plan,
            scope,
            limit=max(limit * 3, limit + 12),
            include_session=True,
        )
    )
    candidates.extend(
        [
            Candidate(
                id=event.event_id,
                type="event",
                text=event.description,
                source="event_timeline_graph",
                scores=scores,
                source_span_ids=event.source_span_ids,
                metadata={},
            )
            for event, scores in service._event_ordering_event_candidates(
                query,
                scope,
                limit=max(limit * 2, 12),
                include_session=True,
            )
        ]
    )
    ordered = _dedupe_candidates(candidates)
    return _candidate_texts(ordered, limit), [candidate.source for candidate in ordered[:limit]]


def _hybrid_items(
    service: Any,
    query: str,
    scope: Scope,
    limit: int,
    category: str,
    hybrid_source: str = "model_pack",
) -> tuple[list[str], list[str], dict[str, Any]]:
    pack = service.answer_context(query, scope, budget={"mode": "benchmark", "query_type_hint": category})
    if hybrid_source == "source_spans":
        items = [
            str(span.get("content") or span.get("conversation_content") or "").strip()
            for span in pack.source_spans
            if isinstance(span, dict) and str(span.get("content") or span.get("conversation_content") or "").strip()
        ]
    else:
        model_pack = _pack_for_model(pack)
        sequence_items = model_pack.get("sequence_items")
        if isinstance(sequence_items, list) and sequence_items:
            items = [_sequence_item_text(item) for item in sequence_items if _sequence_item_text(item)]
        else:
            items = [
                str(span.get("content") or span.get("conversation_content") or "").strip()
                for span in pack.source_spans
                if isinstance(span, dict) and str(span.get("content") or span.get("conversation_content") or "").strip()
            ]
    sources = [str(span.get("candidate_source") or span.get("selector") or "") for span in pack.source_spans if isinstance(span, dict)]
    return items[:limit], sources[:limit], _compact_coverage(pack.coverage)


def _candidate_texts(candidates: list[Candidate], limit: int) -> list[str]:
    return [candidate.text for candidate in candidates[:limit] if candidate.text]


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.id, _normalized_loose(candidate.text))
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _sequence_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("label", "text", "content", "description"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics_records = [record if "graph_fallback" in record else _with_record_diagnostics(dict(record)) for record in records]
    out: dict[str, Any] = {}
    for path in ("graph", "legacy", "hybrid"):
        metrics = [record["paths"][path]["metrics"] for record in diagnostics_records]
        out[path] = {
            "precision": _mean(metrics, "precision"),
            "recall": _mean(metrics, "recall"),
            "f1": _mean(metrics, "f1"),
            "kendall_tau": _mean(metrics, "kendall_tau"),
            "kendall_tau_norm": _mean(metrics, "kendall_tau_norm"),
            "empty_rate": sum(1 for metric in metrics if metric["system_count"] == 0) / len(metrics) if metrics else 0.0,
            "mean_system_count": _mean(metrics, "system_count"),
            "mean_matched": _mean(metrics, "matched"),
        }
    gate = evaluate_gate(out)
    out["graph_vs_legacy_passed"] = gate["passed"]
    out["gate_failures"] = gate["failures"]
    out["path_wins"] = _path_wins(diagnostics_records)
    out["graph_fallback_rate"] = (
        sum(1 for record in diagnostics_records if record.get("graph_fallback")) / len(diagnostics_records)
        if diagnostics_records
        else 0.0
    )
    out["dropped_high_signal_candidate_count"] = sum(
        int(record.get("dropped_high_signal_candidate_count") or 0) for record in diagnostics_records
    )
    out["over_abstract_label_count"] = sum(
        int(record.get("over_abstract_label_count") or 0) for record in diagnostics_records
    )
    return out


def evaluate_gate(summary: dict[str, dict[str, float]]) -> dict[str, object]:
    failures: list[str] = []
    graph = summary.get("graph", {})
    legacy = summary.get("legacy", {})
    hybrid = summary.get("hybrid", {})
    if float(graph.get("f1", 0.0)) < float(legacy.get("f1", 0.0)):
        failures.append("graph_f1_below_legacy")
    if float(graph.get("kendall_tau_norm", 0.0)) < float(legacy.get("kendall_tau_norm", 0.0)):
        failures.append("graph_tau_below_legacy")
    if float(hybrid.get("f1", 0.0)) < float(legacy.get("f1", 0.0)):
        failures.append("hybrid_f1_below_legacy")
    if float(graph.get("empty_rate", 1.0)) > float(legacy.get("empty_rate", 0.0)):
        failures.append("graph_empty_rate_above_legacy")
    return {"passed": not failures, "failures": failures}


def _path_wins(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    paths = ("graph", "legacy", "hybrid")
    out = {
        "f1": {path: 0 for path in paths},
        "kendall_tau_norm": {path: 0 for path in paths},
    }
    for record in records:
        for metric in out:
            scores = {path: float(record["paths"][path]["metrics"].get(metric) or 0.0) for path in paths}
            best = max(scores.values())
            for path, score in scores.items():
                if score == best:
                    out[metric][path] += 1
    return out


def _with_record_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    record.update(_record_diagnostics(record))
    return record


def _record_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    graph = record.get("paths", {}).get("graph", {})
    coverage = record.get("coverage", {}) if isinstance(record.get("coverage"), dict) else {}
    shadow = coverage.get("event_ordering_shadow", {}) if isinstance(coverage.get("event_ordering_shadow"), dict) else {}
    items = [str(item) for item in graph.get("items", []) if str(item).strip()]
    normalized_items = [_normalized_loose(item) for item in items]
    reference_tokens: set[str] = set()
    for item in record.get("reference", []):
        reference_tokens.update(_diagnostic_tokens(str(item)))
    over_abstract_label_count = sum(1 for item in items if _is_over_abstract_label(item))
    topic_drift_count = sum(
        1
        for item in items
        if not _is_over_abstract_label(item)
        and reference_tokens
        and not reference_tokens.intersection(_diagnostic_tokens(item))
    )
    return {
        "topic_drift_count": topic_drift_count,
        "duplicate_label_count": len(normalized_items) - len(set(normalized_items)),
        "graph_empty": int(graph.get("metrics", {}).get("system_count") or 0) == 0,
        "graph_fallback": _record_graph_fallback(record, graph, shadow),
        "dropped_high_signal_candidate_count": len(coverage.get("dropped_high_signal_candidates", []))
        if isinstance(coverage.get("dropped_high_signal_candidates"), list)
        else 0,
        "over_abstract_label_count": over_abstract_label_count,
    }


def _diagnostic_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", value.lower()) if len(token) >= 3}


def _mean(metrics: list[dict[str, Any]], key: str) -> float:
    return sum(float(metric.get(key) or 0.0) for metric in metrics) / len(metrics) if metrics else 0.0


def _compact_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "query_type",
        "selected_candidate_sources",
        "event_ordering_graph",
        "event_ordering_shadow",
        "coverage_insufficient",
        "dropped_high_signal_candidates",
    ]
    return {key: coverage.get(key) for key in keys if key in coverage}


def _record_graph_fallback(record: dict[str, Any], graph: dict[str, Any], shadow: dict[str, Any]) -> bool:
    if "graph_fallback" in record:
        return bool(record.get("graph_fallback"))
    sources = [str(source) for source in graph.get("sources", []) if str(source).strip()]
    if sources:
        return any(source != "event_ordering_persisted_graph" for source in sources)
    if shadow:
        return str(shadow.get("selected_driver") or "none") not in {"graph", "persisted_graph"}
    return False


def _graph_fallback(candidates: list[Candidate]) -> bool:
    if not candidates:
        return True
    for candidate in candidates:
        if candidate.source != "event_ordering_persisted_graph":
            return True
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        telemetry = metadata.get("graph_selector_telemetry") or metadata.get("persisted_graph_telemetry")
        if isinstance(telemetry, dict):
            selected_driver = str(telemetry.get("selected_driver") or "none")
            if selected_driver not in {"graph", "persisted_graph"}:
                return True
    return False


def _summary_for_stdout(report: dict[str, Any]) -> dict[str, Any]:
    if "preflight" in report and "summary" not in report:
        return {
            "preflight": report.get("preflight"),
            "output": "written",
        }
    return {
        "workspace": report.get("workspace"),
        "split": report.get("split"),
        "query_count": report.get("query_count"),
        "summary": report.get("summary"),
        "output": "written",
    }


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalized_loose(value: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", value.lower()))


def _is_over_abstract_label(value: str) -> bool:
    tokens = _diagnostic_tokens(value)
    if not tokens:
        return False
    abstract_tokens = {
        "implementation",
        "summary",
        "phase",
        "work",
        "progress",
        "project",
        "timeline",
        "milestone",
        "event",
        "step",
        "update",
    }
    return tokens.issubset(abstract_tokens)


if __name__ == "__main__":
    main()
