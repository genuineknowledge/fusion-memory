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
from fusion_memory.core.text import stable_hash  # noqa: E402
from fusion_memory.eval.beam_adapter import (  # noqa: E402
    BeamAdapter,
    _align_order_items,
    _event_order_reference,
    _kendall_tau_b,
    _load_official_beam_dataset,
    _normalize_order_item,
)
from fusion_memory.eval.model_adapters import _pack_for_model  # noqa: E402


REPLAY_PATHS = ("graph", "legacy", "dual", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay BEAM event_ordering queries through graph, legacy, dual, and hybrid retrieval paths.")
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
    parser.add_argument("--preflight-min-topics", type=int, default=1)
    parser.add_argument("--preflight-min-nodes", type=int, default=2)
    parser.add_argument("--preflight-min-edges", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("graph_only", "legacy_only", "dual_only", "graph_legacy", "graph_dual_legacy", "hybrid", "all"),
        default="all",
    )
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


def preflight_replay_environment_from_store(
    store: Any,
    *,
    scope: Scope | None = None,
    min_topics: int = 0,
    min_nodes: int = 0,
    min_edges: int = 0,
) -> dict[str, object]:
    chronology_store = getattr(store, "store", store)
    preflight_scope = scope or Scope(workspace_id="preflight", user_id="preflight", agent_id="preflight")
    try:
        topics = chronology_store.list_chronology_topics(preflight_scope, include_session=True)
        nodes = (
            chronology_store.list_chronology_event_nodes(preflight_scope, include_session=True)
            if min_nodes or min_edges
            else []
        )
        edges = chronology_store.list_chronology_event_edges([node.node_id for node in nodes]) if min_edges else []
    except Exception as exc:
        message = str(exc).lower()
        if "chronology_" in message and ("does not exist" in message or "no such table" in message):
            return {
                "status": "failure",
                "error": "missing_chronology_tables",
                "chronology_tables_ready": False,
                "chronology_error": "missing_chronology_tables",
            }
        return {
            "status": "failure",
            "error": type(exc).__name__,
            "chronology_tables_ready": False,
                "chronology_error": type(exc).__name__,
            }
    counts = {"topics": len(topics), "nodes": len(nodes), "edges": len(edges)}
    persisted_graph_ready = (
        counts["topics"] >= min_topics
        and counts["nodes"] >= min_nodes
        and counts["edges"] >= min_edges
    )
    if min_topics or min_nodes or min_edges:
        if not persisted_graph_ready:
            return {
                "status": "failure",
                "error": "persisted_graph_not_backfilled",
                "chronology_tables_ready": True,
                "chronology_error": None,
                "persisted_graph_ready": False,
                "chronology_counts": counts,
                "chronology_thresholds": {"topics": min_topics, "nodes": min_nodes, "edges": min_edges},
            }
    return {
        "status": "ok",
        "error": None,
        "chronology_tables_ready": True,
        "chronology_error": None,
        "persisted_graph_ready": persisted_graph_ready,
        "chronology_counts": counts,
    }


def preflight_replay_query_scopes_from_store(
    store: Any,
    scopes: list[Scope],
    *,
    min_topics: int = 1,
    min_nodes: int = 2,
    min_edges: int = 1,
) -> dict[str, object]:
    chronology_store = getattr(store, "store", store)
    if not all(
        hasattr(chronology_store, name)
        for name in ("list_chronology_topics", "list_chronology_event_nodes", "list_chronology_event_edges")
    ):
        return {
            "checked": 0,
            "ready": 0,
            "not_ready": 0,
            "empty": 0,
            "empty_rate": 0.0,
            "ready_rate": 0.0,
            "thresholds": {"topics": min_topics, "nodes": min_nodes, "edges": min_edges},
            "failure_samples": [],
            "status": "unavailable",
            "error": "chronology_methods_unavailable",
        }
    checked = 0
    ready = 0
    empty = 0
    failures: list[dict[str, object]] = []
    for scope in scopes:
        checked += 1
        counts = _chronology_counts(chronology_store, scope)
        is_ready = (
            counts["topics"] >= min_topics
            and counts["nodes"] >= min_nodes
            and counts["edges"] >= min_edges
        )
        if is_ready:
            ready += 1
        if counts["topics"] == 0 and counts["nodes"] == 0 and counts["edges"] == 0:
            empty += 1
        if not is_ready and len(failures) < 10:
            failures.append(
                {
                    "workspace_id": scope.workspace_id,
                    "run_id": scope.run_id,
                    "session_id": scope.session_id,
                    "counts": counts,
                }
            )
    return {
        "checked": checked,
        "ready": ready,
        "not_ready": checked - ready,
        "empty": empty,
        "empty_rate": empty / checked if checked else 0.0,
        "ready_rate": ready / checked if checked else 0.0,
        "thresholds": {"topics": min_topics, "nodes": min_nodes, "edges": min_edges},
        "failure_samples": failures,
    }


def _chronology_counts(store: Any, scope: Scope) -> dict[str, int]:
    topics = store.list_chronology_topics(scope, include_session=True)
    nodes = store.list_chronology_event_nodes(scope, include_session=True)
    edges = store.list_chronology_event_edges([node.node_id for node in nodes])
    return {"topics": len(topics), "nodes": len(nodes), "edges": len(edges)}


def preflight_replay_environment(args: argparse.Namespace) -> dict[str, object]:
    backend = "postgres" if str(args.db).startswith(("postgresql://", "postgres://")) else None
    try:
        service = memory_service_from_env(args.db, storage_backend=backend)
    except Exception as exc:
        return {
            "status": "failure",
            "error": type(exc).__name__,
            "chronology_tables_ready": False,
            "chronology_error": type(exc).__name__,
        }
    try:
        scope = Scope(
            workspace_id=args.workspace,
            user_id=args.user_id,
            agent_id=args.agent_id,
            run_id=args.run_id or args.workspace,
            session_id=args.session_id,
        )
        return preflight_replay_environment_from_store(
            service,
            scope=scope,
            min_topics=getattr(args, "preflight_min_topics", 1),
            min_nodes=getattr(args, "preflight_min_nodes", 2),
            min_edges=getattr(args, "preflight_min_edges", 1),
        )
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
        preflight = preflight_replay_environment_from_store(
            service,
            scope=base_scope,
            min_topics=getattr(args, "preflight_min_topics", 1),
            min_nodes=getattr(args, "preflight_min_nodes", 2),
            min_edges=getattr(args, "preflight_min_edges", 1),
        )
        query_scope_preflight = preflight_replay_query_scopes_from_store(
            service,
            [adapter._beam_scope(query.id) for query in event_queries],
            min_topics=getattr(args, "preflight_min_topics", 1),
            min_nodes=getattr(args, "preflight_min_nodes", 2),
            min_edges=getattr(args, "preflight_min_edges", 1),
        )
        active_paths = _active_paths(args.mode)
        for query in event_queries:
            query_scope = adapter._beam_scope(query.id)
            reference = _event_order_reference(query)
            graph_items: list[str] = []
            graph_sources: list[str] = []
            graph_fallback = False
            legacy_items: list[str] = []
            legacy_sources: list[str] = []
            dual_items: list[str] = []
            dual_sources: list[str] = []
            hybrid_items: list[str] = []
            hybrid_sources: list[str] = []
            hybrid_coverage: dict[str, Any] = {}

            if args.mode in {"graph_only", "graph_legacy", "graph_dual_legacy", "all"}:
                graph_items, graph_sources, graph_fallback = _graph_items(service, query.query, query_scope, args.limit)
            if args.mode in {"legacy_only", "graph_legacy", "graph_dual_legacy", "all"}:
                legacy_items, legacy_sources = _legacy_items(service, query.query, query_scope, args.limit)
            if args.mode in {"dual_only", "graph_dual_legacy", "all"}:
                dual_items, dual_sources = _dual_graph_legacy_items(service, query.query, query_scope, args.limit)
            if args.mode in {"hybrid", "all"}:
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
                        "query_hash": stable_hash(query.query),
                        "query_length": len(query.query),
                        "reference": reference,
                        "bucket": _event_ordering_bucket(query.query, reference),
                        "graph_fallback": graph_fallback,
                        "coverage": hybrid_coverage,
                        "paths": _build_record_paths(
                            reference,
                            active_paths,
                            graph_items=graph_items,
                            graph_sources=graph_sources,
                            legacy_items=legacy_items,
                            legacy_sources=legacy_sources,
                            dual_items=dual_items,
                            dual_sources=dual_sources,
                            hybrid_items=hybrid_items,
                            hybrid_sources=hybrid_sources,
                            hybrid_coverage=hybrid_coverage,
                        ),
                    },
                )
            )
        output_path = getattr(args, "output", "replay.json")
        report = {
            "workspace": args.workspace,
            "split": args.split,
            "query_count": len(records),
            "limit": args.limit,
            "elapsed_seconds": perf_counter() - started,
            "preflight": preflight,
            "query_scope_preflight": query_scope_preflight,
            "summary": _aggregate(records),
            "bucket_summary": {
                path: _bucket_summary(records, path)
                for path in REPLAY_PATHS
            },
            "route_summary": _route_summary(records),
            "replay_config": {
                "mode": args.mode,
                "hybrid_source": args.hybrid_source,
                "limit": args.limit,
                "artifact_commands": {
                    "rule_audit_json": (
                        f"python3 tools/rule_audit.py --input {output_path} "
                        "--output artifacts/rule-audit.json"
                    ),
                    "rule_audit_csv": (
                        f"python3 tools/rule_audit.py --input {output_path} "
                        "--output artifacts/rule-audit.json --csv artifacts/rule-audit.csv"
                    ),
                },
            },
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
    candidates.extend(_event_ordering_coverage_candidates_for_replay(service, query, scope, limit))
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


def _dual_graph_legacy_items(service: Any, query: str, scope: Scope, limit: int) -> tuple[list[str], list[str]]:
    graph_candidates = [
        candidate
        for candidate in service._event_ordering_graph_selector_candidates(query, scope, limit=limit, include_session=True)
        if candidate.source == "event_ordering_persisted_graph"
    ]
    plan = service.planner.plan(query, query_type_hint="event_ordering")
    legacy_candidates: list[Candidate] = []
    legacy_candidates.extend(_event_ordering_coverage_candidates_for_replay(service, query, scope, limit))
    legacy_candidates.extend(
        service._event_ordering_episode_recall_candidates(
            query,
            scope,
            plan,
            limit=max(limit * 4, limit + 24),
            include_session=True,
        )
    )
    legacy_candidates.extend(
        service._event_ordering_timeline_candidates(
            query,
            plan,
            scope,
            limit=max(limit * 3, limit + 12),
            include_session=True,
        )
    )
    legacy_candidates.extend(
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
    ordered = _rank_legacy_candidates_by_graph_order(_dedupe_candidates(legacy_candidates), graph_candidates)
    return _candidate_texts(ordered, limit), [candidate.source for candidate in ordered[:limit]]


def _event_ordering_coverage_candidates_for_replay(service: Any, query: str, scope: Scope, limit: int) -> list[Candidate]:
    coverage_fn = getattr(service, "_event_ordering_coverage_candidates", None)
    if coverage_fn is None:
        return []
    return list(
        coverage_fn(
            query,
            scope,
            limit=max(limit * 3, limit + 12),
            include_session=True,
        )
    )


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


def _dedupe_dual_candidates(candidates: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        text_key = _normalized_loose(candidate.text)
        span_ids = getattr(candidate, "source_span_ids", []) or []
        span_key = ",".join(str(span_id) for span_id in span_ids if str(span_id).strip())
        key = (span_key, text_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _rank_legacy_candidates_by_graph_order(
    legacy_candidates: list[Candidate],
    graph_candidates: list[Candidate],
) -> list[Candidate]:
    if not legacy_candidates:
        return _dedupe_dual_candidates(graph_candidates)
    if not graph_candidates:
        return legacy_candidates

    matched_legacy_ids: set[int] = set()
    merged: list[Candidate] = []
    for graph_index, graph_candidate in enumerate(graph_candidates):
        best: tuple[float, int] | None = None
        for legacy_index, legacy_candidate in enumerate(legacy_candidates):
            if id(legacy_candidate) in matched_legacy_ids:
                continue
            score = _candidate_alignment_score(graph_candidate, legacy_candidate)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, legacy_index)
        if best is not None:
            legacy_candidate = legacy_candidates[best[1]]
            matched_legacy_ids.add(id(legacy_candidate))
            merged.append(legacy_candidate)
        else:
            merged.append(graph_candidate)

    merged.extend(candidate for candidate in legacy_candidates if id(candidate) not in matched_legacy_ids)
    return _dedupe_dual_candidates(merged)


def _candidate_alignment_score(graph_candidate: Candidate, legacy_candidate: Candidate) -> float:
    graph_span_ids = {str(span_id) for span_id in (getattr(graph_candidate, "source_span_ids", []) or []) if str(span_id).strip()}
    legacy_span_ids = {str(span_id) for span_id in (getattr(legacy_candidate, "source_span_ids", []) or []) if str(span_id).strip()}
    if graph_span_ids and legacy_span_ids and graph_span_ids & legacy_span_ids:
        return 2.0
    graph_tokens = _alignment_tokens(str(getattr(graph_candidate, "text", "") or ""))
    legacy_tokens = _alignment_tokens(str(getattr(legacy_candidate, "text", "") or ""))
    if not graph_tokens or not legacy_tokens:
        return 0.0
    coverage = len(graph_tokens & legacy_tokens) / len(graph_tokens)
    if coverage < 0.50:
        return 0.0
    return coverage


def _alignment_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", value.lower()) if len(token) >= 3}


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
    for path in REPLAY_PATHS:
        metrics = _active_metrics(diagnostics_records, path)
        if not metrics:
            out[path] = {"active": False, "count": 0}
            continue
        out[path] = {
            "active": True,
            "count": len(metrics),
            "precision": _mean(metrics, "precision"),
            "recall": _mean(metrics, "recall"),
            "f1": _mean(metrics, "f1"),
            "kendall_tau": _mean(metrics, "kendall_tau"),
            "kendall_tau_norm": _mean(metrics, "kendall_tau_norm"),
            "empty_rate": sum(1 for metric in metrics if metric["system_count"] == 0) / len(metrics) if metrics else 0.0,
            "mean_system_count": _mean(metrics, "system_count"),
            "mean_matched": _mean(metrics, "matched"),
        }
    if _summary_path_active(out.get("dual", {})) and _summary_path_active(out.get("legacy", {})):
        out["dual_lift_over_legacy_f1"] = float(out.get("dual", {}).get("f1") or 0.0) - float(out.get("legacy", {}).get("f1") or 0.0)
        out["dual_lift_over_legacy_tau"] = float(out.get("dual", {}).get("kendall_tau_norm") or 0.0) - float(out.get("legacy", {}).get("kendall_tau_norm") or 0.0)
    else:
        out["dual_lift_over_legacy_f1"] = None
        out["dual_lift_over_legacy_tau"] = None
    out["cluster_diagnostics"] = _cluster_diagnostics(records)
    gate = evaluate_gate(out)
    out["graph_vs_legacy_passed"] = _graph_vs_legacy_passed(out)
    out["dual_vs_legacy_passed"] = _dual_vs_legacy_passed(out)
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


def _cluster_diagnostics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    expanded_query_count = 0
    selected_topic_counts: list[float] = []
    for record in records:
        coverage = record.get("coverage", {}) if isinstance(record.get("coverage"), dict) else {}
        graph_coverage = coverage.get("event_ordering_graph", {}) if isinstance(coverage.get("event_ordering_graph"), dict) else {}
        expanded_topic_ids = graph_coverage.get("cluster_expanded_topic_ids") or []
        if expanded_topic_ids:
            expanded_query_count += 1
        selected_topic_count = graph_coverage.get("selected_topic_count")
        if isinstance(selected_topic_count, (int, float)):
            selected_topic_counts.append(float(selected_topic_count))
    return {
        "expanded_query_count": expanded_query_count,
        "mean_selected_topic_count": sum(selected_topic_counts) / len(selected_topic_counts) if selected_topic_counts else 0.0,
    }


def _event_ordering_bucket(query: str, reference: list[str]) -> str:
    lower = query.lower()
    joined = " ".join(reference).lower()
    if re.search(r"\b(?:first|then|after|before|next|later)\b|首先|然后|之后|之前", joined):
        return "explicit_order"
    if re.search(r"\b20\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|月\s*\d+\s*日", joined):
        return "dated"
    if len(reference) >= 5 or re.search(r"\b(?:across|throughout|different aspects|long timeline)\b", lower):
        return "long_mixed_topic"
    if re.search(r"[\u4e00-\u9fff]", query):
        return "chinese"
    return "implicit_order"


def _bucket_summary(records: list[dict[str, Any]], path: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        path_info = record.get("paths", {}).get(path, {})
        if not _path_is_active(path_info):
            continue
        bucket = str(record.get("bucket") or "unknown")
        grouped.setdefault(bucket, []).append(path_info.get("metrics", {}))
    summary: dict[str, dict[str, float | int]] = {}
    for bucket, metrics in grouped.items():
        summary[bucket] = {
            "count": len(metrics),
            "precision": _mean(metrics, "precision"),
            "recall": _mean(metrics, "recall"),
            "f1": _mean(metrics, "f1"),
            "kendall_tau_norm": _mean(metrics, "kendall_tau_norm"),
        }
    return summary


def _route_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not _path_is_active(record.get("paths", {}).get("hybrid", {})):
            continue
        coverage = record.get("coverage", {})
        shadow = coverage.get("event_ordering_shadow", {}) if isinstance(coverage, dict) else {}
        route = str(shadow.get("selected_driver") or "unreported")
        counts[route] = counts.get(route, 0) + 1
    return counts


def evaluate_gate(summary: dict[str, dict[str, float]]) -> dict[str, object]:
    failures: list[str] = []
    graph = summary.get("graph", {})
    legacy = summary.get("legacy", {})
    dual = summary.get("dual", {})
    hybrid = summary.get("hybrid", {})
    if not all(_summary_path_active(path_summary) for path_summary in (graph, legacy)):
        return {"passed": False, "failures": ["insufficient_active_paths"]}
    if float(graph.get("f1", 0.0)) < float(legacy.get("f1", 0.0)):
        failures.append("graph_f1_below_legacy")
    if float(graph.get("kendall_tau_norm", 0.0)) < float(legacy.get("kendall_tau_norm", 0.0)):
        failures.append("graph_tau_below_legacy")
    if _summary_path_active(hybrid) and float(hybrid.get("f1", 0.0)) < float(legacy.get("f1", 0.0)):
        failures.append("hybrid_f1_below_legacy")
    if _summary_path_active(dual):
        if float(dual.get("f1", 0.0)) < float(legacy.get("f1", 0.0)):
            failures.append("dual_f1_below_legacy")
        if float(dual.get("kendall_tau_norm", 0.0)) < float(legacy.get("kendall_tau_norm", 0.0)):
            failures.append("dual_tau_below_legacy")
        if float(dual.get("empty_rate", 1.0)) > float(legacy.get("empty_rate", 0.0)):
            failures.append("dual_empty_rate_above_legacy")
    if float(graph.get("empty_rate", 1.0)) > float(legacy.get("empty_rate", 0.0)):
        failures.append("graph_empty_rate_above_legacy")
    return {"passed": not failures, "failures": failures}


def _graph_vs_legacy_passed(summary: dict[str, dict[str, float]]) -> bool:
    graph = summary.get("graph", {})
    legacy = summary.get("legacy", {})
    if not all(_summary_path_active(path_summary) for path_summary in (graph, legacy)):
        return False
    return (
        float(graph.get("f1", 0.0)) >= float(legacy.get("f1", 0.0))
        and float(graph.get("kendall_tau_norm", 0.0)) >= float(legacy.get("kendall_tau_norm", 0.0))
        and float(graph.get("empty_rate", 1.0)) <= float(legacy.get("empty_rate", 0.0))
    )


def _dual_vs_legacy_passed(summary: dict[str, dict[str, float]]) -> bool:
    dual = summary.get("dual", {})
    legacy = summary.get("legacy", {})
    if not all(_summary_path_active(path_summary) for path_summary in (dual, legacy)):
        return False
    return (
        float(dual.get("f1", 0.0)) >= float(legacy.get("f1", 0.0))
        and float(dual.get("kendall_tau_norm", 0.0)) >= float(legacy.get("kendall_tau_norm", 0.0))
        and float(dual.get("empty_rate", 1.0)) <= float(legacy.get("empty_rate", 0.0))
    )


def _path_wins(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    active_paths = sorted(
        {
            path
            for record in records
            for path in REPLAY_PATHS
            if _path_is_active(record.get("paths", {}).get(path, {}))
        }
    )
    out = {
        "f1": {path: 0 for path in active_paths},
        "kendall_tau_norm": {path: 0 for path in active_paths},
    }
    for record in records:
        for metric in out:
            scores = {
                path: float(record["paths"][path]["metrics"].get(metric) or 0.0)
                for path in active_paths
                if _path_is_active(record.get("paths", {}).get(path, {}))
            }
            if not scores:
                continue
            best = max(scores.values())
            for path, score in scores.items():
                if score == best:
                    out[metric][path] += 1
    return out


def _active_paths(mode: str) -> set[str]:
    if mode == "all":
        return set(REPLAY_PATHS)
    mapping = {
        "graph_only": {"graph"},
        "legacy_only": {"legacy"},
        "dual_only": {"dual"},
        "graph_legacy": {"graph", "legacy"},
        "graph_dual_legacy": {"graph", "legacy", "dual"},
        "hybrid": {"hybrid"},
    }
    return mapping.get(mode, set(REPLAY_PATHS))


def _build_record_paths(
    reference: list[str],
    active_paths: set[str],
    *,
    graph_items: list[str],
    graph_sources: list[str],
    legacy_items: list[str],
    legacy_sources: list[str],
    dual_items: list[str],
    dual_sources: list[str],
    hybrid_items: list[str],
    hybrid_sources: list[str],
    hybrid_coverage: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    paths: dict[str, dict[str, Any]] = {
        path: _inactive_path()
        for path in REPLAY_PATHS
    }
    if "graph" in active_paths:
        paths["graph"] = {
            "active": True,
            "items": graph_items,
            "sources": graph_sources,
            "metrics": score_ordering_candidates(reference, graph_items),
        }
    if "legacy" in active_paths:
        paths["legacy"] = {
            "active": True,
            "items": legacy_items,
            "sources": legacy_sources,
            "metrics": score_ordering_candidates(reference, legacy_items),
        }
    if "dual" in active_paths:
        paths["dual"] = {
            "active": True,
            "items": dual_items,
            "sources": dual_sources,
            "metrics": score_ordering_candidates(reference, dual_items),
        }
    if "hybrid" in active_paths:
        paths["hybrid"] = {
            "active": True,
            "items": hybrid_items,
            "sources": hybrid_sources,
            "coverage": hybrid_coverage,
            "metrics": score_ordering_candidates(reference, hybrid_items),
        }
    return paths


def _inactive_path() -> dict[str, Any]:
    return {"active": False, "items": [], "sources": [], "inactive": True}


def _path_is_active(path_info: dict[str, Any]) -> bool:
    if not isinstance(path_info, dict):
        return False
    return bool(path_info.get("active", "metrics" in path_info))


def _active_metrics(records: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    return [
        record["paths"][path]["metrics"]
        for record in records
        if _path_is_active(record.get("paths", {}).get(path, {})) and "metrics" in record["paths"][path]
    ]


def _summary_path_active(path_summary: dict[str, Any]) -> bool:
    if bool(path_summary.get("active")) and int(path_summary.get("count") or 0) > 0:
        return True
    metric_keys = {"f1", "kendall_tau_norm", "precision", "recall", "empty_rate", "mean_system_count", "mean_matched"}
    return any(key in path_summary for key in metric_keys)


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
        "rule_hits",
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
    summary = report.get("summary", {})
    cluster_diagnostics = summary.get("cluster_diagnostics", {}) if isinstance(summary, dict) else {}
    return {
        "workspace": report.get("workspace"),
        "split": report.get("split"),
        "query_count": report.get("query_count"),
        "summary": summary,
        "dual_vs_legacy_passed": summary.get("dual_vs_legacy_passed") if isinstance(summary, dict) else None,
        "dual_lift_over_legacy_f1": summary.get("dual_lift_over_legacy_f1") if isinstance(summary, dict) else None,
        "cluster_expanded_query_count": (
            cluster_diagnostics.get("expanded_query_count") if isinstance(cluster_diagnostics, dict) else None
        ),
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
