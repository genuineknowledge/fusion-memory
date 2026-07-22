from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_memory import Scope  # noqa: E402
from fusion_memory import MemoryService  # noqa: E402
from fusion_memory.cli import _build_eval_models, _jsonable  # noqa: E402
from fusion_memory.core.runtime_config import memory_service_from_env  # noqa: E402
from fusion_memory.eval.beam_adapter import BeamAdapter, _beam_session_id_from_id, _load_official_beam_dataset  # noqa: E402


def _default_beam_dataset() -> str:
    return os.getenv("BEAM_DATASET", "datasets/BEAM")


def main() -> None:
    parser = argparse.ArgumentParser(description="BEAM parallel resume/query runner")
    parser.add_argument("--dataset", default=_default_beam_dataset())
    parser.add_argument("--split", default="100k", choices=["small", "dev", "100k", "500k", "1m", "10m"])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--db", default=os.getenv("FUSION_MEMORY_DB", "postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory"))
    parser.add_argument("--output", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("resume-ingest", help="Ingest only BEAM documents absent from the workspace")
    ingest.add_argument("--workers", type=int, default=1)
    ingest.add_argument("--progress-every", type=int, default=100)
    ingest.add_argument("--devices", default=None, help="Comma-separated CUDA device ids assigned round-robin to workers")

    query = sub.add_parser("query", help="Run answer/judge over an already-ingested workspace")
    query.add_argument("--workers", type=int, default=1)
    query.add_argument("--progress-every", type=int, default=10)
    query.add_argument("--devices", default=None, help="Comma-separated CUDA device ids assigned round-robin to workers")
    query.add_argument("--variant", default=None)
    query.add_argument("--partial-dir", default=None)
    query.add_argument("--model-config-file", default=None)
    query.add_argument("--answer-endpoint", default=None)
    query.add_argument("--answer-model", default=None)
    query.add_argument("--answer-api-key", default=None)
    query.add_argument("--judge-endpoint", default=None)
    query.add_argument("--judge-model", default=None)
    query.add_argument("--judge-api-key", default=None)
    query.add_argument("--model-api-key", default=None)
    query.add_argument("--model-timeout-seconds", type=float, default=None)
    query.add_argument("--use-llm-aggregation", action="store_true", default=None)
    query.add_argument("--llm-aggregation-min-confidence", type=float, default=None)
    query.add_argument("--query-ids", default=None, help="Comma-separated BEAM query ids to run")
    query.add_argument("--query-ids-file", default=None, help="File containing BEAM query ids, one per line or comma-separated")
    query.add_argument("--categories", default=None, help="Comma-separated BEAM categories to run")
    query.add_argument("--from-result", default=None, help="Previous BEAM result JSON used to select failed queries")
    query.add_argument("--score-lt", type=float, default=None, help="With --from-result, select answers below this score")
    query.add_argument(
        "--answer-failed-only",
        action="store_true",
        help="With --from-result, select only retryable answer/judge-failed queries",
    )
    query.add_argument("--per-category", type=int, default=None, help="With --from-result, cap selected queries per category")
    query.add_argument("--diagnostic-output", default=None, help="Write a compact old/new per-query diagnostic JSON")
    query.add_argument("--include-full-pack", action="store_true", help="Include full current EvidencePack in partial records")
    query.add_argument("--max-consecutive-answer-failures", type=int, default=5, help="Abort a worker after this many consecutive answer failures")
    query.add_argument("--answer-failure-retries", type=int, default=0, help="Retry incomplete/answer_failed queries this many extra parallel rounds")

    args = parser.parse_args()
    if args.command == "resume-ingest":
        run_resume_ingest(args)
    elif args.command == "query":
        run_query(args)


def run_resume_ingest(args: argparse.Namespace) -> None:
    documents, queries = _load_required_dataset(args.dataset, args.split)
    existing_turns = _existing_turn_ids(args.db, args.workspace)
    pending = [doc for doc in documents if doc.id not in existing_turns]
    started = perf_counter()
    output: dict[str, Any] = {
        "run": Path(args.output).stem,
        "workspace": args.workspace,
        "split": args.split,
        "status": "ingesting",
        "documents": len(documents),
        "queries": len(queries),
        "existing_documents": len(documents) - len(pending),
        "pending_documents": len(pending),
        "workers": args.workers,
        "started_at": _now(),
    }
    _write_json(args.output, output)
    if not pending:
        output.update(
            {
                "status": "ingest_complete",
                "elapsed_seconds": perf_counter() - started,
                "ingest": _workspace_counts(args.db, args.workspace),
                "completed_at": _now(),
            }
        )
        _write_json(args.output, output)
        print(json.dumps({"phase": "ingest", "status": "complete", **output["ingest"]}, ensure_ascii=False), flush=True)
        return

    chunks = _chunks(pending, max(1, args.workers))
    completed = 0
    accepted_fact_count = 0
    accepted_event_count = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _ingest_chunk,
                [doc.__dict__ for doc in chunk],
                args.dataset,
                args.split,
                args.workspace,
                args.user_id,
                args.agent_id,
                args.run_id,
                args.session_id,
                args.db,
                index,
                args.progress_every,
                args.devices,
            )
            for index, chunk in enumerate(chunks)
            if chunk
        ]
        for future in as_completed(futures):
            result = future.result()
            completed += int(result["documents"])
            accepted_fact_count += int(result["accepted_fact_count"])
            accepted_event_count += int(result["accepted_event_count"])
            counts = _workspace_counts(args.db, args.workspace)
            output.update(
                {
                    "status": "ingesting",
                    "completed_pending_documents": completed,
                    "accepted_fact_count_delta": accepted_fact_count,
                    "accepted_event_count_delta": accepted_event_count,
                    "ingest": counts,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            _write_json(args.output, output)
            print(
                json.dumps(
                    {
                        "phase": "ingest",
                        "completed_pending": completed,
                        "pending_total": len(pending),
                        **counts,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    counts = _workspace_counts(args.db, args.workspace)
    output.update(
        {
            "status": "ingest_complete",
            "completed_pending_documents": completed,
            "accepted_fact_count_delta": accepted_fact_count,
            "accepted_event_count_delta": accepted_event_count,
            "ingest": counts,
            "elapsed_seconds": perf_counter() - started,
            "completed_at": _now(),
        }
    )
    _write_json(args.output, output)


def run_query(args: argparse.Namespace) -> None:
    documents, queries = _load_required_dataset(args.dataset, args.split)
    selected_query_ids = _selected_query_ids(args)
    if selected_query_ids is not None:
        selected = set(selected_query_ids)
        queries = [query for query in queries if query.id in selected]
    selected_categories = _selected_categories(args)
    if selected_categories is not None:
        queries = [query for query in queries if query.category in selected_categories]
    selected_query_id_set = {query.id for query in queries}
    started = perf_counter()
    output: dict[str, Any] = {
        "run": Path(args.output).stem,
        "workspace": args.workspace,
        "split": args.split,
        "variant": args.variant,
        "status": "querying",
        "documents": len(documents),
        "queries": len(queries),
        "query_filter": {
            "query_ids": selected_query_ids,
            "categories": sorted(selected_categories) if selected_categories is not None else None,
            "from_result": args.from_result,
            "score_lt": args.score_lt,
            "answer_failed_only": args.answer_failed_only,
            "per_category": args.per_category,
            "answer_failure_retries": args.answer_failure_retries,
        },
        "workers": args.workers,
        "started_at": _now(),
        "ingest": _workspace_counts(args.db, args.workspace),
    }
    _write_json(args.output, output)
    partial_dir = Path(args.partial_dir) if args.partial_dir else Path(args.output).with_suffix(".partials")
    partial_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_resumable_partial_records(partial_dir, selected_query_id_set)
    existing_ids = _completed_query_ids(existing)
    result_dicts: list[dict[str, Any]] = list(existing)
    worker_failures: list[dict[str, Any]] = []
    if existing:
        output.update(
            {
                "completed_queries": len(existing_ids),
                "resumed_from_partial": len(existing),
                "retryable_partial_failures": sum(1 for record in existing if not _is_completed_record(record)),
            }
        )
        _write_json(args.output, output)
    all_selected_queries = list(queries)
    pending_queries = [query for query in queries if query.id not in existing_ids]
    output["pending_queries"] = len(pending_queries)
    result_dicts, worker_failures = _run_query_round(
        args,
        pending_queries,
        partial_dir,
        result_dicts,
        worker_failures,
        output,
        started,
        round_index=0,
        round_label="initial",
    )
    max_retry_rounds = max(0, int(args.answer_failure_retries or 0))
    for retry_index in range(1, max_retry_rounds + 1):
        retry_ids = _retryable_query_ids(result_dicts, all_selected_queries)
        if not retry_ids:
            break
        retry_queries = [query for query in all_selected_queries if query.id in retry_ids]
        output.update(
            {
                "status": "retrying",
                "retry_round": retry_index,
                "retry_pending_queries": len(retry_queries),
                "elapsed_seconds": perf_counter() - started,
            }
        )
        _write_json(args.output, output)
        result_dicts, worker_failures = _run_query_round(
            args,
            retry_queries,
            partial_dir,
            result_dicts,
            worker_failures,
            output,
            started,
            round_index=retry_index,
            round_label="retry",
        )
    order = {query.id: index for index, query in enumerate(all_selected_queries)}
    result_dicts.sort(key=lambda item: order.get(item["query_id"], math.inf))
    completed_ids = _completed_query_ids(result_dicts)
    output.update(
        {
            "completed_queries": len(completed_ids),
            "partial_records": len(result_dicts),
            "retryable_partial_failures": sum(1 for record in result_dicts if not _is_completed_record(record)),
            "worker_failures": len(worker_failures),
            "pending_queries": max(0, len(all_selected_queries) - len(completed_ids)),
            "elapsed_seconds": perf_counter() - started,
            "completed_at": _now(),
        }
    )
    if not _all_queries_completed(result_dicts, all_selected_queries):
        output["status"] = "partial"
        output["worker_failure_samples"] = worker_failures[:10]
        _write_json(args.output, output)
        if args.diagnostic_output:
            _write_json(args.diagnostic_output, _diagnostic_report(args, result_dicts))
        return
    report = _report_from_result_dicts(args, result_dicts)
    output.update(
        {
            "status": "complete",
            "report": report,
            "worker_failure_samples": worker_failures[:10],
        }
    )
    _write_json(args.output, output)
    if args.diagnostic_output:
        _write_json(args.diagnostic_output, _diagnostic_report(args, result_dicts))


def _run_query_round(
    args: argparse.Namespace,
    pending_queries: list[Any],
    partial_dir: Path,
    result_dicts: list[dict[str, Any]],
    worker_failures: list[dict[str, Any]],
    output: dict[str, Any],
    started: float,
    *,
    round_index: int,
    round_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pending_queries:
        return result_dicts, worker_failures
    chunks = _chunks(pending_queries, max(1, args.workers))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _query_chunk,
                [query.__dict__ for query in chunk],
                args.dataset,
                args.split,
                args.workspace,
                args.user_id,
                args.agent_id,
                args.run_id,
                args.session_id,
                args.db,
                _eval_arg_dict(args),
                index,
                str(partial_dir),
                args.include_full_pack,
                args.max_consecutive_answer_failures,
                args.devices,
            )
            for index, chunk in enumerate(chunks)
            if chunk
        ]
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
            except Exception as exc:
                worker_failures.append(
                    {
                        "worker_failed": True,
                        "error": str(exc),
                        "phase": "query",
                        "round": round_index,
                    }
                )
                continue
            for record in chunk_results:
                if record.get("worker_failed"):
                    record.setdefault("round", round_index)
                    worker_failures.append(record)
            chunk_records = [record for record in chunk_results if record.get("query_id")]
            result_dicts = _merge_partial_records([*result_dicts, *chunk_records])
            completed_ids = _completed_query_ids(result_dicts)
            output.update(
                {
                    "completed_queries": len(completed_ids),
                    "partial_records": len(result_dicts),
                    "retryable_partial_failures": sum(1 for record in result_dicts if not _is_completed_record(record)),
                    "worker_failures": len(worker_failures),
                    "last_query_round": round_index,
                    "pending_queries": max(0, int(output.get("queries") or len(pending_queries)) - len(completed_ids)),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            _write_json(args.output, output)
            print(
                json.dumps(
                    {
                        "phase": "query",
                        "round": round_index,
                        "round_label": round_label,
                        "completed": len(completed_ids),
                        "total": len(pending_queries),
                        "elapsed_seconds": round(perf_counter() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return result_dicts, worker_failures


def _ingest_chunk(
    docs: list[dict[str, Any]],
    dataset: str,
    split: str,
    workspace: str,
    user_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    session_id: str | None,
    db: str,
    worker_index: int,
    progress_every: int,
    devices: str | None,
) -> dict[str, Any]:
    _configure_worker_device(worker_index, devices)
    service = _memory_service(db)
    scope = Scope(workspace_id=workspace, user_id=user_id, agent_id=agent_id, run_id=run_id, session_id=session_id)
    adapter = BeamAdapter(service, scope, split=split)
    accepted_fact_count = 0
    accepted_event_count = 0
    try:
        for i, doc in enumerate(docs, 1):
            before = _workspace_counts(db, workspace) if progress_every and i % progress_every == 1 else None
            timestamp = _parse_dt(doc.get("timestamp")) or datetime.now(timezone.utc)
            result = service.add(
                {
                    "role": doc["speaker"],
                    "content": doc["content"],
                    "turn_id": doc["id"],
                    "timestamp": timestamp.isoformat(),
                },
                _beam_scope(scope, doc.get("id")),
                timestamp,
                {"source_uri": doc["id"]},
            )
            accepted_fact_count += len(result.accepted_fact_ids)
            accepted_event_count += len(result.accepted_event_ids)
            if progress_every and i % progress_every == 0:
                after = _workspace_counts(db, workspace)
                print(
                    json.dumps(
                        {
                            "phase": "ingest-worker",
                            "worker": worker_index,
                            "i": i,
                            "total": len(docs),
                            "spans": after["span_count"],
                            "facts": after["accepted_fact_count"],
                            "events": after["accepted_event_count"],
                            "before": before,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return {
            "worker": worker_index,
            "documents": len(docs),
            "accepted_fact_count": accepted_fact_count,
            "accepted_event_count": accepted_event_count,
        }
    finally:
        service.close()


def _query_chunk(
    queries: list[dict[str, Any]],
    dataset: str,
    split: str,
    workspace: str,
    user_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    session_id: str | None,
    db: str,
    eval_args: dict[str, Any],
    worker_index: int,
    partial_dir: str,
    include_full_pack: bool,
    max_consecutive_answer_failures: int,
    devices: str | None,
) -> list[dict[str, Any]]:
    _configure_worker_device(worker_index, devices)
    service = _memory_service(db)
    scope = Scope(workspace_id=workspace, user_id=user_id, agent_id=agent_id, run_id=run_id, session_id=session_id)
    answer_model, judge_model = _build_eval_models(SimpleNamespace(**eval_args))
    adapter = BeamAdapter(service, scope, split=split, answer_model=answer_model, judge_model=judge_model)
    partial_path = Path(partial_dir) / f"worker_{worker_index}.jsonl"
    selected_query_ids = {str(query.get("id")) for query in queries if query.get("id")}
    out = _load_resumable_partial_records(Path(partial_dir), selected_query_ids)
    completed_ids = _completed_query_ids(out)
    consecutive_answer_failures = 0
    worker_error: dict[str, Any] | None = None
    try:
        for i, query_dict in enumerate(queries, 1):
            if query_dict.get("id") in completed_ids:
                continue
            query_scope = adapter._beam_scope(query_dict.get("id"))
            result = adapter.answer_query(SimpleNamespace(**query_dict))
            record = _jsonable(result)
            if include_full_pack:
                record["full_evidence_pack"] = _jsonable(
                    adapter.retrieval_engine.answer_context(
                        query_dict["query"],
                        query_scope,
                        query_dict.get("category"),
                        budget={"mode": "balanced"},
                    )
                )
            out.append(record)
            _append_partial_record(partial_path, record)
            if _is_completed_record(record):
                completed_ids.add(str(record.get("query_id")))
            if record.get("answer_failed"):
                consecutive_answer_failures += 1
            else:
                consecutive_answer_failures = 0
            if max_consecutive_answer_failures > 0 and consecutive_answer_failures >= max_consecutive_answer_failures:
                worker_error = {
                    "worker_failed": True,
                    "worker_index": worker_index,
                    "error": (
                        f"worker aborted after {consecutive_answer_failures} consecutive answer failures; "
                        f"last_query_id={record.get('query_id')}"
                    ),
                    "last_query_id": record.get("query_id"),
                    "failed_after_consecutive_answer_failures": consecutive_answer_failures,
                    "phase": "query",
                }
                break
            print(
                json.dumps(
                    {"phase": "query-worker", "worker": worker_index, "i": i, "total": len(queries), "query_id": result.query_id},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if worker_error:
            out.append(worker_error)
        return out
    except Exception as exc:
        out.append(
            {
                "worker_failed": True,
                "worker_index": worker_index,
                "error": str(exc),
                "phase": "query",
            }
        )
        return out
    finally:
        service.close()


def _report_from_result_dicts(args: argparse.Namespace, result_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    service = MemoryService()
    scope = Scope(workspace_id=args.workspace, user_id=args.user_id, agent_id=args.agent_id, run_id=args.run_id, session_id=args.session_id)
    answer_model, judge_model = _build_eval_models(SimpleNamespace(**_eval_arg_dict(args)))
    adapter = BeamAdapter(service, scope, split=args.split, answer_model=answer_model, judge_model=judge_model)
    try:
        results = [SimpleNamespace(**item) for item in result_dicts]
        return adapter.report(results)
    finally:
        service.close()


def _eval_arg_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "answer_endpoint": args.answer_endpoint,
        "answer_model": args.answer_model,
        "answer_api_key": args.answer_api_key,
        "judge_endpoint": args.judge_endpoint,
        "judge_model": args.judge_model,
        "judge_api_key": args.judge_api_key,
        "model_api_key": args.model_api_key,
        "model_config_file": args.model_config_file,
        "model_timeout_seconds": args.model_timeout_seconds,
        "use_llm_aggregation": args.use_llm_aggregation,
        "llm_aggregation_min_confidence": args.llm_aggregation_min_confidence,
    }


def _selected_query_ids(args: argparse.Namespace) -> list[str] | None:
    explicit = _split_csv(args.query_ids)
    explicit.extend(_ids_from_file(getattr(args, "query_ids_file", None)))
    selected: list[str] = []
    if explicit:
        selected.extend(explicit)
    if args.from_result:
        selected.extend(
            _query_ids_from_result(
                args.from_result,
                score_lt=args.score_lt,
                per_category=args.per_category,
                answer_failed_only=args.answer_failed_only,
            )
        )
    if not selected:
        return None
    return list(dict.fromkeys(selected))


def _selected_categories(args: argparse.Namespace) -> set[str] | None:
    categories = _split_csv(args.categories)
    return set(categories) if categories else None


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _ids_from_file(path: str | None) -> list[str]:
    if not path:
        return []
    raw = Path(path).read_text(encoding="utf-8")
    ids: list[str] = []
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        ids.extend(_split_csv(line))
    return ids


def _query_ids_from_result(
    path: str,
    *,
    score_lt: float | None,
    per_category: int | None,
    answer_failed_only: bool = False,
) -> list[str]:
    answers = _answer_records_from_result(path)
    if not answers:
        return []
    threshold = 0.5 if score_lt is None else score_lt
    grouped: dict[str, list[dict[str, Any]]] = {}
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        if answer_failed_only:
            if not _record_has_retryable_failure(answer):
                continue
        else:
            if _record_has_retryable_failure(answer):
                pass
            else:
                try:
                    score = float(answer.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                if score >= threshold:
                    continue
        category = str(answer.get("category") or "unknown")
        grouped.setdefault(category, []).append(answer)
    query_ids: list[str] = []
    for category in sorted(grouped):
        items = grouped[category]
        items.sort(key=lambda item: str(item.get("query_id") or ""))
        capped = items[:per_category] if per_category else items
        query_ids.extend(str(item["query_id"]) for item in capped if item.get("query_id"))
    return query_ids


def _answer_records_from_result(path: str) -> list[dict[str, Any]]:
    result_path = Path(path)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = [
        data.get("report", {}).get("answers") if isinstance(data.get("report"), dict) else None,
        data.get("answers"),
        data.get("results"),
        data.get("records"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            return [item for item in candidate if isinstance(item, dict)]
    partial_dir = result_path.with_suffix(".partials")
    if partial_dir.exists():
        return _load_partial_dir_records(partial_dir)
    return []


def _diagnostic_report(args: argparse.Namespace, result_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    old_by_id = _old_answers_by_id(args.from_result) if args.from_result else {}
    records: list[dict[str, Any]] = []
    for result in sorted(result_dicts, key=lambda item: str(item.get("query_id") or "")):
        query_id = str(result.get("query_id") or "")
        old = old_by_id.get(query_id, {})
        old_pack = old.get("evidence_pack") if isinstance(old.get("evidence_pack"), dict) else {}
        new_pack = result.get("evidence_pack") if isinstance(result.get("evidence_pack"), dict) else {}
        full_pack = result.get("full_evidence_pack") if isinstance(result.get("full_evidence_pack"), dict) else {}
        records.append(
            {
                "query_id": query_id,
                "category": result.get("category"),
                "query_type": result.get("query_type"),
                "query_text": result.get("query_text"),
                "old_score": old.get("score"),
                "new_score": result.get("score"),
                "old_answer": old.get("answer"),
                "new_answer": result.get("answer"),
                "old_judge_reason": old.get("judge_reason"),
                "new_judge_reason": result.get("judge_reason"),
                "old_pack_summary": _diagnostic_pack_summary(old_pack),
                "new_pack_summary": _diagnostic_pack_summary(new_pack),
                "new_pack_markers": _diagnostic_pack_markers(full_pack),
            }
        )
    return {
        "run": Path(args.output).stem,
        "workspace": args.workspace,
        "split": args.split,
        "variant": args.variant,
        "status": "complete",
        "from_result": args.from_result,
        "total": len(records),
        "improved": sum(1 for item in records if _score(item.get("new_score")) > _score(item.get("old_score"))),
        "regressed": sum(1 for item in records if _score(item.get("new_score")) < _score(item.get("old_score"))),
        "records": records,
    }


def _old_answers_by_id(path: str) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = data.get("report", {}).get("answers", [])
    if not isinstance(answers, list):
        return {}
    return {str(answer.get("query_id")): answer for answer in answers if isinstance(answer, dict) and answer.get("query_id")}


def _diagnostic_pack_summary(pack: dict[str, Any]) -> dict[str, Any]:
    coverage = pack.get("coverage") if isinstance(pack.get("coverage"), dict) else {}
    return {
        "answer_policy": pack.get("answer_policy"),
        "query_type": coverage.get("query_type"),
        "coverage_insufficient": coverage.get("coverage_insufficient"),
        "topic_group": coverage.get("topic_group") or coverage.get("selected_topic_group"),
        "fact_count": pack.get("fact_count"),
        "event_count": pack.get("event_count"),
        "source_span_count": pack.get("source_span_count"),
        "source_span_ids": pack.get("source_span_ids"),
        "temporal_roles": pack.get("temporal_roles"),
    }


def _diagnostic_pack_markers(pack: dict[str, Any]) -> dict[str, Any]:
    coverage = pack.get("coverage") if isinstance(pack.get("coverage"), dict) else {}
    spans = pack.get("source_spans") if isinstance(pack.get("source_spans"), list) else []
    facts = pack.get("facts") if isinstance(pack.get("facts"), list) else []
    events = pack.get("events") if isinstance(pack.get("events"), list) else []
    claim_counts = coverage.get("claim_polarity_counts")
    value_mentions = []
    for record in list(spans) + list(facts):
        if isinstance(record, dict) and record.get("value_mentions"):
            value_mentions.extend(record.get("value_mentions") or [])
    return {
        "coverage_keys": sorted(coverage.keys()),
        "topic_groups": sorted({str(item.get("topic_group")) for item in spans if isinstance(item, dict) and item.get("topic_group")}),
        "claim_polarity_counts": claim_counts,
        "value_mentions": value_mentions[:20],
        "timeline": [
            {
                "timeline_index": item.get("timeline_index"),
                "speaker": item.get("speaker"),
                "content": str(item.get("content") or item.get("description") or "")[:500],
            }
            for item in (spans if spans else events)[:12]
            if isinstance(item, dict)
        ],
    }


def _score(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _memory_service(db: str):
    backend = "postgres" if db.startswith(("postgresql://", "postgres://")) else None
    service = memory_service_from_env(db, storage_backend=backend)
    service.config.auto_session_summary_tasks = False
    return service


def _configure_worker_device(worker_index: int, devices: str | None) -> None:
    values = _split_csv(devices)
    if not values:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = values[worker_index % len(values)]
    os.environ.setdefault("FUSION_MEMORY_EMBEDDING_DEVICE", "cuda:0")
    os.environ.setdefault("FUSION_MEMORY_RERANKER_DEVICE", "cuda:0")


def _beam_scope(scope: Scope, item_id: str | None) -> Scope:
    session_id = _beam_session_id_from_id(item_id)
    if not session_id:
        return scope
    return Scope(
        workspace_id=scope.workspace_id,
        user_id=scope.user_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        session_id=session_id,
        app_id=scope.app_id,
    )


def _load_required_dataset(dataset: str, split: str):
    loaded = _load_official_beam_dataset(dataset, split)
    if not loaded:
        raise ValueError("official BEAM dataset layout is required")
    return loaded


def _load_partial_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                json.dumps(
                    {
                        "phase": "partial-load",
                        "status": "skipped_malformed_jsonl",
                        "path": str(path),
                        "line": line_number,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _load_partial_dir_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for partial in sorted(path.glob("*.jsonl")):
        records.extend(_load_partial_records(partial))
    return _merge_partial_records(records)


def _load_resumable_partial_records(partial_dir: Path, query_ids: set[str]) -> list[dict[str, Any]]:
    if not query_ids:
        return []
    return [
        record
        for record in _load_partial_dir_records(partial_dir)
        if str(record.get("query_id") or "") in query_ids
    ]


def _merge_partial_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = record.get("query_id")
        if not query_id:
            continue
        key = str(query_id)
        previous = by_id.get(key)
        if previous is not None and _is_completed_record(previous) and not _is_completed_record(record):
            continue
        by_id[key] = record
    return list(by_id.values())


def _completed_query_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record.get("query_id")) for record in records if _is_completed_record(record)}


def _retryable_query_ids(records: list[dict[str, Any]], queries: list[Any]) -> set[str]:
    completed = _completed_query_ids(records)
    return {str(query.id) for query in queries if str(query.id) not in completed}


def _all_queries_completed(records: list[dict[str, Any]], queries: list[Any]) -> bool:
    return not _retryable_query_ids(records, queries)


def _is_completed_record(record: dict[str, Any]) -> bool:
    return bool(record.get("query_id")) and not _record_has_retryable_failure(record)


def _record_has_retryable_failure(record: dict[str, Any]) -> bool:
    if bool(record.get("answer_failed")) or bool(record.get("judge_failed")):
        return True
    judge_reason = str(record.get("judge_reason") or "").lower()
    return "rubric scoring failed" in judge_reason or "judge scoring failed" in judge_reason


def _append_partial_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _existing_turn_ids(db: str, workspace: str) -> set[str]:
    psycopg2 = _import_psycopg2()
    conn = psycopg2.connect(db)
    try:
        cur = conn.cursor()
        cur.execute("select turn_id from evidence_spans where workspace_id=%s and turn_id is not null", (workspace,))
        return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def _workspace_counts(db: str, workspace: str) -> dict[str, int]:
    psycopg2 = _import_psycopg2()
    conn = psycopg2.connect(db)
    try:
        cur = conn.cursor()
        counts: dict[str, int] = {}
        for table, key in [
            ("evidence_spans", "span_count"),
            ("memory_facts", "accepted_fact_count"),
            ("events", "accepted_event_count"),
        ]:
            cur.execute(f"select count(*) from {table} where workspace_id=%s", (workspace,))
            counts[key] = int(cur.fetchone()[0])
        cur.execute(
            """
            select count(*)
            from event_edges ee
            join events e on e.event_id = ee.from_event_id
            where e.workspace_id = %s
            """,
            (workspace,),
        )
        counts["event_edge_count"] = int(cur.fetchone()[0])
        return counts
    finally:
        conn.close()


def _import_psycopg2():
    try:
        import psycopg2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("psycopg2 is required for Postgres-backed BEAM runner operations") from exc
    return psycopg2


def _chunks(items: list[Any], workers: int) -> list[list[Any]]:
    if workers <= 1:
        return [items]
    return [items[index::workers] for index in range(workers)]


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


if __name__ == "__main__":
    main()
