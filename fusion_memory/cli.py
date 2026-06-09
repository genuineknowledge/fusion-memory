from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.core.config import DEFAULT_CONFIG
from fusion_memory.core.llm import OpenAICompatibleLLMClient
from fusion_memory.eval.adapter import BenchmarkAdapter
from fusion_memory.eval.beam_adapter import BEAM_SPLITS, BeamAdapter
from fusion_memory.eval.longmemeval_adapter import LONGMEMEVAL_SPLITS, LongMemEvalAdapter
from fusion_memory.eval.model_adapters import OpenAICompatibleAnswerModel, OpenAICompatibleJudgeModel
from fusion_memory.storage.postgres_store import PostgresMigrationRunner
from fusion_memory.storage.postgres_verifier import verify_postgres_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion Memory local CLI")
    parser.add_argument("--db", default="fusion-memory.sqlite3", help="SQLite database path")
    parser.add_argument("--workspace-id")
    parser.add_argument("--user-id")
    parser.add_argument("--agent-id")
    parser.add_argument("--run-id")
    parser.add_argument("--session-id")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Add a memory input")
    add.add_argument("content")
    add.add_argument("--role", default="user")
    add.add_argument("--time")

    search = sub.add_parser("search", help="Search memory")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=DEFAULT_CONFIG.retrieval_output_n)
    search.add_argument("--allow-cross-session", action="store_true")

    context = sub.add_parser("answer-context", help="Build an evidence pack")
    context.add_argument("query")
    context.add_argument("--limit", type=int, default=DEFAULT_CONFIG.retrieval_output_n)
    context.add_argument("--allow-cross-session", action="store_true")

    get = sub.add_parser("get", help="Get a raw span or memory fact by id")
    get.add_argument("object_id")
    get.add_argument("--type", choices=["span", "fact"], default=None)

    history = sub.add_parser("history", help="List fact, relation, and event history for the current scope")
    history.add_argument("--entity", default=None)
    history.add_argument("--fact-id", default=None)
    history.add_argument("--allow-cross-session", action="store_true")

    trace = sub.add_parser("debug-trace", help="Read an add/search debug trace by id")
    trace.add_argument("trace_id")

    audit = sub.add_parser("audit", help="List append-only audit events for the current scope")
    audit.add_argument("--type", default=None)
    audit.add_argument("--limit", type=int, default=100)

    timeline = sub.add_parser("timeline", help="List events for the current scope")
    timeline.add_argument("--entity", default=None)
    timeline.add_argument("--start", default=None)
    timeline.add_argument("--end", default=None)
    timeline.add_argument("--allow-cross-session", action="store_true")

    views = sub.add_parser("views", help="List current views for the current scope")
    views.add_argument("--type", default=None)
    views.add_argument("--allow-cross-session", action="store_true")

    profiles = sub.add_parser("profiles", help="List entity profiles for the current scope")
    profiles.add_argument("entity_id")
    profiles.add_argument("--type", default=None)
    profiles.add_argument("--allow-cross-session", action="store_true")

    summaries = sub.add_parser("summaries", help="List or refresh session summary spans")
    summaries.add_argument("--refresh", action="store_true")
    summaries.add_argument("--max-source-spans", type=int, default=None)

    tasks = sub.add_parser("tasks", help="List or process background memory tasks")
    tasks.add_argument("--status", default=None)
    tasks.add_argument("--limit", type=int, default=100)
    tasks.add_argument("--process", action="store_true")
    tasks.add_argument("--allow-cross-session", action="store_true")

    reports = sub.add_parser("report", help="Show local quality/coverage reports")
    reports.add_argument("name", choices=["encoding", "profiles"])

    train = sub.add_parser("train-utility", help="Train the local retrieval utility scorer from collected weak labels")
    train.add_argument("--save-model", default=None)

    bench = sub.add_parser("run-benchmark", help="Run a local retrieval benchmark dataset")
    bench.add_argument("dataset_path")
    bench.add_argument("--split", default=None)
    bench.add_argument("--ablate", action="store_true", help="Also report Fast/Balanced/Benchmark mode ablations")
    _add_eval_model_args(bench)

    beam = sub.add_parser("run-beam", help="Run a BEAM-style local benchmark split")
    beam.add_argument("dataset_path")
    beam.add_argument("--split", default="small", choices=sorted(BEAM_SPLITS))
    beam.add_argument("--ablate", action="store_true", help="Also report retrieval-mode and component ablations")
    _add_eval_model_args(beam)

    longmem = sub.add_parser("run-longmemeval", help="Run a LongMemEval-style local benchmark split")
    longmem.add_argument("dataset_path")
    longmem.add_argument("--split", default="dev", choices=sorted(LONGMEMEVAL_SPLITS))
    longmem.add_argument("--ablate", action="store_true", help="Also report retrieval-mode and component ablations")
    _add_eval_model_args(longmem)

    pg = sub.add_parser("migrate-postgres", help="Apply the Postgres/pgvector production schema")
    pg.add_argument("dsn", help="Postgres DSN, for example postgresql://user:pass@localhost:5432/fusion_memory")

    pg_verify = sub.add_parser("verify-postgres", help="Run a live Postgres backend smoke verification")
    pg_verify.add_argument("dsn", help="Postgres DSN, for example postgresql://user:pass@localhost:5432/fusion_memory")
    pg_verify.add_argument("--skip-migrate", action="store_true", help="Skip migration and only run the service smoke")

    args = parser.parse_args()
    if args.command == "migrate-postgres":
        runner = PostgresMigrationRunner(args.dsn)
        try:
            report = runner.migrate()
            print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
        finally:
            runner.close()
        return
    if args.command == "verify-postgres":
        report = verify_postgres_backend(args.dsn, migrate=not args.skip_migrate)
        print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
        return

    scope = Scope(
        workspace_id=args.workspace_id,
        user_id=args.user_id,
        agent_id=args.agent_id,
        run_id=args.run_id,
        session_id=args.session_id,
    )
    service = MemoryService(args.db)
    try:
        if args.command == "add":
            session_time = datetime.fromisoformat(args.time) if args.time else datetime.now(timezone.utc)
            result = service.add({"role": args.role, "content": args.content}, scope, session_time)
            print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        elif args.command == "search":
            result = service.search(args.query, scope, options={"limit": args.limit, "allow_cross_session": args.allow_cross_session})
            print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
        elif args.command == "answer-context":
            pack = service.answer_context(args.query, scope, budget={"limit": args.limit, "allow_cross_session": args.allow_cross_session})
            print(json.dumps(_jsonable(pack), ensure_ascii=False, indent=2))
        elif args.command == "get":
            print(json.dumps(_jsonable(service.get(args.object_id, args.type)), ensure_ascii=False, indent=2))
        elif args.command == "history":
            print(json.dumps(_jsonable(service.history(scope, entity=args.entity, fact_id=args.fact_id, allow_cross_session=args.allow_cross_session)), ensure_ascii=False, indent=2))
        elif args.command == "debug-trace":
            print(json.dumps(_jsonable(service.debug_trace(args.trace_id)), ensure_ascii=False, indent=2))
        elif args.command == "audit":
            print(json.dumps(_jsonable(service.audit_events(scope, event_type=args.type, limit=args.limit)), ensure_ascii=False, indent=2))
        elif args.command == "timeline":
            events = service.timeline(args.entity, scope, start=args.start, end=args.end, allow_cross_session=args.allow_cross_session)
            print(json.dumps(_jsonable(events), ensure_ascii=False, indent=2))
        elif args.command == "views":
            print(json.dumps(_jsonable(service.get_current_views(scope, view_type=args.type, allow_cross_session=args.allow_cross_session)), ensure_ascii=False, indent=2))
        elif args.command == "profiles":
            profiles = service.get_entity_profile(args.entity_id, scope, profile_type=args.type, allow_cross_session=args.allow_cross_session)
            print(json.dumps(_jsonable(profiles), ensure_ascii=False, indent=2))
        elif args.command == "summaries":
            if args.refresh:
                summary = service.refresh_session_summary(scope, max_source_spans=args.max_source_spans)
                print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
            else:
                print(json.dumps(_jsonable(service.get_session_summaries(scope)), ensure_ascii=False, indent=2))
        elif args.command == "tasks":
            if args.process:
                result = service.process_background_tasks(scope, limit=args.limit, allow_cross_session=args.allow_cross_session)
                print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
            else:
                tasks_out = service.list_background_tasks(
                    scope,
                    status=args.status,
                    limit=args.limit,
                    allow_cross_session=args.allow_cross_session,
                )
                print(json.dumps(_jsonable(tasks_out), ensure_ascii=False, indent=2))
        elif args.command == "report":
            if args.name == "encoding":
                print(json.dumps(_jsonable(service.encoding_report(scope)), ensure_ascii=False, indent=2))
            elif args.name == "profiles":
                print(json.dumps(_jsonable(service.profile_report(scope)), ensure_ascii=False, indent=2))
        elif args.command == "train-utility":
            report = service.train_utility_scorer()
            if args.save_model:
                service.save_utility_scorer(args.save_model)
            print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
        elif args.command == "run-benchmark":
            answer_model, judge_model = _build_eval_models(args)
            adapter = BenchmarkAdapter(service, scope, answer_model=answer_model, judge_model=judge_model)
            ingest = adapter.ingest_dataset(args.dataset_path, split=args.split)
            queries = adapter.build_queries(args.dataset_path, split=args.split)
            results = adapter.run_queries(queries)
            output = {"ingest": ingest, "report": adapter.report(results)}
            if args.ablate:
                output["ablation"] = {
                    "retrieval_modes": adapter.run_ablation(queries),
                    "components": adapter.run_component_ablation(queries),
                }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        elif args.command == "run-beam":
            answer_model, judge_model = _build_eval_models(args)
            adapter = BeamAdapter(service, scope, split=args.split, answer_model=answer_model, judge_model=judge_model)
            output = adapter.run_dataset(args.dataset_path, split=args.split, ablate=args.ablate)
            print(json.dumps(output, ensure_ascii=False, indent=2))
        elif args.command == "run-longmemeval":
            answer_model, judge_model = _build_eval_models(args)
            adapter = LongMemEvalAdapter(service, scope, split=args.split, answer_model=answer_model, judge_model=judge_model)
            output = adapter.run_dataset(args.dataset_path, split=args.split, ablate=args.ablate)
            print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        service.close()


def _jsonable(value):
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _add_eval_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--answer-endpoint", default=None, help="OpenAI-compatible chat/completions endpoint for benchmark answers")
    parser.add_argument("--answer-model", default="eval-answer", help="Model name sent to --answer-endpoint")
    parser.add_argument("--answer-api-key", default=None, help="Bearer token for --answer-endpoint")
    parser.add_argument("--judge-endpoint", default=None, help="OpenAI-compatible chat/completions endpoint for semantic judging")
    parser.add_argument("--judge-model", default="eval-judge", help="Model name sent to --judge-endpoint")
    parser.add_argument("--judge-api-key", default=None, help="Bearer token for --judge-endpoint")
    parser.add_argument("--model-api-key", default=None, help="Shared fallback Bearer token for answer/judge endpoints")
    parser.add_argument("--model-timeout-seconds", type=float, default=30.0, help="HTTP timeout for answer/judge model calls")


def _build_eval_models(args: argparse.Namespace):
    answer_model = None
    judge_model = None
    if getattr(args, "answer_endpoint", None):
        answer_client = OpenAICompatibleLLMClient(
            args.answer_endpoint,
            api_key=args.answer_api_key or args.model_api_key,
            model=args.answer_model,
            timeout_seconds=args.model_timeout_seconds,
        )
        answer_model = OpenAICompatibleAnswerModel(answer_client)
    if getattr(args, "judge_endpoint", None):
        judge_client = OpenAICompatibleLLMClient(
            args.judge_endpoint,
            api_key=args.judge_api_key or args.model_api_key,
            model=args.judge_model,
            timeout_seconds=args.model_timeout_seconds,
        )
        judge_model = OpenAICompatibleJudgeModel(judge_client)
    return answer_model, judge_model


if __name__ == "__main__":
    main()
