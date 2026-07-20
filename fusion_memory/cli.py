from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from fusion_memory import Scope
from fusion_memory.alpha_beta import run_alpha, run_beta
from fusion_memory.agent_installer import install_agent
from fusion_memory.adapters.haitun_history_watcher import (
    config_from_workspace,
    start_history_watcher_daemon,
    status_history_watcher_daemon,
    stop_history_watcher_daemon,
    sync_history_once,
    watch_history,
)
from fusion_memory.core.config import DEFAULT_CONFIG
from fusion_memory.core.llm import OpenAICompatibleLLMClient
from fusion_memory.product import (
    backup_data,
    configure_interactive,
    doctor,
    init_home,
    install_readiness,
    product_paths,
    render_human,
    safe_product_error,
    service_status,
    start_service,
    stop_service,
    upgrade,
)
from fusion_memory.core.runtime_config import memory_service_from_env
from fusion_memory.eval.beam_adapter import BEAM_SPLITS, BeamAdapter
from fusion_memory.eval.model_adapters import OpenAICompatibleAnswerModel, OpenAICompatibleJudgeModel
from fusion_memory.storage.postgres_store import PostgresMigrationRunner
from fusion_memory.storage.token_store import PostgresTokenStore, TokenRecord
from fusion_memory.storage.postgres_verifier import verify_postgres_backend

sync_haitun_history_once = sync_history_once
sync_dolphin_history_once = sync_history_once

_COMPAT_COMMAND_ALIASES = {
    "sync-dolphin-history": "sync-haitun-history",
}


class FusionMemoryArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if "--json" in sys.argv:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "invalid_command",
                        "message": _friendly_argparse_message(message),
                        "next_step": "Run fusion-memory doctor or check the help text.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2)
        super().error(message)


def main() -> None:
    parser = FusionMemoryArgumentParser(description="Fusion Memory local CLI")
    parser.add_argument("--db", default="fusion-memory.sqlite3", help="SQLite database path")
    parser.add_argument("--workspace-id")
    parser.add_argument("--user-id")
    parser.add_argument("--agent-id")
    parser.add_argument("--run-id")
    parser.add_argument("--session-id")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="Set up the beginner-friendly local configuration")
    init_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    init_cmd.add_argument("--host", default="127.0.0.1")
    init_cmd.add_argument("--port", type=int, default=8700)
    init_cmd.add_argument("--wizard", action="store_true", help="Ask for database and model configuration")
    init_cmd.add_argument("--force", action="store_true", help="Overwrite existing local configuration")
    init_cmd.add_argument("--local-test", action="store_true", help="Use SQLite and built-in lightweight models for temporary local testing")
    init_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    install_check_cmd = sub.add_parser("install-check", help="Initialize after install using home-local models or a safe fallback")
    install_check_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    install_check_cmd.add_argument("--force", action="store_true", help="Overwrite existing local configuration")
    install_check_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    download_models_cmd = sub.add_parser("download-models", help="Download local Qwen models into Fusion Memory home")
    download_models_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    download_models_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    doctor_cmd = sub.add_parser("doctor", help="Check whether Fusion Memory is ready to run")
    doctor_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    doctor_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    start_cmd = sub.add_parser("start", help="Start the local Fusion Memory service")
    start_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    start_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    start_cmd.add_argument("--wait-seconds", type=float, default=10.0)

    stop_cmd = sub.add_parser("stop", help="Stop the local Fusion Memory service")
    stop_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    stop_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    status_cmd = sub.add_parser("status", help="Show local Fusion Memory service status")
    status_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    status_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    backup_cmd = sub.add_parser("backup", help="Back up local configuration and SQLite data")
    backup_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    backup_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    upgrade_cmd = sub.add_parser("upgrade", help="Back up data and upgrade Fusion Memory")
    upgrade_cmd.add_argument("--home", default=None, help="Fusion Memory data directory")
    upgrade_cmd.add_argument("--package", default=None, help="Package/path to upgrade from")
    upgrade_cmd.add_argument("--dry-run", action="store_true")
    upgrade_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    install_agent_cmd = sub.add_parser("install-agent", help="Install or configure Agent adapters")
    install_agent_cmd.add_argument("--target", default="all")
    install_agent_cmd.add_argument("--dry-run", action="store_true")
    install_agent_cmd.add_argument("--home", default=None)
    install_agent_cmd.add_argument("--json", action="store_true")

    sync_haitun = sub.add_parser("sync-haitun-history", help="Sync Haitun saved history JSONL into Fusion Memory")
    _add_haitun_history_sync_args(sync_haitun)
    start_haitun_watcher = sub.add_parser(
        "start-haitun-history-watcher",
        help="Start Haitun passive history sync as a background process",
    )
    _add_haitun_history_watcher_args(start_haitun_watcher)
    status_haitun_watcher = sub.add_parser(
        "status-haitun-history-watcher",
        help="Show Haitun passive history sync process status",
    )
    _add_haitun_history_watcher_status_args(status_haitun_watcher)
    stop_haitun_watcher = sub.add_parser(
        "stop-haitun-history-watcher",
        help="Stop Haitun passive history sync background process",
    )
    _add_haitun_history_watcher_status_args(stop_haitun_watcher)

    alpha_cmd = sub.add_parser("alpha-test", help="Run local Fusion Memory alpha simulation")
    alpha_cmd.add_argument("--report", default=None)
    alpha_cmd.add_argument("--json", action="store_true")

    beta_cmd = sub.add_parser("beta-test", help="Run Fusion Memory beta simulation checks")
    beta_cmd.add_argument("--report", default=None)
    beta_cmd.add_argument("--json", action="store_true")

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

    beam = sub.add_parser("run-beam", help="Run a BEAM-style local benchmark split")
    beam.add_argument("dataset_path")
    beam.add_argument("--split", default="small", choices=sorted(BEAM_SPLITS))
    beam.add_argument("--ablate", action="store_true", help="Also report retrieval-mode and component ablations")
    _add_eval_model_args(beam)

    pg = sub.add_parser("migrate-postgres", help="Apply the Postgres/pgvector production schema")
    pg.add_argument("dsn", help="Postgres DSN, for example postgresql://user:pass@localhost:5432/fusion_memory")

    pg_verify = sub.add_parser("verify-postgres", help="Run a live Postgres backend smoke verification")
    pg_verify.add_argument("dsn", help="Postgres DSN, for example postgresql://user:pass@localhost:5432/fusion_memory")
    pg_verify.add_argument("--skip-migrate", action="store_true", help="Skip migration and only run the service smoke")

    embedding_server = sub.add_parser("embedding-server", help="Run one long-lived local embedding model server")
    _add_model_server_args(embedding_server)
    reranker_server = sub.add_parser("reranker-server", help="Run one long-lived local reranker model server")
    _add_model_server_args(reranker_server)

    token = sub.add_parser("token", help="Manage MCP bearer tokens")
    token_sub = token.add_subparsers(dest="token_command", required=True)
    token_create_cmd = token_sub.add_parser("create", help="Create a bearer token")
    _add_token_database_args(token_create_cmd)
    token_create_cmd.add_argument("--user-id", required=True)
    token_create_cmd.add_argument("--scopes", required=True)
    token_create_cmd.add_argument("--expires-at", default=None)
    token_list_cmd = token_sub.add_parser("list", help="List bearer tokens")
    _add_token_database_args(token_list_cmd)
    token_list_cmd.add_argument("--user-id", required=True)
    token_revoke_cmd = token_sub.add_parser("revoke", help="Revoke a bearer token")
    _add_token_database_args(token_revoke_cmd)
    token_revoke_cmd.add_argument("--token-id", required=True)

    _rewrite_compat_command_aliases(sys.argv)
    args = parser.parse_args()
    try:
        if args.command == "init":
            if args.wizard:
                return _print_product_result(
                    configure_interactive(args.home, host=args.host, port=args.port, force=args.force),
                    json_output=args.json,
                )
            elif args.local_test:
                return _print_product_result(
                    init_home(args.home, host=args.host, port=args.port, force=args.force, local_test=True),
                    json_output=args.json,
                )
            else:
                return _print_product_result(
                    init_home(args.home, host=args.host, port=args.port, force=args.force),
                    json_output=args.json,
                )
        if args.command == "doctor":
            return _print_product_result(doctor(args.home), json_output=args.json)
        if args.command == "install-check":
            return _print_product_result(install_readiness(args.home, force=args.force), json_output=args.json)
        if args.command == "download-models":
            from fusion_memory.windows_installer import download_qwen_models

            paths = product_paths(args.home)
            paths.home.mkdir(parents=True, exist_ok=True)
            result = download_qwen_models(
                Path.cwd(),
                log_dir=paths.home / "logs",
                models_dir=paths.models,
            )
            return _print_product_result(
                {
                    "ok": result.ok,
                    "step": result.step_name,
                    "error": result.error,
                    "models": str(paths.models),
                    "log": str(result.log_path) if result.log_path else "",
                },
                json_output=args.json,
            )
        if args.command == "start":
            return _print_product_result(start_service(args.home, wait_seconds=args.wait_seconds), json_output=args.json)
        if args.command == "stop":
            return _print_product_result(stop_service(args.home), json_output=args.json)
        if args.command == "status":
            return _print_product_result(service_status(args.home), json_output=args.json)
        if args.command == "backup":
            return _print_product_result(backup_data(args.home), json_output=args.json)
        if args.command == "upgrade":
            return _print_product_result(upgrade(args.home, package=args.package, dry_run=args.dry_run), json_output=args.json)
        if args.command == "install-agent":
            return _print_product_result(
                install_agent(args.target, dry_run=args.dry_run, home=args.home),
                json_output=args.json,
            )
        if args.command == "sync-haitun-history":
            config = config_from_workspace(
                workspace=Path(args.workspace),
                session_id=args.session_id,
                db_path=args.db,
                base_url=args.memory_url,
            )
            if args.background:
                return _print_product_result(
                    start_history_watcher_daemon(
                        config,
                        poll_interval_seconds=args.poll_interval_seconds,
                    ),
                    json_output=args.json,
                )
            if args.once:
                return _print_product_result(sync_haitun_history_once(config), json_output=args.json)
            else:
                watch_history(config, poll_interval_seconds=args.poll_interval_seconds)
            return
        if args.command == "start-haitun-history-watcher":
            config = config_from_workspace(
                workspace=Path(args.workspace),
                session_id=args.session_id,
                db_path=args.db,
                base_url=args.memory_url,
            )
            return _print_product_result(
                start_history_watcher_daemon(
                    config,
                    poll_interval_seconds=args.poll_interval_seconds,
                ),
                json_output=args.json,
            )
        if args.command == "status-haitun-history-watcher":
            config = config_from_workspace(
                workspace=Path(args.workspace),
                session_id=args.session_id,
                db_path=args.db,
                base_url=args.memory_url,
            )
            return _print_product_result(status_history_watcher_daemon(config), json_output=args.json)
        if args.command == "stop-haitun-history-watcher":
            config = config_from_workspace(
                workspace=Path(args.workspace),
                session_id=args.session_id,
                db_path=args.db,
                base_url=args.memory_url,
            )
            return _print_product_result(stop_history_watcher_daemon(config), json_output=args.json)
        if args.command == "alpha-test":
            return _print_product_result(run_alpha(report_path=args.report), json_output=args.json)
        if args.command == "beta-test":
            return _print_product_result(run_beta(report_path=args.report), json_output=args.json)
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
        if args.command == "embedding-server":
            from fusion_memory.model_server import run_embedding_server

            return run_embedding_server(args.host, args.port, args.model, args.device, args.max_concurrency)
        if args.command == "reranker-server":
            from fusion_memory.model_server import run_reranker_server

            return run_reranker_server(args.host, args.port, args.model, args.device, args.max_concurrency)
        if args.command == "token":
            store = _token_store_from_args(args)
            if args.token_command == "create":
                expires_at = datetime.fromisoformat(args.expires_at) if args.expires_at else None
                plaintext = token_create(store, args.user_id, _parse_token_scopes(args.scopes), expires_at)
                print(json.dumps({"token": plaintext}, ensure_ascii=False))
            elif args.token_command == "list":
                print(json.dumps(token_list(store, args.user_id), default=_jsonable, ensure_ascii=False, indent=2))
            else:
                print(json.dumps({"revoked": token_revoke(store, args.token_id)}, ensure_ascii=False))
            return

        scope = Scope(
            workspace_id=args.workspace_id,
            user_id=args.user_id,
            agent_id=args.agent_id,
            run_id=args.run_id,
            session_id=args.session_id,
        )
        service = memory_service_from_env(args.db)
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
            elif args.command == "run-beam":
                answer_model, judge_model = _build_eval_models(args)
                adapter = BeamAdapter(service, scope, split=args.split, answer_model=answer_model, judge_model=judge_model)
                output = adapter.run_dataset(args.dataset_path, split=args.split, ablate=args.ablate)
                print(json.dumps(output, ensure_ascii=False, indent=2))
        finally:
            service.close()
    except Exception as exc:
        payload = {"ok": False, **safe_product_error(exc)}
        return _print_product_result(payload, json_output=getattr(args, "json", False))


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


_VALID_TOKEN_SCOPES = {"memory:read", "memory:write", "memory:sync"}


def token_create(store: PostgresTokenStore, user_id: str, scopes: tuple[str, ...], expires_at: datetime | None) -> str:
    validate_token_scopes(scopes)
    plaintext, _record = store.create_token(user_id, scopes, expires_at)
    return plaintext


def token_list(store: PostgresTokenStore, user_id: str) -> list[dict[str, object]]:
    return [_token_record_output(record) for record in store.list_tokens(user_id)]


def token_revoke(store: PostgresTokenStore, token_id: str) -> bool:
    return store.revoke_token(token_id)


def validate_token_scopes(scopes: tuple[str, ...]) -> tuple[str, ...]:
    unknown = set(scopes) - _VALID_TOKEN_SCOPES
    if unknown:
        raise ValueError(f"unsupported token scopes: {', '.join(sorted(unknown))}")
    if "memory:sync" in scopes and "memory:write" not in scopes:
        raise ValueError("memory:sync requires memory:write")
    return scopes


def _token_record_output(record: TokenRecord) -> dict[str, object]:
    return {
        "token_id": record.token_id,
        "user_id": record.user_id,
        "scopes": list(record.scopes),
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "revoked_at": record.revoked_at.isoformat() if record.revoked_at else None,
        "created_at": record.created_at.isoformat(),
        "last_used_at": record.last_used_at.isoformat() if record.last_used_at else None,
    }


def _parse_token_scopes(value: str) -> tuple[str, ...]:
    scopes = tuple(scope.strip() for scope in value.split(",") if scope.strip())
    if not scopes:
        raise ValueError("at least one scope is required")
    return validate_token_scopes(scopes)


def _add_token_database_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dsn", default=None, help="Postgres DSN; defaults to FUSION_MEMORY_PG_DSN")


def _token_store_from_args(args: argparse.Namespace) -> PostgresTokenStore:
    dsn = args.dsn or os.getenv("FUSION_MEMORY_PG_DSN")
    pepper = os.getenv("FUSION_MEMORY_TOKEN_PEPPER")
    if not dsn:
        raise ValueError("Postgres DSN is required via --dsn or FUSION_MEMORY_PG_DSN")
    if not pepper:
        raise ValueError("FUSION_MEMORY_TOKEN_PEPPER is required")

    def connect():
        try:
            import psycopg2
        except ImportError as exc:
            raise RuntimeError("Postgres token commands require fusion-memory[postgres]") from exc
        return psycopg2.connect(dsn)

    return PostgresTokenStore(connect, pepper=pepper)


def _add_haitun_history_sync_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="Haitun workspace path")
    parser.add_argument("--session-id", required=True, help="Haitun session id")
    parser.add_argument("--memory-url", default=None, help="Fusion Memory service URL; defaults to PSI_MEMORY_BASE_URL or http://127.0.0.1:8700")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--background", action="store_true", help="Start the long-running watcher in the background and return its OS pid")
    parser.add_argument("--json", action="store_true")


def _add_haitun_history_watcher_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="Haitun workspace path")
    parser.add_argument("--session-id", required=True, help="Haitun session id")
    parser.add_argument("--memory-url", default=None, help="Fusion Memory service URL; defaults to PSI_MEMORY_BASE_URL or http://127.0.0.1:8700")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")


def _add_haitun_history_watcher_status_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="Haitun workspace path")
    parser.add_argument("--session-id", required=True, help="Haitun session id")
    parser.add_argument("--memory-url", default=None, help="Fusion Memory service URL; defaults to PSI_MEMORY_BASE_URL or http://127.0.0.1:8700")
    parser.add_argument("--json", action="store_true")


def _rewrite_compat_command_aliases(argv: list[str]) -> None:
    for index, value in enumerate(argv[1:], start=1):
        if value in _COMPAT_COMMAND_ALIASES:
            argv[index] = _COMPAT_COMMAND_ALIASES[value]
            return


def _print_product_result(result: dict, *, json_output: bool = False) -> int:
    result = _normalize_product_result(result)
    if json_output:
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    else:
        print(render_human(result))
    return 0 if result.get("ok", True) else 1


def _normalize_product_result(result: dict[str, object]) -> dict[str, object]:
    if result.get("ok", True):
        return result
    if "checks" in result:
        return result
    normalized = dict(result)
    normalized.setdefault("error", "unexpected_error")
    normalized.setdefault("message", "Fusion Memory could not complete the request.")
    normalized.setdefault("next_step", "Run fusion-memory doctor and check the help text or local log file.")
    return normalized


def _friendly_argparse_message(message: str) -> str:
    if "invalid choice" in message and "--target" in message:
        return "Unknown Agent target. Choose one of: all, dolphin, openclaw, hermes, fusion-agent."
    if "the following arguments are required" in message:
        return "A required command option is missing. Run fusion-memory doctor or check the help text."
    return "Fusion Memory could not understand that command. Check the command and try again."


def _add_eval_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-config-file",
        default=None,
        help="Loose key/base_url/model_use config file for benchmark answer and judge models",
    )
    parser.add_argument("--answer-endpoint", default=None, help="OpenAI-compatible chat/completions endpoint for benchmark answers")
    parser.add_argument("--answer-model", default=None, help="Model name sent to --answer-endpoint")
    parser.add_argument("--answer-api-key", default=None, help="Bearer token for --answer-endpoint")
    parser.add_argument("--judge-endpoint", default=None, help="OpenAI-compatible chat/completions endpoint for semantic judging")
    parser.add_argument("--judge-model", default=None, help="Model name sent to --judge-endpoint")
    parser.add_argument("--judge-api-key", default=None, help="Bearer token for --judge-endpoint")
    parser.add_argument("--model-api-key", default=None, help="Shared fallback Bearer token for answer/judge endpoints")
    parser.add_argument("--model-timeout-seconds", type=float, default=None, help="HTTP timeout for answer/judge model calls")
    parser.add_argument(
        "--use-llm-aggregation",
        action="store_true",
        default=None,
        help="Use a strict LLM aggregation pass for multi-session count/list evidence before answering",
    )
    parser.add_argument(
        "--llm-aggregation-min-confidence",
        type=float,
        default=None,
        help="Minimum confidence accepted from the LLM aggregation pass",
    )


def _add_model_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face model identifier")
    parser.add_argument("--device", default=None, help="Optional model device, for example cuda:0")
    parser.add_argument("--max-concurrency", type=int, default=1)


def _build_eval_models(args: argparse.Namespace):
    answer_model = None
    judge_model = None
    file_config = _eval_model_config_from_file(getattr(args, "model_config_file", None))
    shared_endpoint = _env_endpoint("FUSION_MEMORY_EVAL_ENDPOINT", "FUSION_MEMORY_EVAL_BASE_URL")
    answer_endpoint = (
        getattr(args, "answer_endpoint", None)
        or _env_endpoint("FUSION_MEMORY_EVAL_ANSWER_ENDPOINT", "FUSION_MEMORY_EVAL_ANSWER_BASE_URL")
        or shared_endpoint
        or file_config.get("endpoint")
    )
    judge_endpoint = (
        getattr(args, "judge_endpoint", None)
        or _env_endpoint("FUSION_MEMORY_EVAL_JUDGE_ENDPOINT", "FUSION_MEMORY_EVAL_JUDGE_BASE_URL")
        or shared_endpoint
        or file_config.get("endpoint")
    )
    shared_api_key = getattr(args, "model_api_key", None) or os.getenv("FUSION_MEMORY_EVAL_MODEL_API_KEY") or file_config.get("api_key")
    shared_model = os.getenv("FUSION_MEMORY_EVAL_MODEL") or file_config.get("model")
    timeout_seconds = getattr(args, "model_timeout_seconds", None) or _float_env("FUSION_MEMORY_EVAL_TIMEOUT_SECONDS", 30.0)
    retry_attempts = _int_env("FUSION_MEMORY_EVAL_RETRY_ATTEMPTS", 5)
    retry_backoff_seconds = _float_env("FUSION_MEMORY_EVAL_RETRY_BACKOFF_SECONDS", 2.0)
    retry_max_backoff_seconds = _float_env("FUSION_MEMORY_EVAL_RETRY_MAX_BACKOFF_SECONDS", 60.0)
    min_interval_seconds = _float_env("FUSION_MEMORY_EVAL_MIN_INTERVAL_SECONDS", 1.0)
    use_llm_aggregation = _bool_arg_or_env(args, "use_llm_aggregation", "FUSION_MEMORY_EVAL_USE_LLM_AGGREGATION", False)
    llm_aggregation_min_confidence = (
        getattr(args, "llm_aggregation_min_confidence", None)
        if getattr(args, "llm_aggregation_min_confidence", None) is not None
        else _float_env("FUSION_MEMORY_EVAL_LLM_AGGREGATION_MIN_CONFIDENCE", 0.70)
    )
    if answer_endpoint:
        answer_client = OpenAICompatibleLLMClient(
            answer_endpoint,
            api_key=getattr(args, "answer_api_key", None) or os.getenv("FUSION_MEMORY_EVAL_ANSWER_API_KEY") or shared_api_key,
            model=getattr(args, "answer_model", None) or os.getenv("FUSION_MEMORY_EVAL_ANSWER_MODEL") or shared_model or "eval-answer",
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_max_backoff_seconds=retry_max_backoff_seconds,
            min_interval_seconds=min_interval_seconds,
        )
        answer_model = OpenAICompatibleAnswerModel(
            answer_client,
            use_llm_aggregation=use_llm_aggregation,
            llm_aggregation_min_confidence=llm_aggregation_min_confidence,
        )
    if judge_endpoint:
        judge_client = OpenAICompatibleLLMClient(
            judge_endpoint,
            api_key=getattr(args, "judge_api_key", None) or os.getenv("FUSION_MEMORY_EVAL_JUDGE_API_KEY") or shared_api_key,
            model=getattr(args, "judge_model", None) or os.getenv("FUSION_MEMORY_EVAL_JUDGE_MODEL") or shared_model or "eval-judge",
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_max_backoff_seconds=retry_max_backoff_seconds,
            min_interval_seconds=min_interval_seconds,
        )
        judge_model = OpenAICompatibleJudgeModel(judge_client)
    return answer_model, judge_model


def _eval_model_config_from_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    values: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, raw_value = line.split("=", 1)
            key = _normalize_eval_config_key(key)
            value = _strip_loose_quotes(raw_value.strip())
            _store_eval_config_value(values, key, value)
            continue
        if ":" in line:
            key, raw_value = line.split(":", 1)
            key = _normalize_eval_config_key(key)
            if key:
                _store_eval_config_value(values, key, _strip_loose_quotes(raw_value.strip()))
                continue
        if "api_key" not in values:
            values["api_key"] = _strip_loose_quotes(line)
    return values


def _normalize_eval_config_key(key: str) -> str:
    return _strip_loose_quotes(key).strip().lower().replace("-", "_")


def _store_eval_config_value(values: dict[str, str], key: str, value: str) -> None:
    if key in {"base_url", "url"}:
        values["endpoint"] = _endpoint_from_eval_base_url(value)
    elif key in {"endpoint", "answer_endpoint", "judge_endpoint"}:
        values["endpoint"] = _strip_loose_quotes(value)
    elif key in {"api_key", "key", "token", "model_api_key", "openai_api_key"}:
        values["api_key"] = value
    elif key in {"model", "model_use", "model_name"}:
        model = _first_model_name(value)
        if model:
            values["model"] = model


def _strip_loose_quotes(value: str) -> str:
    value = value.strip().strip(",")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def _first_model_name(value: str) -> str | None:
    value = _strip_loose_quotes(value)
    # Local key files often document alternatives as "model_a or model_b".
    value = re.split(r"\s+or\s+|[,/]", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    value = re.sub(r"^gpt(?=\d)", "gpt-", value, flags=re.IGNORECASE)
    return value or None


def _endpoint_from_eval_base_url(base_url: str) -> str:
    base_url = _strip_loose_quotes(base_url).rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _env_endpoint(endpoint_name: str, base_url_name: str) -> str | None:
    endpoint = os.getenv(endpoint_name)
    if endpoint:
        return endpoint
    base_url = os.getenv(base_url_name)
    if not base_url:
        return None
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _bool_arg_or_env(args: argparse.Namespace, arg_name: str, env_name: str, default: bool) -> bool:
    value = getattr(args, arg_name, None)
    if value is not None:
        return bool(value)
    raw = os.getenv(env_name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
