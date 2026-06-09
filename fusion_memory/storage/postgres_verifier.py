from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from fusion_memory.api.service import MemoryService
from fusion_memory.core.models import Scope
from fusion_memory.storage.postgres_store import PostgresMigrationReport, PostgresMigrationRunner


@dataclass
class PostgresVerificationReport:
    backend: str
    dsn: str
    migrated: bool
    migration: dict[str, Any] | None
    scope: dict[str, str | None]
    checks: dict[str, bool]
    add: dict[str, Any]
    search: dict[str, Any]
    answer_context: dict[str, Any]
    tasks: dict[str, Any]


def verify_postgres_backend(
    dsn: str,
    *,
    migrate: bool = True,
    runner_factory: Callable[[str], Any] | None = None,
    service_factory: Callable[[str], Any] | None = None,
) -> PostgresVerificationReport:
    migration: PostgresMigrationReport | None = None
    if migrate:
        runner = runner_factory(dsn) if runner_factory else PostgresMigrationRunner(dsn)
        try:
            migration = runner.migrate()
        finally:
            runner.close()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    scope = Scope(workspace_id=f"fm_verify_{stamp}", user_id="verify_user", agent_id="verify_agent", session_id=f"verify_session_{stamp}")
    service = service_factory(dsn) if service_factory else MemoryService(dsn, storage_backend="postgres")
    try:
        add_result = service.add(
            [
                {"role": "user", "content": f"Atlas verification {stamp} uses Qdrant for retrieval."},
                {"role": "assistant", "content": "I noted the Atlas retrieval backend."},
                {"role": "user", "content": "Reports verification uses PostgreSQL."},
                {"role": "assistant", "content": "I will keep the report backend on PostgreSQL."},
                {"role": "user", "content": "Reranking verification should use a cross encoder."},
                {"role": "assistant", "content": "I will include reranking in the retrieval plan."},
            ],
            scope,
            datetime.now(timezone.utc),
            {"min_window_spans": 3, "window_size": 3},
        )
        search_result = service.search(f"Qdrant Atlas verification {stamp}", scope, options={"limit": 6})
        pack = service.answer_context(f"What retrieval backend did Atlas verification {stamp} use?", scope, budget={"limit": 6})
        tasks_before = service.list_background_tasks(scope, status="pending")
        processed = service.process_background_tasks(scope, limit=10)
        audits = service.audit_events(scope, limit=20)

        checks = {
            "migration_applied": (not migrate) or migration is not None,
            "spans_written": bool(add_result.span_ids),
            "facts_or_events_written": bool(add_result.accepted_fact_ids or add_result.accepted_event_ids),
            "search_returned_candidates": bool(search_result.candidates),
            "answer_context_has_evidence": bool(pack.source_spans or pack.facts or pack.current_views),
            "audit_written": bool(audits),
            "tasks_visible": isinstance(tasks_before, list),
            "task_processor_ran": processed.get("processed_count", 0) >= 0,
        }
        return PostgresVerificationReport(
            backend="postgres",
            dsn=_mask_dsn(dsn),
            migrated=migrate,
            migration=_report_dict(migration),
            scope=scope.__dict__,
            checks=checks,
            add={
                "span_count": len(add_result.span_ids),
                "accepted_fact_count": len(add_result.accepted_fact_ids),
                "accepted_event_count": len(add_result.accepted_event_ids),
                "trace_id": add_result.trace_id,
            },
            search={"candidate_count": len(search_result.candidates), "trace_id": search_result.trace_id},
            answer_context={
                "source_span_count": len(pack.source_spans),
                "fact_count": len(pack.facts),
                "event_count": len(pack.events),
                "current_view_count": len(pack.current_views),
                "answer_policy": pack.answer_policy,
            },
            tasks={
                "pending_before": len(tasks_before),
                "processed_count": processed.get("processed_count", 0),
                "status_counts": processed.get("status_counts", {}),
            },
        )
    finally:
        service.close()


def _report_dict(report: PostgresMigrationReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "backend": report.backend,
        "migration_path": report.migration_path,
        "applied_statements": report.applied_statements,
        "tables": report.tables,
    }


def _mask_dsn(dsn: str) -> str:
    parsed = urlsplit(dsn)
    if not parsed.password:
        return dsn
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    user = parsed.username or ""
    netloc = f"{user}:***@{host}" if user else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
