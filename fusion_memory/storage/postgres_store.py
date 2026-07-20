from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator, Mapping

from fusion_memory.core.chronology import (
    ChronologyEventEdge,
    ChronologyEventNode,
    ChronologyPhase,
    ChronologyTopic,
)
from fusion_memory.core.embedding import DeterministicEmbedder, Embedder
from fusion_memory.core.models import (
    CurrentView,
    EncodingDecision,
    EntityProfile,
    EntityRecord,
    EventEdge,
    EvidenceSpan,
    FactRelation,
    MemoryEvent,
    MemoryFact,
    Scope,
    new_id,
)
from fusion_memory.core.text import keyword_score, stable_hash
from fusion_memory.storage.postgres_pool import PostgresConnectionPool, RetryableOperationError


DEFAULT_POSTGRES_MIGRATION = Path(__file__).parent / "migrations" / "postgres" / "001_init.sql"
POSTGRES_TABLES = [
    "evidence_spans",
    "memory_facts",
    "fact_relations",
    "events",
    "event_edges",
    "current_views",
    "entity_profiles",
    "entities",
    "encoding_decisions",
    "retrieval_utility_examples",
    "debug_traces",
    "audit_events",
    "background_tasks",
    "chronology_topics",
    "chronology_phases",
    "chronology_event_nodes",
    "chronology_event_edges",
]


class PostgresBackendUnavailable(RuntimeError):
    pass


class _PostgresRepository:
    """Common transaction ownership behavior for standalone and bound facades."""

    def _commit_if_owner(self) -> None:
        if getattr(self, "_manage_transaction", True):
            self.connect().commit()

    def _rollback_if_owner(self) -> None:
        if getattr(self, "_manage_transaction", True):
            self.connect().rollback()


@dataclass
class PostgresMigrationReport:
    backend: str
    migration_path: str
    applied_statements: int
    tables: list[str]


class PostgresMigrationRunner:
    """Apply the Fusion Memory Postgres/pgvector schema.

    This is intentionally separated from the SQLite runtime store. The local
    MVP remains dependency-free, while production deployments can opt into a
    real Postgres schema migration by installing `psycopg2`.
    """

    def __init__(
        self,
        dsn: str,
        *,
        connect: Callable[[str], Any] | None = None,
        migration_path: str | Path = DEFAULT_POSTGRES_MIGRATION,
    ) -> None:
        self.dsn = dsn
        self._connect = connect or _default_connect
        self.migration_path = Path(migration_path)
        self.conn: Any | None = None

    def connect(self) -> Any:
        if self.conn is None:
            self.conn = self._connect(self.dsn)
        return self.conn

    def migrate(self) -> PostgresMigrationReport:
        conn = self.connect()
        cursor = conn.cursor()
        applied_statements = 0
        try:
            cursor.execute("select pg_advisory_xact_lock(hashtext('fusion_memory_schema_migrations'))")
            cursor.execute(
                """
                create table if not exists fusion_memory_schema_migrations (
                    version text primary key,
                    applied_at timestamptz not null default now()
                )
                """
            )
            for migration in sorted(self.migration_path.parent.glob("*.sql")):
                _execute_migration_query(
                    cursor,
                    "select 1 from fusion_memory_schema_migrations where version = %s",
                    (migration.name,),
                )
                row = cursor.fetchone() if hasattr(cursor, "fetchone") else None
                if row is not None:
                    continue
                statements = _split_sql_statements(migration.read_text(encoding="utf-8"))
                for statement in statements:
                    cursor.execute(statement)
                _execute_migration_query(
                    cursor,
                    "insert into fusion_memory_schema_migrations (version) values (%s)",
                    (migration.name,),
                )
                applied_statements += len(statements)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
        return PostgresMigrationReport(
            backend="postgres",
            migration_path=str(self.migration_path),
            applied_statements=applied_statements,
            tables=list(POSTGRES_TABLES),
        )

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PostgresMigrationRunner":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PostgresEvidenceRepository(_PostgresRepository):
    """Postgres/pgvector CRUD boundary for Layer 1 evidence spans.

    The local MVP uses SQLite by default, but production deployments need a
    repository that speaks the migration schema directly. This class keeps the
    dependency boundary optional: callers can inject a fake connection in tests,
    or install psycopg2 for a real Postgres deployment.
    """

    def __init__(
        self,
        dsn: str,
        *,
        connect: Callable[[str], Any] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.dsn = dsn
        self._connect = connect or _default_connect
        self.embedder = embedder or DeterministicEmbedder()
        self.conn: Any | None = None

    def connect(self) -> Any:
        if self.conn is None:
            self.conn = self._connect(self.dsn)
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PostgresEvidenceRepository":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def insert_span(self, span: EvidenceSpan) -> bool:
        metadata = dict(span.metadata)
        line_start = metadata.get("line_start")
        line_end = metadata.get("line_end")
        embedding = _pgvector_literal(self.embedder.embed_text(span.content))
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into evidence_spans
                (span_id, workspace_id, user_id, agent_id, run_id, session_id, app_id,
                 turn_id, speaker, span_type, content, content_hash, timestamp, source_uri,
                 line_start, line_end, parent_span_id, entities, topics, embedding_dense, metadata)
                values
                (%s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s::jsonb, %s::jsonb, cast(%s as vector), %s::jsonb)
                on conflict (span_id) do nothing
                """,
                (
                    span.span_id,
                    span.scope.workspace_id,
                    span.scope.user_id,
                    span.scope.agent_id,
                    span.scope.run_id,
                    span.scope.session_id,
                    span.scope.app_id,
                    span.turn_id,
                    span.speaker,
                    span.span_type,
                    span.content,
                    span.content_hash,
                    _dt_to_pg(span.timestamp),
                    span.source_uri,
                    line_start,
                    line_end,
                    span.parent_span_id,
                    _json_dumps(span.entities),
                    _json_dumps(span.topics),
                    embedding,
                    _json_dumps(metadata),
                ),
            )
            inserted = cursor.rowcount > 0
            self._commit_if_owner()
            return inserted
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def get_span(self, span_id: str, scope: Scope | None = None, *, include_session: bool = False) -> EvidenceSpan | None:
        where = "span_id = %s"
        params: list[Any] = [span_id]
        if scope:
            scope_where, scope_params = _scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        rows = self._fetch_all(f"select * from evidence_spans where {where} limit 1", params)
        return _row_to_span(rows[0]) if rows else None

    def list_spans(self, scope: Scope, *, include_session: bool = False, limit: int | None = None) -> list[EvidenceSpan]:
        where, params = _scope_where(scope, include_session=include_session)
        sql = f"select * from evidence_spans where {where} order by timestamp, created_at"
        if limit is not None:
            sql += " limit %s"
            params.append(limit)
        return [_row_to_span(row) for row in self._fetch_all(sql, params)]

    def find_duplicate_span(self, content_hash: str, scope: Scope) -> EvidenceSpan | None:
        where, params = _scope_where(scope)
        rows = self._fetch_all(
            f"select * from evidence_spans where {where} and content_hash = %s limit 1",
            [*params, content_hash],
        )
        return _row_to_span(rows[0]) if rows else None

    def search_spans(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        speaker: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[EvidenceSpan, dict[str, float]]]:
        where, params = _scope_where(scope, include_session=include_session, alias="e")
        if speaker:
            where += " and e.speaker = %s"
            params.append(speaker)
        query_text = query.strip()
        query_vector = _pgvector_literal(self.embedder.embed_text(query_text))
        like_query = f"%{query_text}%"
        rows = self._fetch_all(
            f"""
            with scored as (
              select
                e.*,
                ts_rank_cd(e.search_tsv, plainto_tsquery('simple', %s)) as bm25_score,
                coalesce(1.0 - (e.embedding_dense <=> cast(%s as vector)), 0.0) as semantic_score
              from evidence_spans e
              where {where}
            )
            select
              *,
              (0.55 * semantic_score + 0.45 * bm25_score) as score
            from scored
            where bm25_score > 0 or semantic_score > 0 or content ilike %s
            order by score desc, timestamp desc nulls last
            limit %s
            """,
            [query_text, query_vector, *params, like_query, limit],
        )
        out: list[tuple[EvidenceSpan, dict[str, float]]] = []
        for row in rows:
            bm25 = float(row.get("bm25_score") or 0.0)
            semantic = float(row.get("semantic_score") or 0.0)
            score = float(row.get("score") if row.get("score") is not None else 0.55 * semantic + 0.45 * bm25)
            out.append(
                (
                    _row_to_span(row),
                    {
                        "semantic_score": semantic,
                        "bm25_score": bm25,
                        "sparse_source": 1.0 if bm25 > 0 else 0.0,
                        "score": score,
                    },
                )
            )
        return out

    def _fetch_all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self.connect().cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [_row_as_dict(row, cursor.description) for row in rows]
        finally:
            cursor.close()


class PostgresFactRepository(_PostgresRepository):
    """Postgres/pgvector CRUD boundary for Layer 3 facts and fact relations."""

    def __init__(
        self,
        dsn: str,
        *,
        connect: Callable[[str], Any] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.dsn = dsn
        self._connect = connect or _default_connect
        self.embedder = embedder or DeterministicEmbedder()
        self.conn: Any | None = None

    def connect(self) -> Any:
        if self.conn is None:
            self.conn = self._connect(self.dsn)
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PostgresFactRepository":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def insert_fact(self, fact: MemoryFact) -> bool:
        embedding = _pgvector_literal(self.embedder.embed_text(fact.text))
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into memory_facts
                (fact_id, workspace_id, user_id, agent_id, run_id, session_id, app_id,
                 subject, predicate, object, text, category, polarity, confidence, salience,
                 observed_at, valid_from, valid_to, source_span_ids, linked_fact_ids,
                 embedding_dense, hash, metadata, created_at)
                values
                (%s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s::jsonb, %s::jsonb,
                 cast(%s as vector), %s, %s::jsonb, %s)
                on conflict (fact_id) do nothing
                """,
                (
                    fact.fact_id,
                    fact.scope.workspace_id,
                    fact.scope.user_id,
                    fact.scope.agent_id,
                    fact.scope.run_id,
                    fact.scope.session_id,
                    fact.scope.app_id,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.text,
                    fact.category,
                    fact.polarity,
                    fact.confidence,
                    fact.salience,
                    _dt_to_pg(fact.observed_at),
                    _dt_to_pg(fact.valid_from),
                    _dt_to_pg(fact.valid_to),
                    _json_dumps(fact.source_span_ids),
                    _json_dumps(fact.linked_fact_ids),
                    embedding,
                    fact.metadata.get("hash"),
                    _json_dumps(fact.metadata),
                    _dt_to_pg(fact.created_at),
                ),
            )
            inserted = cursor.rowcount > 0
            self._commit_if_owner()
            return inserted
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def get_fact(self, fact_id: str, scope: Scope | None = None, *, include_session: bool = False) -> MemoryFact | None:
        where = "fact_id = %s"
        params: list[Any] = [fact_id]
        if scope:
            scope_where, scope_params = _scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        rows = self._fetch_all(f"select * from memory_facts where {where} limit 1", params)
        return _row_to_fact(rows[0]) if rows else None

    def list_facts(
        self,
        scope: Scope,
        category: str | None = None,
        *,
        include_session: bool = False,
        limit: int | None = None,
    ) -> list[MemoryFact]:
        where, params = _scope_where(scope, include_session=include_session)
        if category:
            where += " and category = %s"
            params.append(category)
        sql = f"select * from memory_facts where {where} order by created_at"
        if limit is not None:
            sql += " limit %s"
            params.append(limit)
        return [_row_to_fact(row) for row in self._fetch_all(sql, params)]

    def search_facts(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        category: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[MemoryFact, dict[str, float]]]:
        where, params = _scope_where(scope, include_session=include_session, alias="f")
        if category:
            where += " and f.category = %s"
            params.append(category)
        query_text = query.strip()
        query_vector = _pgvector_literal(self.embedder.embed_text(query_text))
        rows = self._fetch_all(
            f"""
            with scored as (
              select
                f.*,
                ts_rank_cd(f.search_tsv, plainto_tsquery('simple', %s)) as bm25_score,
                coalesce(1.0 - (f.embedding_dense <=> cast(%s as vector)), 0.0) as semantic_score,
                case
                  when exists (
                    select 1 from fact_relations r
                    where r.relation_type = 'supersedes' and r.to_fact_id = f.fact_id
                  )
                  then -0.15 else 0.0
                end as active_prior
              from memory_facts f
              where {where}
            )
            select
              *,
              (0.50 * semantic_score + 0.35 * bm25_score + 0.10 * confidence + 0.05 * salience + active_prior) as score
            from scored
            where bm25_score > 0 or semantic_score > 0
            order by score desc, created_at desc
            limit %s
            """,
            [query_text, query_vector, *params, limit],
        )
        out: list[tuple[MemoryFact, dict[str, float]]] = []
        for row in rows:
            bm25 = float(row.get("bm25_score") or 0.0)
            semantic = float(row.get("semantic_score") or 0.0)
            score = float(row.get("score") or 0.0)
            out.append(
                (
                    _row_to_fact(row),
                    {
                        "semantic_score": semantic,
                        "bm25_score": bm25,
                        "sparse_source": 1.0 if bm25 > 0 else 0.0,
                        "score": score,
                    },
                )
            )
        return out

    def insert_fact_relation(self, relation: FactRelation) -> bool:
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into fact_relations
                (relation_id, from_fact_id, to_fact_id, relation_type, source_span_ids, confidence)
                values (%s, %s, %s, %s, %s::jsonb, %s)
                on conflict (relation_id) do nothing
                """,
                (
                    relation.relation_id,
                    relation.from_fact_id,
                    relation.to_fact_id,
                    relation.relation_type,
                    _json_dumps(relation.source_span_ids),
                    relation.confidence,
                ),
            )
            inserted = cursor.rowcount > 0
            self._commit_if_owner()
            return inserted
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def list_fact_relations(self, fact_id: str | None = None, relation_type: str | None = None) -> list[FactRelation]:
        clauses: list[str] = []
        params: list[Any] = []
        if fact_id:
            clauses.append("(from_fact_id = %s or to_fact_id = %s)")
            params.extend([fact_id, fact_id])
        if relation_type:
            clauses.append("relation_type = %s")
            params.append(relation_type)
        where = " and ".join(clauses) if clauses else "1=1"
        rows = self._fetch_all(f"select * from fact_relations where {where} order by created_at", params)
        return [_row_to_fact_relation(row) for row in rows]

    def superseded_fact_ids(self) -> set[str]:
        rows = self._fetch_all("select to_fact_id from fact_relations where relation_type = 'supersedes'", [])
        return {str(row["to_fact_id"]) for row in rows}

    def _fetch_all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self.connect().cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [_row_as_dict(row, cursor.description) for row in rows]
        finally:
            cursor.close()


class PostgresEventRepository(_PostgresRepository):
    """Postgres CRUD boundary for Layer 4 events and temporal edges."""

    def __init__(self, dsn: str, *, connect: Callable[[str], Any] | None = None) -> None:
        self.dsn = dsn
        self._connect = connect or _default_connect
        self.conn: Any | None = None

    def connect(self) -> Any:
        if self.conn is None:
            self.conn = self._connect(self.dsn)
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PostgresEventRepository":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def insert_event(self, event: MemoryEvent) -> bool:
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into events
                (event_id, workspace_id, user_id, agent_id, run_id, session_id, app_id,
                 event_type, participants, description, time_start, time_end, time_granularity,
                 time_source, source_span_ids, fact_ids, confidence)
                values
                (%s, %s, %s, %s, %s, %s, %s,
                 %s, %s::jsonb, %s, %s, %s, %s,
                 %s, %s::jsonb, %s::jsonb, %s)
                on conflict (event_id) do nothing
                """,
                (
                    event.event_id,
                    event.scope.workspace_id,
                    event.scope.user_id,
                    event.scope.agent_id,
                    event.scope.run_id,
                    event.scope.session_id,
                    event.scope.app_id,
                    event.event_type,
                    _json_dumps(event.participants),
                    event.description,
                    _dt_to_pg(event.time_start),
                    _dt_to_pg(event.time_end),
                    event.time_granularity,
                    event.time_source,
                    _json_dumps(event.source_span_ids),
                    _json_dumps(event.fact_ids),
                    event.confidence,
                ),
            )
            inserted = cursor.rowcount > 0
            self._commit_if_owner()
            return inserted
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def get_event(self, event_id: str, scope: Scope | None = None, *, include_session: bool = False) -> MemoryEvent | None:
        where = "event_id = %s"
        params: list[Any] = [event_id]
        if scope:
            scope_where, scope_params = _scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        rows = self._fetch_all(f"select * from events where {where} limit 1", params)
        return _row_to_event(rows[0]) if rows else None

    def list_events(self, scope: Scope, *, include_session: bool = False, limit: int | None = None) -> list[MemoryEvent]:
        where, params = _scope_where(scope, include_session=include_session)
        sql = f"select * from events where {where} order by time_start, created_at"
        if limit is not None:
            sql += " limit %s"
            params.append(limit)
        return [_row_to_event(row) for row in self._fetch_all(sql, params)]

    def search_events(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        *,
        include_session: bool = False,
    ) -> list[tuple[MemoryEvent, dict[str, float]]]:
        where, params = _scope_where(scope, include_session=include_session, alias="e")
        query_text = query.strip()
        rows = self._fetch_all(
            f"""
            with scored as (
              select
                e.*,
                ts_rank_cd(e.search_tsv, plainto_tsquery('simple', %s)) as bm25_score,
                case when e.time_start is null then 0.0 else 0.2 end as temporal_fit
              from events e
              where {where}
            )
            select
              *,
              (0.75 * bm25_score + temporal_fit + 0.05 * confidence) as score
            from scored
            where bm25_score > 0 or description ilike %s
            order by score desc, time_start desc nulls last
            limit %s
            """,
            [query_text, *params, f"%{query_text}%", limit],
        )
        out: list[tuple[MemoryEvent, dict[str, float]]] = []
        for row in rows:
            bm25 = float(row.get("bm25_score") or 0.0)
            temporal = float(row.get("temporal_fit") or 0.0)
            score = float(row.get("score") or 0.0)
            out.append(
                (
                    _row_to_event(row),
                    {
                        "bm25_score": bm25,
                        "sparse_source": 1.0 if bm25 > 0 else 0.0,
                        "temporal_fit": temporal,
                        "score": score,
                    },
                )
            )
        return out

    def insert_event_edge(self, edge: EventEdge) -> bool:
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into event_edges
                (edge_id, from_event_id, to_event_id, edge_type, source_span_ids, confidence)
                values (%s, %s, %s, %s, %s::jsonb, %s)
                on conflict (edge_id) do nothing
                """,
                (
                    edge.edge_id,
                    edge.from_event_id,
                    edge.to_event_id,
                    edge.edge_type,
                    _json_dumps(edge.source_span_ids),
                    edge.confidence,
                ),
            )
            inserted = cursor.rowcount > 0
            self._commit_if_owner()
            return inserted
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def has_event_edge(self, from_event_id: str, to_event_id: str, edge_type: str | None = None) -> bool:
        where = "from_event_id = %s and to_event_id = %s"
        params: list[Any] = [from_event_id, to_event_id]
        if edge_type:
            where += " and edge_type = %s"
            params.append(edge_type)
        rows = self._fetch_all(f"select 1 as present from event_edges where {where} limit 1", params)
        return bool(rows)

    def get_event_edge(self, from_event_id: str, to_event_id: str) -> dict[str, Any] | None:
        rows = self._fetch_all(
            """
            select * from event_edges
            where from_event_id = %s and to_event_id = %s
            order by confidence desc
            limit 1
            """,
            [from_event_id, to_event_id],
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "edge_id": row["edge_id"],
            "edge_type": row["edge_type"],
            "source_span_ids": _json_loads(row.get("source_span_ids"), []),
            "confidence": float(row["confidence"]),
        }

    def _fetch_all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self.connect().cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [_row_as_dict(row, cursor.description) for row in rows]
        finally:
            cursor.close()


class PostgresViewProfileRepository(_PostgresRepository):
    """Postgres CRUD/search boundary for Layer 5 views, profiles, and entities."""

    def __init__(
        self,
        dsn: str,
        *,
        connect: Callable[[str], Any] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.dsn = dsn
        self._connect = connect or _default_connect
        self.embedder = embedder or DeterministicEmbedder()
        self.conn: Any | None = None

    def connect(self) -> Any:
        if self.conn is None:
            self.conn = self._connect(self.dsn)
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PostgresViewProfileRepository":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def upsert_current_view(self, view: CurrentView) -> bool:
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into current_views
                (view_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, view_type, subject, text,
                 state_json, source_fact_ids, source_event_ids, source_span_ids, confidence, updated_at, expires_at)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, null)
                on conflict (view_id) do update set
                  text = excluded.text,
                  state_json = excluded.state_json,
                  source_fact_ids = excluded.source_fact_ids,
                  source_event_ids = excluded.source_event_ids,
                  source_span_ids = excluded.source_span_ids,
                  confidence = excluded.confidence,
                  updated_at = excluded.updated_at
                """,
                (
                    view.view_id,
                    view.scope.workspace_id,
                    view.scope.user_id,
                    view.scope.agent_id,
                    view.scope.run_id,
                    view.scope.session_id,
                    view.scope.app_id,
                    view.view_type,
                    view.subject,
                    view.text,
                    _json_dumps(view.state_json),
                    _json_dumps(view.source_fact_ids),
                    _json_dumps(view.source_event_ids),
                    _json_dumps(view.source_span_ids),
                    view.confidence,
                    _dt_to_pg(view.updated_at),
                ),
            )
            self._commit_if_owner()
            return cursor.rowcount > 0
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def list_current_views(self, scope: Scope, view_type: str | None = None, *, include_session: bool = False) -> list[CurrentView]:
        where, params = _scope_where(scope, include_session=include_session)
        if view_type:
            where += " and view_type = %s"
            params.append(view_type)
        rows = self._fetch_all(f"select * from current_views where {where} order by updated_at desc", params)
        return [_row_to_view(row) for row in rows]

    def upsert_entity_profile(self, profile: EntityProfile) -> bool:
        embedding = _pgvector_literal(self.embedder.embed_text(profile.text))
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into entity_profiles
                (profile_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, entity_id, entity_type,
                 profile_type, text, state_json, source_fact_ids, source_event_ids, source_span_ids, confidence,
                 support_count, last_observed_at, updated_at, expires_at, embedding_dense)
                values
                (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                 %s, %s, %s, null, cast(%s as vector))
                on conflict (profile_id) do update set
                  text = excluded.text,
                  state_json = excluded.state_json,
                  source_fact_ids = excluded.source_fact_ids,
                  source_event_ids = excluded.source_event_ids,
                  source_span_ids = excluded.source_span_ids,
                  confidence = excluded.confidence,
                  support_count = excluded.support_count,
                  last_observed_at = excluded.last_observed_at,
                  updated_at = excluded.updated_at,
                  embedding_dense = excluded.embedding_dense
                """,
                (
                    profile.profile_id,
                    profile.scope.workspace_id,
                    profile.scope.user_id,
                    profile.scope.agent_id,
                    profile.scope.run_id,
                    profile.scope.session_id,
                    profile.scope.app_id,
                    profile.entity_id,
                    profile.entity_type,
                    profile.profile_type,
                    profile.text,
                    _json_dumps(profile.state_json),
                    _json_dumps(profile.source_fact_ids),
                    _json_dumps(profile.source_event_ids),
                    _json_dumps(profile.source_span_ids),
                    profile.confidence,
                    profile.support_count,
                    _dt_to_pg(profile.last_observed_at),
                    _dt_to_pg(profile.updated_at),
                    embedding,
                ),
            )
            self._commit_if_owner()
            return cursor.rowcount > 0
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def list_entity_profiles(self, scope: Scope, entity_id: str | None = None, *, include_session: bool = False) -> list[EntityProfile]:
        where, params = _scope_where(scope, include_session=include_session)
        if entity_id:
            where += " and lower(entity_id) = lower(%s)"
            params.append(entity_id)
        rows = self._fetch_all(f"select * from entity_profiles where {where} order by updated_at desc", params)
        return [_row_to_profile(row) for row in rows]

    def search_entity_profiles(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        *,
        include_session: bool = False,
    ) -> list[tuple[EntityProfile, dict[str, float]]]:
        where, params = _scope_where(scope, include_session=include_session, alias="p")
        query_text = query.strip()
        query_vector = _pgvector_literal(self.embedder.embed_text(query_text))
        rows = self._fetch_all(
            f"""
            with scored as (
              select
                p.*,
                ts_rank_cd(p.search_tsv, plainto_tsquery('simple', %s)) as bm25_score,
                coalesce(1.0 - (p.embedding_dense <=> cast(%s as vector)), 0.0) as semantic_score
              from entity_profiles p
              where {where}
            )
            select
              *,
              (0.45 * semantic_score + 0.40 * bm25_score + 0.10 * confidence + least(0.05, support_count * 0.01)) as score
            from scored
            where bm25_score > 0 or semantic_score > 0
            order by score desc, updated_at desc
            limit %s
            """,
            [query_text, query_vector, *params, limit],
        )
        out: list[tuple[EntityProfile, dict[str, float]]] = []
        for row in rows:
            bm25 = float(row.get("bm25_score") or 0.0)
            semantic = float(row.get("semantic_score") or 0.0)
            out.append(
                (
                    _row_to_profile(row),
                    {
                        "semantic_score": semantic,
                        "bm25_score": bm25,
                        "sparse_source": 1.0 if bm25 > 0 else 0.0,
                        "score": float(row.get("score") or 0.0),
                    },
                )
            )
        return out

    def upsert_entity(
        self,
        scope: Scope,
        name: str,
        *,
        entity_type: str = "unknown",
        source_span_ids: list[str] | None = None,
        aliases: list[str] | None = None,
        observed_at: datetime | None = None,
    ) -> EntityRecord:
        normalized = name.strip()
        entity_id = "entity_" + stable_hash(
            "|".join([scope.workspace_id or "", scope.user_id or "", scope.agent_id or "", scope.run_id or "", normalized.lower()])
        )[:24]
        now = datetime.now(timezone.utc)
        source_span_ids = source_span_ids or []
        aliases = aliases or []
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into entities
                (entity_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, name, entity_type,
                 aliases, source_span_ids, observed_count, last_observed_at, created_at, updated_at)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, 1, %s, %s, %s)
                on conflict (entity_id) do update set
                  aliases = (
                    select coalesce(jsonb_agg(value order by value), '[]'::jsonb)
                    from (
                      select distinct value
                      from jsonb_array_elements_text(entities.aliases || excluded.aliases) as merged(value)
                    ) as deduped
                  ),
                  source_span_ids = (
                    select coalesce(jsonb_agg(value order by value), '[]'::jsonb)
                    from (
                      select distinct value
                      from jsonb_array_elements_text(entities.source_span_ids || excluded.source_span_ids) as merged(value)
                    ) as deduped
                  ),
                  observed_count = entities.observed_count + 1,
                  last_observed_at = excluded.last_observed_at,
                  updated_at = excluded.updated_at
                """,
                (
                    entity_id,
                    scope.workspace_id,
                    scope.user_id,
                    scope.agent_id,
                    scope.run_id,
                    scope.session_id,
                    scope.app_id,
                    normalized,
                    entity_type,
                    _json_dumps(aliases),
                    _json_dumps(source_span_ids),
                    _dt_to_pg(observed_at or now),
                    _dt_to_pg(now),
                    _dt_to_pg(now),
                ),
            )
            self._commit_if_owner()
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()
        row = self._fetch_all("select * from entities where entity_id = %s limit 1", [entity_id])[0]
        return _row_to_entity(row)

    def list_entities(self, scope: Scope, *, include_session: bool = False) -> list[EntityRecord]:
        where, params = _scope_where(scope, include_session=include_session)
        rows = self._fetch_all(f"select * from entities where {where} order by observed_count desc, name", params)
        return [_row_to_entity(row) for row in rows]

    def search_entities(self, query: str, scope: Scope, limit: int = 20, *, include_session: bool = False) -> list[tuple[EntityRecord, dict[str, float]]]:
        scored: list[tuple[EntityRecord, dict[str, float]]] = []
        lower = query.lower()
        for entity in self.list_entities(scope, include_session=include_session):
            exact = 1.0 if entity.name.lower() in lower else 0.0
            alias = max((1.0 if alias.lower() in lower else 0.0 for alias in entity.aliases), default=0.0)
            lexical = keyword_score(query, entity.name + " " + " ".join(entity.aliases))
            score = max(exact, alias, lexical) + min(0.25, entity.observed_count * 0.03)
            if score > 0:
                scored.append((entity, {"entity_overlap": max(exact, alias, lexical), "score": score}))
        scored.sort(key=lambda item: item[1]["score"], reverse=True)
        return scored[:limit]

    def _fetch_all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self.connect().cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [_row_as_dict(row, cursor.description) for row in rows]
        finally:
            cursor.close()


class PostgresRuntimeRepository(_PostgresRepository):
    """Postgres CRUD boundary for encoding, utility examples, traces, audit, and background tasks."""

    def __init__(self, dsn: str, *, connect: Callable[[str], Any] | None = None) -> None:
        self.dsn = dsn
        self._connect = connect or _default_connect
        self.conn: Any | None = None

    def connect(self) -> Any:
        if self.conn is None:
            self.conn = self._connect(self.dsn)
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PostgresRuntimeRepository":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def insert_encoding_decision(self, scope: Scope, decision: EncodingDecision) -> bool:
        candidate = {
            "local_id": decision.candidate.local_id,
            "candidate_type": decision.candidate.candidate_type,
            "text": decision.candidate.text,
            "structured": decision.candidate.structured,
            "confidence": decision.candidate.confidence,
            "source_span_ids": decision.candidate.source_span_ids,
            "extractor_name": decision.candidate.extractor_name,
            "prompt_version": decision.candidate.prompt_version,
        }
        return self._execute_commit(
            """
            insert into encoding_decisions
            (decision_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, candidate_type,
             candidate_json, source_span_ids, decision, reason_codes, scores_json, matched_existing_ids)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            on conflict (decision_id) do nothing
            """,
            [
                decision.decision_id,
                scope.workspace_id,
                scope.user_id,
                scope.agent_id,
                scope.run_id,
                scope.session_id,
                scope.app_id,
                decision.candidate_type,
                _json_dumps(candidate),
                _json_dumps(decision.candidate.source_span_ids),
                decision.decision,
                _json_dumps(decision.reason_codes),
                _json_dumps(decision.scores),
                _json_dumps(decision.matched_existing_ids),
            ],
        )

    def list_encoding_decisions(self, scope: Scope, candidate_type: str | None = None) -> list[dict[str, Any]]:
        where, params = _scope_where(scope)
        if candidate_type:
            where += " and candidate_type = %s"
            params.append(candidate_type)
        rows = self._fetch_all(f"select * from encoding_decisions where {where} order by created_at", params)
        return [_row_to_encoding_decision_dict(row) for row in rows]

    def insert_utility_example(self, example: dict[str, Any]) -> bool:
        return self._execute_commit(
            """
            insert into retrieval_utility_examples
            (example_id, query_id, query_text, query_type, candidate_id, candidate_type, features_json,
             label, label_source, answer_correct)
            values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            on conflict (example_id) do nothing
            """,
            [
                example["example_id"],
                example.get("query_id"),
                example["query_text"],
                example.get("query_type"),
                example["candidate_id"],
                example["candidate_type"],
                _json_dumps(example["features"]),
                example["label"],
                example["label_source"],
                None if example.get("answer_correct") is None else bool(example["answer_correct"]),
            ],
        )

    def list_utility_examples(self, label: str | None = None) -> list[dict[str, Any]]:
        where = "1=1"
        params: list[Any] = []
        if label:
            where += " and label = %s"
            params.append(label)
        rows = self._fetch_all(f"select * from retrieval_utility_examples where {where} order by created_at", params)
        return [_row_to_utility_example(row) for row in rows]

    def save_trace(self, trace_id: str, trace: dict[str, Any], scope: Scope | None = None) -> bool:
        columns = _scope_columns(scope) if scope else {key: None for key in _scope_columns(Scope()).keys()}
        return self._execute_commit(
            """
            insert into debug_traces
            (trace_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, trace_json)
            values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            on conflict (trace_id) do update set
              workspace_id = excluded.workspace_id,
              user_id = excluded.user_id,
              agent_id = excluded.agent_id,
              run_id = excluded.run_id,
              session_id = excluded.session_id,
              app_id = excluded.app_id,
              trace_json = excluded.trace_json
            """,
            [
                trace_id,
                columns["workspace_id"],
                columns["user_id"],
                columns["agent_id"],
                columns["run_id"],
                columns["session_id"],
                columns["app_id"],
                _json_dumps(trace),
            ],
        )

    def get_trace(self, trace_id: str, scope: Scope | None = None, *, include_session: bool = False) -> dict[str, Any] | None:
        where = "trace_id = %s"
        params: list[Any] = [trace_id]
        if scope:
            scope_where, scope_params = _scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        rows = self._fetch_all(f"select trace_json from debug_traces where {where} limit 1", params)
        return _json_loads(rows[0]["trace_json"], None) if rows else None

    def insert_audit_event(
        self,
        scope: Scope,
        event_type: str,
        *,
        object_type: str | None = None,
        object_id: str | None = None,
        trace_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        audit_id = new_id("audit")
        self._execute_commit(
            """
            insert into audit_events
            (audit_id, workspace_id, user_id, agent_id, run_id, session_id, app_id,
             event_type, object_type, object_id, trace_id, payload_json)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            [
                audit_id,
                scope.workspace_id,
                scope.user_id,
                scope.agent_id,
                scope.run_id,
                scope.session_id,
                scope.app_id,
                event_type,
                object_type,
                object_id,
                trace_id,
                _json_dumps(payload or {}),
            ],
        )
        return audit_id

    def list_audit_events(self, scope: Scope, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        where, params = _scope_where(scope)
        if event_type:
            where += " and event_type = %s"
            params.append(event_type)
        rows = self._fetch_all(f"select * from audit_events where {where} order by created_at desc limit %s", [*params, limit])
        return [_row_to_audit_event(row) for row in rows]

    def enqueue_background_task(
        self,
        scope: Scope,
        task_type: str,
        *,
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        run_after: datetime | None = None,
    ) -> dict[str, Any]:
        if dedupe_key:
            existing = self._fetch_all("select * from background_tasks where dedupe_key = %s limit 1", [dedupe_key])
            if existing:
                return _row_to_background_task(existing[0])
        task_id = new_id("task")
        now = datetime.now(timezone.utc)
        self._execute_commit(
            """
            insert into background_tasks
            (task_id, task_type, workspace_id, user_id, agent_id, run_id, session_id, app_id,
             status, dedupe_key, payload_json, attempts, last_error, run_after, created_at, updated_at)
            values (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s::jsonb, 0, null, %s, %s, %s)
            """,
            [
                task_id,
                task_type,
                scope.workspace_id,
                scope.user_id,
                scope.agent_id,
                scope.run_id,
                scope.session_id,
                scope.app_id,
                dedupe_key,
                _json_dumps(payload or {}),
                _dt_to_pg(run_after or now),
                _dt_to_pg(now),
                _dt_to_pg(now),
            ],
        )
        return _row_to_background_task(self._fetch_all("select * from background_tasks where task_id = %s limit 1", [task_id])[0])

    def list_background_tasks(
        self,
        scope: Scope | None = None,
        *,
        status: str | None = None,
        limit: int = 100,
        include_session: bool = False,
    ) -> list[dict[str, Any]]:
        where = "1=1"
        params: list[Any] = []
        if scope:
            where, params = _scope_where(scope, include_session=include_session)
        if status:
            where += " and status = %s"
            params.append(status)
        rows = self._fetch_all(f"select * from background_tasks where {where} order by created_at desc limit %s", [*params, limit])
        return [_row_to_background_task(row) for row in rows]

    def next_background_tasks(self, *, limit: int = 10, scope: Scope | None = None, include_session: bool = False) -> list[dict[str, Any]]:
        where = "status = 'pending' and (run_after is null or run_after <= %s)"
        params: list[Any] = [_dt_to_pg(datetime.now(timezone.utc))]
        if scope:
            scope_where, scope_params = _scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        rows = self._fetch_all(f"select * from background_tasks where {where} order by run_after, created_at limit %s", [*params, limit])
        return [_row_to_background_task(row) for row in rows]

    def update_background_task(
        self,
        task_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        rows = self._fetch_all("select * from background_tasks where task_id = %s limit 1", [task_id])
        if not rows:
            return None
        row = rows[0]
        payload = _json_loads(row.get("payload_json"), {})
        if result is not None:
            payload["result"] = result
        attempts = int(row.get("attempts") or 0) + (1 if status == "running" and row.get("status") != "running" else 0)
        self._execute_commit(
            """
            update background_tasks
            set status = %s, payload_json = %s::jsonb, attempts = %s, last_error = %s, updated_at = %s
            where task_id = %s
            """,
            [status, _json_dumps(payload), attempts, error, _dt_to_pg(datetime.now(timezone.utc)), task_id],
        )
        updated = self._fetch_all("select * from background_tasks where task_id = %s limit 1", [task_id])
        return _row_to_background_task(updated[0]) if updated else None

    def _execute_commit(self, sql: str, params: list[Any]) -> bool:
        cursor = self.connect().cursor()
        try:
            cursor.execute(sql, params)
            changed = cursor.rowcount > 0
            self._commit_if_owner()
            return changed
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def _fetch_all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self.connect().cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [_row_as_dict(row, cursor.description) for row in rows]
        finally:
            cursor.close()


class PostgresMemoryStore(_PostgresRepository):
    """SQLiteMemoryStore-compatible facade over production Postgres repositories."""

    def __init__(
        self,
        dsn: str,
        *,
        connect: Callable[[str], Any] | None = None,
        embedder: Embedder | None = None,
        pool: PostgresConnectionPool | None = None,
        acquire_timeout_seconds: float = 5.0,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self.dsn = dsn
        self._connect = connect or _default_connect
        self.embedder = embedder or DeterministicEmbedder()
        self.pool = pool
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self.conn: Any | None = None
        self._owns_connection = True
        shared_connect = lambda dsn: self.connect()
        self.evidence = PostgresEvidenceRepository(dsn, connect=shared_connect, embedder=self.embedder)
        self.facts = PostgresFactRepository(dsn, connect=shared_connect, embedder=self.embedder)
        self.events = PostgresEventRepository(dsn, connect=shared_connect)
        self.views_profiles = PostgresViewProfileRepository(dsn, connect=shared_connect, embedder=self.embedder)
        self.runtime = PostgresRuntimeRepository(dsn, connect=shared_connect)

    @contextmanager
    def operation(self, *, user_id: str | None = None, write: bool = False) -> Iterator["PostgresMemoryStore"]:
        """Run one logical request in a single Postgres transaction."""
        if write and not (user_id and user_id.strip()):
            raise ValueError("write operations require an authenticated user_id")
        if self.pool is None:
            connection_context = nullcontext(self.connect())
        else:
            connection_context = self.pool.connection(self.acquire_timeout_seconds)
        with connection_context as connection:
            bound = self.for_connection(connection, manage_transaction=False)
            try:
                if write:
                    self._acquire_user_write_lock(connection, user_id)
                yield bound
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def for_connection(self, connection: Any, *, manage_transaction: bool = False) -> "PostgresMemoryStore":
        facade = PostgresMemoryStore(
            self.dsn,
            connect=lambda _dsn: connection,
            embedder=self.embedder,
            acquire_timeout_seconds=self.acquire_timeout_seconds,
            lock_timeout_seconds=self.lock_timeout_seconds,
        )
        facade.conn = connection
        facade._owns_connection = False
        facade._manage_transaction = manage_transaction
        for repository in [facade.evidence, facade.facts, facade.events, facade.views_profiles, facade.runtime]:
            repository.conn = connection
            repository._manage_transaction = manage_transaction
        return facade

    def _acquire_user_write_lock(self, connection: Any, user_id: str) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute("set local lock_timeout = %s", (f"{int(self.lock_timeout_seconds * 1000)}ms",))
            cursor.execute("select pg_advisory_xact_lock(hashtextextended(%s, 0))", (user_id,))
        except Exception as exc:
            if _is_lock_not_available(exc):
                raise RetryableOperationError("write for this user is busy", code="user_write_busy") from exc
            raise
        finally:
            cursor.close()

    def connect(self) -> Any:
        if self.conn is None:
            self.conn = self._connect(self.dsn)
        return self.conn

    def close(self) -> None:
        if self.conn is not None and self._owns_connection:
            self.conn.close()
        self.conn = None
        if self.pool is not None:
            self.pool.close()
        for repo in [self.evidence, self.facts, self.events, self.views_profiles, self.runtime]:
            repo.conn = None

    def __enter__(self) -> "PostgresMemoryStore":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def insert_span(self, span: EvidenceSpan) -> bool:
        return self.evidence.insert_span(span)

    def get_span(self, span_id: str, scope: Scope | None = None, *, include_session: bool = False) -> EvidenceSpan | None:
        return self.evidence.get_span(span_id, scope, include_session=include_session)

    def list_spans(self, scope: Scope, *, include_session: bool = False) -> list[EvidenceSpan]:
        return self.evidence.list_spans(scope, include_session=include_session)

    def search_spans(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        speaker: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[EvidenceSpan, dict[str, float]]]:
        return self.evidence.search_spans(query, scope, limit=limit, speaker=speaker, include_session=include_session)

    def find_duplicate_span(self, content_hash: str, scope: Scope) -> EvidenceSpan | None:
        return self.evidence.find_duplicate_span(content_hash, scope)

    def insert_fact(self, fact: MemoryFact) -> bool:
        return self.facts.insert_fact(fact)

    def get_fact(self, fact_id: str, scope: Scope | None = None, *, include_session: bool = False) -> MemoryFact | None:
        return self.facts.get_fact(fact_id, scope, include_session=include_session)

    def list_facts(self, scope: Scope, category: str | None = None, *, include_session: bool = False) -> list[MemoryFact]:
        return self.facts.list_facts(scope, category=category, include_session=include_session)

    def search_facts(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        category: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[MemoryFact, dict[str, float]]]:
        return self.facts.search_facts(query, scope, limit=limit, category=category, include_session=include_session)

    def insert_fact_relation(self, relation: FactRelation) -> bool:
        return self.facts.insert_fact_relation(relation)

    def list_fact_relations(self, fact_id: str | None = None, relation_type: str | None = None) -> list[FactRelation]:
        return self.facts.list_fact_relations(fact_id=fact_id, relation_type=relation_type)

    def superseded_fact_ids(self) -> set[str]:
        return self.facts.superseded_fact_ids()

    def insert_event(self, event: MemoryEvent) -> bool:
        return self.events.insert_event(event)

    def get_event(self, event_id: str, scope: Scope | None = None, *, include_session: bool = False) -> MemoryEvent | None:
        return self.events.get_event(event_id, scope, include_session=include_session)

    def list_events(self, scope: Scope, *, include_session: bool = False) -> list[MemoryEvent]:
        return self.events.list_events(scope, include_session=include_session)

    def search_events(self, query: str, scope: Scope, limit: int = 20, *, include_session: bool = False) -> list[tuple[MemoryEvent, dict[str, float]]]:
        return self.events.search_events(query, scope, limit=limit, include_session=include_session)

    def insert_event_edge(self, edge: EventEdge) -> bool:
        return self.events.insert_event_edge(edge)

    def has_event_edge(self, from_event_id: str, to_event_id: str, edge_type: str | None = None) -> bool:
        return self.events.has_event_edge(from_event_id, to_event_id, edge_type=edge_type)

    def get_event_edge(self, from_event_id: str, to_event_id: str) -> dict[str, Any] | None:
        return self.events.get_event_edge(from_event_id, to_event_id)

    def upsert_chronology_topic(self, topic: ChronologyTopic) -> None:
        values = {
            **_scope_columns(topic.scope),
            "topic_id": topic.topic_id,
            "canonical_label": topic.canonical_label,
            "aliases": _json_dumps(topic.aliases),
            "language": topic.language,
            "taxonomy_tags": _json_dumps(topic.taxonomy_tags),
            "source_span_ids": _json_dumps(topic.source_span_ids),
            "confidence": topic.confidence,
            "created_at": _dt_to_pg(topic.created_at),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join(["%s"] * len(values))
        updates = ", ".join(f"{key}=excluded.{key}" for key in values if key != "topic_id")
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                f"insert into chronology_topics ({columns}) values ({placeholders}) "
                f"on conflict(topic_id) do update set {updates}",
                list(values.values()),
            )
            self._commit_if_owner()
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def upsert_chronology_phase(self, phase: ChronologyPhase) -> None:
        values = {
            "phase_id": phase.phase_id,
            "topic_id": phase.topic_id,
            "phase_type": phase.phase_type,
            "order_hint": phase.order_hint,
            "source_span_ids": _json_dumps(phase.source_span_ids),
            "confidence": phase.confidence,
            "created_at": _dt_to_pg(phase.created_at),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join(["%s"] * len(values))
        updates = ", ".join(f"{key}=excluded.{key}" for key in values if key != "phase_id")
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                f"insert into chronology_phases ({columns}) values ({placeholders}) "
                f"on conflict(phase_id) do update set {updates}",
                list(values.values()),
            )
            self._commit_if_owner()
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def upsert_chronology_event_node(self, node: ChronologyEventNode) -> None:
        values = {
            **_scope_columns(node.scope),
            "node_id": node.node_id,
            "actor": node.actor,
            "action": node.action,
            "object": node.object,
            "topic_id": node.topic_id,
            "phase_id": node.phase_id,
            "timestamp": _dt_to_pg(node.timestamp),
            "source_span_id": node.source_span_id,
            "source_turn_id": node.source_turn_id,
            "text": node.text,
            "language": node.language,
            "confidence": node.confidence,
            "explicit_order_marker": node.explicit_order_marker,
            "created_at": _dt_to_pg(node.created_at),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join(["%s"] * len(values))
        updates = ", ".join(f"{key}=excluded.{key}" for key in values if key != "node_id")
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                f"insert into chronology_event_nodes ({columns}) values ({placeholders}) "
                f"on conflict(node_id) do update set {updates}",
                list(values.values()),
            )
            self._commit_if_owner()
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def insert_chronology_event_edge(self, edge: ChronologyEventEdge) -> bool:
        cursor = self.connect().cursor()
        try:
            cursor.execute(
                """
                insert into chronology_event_edges
                (edge_id, from_node_id, to_node_id, edge_type, evidence_type, source_span_ids, confidence, created_at)
                values (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                on conflict (from_node_id, to_node_id, edge_type, evidence_type) do nothing
                """,
                (
                    edge.edge_id,
                    edge.from_node_id,
                    edge.to_node_id,
                    edge.edge_type,
                    edge.evidence_type,
                    _json_dumps(edge.source_span_ids),
                    edge.confidence,
                    _dt_to_pg(edge.created_at),
                ),
            )
            inserted = cursor.rowcount > 0
            self._commit_if_owner()
            return inserted
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()

    def list_chronology_topics(self, scope: Scope, include_session: bool = False) -> list[ChronologyTopic]:
        where, params = _scope_where(scope, include_session=include_session)
        rows = self._fetch_chronology_rows(
            f"select * from chronology_topics where {where} order by confidence desc, created_at, canonical_label",
            params,
        )
        return [_row_to_chronology_topic(row) for row in rows]

    def list_chronology_phases(self, topic_ids: list[str]) -> list[ChronologyPhase]:
        if not topic_ids:
            return []
        placeholders = ", ".join(["%s"] * len(topic_ids))
        rows = self._fetch_chronology_rows(
            f"select * from chronology_phases where topic_id in ({placeholders}) order by order_hint, created_at, phase_id",
            topic_ids,
        )
        return [_row_to_chronology_phase(row) for row in rows]

    def list_chronology_event_nodes(
        self,
        scope: Scope,
        include_session: bool = False,
        topic_ids: list[str] | None = None,
    ) -> list[ChronologyEventNode]:
        where, params = _scope_where(scope, include_session=include_session)
        if topic_ids is not None:
            if not topic_ids:
                return []
            placeholders = ", ".join(["%s"] * len(topic_ids))
            where += f" and topic_id in ({placeholders})"
            params.extend(topic_ids)
        rows = self._fetch_chronology_rows(
            f"select * from chronology_event_nodes where {where} order by timestamp, created_at, node_id",
            params,
        )
        return [_row_to_chronology_event_node(row) for row in rows]

    def list_chronology_event_edges(self, node_ids: list[str]) -> list[ChronologyEventEdge]:
        if not node_ids:
            return []
        placeholders = ", ".join(["%s"] * len(node_ids))
        rows = self._fetch_chronology_rows(
            f"""
            select * from chronology_event_edges
            where from_node_id in ({placeholders}) or to_node_id in ({placeholders})
            order by created_at, edge_id
            """,
            [*node_ids, *node_ids],
        )
        return [_row_to_chronology_event_edge(row) for row in rows]

    def _fetch_chronology_rows(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self.connect().cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [_row_as_dict(row, cursor.description) for row in rows]
        finally:
            cursor.close()

    def upsert_current_view(self, view: CurrentView) -> bool:
        return self.views_profiles.upsert_current_view(view)

    def list_current_views(self, scope: Scope, view_type: str | None = None, *, include_session: bool = False) -> list[CurrentView]:
        return self.views_profiles.list_current_views(scope, view_type=view_type, include_session=include_session)

    def upsert_entity_profile(self, profile: EntityProfile) -> bool:
        return self.views_profiles.upsert_entity_profile(profile)

    def list_entity_profiles(self, scope: Scope, entity_id: str | None = None, *, include_session: bool = False) -> list[EntityProfile]:
        return self.views_profiles.list_entity_profiles(scope, entity_id=entity_id, include_session=include_session)

    def search_entity_profiles(self, query: str, scope: Scope, limit: int = 20, *, include_session: bool = False) -> list[tuple[EntityProfile, dict[str, float]]]:
        return self.views_profiles.search_entity_profiles(query, scope, limit=limit, include_session=include_session)

    def upsert_entity(
        self,
        scope: Scope,
        name: str,
        *,
        entity_type: str = "unknown",
        source_span_ids: list[str] | None = None,
        aliases: list[str] | None = None,
        observed_at: datetime | None = None,
    ) -> EntityRecord:
        return self.views_profiles.upsert_entity(
            scope,
            name,
            entity_type=entity_type,
            source_span_ids=source_span_ids,
            aliases=aliases,
            observed_at=observed_at,
        )

    def list_entities(self, scope: Scope, *, include_session: bool = False) -> list[EntityRecord]:
        return self.views_profiles.list_entities(scope, include_session=include_session)

    def search_entities(self, query: str, scope: Scope, limit: int = 20, *, include_session: bool = False) -> list[tuple[EntityRecord, dict[str, float]]]:
        return self.views_profiles.search_entities(query, scope, limit=limit, include_session=include_session)

    def insert_encoding_decision(self, scope: Scope, decision: EncodingDecision) -> bool:
        return self.runtime.insert_encoding_decision(scope, decision)

    def list_encoding_decisions(self, scope: Scope, candidate_type: str | None = None) -> list[dict[str, Any]]:
        return self.runtime.list_encoding_decisions(scope, candidate_type=candidate_type)

    def insert_utility_example(self, example: dict[str, Any]) -> bool:
        return self.runtime.insert_utility_example(example)

    def list_utility_examples(self, label: str | None = None) -> list[dict[str, Any]]:
        return self.runtime.list_utility_examples(label=label)

    def save_trace(self, trace_id: str, trace: dict[str, Any], scope: Scope | None = None) -> bool:
        return self.runtime.save_trace(trace_id, trace, scope)

    def get_trace(self, trace_id: str, scope: Scope | None = None, *, include_session: bool = False) -> dict[str, Any] | None:
        return self.runtime.get_trace(trace_id, scope, include_session=include_session)

    def clear_scope(self, scope: Scope, *, include_session: bool = False) -> dict[str, Any]:
        where, params = _scope_where(scope, include_session=include_session)
        conn = self.connect()
        cursor = conn.cursor()
        scoped_tables = [
            "evidence_spans",
            "memory_facts",
            "events",
            "current_views",
            "entity_profiles",
            "entities",
            "encoding_decisions",
            "debug_traces",
            "audit_events",
            "background_tasks",
        ]
        counts: dict[str, int] = {}
        try:
            fact_ids = self._select_ids(cursor, "memory_facts", "fact_id", where, params)
            event_ids = self._select_ids(cursor, "events", "event_id", where, params)
            chronology_topic_ids = self._select_ids(cursor, "chronology_topics", "topic_id", where, params)
            chronology_node_ids = self._select_ids(cursor, "chronology_event_nodes", "node_id", where, params)
            counts["fact_relations"] = self._delete_related_ids(
                cursor,
                "fact_relations",
                ("from_fact_id", "to_fact_id"),
                fact_ids,
            )
            counts["event_edges"] = self._delete_related_ids(
                cursor,
                "event_edges",
                ("from_event_id", "to_event_id"),
                event_ids,
            )
            counts["chronology_event_edges"] = self._delete_related_ids(
                cursor,
                "chronology_event_edges",
                ("from_node_id", "to_node_id"),
                chronology_node_ids,
            )
            counts["chronology_phases"] = self._delete_ids(
                cursor,
                "chronology_phases",
                "topic_id",
                chronology_topic_ids,
            )
            cursor.execute(f"delete from chronology_event_nodes where {where}", params)
            counts["chronology_event_nodes"] = cursor.rowcount if cursor.rowcount >= 0 else 0
            cursor.execute(f"delete from chronology_topics where {where}", params)
            counts["chronology_topics"] = cursor.rowcount if cursor.rowcount >= 0 else 0
            for table in scoped_tables:
                cursor.execute(f"delete from {table} where {where}", params)
                counts[table] = cursor.rowcount if cursor.rowcount >= 0 else 0
            self._commit_if_owner()
        except Exception:
            self._rollback_if_owner()
            raise
        finally:
            cursor.close()
        return {"deleted": counts, "include_session": include_session}

    def _select_ids(
        self,
        cursor: Any,
        table: str,
        id_column: str,
        where: str,
        params: list[Any],
    ) -> list[str]:
        cursor.execute(f"select {id_column} from {table} where {where}", params)
        return [row[0] for row in cursor.fetchall()]

    def _delete_ids(
        self,
        cursor: Any,
        table: str,
        id_column: str,
        object_ids: list[str],
    ) -> int:
        if not object_ids:
            return 0
        placeholders = ", ".join(["%s"] * len(object_ids))
        cursor.execute(f"delete from {table} where {id_column} in ({placeholders})", object_ids)
        return cursor.rowcount if cursor.rowcount >= 0 else 0

    def _delete_related_ids(
        self,
        cursor: Any,
        table: str,
        id_columns: tuple[str, str],
        object_ids: list[str],
    ) -> int:
        if not object_ids:
            return 0
        placeholders = ", ".join(["%s"] * len(object_ids))
        first, second = id_columns
        cursor.execute(
            f"delete from {table} where {first} in ({placeholders}) or {second} in ({placeholders})",
            [*object_ids, *object_ids],
        )
        return cursor.rowcount if cursor.rowcount >= 0 else 0

    def insert_audit_event(
        self,
        scope: Scope,
        event_type: str,
        *,
        object_type: str | None = None,
        object_id: str | None = None,
        trace_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        return self.runtime.insert_audit_event(scope, event_type, object_type=object_type, object_id=object_id, trace_id=trace_id, payload=payload)

    def list_audit_events(self, scope: Scope, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.runtime.list_audit_events(scope, event_type=event_type, limit=limit)

    def enqueue_background_task(
        self,
        scope: Scope,
        task_type: str,
        *,
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        run_after: datetime | None = None,
    ) -> dict[str, Any]:
        return self.runtime.enqueue_background_task(scope, task_type, payload=payload, dedupe_key=dedupe_key, run_after=run_after)

    def list_background_tasks(
        self,
        scope: Scope | None = None,
        *,
        status: str | None = None,
        limit: int = 100,
        include_session: bool = False,
    ) -> list[dict[str, Any]]:
        return self.runtime.list_background_tasks(scope, status=status, limit=limit, include_session=include_session)

    def next_background_tasks(self, *, limit: int = 10, scope: Scope | None = None, include_session: bool = False) -> list[dict[str, Any]]:
        return self.runtime.next_background_tasks(limit=limit, scope=scope, include_session=include_session)

    def update_background_task(
        self,
        task_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        return self.runtime.update_background_task(task_id, status=status, result=result, error=error)


def _execute_migration_query(cursor: Any, statement: str, params: tuple[str]) -> None:
    try:
        cursor.execute(statement, params)
    except TypeError:
        # Existing injected migration cursors only accept a single SQL argument.
        literal = params[0].replace("'", "''")
        cursor.execute(statement.replace("%s", f"'{literal}'"))


def _default_connect(dsn: str) -> Any:
    try:
        import psycopg2
    except ImportError as exc:
        raise PostgresBackendUnavailable(
            "Postgres migration requires psycopg2. Install with `pip install psycopg2-binary` "
            "or use the local SQLite backend."
        ) from exc
    return psycopg2.connect(dsn)


def _is_lock_not_available(error: Exception) -> bool:
    try:
        from psycopg2.errors import LockNotAvailable
    except ImportError:
        return False
    return isinstance(error, LockNotAvailable)


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if not in_single_quote and not in_double_quote and char == "-" and next_char == "-":
            while index < len(sql) and sql[index] != "\n":
                index += 1
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        if char == ";" and not in_single_quote and not in_double_quote:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(char)
        index += 1
    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def _scope_columns(scope: Scope) -> dict[str, str | None]:
    return {
        "workspace_id": scope.workspace_id,
        "user_id": scope.user_id,
        "agent_id": scope.agent_id,
        "run_id": scope.run_id,
        "session_id": scope.session_id,
        "app_id": scope.app_id,
    }


def _scope_where(scope: Scope, *, include_session: bool = False, alias: str | None = None) -> tuple[str, list[Any]]:
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in _scope_columns(scope).items():
        if key == "session_id" and not include_session:
            continue
        if value is not None:
            clauses.append(f"{prefix}{key} = %s")
            params.append(value)
    if not clauses:
        return "1=1", []
    return " and ".join(clauses), params


def _row_as_dict(row: Any, description: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    columns = [item[0] for item in description or []]
    return dict(zip(columns, row))


def _row_to_span(row: dict[str, Any]) -> EvidenceSpan:
    metadata = _json_loads(row.get("metadata"), {})
    if row.get("line_start") is not None and "line_start" not in metadata:
        metadata["line_start"] = row["line_start"]
    if row.get("line_end") is not None and "line_end" not in metadata:
        metadata["line_end"] = row["line_end"]
    return EvidenceSpan(
        span_id=row["span_id"],
        scope=Scope(
            workspace_id=row.get("workspace_id"),
            user_id=row.get("user_id"),
            agent_id=row.get("agent_id"),
            run_id=row.get("run_id"),
            session_id=row.get("session_id"),
            app_id=row.get("app_id"),
        ),
        turn_id=row.get("turn_id"),
        speaker=row["speaker"],
        span_type=row["span_type"],
        content=row["content"],
        content_hash=row["content_hash"],
        timestamp=_dt_from_pg(row.get("timestamp")) or datetime.now(timezone.utc),
        source_uri=row.get("source_uri"),
        parent_span_id=row.get("parent_span_id"),
        entities=_json_loads(row.get("entities"), []),
        topics=_json_loads(row.get("topics"), []),
        metadata=metadata,
    )


def _row_to_fact(row: dict[str, Any]) -> MemoryFact:
    return MemoryFact(
        fact_id=row["fact_id"],
        scope=_row_scope(row),
        subject=row.get("subject") or "",
        predicate=row.get("predicate") or "",
        object=row.get("object") or "",
        text=row["text"],
        category=row["category"],
        polarity=row.get("polarity") or "unknown",
        confidence=float(row["confidence"]),
        salience=float(row["salience"]),
        observed_at=_dt_from_pg(row.get("observed_at")),
        valid_from=_dt_from_pg(row.get("valid_from")),
        valid_to=_dt_from_pg(row.get("valid_to")),
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        linked_fact_ids=_json_loads(row.get("linked_fact_ids"), []),
        metadata=_json_loads(row.get("metadata"), {}),
        created_at=_dt_from_pg(row.get("created_at")) or datetime.now(timezone.utc),
    )


def _row_to_fact_relation(row: dict[str, Any]) -> FactRelation:
    return FactRelation(
        relation_id=row["relation_id"],
        from_fact_id=row["from_fact_id"],
        to_fact_id=row["to_fact_id"],
        relation_type=row["relation_type"],
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        confidence=float(row["confidence"]),
    )


def _row_to_event(row: dict[str, Any]) -> MemoryEvent:
    return MemoryEvent(
        event_id=row["event_id"],
        scope=_row_scope(row),
        event_type=row["event_type"],
        description=row["description"],
        participants=_json_loads(row.get("participants"), []),
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        fact_ids=_json_loads(row.get("fact_ids"), []),
        time_start=_dt_from_pg(row.get("time_start")),
        time_end=_dt_from_pg(row.get("time_end")),
        time_granularity=row.get("time_granularity") or "unknown",
        time_source=row.get("time_source") or "unknown",
        confidence=float(row.get("confidence") or 0.0),
    )


def _row_to_chronology_topic(row: dict[str, Any]) -> ChronologyTopic:
    return ChronologyTopic(
        topic_id=row["topic_id"],
        scope=_row_scope(row),
        canonical_label=row["canonical_label"],
        aliases=_json_loads(row.get("aliases"), []),
        language=row.get("language") or "unknown",
        taxonomy_tags=_json_loads(row.get("taxonomy_tags"), []),
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        confidence=float(row["confidence"]),
        created_at=_dt_from_pg(row.get("created_at")) or datetime.now(timezone.utc),
    )


def _row_to_chronology_phase(row: dict[str, Any]) -> ChronologyPhase:
    return ChronologyPhase(
        phase_id=row["phase_id"],
        topic_id=row["topic_id"],
        phase_type=row["phase_type"],
        order_hint=row.get("order_hint"),
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        confidence=float(row["confidence"]),
        created_at=_dt_from_pg(row.get("created_at")) or datetime.now(timezone.utc),
    )


def _row_to_chronology_event_node(row: dict[str, Any]) -> ChronologyEventNode:
    return ChronologyEventNode(
        node_id=row["node_id"],
        scope=_row_scope(row),
        actor=row["actor"],
        action=row["action"],
        object=row["object"],
        topic_id=row.get("topic_id"),
        phase_id=row.get("phase_id"),
        timestamp=_dt_from_pg(row.get("timestamp")),
        source_span_id=row.get("source_span_id"),
        source_turn_id=row.get("source_turn_id"),
        text=row["text"],
        language=row.get("language") or "unknown",
        confidence=float(row["confidence"]),
        explicit_order_marker=row.get("explicit_order_marker"),
        created_at=_dt_from_pg(row.get("created_at")) or datetime.now(timezone.utc),
    )


def _row_to_chronology_event_edge(row: dict[str, Any]) -> ChronologyEventEdge:
    return ChronologyEventEdge(
        edge_id=row["edge_id"],
        from_node_id=row["from_node_id"],
        to_node_id=row["to_node_id"],
        edge_type=row["edge_type"],
        evidence_type=row["evidence_type"],
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        confidence=float(row["confidence"]),
        created_at=_dt_from_pg(row.get("created_at")) or datetime.now(timezone.utc),
    )


def _row_to_view(row: dict[str, Any]) -> CurrentView:
    return CurrentView(
        view_id=row["view_id"],
        scope=_row_scope(row),
        view_type=row["view_type"],
        subject=row["subject"],
        text=row["text"],
        state_json=_json_loads(row.get("state_json"), {}),
        source_fact_ids=_json_loads(row.get("source_fact_ids"), []),
        source_event_ids=_json_loads(row.get("source_event_ids"), []),
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        confidence=float(row["confidence"]),
        updated_at=_dt_from_pg(row.get("updated_at")) or datetime.now(timezone.utc),
    )


def _row_to_profile(row: dict[str, Any]) -> EntityProfile:
    return EntityProfile(
        profile_id=row["profile_id"],
        scope=_row_scope(row),
        entity_id=row["entity_id"],
        entity_type=row["entity_type"],
        profile_type=row["profile_type"],
        text=row["text"],
        state_json=_json_loads(row.get("state_json"), {}),
        source_fact_ids=_json_loads(row.get("source_fact_ids"), []),
        source_event_ids=_json_loads(row.get("source_event_ids"), []),
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        confidence=float(row["confidence"]),
        support_count=int(row["support_count"]),
        last_observed_at=_dt_from_pg(row.get("last_observed_at")),
        updated_at=_dt_from_pg(row.get("updated_at")) or datetime.now(timezone.utc),
    )


def _row_to_entity(row: dict[str, Any]) -> EntityRecord:
    return EntityRecord(
        entity_id=row["entity_id"],
        scope=_row_scope(row),
        name=row["name"],
        entity_type=row["entity_type"],
        aliases=_json_loads(row.get("aliases"), []),
        source_span_ids=_json_loads(row.get("source_span_ids"), []),
        observed_count=int(row["observed_count"]),
        last_observed_at=_dt_from_pg(row.get("last_observed_at")),
    )


def _row_to_encoding_decision_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": row["decision_id"],
        "scope": _row_scope(row).__dict__,
        "candidate_type": row["candidate_type"],
        "candidate": _json_loads(row.get("candidate_json"), {}),
        "source_span_ids": _json_loads(row.get("source_span_ids"), []),
        "decision": row["decision"],
        "reason_codes": _json_loads(row.get("reason_codes"), []),
        "scores": _json_loads(row.get("scores_json"), {}),
        "matched_existing_ids": _json_loads(row.get("matched_existing_ids"), []),
        "created_at": row.get("created_at"),
    }


def _row_to_utility_example(row: dict[str, Any]) -> dict[str, Any]:
    answer_correct = row.get("answer_correct")
    return {
        "example_id": row["example_id"],
        "query_id": row.get("query_id"),
        "query_text": row["query_text"],
        "query_type": row.get("query_type"),
        "candidate_id": row["candidate_id"],
        "candidate_type": row["candidate_type"],
        "features": _json_loads(row.get("features_json"), {}),
        "label": row["label"],
        "label_source": row["label_source"],
        "answer_correct": None if answer_correct is None else bool(answer_correct),
        "created_at": row.get("created_at"),
    }


def _row_to_audit_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": row["audit_id"],
        "scope": _row_scope(row).__dict__,
        "event_type": row["event_type"],
        "object_type": row.get("object_type"),
        "object_id": row.get("object_id"),
        "trace_id": row.get("trace_id"),
        "payload": _json_loads(row.get("payload_json"), {}),
        "created_at": row.get("created_at"),
    }


def _row_to_background_task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "scope": _row_scope(row).__dict__,
        "status": row["status"],
        "dedupe_key": row.get("dedupe_key"),
        "payload": _json_loads(row.get("payload_json"), {}),
        "attempts": int(row.get("attempts") or 0),
        "last_error": row.get("last_error"),
        "run_after": row.get("run_after"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _row_scope(row: dict[str, Any]) -> Scope:
    return Scope(
        workspace_id=row.get("workspace_id"),
        user_id=row.get("user_id"),
        agent_id=row.get("agent_id"),
        run_id=row.get("run_id"),
        session_id=row.get("session_id"),
        app_id=row.get("app_id"),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _dt_to_pg(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _dt_from_pg(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"
