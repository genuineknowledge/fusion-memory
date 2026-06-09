from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fusion_memory.core.embedding import DeterministicEmbedder, Embedder, cosine_dense
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
from fusion_memory.core.text import keyword_score, stable_hash, tokenize


def dt_to_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def dt_from_str(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


class SQLiteMemoryStore:
    def __init__(self, path: str | Path = ":memory:", embedder: Embedder | None = None) -> None:
        self.path = str(path)
        self.embedder = embedder or DeterministicEmbedder()
        self.conn = sqlite3.connect(self.path)
        self._closed = False
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("pragma foreign_keys = on")
        self.fts_enabled = False
        self.migrate()

    def close(self) -> None:
        if not self._closed:
            self.conn.close()
            self._closed = True

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def migrate(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            create table if not exists evidence_spans (
              span_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              turn_id text,
              speaker text not null,
              span_type text not null,
              content text not null,
              content_hash text not null,
              timestamp text,
              source_uri text,
              parent_span_id text,
              entities text not null default '[]',
              topics text not null default '[]',
              embedding_dense text not null default '[]',
              metadata text not null default '{}',
              created_at text not null
            );
            create index if not exists evidence_scope_idx on evidence_spans(workspace_id, user_id, agent_id, run_id, session_id);
            create index if not exists evidence_hash_idx on evidence_spans(content_hash);

            create table if not exists memory_facts (
              fact_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              subject text,
              predicate text,
              object text,
              text text not null,
              category text not null,
              polarity text not null default 'unknown',
              confidence real not null,
              salience real not null,
              observed_at text,
              valid_from text,
              valid_to text,
              source_span_ids text not null,
              linked_fact_ids text not null default '[]',
              embedding_dense text not null default '[]',
              hash text,
              metadata text not null default '{}',
              created_at text not null
            );
            create index if not exists facts_scope_idx on memory_facts(workspace_id, user_id, agent_id, run_id, session_id);
            create index if not exists facts_category_idx on memory_facts(category);

            create table if not exists fact_relations (
              relation_id text primary key,
              from_fact_id text not null,
              to_fact_id text not null,
              relation_type text not null,
              source_span_ids text not null default '[]',
              confidence real not null,
              created_at text not null
            );
            create index if not exists fact_rel_from_idx on fact_relations(from_fact_id);
            create index if not exists fact_rel_to_idx on fact_relations(to_fact_id);

            create table if not exists events (
              event_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              event_type text not null,
              participants text not null default '[]',
              description text not null,
              time_start text,
              time_end text,
              time_granularity text,
              time_source text,
              source_span_ids text not null default '[]',
              fact_ids text not null default '[]',
              confidence real not null,
              created_at text not null
            );
            create index if not exists events_scope_idx on events(workspace_id, user_id, agent_id, run_id, session_id);
            create index if not exists events_time_idx on events(time_start);

            create table if not exists event_edges (
              edge_id text primary key,
              from_event_id text not null,
              to_event_id text not null,
              edge_type text not null,
              source_span_ids text not null default '[]',
              confidence real not null,
              created_at text not null
            );

            create table if not exists current_views (
              view_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              view_type text not null,
              subject text not null,
              text text not null,
              state_json text not null default '{}',
              source_fact_ids text not null default '[]',
              source_event_ids text not null default '[]',
              source_span_ids text not null default '[]',
              confidence real not null,
              updated_at text not null,
              expires_at text
            );
            create index if not exists current_views_scope_idx on current_views(workspace_id, user_id, agent_id, run_id, session_id);

            create table if not exists entity_profiles (
              profile_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              entity_id text not null,
              entity_type text not null,
              profile_type text not null,
              text text not null,
              state_json text not null default '{}',
              source_fact_ids text not null default '[]',
              source_event_ids text not null default '[]',
              source_span_ids text not null default '[]',
              confidence real not null,
              support_count integer not null,
              last_observed_at text,
              updated_at text not null,
              expires_at text,
              embedding_dense text not null default '[]'
            );
            create index if not exists profiles_scope_idx on entity_profiles(workspace_id, user_id, agent_id, run_id, session_id);
            create index if not exists profiles_entity_idx on entity_profiles(entity_id, profile_type);

            create table if not exists entities (
              entity_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              name text not null,
              entity_type text not null,
              aliases text not null default '[]',
              source_span_ids text not null default '[]',
              observed_count integer not null default 1,
              last_observed_at text,
              created_at text not null,
              updated_at text not null
            );
            create index if not exists entities_scope_idx on entities(workspace_id, user_id, agent_id, run_id, session_id);
            create index if not exists entities_name_idx on entities(name);

            create table if not exists encoding_decisions (
              decision_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              candidate_type text not null,
              candidate_json text not null,
              source_span_ids text not null default '[]',
              decision text not null,
              reason_codes text not null default '[]',
              scores_json text not null default '{}',
              matched_existing_ids text not null default '[]',
              created_at text not null
            );

            create table if not exists retrieval_utility_examples (
              example_id text primary key,
              query_id text,
              query_text text not null,
              query_type text,
              candidate_id text not null,
              candidate_type text not null,
              features_json text not null,
              label text not null,
              label_source text not null,
              answer_correct integer,
              created_at text not null
            );

            create table if not exists debug_traces (
              trace_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              trace_json text not null,
              created_at text not null
            );
            create index if not exists debug_traces_scope_idx on debug_traces(workspace_id, user_id, agent_id, run_id, session_id);

            create table if not exists audit_events (
              audit_id text primary key,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              event_type text not null,
              object_type text,
              object_id text,
              trace_id text,
              payload_json text not null default '{}',
              created_at text not null
            );
            create index if not exists audit_scope_idx on audit_events(workspace_id, user_id, agent_id, run_id, session_id);
            create index if not exists audit_trace_idx on audit_events(trace_id);

            create table if not exists background_tasks (
              task_id text primary key,
              task_type text not null,
              workspace_id text,
              user_id text,
              agent_id text,
              run_id text,
              session_id text,
              app_id text,
              status text not null,
              dedupe_key text,
              payload_json text not null default '{}',
              attempts integer not null default 0,
              last_error text,
              run_after text,
              created_at text not null,
              updated_at text not null
            );
            create unique index if not exists background_tasks_dedupe_idx on background_tasks(dedupe_key) where dedupe_key is not null;
            create index if not exists background_tasks_status_idx on background_tasks(status, run_after, created_at);
            create index if not exists background_tasks_scope_idx on background_tasks(workspace_id, user_id, agent_id, run_id, session_id);
            """
        )
        self.conn.commit()
        self._migrate_columns()
        self._migrate_fts()

    def _migrate_columns(self) -> None:
        self._ensure_columns(
            "debug_traces",
            {
                "workspace_id": "text",
                "user_id": "text",
                "agent_id": "text",
                "run_id": "text",
                "session_id": "text",
                "app_id": "text",
            },
        )
        self.conn.execute(
            "create index if not exists debug_traces_scope_idx on debug_traces(workspace_id, user_id, agent_id, run_id, session_id)"
        )
        self.conn.commit()

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        existing = {row["name"] for row in self.conn.execute(f"pragma table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                self.conn.execute(f"alter table {table} add column {name} {definition}")

    def _migrate_fts(self) -> None:
        try:
            self.conn.executescript(
                """
                create virtual table if not exists fts_evidence using fts5(span_id unindexed, content);
                create virtual table if not exists fts_facts using fts5(fact_id unindexed, text);
                create virtual table if not exists fts_events using fts5(event_id unindexed, description);
                create virtual table if not exists fts_profiles using fts5(profile_id unindexed, text);
                """
            )
            self.conn.commit()
            self.fts_enabled = True
            self._backfill_fts()
        except sqlite3.Error:
            self.fts_enabled = False

    def _backfill_fts(self) -> None:
        if not self.fts_enabled:
            return
        self.conn.execute(
            """
            insert into fts_evidence(rowid, span_id, content)
            select rowid, span_id, content from evidence_spans
            where rowid not in (select rowid from fts_evidence)
            """
        )
        self.conn.execute(
            """
            insert into fts_facts(rowid, fact_id, text)
            select rowid, fact_id, text from memory_facts
            where rowid not in (select rowid from fts_facts)
            """
        )
        self.conn.execute(
            """
            insert into fts_events(rowid, event_id, description)
            select rowid, event_id, description from events
            where rowid not in (select rowid from fts_events)
            """
        )
        self.conn.execute(
            """
            insert into fts_profiles(rowid, profile_id, text)
            select rowid, profile_id, text from entity_profiles
            where rowid not in (select rowid from fts_profiles)
            """
        )
        self.conn.commit()

    def _scope_columns(self, scope: Scope) -> dict[str, str | None]:
        return {
            "workspace_id": scope.workspace_id,
            "user_id": scope.user_id,
            "agent_id": scope.agent_id,
            "run_id": scope.run_id,
            "session_id": scope.session_id,
            "app_id": scope.app_id,
        }

    def _scope_where(self, scope: Scope, *, include_session: bool = False, alias: str | None = None) -> tuple[str, list[Any]]:
        prefix = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in self._scope_columns(scope).items():
            if key == "session_id" and not include_session:
                continue
            if value is not None:
                clauses.append(f"{prefix}{key} = ?")
                params.append(value)
        if not clauses:
            return "1=1", []
        return " and ".join(clauses), params

    def insert_span(self, span: EvidenceSpan) -> bool:
        embedding = self.embedder.embed_text(span.content)
        values = {
            **self._scope_columns(span.scope),
            "span_id": span.span_id,
            "turn_id": span.turn_id,
            "speaker": span.speaker,
            "span_type": span.span_type,
            "content": span.content,
            "content_hash": span.content_hash,
            "timestamp": dt_to_str(span.timestamp),
            "source_uri": span.source_uri,
            "parent_span_id": span.parent_span_id,
            "entities": dumps(span.entities),
            "topics": dumps(span.topics),
            "embedding_dense": dumps(embedding),
            "metadata": dumps(span.metadata),
            "created_at": dt_to_str(datetime.now(timezone.utc)),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join(["?"] * len(values))
        cur = self.conn.execute(
            f"insert or ignore into evidence_spans ({columns}) values ({placeholders})",
            list(values.values()),
        )
        if cur.rowcount > 0:
            self._upsert_fts("fts_evidence", "span_id", span.span_id, span.content)
        self.conn.commit()
        return cur.rowcount > 0

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
            "|".join(
                [
                    scope.workspace_id or "",
                    scope.user_id or "",
                    scope.agent_id or "",
                    scope.run_id or "",
                    normalized.lower(),
                ]
            )
        )[:24]
        existing = self.conn.execute("select * from entities where entity_id = ?", (entity_id,)).fetchone()
        now = datetime.now(timezone.utc)
        source_span_ids = source_span_ids or []
        aliases = aliases or []
        if existing:
            merged_sources = list(dict.fromkeys(loads(existing["source_span_ids"], []) + source_span_ids))
            merged_aliases = list(dict.fromkeys(loads(existing["aliases"], []) + aliases))
            observed_count = int(existing["observed_count"]) + 1
            self.conn.execute(
                """
                update entities set
                  aliases = ?,
                  source_span_ids = ?,
                  observed_count = ?,
                  last_observed_at = ?,
                  updated_at = ?
                where entity_id = ?
                """,
                (
                    dumps(merged_aliases),
                    dumps(merged_sources),
                    observed_count,
                    dt_to_str(observed_at or now),
                    dt_to_str(now),
                    entity_id,
                ),
            )
        else:
            values = {
                **self._scope_columns(scope),
                "entity_id": entity_id,
                "name": normalized,
                "entity_type": entity_type,
                "aliases": dumps(aliases),
                "source_span_ids": dumps(source_span_ids),
                "observed_count": 1,
                "last_observed_at": dt_to_str(observed_at or now),
                "created_at": dt_to_str(now),
                "updated_at": dt_to_str(now),
            }
            columns = ", ".join(values.keys())
            placeholders = ", ".join(["?"] * len(values))
            self.conn.execute(f"insert into entities ({columns}) values ({placeholders})", list(values.values()))
        self.conn.commit()
        row = self.conn.execute("select * from entities where entity_id = ?", (entity_id,)).fetchone()
        return self._row_to_entity(row)

    def list_entities(self, scope: Scope, *, include_session: bool = False) -> list[EntityRecord]:
        where, params = self._scope_where(scope, include_session=include_session)
        rows = self.conn.execute(f"select * from entities where {where} order by observed_count desc, name", params).fetchall()
        return [self._row_to_entity(row) for row in rows]

    def search_entities(self, query: str, scope: Scope, limit: int = 20, *, include_session: bool = False) -> list[tuple[EntityRecord, dict[str, float]]]:
        entities = self.list_entities(scope, include_session=include_session)
        scored: list[tuple[EntityRecord, dict[str, float]]] = []
        lower = query.lower()
        for entity in entities:
            exact = 1.0 if entity.name.lower() in lower else 0.0
            alias = max((1.0 if alias.lower() in lower else 0.0 for alias in entity.aliases), default=0.0)
            lexical = keyword_score(query, entity.name + " " + " ".join(entity.aliases))
            score = max(exact, alias, lexical) + min(0.25, entity.observed_count * 0.03)
            if score > 0:
                scored.append((entity, {"entity_overlap": max(exact, alias, lexical), "score": score}))
        scored.sort(key=lambda item: item[1]["score"], reverse=True)
        return scored[:limit]

    def get_span(self, span_id: str, scope: Scope | None = None, *, include_session: bool = False) -> EvidenceSpan | None:
        where = "span_id = ?"
        params: list[Any] = [span_id]
        if scope:
            scope_where, scope_params = self._scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        row = self.conn.execute(f"select * from evidence_spans where {where}", params).fetchone()
        return self._row_to_span(row) if row else None

    def list_spans(self, scope: Scope, *, include_session: bool = False) -> list[EvidenceSpan]:
        where, params = self._scope_where(scope, include_session=include_session)
        rows = self.conn.execute(f"select * from evidence_spans where {where} order by timestamp, created_at", params).fetchall()
        return [self._row_to_span(row) for row in rows]

    def search_spans(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        speaker: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[EvidenceSpan, dict[str, float]]]:
        where, params = self._scope_where(scope, include_session=include_session)
        if speaker:
            where += " and speaker = ?"
            params.append(speaker)
        rows = self.conn.execute(f"select * from evidence_spans where {where}", params).fetchall()
        fts_scores = self._fts_scores("fts_evidence", "span_id", query, limit=max(limit * 4, 20))
        qvec = self.embedder.embed_text(query)
        scored: list[tuple[EvidenceSpan, dict[str, float]]] = []
        for row in rows:
            span = self._row_to_span(row)
            dense = cosine_dense(qvec, loads(row["embedding_dense"], []))
            fts_score = fts_scores.get(span.span_id)
            bm25 = fts_score if fts_score is not None else keyword_score(query, span.content)
            score = 0.55 * dense + 0.45 * bm25
            if score > 0 or any(token in span.content.lower() for token in tokenize(query)):
                scored.append(
                    (
                        span,
                        {
                            "semantic_score": dense,
                            "bm25_score": bm25,
                            "sparse_source": 1.0 if fts_score is not None else 0.0,
                            "score": score,
                        },
                    )
                )
        scored.sort(key=lambda item: item[1]["score"], reverse=True)
        return scored[:limit]

    def find_duplicate_span(self, content_hash: str, scope: Scope) -> EvidenceSpan | None:
        where, params = self._scope_where(scope)
        row = self.conn.execute(
            f"select * from evidence_spans where {where} and content_hash = ? limit 1",
            [*params, content_hash],
        ).fetchone()
        return self._row_to_span(row) if row else None

    def insert_fact(self, fact: MemoryFact) -> None:
        embedding = self.embedder.embed_text(fact.text)
        values = {
            **self._scope_columns(fact.scope),
            "fact_id": fact.fact_id,
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "text": fact.text,
            "category": fact.category,
            "polarity": fact.polarity,
            "confidence": fact.confidence,
            "salience": fact.salience,
            "observed_at": dt_to_str(fact.observed_at),
            "valid_from": dt_to_str(fact.valid_from),
            "valid_to": dt_to_str(fact.valid_to),
            "source_span_ids": dumps(fact.source_span_ids),
            "linked_fact_ids": dumps(fact.linked_fact_ids),
            "embedding_dense": dumps(embedding),
            "hash": fact.metadata.get("hash"),
            "metadata": dumps(fact.metadata),
            "created_at": dt_to_str(fact.created_at),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join(["?"] * len(values))
        cur = self.conn.execute(f"insert into memory_facts ({columns}) values ({placeholders})", list(values.values()))
        self._upsert_fts("fts_facts", "fact_id", fact.fact_id, fact.text, rowid=cur.lastrowid)
        self.conn.commit()

    def get_fact(self, fact_id: str, scope: Scope | None = None, *, include_session: bool = False) -> MemoryFact | None:
        where = "fact_id = ?"
        params: list[Any] = [fact_id]
        if scope:
            scope_where, scope_params = self._scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        row = self.conn.execute(f"select * from memory_facts where {where}", params).fetchone()
        return self._row_to_fact(row) if row else None

    def list_facts(self, scope: Scope, category: str | None = None, *, include_session: bool = False) -> list[MemoryFact]:
        where, params = self._scope_where(scope, include_session=include_session)
        if category:
            where += " and category = ?"
            params.append(category)
        rows = self.conn.execute(f"select * from memory_facts where {where} order by created_at", params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def search_facts(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        category: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[MemoryFact, dict[str, float]]]:
        facts = self.list_facts(scope, category=category, include_session=include_session)
        qvec = self.embedder.embed_text(query)
        fts_scores = self._fts_scores("fts_facts", "fact_id", query, limit=max(limit * 4, 20))
        scored: list[tuple[MemoryFact, dict[str, float]]] = []
        superseded = self.superseded_fact_ids()
        for fact in facts:
            dense = cosine_dense(qvec, self.fact_embedding(fact.fact_id))
            fts_score = fts_scores.get(fact.fact_id)
            bm25 = fts_score if fts_score is not None else keyword_score(query, fact.text)
            active_prior = -0.15 if fact.fact_id in superseded else 0.0
            score = 0.50 * dense + 0.35 * bm25 + 0.10 * fact.confidence + 0.05 * fact.salience + active_prior
            if score > 0:
                scored.append((fact, {"semantic_score": dense, "bm25_score": bm25, "sparse_source": 1.0 if fts_score is not None else 0.0, "score": score}))
        scored.sort(key=lambda item: item[1]["score"], reverse=True)
        return scored[:limit]

    def fact_embedding(self, fact_id: str) -> list[float]:
        row = self.conn.execute("select embedding_dense from memory_facts where fact_id = ?", (fact_id,)).fetchone()
        return loads(row["embedding_dense"], []) if row else []

    def insert_fact_relation(self, relation: FactRelation) -> None:
        self.conn.execute(
            """
            insert into fact_relations
            (relation_id, from_fact_id, to_fact_id, relation_type, source_span_ids, confidence, created_at)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relation.relation_id,
                relation.from_fact_id,
                relation.to_fact_id,
                relation.relation_type,
                dumps(relation.source_span_ids),
                relation.confidence,
                dt_to_str(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()

    def list_fact_relations(self, fact_id: str | None = None, relation_type: str | None = None) -> list[FactRelation]:
        clauses: list[str] = []
        params: list[Any] = []
        if fact_id:
            clauses.append("(from_fact_id = ? or to_fact_id = ?)")
            params.extend([fact_id, fact_id])
        if relation_type:
            clauses.append("relation_type = ?")
            params.append(relation_type)
        where = " and ".join(clauses) if clauses else "1=1"
        rows = self.conn.execute(f"select * from fact_relations where {where}", params).fetchall()
        return [self._row_to_fact_relation(row) for row in rows]

    def superseded_fact_ids(self) -> set[str]:
        rows = self.conn.execute("select to_fact_id from fact_relations where relation_type = 'supersedes'").fetchall()
        return {row["to_fact_id"] for row in rows}

    def insert_event(self, event: MemoryEvent) -> None:
        values = {
            **self._scope_columns(event.scope),
            "event_id": event.event_id,
            "event_type": event.event_type,
            "participants": dumps(event.participants),
            "description": event.description,
            "time_start": dt_to_str(event.time_start),
            "time_end": dt_to_str(event.time_end),
            "time_granularity": event.time_granularity,
            "time_source": event.time_source,
            "source_span_ids": dumps(event.source_span_ids),
            "fact_ids": dumps(event.fact_ids),
            "confidence": event.confidence,
            "created_at": dt_to_str(datetime.now(timezone.utc)),
        }
        columns = ", ".join(values.keys())
        placeholders = ", ".join(["?"] * len(values))
        cur = self.conn.execute(f"insert into events ({columns}) values ({placeholders})", list(values.values()))
        self._upsert_fts("fts_events", "event_id", event.event_id, event.description, rowid=cur.lastrowid)
        self.conn.commit()

    def search_events(self, query: str, scope: Scope, limit: int = 20, *, include_session: bool = False) -> list[tuple[MemoryEvent, dict[str, float]]]:
        where, params = self._scope_where(scope, include_session=include_session)
        rows = self.conn.execute(f"select * from events where {where}", params).fetchall()
        fts_scores = self._fts_scores("fts_events", "event_id", query, limit=max(limit * 4, 20))
        scored: list[tuple[MemoryEvent, dict[str, float]]] = []
        for row in rows:
            event = self._row_to_event(row)
            fts_score = fts_scores.get(event.event_id)
            bm25 = fts_score if fts_score is not None else keyword_score(query, event.description + " " + " ".join(event.participants))
            temporal = 0.2 if event.time_start else 0.0
            score = 0.75 * bm25 + temporal + 0.05 * event.confidence
            if score > 0:
                scored.append((event, {"bm25_score": bm25, "sparse_source": 1.0 if fts_score is not None else 0.0, "temporal_fit": temporal, "score": score}))
        scored.sort(key=lambda item: item[1]["score"], reverse=True)
        return scored[:limit]

    def list_events(self, scope: Scope, *, include_session: bool = False) -> list[MemoryEvent]:
        where, params = self._scope_where(scope, include_session=include_session)
        rows = self.conn.execute(f"select * from events where {where} order by time_start, created_at", params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: str, scope: Scope | None = None, *, include_session: bool = False) -> MemoryEvent | None:
        where = "event_id = ?"
        params: list[Any] = [event_id]
        if scope:
            scope_where, scope_params = self._scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        row = self.conn.execute(f"select * from events where {where}", params).fetchone()
        return self._row_to_event(row) if row else None

    def insert_event_edge(self, edge: EventEdge) -> None:
        self.conn.execute(
            """
            insert into event_edges
            (edge_id, from_event_id, to_event_id, edge_type, source_span_ids, confidence, created_at)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.from_event_id,
                edge.to_event_id,
                edge.edge_type,
                dumps(edge.source_span_ids),
                edge.confidence,
                dt_to_str(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()

    def has_event_edge(self, from_event_id: str, to_event_id: str, edge_type: str | None = None) -> bool:
        where = "from_event_id = ? and to_event_id = ?"
        params: list[Any] = [from_event_id, to_event_id]
        if edge_type:
            where += " and edge_type = ?"
            params.append(edge_type)
        row = self.conn.execute(f"select 1 from event_edges where {where} limit 1", params).fetchone()
        return row is not None

    def get_event_edge(self, from_event_id: str, to_event_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            select * from event_edges
            where from_event_id = ? and to_event_id = ?
            order by confidence desc
            limit 1
            """,
            (from_event_id, to_event_id),
        ).fetchone()
        if not row:
            return None
        return {
            "edge_id": row["edge_id"],
            "edge_type": row["edge_type"],
            "source_span_ids": loads(row["source_span_ids"], []),
            "confidence": row["confidence"],
        }

    def upsert_current_view(self, view: CurrentView) -> None:
        self.conn.execute(
            """
            insert into current_views
            (view_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, view_type, subject, text,
             state_json, source_fact_ids, source_event_ids, source_span_ids, confidence, updated_at, expires_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, null)
            on conflict(view_id) do update set
              text=excluded.text,
              state_json=excluded.state_json,
              source_fact_ids=excluded.source_fact_ids,
              source_event_ids=excluded.source_event_ids,
              source_span_ids=excluded.source_span_ids,
              confidence=excluded.confidence,
              updated_at=excluded.updated_at
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
                dumps(view.state_json),
                dumps(view.source_fact_ids),
                dumps(view.source_event_ids),
                dumps(view.source_span_ids),
                view.confidence,
                dt_to_str(view.updated_at),
            ),
        )
        self.conn.commit()

    def list_current_views(self, scope: Scope, view_type: str | None = None, *, include_session: bool = False) -> list[CurrentView]:
        where, params = self._scope_where(scope, include_session=include_session)
        if view_type:
            where += " and view_type = ?"
            params.append(view_type)
        rows = self.conn.execute(f"select * from current_views where {where}", params).fetchall()
        return [self._row_to_view(row) for row in rows]

    def upsert_entity_profile(self, profile: EntityProfile) -> None:
        embedding = self.embedder.embed_text(profile.text)
        cur = self.conn.execute(
            """
            insert into entity_profiles
            (profile_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, entity_id, entity_type,
             profile_type, text, state_json, source_fact_ids, source_event_ids, source_span_ids, confidence,
             support_count, last_observed_at, updated_at, expires_at, embedding_dense)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, null, ?)
            on conflict(profile_id) do update set
              text=excluded.text,
              state_json=excluded.state_json,
              source_fact_ids=excluded.source_fact_ids,
              source_event_ids=excluded.source_event_ids,
              source_span_ids=excluded.source_span_ids,
              confidence=excluded.confidence,
              support_count=excluded.support_count,
              last_observed_at=excluded.last_observed_at,
              updated_at=excluded.updated_at,
              embedding_dense=excluded.embedding_dense
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
                dumps(profile.state_json),
                dumps(profile.source_fact_ids),
                dumps(profile.source_event_ids),
                dumps(profile.source_span_ids),
                profile.confidence,
                profile.support_count,
                dt_to_str(profile.last_observed_at),
                dt_to_str(profile.updated_at),
                dumps(embedding),
            ),
        )
        row = self.conn.execute("select rowid from entity_profiles where profile_id = ?", (profile.profile_id,)).fetchone()
        self._upsert_fts("fts_profiles", "profile_id", profile.profile_id, profile.text, rowid=row["rowid"] if row else cur.lastrowid)
        self.conn.commit()

    def list_entity_profiles(self, scope: Scope, entity_id: str | None = None, *, include_session: bool = False) -> list[EntityProfile]:
        where, params = self._scope_where(scope, include_session=include_session)
        if entity_id:
            where += " and lower(entity_id) = lower(?)"
            params.append(entity_id)
        rows = self.conn.execute(f"select * from entity_profiles where {where}", params).fetchall()
        return [self._row_to_profile(row) for row in rows]

    def search_entity_profiles(self, query: str, scope: Scope, limit: int = 20, *, include_session: bool = False) -> list[tuple[EntityProfile, dict[str, float]]]:
        profiles = self.list_entity_profiles(scope, include_session=include_session)
        qvec = self.embedder.embed_text(query)
        fts_scores = self._fts_scores("fts_profiles", "profile_id", query, limit=max(limit * 4, 20))
        scored: list[tuple[EntityProfile, dict[str, float]]] = []
        for profile in profiles:
            row = self.conn.execute("select embedding_dense from entity_profiles where profile_id = ?", (profile.profile_id,)).fetchone()
            dense = cosine_dense(qvec, loads(row["embedding_dense"], []) if row else [])
            fts_score = fts_scores.get(profile.profile_id)
            bm25 = fts_score if fts_score is not None else keyword_score(query, profile.text)
            score = 0.45 * dense + 0.40 * bm25 + 0.10 * profile.confidence + min(0.05, profile.support_count * 0.01)
            if score > 0:
                scored.append((profile, {"semantic_score": dense, "bm25_score": bm25, "sparse_source": 1.0 if fts_score is not None else 0.0, "score": score}))
        scored.sort(key=lambda item: item[1]["score"], reverse=True)
        return scored[:limit]

    def insert_encoding_decision(self, scope: Scope, decision: EncodingDecision) -> None:
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
        self.conn.execute(
            """
            insert into encoding_decisions
            (decision_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, candidate_type,
             candidate_json, source_span_ids, decision, reason_codes, scores_json, matched_existing_ids, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                scope.workspace_id,
                scope.user_id,
                scope.agent_id,
                scope.run_id,
                scope.session_id,
                scope.app_id,
                decision.candidate_type,
                dumps(candidate),
                dumps(decision.candidate.source_span_ids),
                decision.decision,
                dumps(decision.reason_codes),
                dumps(decision.scores),
                dumps(decision.matched_existing_ids),
                dt_to_str(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()

    def list_encoding_decisions(self, scope: Scope, candidate_type: str | None = None) -> list[dict[str, Any]]:
        where, params = self._scope_where(scope)
        if candidate_type:
            where += " and candidate_type = ?"
            params.append(candidate_type)
        rows = self.conn.execute(
            f"select * from encoding_decisions where {where} order by created_at",
            params,
        ).fetchall()
        return [
            {
                "decision_id": row["decision_id"],
                "scope": self._row_scope(row).__dict__,
                "candidate_type": row["candidate_type"],
                "candidate": loads(row["candidate_json"], {}),
                "source_span_ids": loads(row["source_span_ids"], []),
                "decision": row["decision"],
                "reason_codes": loads(row["reason_codes"], []),
                "scores": loads(row["scores_json"], {}),
                "matched_existing_ids": loads(row["matched_existing_ids"], []),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def insert_utility_example(self, example: dict[str, Any]) -> None:
        self.conn.execute(
            """
            insert into retrieval_utility_examples
            (example_id, query_id, query_text, query_type, candidate_id, candidate_type, features_json,
             label, label_source, answer_correct, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                example["example_id"],
                example.get("query_id"),
                example["query_text"],
                example.get("query_type"),
                example["candidate_id"],
                example["candidate_type"],
                dumps(example["features"]),
                example["label"],
                example["label_source"],
                None if example.get("answer_correct") is None else int(example["answer_correct"]),
                dt_to_str(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()

    def list_utility_examples(self, label: str | None = None) -> list[dict[str, Any]]:
        where = "1=1"
        params: list[Any] = []
        if label:
            where += " and label = ?"
            params.append(label)
        rows = self.conn.execute(
            f"select * from retrieval_utility_examples where {where} order by created_at",
            params,
        ).fetchall()
        return [
            {
                "example_id": row["example_id"],
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "query_type": row["query_type"],
                "candidate_id": row["candidate_id"],
                "candidate_type": row["candidate_type"],
                "features": loads(row["features_json"], {}),
                "label": row["label"],
                "label_source": row["label_source"],
                "answer_correct": None if row["answer_correct"] is None else bool(row["answer_correct"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_trace(self, trace_id: str, trace: dict[str, Any], scope: Scope | None = None) -> None:
        scope_columns = self._scope_columns(scope) if scope else {
            "workspace_id": None,
            "user_id": None,
            "agent_id": None,
            "run_id": None,
            "session_id": None,
            "app_id": None,
        }
        self.conn.execute(
            """
            insert or replace into debug_traces
            (trace_id, workspace_id, user_id, agent_id, run_id, session_id, app_id, trace_json, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                scope_columns["workspace_id"],
                scope_columns["user_id"],
                scope_columns["agent_id"],
                scope_columns["run_id"],
                scope_columns["session_id"],
                scope_columns["app_id"],
                dumps(trace),
                dt_to_str(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()

    def get_trace(self, trace_id: str, scope: Scope | None = None, *, include_session: bool = False) -> dict[str, Any] | None:
        where = "trace_id = ?"
        params: list[Any] = [trace_id]
        if scope:
            scope_where, scope_params = self._scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        row = self.conn.execute(f"select trace_json from debug_traces where {where}", params).fetchone()
        return loads(row["trace_json"], None) if row else None

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
        self.conn.execute(
            """
            insert into audit_events
            (audit_id, workspace_id, user_id, agent_id, run_id, session_id, app_id,
             event_type, object_type, object_id, trace_id, payload_json, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
                dumps(payload or {}),
                dt_to_str(datetime.now(timezone.utc)),
            ),
        )
        self.conn.commit()
        return audit_id

    def list_audit_events(self, scope: Scope, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        where, params = self._scope_where(scope)
        if event_type:
            where += " and event_type = ?"
            params.append(event_type)
        rows = self.conn.execute(
            f"select * from audit_events where {where} order by created_at desc limit ?",
            [*params, limit],
        ).fetchall()
        return [
            {
                "audit_id": row["audit_id"],
                "scope": self._row_scope(row).__dict__,
                "event_type": row["event_type"],
                "object_type": row["object_type"],
                "object_id": row["object_id"],
                "trace_id": row["trace_id"],
                "payload": loads(row["payload_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def enqueue_background_task(
        self,
        scope: Scope,
        task_type: str,
        *,
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        run_after: datetime | None = None,
    ) -> dict[str, Any]:
        existing = None
        if dedupe_key:
            existing = self.conn.execute("select * from background_tasks where dedupe_key = ?", (dedupe_key,)).fetchone()
        if existing:
            return self._row_to_background_task(existing)
        now = datetime.now(timezone.utc)
        task_id = new_id("task")
        self.conn.execute(
            """
            insert into background_tasks
            (task_id, task_type, workspace_id, user_id, agent_id, run_id, session_id, app_id,
             status, dedupe_key, payload_json, attempts, last_error, run_after, created_at, updated_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, 0, null, ?, ?, ?)
            """,
            (
                task_id,
                task_type,
                scope.workspace_id,
                scope.user_id,
                scope.agent_id,
                scope.run_id,
                scope.session_id,
                scope.app_id,
                dedupe_key,
                dumps(payload or {}),
                dt_to_str(run_after or now),
                dt_to_str(now),
                dt_to_str(now),
            ),
        )
        self.conn.commit()
        row = self.conn.execute("select * from background_tasks where task_id = ?", (task_id,)).fetchone()
        return self._row_to_background_task(row)

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
            where, params = self._scope_where(scope, include_session=include_session)
        if status:
            where += " and status = ?"
            params.append(status)
        rows = self.conn.execute(
            f"select * from background_tasks where {where} order by created_at desc limit ?",
            [*params, limit],
        ).fetchall()
        return [self._row_to_background_task(row) for row in rows]

    def next_background_tasks(self, *, limit: int = 10, scope: Scope | None = None, include_session: bool = False) -> list[dict[str, Any]]:
        where = "status = 'pending' and (run_after is null or run_after <= ?)"
        params: list[Any] = [dt_to_str(datetime.now(timezone.utc))]
        if scope:
            scope_where, scope_params = self._scope_where(scope, include_session=include_session)
            where += f" and {scope_where}"
            params.extend(scope_params)
        rows = self.conn.execute(
            f"select * from background_tasks where {where} order by run_after, created_at limit ?",
            [*params, limit],
        ).fetchall()
        return [self._row_to_background_task(row) for row in rows]

    def update_background_task(
        self,
        task_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        row = self.conn.execute("select * from background_tasks where task_id = ?", (task_id,)).fetchone()
        if not row:
            return None
        payload = loads(row["payload_json"], {})
        if result is not None:
            payload["result"] = result
        attempts = int(row["attempts"]) + (1 if status == "running" and row["status"] != "running" else 0)
        self.conn.execute(
            """
            update background_tasks
            set status = ?, payload_json = ?, attempts = ?, last_error = ?, updated_at = ?
            where task_id = ?
            """,
            (
                status,
                dumps(payload),
                attempts,
                error,
                dt_to_str(datetime.now(timezone.utc)),
                task_id,
            ),
        )
        self.conn.commit()
        updated = self.conn.execute("select * from background_tasks where task_id = ?", (task_id,)).fetchone()
        return self._row_to_background_task(updated)

    def _fts_scores(self, table: str, id_column: str, query: str, limit: int) -> dict[str, float]:
        if not self.fts_enabled:
            return {}
        fts_query = self._fts_query(query)
        if not fts_query:
            return {}
        try:
            rows = self.conn.execute(
                f"select {id_column} as id, bm25({table}) as rank from {table} where {table} match ? order by rank limit ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.Error:
            return {}
        return {row["id"]: 1.0 / index for index, row in enumerate(rows, start=1)}

    def _upsert_fts(self, table: str, id_column: str, object_id: str, text: str, rowid: int | None = None) -> None:
        if not self.fts_enabled:
            return
        try:
            self.conn.execute(f"delete from {table} where {id_column} = ?", (object_id,))
            if rowid is None:
                self.conn.execute(f"insert into {table} ({id_column}, content) values (?, ?)", (object_id, text))
            else:
                content_column = "content" if table == "fts_evidence" else "text" if table in {"fts_facts", "fts_profiles"} else "description"
                self.conn.execute(
                    f"insert into {table} (rowid, {id_column}, {content_column}) values (?, ?, ?)",
                    (rowid, object_id, text),
                )
        except sqlite3.Error:
            self.fts_enabled = False

    def _fts_query(self, query: str) -> str:
        tokens = tokenize(query)
        if not tokens:
            return ""
        quoted = ['"' + token.replace('"', '""') + '"' for token in tokens[:12]]
        return " OR ".join(quoted)

    def _row_scope(self, row: sqlite3.Row) -> Scope:
        return Scope(
            workspace_id=row["workspace_id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            app_id=row["app_id"],
        )

    def _row_to_span(self, row: sqlite3.Row) -> EvidenceSpan:
        return EvidenceSpan(
            span_id=row["span_id"],
            scope=self._row_scope(row),
            turn_id=row["turn_id"],
            speaker=row["speaker"],
            span_type=row["span_type"],
            content=row["content"],
            content_hash=row["content_hash"],
            timestamp=dt_from_str(row["timestamp"]) or datetime.now(timezone.utc),
            source_uri=row["source_uri"],
            parent_span_id=row["parent_span_id"],
            entities=loads(row["entities"], []),
            topics=loads(row["topics"], []),
            metadata=loads(row["metadata"], {}),
        )

    def _row_to_fact(self, row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(
            fact_id=row["fact_id"],
            scope=self._row_scope(row),
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            text=row["text"],
            category=row["category"],
            polarity=row["polarity"],
            confidence=row["confidence"],
            salience=row["salience"],
            observed_at=dt_from_str(row["observed_at"]),
            valid_from=dt_from_str(row["valid_from"]),
            valid_to=dt_from_str(row["valid_to"]),
            source_span_ids=loads(row["source_span_ids"], []),
            linked_fact_ids=loads(row["linked_fact_ids"], []),
            metadata=loads(row["metadata"], {}),
            created_at=dt_from_str(row["created_at"]) or datetime.now(timezone.utc),
        )

    def _row_to_fact_relation(self, row: sqlite3.Row) -> FactRelation:
        return FactRelation(
            relation_id=row["relation_id"],
            from_fact_id=row["from_fact_id"],
            to_fact_id=row["to_fact_id"],
            relation_type=row["relation_type"],
            source_span_ids=loads(row["source_span_ids"], []),
            confidence=row["confidence"],
        )

    def _row_to_event(self, row: sqlite3.Row) -> MemoryEvent:
        return MemoryEvent(
            event_id=row["event_id"],
            scope=self._row_scope(row),
            event_type=row["event_type"],
            participants=loads(row["participants"], []),
            description=row["description"],
            time_start=dt_from_str(row["time_start"]),
            time_end=dt_from_str(row["time_end"]),
            time_granularity=row["time_granularity"] or "unknown",
            time_source=row["time_source"] or "unknown",
            source_span_ids=loads(row["source_span_ids"], []),
            fact_ids=loads(row["fact_ids"], []),
            confidence=row["confidence"],
        )

    def _row_to_view(self, row: sqlite3.Row) -> CurrentView:
        return CurrentView(
            view_id=row["view_id"],
            scope=self._row_scope(row),
            view_type=row["view_type"],
            subject=row["subject"],
            text=row["text"],
            state_json=loads(row["state_json"], {}),
            source_fact_ids=loads(row["source_fact_ids"], []),
            source_event_ids=loads(row["source_event_ids"], []),
            source_span_ids=loads(row["source_span_ids"], []),
            confidence=row["confidence"],
            updated_at=dt_from_str(row["updated_at"]) or datetime.now(timezone.utc),
        )

    def _row_to_profile(self, row: sqlite3.Row) -> EntityProfile:
        return EntityProfile(
            profile_id=row["profile_id"],
            scope=self._row_scope(row),
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            profile_type=row["profile_type"],
            text=row["text"],
            state_json=loads(row["state_json"], {}),
            source_fact_ids=loads(row["source_fact_ids"], []),
            source_event_ids=loads(row["source_event_ids"], []),
            source_span_ids=loads(row["source_span_ids"], []),
            confidence=row["confidence"],
            support_count=row["support_count"],
            last_observed_at=dt_from_str(row["last_observed_at"]),
            updated_at=dt_from_str(row["updated_at"]) or datetime.now(timezone.utc),
        )

    def _row_to_entity(self, row: sqlite3.Row) -> EntityRecord:
        return EntityRecord(
            entity_id=row["entity_id"],
            scope=self._row_scope(row),
            name=row["name"],
            entity_type=row["entity_type"],
            aliases=loads(row["aliases"], []),
            source_span_ids=loads(row["source_span_ids"], []),
            observed_count=row["observed_count"],
            last_observed_at=dt_from_str(row["last_observed_at"]),
        )

    def _row_to_background_task(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "scope": self._row_scope(row).__dict__,
            "status": row["status"],
            "dedupe_key": row["dedupe_key"],
            "payload": loads(row["payload_json"], {}),
            "attempts": int(row["attempts"]),
            "last_error": row["last_error"],
            "run_after": row["run_after"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def flatten_source_span_ids(items: Iterable[MemoryFact | MemoryEvent | CurrentView | EntityProfile]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        for span_id in item.source_span_ids:
            if span_id not in seen:
                seen.add(span_id)
                out.append(span_id)
    return out
