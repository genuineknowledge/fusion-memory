from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any

from fusion_memory.core.models import CurrentView, EncodingDecision, EntityProfile, ExtractedCandidate, Scope
from fusion_memory.storage.postgres_store import PostgresRuntimeRepository, PostgresViewProfileRepository


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class PostgresViewProfileRepositoryTests(unittest.TestCase):
    def test_views_profiles_entities_and_profile_search(self) -> None:
        fake = FakePostgresConnection()
        repo = PostgresViewProfileRepository("postgresql://example/fusion", connect=lambda dsn: fake)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")

        view = CurrentView(
            view_id="view_1",
            scope=scope,
            view_type="current_preferences",
            subject="Atlas",
            text="Atlas currently uses Qdrant.",
            state_json={"backend": "Qdrant"},
            source_fact_ids=["fact_1"],
            source_event_ids=[],
            source_span_ids=["span_1"],
            confidence=0.9,
            updated_at=ts("2026-06-02T10:00:00+00:00"),
        )
        self.assertTrue(repo.upsert_current_view(view))
        views = repo.list_current_views(scope, view_type="current_preferences", include_session=True)
        self.assertEqual([item.view_id for item in views], ["view_1"])
        self.assertEqual(views[0].state_json["backend"], "Qdrant")

        profile = EntityProfile(
            profile_id="profile_1",
            scope=scope,
            entity_id="u",
            entity_type="user",
            profile_type="communication_style",
            text="User prefers concise technical tradeoffs.",
            state_json={"style": "concise"},
            source_fact_ids=["fact_2"],
            source_event_ids=[],
            source_span_ids=["span_2"],
            confidence=0.84,
            support_count=3,
            last_observed_at=ts("2026-06-02T10:00:00+00:00"),
            updated_at=ts("2026-06-03T10:00:00+00:00"),
        )
        self.assertTrue(repo.upsert_entity_profile(profile))
        profiles = repo.list_entity_profiles(scope, entity_id="u", include_session=True)
        self.assertEqual([item.profile_id for item in profiles], ["profile_1"])
        self.assertEqual(profiles[0].support_count, 3)

        search = repo.search_entity_profiles("concise tradeoffs", scope, include_session=True)
        self.assertEqual(search[0][0].profile_id, "profile_1")
        self.assertGreater(search[0][1]["score"], 0)
        profile_search_sql = next(sql for sql, _ in fake.executed if "from entity_profiles p" in sql.lower())
        self.assertIn("embedding_dense <=> cast(%s as vector)", profile_search_sql)

        entity = repo.upsert_entity(scope, "Qdrant", entity_type="project", source_span_ids=["span_1"], aliases=["qdrantdb"])
        self.assertEqual(entity.name, "Qdrant")
        entity = repo.upsert_entity(scope, "Qdrant", entity_type="project", source_span_ids=["span_2"], aliases=["qdrant-vector"])
        self.assertEqual(entity.observed_count, 2)
        self.assertEqual(entity.source_span_ids, ["span_1", "span_2"])
        self.assertEqual(entity.aliases, ["qdrantdb", "qdrant-vector"])

        entity_hits = repo.search_entities("qdrant-vector", scope, include_session=True)
        self.assertEqual(entity_hits[0][0].entity_id, entity.entity_id)
        self.assertGreater(entity_hits[0][1]["score"], 0)


class PostgresRuntimeRepositoryTests(unittest.TestCase):
    def test_encoding_utility_trace_audit_and_background_tasks(self) -> None:
        fake = FakePostgresConnection()
        repo = PostgresRuntimeRepository("postgresql://example/fusion", connect=lambda dsn: fake)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")

        decision = EncodingDecision(
            decision_id="decision_1",
            candidate_type="fact",
            candidate=ExtractedCandidate(
                local_id="cand_1",
                candidate_type="fact",
                text="Atlas uses Qdrant.",
                structured={"subject": "Atlas", "object": "Qdrant"},
                confidence=0.88,
                source_span_ids=["span_1"],
                extractor_name="rules",
            ),
            decision="accept",
            reason_codes=["source_present"],
            scores={"confidence": 0.88},
        )
        self.assertTrue(repo.insert_encoding_decision(scope, decision))
        decisions = repo.list_encoding_decisions(scope, candidate_type="fact")
        self.assertEqual(decisions[0]["candidate"]["local_id"], "cand_1")
        self.assertEqual(decisions[0]["reason_codes"], ["source_present"])

        example = {
            "example_id": "utility_1",
            "query_id": "q1",
            "query_text": "What does Atlas use?",
            "query_type": "factual_exact",
            "candidate_id": "fact_1",
            "candidate_type": "fact",
            "features": {"bm25_score": 0.8},
            "label": "positive",
            "label_source": "weak",
            "answer_correct": True,
        }
        self.assertTrue(repo.insert_utility_example(example))
        self.assertEqual(repo.list_utility_examples(label="positive")[0]["features"], {"bm25_score": 0.8})

        self.assertTrue(repo.save_trace("trace_1", {"operation": "search", "steps": [1]}, scope))
        self.assertEqual(repo.get_trace("trace_1", scope, include_session=True)["operation"], "search")

        audit_id = repo.insert_audit_event(scope, "memory.search", object_type="trace", object_id="trace_1", trace_id="trace_1", payload={"n": 1})
        audits = repo.list_audit_events(scope, event_type="memory.search", limit=10)
        self.assertEqual(audits[0]["audit_id"], audit_id)
        self.assertEqual(audits[0]["payload"], {"n": 1})

        task = repo.enqueue_background_task(
            scope,
            "refresh_session_summary",
            payload={"source_hash": "abc"},
            dedupe_key="summary:s:abc",
        )
        duplicate = repo.enqueue_background_task(scope, "refresh_session_summary", dedupe_key="summary:s:abc")
        self.assertEqual(duplicate["task_id"], task["task_id"])
        self.assertEqual(repo.list_background_tasks(scope, status="pending", include_session=True)[0]["task_id"], task["task_id"])
        self.assertEqual(repo.next_background_tasks(scope=scope, include_session=True)[0]["task_id"], task["task_id"])

        running = repo.update_background_task(task["task_id"], status="running")
        self.assertEqual(running["attempts"], 1)
        done = repo.update_background_task(task["task_id"], status="succeeded", result={"summary_span_id": "span_summary"})
        self.assertEqual(done["payload"]["result"]["summary_span_id"], "span_summary")
        self.assertEqual(done["attempts"], 1)

        task_sql = next(sql for sql, _ in fake.executed if "insert into background_tasks" in sql.lower())
        self.assertIn("payload_json", task_sql)


class FakePostgresCursor:
    def __init__(self, conn: "FakePostgresConnection") -> None:
        self.conn = conn
        self.description: list[tuple[str]] = []
        self.rowcount = -1
        self._results: list[dict[str, Any]] = []
        self.closed = False

    def execute(self, statement: str, params: Any = None) -> None:
        params = list(params or [])
        self.conn.executed.append((statement, params))
        normalized = " ".join(statement.lower().split())
        if "insert into current_views" in normalized:
            self._upsert("current_views", params[0], _current_view_row(params))
        elif "from current_views" in normalized:
            self._set_results(list(self.conn.tables["current_views"].values()))
        elif "insert into entity_profiles" in normalized:
            self._upsert("entity_profiles", params[0], _profile_row(params))
        elif normalized.startswith("with scored") and "from entity_profiles p" in normalized:
            self._search_profiles(params)
        elif "from entity_profiles" in normalized:
            self._set_results(list(self.conn.tables["entity_profiles"].values()))
        elif "select * from entities where entity_id" in normalized:
            self._get_by_id("entities", params[0])
        elif "insert into entities" in normalized:
            self._upsert("entities", params[0], _entity_row(params))
        elif "update entities set" in normalized:
            self._update_entity(params)
        elif "from entities" in normalized:
            self._set_results(sorted(self.conn.tables["entities"].values(), key=lambda row: (-row["observed_count"], row["name"])))
        elif "insert into encoding_decisions" in normalized:
            self._upsert("encoding_decisions", params[0], _encoding_row(params))
        elif "from encoding_decisions" in normalized:
            self._set_results(list(self.conn.tables["encoding_decisions"].values()))
        elif "insert into retrieval_utility_examples" in normalized:
            self._upsert("retrieval_utility_examples", params[0], _utility_row(params))
        elif "from retrieval_utility_examples" in normalized:
            self._set_results(list(self.conn.tables["retrieval_utility_examples"].values()))
        elif "insert into debug_traces" in normalized:
            self._upsert("debug_traces", params[0], _trace_row(params))
        elif "from debug_traces" in normalized:
            self._get_by_id("debug_traces", params[0])
        elif "insert into audit_events" in normalized:
            self._upsert("audit_events", params[0], _audit_row(params))
        elif "from audit_events" in normalized:
            self._set_results(list(self.conn.tables["audit_events"].values()))
        elif "from background_tasks where dedupe_key" in normalized:
            self._set_results([row for row in self.conn.tables["background_tasks"].values() if row["dedupe_key"] == params[0]][:1])
        elif "insert into background_tasks" in normalized:
            self._upsert("background_tasks", params[0], _task_row(params))
        elif "from background_tasks where task_id" in normalized:
            self._get_by_id("background_tasks", params[0])
        elif "update background_tasks" in normalized:
            self._update_task(params)
        elif "from background_tasks" in normalized:
            self._set_results(list(self.conn.tables["background_tasks"].values()))
        else:
            self._set_results([])

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._results)

    def close(self) -> None:
        self.closed = True

    def _upsert(self, table: str, object_id: str, row: dict[str, Any]) -> None:
        self.conn.tables[table][object_id] = row
        self.rowcount = 1
        self._set_results([])

    def _get_by_id(self, table: str, object_id: str) -> None:
        row = self.conn.tables[table].get(object_id)
        self._set_results([row] if row else [])

    def _update_entity(self, params: list[Any]) -> None:
        row = self.conn.tables["entities"][params[5]]
        row["aliases"] = params[0]
        row["source_span_ids"] = params[1]
        row["observed_count"] = params[2]
        row["last_observed_at"] = params[3]
        row["updated_at"] = params[4]
        self.rowcount = 1
        self._set_results([])

    def _update_task(self, params: list[Any]) -> None:
        row = self.conn.tables["background_tasks"][params[5]]
        row["status"] = params[0]
        row["payload_json"] = params[1]
        row["attempts"] = params[2]
        row["last_error"] = params[3]
        row["updated_at"] = params[4]
        self.rowcount = 1
        self._set_results([])

    def _search_profiles(self, params: list[Any]) -> None:
        terms = {term.lower() for term in params[0].split()}
        rows: list[dict[str, Any]] = []
        for row in self.conn.tables["entity_profiles"].values():
            hits = sum(1 for term in terms if term in row["text"].lower())
            if not hits:
                continue
            scored = dict(row)
            scored["bm25_score"] = hits / max(1, len(terms))
            scored["semantic_score"] = 0.5
            scored["score"] = 0.45 * scored["semantic_score"] + 0.40 * scored["bm25_score"] + 0.10 * scored["confidence"] + min(0.05, scored["support_count"] * 0.01)
            rows.append(scored)
        rows.sort(key=lambda row: row["score"], reverse=True)
        self._set_results(rows[: int(params[-1])])

    def _set_results(self, rows: list[dict[str, Any]]) -> None:
        self._results = rows
        keys = list(rows[0].keys()) if rows else []
        self.description = [(key,) for key in keys]


class FakePostgresConnection:
    def __init__(self) -> None:
        self.tables: dict[str, dict[str, dict[str, Any]]] = {
            "current_views": {},
            "entity_profiles": {},
            "entities": {},
            "encoding_decisions": {},
            "retrieval_utility_examples": {},
            "debug_traces": {},
            "audit_events": {},
            "background_tasks": {},
        }
        self.executed: list[tuple[str, list[Any]]] = []
        self.committed = 0
        self.rolled_back = 0
        self.closed = False

    def cursor(self) -> FakePostgresCursor:
        return FakePostgresCursor(self)

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1

    def close(self) -> None:
        self.closed = True


def _current_view_row(params: list[Any]) -> dict[str, Any]:
    return {
        "view_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "view_type": params[7],
        "subject": params[8],
        "text": params[9],
        "state_json": params[10],
        "source_fact_ids": params[11],
        "source_event_ids": params[12],
        "source_span_ids": params[13],
        "confidence": params[14],
        "updated_at": params[15],
        "expires_at": None,
    }


def _profile_row(params: list[Any]) -> dict[str, Any]:
    return {
        "profile_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "entity_id": params[7],
        "entity_type": params[8],
        "profile_type": params[9],
        "text": params[10],
        "state_json": params[11],
        "source_fact_ids": params[12],
        "source_event_ids": params[13],
        "source_span_ids": params[14],
        "confidence": params[15],
        "support_count": params[16],
        "last_observed_at": params[17],
        "updated_at": params[18],
        "embedding_dense": params[19],
    }


def _entity_row(params: list[Any]) -> dict[str, Any]:
    return {
        "entity_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "name": params[7],
        "entity_type": params[8],
        "aliases": params[9],
        "source_span_ids": params[10],
        "observed_count": 1,
        "last_observed_at": params[11],
        "created_at": params[12],
        "updated_at": params[13],
    }


def _encoding_row(params: list[Any]) -> dict[str, Any]:
    return {
        "decision_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "candidate_type": params[7],
        "candidate_json": params[8],
        "source_span_ids": params[9],
        "decision": params[10],
        "reason_codes": params[11],
        "scores_json": params[12],
        "matched_existing_ids": params[13],
        "created_at": "2026-06-01T10:00:00+00:00",
    }


def _utility_row(params: list[Any]) -> dict[str, Any]:
    return {
        "example_id": params[0],
        "query_id": params[1],
        "query_text": params[2],
        "query_type": params[3],
        "candidate_id": params[4],
        "candidate_type": params[5],
        "features_json": params[6],
        "label": params[7],
        "label_source": params[8],
        "answer_correct": params[9],
        "created_at": "2026-06-01T10:00:00+00:00",
    }


def _trace_row(params: list[Any]) -> dict[str, Any]:
    return {
        "trace_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "trace_json": params[7],
        "created_at": "2026-06-01T10:00:00+00:00",
    }


def _audit_row(params: list[Any]) -> dict[str, Any]:
    return {
        "audit_id": params[0],
        "workspace_id": params[1],
        "user_id": params[2],
        "agent_id": params[3],
        "run_id": params[4],
        "session_id": params[5],
        "app_id": params[6],
        "event_type": params[7],
        "object_type": params[8],
        "object_id": params[9],
        "trace_id": params[10],
        "payload_json": params[11],
        "created_at": "2026-06-01T10:00:00+00:00",
    }


def _task_row(params: list[Any]) -> dict[str, Any]:
    return {
        "task_id": params[0],
        "task_type": params[1],
        "workspace_id": params[2],
        "user_id": params[3],
        "agent_id": params[4],
        "run_id": params[5],
        "session_id": params[6],
        "app_id": params[7],
        "status": "pending",
        "dedupe_key": params[8],
        "payload_json": params[9],
        "attempts": 0,
        "last_error": None,
        "run_after": params[10],
        "created_at": params[11],
        "updated_at": params[12],
    }


if __name__ == "__main__":
    unittest.main()
