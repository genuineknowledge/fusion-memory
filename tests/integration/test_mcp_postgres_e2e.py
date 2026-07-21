from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
import uuid
from typing import Any

import pytest

from tests.integration.mcp_stack import DeployedMcpStack, MissingIntegrationConfig


@pytest.fixture
def deployed_stack() -> DeployedMcpStack:
    try:
        return DeployedMcpStack()
    except MissingIntegrationConfig as exc:
        pytest.skip(str(exc))


@pytest.fixture
def postgres_connection():
    dsn = os.environ.get("FUSION_MEMORY_PG_DSN", "").strip()
    if not dsn:
        pytest.skip("set FUSION_MEMORY_PG_DSN before Postgres persistence integration tests")
    psycopg2 = pytest.importorskip("psycopg2", reason="install the Postgres test dependency")
    connection = psycopg2.connect(dsn)
    connection.autocommit = True
    try:
        yield connection
    finally:
        connection.close()


@pytest.mark.integration
def test_user_scope_and_cross_session_workspace_retrieval(deployed_stack: DeployedMcpStack):
    marker = f"user-a-e2e-{uuid.uuid4().hex}"
    added = deployed_stack.client(user="a", workspace="ws-a", session="s1").call(
        "memory_add", {"content": marker, "source": "task10-e2e"}
    )
    assert added["ok"] is True

    same_user = deployed_stack.client(user="a", workspace="ws-b", session="s2").call(
        "memory_search", {"query": marker, "limit": 12}
    )
    other_user = deployed_stack.client(user="b", workspace="ws-b", session="s2").call(
        "memory_search", {"query": marker, "limit": 12}
    )
    assert any(marker in text for text in _candidate_texts(same_user))
    assert all(marker not in text for text in _candidate_texts(other_user))


@pytest.mark.integration
def test_different_users_are_served_concurrently(deployed_stack: DeployedMcpStack):
    try:
        deployed_stack.reset_worker_stats()
    except MissingIntegrationConfig as exc:
        pytest.skip(str(exc))

    barrier = __import__("threading").Barrier(2)
    queries = {user: f"parallel-{user}-{uuid.uuid4().hex}" for user in ("a", "b")}

    def add(user: str) -> dict[str, object]:
        barrier.wait(timeout=5)
        return deployed_stack.client(user=user, workspace=f"ws-{user}", session="parallel").call(
            "memory_add", {"content": queries[user], "source": "task10-concurrency"}
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(add, user) for user in ("a", "b")]
        results = [future.result(timeout=30) for future in futures]
    assert all(result["ok"] is True for result in results)
    deployed_stack.assert_worker_overlap((queries["a"], queries["b"]))


@pytest.mark.integration
def test_history_batch_replay_is_idempotent(deployed_stack: DeployedMcpStack, postgres_connection: Any):
    marker = f"batch-e2e-{uuid.uuid4().hex}"
    arguments = {
        "batch_id": marker,
        "messages": [
            {"role": "user", "content": f"{marker} request", "turn_id": "turn-1"},
            {"role": "assistant", "content": f"{marker} response", "turn_id": "turn-1"},
        ],
        "metadata": {"source": "task10-history-replay"},
    }
    client = deployed_stack.client(user="a", workspace="ws-history", session="history-s1")
    first = client.call("memory_add_batch", arguments)
    durable_before_replay = _postgres_batch_snapshot(postgres_connection, marker)
    before_replay = deployed_stack.client(user="a", workspace="ws-other", session="history-s2").call(
        "memory_search", {"query": marker, "limit": 12}
    )
    second = client.call("memory_add_batch", arguments)
    durable_after_replay = _postgres_batch_snapshot(postgres_connection, marker)
    after_replay = deployed_stack.client(user="a", workspace="ws-other", session="history-s2").call(
        "memory_search", {"query": marker, "limit": 12}
    )
    assert first["ok"] is True
    assert second == first
    assert first["result"]["message_count"] == 2
    assert _candidate_identities(before_replay) == _candidate_identities(after_replay)
    assert _candidate_identities(after_replay)
    assert marker in json.dumps(after_replay, ensure_ascii=False)
    assert durable_after_replay == durable_before_replay
    ledger = durable_after_replay["ledger"]
    evidence = durable_after_replay["evidence"]
    assert len(ledger) == 1
    assert ledger[0][1] == marker
    assert ledger[0][3] == "completed"
    assert len(evidence) == 2
    assert {row[1] for row in evidence} == {ledger[0][0]}
    assert {row[6] for row in evidence} == {f"{marker} request", f"{marker} response"}


@pytest.mark.integration
def test_user_systemd_restart_reconnects_same_user(deployed_stack: DeployedMcpStack):
    if os.environ.get("FUSION_MEMORY_E2E_SYSTEMD") != "1":
        pytest.skip("set FUSION_MEMORY_E2E_SYSTEMD=1 only for an installed user-systemd service")
    marker = f"systemd-e2e-{uuid.uuid4().hex}"
    client = deployed_stack.client(user="a", workspace="ws-systemd", session="restart")
    assert client.call("memory_add", {"content": marker})["ok"] is True
    subprocess.run(
        ["systemctl", "--user", "restart", "fusion-memory-mcp.service"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    deadline = time.monotonic() + 30.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            result = client.call("memory_search", {"query": marker, "limit": 8})
            if result.get("ok") is True and marker in json.dumps(result, ensure_ascii=False):
                return
        except BaseException as exc:
            last_error = exc
        time.sleep(0.25)
    raise AssertionError("MCP did not reconnect after user-systemd restart") from last_error


def _candidate_identities(result: dict[str, object]) -> set[tuple[str, str, str]]:
    payload = result.get("result")
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    return {
        (str(candidate.get("id")), str(candidate.get("type")), str(candidate.get("text")))
        for candidate in candidates
        if isinstance(candidate, dict)
    }


def _candidate_texts(result: dict[str, object]) -> list[str]:
    return [identity[2] for identity in _candidate_identities(result)]


def _postgres_batch_snapshot(connection: Any, marker: str) -> dict[str, tuple[tuple[object, ...], ...]]:
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            select user_id, batch_id, request_hash, status,
                   coalesce(result::text, ''), coalesce(trace_id, '')
            from fusion_memory_mcp_batches
            where batch_id = %s
            order by user_id, batch_id
            """,
            (marker,),
        )
        ledger = tuple(tuple(row) for row in cursor.fetchall())
        cursor.execute(
            """
            select span_id, user_id, workspace_id, session_id, turn_id,
                   speaker, content, content_hash
            from evidence_spans
            where position(%s in content) > 0
            order by span_id
            """,
            (marker,),
        )
        evidence = tuple(tuple(row) for row in cursor.fetchall())
        return {"ledger": ledger, "evidence": evidence}
    finally:
        cursor.close()


def test_postgres_batch_snapshot_preserves_durable_row_identities():
    marker = "batch-e2e-marker"
    ledger = [("user-a", marker, "request-hash", "completed", '{"message_count":2}', "trace-1")]
    evidence = [
        ("span-1", "user-a", "ws-history", "history-s1", "turn-1", "user", f"{marker} request", "hash-1"),
        (
            "span-2",
            "user-a",
            "ws-history",
            "history-s1",
            "turn-1",
            "assistant",
            f"{marker} response",
            "hash-2",
        ),
    ]

    class Cursor:
        def __init__(self) -> None:
            self.rows: list[tuple[object, ...]] = []
            self.calls: list[tuple[str, tuple[str, ...]]] = []

        def execute(self, sql: str, params: tuple[str, ...]) -> None:
            self.calls.append((sql, params))
            self.rows = ledger if "fusion_memory_mcp_batches" in sql else evidence

        def fetchall(self):
            return self.rows

        def close(self) -> None:
            pass

    class Connection:
        def __init__(self) -> None:
            self.cursor_instance = Cursor()

        def cursor(self):
            return self.cursor_instance

    connection = Connection()

    assert _postgres_batch_snapshot(connection, marker) == {
        "ledger": tuple(ledger),
        "evidence": tuple(evidence),
    }
    assert [params for _sql, params in connection.cursor_instance.calls] == [(marker,), (marker,)]
