from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
import uuid

import pytest

from tests.integration.mcp_stack import DeployedMcpStack


@pytest.fixture
def deployed_stack() -> DeployedMcpStack:
    try:
        return DeployedMcpStack()
    except RuntimeError as exc:
        pytest.skip(str(exc))


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
    except RuntimeError as exc:
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
def test_history_batch_replay_is_idempotent(deployed_stack: DeployedMcpStack):
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
    before_replay = deployed_stack.client(user="a", workspace="ws-other", session="history-s2").call(
        "memory_search", {"query": marker, "limit": 12}
    )
    second = client.call("memory_add_batch", arguments)
    after_replay = deployed_stack.client(user="a", workspace="ws-other", session="history-s2").call(
        "memory_search", {"query": marker, "limit": 12}
    )
    assert first["ok"] is True
    assert second == first
    assert first["result"]["message_count"] == 2
    assert _candidate_identities(before_replay) == _candidate_identities(after_replay)
    assert _candidate_identities(after_replay)
    assert marker in json.dumps(after_replay, ensure_ascii=False)


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
