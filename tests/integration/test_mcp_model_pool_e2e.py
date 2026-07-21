from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from tests.integration.mcp_stack import DeployedMcpStack, E2EConfig


@dataclass
class ManagedPoolStack:
    stack: DeployedMcpStack
    workers: list[subprocess.Popen[bytes]]
    worker_urls: list[str]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, process: subprocess.Popen[bytes], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"managed process exited with status {process.returncode}")
        try:
            if httpx.get(url, timeout=1.0).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"managed process did not become ready at {url}")


@pytest.fixture
def managed_pool_stack() -> ManagedPoolStack:
    try:
        base = DeployedMcpStack()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    dsn = os.environ.get("FUSION_MEMORY_PG_DSN", "").strip()
    pepper = os.environ.get("FUSION_MEMORY_TOKEN_PEPPER", "").strip()
    if not dsn or not pepper:
        pytest.skip("set FUSION_MEMORY_PG_DSN and FUSION_MEMORY_TOKEN_PEPPER for managed pool E2E")

    runtime_dir = Path(__file__).parents[2] / ".runtime" / "task10" / f"model-pool-{uuid.uuid4().hex}"
    runtime_dir.mkdir(parents=True, mode=0o700)
    os.chmod(runtime_dir, 0o700)
    worker_ports = [_free_port(), _free_port()]
    mcp_port = _free_port()
    worker_urls = [f"http://127.0.0.1:{port}" for port in worker_ports]
    log_paths = [runtime_dir / f"worker-{index}.log" for index in range(2)]
    mcp_log_path = runtime_dir / "mcp.log"
    for path in [*log_paths, mcp_log_path]:
        path.touch(mode=0o600)
        os.chmod(path, 0o600)
    log_files = [path.open("wb") for path in log_paths]
    mcp_log = mcp_log_path.open("wb")
    workers: list[subprocess.Popen[bytes]] = []
    mcp_process: subprocess.Popen[bytes] | None = None
    try:
        adapter = Path(__file__).with_name("mcp_stack.py")
        for port, log_file in zip(worker_ports, log_files):
            workers.append(
                subprocess.Popen(
                    [sys.executable, str(adapter), "--fake-worker", "--port", str(port)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            )
        for url, process in zip(worker_urls, workers):
            _wait_http(f"{url}/health", process)

        env = dict(os.environ)
        env.update(
            {
                "FUSION_MEMORY_EMBEDDING_PROVIDER": "http",
                "FUSION_MEMORY_EMBEDDING_ENDPOINTS": ",".join(f"{url}/v1/embeddings" for url in worker_urls),
                "FUSION_MEMORY_EMBEDDING_DIMENSION": "1024",
                "FUSION_MEMORY_EMBEDDING_FAILURE_THRESHOLD": "2",
                "FUSION_MEMORY_EMBEDDING_RECOVERY_SECONDS": "60",
                "FUSION_MEMORY_EMBEDDING_MAX_IN_FLIGHT": "1",
                "FUSION_MEMORY_RERANKER_PROVIDER": "http",
                "FUSION_MEMORY_RERANKER_ENDPOINTS": ",".join(f"{url}/v1/rerank" for url in worker_urls),
                "FUSION_MEMORY_RERANKER_FAILURE_THRESHOLD": "2",
                "FUSION_MEMORY_RERANKER_RECOVERY_SECONDS": "60",
                "FUSION_MEMORY_RERANKER_MAX_IN_FLIGHT": "1",
                "FUSION_MEMORY_EXTRACTOR_MODE": "off",
                "FUSION_MEMORY_QUERY_INTENT_MODE": "off",
                "FUSION_MEMORY_MCP_SEARCH_MODE": "balanced",
                "FUSION_MEMORY_MCP_PUBLIC_URL": f"http://127.0.0.1:{mcp_port}/mcp",
            }
        )
        mcp_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fusion_memory.cli",
                "mcp-server",
                "--host",
                "127.0.0.1",
                "--port",
                str(mcp_port),
                "--path",
                "/mcp",
            ],
            cwd=Path(__file__).parents[2],
            env=env,
            stdout=mcp_log,
            stderr=subprocess.STDOUT,
        )
        _wait_http(f"http://127.0.0.1:{mcp_port}/mcp", mcp_process)
        config = E2EConfig(
            url=f"http://127.0.0.1:{mcp_port}/mcp",
            token_a=base.config.token_a,
            token_b=base.config.token_b,
        )
        yield ManagedPoolStack(DeployedMcpStack(config), workers, worker_urls)
    finally:
        for process in ([mcp_process] if mcp_process is not None else []) + workers:
            if process.poll() is None:
                process.terminate()
        for process in ([mcp_process] if mcp_process is not None else []) + workers:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        for log_file in [*log_files, mcp_log]:
            log_file.close()


@pytest.mark.integration
def test_model_pool_survives_one_of_two_worker_processes(managed_pool_stack: ManagedPoolStack):
    stack = managed_pool_stack.stack
    marker = f"pool-e2e-{uuid.uuid4().hex}"
    client = stack.client(user="a", workspace="ws-pool", session="pool")
    assert client.call("memory_add", {"content": marker})["ok"] is True
    assert client.call("memory_search", {"query": marker, "limit": 8})["ok"] is True

    failed_worker = managed_pool_stack.workers[0]
    failed_worker.terminate()
    failed_worker.wait(timeout=5)
    for _ in range(4):
        assert client.call("memory_search", {"query": marker, "limit": 8})["ok"] is True

    health = client.call("memory_health", {})
    pools = health["result"]["background"]["model_pools"]
    for label in ("embedding", "reranker"):
        snapshots = pools[label]
        assert len(snapshots) == 2
        failed = [snapshot for snapshot in snapshots if snapshot["healthy"] is False]
        healthy = [snapshot for snapshot in snapshots if snapshot["healthy"] is True]
        assert len(failed) == 1
        assert int(failed[0]["failure_count"]) >= 2
        assert len(healthy) == 1
