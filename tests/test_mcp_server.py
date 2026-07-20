import contextlib
import json
import subprocess
import sys
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken

from fusion_memory.core.models import Scope
from fusion_memory.mcp_runtime import FusionMemoryRuntime, runtime_from_env
from fusion_memory.mcp_server import _pooled_connection_factory, create_mcp_app, create_mcp_server, run_mcp_server
from fusion_memory.storage.postgres_store import PostgresMemoryStore
from fusion_memory.storage.token_store import PostgresTokenStore


class FakeTokenVerifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        tokens = {
            "token-a": ("user-a", ["memory:read", "memory:write", "memory:sync"]),
            "token-read": ("user-a", ["memory:read"]),
        }
        entry = tokens.get(token)
        if entry is None:
            return None
        user_id, scopes = entry
        return AccessToken(token=token, client_id=f"client-{user_id}", subject=user_id, scopes=scopes)


class FakeMemoryRuntime:
    def __init__(self) -> None:
        self.items: list[tuple[Scope, str]] = []
        self.last_scope = Scope()

    async def add(self, scope: Scope, content: str, source: str | None) -> dict[str, Any]:
        self.last_scope = scope
        self.items.append((scope, content))
        return {"saved": True, "source": source}

    async def search(self, scope: Scope, query: str, limit: int) -> dict[str, Any]:
        self.last_scope = scope
        matches = [content for item_scope, content in self.items if item_scope.user_id == scope.user_id and query in content]
        return {"candidates": matches[:limit]}

    async def answer_context(self, scope: Scope, query: str, limit: int) -> dict[str, Any]:
        return await self.search(scope, query, limit)

    async def add_batch(
        self, scope: Scope, messages: list[dict[str, Any]], batch_id: str, metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        del metadata
        self.last_scope = scope
        self.items.extend((scope, str(message.get("content") or "")) for message in messages)
        return {"batch_id": batch_id, "message_count": len(messages)}


@pytest.fixture
def fake_runtime() -> FakeMemoryRuntime:
    return FakeMemoryRuntime()


@pytest.fixture
def mcp_app(fake_runtime: FakeMemoryRuntime):
    return create_mcp_app(
        runtime=fake_runtime,
        token_verifier=FakeTokenVerifier(),
        path="/mcp",
        public_url="http://test/mcp",
        stateless_http=True,
        json_response=True,
    )


@pytest.fixture
async def mcp_client_factory(mcp_app):
    async with mcp_app.router.lifespan_context(mcp_app):
        @contextlib.asynccontextmanager
        async def open_client(*, token: str, session_id: str, workspace_id: str = "ws-1"):
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Fusion-Memory-Workspace": workspace_id,
                "X-Fusion-Memory-Session": session_id,
            }
            transport = httpx.ASGITransport(app=mcp_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers) as http_client:
                async with streamable_http_client("http://test/mcp", http_client=http_client) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session

        yield open_client


@pytest.mark.anyio
async def test_tools_list_contains_memory_tools(mcp_client_factory):
    async with mcp_client_factory(token="token-a", session_id="s1") as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert {"memory_add", "memory_search", "memory_answer_context", "memory_add_batch"} <= names


@pytest.mark.anyio
async def test_same_user_cross_session_searches_all_sessions(mcp_client_factory):
    async with mcp_client_factory(token="token-a", session_id="s1") as client:
        await client.call_tool("memory_add", {"content": "prefers postgres"})
    async with mcp_client_factory(token="token-a", session_id="s2") as client:
        result = await client.call_tool("memory_search", {"query": "postgres"})
    assert "prefers postgres" in json.dumps(result.structuredContent, ensure_ascii=False)


@pytest.mark.anyio
async def test_user_id_comes_from_token_and_is_not_a_tool_parameter(mcp_client_factory, fake_runtime):
    async with mcp_client_factory(token="token-a", session_id="s1") as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert "user_id" not in tools["memory_search"].inputSchema["properties"]
        await client.call_tool("memory_search", {"query": "secret"})
    assert fake_runtime.last_scope.user_id == "user-a"


@pytest.mark.anyio
async def test_invalid_token_is_rejected_at_http_auth_layer(mcp_app):
    transport = httpx.ASGITransport(app=mcp_app)
    async with mcp_app.router.lifespan_context(mcp_app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                headers={"Authorization": "Bearer invalid"},
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            )
    assert response.status_code == 401


@pytest.mark.anyio
async def test_public_origin_without_mcp_path_passes_transport_security():
    server = create_mcp_server(
        runtime=FakeMemoryRuntime(),
        token_verifier=FakeTokenVerifier(),
        path="/mcp",
        public_url="https://memory.example/mcp",
    )
    assert server.settings.transport_security.allowed_origins == ["https://memory.example"]

    app = server.streamable_http_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="https://memory.example") as client:
            response = await client.post(
                "/mcp",
                headers={"Authorization": "Bearer invalid", "Origin": "https://memory.example"},
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            )
    assert response.status_code == 401


@pytest.mark.anyio
async def test_runtime_lifespan_supervises_health_pools_without_blocking_shutdown():
    import threading

    started = threading.Event()
    release = threading.Event()

    class BlockingPool:
        def healthy_endpoints(self):
            started.set()
            release.wait(timeout=5)
            return []

    class Runtime:
        endpoint_pools = (BlockingPool(),)
        health_check_interval_seconds = 0.01

    server = create_mcp_server(
        runtime=Runtime(),
        token_verifier=FakeTokenVerifier(),
        public_url="http://test/mcp",
    )
    app = server.streamable_http_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer token-a"},
        ) as http_client:
            async with streamable_http_client("http://test/mcp", http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as client:
                    await client.initialize()
                    assert await __import__("anyio").to_thread.run_sync(started.wait, 0.2)
                    release.set()


@pytest.mark.anyio
async def test_write_scope_is_required(mcp_client_factory):
    async with mcp_client_factory(token="token-read", session_id="s1") as client:
        result = await client.call_tool("memory_add", {"content": "denied"})
    assert result.structuredContent["ok"] is False
    assert result.structuredContent["error"]["code"] == "insufficient_scope"


def test_cli_starts_only_mcp_server(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("fusion_memory.mcp_server.run_mcp_server", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fusion-memory", "mcp-server", "--host", "0.0.0.0", "--port", "9123", "--path", "/native", "--public-url", "http://test/native"],
    )

    from fusion_memory.cli import main

    main()

    assert captured == {"host": "0.0.0.0", "port": 9123, "path": "/native", "public_url": "http://test/native"}


def test_package_does_not_eagerly_import_mcp_server():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import fusion_memory.mcp_runtime; assert 'fusion_memory.mcp_server' not in sys.modules"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_mcp_server_checks_pepper_before_constructing_runtime(monkeypatch):
    constructed = False

    def fake_runtime_from_env():
        nonlocal constructed
        constructed = True
        raise AssertionError("runtime must not be constructed")

    monkeypatch.delenv("FUSION_MEMORY_TOKEN_PEPPER", raising=False)
    monkeypatch.setenv("FUSION_MEMORY_MCP_PUBLIC_URL", "http://test/mcp")
    monkeypatch.setattr("fusion_memory.mcp_server.runtime_from_env", fake_runtime_from_env)

    with pytest.raises(ValueError, match="FUSION_MEMORY_TOKEN_PEPPER is required"):
        run_mcp_server()

    assert constructed is False


def test_runtime_from_env_accepts_pg_dsn(monkeypatch):
    captured: dict[str, object] = {}

    class FakePool:
        def __init__(self, dsn, **kwargs):
            captured["dsn"] = dsn

        def close(self):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.delenv("FUSION_MEMORY_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("FUSION_MEMORY_PG_DSN", "postgresql://memory")
    monkeypatch.setattr("fusion_memory.mcp_runtime.PostgresConnectionPool", FakePool)
    monkeypatch.setattr("fusion_memory.mcp_runtime.PostgresMemoryStore", FakeStore)
    monkeypatch.setattr("fusion_memory.mcp_runtime.PostgresOperationExecutor", FakeExecutor)
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_embedder", lambda: object())
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_reranker", lambda: object())
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_extractor", lambda: object())
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_async_extractor", lambda: None)
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_query_intent_refiner", lambda: None)
    monkeypatch.setattr("fusion_memory.mcp_runtime.build_runtime_retrieval_flags", lambda: object())

    runtime, _ = runtime_from_env()

    assert isinstance(runtime, FusionMemoryRuntime)
    assert captured["dsn"] == "postgresql://memory"


def test_runtime_factory_creates_request_local_extractor_and_refiner(monkeypatch):
    import concurrent.futures

    class FakePool:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    created_extractors: list[object] = []
    created_refiners: list[object] = []
    monkeypatch.setenv("FUSION_MEMORY_PG_DSN", "postgresql://memory")
    monkeypatch.setattr("fusion_memory.mcp_runtime.PostgresConnectionPool", FakePool)
    monkeypatch.setattr("fusion_memory.mcp_runtime.PostgresMemoryStore", FakeStore)
    monkeypatch.setattr("fusion_memory.mcp_runtime.PostgresOperationExecutor", FakeExecutor)
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_embedder", lambda: None)
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_reranker", lambda: None)
    monkeypatch.setattr("fusion_memory.mcp_runtime._build_extractor", lambda: None)
    monkeypatch.setattr(
        "fusion_memory.mcp_runtime._build_async_extractor",
        lambda: created_extractors.append(object()) or created_extractors[-1],
    )
    monkeypatch.setattr(
        "fusion_memory.mcp_runtime._build_query_intent_refiner",
        lambda: created_refiners.append(object()) or created_refiners[-1],
    )
    monkeypatch.setattr("fusion_memory.mcp_runtime.build_runtime_retrieval_flags", lambda: object())

    runtime, _ = runtime_from_env()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(runtime._service_factory, object()) for _ in range(2)]
        first, second = [future.result() for future in futures]

    assert first.async_extractor is not second.async_extractor
    assert first.planner.intent_refiner is not second.planner.intent_refiner


def test_runtime_close_runs_resource_closers_once():
    calls: list[str] = []
    runtime = FusionMemoryRuntime(
        object(),
        lambda _store: object(),
        closers=(lambda: calls.append("executor"), lambda: calls.append("store"), lambda: calls.append("pool")),
    )

    runtime.close()
    runtime.close()

    assert calls == ["executor", "store", "pool"]


def test_runtime_close_attempts_later_resources_after_a_close_failure():
    calls: list[str] = []

    def fail_executor_close():
        calls.append("executor")
        raise RuntimeError("executor close failed")

    runtime = FusionMemoryRuntime(
        object(),
        lambda _store: object(),
        closers=(fail_executor_close, lambda: calls.append("store"), lambda: calls.append("pool")),
    )

    with pytest.raises(RuntimeError, match="executor close failed"):
        runtime.close()

    assert calls == ["executor", "store", "pool"]


def test_root_postgres_store_closes_its_pool_once():
    class Pool:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1

    pool = Pool()
    store = PostgresMemoryStore("postgresql://memory", pool=pool)

    store.close()
    store.close()

    assert pool.closes == 1


def test_pooled_token_connection_releases_once_when_cursor_creation_fails():
    class BrokenConnection:
        def cursor(self):
            raise RuntimeError("cursor unavailable")

        def rollback(self):
            pass

    class Lease:
        def __init__(self):
            self.exits = 0

        def __enter__(self):
            return BrokenConnection()

        def __exit__(self, exc_type, exc, tb):
            self.exits += 1

    class Pool:
        def __init__(self):
            self.lease = Lease()

        def connection(self, timeout_seconds):
            assert timeout_seconds == 5.0
            return self.lease

    pool = Pool()
    store = PostgresTokenStore(_pooled_connection_factory(pool), pepper="pepper")

    with pytest.raises(RuntimeError, match="cursor unavailable"):
        store.list_tokens("user-a")

    assert pool.lease.exits == 1


def test_pooled_token_connection_close_is_idempotent():
    class Connection:
        pass

    class Lease:
        def __init__(self):
            self.exits = 0

        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            self.exits += 1

    class Pool:
        def __init__(self):
            self.lease = Lease()

        def connection(self, timeout_seconds):
            return self.lease

    pool = Pool()
    connection = _pooled_connection_factory(pool)()

    connection.close()
    connection.close()

    assert pool.lease.exits == 1


def test_pooled_token_connection_releases_when_cursor_close_fails():
    class Cursor:
        def execute(self, *args):
            pass

        def fetchall(self):
            return []

        def close(self):
            raise RuntimeError("cursor close failed")

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def rollback(self):
            pass

    class Lease:
        def __init__(self):
            self.exits = 0

        def __enter__(self):
            return Connection()

        def __exit__(self, exc_type, exc, tb):
            self.exits += 1

    class Pool:
        def __init__(self):
            self.lease = Lease()

        def connection(self, timeout_seconds):
            return self.lease

    pool = Pool()
    store = PostgresTokenStore(_pooled_connection_factory(pool), pepper="pepper")

    with pytest.raises(RuntimeError, match="cursor close failed"):
        store.list_tokens("user-a")

    assert pool.lease.exits == 1
