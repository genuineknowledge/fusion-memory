from __future__ import annotations

import os
import json
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import anyio

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope as ASGIScope, Send

from fusion_memory.core.auth import AuthorizationError
from fusion_memory.core.models import Scope
from fusion_memory.mcp_auth import FusionMemoryTokenVerifier
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.mcp_runtime import runtime_from_env
from fusion_memory.storage.postgres_pool import PoolAcquireTimeout, RetryableOperationError
from fusion_memory.storage.postgres_store import PostgresBackendUnavailable
from fusion_memory.storage.token_store import PostgresTokenStore
from fusion_memory.storage.batch_ledger import BatchIdConflictError


MAX_SEARCH_LIMIT = 32
_request_provenance: ContextVar["RequestProvenance | None"] = ContextVar("mcp_request_provenance", default=None)


@dataclass(frozen=True)
class RequestProvenance:
    workspace_id: str | None
    session_id: str | None


class RequestProvenanceMiddleware:
    """Make MCP provenance headers request-local without trusting tool payloads."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: ASGIScope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        provenance = RequestProvenance(
            workspace_id=headers.get("x-fusion-memory-workspace") or None,
            session_id=headers.get("x-fusion-memory-session") or None,
        )
        token = _request_provenance.set(provenance)
        try:
            await self.app(scope, receive, send)
        finally:
            _request_provenance.reset(token)


def create_mcp_server(
    *,
    runtime: Any,
    token_verifier: Any,
    host: str = "127.0.0.1",
    port: int = 8700,
    path: str = "/mcp",
    public_url: str,
    lifespan: Any = None,
) -> FastMCP:
    """Create the authenticated MCP server and its structured tools."""
    if not path.startswith("/"):
        raise ValueError("MCP path must start with '/'")
    public_parts = urlsplit(public_url)
    public_host = public_parts.netloc
    allowed_hosts = [f"{host}:*", host, public_host]
    allowed_origins = [f"{public_parts.scheme}://{public_host}"]
    server = FastMCP(
        "Fusion Memory",
        host=host,
        port=port,
        streamable_http_path=path,
        token_verifier=token_verifier,
        auth=AuthSettings(issuer_url=public_url, resource_server_url=public_url),
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
    )
    _install_runtime_lifespan(server, runtime, lifespan)

    @server.tool(structured_output=True)
    async def memory_add(content: str, source: str | None = None) -> dict[str, Any]:
        """Persist a memory item for the authenticated user and request provenance."""
        return await _call_tool(runtime, {"memory:write"}, write=True, content=content, source=source)

    @server.tool(structured_output=True)
    async def memory_search(query: str, limit: int = 12) -> dict[str, Any]:
        """Search all workspaces and sessions belonging to the authenticated user."""
        return await _call_tool(runtime, {"memory:read"}, query=query, limit=limit)

    @server.tool(structured_output=True)
    async def memory_answer_context(query: str, limit: int = 12) -> dict[str, Any]:
        """Build answer context from all memory owned by the authenticated user."""
        return await _call_tool(runtime, {"memory:read"}, answer_context=True, query=query, limit=limit)

    @server.tool(structured_output=True)
    async def memory_add_batch(
        messages: list[dict[str, Any]], batch_id: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Persist a bounded batch with the authenticated user's request provenance."""
        return await _call_tool(
            runtime,
            {"memory:write", "memory:sync"},
            write=True,
            messages=messages,
            batch_id=batch_id,
            metadata=metadata,
        )

    @server.tool(structured_output=True)
    async def memory_health() -> dict[str, Any]:
        """Return authenticated, read-scoped MCP supervisor liveness."""
        return await _call_tool(runtime, {"memory:read"}, health=True)

    return server


def create_mcp_app(
    *,
    runtime: Any,
    token_verifier: Any,
    host: str = "127.0.0.1",
    port: int = 8700,
    path: str = "/mcp",
    public_url: str,
    stateless_http: bool = False,
    json_response: bool = False,
) -> ASGIApp:
    server = create_mcp_server(
        runtime=runtime,
        token_verifier=token_verifier,
        host=host,
        port=port,
        path=path,
        public_url=public_url,
    )
    server.settings.stateless_http = stateless_http
    server.settings.json_response = json_response
    app = server.streamable_http_app()
    app.add_middleware(RequestProvenanceMiddleware)
    return app


def run_mcp_server(
    host: str = "127.0.0.1", port: int = 8700, path: str = "/mcp", public_url: str | None = None
) -> None:
    public_url = public_url or os.getenv("FUSION_MEMORY_MCP_PUBLIC_URL", "").strip()
    if not public_url:
        raise ValueError("FUSION_MEMORY_MCP_PUBLIC_URL is required")
    pepper = os.getenv("FUSION_MEMORY_TOKEN_PEPPER", "").strip()
    if not pepper:
        raise ValueError("FUSION_MEMORY_TOKEN_PEPPER is required")
    runtime, pool = runtime_from_env()
    token_store = PostgresTokenStore(_pooled_connection_factory(pool), pepper=pepper)
    verifier = FusionMemoryTokenVerifier(token_store, pepper=pepper)
    server = create_mcp_server(
        runtime=runtime,
        token_verifier=verifier,
        host=host,
        port=port,
        path=path,
        public_url=public_url,
    )
    server.run(transport="streamable-http")


async def _call_tool(runtime: Any, required_scopes: set[str], **payload: Any) -> dict[str, Any]:
    try:
        access_token = get_access_token()
        if access_token is None or not (access_token.subject or "").strip():
            raise AuthorizationError("authenticated token subject is required")
        if not required_scopes.issubset(set(access_token.scopes)):
            return _error("insufficient_scope", retryable=False)
        user_id = access_token.subject.strip()
        provenance = _request_provenance.get() or RequestProvenance(None, None)
        if payload.pop("write", False):
            scope = Scope(
                user_id=user_id,
                workspace_id=provenance.workspace_id,
                session_id=provenance.session_id,
                app_id="mcp",
            )
        else:
            scope = Scope(user_id=user_id, app_id="mcp")
        if "content" in payload:
            content = _bounded_text(payload["content"], _positive_env("FUSION_MEMORY_MCP_MAX_CONTENT_BYTES", 65536))
            result = await runtime.add(scope, content, payload.get("source"))
        elif payload.pop("answer_context", False):
            result = await runtime.answer_context(scope, _required_text(payload["query"]), _limit(payload["limit"]))
        elif "query" in payload:
            result = await runtime.search(scope, _required_text(payload["query"]), _limit(payload["limit"]))
        else:
            if payload.pop("health", False):
                check = getattr(runtime, "background_health", None)
                result = check() if callable(check) else {"ok": False, "error": "supervisor_state_unavailable"}
                if not isinstance(result, dict):
                    result = {"ok": False, "error": "invalid_supervisor_state"}
                return {"ok": True, "result": {"background": _json_safe(result)}}
            messages = _bounded_messages(payload["messages"])
            batch_id = _required_text(payload["batch_id"])
            metadata = _bounded_batch_metadata(payload.get("metadata"))
            _check_batch_payload_size(messages, batch_id, metadata)
            result = await runtime.add_batch(scope, messages, batch_id, metadata)
        return {"ok": True, "result": _json_safe(result)}
    except BatchIdConflictError:
        return _error("batch_id_conflict", retryable=False)
    except (ValueError, TypeError) as exc:
        return _error("invalid_request", retryable=False, message=str(exc))
    except AuthorizationError:
        return _error("unauthorized", retryable=False)
    except RetryableOperationError as exc:
        return _error(exc.code, retryable=True)
    except PoolAcquireTimeout:
        return _error("pool_acquire_timeout", retryable=True)
    except EndpointUnavailable:
        return _error("model_unavailable", retryable=True)
    except (TimeoutError, ConnectionError, OSError):
        return _error("network_error", retryable=True)
    except PostgresBackendUnavailable:
        return _error("postgres_unavailable", retryable=True)
    except Exception:
        return _error("operation_failed", retryable=True)


def _runtime_lifespan(
    runtime: Any,
    session_manager_lifespan: Any,
    custom_lifespan: Any,
    server: FastMCP,
    custom_state: dict[str, Any],
):
    async def lifespan() -> AsyncIterator[Any]:
        async with AsyncExitStack() as stack:
            try:
                if custom_lifespan is not None:
                    custom_state["value"] = await stack.enter_async_context(custom_lifespan(server))
                else:
                    custom_state["value"] = {}
                manager = getattr(runtime, "lifespan", None)
                if manager is not None:
                    await stack.enter_async_context(manager())
                await stack.enter_async_context(_health_supervisor(runtime))
                await stack.enter_async_context(session_manager_lifespan())
                custom_state["active"] = True
                yield
            finally:
                custom_state["active"] = False
                custom_state.pop("value", None)

    return asynccontextmanager(lifespan)


def _session_lifespan_proxy(custom_state: dict[str, Any]):
    @asynccontextmanager
    async def lifespan(_server: Any) -> AsyncIterator[Any]:
        if not custom_state.get("active"):
            raise RuntimeError("MCP session started outside the application lifespan")
        yield custom_state["value"]

    return lifespan


def _install_runtime_lifespan(server: FastMCP, runtime: Any, custom_lifespan: Any) -> None:
    streamable_http_app = server.streamable_http_app
    installed = False
    custom_state: dict[str, Any] = {}

    def build_app():
        nonlocal installed
        app = streamable_http_app()
        if not installed:
            server.session_manager.run = _runtime_lifespan(
                runtime,
                server.session_manager.run,
                custom_lifespan,
                server,
                custom_state,
            )
            server._mcp_server.lifespan = _session_lifespan_proxy(custom_state)
            installed = True
        return app

    server.streamable_http_app = build_app


@asynccontextmanager
async def _health_supervisor(runtime: Any) -> AsyncIterator[None]:
    pools = tuple(pool for pool in getattr(runtime, "endpoint_pools", ()) if callable(getattr(pool, "healthy_endpoints", None)))
    interval = float(getattr(runtime, "health_check_interval_seconds", 5.0))
    if interval <= 0:
        interval = 5.0
    mark_alive = getattr(runtime, "mark_supervisor_alive", None)
    mark_unhealthy = getattr(runtime, "mark_supervisor_unhealthy", None)
    async with anyio.create_task_group() as task_group:
        if callable(mark_alive):
            mark_alive()
        for pool in pools:
            task_group.start_soon(_supervise_endpoint_pool, pool, interval)
        try:
            yield
        except BaseException as exc:
            if callable(mark_unhealthy):
                mark_unhealthy(exc)
            raise
        finally:
            task_group.cancel_scope.cancel()
            if callable(mark_unhealthy):
                mark_unhealthy()


async def _supervise_endpoint_pool(pool: Any, interval: float) -> None:
    while True:
        try:
            await anyio.to_thread.run_sync(pool.healthy_endpoints, abandon_on_cancel=True)
        except Exception:
            # EndpointPool records probe failures itself; a supervisor must not
            # bring down the MCP lifespan for a transient health-check failure.
            pass
        await anyio.sleep(interval)


def _pooled_connection_factory(pool: Any):
    def factory() -> Any:
        context = pool.connection(5.0)
        connection = context.__enter__()

        class PooledConnection:
            def __init__(self) -> None:
                self._released = False

            def __getattr__(self, name: str) -> Any:
                return getattr(connection, name)

            def close(self) -> None:
                if self._released:
                    return
                self._released = True
                context.__exit__(None, None, None)

        return PooledConnection()

    return factory


def _limit(value: int) -> int:
    if not isinstance(value, int):
        raise ValueError("limit must be an integer")
    return min(MAX_SEARCH_LIMIT, max(1, value))


def _required_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("text must be non-empty")
    return value.strip()


def _bounded_text(value: Any, max_bytes: int) -> str:
    text = _required_text(value)
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("content exceeds configured byte limit")
    return text


def _bounded_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    max_messages = _positive_env("FUSION_MEMORY_MCP_MAX_BATCH_MESSAGES", 100)
    max_bytes = _positive_env("FUSION_MEMORY_MCP_MAX_BATCH_BYTES", 524288)
    if len(value) > max_messages:
        raise ValueError("batch exceeds configured message limit")
    allowed_fields = {"role", "content", "source", "source_uri", "timestamp", "turn_id"}
    messages: list[dict[str, Any]] = []
    total_bytes = 0
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        unexpected = set(message) - allowed_fields
        if unexpected:
            raise ValueError("unsupported message fields")
        if "source" in message and "source_uri" in message:
            raise ValueError("provide only one of source or source_uri")
        content = _required_text(message.get("content"))
        normalized: dict[str, str] = {"content": content}
        for field in ("role", "source_uri", "timestamp", "turn_id"):
            if field not in message:
                continue
            field_value = _history_string(message[field], field)
            if field == "timestamp":
                _validate_history_timestamp(field_value)
            normalized[field] = field_value
        if "source" in message:
            source = _history_string(message["source"], "source")
            if "source_uri" not in normalized:
                normalized["source_uri"] = source
        total_bytes += _json_byte_length(normalized)
        if total_bytes > max_bytes:
            raise ValueError("batch exceeds configured byte limit")
        messages.append(normalized)
    return messages


def _history_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_history_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc


def _bounded_batch_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    protected = {"window_size", "min_window_spans", "chunk_size_tokens", "chunk_overlap_tokens"}
    if protected.intersection(value):
        raise ValueError("metadata cannot configure ingestion limits")
    return dict(value)


def _check_batch_payload_size(messages: list[dict[str, Any]], batch_id: str, metadata: dict[str, Any] | None) -> None:
    max_bytes = _positive_env("FUSION_MEMORY_MCP_MAX_BATCH_BYTES", 524288)
    if _json_byte_length({"messages": messages, "batch_id": batch_id, "metadata": metadata}) > max_bytes:
        raise ValueError("batch exceeds configured byte limit")


def _json_byte_length(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("batch data must be JSON serializable") from exc


def _positive_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _error(code: str, *, retryable: bool, message: str | None = None) -> dict[str, Any]:
    del message
    error: dict[str, Any] = {"code": code, "retryable": retryable}
    return {"ok": False, "error": error}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
