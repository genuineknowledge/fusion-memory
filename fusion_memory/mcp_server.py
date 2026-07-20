from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope as ASGIScope, Send

from fusion_memory.core.auth import AuthorizationError
from fusion_memory.core.models import Scope
from fusion_memory.mcp_auth import FusionMemoryTokenVerifier
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.mcp_runtime import FusionMemoryRuntime, runtime_from_env
from fusion_memory.storage.postgres_pool import PoolAcquireTimeout, RetryableOperationError
from fusion_memory.storage.postgres_store import PostgresBackendUnavailable
from fusion_memory.storage.token_store import PostgresTokenStore


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
    lifespan = lifespan or _runtime_lifespan(runtime)
    public_host = urlsplit(public_url).netloc
    allowed_hosts = [f"{host}:*", host, public_host]
    allowed_origins = [public_url.rstrip("/")]
    server = FastMCP(
        "Fusion Memory",
        host=host,
        port=port,
        streamable_http_path=path,
        token_verifier=token_verifier,
        auth=AuthSettings(issuer_url=public_url, resource_server_url=public_url),
        lifespan=lifespan,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        ),
    )

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
            messages = _bounded_messages(payload["messages"])
            result = await runtime.add_batch(scope, messages, _required_text(payload["batch_id"]), payload.get("metadata"))
        return {"ok": True, "result": _json_safe(result)}
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


def _runtime_lifespan(runtime: Any):
    async def lifespan(_server: FastMCP) -> AsyncIterator[Any]:
        manager = getattr(runtime, "lifespan", None)
        if manager is None:
            yield runtime
        else:
            async with manager():
                yield runtime

    from contextlib import asynccontextmanager

    return asynccontextmanager(lifespan)


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
    messages: list[dict[str, Any]] = []
    total_bytes = 0
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        content = _required_text(message.get("content"))
        total_bytes += len(content.encode("utf-8"))
        if total_bytes > max_bytes:
            raise ValueError("batch exceeds configured byte limit")
        messages.append(dict(message, content=content))
    return messages


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
