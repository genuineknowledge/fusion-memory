from __future__ import annotations

from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MemoryMcpClient:
    """Reusable authenticated Streamable HTTP MCP client for watcher processes."""

    def __init__(self, url: str, token: str, workspace_id: str, session_id: str, *, timeout_seconds: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.workspace_id = workspace_id
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = await self._ensure_session()
        try:
            result = await session.call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
            )
            return _tool_result(result)
        except Exception:
            await self._reset()
            raise

    async def close(self) -> None:
        await self._reset()

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session
        stack = AsyncExitStack()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Fusion-Memory-Workspace": self.workspace_id,
            "X-Fusion-Memory-Session": self.session_id,
        }
        try:
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(headers=headers, timeout=self.timeout_seconds)
            )
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self.url, http_client=http_client)
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        return session

    async def _reset(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        if stack is not None:
            await stack.aclose()


def _tool_result(result: Any) -> dict[str, Any]:
    if bool(getattr(result, "isError", False) or getattr(result, "is_error", False)):
        return {"ok": False, "isError": True}
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    return {"ok": False, "error": {"code": "unstructured_mcp_result", "retryable": True}}
