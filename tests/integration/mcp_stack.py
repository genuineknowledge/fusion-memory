from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import ssl
import ipaddress
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx
from aiohttp import web
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


@dataclass(frozen=True)
class E2EConfig:
    url: str
    token_a: str = field(repr=False)
    token_b: str = field(repr=False)
    ca_file: str | None = None


class MissingIntegrationConfig(RuntimeError):
    """Required operator configuration was not supplied."""


class DeployedMcpClient:
    def __init__(self, config: E2EConfig, *, token: str, workspace_id: str, session_id: str) -> None:
        self.config = config
        self.token = token
        self.workspace_id = workspace_id
        self.session_id = session_id

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return anyio.run(self._call, name, arguments)

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Fusion-Memory-Workspace": self.workspace_id,
            "X-Fusion-Memory-Session": self.session_id,
        }
        verify: bool | ssl.SSLContext = (
            ssl.create_default_context(cafile=self.config.ca_file) if self.config.ca_file else True
        )
        async with httpx.AsyncClient(headers=headers, timeout=30.0, verify=verify) as http_client:
            async with streamable_http_client(self.config.url, http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    structured = getattr(result, "structuredContent", None)
                    if isinstance(structured, dict):
                        return structured
                    return {
                        "ok": not bool(getattr(result, "isError", False)),
                        "content": [getattr(block, "text", str(block)) for block in getattr(result, "content", [])],
                    }


class DeployedMcpStack:
    def __init__(self, config: E2EConfig | None = None) -> None:
        if config is None:
            url = os.environ.get("FUSION_MEMORY_E2E_URL", "").strip()
            token_a = os.environ.get("FUSION_MEMORY_E2E_TOKEN_A", "").strip()
            token_b = os.environ.get("FUSION_MEMORY_E2E_TOKEN_B", "").strip()
            ca_file = os.environ.get("FUSION_MEMORY_E2E_CA_FILE", "").strip() or None
            if url:
                _validate_e2e_url(url)
            if ca_file and not os.path.isfile(ca_file):
                raise ValueError("FUSION_MEMORY_E2E_CA_FILE does not exist")
            if not url or not token_a or not token_b:
                raise MissingIntegrationConfig(
                    "set FUSION_MEMORY_E2E_URL, FUSION_MEMORY_E2E_TOKEN_A, and "
                    "FUSION_MEMORY_E2E_TOKEN_B before integration tests"
                )
            config = E2EConfig(url=url, token_a=token_a, token_b=token_b, ca_file=ca_file)
        _validate_e2e_url(config.url)
        self.config = config

    def client(self, *, user: str, workspace: str = "ws", session: str) -> DeployedMcpClient:
        if user == "a":
            token = self.config.token_a
        elif user == "b":
            token = self.config.token_b
        else:
            raise ValueError("user must be 'a' or 'b'")
        return DeployedMcpClient(
            self.config,
            token=token,
            workspace_id=workspace,
            session_id=session,
        )

    def reset_worker_stats(self) -> None:
        for url in self.worker_stats_urls():
            response = httpx.post(f"{url}/stats/reset", timeout=5.0)
            response.raise_for_status()

    def assert_worker_overlap(self, expected_texts: tuple[str, str]) -> None:
        expected_tags = tuple(hashlib.sha256(text.encode()).hexdigest() for text in expected_texts)
        intervals: list[tuple[str, float, float, set[str]]] = []
        for url in self.worker_stats_urls():
            response = httpx.get(f"{url}/stats", timeout=5.0)
            response.raise_for_status()
            for record in response.json().get("requests", []):
                if record.get("path") == "/v1/embeddings":
                    intervals.append(
                        (
                            url,
                            float(record["started"]),
                            float(record["finished"]),
                            set(record.get("tags", [])),
                        )
                    )
        for left_worker, left_start, left_end, left_tags in intervals:
            if expected_tags[0] not in left_tags:
                continue
            for right_worker, right_start, right_end, right_tags in intervals:
                if (
                    expected_tags[1] in right_tags
                    and left_worker != right_worker
                    and max(left_start, right_start) < min(left_end, right_end)
                ):
                    return
        raise AssertionError("the two user-correlated embedding requests did not overlap across fake workers")

    @staticmethod
    def worker_stats_urls() -> list[str]:
        value = os.environ.get("FUSION_MEMORY_E2E_WORKER_STATS_URLS", "")
        urls = [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
        if len(urls) < 2:
            raise MissingIntegrationConfig("set FUSION_MEMORY_E2E_WORKER_STATS_URLS to at least two fake worker URLs")
        return urls


def _validate_e2e_url(url: str) -> None:
    try:
        parts = urlsplit(url)
        _ = parts.port
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("FUSION_MEMORY_E2E_URL is invalid") from exc
    if not parts.hostname or parts.path != "/mcp" or parts.username is not None or parts.password is not None:
        raise ValueError("FUSION_MEMORY_E2E_URL must use exact /mcp without credentials")
    if parts.query or parts.fragment:
        raise ValueError("FUSION_MEMORY_E2E_URL must not contain query or fragment")
    loopback = parts.hostname.lower() == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        pass
    if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
        raise ValueError("FUSION_MEMORY_E2E_URL must use HTTPS except for loopback development")


class _FakeWorkerState:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.requests: list[dict[str, Any]] = []
        self.fail_embeddings = False

    async def record(self, path: str, response_factory: Any, *, tags: list[str] | None = None) -> web.Response:
        started = time.monotonic()
        await asyncio.sleep(self.delay_seconds)
        response = await response_factory()
        self.requests.append(
            {"path": path, "started": started, "finished": time.monotonic(), "tags": list(tags or [])}
        )
        return response


def create_fake_worker_app(*, delay_seconds: float = 0.0) -> web.Application:
    state = _FakeWorkerState(delay_seconds)
    app = web.Application()

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def embeddings(request: web.Request) -> web.Response:
        payload = await request.json()
        texts = payload.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        dimensions = int(payload.get("dimensions") or 1024)

        async def response() -> web.Response:
            if state.fail_embeddings:
                return web.json_response({"error": "configured inference failure"}, status=500)
            vectors = []
            for index, _text in enumerate(texts):
                vector = [0.0] * dimensions
                vector[index % dimensions] = 1.0
                vectors.append(vector)
            return web.json_response({"data": [{"embedding": vector} for vector in vectors]})

        tags = [hashlib.sha256(str(text).encode()).hexdigest() for text in texts]
        return await state.record("/v1/embeddings", response, tags=tags)

    async def rerank(request: web.Request) -> web.Response:
        payload = await request.json()
        documents = payload.get("documents", [])

        async def response() -> web.Response:
            return web.json_response({"scores": [1.0 / (index + 1) for index, _ in enumerate(documents)]})

        return await state.record("/v1/rerank", response)

    async def stats(_request: web.Request) -> web.Response:
        return web.json_response({"requests": list(state.requests)})

    async def reset_stats(_request: web.Request) -> web.Response:
        state.requests.clear()
        return web.json_response({"ok": True})

    async def configure_failure(request: web.Request) -> web.Response:
        state.fail_embeddings = bool((await request.json()).get("enabled"))
        return web.json_response({"ok": True, "enabled": state.fail_embeddings})

    app.router.add_get("/health", health)
    app.router.add_post("/v1/embeddings", embeddings)
    app.router.add_post("/v1/rerank", rerank)
    app.router.add_get("/stats", stats)
    app.router.add_post("/stats/reset", reset_stats)
    app.router.add_post("/fail/embeddings", configure_failure)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-worker", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if not args.fake_worker:
        parser.error("--fake-worker is required")
    web.run_app(
        create_fake_worker_app(delay_seconds=max(0.0, args.delay_seconds)),
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    main()
