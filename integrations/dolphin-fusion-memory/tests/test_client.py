from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest
from aiohttp import web

TOOLS_DIR = Path(__file__).resolve().parents[1] / "workspace" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from _client import format_context_pack, post_json


@pytest.mark.anyio
async def test_post_json_sends_payload_and_returns_json() -> None:
    seen: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(await request.json())
        return web.json_response({"ok": True, "facts": [{"text": "alpha"}]})

    app = web.Application()
    app.router.add_post("/answer-context", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()

    try:
        data = await post_json(
            f"http://127.0.0.1:{port}",
            "/answer-context",
            {"query": "alpha", "scope": {"agent_id": "dolphin"}, "budget": {"limit": 8}},
            2.0,
        )
        assert seen[0]["query"] == "alpha"
        assert data["ok"] is True
        assert format_context_pack(data).startswith("Fusion Memory context:")
    finally:
        await runner.cleanup()
