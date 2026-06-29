from __future__ import annotations

# ruff: noqa: E402

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

BASE_URL = (
    os.getenv("FUSION_MEMORY_SMOKE_MEMORY_URL")
    or os.getenv("PSI_MEMORY_BASE_URL")
    or "http://127.0.0.1:8700"
)
os.environ.setdefault("PSI_MEMORY_BASE_URL", BASE_URL)

TOOLS_DIR = Path(__file__).resolve().parent / "workspace" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from memory_add import memory_add
from memory_answer_context import memory_answer_context
from memory_search import memory_search


def _has_failure_message(text: str) -> bool:
    lowered = text.lower()
    return '"ok": false' in lowered or "fusion memory is not available" in lowered


def _write_succeeded(write_result: str) -> bool:
    if _has_failure_message(write_result):
        return False
    try:
        payload = json.loads(write_result)
    except json.JSONDecodeError:
        return '"saved": true' in write_result.lower()
    if isinstance(payload, dict):
        if payload.get("saved") is True:
            return True
        result = payload.get("result")
        if isinstance(result, dict):
            return result.get("saved") is True
    return False


async def main() -> int:
    token = f"dolphin-smoke-{uuid4().hex}"
    content = f"Dolphin-Agent Fusion Memory smoke token {token}"

    write_result = await memory_add(content, source="smoke")
    search_result = await memory_search(token, limit=3)
    context_result = await memory_answer_context(token)

    ok = (
        _write_succeeded(write_result)
        and token in search_result
        and not _has_failure_message(search_result)
        and token in context_result
        and not _has_failure_message(context_result)
    )

    print(
        json.dumps(
            {
                "ok": ok,
                "url": BASE_URL,
                "token": token,
                "write": write_result,
                "search": search_result,
                "context": context_result,
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
