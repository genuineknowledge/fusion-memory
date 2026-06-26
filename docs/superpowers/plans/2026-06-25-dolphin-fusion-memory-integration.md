# Dolphin-Agent × Fusion Memory Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Dolphin-Agent 在保留显式 `memory_add / memory_search / memory_answer_context` 的同时，于每轮结束后把本轮新增 history 自动持久化到 Fusion Memory。

**Architecture:** 先在 Dolphin-Agent 侧补一个极薄的 `FusionMemoryClient`，只负责把 turn delta 通过 HTTP 发给 Fusion Memory 的 `/ingest-turn`。再在 `SessionAgent.run()` 中记录本轮开始前的 history 长度，并在 stop / error / unexpected-finish 三类出口上统一调用非阻断 flush。最后收尾 `memory/integrations/dolphin-fusion-memory/` 的系统提示文案和 Dolphin 端到端测试，保证显式工具与自动持久化同时成立。

**Tech Stack:** Python 3.14, `aiohttp`, `anyio`, Dolphin-Agent `SessionAgent`, Fusion Memory HTTP server, pytest integration tests.

## Global Constraints

- 保留显式 `memory_add / memory_search / memory_answer_context` 工具，让模型仍可主动读写记忆。
- 在 Dolphin session 内增加 turn 结束后的自动持久化。
- 自动持久化的数据源是当前 session history 的“本轮新增 message 列表”。
- Dolphin 不在适配层过滤 assistant/tool 噪声。
- 检索默认允许跨 session 长期视图，但由 memory core 保证 `current session` 优先。
- Dolphin 侧不实现记忆价值判断。
- Dolphin 侧不做 spans 预拆分。
- 自动持久化必须是“非阻断”的。

## File Map

- Create: `/public/home/wwb/Dolphin-Agent/src/psi_agent/session/memory_client.py`
  - 定义 Dolphin 侧 Fusion Memory HTTP client。
  - 读取 `PSI_MEMORY_*` 环境变量，复用现有 integration 配置约定。
- Modify: `/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
  - 为 `SessionAgent` 增加可选 `memory_client`。
  - 在 `create()` 中按环境变量构造 client。
  - 在 `run()` 中计算 turn delta 并非阻断 flush。
- Modify: `/public/home/wwb/Dolphin-Agent/tests/psi_agent/session/test_agent.py`
  - 覆盖 stop / error 两种 turn flush 路径。
- Create: `/public/home/wwb/Dolphin-Agent/tests/psi_agent/session/test_memory_client.py`
  - 覆盖 config 构造与 `/ingest-turn` payload。
- Modify: `/public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace/systems/system.py`
  - 更新系统提示，明确“显式 add + 系统 auto-persist 并存”。
- Modify: `/public/home/wwb/memory/integrations/dolphin-fusion-memory/tests/test_tools.py`
  - 覆盖系统提示文案。
- Create: `/public/home/wwb/Dolphin-Agent/tests/integration/test_fusion_memory_auto_persist.py`
  - 覆盖端到端 turn auto-persist 与 memory down 非阻断降级。

---

### Task 1: Add a minimal Dolphin-side Fusion Memory client

**Files:**
- Create: `/public/home/wwb/Dolphin-Agent/src/psi_agent/session/memory_client.py`
- Create: `/public/home/wwb/Dolphin-Agent/tests/psi_agent/session/test_memory_client.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class MemoryClientConfig`
  - `build_memory_client_config(env: Mapping[str, str] | None = None) -> MemoryClientConfig`
  - `class FusionMemoryClient`
  - `async def ingest_turn(self, messages: list[dict[str, Any]], *, turn_id: str | None, turn_index: int | None, ended_with_error: bool) -> None`
- Consumes:
  - Fusion Memory HTTP endpoint `/ingest-turn`
  - env vars `PSI_MEMORY_BASE_URL`, `PSI_MEMORY_TIMEOUT_SECONDS`, `PSI_MEMORY_WORKSPACE_ID`, `PSI_MEMORY_USER_ID`, `PSI_MEMORY_AGENT_ID`, `PSI_MEMORY_SESSION_ID`

- [ ] **Step 1: Write the failing tests**

`/public/home/wwb/Dolphin-Agent/tests/psi_agent/session/test_memory_client.py`
```python
from __future__ import annotations

import socket

import pytest
from aiohttp import web

from psi_agent.session.memory_client import FusionMemoryClient, MemoryClientConfig, build_memory_client_config


def test_build_memory_client_config_reads_expected_env_keys() -> None:
    config = build_memory_client_config(
        {
            "PSI_MEMORY_BASE_URL": "http://127.0.0.1:8700",
            "PSI_MEMORY_TIMEOUT_SECONDS": "3.5",
            "PSI_MEMORY_WORKSPACE_ID": "ws",
            "PSI_MEMORY_USER_ID": "u",
            "PSI_MEMORY_AGENT_ID": "agent",
            "PSI_MEMORY_SESSION_ID": "session-1",
        }
    )

    assert config.base_url == "http://127.0.0.1:8700"
    assert config.timeout_seconds == 3.5
    assert config.workspace_id == "ws"
    assert config.user_id == "u"
    assert config.agent_id == "agent"
    assert config.session_id == "session-1"


@pytest.mark.anyio
async def test_ingest_turn_posts_messages_scope_and_error_flag() -> None:
    seen: dict[str, object] = {}

    async def handler(request: web.Request) -> web.Response:
        seen.update(await request.json())
        return web.json_response({"span_ids": ["span-1"]})

    app = web.Application()
    app.router.add_post("/ingest-turn", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()

    client = FusionMemoryClient(
        MemoryClientConfig(
            base_url=f"http://127.0.0.1:{port}",
            timeout_seconds=2.0,
            workspace_id="ws",
            user_id="u",
            agent_id="agent",
            session_id="session-1",
        )
    )

    try:
        await client.ingest_turn(
            [{"role": "user", "content": "remember my aisle seat preference"}],
            turn_id="turn-1",
            turn_index=1,
            ended_with_error=False,
        )
        assert seen["messages"] == [{"role": "user", "content": "remember my aisle seat preference"}]
        assert seen["scope"]["workspace_id"] == "ws"
        assert seen["scope"]["session_id"] == "session-1"
        assert seen["metadata"]["ended_with_error"] is False
    finally:
        await runner.cleanup()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /public/home/wwb/Dolphin-Agent
PYTHONPATH=/public/home/wwb/Dolphin-Agent/src ./.venv/bin/python -m pytest tests/psi_agent/session/test_memory_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'psi_agent.session.memory_client'`.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/memory_client.py`
```python
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import aiohttp


@dataclass(frozen=True)
class MemoryClientConfig:
    base_url: str
    timeout_seconds: float
    workspace_id: str
    user_id: str
    agent_id: str
    session_id: str | None


def build_memory_client_config(env: Mapping[str, str] | None = None) -> MemoryClientConfig:
    env = os.environ if env is None else env
    return MemoryClientConfig(
        base_url=(env.get("PSI_MEMORY_BASE_URL") or "http://127.0.0.1:8700").rstrip("/"),
        timeout_seconds=float(env.get("PSI_MEMORY_TIMEOUT_SECONDS") or "2.0"),
        workspace_id=env.get("PSI_MEMORY_WORKSPACE_ID") or "dolphin",
        user_id=env.get("PSI_MEMORY_USER_ID") or env.get("USER") or "user",
        agent_id=env.get("PSI_MEMORY_AGENT_ID") or "dolphin",
        session_id=env.get("PSI_MEMORY_SESSION_ID") or None,
    )


class FusionMemoryClient:
    def __init__(self, config: MemoryClientConfig) -> None:
        self._config = config

    async def ingest_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        turn_id: str | None,
        turn_index: int | None,
        ended_with_error: bool,
    ) -> None:
        payload = {
            "messages": messages,
            "scope": {
                "workspace_id": self._config.workspace_id,
                "user_id": self._config.user_id,
                "agent_id": self._config.agent_id,
                "session_id": self._config.session_id,
                "app_id": "dolphin",
            },
            "turn_id": turn_id,
            "turn_index": turn_index,
            "metadata": {"ended_with_error": ended_with_error},
        }
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self._config.base_url}/ingest-turn", json=payload) as response:
                response.raise_for_status()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cd /public/home/wwb/Dolphin-Agent
PYTHONPATH=/public/home/wwb/Dolphin-Agent/src ./.venv/bin/python -m pytest tests/psi_agent/session/test_memory_client.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/Dolphin-Agent
git add src/psi_agent/session/memory_client.py tests/psi_agent/session/test_memory_client.py
git commit -m "feat(session): add Fusion Memory client"
```

### Task 2: Hook non-blocking turn-delta persistence into `SessionAgent.run()`

**Files:**
- Modify: `/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
- Modify: `/public/home/wwb/Dolphin-Agent/tests/psi_agent/session/test_agent.py`

**Interfaces:**
- Produces:
  - `SessionAgent(..., memory_client: FusionMemoryClient | None = None)`
  - `self._turn_index: int`
  - `async def _flush_turn_memory(self, history_len_before: int, *, turn_index: int, ended_with_error: bool) -> None`
- Consumes:
  - `FusionMemoryClient.ingest_turn(...)`
  - existing `history` append order in `SessionAgent.run()`

- [ ] **Step 1: Write the failing tests**

`/public/home/wwb/Dolphin-Agent/tests/psi_agent/session/test_agent.py`
```python
@pytest.mark.anyio
async def test_turn_delta_is_flushed_on_stop(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class FakeMemoryClient:
        async def ingest_turn(self, messages, *, turn_id, turn_index, ended_with_error):
            seen["messages"] = messages
            seen["turn_index"] = turn_index
            seen["ended_with_error"] = ended_with_error

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, reason="OK", headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(_sse_chunk(content="stored", finish="stop").encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    mock_server = MockAIServer(tmp_path)
    ai_socket = await mock_server.start(handler)
    try:
        agent = SessionAgent(ai_socket=ai_socket, tools={}, memory_client=FakeMemoryClient())
        async for _ in agent.run({"role": "user", "content": "remember my seat preference"}):
            pass

        assert [item["role"] for item in seen["messages"]] == ["user", "assistant"]
        assert seen["turn_index"] == 1
        assert seen["ended_with_error"] is False
    finally:
        await mock_server.cleanup()


@pytest.mark.anyio
async def test_turn_delta_is_flushed_on_error(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class FakeMemoryClient:
        async def ingest_turn(self, messages, *, turn_id, turn_index, ended_with_error):
            seen["messages"] = messages
            seen["turn_index"] = turn_index
            seen["ended_with_error"] = ended_with_error

    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=500)

    mock_server = MockAIServer(tmp_path)
    ai_socket = await mock_server.start(handler)
    try:
        agent = SessionAgent(ai_socket=ai_socket, tools={}, memory_client=FakeMemoryClient())
        async for _ in agent.run({"role": "user", "content": "danger"}):
            pass

        assert [item["role"] for item in seen["messages"]] == ["user"]
        assert seen["turn_index"] == 1
        assert seen["ended_with_error"] is True
    finally:
        await mock_server.cleanup()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /public/home/wwb/Dolphin-Agent
PYTHONPATH=/public/home/wwb/Dolphin-Agent/src ./.venv/bin/python -m pytest tests/psi_agent/session/test_agent.py -k "turn_delta_is_flushed" -v
```

Expected: FAIL because `SessionAgent.__init__` has no `memory_client` parameter and there is no flush hook.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
from psi_agent.session.memory_client import FusionMemoryClient, build_memory_client_config
```

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
def __init__(
    self,
    *,
    ai_socket: str,
    tools: dict[str, ToolFunction],
    tool_funcs: dict[str, Callable[..., Any]] | None = None,
    schedules: list | None = None,
    system_prompt_builder: Callable[..., Any] | None = None,
    max_tool_rounds: int = 128,
    history: list[dict] | None = None,
    history_path: Path | None = None,
    memory_client: FusionMemoryClient | None = None,
) -> None:
    self.ai_socket = ai_socket
    self.tools = tools
    self._tool_funcs = tool_funcs if tool_funcs else {}
    self.schedules = schedules if schedules is not None else []
    self._system_prompt_builder = system_prompt_builder
    self.max_tool_rounds = max_tool_rounds
    self.history = history if history is not None else []
    self._history_path = history_path
    self._pending_schedule_chunks = []
    self._memory_client = memory_client
    self._turn_index = 0
```

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
memory_client = None
try:
    memory_client = FusionMemoryClient(build_memory_client_config())
except Exception as exc:
    logger.warning(f"Fusion Memory client disabled during session startup: {exc}")

return cls(
    ai_socket=ai_socket,
    tools=tools,
    tool_funcs=tool_funcs,
    schedules=schedules,
    system_prompt_builder=_load_system_prompt_builder(workspace_path),
    max_tool_rounds=max_tool_rounds,
    history=history,
    history_path=history_path,
    memory_client=memory_client,
)
```

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
self._turn_index += 1
turn_index = self._turn_index
history_len_before = len(self.history)
self.history.append(user_message)
```

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
if finish_reason == "error":
    await self._flush_turn_memory(history_len_before, turn_index=turn_index, ended_with_error=True)
    return

if finish_reason == "stop":
    if accumulated_content or accumulated_reasoning:
        assistant_msg: dict = {"role": "assistant"}
        if accumulated_content:
            assistant_msg["content"] = accumulated_content
        if accumulated_reasoning:
            assistant_msg["reasoning_content"] = accumulated_reasoning
        self.history.append(assistant_msg)
        if self._history_path is not None:
            await _save_history(self._history_path, self.history)
    await self._flush_turn_memory(history_len_before, turn_index=turn_index, ended_with_error=False)
    return
```

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
if finish_reason not in ("error", "stop", "tool_calls"):
    if accumulated_content or accumulated_reasoning:
        assistant_msg: dict = {"role": "assistant"}
        if accumulated_content:
            assistant_msg["content"] = accumulated_content
        if accumulated_reasoning:
            assistant_msg["reasoning_content"] = accumulated_reasoning
        self.history.append(assistant_msg)
    await self._flush_turn_memory(history_len_before, turn_index=turn_index, ended_with_error=False)
    return
```

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
async def _flush_turn_memory(
    self,
    history_len_before: int,
    *,
    turn_index: int,
    ended_with_error: bool,
) -> None:
    if self._memory_client is None:
        return
    delta = self.history[history_len_before:]
    if not delta:
        return
    try:
        await self._memory_client.ingest_turn(
            delta,
            turn_id=None,
            turn_index=turn_index,
            ended_with_error=ended_with_error,
        )
    except Exception as exc:
        logger.warning(f"Fusion Memory auto-persist skipped: {exc}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cd /public/home/wwb/Dolphin-Agent
PYTHONPATH=/public/home/wwb/Dolphin-Agent/src ./.venv/bin/python -m pytest tests/psi_agent/session/test_agent.py -k "turn_delta_is_flushed" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/Dolphin-Agent
git add src/psi_agent/session/agent.py tests/psi_agent/session/test_agent.py
git commit -m "feat(session): auto-persist turn deltas to Fusion Memory"
```

### Task 3: Update the explicit-tool system prompt to mention auto persistence

**Files:**
- Modify: `/public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace/systems/system.py`
- Modify: `/public/home/wwb/memory/integrations/dolphin-fusion-memory/tests/test_tools.py`

**Interfaces:**
- Produces:
  - system prompt copy that explicitly distinguishes:
    - explicit `memory_add`
    - automatic turn auto-persist
- Consumes:
  - existing public tool names `memory_add`, `memory_search`, `memory_answer_context`

- [ ] **Step 1: Write the failing test**

`/public/home/wwb/memory/integrations/dolphin-fusion-memory/tests/test_tools.py`
```python
@pytest.mark.anyio
async def test_system_prompt_mentions_explicit_add_and_auto_persist() -> None:
    prompt = await system.system_prompt_builder()
    assert "memory_add" in prompt
    assert "auto-persist" in prompt
    assert "memory_search" in prompt
    assert "memory_answer_context" in prompt
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest memory/integrations/dolphin-fusion-memory/tests/test_tools.py -k "explicit_add_and_auto_persist" -v
```

Expected: FAIL because the current prompt only describes the three explicit tools and does not mention session auto-persist.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace/systems/system.py`
```python
async def system_prompt_builder() -> str:
    return (
        "You have access to durable Fusion Memory via three explicit tools:\n"
        "- memory_add: store a stable user preference, project fact, or decision\n"
        "- memory_search: retrieve raw evidence by keyword\n"
        "- memory_answer_context: retrieve a query-grounded context pack\n\n"
        "The session may also auto-persist the current turn's raw history after each turn. "
        "Use memory_add when you intentionally want to promote a durable fact or preference. "
        "Use memory_answer_context when answering questions about the user's history, preferences, or prior context. "
        "Use memory_search when you need raw supporting evidence."
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest memory/integrations/dolphin-fusion-memory/tests/test_tools.py -k "explicit_add_and_auto_persist" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/memory
git add \
  memory/integrations/dolphin-fusion-memory/workspace/systems/system.py \
  memory/integrations/dolphin-fusion-memory/tests/test_tools.py
git commit -m "docs(memory): clarify explicit add and auto persist prompt"
```

### Task 4: Add end-to-end coverage for auto persistence and non-blocking degradation

**Files:**
- Create: `/public/home/wwb/Dolphin-Agent/tests/integration/test_fusion_memory_auto_persist.py`

**Interfaces:**
- Produces regression coverage for:
  - stop-path auto persistence
  - memory-server failure non-blocking behavior
- Consumes:
  - `tests.integration.conftest._psi_process_spec`
  - `tests.integration.conftest.read_sse`
  - `FusionMemoryClient.ingest_turn(...)`

- [ ] **Step 1: Write the failing tests**

`/public/home/wwb/Dolphin-Agent/tests/integration/test_fusion_memory_auto_persist.py`
```python
from __future__ import annotations

import socket
from pathlib import Path

import anyio
import pytest
from aiohttp import web

from tests.integration.conftest import _psi_process_spec, read_sse
from tests.integration.test_end_to_end import _chunk, _stop_process, _wait_for_socket


async def _start_memory_server(status: int, seen: list[dict]) -> tuple[web.AppRunner, int]:
    async def handler(request: web.Request) -> web.Response:
        seen.append(await request.json())
        return web.json_response({"span_ids": ["span-1"]}, status=status)

    app = web.Application()
    app.router.add_post("/ingest-turn", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()
    return runner, port


@pytest.mark.anyio
async def test_turn_auto_persist_posts_new_messages_without_blocking(
    tmp_path: Path,
    mock_ai_server,
) -> None:
    seen: list[dict] = []
    memory_runner, memory_port = await _start_memory_server(200, seen)
    mock_ai_server.set_responses([_chunk(content="stored", finish_reason="stop")])
    base_url = await mock_ai_server.start()

    ai_socket = str(tmp_path / "ai.sock")
    channel_socket = str(tmp_path / "channel.sock")

    ai_cmd, ai_env, ai_cwd = _psi_process_spec(
        "ai",
        "--provider",
        "openai",
        "--session-socket",
        ai_socket,
        "--model",
        "test",
        "--api-key",
        "k",
        "--base-url",
        base_url,
    )
    ai_proc = await anyio.open_process(ai_cmd, env=ai_env, cwd=str(ai_cwd))

    ses_cmd, ses_env, ses_cwd = _psi_process_spec(
        "session",
        "--workspace",
        "examples/a-simple-schedule-workspace",
        "--channel-socket",
        channel_socket,
        "--ai-socket",
        ai_socket,
    )
    ses_env = dict(ses_env or {})
    ses_env.update(
        {
            "PSI_MEMORY_BASE_URL": f"http://127.0.0.1:{memory_port}",
            "PSI_MEMORY_WORKSPACE_ID": "ws",
            "PSI_MEMORY_USER_ID": "u",
            "PSI_MEMORY_AGENT_ID": "dolphin",
            "PSI_MEMORY_SESSION_ID": "session-1",
        }
    )
    ses_proc = await anyio.open_process(ses_cmd, env=ses_env, cwd=str(ses_cwd))

    try:
        assert await _wait_for_socket(ai_socket)
        assert await _wait_for_socket(channel_socket)
        chunks = await read_sse(channel_socket, "remember my aisle seat preference")
        assert chunks
        assert seen
        assert seen[0]["messages"][0]["role"] == "user"
        assert seen[0]["messages"][0]["content"] == "remember my aisle seat preference"
    finally:
        await _stop_process(ses_proc)
        await _stop_process(ai_proc)
        await memory_runner.cleanup()


@pytest.mark.anyio
async def test_turn_auto_persist_failure_does_not_fail_session(
    tmp_path: Path,
    mock_ai_server,
) -> None:
    seen: list[dict] = []
    memory_runner, memory_port = await _start_memory_server(500, seen)
    mock_ai_server.set_responses([_chunk(content="still answered", finish_reason="stop")])
    base_url = await mock_ai_server.start()

    ai_socket = str(tmp_path / "ai.sock")
    channel_socket = str(tmp_path / "channel.sock")

    ai_cmd, ai_env, ai_cwd = _psi_process_spec(
        "ai",
        "--provider",
        "openai",
        "--session-socket",
        ai_socket,
        "--model",
        "test",
        "--api-key",
        "k",
        "--base-url",
        base_url,
    )
    ai_proc = await anyio.open_process(ai_cmd, env=ai_env, cwd=str(ai_cwd))

    ses_cmd, ses_env, ses_cwd = _psi_process_spec(
        "session",
        "--workspace",
        "examples/a-simple-schedule-workspace",
        "--channel-socket",
        channel_socket,
        "--ai-socket",
        ai_socket,
    )
    ses_env = dict(ses_env or {})
    ses_env.update(
        {
            "PSI_MEMORY_BASE_URL": f"http://127.0.0.1:{memory_port}",
            "PSI_MEMORY_WORKSPACE_ID": "ws",
            "PSI_MEMORY_USER_ID": "u",
            "PSI_MEMORY_AGENT_ID": "dolphin",
            "PSI_MEMORY_SESSION_ID": "session-1",
        }
    )
    ses_proc = await anyio.open_process(ses_cmd, env=ses_env, cwd=str(ses_cwd))

    try:
        assert await _wait_for_socket(ai_socket)
        assert await _wait_for_socket(channel_socket)
        chunks = await read_sse(channel_socket, "hello")
        text = "".join(chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") for chunk in chunks)
        assert "still answered" in text
        assert seen
    finally:
        await _stop_process(ses_proc)
        await _stop_process(ai_proc)
        await memory_runner.cleanup()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /public/home/wwb/Dolphin-Agent
PATH=/public/home/wwb/Dolphin-Agent/.venv/bin:$PATH ./.venv/bin/python -m pytest tests/integration/test_fusion_memory_auto_persist.py -v
```

Expected: FAIL because the session does not yet attempt `/ingest-turn`.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/Dolphin-Agent/src/psi_agent/session/agent.py`
```python
# No additional production files beyond Task 2 should be needed here.
# The implementation work for this task is the integration wiring already added:
# - SessionAgent.create() builds FusionMemoryClient from PSI_MEMORY_* env
# - SessionAgent.run() flushes turn deltas on stop/error/unexpected-finish
# - _flush_turn_memory() swallows memory errors and logs them
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cd /public/home/wwb/Dolphin-Agent
PATH=/public/home/wwb/Dolphin-Agent/.venv/bin:$PATH ./.venv/bin/python -m pytest tests/integration/test_fusion_memory_auto_persist.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/Dolphin-Agent
git add tests/integration/test_fusion_memory_auto_persist.py
git commit -m "test(session): cover Fusion Memory auto persistence"
```
