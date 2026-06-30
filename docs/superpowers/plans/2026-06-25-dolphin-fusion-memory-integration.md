# Dolphin-Agent x Fusion Memory History Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Dolphin-Agent 在不修改 `Dolphin-Agent/src` 的前提下，通过外部 history watcher 将已落盘 session history 同步到 Fusion Memory，同时保留显式 memory tools。

**Architecture:** Dolphin 继续只负责现有 workspace tools 和 `workspace/histories/{session_id}.jsonl`。Fusion Memory 在 memory 仓库内新增 Dolphin history watcher：读取 JSONL、按 user-message 边界做轻量 batch、维护 checkpoint、向 `/ingest-turn` 提交稳定 `turn_id`。显式 `memory_add / memory_search / memory_answer_context` 仍由 Dolphin workspace tools 提供。

**Tech Stack:** Python 3.11+, standard library `json/pathlib/asyncio/hashlib`, `aiohttp` for HTTP tests and workspace tool client, Fusion Memory HTTP server, pytest.

## Global Constraints

- 不修改 Dolphin `src`、不新增 Dolphin 内部 memory client、不改 `SessionAgent.run()`。
- watcher 同步的是已落盘 history JSONL，不保证捕获 Dolphin in-memory 未保存消息。
- 保留显式 `memory_add / memory_search / memory_answer_context` 工具。
- watcher 不过滤 assistant/tool 噪声，不做长期记忆价值判断。
- `/ingest-turn` 默认提交到 `http://127.0.0.1:8700`。
- watcher 必须维护 checkpoint，重启后不重复提交已经确认的 batch。
- Fusion Memory 不可用时，不影响 Dolphin session；watcher 保留未确认 batch 并重试。
- Dolphin 生产运行推荐显式传入 `--session-id`，并设置同值 `PSI_MEMORY_SESSION_ID`。
- 所有改动只落在 memory 仓库；Dolphin Fusion Memory PR 可以关闭或撤销。

## File Map

- Create: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/adapters/dolphin_history_watcher.py`
  - 解析 Dolphin JSONL history。
  - 维护 checkpoint。
  - 生成稳定 turn batches。
  - 提交 `/ingest-turn`。
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/cli.py`
  - 增加 `watch-dolphin-history` 子命令。
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/workspace/systems/system.py`
  - 将提示词从“session auto-persist after each response”改为“external history watcher may sync saved history”。
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/README.md`
  - 增加 watcher 启动流程、session id 要求、降级边界。
- Create: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/tests/test_dolphin_history_watcher.py`
  - 覆盖 parser、batching、checkpoint、HTTP submit、retry。
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/tests/test_tools.py`
  - 覆盖系统提示不再暗示 Dolphin 内部 hook。

---

### Task 1: Add pure history parsing, batching, and checkpoint state

**Files:**
- Create: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/adapters/dolphin_history_watcher.py`
- Create: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/tests/test_dolphin_history_watcher.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class HistoryMessage`
  - `@dataclass(frozen=True) class HistoryBatch`
  - `@dataclass class WatcherCheckpoint`
  - `read_history_messages(path: Path) -> list[HistoryMessage]`
  - `build_batches(messages: list[HistoryMessage], *, session_id: str) -> list[HistoryBatch]`
  - `load_checkpoint(path: Path) -> WatcherCheckpoint`
  - `save_checkpoint(path: Path, checkpoint: WatcherCheckpoint) -> None`
- Consumes:
  - Dolphin JSONL file containing one message dict per line.

- [ ] **Step 1: Write failing tests**

`/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/tests/test_dolphin_history_watcher.py`
```python
from __future__ import annotations

import json
from pathlib import Path

from fusion_memory.adapters.dolphin_history_watcher import (
    WatcherCheckpoint,
    build_batches,
    load_checkpoint,
    read_history_messages,
    save_checkpoint,
)


def test_read_history_messages_skips_blank_lines_and_preserves_line_numbers(tmp_path: Path) -> None:
    history = tmp_path / "session-1.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "first"}),
                "",
                json.dumps({"role": "assistant", "content": "answer"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    messages = read_history_messages(history)

    assert [message.line_number for message in messages] == [1, 3]
    assert [message.data["role"] for message in messages] == ["user", "assistant"]


def test_build_batches_starts_new_batch_on_user_message() -> None:
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "tool result", "name": "lookup"},
        {"role": "user", "content": "second"},
    ]
    path = Path("session-1.jsonl")
    messages = [
        read_message
        for index, item in enumerate(history, start=1)
        for read_message in read_history_messages_from_items(path, index, item)
    ]

    batches = build_batches(messages, session_id="session-1")

    assert len(batches) == 2
    assert [message["role"] for message in batches[0].messages] == ["user", "assistant", "tool"]
    assert [message["role"] for message in batches[1].messages] == ["user"]
    assert batches[0].turn_id.startswith("dolphin:session-1:lines:2-4:")
    assert batches[1].turn_id.startswith("dolphin:session-1:lines:5-5:")


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / ".fusion-memory" / "dolphin-history-watcher" / "session-1.json"
    checkpoint = WatcherCheckpoint(
        history_path="/workspace/histories/session-1.jsonl",
        session_id="session-1",
        line_count=4,
        file_size=123,
        file_mtime_ns=456,
        last_message_hash="abc",
        submitted_batches=["batch-1"],
    )

    save_checkpoint(checkpoint_path, checkpoint)

    assert load_checkpoint(checkpoint_path) == checkpoint


def read_history_messages_from_items(path: Path, line_number: int, item: dict) -> list:
    tmp = path.parent / f".{line_number}.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(item) + "\n", encoding="utf-8")
    message = read_history_messages(tmp)[0]
    return [type(message)(line_number=line_number, data=message.data, raw_hash=message.raw_hash)]
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest tests/test_dolphin_history_watcher.py -k "read_history_messages or build_batches or checkpoint" -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_memory.adapters'`.

- [ ] **Step 3: Implement pure logic**

`/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/adapters/dolphin_history_watcher.py`
```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HistoryMessage:
    line_number: int
    data: dict[str, Any]
    raw_hash: str


@dataclass(frozen=True)
class HistoryBatch:
    turn_id: str
    batch_hash: str
    line_start: int
    line_end: int
    messages: list[dict[str, Any]]


@dataclass
class WatcherCheckpoint:
    history_path: str
    session_id: str
    line_count: int = 0
    file_size: int = 0
    file_mtime_ns: int = 0
    last_message_hash: str | None = None
    submitted_batches: list[str] = field(default_factory=list)


def read_history_messages(path: Path) -> list[HistoryMessage]:
    if not path.exists():
        return []
    messages: list[HistoryMessage] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        messages.append(
            HistoryMessage(
                line_number=line_number,
                data=data,
                raw_hash=_stable_hash(json.dumps(data, ensure_ascii=False, sort_keys=True)),
            )
        )
    return messages


def build_batches(messages: list[HistoryMessage], *, session_id: str) -> list[HistoryBatch]:
    batches: list[list[HistoryMessage]] = []
    current: list[HistoryMessage] = []
    for message in messages:
        role = str(message.data.get("role") or "")
        if role == "system":
            continue
        if role == "user" and current:
            batches.append(current)
            current = [message]
            continue
        if role == "user" or current:
            current.append(message)
    if current:
        batches.append(current)
    return [_batch_from_messages(batch, session_id=session_id) for batch in batches]


def load_checkpoint(path: Path) -> WatcherCheckpoint:
    if not path.exists():
        session_id = path.stem
        return WatcherCheckpoint(history_path="", session_id=session_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    return WatcherCheckpoint(
        history_path=str(data.get("history_path") or ""),
        session_id=str(data.get("session_id") or path.stem),
        line_count=int(data.get("line_count") or 0),
        file_size=int(data.get("file_size") or 0),
        file_mtime_ns=int(data.get("file_mtime_ns") or 0),
        last_message_hash=data.get("last_message_hash"),
        submitted_batches=list(data.get("submitted_batches") or []),
    )


def save_checkpoint(path: Path, checkpoint: WatcherCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _batch_from_messages(messages: list[HistoryMessage], *, session_id: str) -> HistoryBatch:
    line_start = messages[0].line_number
    line_end = messages[-1].line_number
    raw = "\n".join(message.raw_hash for message in messages)
    batch_hash = _stable_hash(raw)[:16]
    return HistoryBatch(
        turn_id=f"dolphin:{session_id}:lines:{line_start}-{line_end}:{batch_hash}",
        batch_hash=batch_hash,
        line_start=line_start,
        line_end=line_end,
        messages=[dict(message.data) for message in messages],
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify pass**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest tests/test_dolphin_history_watcher.py -k "read_history_messages or build_batches or checkpoint" -v
```

Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
git add fusion_memory/adapters/dolphin_history_watcher.py tests/test_dolphin_history_watcher.py
git commit -m "feat(memory): parse Dolphin history watcher batches"
```

### Task 2: Add HTTP submission and retry-safe checkpoint advancement

**Files:**
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/adapters/dolphin_history_watcher.py`
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/tests/test_dolphin_history_watcher.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class WatcherConfig`
  - `async def submit_batch(config: WatcherConfig, batch: HistoryBatch) -> None`
  - `async def sync_history_once(config: WatcherConfig) -> int`
- Consumes:
  - Fusion Memory HTTP endpoint `POST /ingest-turn`.
  - `WatcherCheckpoint.submitted_batches`.

- [ ] **Step 1: Write failing tests**

Append to `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/tests/test_dolphin_history_watcher.py`:
```python
import socket

import pytest
from aiohttp import web

from fusion_memory.adapters.dolphin_history_watcher import WatcherConfig, sync_history_once


@pytest.mark.anyio
async def test_sync_history_once_posts_new_batch_and_advances_checkpoint(tmp_path: Path) -> None:
    seen: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        seen.append(await request.json())
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/ingest-turn", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()

    history = tmp_path / "histories" / "session-1.jsonl"
    history.parent.mkdir()
    history.write_text(
        json.dumps({"role": "user", "content": "remember blue"}) + "\n"
        + json.dumps({"role": "assistant", "content": "stored"}) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / ".fusion-memory" / "dolphin-history-watcher" / "session-1.json"

    try:
        count = await sync_history_once(
            WatcherConfig(
                history_path=history,
                checkpoint_path=checkpoint,
                base_url=f"http://127.0.0.1:{port}",
                workspace_id="ws",
                user_id="u",
                agent_id="dolphin",
                session_id="session-1",
                timeout_seconds=2.0,
            )
        )
    finally:
        await runner.cleanup()

    assert count == 1
    assert seen[0]["turn_id"].startswith("dolphin:session-1:lines:1-2:")
    assert seen[0]["scope"]["session_id"] == "session-1"
    assert seen[0]["metadata"]["source"] == "dolphin-history-watcher"
    assert load_checkpoint(checkpoint).submitted_batches == [seen[0]["metadata"]["batch_hash"]]


@pytest.mark.anyio
async def test_sync_history_once_does_not_advance_checkpoint_when_post_fails(tmp_path: Path) -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"message": "down"}, status=503)

    app = web.Application()
    app.router.add_post("/ingest-turn", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()

    history = tmp_path / "histories" / "session-1.jsonl"
    history.parent.mkdir()
    history.write_text(json.dumps({"role": "user", "content": "remember blue"}) + "\n", encoding="utf-8")
    checkpoint = tmp_path / ".fusion-memory" / "dolphin-history-watcher" / "session-1.json"

    try:
        with pytest.raises(RuntimeError):
            await sync_history_once(
                WatcherConfig(
                    history_path=history,
                    checkpoint_path=checkpoint,
                    base_url=f"http://127.0.0.1:{port}",
                    workspace_id="ws",
                    user_id="u",
                    agent_id="dolphin",
                    session_id="session-1",
                    timeout_seconds=2.0,
                )
            )
    finally:
        await runner.cleanup()

    assert load_checkpoint(checkpoint).submitted_batches == []
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest tests/test_dolphin_history_watcher.py -k "sync_history_once" -v
```

Expected: FAIL with `ImportError` for `WatcherConfig` or `sync_history_once`.

- [ ] **Step 3: Implement submit and checkpoint advancement**

Add to `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/adapters/dolphin_history_watcher.py`:
```python
import aiohttp


@dataclass(frozen=True)
class WatcherConfig:
    history_path: Path
    checkpoint_path: Path
    base_url: str
    workspace_id: str
    user_id: str
    agent_id: str
    session_id: str
    timeout_seconds: float = 2.0
    app_id: str = "dolphin"


async def submit_batch(config: WatcherConfig, batch: HistoryBatch) -> None:
    payload = {
        "messages": batch.messages,
        "scope": {
            "workspace_id": config.workspace_id,
            "user_id": config.user_id,
            "agent_id": config.agent_id,
            "session_id": config.session_id,
            "app_id": config.app_id,
        },
        "turn_id": batch.turn_id,
        "turn_index": None,
        "metadata": {
            "source": "dolphin-history-watcher",
            "history_path": str(config.history_path),
            "line_start": batch.line_start,
            "line_end": batch.line_end,
            "batch_hash": batch.batch_hash,
            "ended_with_error": "unknown",
        },
    }
    timeout = aiohttp.ClientTimeout(total=max(0.1, min(5.0, config.timeout_seconds)))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{config.base_url.rstrip('/')}/ingest-turn", json=payload) as response:
            if response.status >= 400:
                raise RuntimeError(f"Fusion Memory ingest-turn failed with status {response.status}")


async def sync_history_once(config: WatcherConfig) -> int:
    messages = read_history_messages(config.history_path)
    batches = build_batches(messages, session_id=config.session_id)
    checkpoint = load_checkpoint(config.checkpoint_path)
    submitted = set(checkpoint.submitted_batches)
    submitted_count = 0
    for batch in batches:
        if batch.batch_hash in submitted:
            continue
        await submit_batch(config, batch)
        submitted.add(batch.batch_hash)
        checkpoint.submitted_batches.append(batch.batch_hash)
        submitted_count += 1
    stat = config.history_path.stat() if config.history_path.exists() else None
    checkpoint.history_path = str(config.history_path)
    checkpoint.session_id = config.session_id
    checkpoint.line_count = messages[-1].line_number if messages else 0
    checkpoint.file_size = stat.st_size if stat else 0
    checkpoint.file_mtime_ns = stat.st_mtime_ns if stat else 0
    checkpoint.last_message_hash = messages[-1].raw_hash if messages else None
    save_checkpoint(config.checkpoint_path, checkpoint)
    return submitted_count
```

- [ ] **Step 4: Run tests to verify pass**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest tests/test_dolphin_history_watcher.py -k "sync_history_once" -v
```

Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
git add fusion_memory/adapters/dolphin_history_watcher.py tests/test_dolphin_history_watcher.py
git commit -m "feat(memory): sync Dolphin history batches"
```

### Task 3: Add CLI command for the watcher loop

**Files:**
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/adapters/dolphin_history_watcher.py`
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/cli.py`
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/tests/test_dolphin_history_watcher.py`

**Interfaces:**
- Produces:
  - `async def watch_history(config: WatcherConfig, *, poll_interval_seconds: float = 1.0) -> None`
  - CLI: `fusion-memory watch-dolphin-history --workspace <path> --session-id <id>`
- Consumes:
  - env vars `PSI_MEMORY_BASE_URL`, `PSI_MEMORY_WORKSPACE_ID`, `PSI_MEMORY_USER_ID`, `PSI_MEMORY_AGENT_ID`, `PSI_MEMORY_TIMEOUT_SECONDS`.

- [ ] **Step 1: Write failing CLI test**

Append to `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/tests/test_dolphin_history_watcher.py`:
```python
from fusion_memory.adapters.dolphin_history_watcher import config_from_workspace


def test_config_from_workspace_uses_expected_history_and_checkpoint_paths(tmp_path: Path) -> None:
    cfg = config_from_workspace(
        workspace=tmp_path,
        session_id="session-1",
        env={
            "PSI_MEMORY_BASE_URL": "http://127.0.0.1:8700",
            "PSI_MEMORY_WORKSPACE_ID": "ws",
            "PSI_MEMORY_USER_ID": "u",
            "PSI_MEMORY_AGENT_ID": "dolphin",
            "PSI_MEMORY_TIMEOUT_SECONDS": "3",
        },
    )

    assert cfg.history_path == tmp_path / "histories" / "session-1.jsonl"
    assert cfg.checkpoint_path == tmp_path / ".fusion-memory" / "dolphin-history-watcher" / "session-1.json"
    assert cfg.base_url == "http://127.0.0.1:8700"
    assert cfg.workspace_id == "ws"
    assert cfg.timeout_seconds == 3.0
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest tests/test_dolphin_history_watcher.py -k "config_from_workspace" -v
```

Expected: FAIL with `ImportError` for `config_from_workspace`.

- [ ] **Step 3: Implement config builder and watch loop**

Add to `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/adapters/dolphin_history_watcher.py`:
```python
import asyncio
import os
from collections.abc import Mapping


def config_from_workspace(
    *,
    workspace: Path,
    session_id: str,
    env: Mapping[str, str] | None = None,
) -> WatcherConfig:
    env_map = os.environ if env is None else env
    timeout_raw = env_map.get("PSI_MEMORY_TIMEOUT_SECONDS")
    try:
        timeout = float(timeout_raw) if timeout_raw else 2.0
    except ValueError:
        timeout = 2.0
    return WatcherConfig(
        history_path=workspace / "histories" / f"{session_id}.jsonl",
        checkpoint_path=workspace / ".fusion-memory" / "dolphin-history-watcher" / f"{session_id}.json",
        base_url=(env_map.get("PSI_MEMORY_BASE_URL") or "http://127.0.0.1:8700").rstrip("/"),
        workspace_id=env_map.get("PSI_MEMORY_WORKSPACE_ID") or "dolphin",
        user_id=env_map.get("PSI_MEMORY_USER_ID") or env_map.get("USER") or env_map.get("USERNAME") or "user",
        agent_id=env_map.get("PSI_MEMORY_AGENT_ID") or "dolphin",
        session_id=session_id,
        timeout_seconds=max(0.1, min(5.0, timeout)),
    )


async def watch_history(config: WatcherConfig, *, poll_interval_seconds: float = 1.0) -> None:
    while True:
        try:
            await sync_history_once(config)
        except Exception as exc:
            print(f"Fusion Memory Dolphin history watcher skipped sync: {exc}", flush=True)
        await asyncio.sleep(max(0.1, poll_interval_seconds))
```

Modify `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/fusion_memory/cli.py`:
```python
def _add_watch_dolphin_history_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("watch-dolphin-history", help="Sync Dolphin-Agent saved history JSONL into Fusion Memory")
    parser.add_argument("--workspace", required=True, help="Dolphin workspace path")
    parser.add_argument("--session-id", required=True, help="Dolphin session id")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
```

In the existing CLI dispatch, add:
```python
elif args.command == "watch-dolphin-history":
    import asyncio
    from pathlib import Path
    from fusion_memory.adapters.dolphin_history_watcher import config_from_workspace, watch_history

    config = config_from_workspace(workspace=Path(args.workspace), session_id=args.session_id)
    asyncio.run(watch_history(config, poll_interval_seconds=args.poll_interval_seconds))
```

- [ ] **Step 4: Run focused tests**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest tests/test_dolphin_history_watcher.py -k "config_from_workspace" -v
python -m fusion_memory.cli watch-dolphin-history --help
```

Expected: pytest PASS, help output includes `--workspace` and `--session-id`.

- [ ] **Step 5: Commit locally**

```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
git add fusion_memory/adapters/dolphin_history_watcher.py fusion_memory/cli.py tests/test_dolphin_history_watcher.py
git commit -m "feat(memory): add Dolphin history watcher CLI"
```

### Task 4: Update Dolphin integration prompt and docs

**Files:**
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/workspace/systems/system.py`
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/tests/test_tools.py`
- Modify: `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/README.md`

**Interfaces:**
- Produces:
  - System prompt that does not claim Dolphin internal turn hook exists.
  - README with two-process startup: Dolphin session plus memory watcher.
- Consumes:
  - Existing workspace tool names.

- [ ] **Step 1: Write failing prompt test**

Modify `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/tests/test_tools.py`:
```python
@pytest.mark.anyio
async def test_system_prompt_mentions_external_history_watcher_without_internal_hook_claim() -> None:
    prompt = await system.system_prompt_builder()
    assert "memory_add" in prompt
    assert "memory_search" in prompt
    assert "memory_answer_context" in prompt
    assert "history watcher" in prompt
    assert "after each response" not in prompt
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest integrations/dolphin-fusion-memory/tests/test_tools.py -k "history_watcher" -v
```

Expected: FAIL because the prompt still says the current turn may auto-persist after each response.

- [ ] **Step 3: Update system prompt**

`/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/workspace/systems/system.py`
```python
from __future__ import annotations


async def system_prompt_builder() -> str:
    """Build the Dolphin-Agent system prompt for Fusion Memory tools."""
    return (
        "You have access to durable Fusion Memory via three tools:\n"
        "- memory_add: store a stable user preference, project fact, or decision\n"
        "- memory_search: retrieve raw evidence by keyword\n"
        "- memory_answer_context: retrieve a query-grounded context pack\n\n"
        "An external history watcher may sync Dolphin's saved JSONL history into Fusion Memory. "
        "That watcher only sees messages after Dolphin writes them to the history file. "
        "Use memory_add when you intentionally want to store a durable fact or preference. "
        "Use memory_answer_context when answering questions about the user's history, preferences, or prior context. "
        "Use memory_search when you need raw supporting evidence. "
        "Use memory_add only for durable, reusable facts, not transient conversation."
    )
```

- [ ] **Step 4: Update README run instructions**

Add to `/public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd/integrations/dolphin-fusion-memory/README.md`:
````markdown
## Saved-History Watcher

This integration does not patch Dolphin-Agent source code. Automatic persistence
is provided by an external watcher that reads Dolphin's saved history file:

```text
<workspace>/histories/<session-id>.jsonl
```

Start Dolphin with a stable session id:

```bash
export PSI_MEMORY_BASE_URL=http://127.0.0.1:8700
export PSI_MEMORY_SESSION_ID=dolphin-demo

uv run psi-agent session \
  --workspace /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace \
  --session-id dolphin-demo \
  --channel-socket ./channel.sock \
  --ai-socket ./ai.sock
```

Start the watcher in a second shell:

```bash
cd /public/home/wwb/memory
fusion-memory watch-dolphin-history \
  --workspace /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace \
  --session-id dolphin-demo
```

The watcher syncs saved JSONL history only. It cannot see messages that Dolphin
kept only in memory before an error or process exit.
````

- [ ] **Step 5: Run tests and docs smoke**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest integrations/dolphin-fusion-memory/tests/test_tools.py -k "history_watcher" -v
python -m pytest tests/test_dolphin_history_watcher.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit locally**

```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
git add \
  integrations/dolphin-fusion-memory/workspace/systems/system.py \
  integrations/dolphin-fusion-memory/tests/test_tools.py \
  integrations/dolphin-fusion-memory/README.md \
  tests/test_dolphin_history_watcher.py
git commit -m "docs(memory): document Dolphin history watcher integration"
```

### Task 5: Verification and Dolphin PR cleanup

**Files:**
- No production file changes.

**Interfaces:**
- Consumes:
  - All tasks above.
- Produces:
  - Verification evidence.
  - Clear decision that Dolphin Fusion Memory PR is no longer needed.

- [ ] **Step 1: Verify Dolphin source is untouched by this implementation**

Run:
```bash
cd /public/home/wwb/Dolphin-Agent
git diff -- src/psi_agent/session/agent.py src/psi_agent/session/__init__.py
```

Expected: no diff for the watcher-only implementation branch. If an old Fusion Memory PR branch still contains memory hook changes, close or abandon that PR instead of merging it.

- [ ] **Step 2: Run memory watcher tests**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest tests/test_dolphin_history_watcher.py integrations/dolphin-fusion-memory/tests/test_tools.py -v
```

Expected: PASS.

- [ ] **Step 3: Run existing memory integration tests**

Run:
```bash
cd /public/home/wwb/memory/.worktrees/memory-turn-ingestion-sdd
python -m pytest integrations/dolphin-fusion-memory/tests -v
```

Expected: PASS.

- [ ] **Step 4: Report close-PR action**

Record in the handoff or PR comment:

```text
Dolphin Fusion Memory core hook PR is no longer needed. The accepted design uses a memory-side external history watcher that reads Dolphin's existing workspace/histories/{session_id}.jsonl and submits saved history to Fusion Memory. No Dolphin src changes are required.
```
