from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fusion_memory.api.service import MemoryService
from fusion_memory.core.models import Scope
from fusion_memory.core.text import stable_hash
from fusion_memory.core.runtime_config import memory_service_from_env


@dataclass(frozen=True)
class WatcherConfig:
    history_path: Path
    checkpoint_path: Path
    base_url: str
    workspace_id: str
    user_id: str
    agent_id: str
    session_id: str
    db_path: Path | str
    timeout_seconds: float = 2.0
    app_id: str = "haitun"


def config_from_workspace(
    *,
    workspace: Path,
    session_id: str,
    db_path: str | Path = "fusion-memory.sqlite3",
    env: Mapping[str, str] | None = None,
) -> WatcherConfig:
    env_map = os.environ if env is None else env
    timeout = _float(env_map.get("PSI_MEMORY_TIMEOUT_SECONDS"), 2.0)
    return WatcherConfig(
        history_path=workspace / "histories" / f"{session_id}.jsonl",
        checkpoint_path=workspace / ".fusion-memory" / "haitun-history-watcher" / f"{session_id}.json",
        base_url=(env_map.get("PSI_MEMORY_BASE_URL") or "http://127.0.0.1:8700").rstrip("/"),
        workspace_id=env_map.get("PSI_MEMORY_WORKSPACE_ID") or "haitun",
        user_id=env_map.get("PSI_MEMORY_USER_ID") or env_map.get("USER") or env_map.get("USERNAME") or "user",
        agent_id=env_map.get("PSI_MEMORY_AGENT_ID") or "haitun",
        session_id=session_id,
        db_path=db_path,
        timeout_seconds=max(0.1, min(5.0, timeout)),
    )


def sync_history_once(config: WatcherConfig) -> dict[str, Any]:
    messages = _read_history_messages(config.history_path)
    batches = _build_batches(messages, session_id=config.session_id)
    checkpoint = load_checkpoint(config.checkpoint_path)
    submitted = set(checkpoint.get("submitted_batches") or [])
    submitted_count = 0
    service = memory_service_from_env(config.db_path)
    try:
        scope = Scope(
            workspace_id=config.workspace_id,
            user_id=config.user_id,
            agent_id=config.agent_id,
            session_id=config.session_id,
            app_id=config.app_id,
        )
        for batch in batches:
            if batch["batch_hash"] in submitted:
                continue
            service.add(
                {"messages": batch["messages"]},
                scope,
                datetime.now(timezone.utc),
                metadata={
                    "source": "haitun-history-watcher",
                    "history_path": str(config.history_path),
                    "line_start": batch["line_start"],
                    "line_end": batch["line_end"],
                    "batch_hash": batch["batch_hash"],
                    "turn_id": batch["turn_id"],
                    "ended_with_error": "unknown",
                },
            )
            submitted.add(batch["batch_hash"])
            checkpoint.setdefault("submitted_batches", []).append(batch["batch_hash"])
            submitted_count += 1
    finally:
        service.close()

    stat = config.history_path.stat() if config.history_path.exists() else None
    checkpoint.update(
        {
            "history_path": str(config.history_path),
            "session_id": config.session_id,
            "line_count": messages[-1]["line_number"] if messages else 0,
            "file_size": stat.st_size if stat else 0,
            "file_mtime_ns": stat.st_mtime_ns if stat else 0,
            "last_message_hash": messages[-1]["raw_hash"] if messages else None,
        }
    )
    save_checkpoint(config.checkpoint_path, checkpoint)
    return {"ok": True, "submitted_count": submitted_count, "batch_count": len(batches)}


def watch_history(config: WatcherConfig, *, poll_interval_seconds: float = 1.0) -> None:
    while True:
        try:
            sync_history_once(config)
        except Exception as exc:
            print(f"Fusion Memory Haitun history watcher skipped sync: {exc}", flush=True)
        time.sleep(max(0.1, poll_interval_seconds))


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"submitted_batches": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"submitted_batches": []}
    if not isinstance(data, dict):
        return {"submitted_batches": []}
    if not isinstance(data.get("submitted_batches"), list):
        data["submitted_batches"] = []
    return data


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_history_messages(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        role = str(item.get("role") or item.get("speaker") or "user").strip() or "user"
        if not content:
            continue
        out.append({"role": role, "content": content, "line_number": line_number, "raw_hash": stable_hash(raw)})
    return out


def _build_batches(messages: list[dict[str, Any]], *, session_id: str) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    start_line = 0
    end_line = 0
    for message in messages:
        role = message["role"]
        if role == "user" and current:
            batches.append(_batch(current, session_id=session_id, start_line=start_line, end_line=end_line))
            current = []
        if not current:
            start_line = int(message["line_number"])
        end_line = int(message["line_number"])
        current.append({"role": role, "content": message["content"]})
    if current:
        batches.append(_batch(current, session_id=session_id, start_line=start_line, end_line=end_line))
    return batches


def _batch(messages: list[dict[str, Any]], *, session_id: str, start_line: int, end_line: int) -> dict[str, Any]:
    identity = {"session_id": session_id, "line_start": start_line, "line_end": end_line, "messages": messages}
    batch_hash = stable_hash(json.dumps(identity, ensure_ascii=False, sort_keys=True))[:16]
    return {
        "messages": messages,
        "line_start": start_line,
        "line_end": end_line,
        "batch_hash": batch_hash,
        "turn_id": f"haitun:{session_id}:lines:{start_line}-{end_line}:{batch_hash}",
    }


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value else default
    except (TypeError, ValueError):
        return default
