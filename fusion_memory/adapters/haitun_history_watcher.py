from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import request

from fusion_memory.core.text import stable_hash


@dataclass(frozen=True)
class WatcherConfig:
    workspace: Path | None
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
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> WatcherConfig:
    env_map = os.environ if env is None else env
    timeout = _float(env_map.get("PSI_MEMORY_TIMEOUT_SECONDS"), 2.0)
    return WatcherConfig(
        workspace=workspace,
        history_path=workspace / "histories" / f"{session_id}.jsonl",
        checkpoint_path=workspace / ".fusion-memory" / "haitun-history-watcher" / f"{session_id}.json",
        base_url=(base_url or env_map.get("PSI_MEMORY_BASE_URL") or "http://127.0.0.1:8700").rstrip("/"),
        workspace_id=env_map.get("PSI_MEMORY_WORKSPACE_ID") or "haitun",
        user_id=env_map.get("PSI_MEMORY_USER_ID") or env_map.get("USER") or env_map.get("USERNAME") or "user",
        agent_id=env_map.get("PSI_MEMORY_AGENT_ID") or "haitun",
        session_id=session_id,
        db_path=db_path,
        timeout_seconds=max(0.1, min(5.0, timeout)),
    )


def sync_history_once(
    config: WatcherConfig,
    *,
    submit_add: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    messages = _read_history_messages(config.history_path)
    batches = _build_batches(messages, session_id=config.session_id)
    checkpoint = load_checkpoint(config.checkpoint_path)
    submitted = set(checkpoint.get("submitted_batches") or [])
    submitted_count = 0
    submit = submit_add or (lambda payload: _post_add(config, payload))
    for batch in batches:
        if batch["batch_hash"] in submitted:
            continue
        payload = {
            "input": {"messages": batch["messages"]},
            "scope": {
                "workspace_id": config.workspace_id,
                "user_id": config.user_id,
                "agent_id": config.agent_id,
                "session_id": config.session_id,
                "app_id": config.app_id,
            },
            "session_time": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "source": "haitun-history-watcher",
                "history_path": str(config.history_path),
                "line_start": batch["line_start"],
                "line_end": batch["line_end"],
                "batch_hash": batch["batch_hash"],
                "turn_id": batch["turn_id"],
                "ended_with_error": "unknown",
            },
        }
        submit(payload)
        submitted.add(batch["batch_hash"])
        checkpoint.setdefault("submitted_batches", []).append(batch["batch_hash"])
        submitted_count += 1

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


def _post_add(config: WatcherConfig, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{config.base_url.rstrip('/')}/add",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=config.timeout_seconds) as response:
        raw = response.read().decode("utf-8")
        if not raw.strip():
            return {"ok": True}
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("ok") is False:
            raise RuntimeError(
                "Fusion Memory HTTP /add rejected passive sync payload: "
                + str(data.get("message") or data.get("error") or "unknown error")
            )
        return data if isinstance(data, dict) else {"ok": True}


def watch_history(config: WatcherConfig, *, poll_interval_seconds: float = 1.0) -> None:
    while True:
        try:
            sync_history_once(config)
        except Exception as exc:
            print(f"Fusion Memory Haitun history watcher skipped sync: {exc}", flush=True)
        time.sleep(max(0.1, poll_interval_seconds))


def history_watcher_pid_file(config: WatcherConfig) -> Path:
    return config.checkpoint_path.with_suffix(".pid")


def history_watcher_log_file(config: WatcherConfig) -> Path:
    return config.checkpoint_path.with_suffix(".log")


def start_history_watcher_daemon(
    config: WatcherConfig,
    *,
    poll_interval_seconds: float = 1.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    pid_file = history_watcher_pid_file(config)
    log_file = history_watcher_log_file(config)
    existing_pid = _read_pid(pid_file)
    if existing_pid is not None and _process_exists(existing_pid):
        return _watcher_result(
            config,
            pid=existing_pid,
            running=True,
            already_running=True,
        )

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    command = _watcher_daemon_command(config, poll_interval_seconds=poll_interval_seconds)
    daemon_env = _daemon_env(env, memory_url=config.base_url)
    try:
        with log_file.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                **_daemon_popen_kwargs(
                    log_handle,
                    cwd=str(config.workspace or config.history_path.parent.parent),
                    env=daemon_env,
                ),
            )
    except OSError as exc:
        return {
            "ok": False,
            "running": False,
            "error": "watcher_start_failed",
            "message": f"Could not start Haitun history watcher: {exc}",
            "pid_file": str(pid_file),
            "log_file": str(log_file),
        }

    pid_file.write_text(str(process.pid), encoding="utf-8")
    return _watcher_result(config, pid=process.pid, running=True, started=True)


def status_history_watcher_daemon(config: WatcherConfig) -> dict[str, Any]:
    pid = _read_pid(history_watcher_pid_file(config))
    running = bool(pid is not None and _process_exists(pid))
    return _watcher_result(config, pid=pid, running=running, ok=running)


def stop_history_watcher_daemon(
    config: WatcherConfig, *, wait_seconds: float = 5.0
) -> dict[str, Any]:
    pid_file = history_watcher_pid_file(config)
    pid = _read_pid(pid_file)
    if pid is None:
        return _watcher_result(config, pid=None, running=False, already_stopped=True)
    if not _process_exists(pid):
        pid_file.unlink(missing_ok=True)
        return _watcher_result(config, pid=pid, running=False, already_stopped=True)

    try:
        _terminate_process_tree(pid)
    except OSError as exc:
        return {
            "ok": False,
            "running": True,
            "pid": pid,
            "error": "watcher_stop_failed",
            "message": f"Could not stop Haitun history watcher: {exc}",
            "pid_file": str(pid_file),
            "log_file": str(history_watcher_log_file(config)),
        }

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not _process_exists(pid):
            pid_file.unlink(missing_ok=True)
            return _watcher_result(config, pid=pid, running=False, stopped=True)
        time.sleep(0.1)

    if os.name != "nt":
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return _watcher_result(config, pid=pid, running=False, stopped=True)
        except OSError as exc:
            return {
                "ok": False,
                "running": True,
                "pid": pid,
                "error": "watcher_stop_timeout",
                "message": f"Could not force-stop Haitun history watcher: {exc}",
                "pid_file": str(pid_file),
                "log_file": str(history_watcher_log_file(config)),
            }
        if not _process_exists(pid):
            pid_file.unlink(missing_ok=True)
            return _watcher_result(config, pid=pid, running=False, forced=True)

    return {
        "ok": False,
        "running": True,
        "pid": pid,
        "error": "watcher_stop_timeout",
        "message": f"Haitun history watcher did not stop within {wait_seconds:.1f}s.",
        "pid_file": str(pid_file),
        "log_file": str(history_watcher_log_file(config)),
    }


def _watcher_daemon_command(
    config: WatcherConfig, *, poll_interval_seconds: float
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "fusion_memory.cli",
        "--db",
        str(config.db_path),
        "sync-haitun-history",
        "--workspace",
        str(config.workspace or config.history_path.parent.parent),
        "--session-id",
        config.session_id,
        "--poll-interval-seconds",
        str(max(0.1, poll_interval_seconds)),
    ]


def _watcher_result(
    config: WatcherConfig,
    *,
    pid: int | None,
    running: bool,
    ok: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "running": running,
        "pid": pid,
        "pid_file": str(history_watcher_pid_file(config)),
        "log_file": str(history_watcher_log_file(config)),
        "checkpoint_file": str(config.checkpoint_path),
        "memory_url": config.base_url,
        **extra,
    }


def _daemon_popen_kwargs(
    log_handle: Any,
    *,
    cwd: str,
    env: Mapping[str, str],
    os_name: str | None = None,
) -> dict[str, Any]:
    name = os.name if os_name is None else os_name
    kwargs: dict[str, Any] = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "cwd": cwd,
        "env": dict(env),
    }
    if name == "nt":
        flags = 0
        for flag_name in (
            "CREATE_NEW_PROCESS_GROUP",
            "DETACHED_PROCESS",
            "CREATE_NO_WINDOW",
        ):
            flags |= int(getattr(subprocess, flag_name, 0))
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _daemon_env(
    env: Mapping[str, str] | None = None,
    *,
    memory_url: str | None = None,
    os_name: str | None = None,
) -> dict[str, str]:
    name = os.name if os_name is None else os_name
    source = os.environ if env is None else env
    normalized: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in source.items():
        if value is None:
            continue
        out_key = "Path" if name == "nt" and key.lower() == "path" else str(key)
        dedupe_key = out_key.lower() if name == "nt" else out_key
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized[out_key] = str(value)
    if memory_url:
        normalized["PSI_MEMORY_BASE_URL"] = memory_url.rstrip("/")
    return normalized


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _windows_process_exists(pid: int) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def _terminate_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    os.kill(pid, signal.SIGTERM)


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
