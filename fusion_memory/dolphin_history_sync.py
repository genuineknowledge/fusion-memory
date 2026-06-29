from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
from urllib import error, request


DEFAULT_DOLPHIN_GATEWAY_URL = "http://127.0.0.1:8080"
DEFAULT_MEMORY_URL = "http://127.0.0.1:8700"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class DolphinHistoryTurn:
    index: int
    role: str
    text: str

    @property
    def sync_key(self) -> str:
        return sha256(f"{self.index}\0{self.role}\0{self.text}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DolphinHistorySyncConfig:
    memory_url: str = DEFAULT_MEMORY_URL
    session_id: str = ""
    workspace: Path | None = None
    gateway_url: str | None = None
    scope: dict[str, Any] = field(default_factory=lambda: {"workspace_id": "dolphin", "agent_id": "dolphin", "app_id": "dolphin"})
    state_file: Path | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS

    def normalized_memory_url(self) -> str:
        return self.memory_url.rstrip("/")

    def normalized_gateway_url(self) -> str | None:
        if not self.gateway_url:
            return None
        return self.gateway_url.rstrip("/")

    def history_file(self) -> Path:
        if self.workspace is None:
            raise ValueError("workspace is required when gateway_url is not set")
        if not self.session_id:
            raise ValueError("session_id is required")
        return self.workspace / "histories" / f"{self.session_id}.jsonl"

    def resolved_state_file(self) -> Path:
        if self.state_file is not None:
            return self.state_file
        if self.workspace is not None and self.session_id:
            return self.workspace / "histories" / f"{self.session_id}.fusion-memory-sync.json"
        return Path(".fusion-memory-history-sync.json")


PostAdd = Callable[[str, dict[str, Any], float], dict[str, Any]]
FetchHistory = Callable[[DolphinHistorySyncConfig], list[DolphinHistoryTurn]]


def load_history_file(path: Path) -> list[DolphinHistoryTurn]:
    if not path.exists():
        return []
    turns: list[DolphinHistoryTurn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        text = _message_text(item.get("content"))
        if not text:
            continue
        turns.append(DolphinHistoryTurn(index=len(turns), role=role, text=text))
    return turns


def load_gateway_history(config: DolphinHistorySyncConfig) -> list[DolphinHistoryTurn]:
    gateway_url = config.normalized_gateway_url()
    if not gateway_url:
        return load_history_file(config.history_file())
    if not config.session_id:
        raise ValueError("session_id is required")
    url = f"{gateway_url}/sessions/{config.session_id}/history"
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Dolphin gateway history request failed: HTTP {exc.code} {detail}") from exc
    if not isinstance(data, list):
        raise RuntimeError("Dolphin gateway history response must be a JSON list")
    turns: list[DolphinHistoryTurn] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        text = _message_text(item.get("text") if "text" in item else item.get("content"))
        if text:
            turns.append(DolphinHistoryTurn(index=len(turns), role=role, text=text))
    return turns


def sync_once(
    config: DolphinHistorySyncConfig,
    *,
    post_add: PostAdd | None = None,
    fetch_history: FetchHistory | None = None,
) -> dict[str, Any]:
    if not config.session_id:
        raise ValueError("session_id is required")
    post_add = post_add or _post_add
    fetch_history = fetch_history or load_gateway_history
    turns = fetch_history(config)
    state_path = config.resolved_state_file()
    state = _load_state(state_path)
    synced = set(state.get("synced", []))
    added = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    for turn in turns:
        key = turn.sync_key
        if key in synced:
            skipped += 1
            continue
        payload = _add_payload(config, turn, key)
        try:
            result = post_add(config.normalized_memory_url(), payload, config.timeout_seconds)
        except Exception as exc:
            errors.append({"history_index": turn.index, "message": str(exc)})
            continue
        if result.get("ok", True) is False:
            errors.append({"history_index": turn.index, "message": str(result.get("message") or result.get("error") or "Fusion Memory add failed")})
            continue
        synced.add(key)
        added += 1
    _save_state(state_path, {"synced": sorted(synced), "updated_at": datetime.now(timezone.utc).isoformat()})
    return {"ok": not errors, "added": added, "skipped": skipped, "errors": errors, "state_file": str(state_path)}


def sync_forever(
    config: DolphinHistorySyncConfig,
    *,
    post_add: PostAdd | None = None,
    fetch_history: FetchHistory | None = None,
) -> None:
    while True:
        sync_once(config, post_add=post_add, fetch_history=fetch_history)
        time.sleep(config.interval_seconds)


def _message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(value).strip()


def _add_payload(config: DolphinHistorySyncConfig, turn: DolphinHistoryTurn, sync_key: str) -> dict[str, Any]:
    scope = dict(config.scope)
    scope.setdefault("workspace_id", "dolphin")
    scope.setdefault("agent_id", "dolphin")
    scope.setdefault("app_id", "dolphin")
    scope["session_id"] = config.session_id
    return {
        "input": {
            "role": turn.role,
            "content": turn.text,
            "turn_id": f"dolphin_history_{turn.index}",
            "metadata": {
                "source": "dolphin_history_sync",
                "write_mode": "history_sync",
                "session_id": config.session_id,
                "history_index": turn.index,
                "history_hash": sync_key,
            },
        },
        "scope": scope,
        "session_time": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "source": "dolphin_history_sync",
            "write_mode": "history_sync",
            "session_id": config.session_id,
            "history_index": turn.index,
            "history_hash": sync_key,
            "auto_persisted": True,
        },
    }


def _post_add(memory_url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{memory_url.rstrip('/')}/add",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Fusion Memory add failed: HTTP {exc.code} {detail}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Fusion Memory add response must be a JSON object")
    return data


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"synced": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"synced": []}
    if not isinstance(data, dict):
        return {"synced": []}
    synced = data.get("synced")
    if not isinstance(synced, list):
        data["synced"] = []
    return data


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
