from __future__ import annotations

import argparse
import json
import os
import re
import signal
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from fusion_memory import Scope
from fusion_memory.api.service import MemoryService
from fusion_memory.core.runtime_config import memory_service_from_env
from fusion_memory.product import runtime_status_payload


class MemoryServerState:
    def __init__(
        self,
        service: MemoryService,
        *,
        background_task_interval_seconds: float = 1.0,
        background_task_batch_size: int = 5,
    ) -> None:
        self.service = service
        self.lock = threading.RLock()
        self.background_task_interval_seconds = max(0.0, background_task_interval_seconds)
        self.background_task_batch_size = max(1, background_task_batch_size)
        self.next_background_task_run = time.monotonic() + self.background_task_interval_seconds
        self.background_task_launch_lock = threading.Lock()
        self.background_task_thread: threading.Thread | None = None
        self.last_background_task_error: str | None = None


class FusionMemoryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], state: MemoryServerState) -> None:
        super().__init__(server_address, handler_class)
        self.state = state
        self.storage_backend = state.service.storage_backend

    def service_actions(self) -> None:
        super().service_actions()
        interval = self.state.background_task_interval_seconds
        if interval <= 0:
            return
        now = time.monotonic()
        if now < self.state.next_background_task_run:
            return
        self.state.next_background_task_run = now + interval
        with self.state.background_task_launch_lock:
            thread = self.state.background_task_thread
            if thread is not None and thread.is_alive():
                return
            self.state.background_task_thread = threading.Thread(
                target=self._run_background_tasks,
                daemon=True,
            )
            self.state.background_task_thread.start()

    def _run_background_tasks(self) -> None:
        with self.state.lock:
            try:
                self.state.service.process_server_background_tasks(limit=self.state.background_task_batch_size)
            except Exception as exc:
                self.state.last_background_task_error = f"{exc.__class__.__name__}: {exc}"

    def server_close(self) -> None:
        thread = self.state.background_task_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        super().server_close()


def make_handler(state: MemoryServerState) -> type[BaseHTTPRequestHandler]:
    class FusionMemoryHandler(BaseHTTPRequestHandler):
        server_version = "FusionMemoryHTTP/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._write_json(200, {"ok": True})
                return
            if path == "/status":
                self._write_json(
                    200,
                    runtime_status_payload(
                        storage_backend=self.server.storage_backend if hasattr(self.server, "storage_backend") else "sqlite"
                    ),
                )
                return
            self._write_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                with state.lock:
                    if path == "/add":
                        result = state.service.add(
                            payload.get("input"),
                            _scope(payload),
                            _optional_datetime(payload.get("session_time")),
                            metadata=payload.get("metadata"),
                        )
                    elif path == "/search":
                        result = state.service.search(
                            _required(payload, "query"),
                            _scope(payload),
                            options=payload.get("options") or {},
                        )
                    elif path == "/answer-context":
                        result = state.service.answer_context(
                            _required(payload, "query"),
                            _scope(payload),
                            budget=payload.get("budget"),
                        )
                    elif path in {"/clear", "/delete"}:
                        result = state.service.clear(
                            _scope(payload),
                            allow_cross_session=bool(payload.get("allow_cross_session", False)),
                        )
                    else:
                        self._write_json(404, {"error": "not_found"})
                        return
                self._write_json(200, _jsonable(result))
            except Exception as exc:
                status, payload = _error_response(exc)
                self._write_json(status, payload)

        def log_message(self, format: str, *args: Any) -> None:
            if os.getenv("FUSION_MEMORY_SERVER_LOG_REQUESTS", "").lower() in {"1", "true", "yes"}:
                super().log_message(format, *args)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return _repair_request_payload(data)

        def _write_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

    return FusionMemoryHandler


def serve(
    service: MemoryService,
    *,
    host: str = "127.0.0.1",
    port: int = 8700,
    background_task_interval_seconds: float = 1.0,
    background_task_batch_size: int = 5,
) -> HTTPServer:
    state = MemoryServerState(
        service,
        background_task_interval_seconds=background_task_interval_seconds,
        background_task_batch_size=background_task_batch_size,
    )
    server = FusionMemoryHTTPServer((host, port), make_handler(state), state)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fusion Memory as a persistent local HTTP service")
    parser.add_argument("--host", default=os.getenv("FUSION_MEMORY_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FUSION_MEMORY_SERVER_PORT", "8700")))
    parser.add_argument("--db", default=os.getenv("FUSION_MEMORY_DB", "fusion-memory.sqlite3"))
    parser.add_argument("--storage-backend", default=os.getenv("FUSION_MEMORY_STORAGE_BACKEND"))
    args = parser.parse_args()

    service = memory_service_from_env(args.db, storage_backend=args.storage_backend)
    server = serve(service, host=args.host, port=args.port)

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        print(json.dumps({"host": args.host, "port": args.port, "storage_backend": service.storage_backend}), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


def _scope(payload: dict[str, Any]) -> Scope:
    raw = payload.get("scope")
    if raw is None:
        raw = {
            key: payload.get(key)
            for key in ("workspace_id", "user_id", "agent_id", "run_id", "session_id", "app_id")
            if payload.get(key) is not None
        }
        if not raw:
            raise ValueError("scope is required")
    if not isinstance(raw, dict):
        raise ValueError("scope is required")
    return Scope(
        workspace_id=raw.get("workspace_id"),
        user_id=raw.get("user_id"),
        agent_id=raw.get("agent_id"),
        run_id=raw.get("run_id"),
        session_id=raw.get("session_id"),
        app_id=raw.get("app_id"),
    )


def _repair_request_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _repair_likely_mojibake(value)
    if isinstance(value, list):
        return [_repair_request_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_request_payload(item) for key, item in value.items()}
    return value


_MOJIBAKE_SIGNATURES = (
    "浣犲",
    "鍠滄",
    "濆啺",
    "缇庡",
    "鍜栧",
    "枃娴嬭",
    "榛樿",
    "鐢ㄤ",
    "腑鏂",
    "囧洖",
    "澶嶆",
)
_PRIVATE_OR_REPLACEMENT_RE = re.compile(r"[\ue000-\uf8ff\ufffd\x80-\x9f]")
_LATIN1_MOJIBAKE_RE = re.compile(
    r"[ÃÂâÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßåæçèéêëìíîïðñòóôõöøùúûüýþÿ]"
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def _repair_likely_mojibake(text: str) -> str:
    score = _mojibake_score(text)
    if score <= 0:
        return text

    best = text
    best_score = score
    for encoding in ("gb18030", "latin-1"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        candidate_score = _mojibake_score(candidate)
        if candidate != text and candidate_score < best_score and _plausible_repair(text, candidate):
            best = candidate
            best_score = candidate_score
    return best


def _mojibake_score(text: str) -> int:
    score = 0
    score += 5 * len(_PRIVATE_OR_REPLACEMENT_RE.findall(text))
    score += 2 * len(_LATIN1_MOJIBAKE_RE.findall(text))
    for signature in _MOJIBAKE_SIGNATURES:
        if signature in text:
            score += 4
    return score


def _plausible_repair(original: str, candidate: str) -> bool:
    if not candidate.strip():
        return False
    if _PRIVATE_OR_REPLACEMENT_RE.search(candidate):
        return False
    if len(_CJK_RE.findall(original)) and not len(_CJK_RE.findall(candidate)):
        return False
    return True


def _required(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("session_time must be an ISO datetime string")
    return datetime.fromisoformat(value)


def _error_response(exc: Exception) -> tuple[int, dict[str, str]]:
    if isinstance(exc, json.JSONDecodeError):
        return (
            400,
            {
                "error": "bad_request",
                "cause": "invalid_json",
                "message": "Request body must be valid JSON.",
            },
        )
    if isinstance(exc, ValueError):
        message = str(exc)
        if message == "scope is required":
            return (
                400,
                {
                    "error": "bad_request",
                    "cause": "missing_scope",
                    "message": "Request must include a scope with at least one of workspace_id, user_id, agent_id, or run_id.",
                },
            )
        if message.endswith(" is required"):
            field = message.removesuffix(" is required")
            return (
                400,
                {
                    "error": "bad_request",
                    "cause": f"missing_{field}",
                    "message": f"Request must include {field}.",
                },
            )
        if message in {
            "request body must be a JSON object",
            "session_time must be an ISO datetime string",
            "add requires at least one of workspace_id, user_id, agent_id, or run_id",
            "read requires at least one of workspace_id, user_id, agent_id, or run_id",
        }:
            return (
                400,
                {
                    "error": "bad_request",
                    "cause": _safe_cause(message),
                    "message": message,
                },
            )
    return (
        500,
        {
            "error": "request_failed",
            "cause": "server_error",
            "message": "Fusion Memory could not complete that request. Run fusion-memory doctor.",
        },
    )


def _safe_cause(message: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")[:80]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
