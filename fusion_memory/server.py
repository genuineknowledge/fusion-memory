from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

from fusion_memory import Scope
from fusion_memory.api.service import MemoryService
from fusion_memory.core.runtime_config import memory_service_from_env
from fusion_memory.product import runtime_status_payload


class MemoryServerState:
    def __init__(self, service: MemoryService) -> None:
        self.service = service
        self.lock = threading.RLock()


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
                        storage_backend=self.server.storage_backend if hasattr(self.server, "storage_backend") else "postgres"
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
            except Exception:
                self._write_json(
                    400,
                    {
                        "error": "request_failed",
                        "message": "Fusion Memory could not complete that request. Run fusion-memory doctor.",
                    },
                )

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
            return data

        def _write_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return FusionMemoryHandler


def serve(
    service: MemoryService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> HTTPServer:
    state = MemoryServerState(service)
    server = HTTPServer((host, port), make_handler(state))
    server.storage_backend = getattr(service, "storage_backend", "postgres")  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fusion Memory as a persistent local HTTP service")
    parser.add_argument("--host", default=os.getenv("FUSION_MEMORY_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FUSION_MEMORY_SERVER_PORT", "8765")))
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
        print(json.dumps({"host": args.host, "port": args.port, "storage_backend": args.storage_backend or "postgres"}), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


def _scope(payload: dict[str, Any]) -> Scope:
    raw = payload.get("scope")
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
