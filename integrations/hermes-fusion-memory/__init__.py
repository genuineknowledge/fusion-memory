from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider
except Exception:
    class MemoryProvider:  # type: ignore[no-redef]
        pass


DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 1.5
MIN_TIMEOUT_SECONDS = 0.1
MAX_TIMEOUT_SECONDS = 2.0


class FusionMemoryHTTPError(RuntimeError):
    def __init__(self, error: str, cause: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.cause = cause
        self.message = message


class FusionMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "fusion_memory"

    def __init__(self) -> None:
        self.base_url = os.getenv("FUSION_MEMORY_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = _timeout_seconds(os.getenv("FUSION_MEMORY_TIMEOUT_SECONDS"))
        self.scope: dict[str, Any] = {
            "agent_id": "hermes",
            "app_id": "fusion-memory",
        }

    def is_available(self) -> bool:
        return bool(self.base_url)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = str(kwargs.get("hermes_home") or os.getenv("HERMES_HOME") or "")
        workspace = str(kwargs.get("agent_workspace") or "hermes")
        user_id = str(kwargs.get("user_id") or kwargs.get("user_id_alt") or os.getenv("USER") or os.getenv("USERNAME") or "user")
        self.scope = {
            "workspace_id": workspace or (os.path.basename(hermes_home) if hermes_home else "hermes"),
            "user_id": user_id,
            "agent_id": "hermes",
            "session_id": session_id,
            "app_id": "fusion-memory",
        }

    def system_prompt_block(self) -> str:
        return "Fusion Memory may provide durable user preferences and project facts when available."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if session_id:
            self.scope["session_id"] = session_id
        try:
            pack = self._post_json(
                "/answer-context",
                {"query": query, "scope": self.scope, "budget": {"limit": 8, "allow_cross_session": True}},
            )
        except Exception:
            return ""
        return _format_context(pack)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if session_id:
            self.scope["session_id"] = session_id
        if not user_content and not assistant_content:
            return
        try:
            self._post_json(
                "/add",
                {
                    "input": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "scope": self.scope,
                    "metadata": {"source": "hermes", "mode": "auto-turn"},
                },
            )
        except Exception:
            return

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action not in {"add", "replace"} or not content.strip():
            return
        try:
            self._post_json(
                "/add",
                {
                    "input": {"role": "user", "content": content},
                    "scope": self.scope,
                    "metadata": {"source": "hermes-memory-tool", "target": target, **(metadata or {})},
                },
            )
        except Exception:
            return

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "fusion_memory_search",
                "description": "Search durable Fusion Memory for relevant preferences, facts, and prior context.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"],
                },
            },
            {
                "name": "fusion_memory_store",
                "description": "Store a durable user preference, project fact, or stable decision in Fusion Memory.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            },
            {
                "name": "fusion_memory_clear",
                "description": "Clear Fusion Memory rows for the current Hermes scope when the user explicitly asks.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        try:
            if tool_name == "fusion_memory_search":
                query = str(args.get("query") or "").strip()
                limit = int(args.get("limit") or 8)
                pack = self._post_json(
                    "/answer-context",
                    {"query": query, "scope": self.scope, "budget": {"limit": limit, "allow_cross_session": True}},
                )
                return json.dumps({"ok": True, "context": _format_context(pack)}, ensure_ascii=False)
            if tool_name == "fusion_memory_store":
                content = str(args.get("content") or "").strip()
                if not content:
                    return json.dumps({"ok": False, "message": "Memory content is empty."})
                result = self._post_json(
                    "/add",
                    {"input": {"role": "user", "content": content}, "scope": self.scope, "metadata": {"source": "hermes-tool"}},
                )
                return json.dumps({"ok": True, "saved": True, "result": result}, ensure_ascii=False)
            if tool_name == "fusion_memory_clear":
                result = self._post_json("/clear", {"scope": self.scope, "allow_cross_session": True})
                return json.dumps({"ok": True, "cleared": True, "result": result}, ensure_ascii=False)
            return json.dumps({"ok": False, "message": f"Unknown Fusion Memory tool: {tool_name}"})
        except Exception as exc:
            return json.dumps(_safe_failure(exc), ensure_ascii=False)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "Fusion Memory service URL",
                "default": DEFAULT_BASE_URL,
                "required": False,
            }
        ]

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                payload = {}
            raise FusionMemoryHTTPError(
                str(payload.get("error") or "request_failed"),
                str(payload.get("cause") or f"http_{exc.code}"),
                str(payload.get("message") or "Fusion Memory request failed."),
            ) from exc
        return data if isinstance(data, dict) else {}


def register(ctx: Any) -> None:
    ctx.register_memory_provider(FusionMemoryProvider())


def _format_context(pack: Dict[str, Any]) -> str:
    lines: list[str] = []
    for key in ("current_views", "entity_profiles", "facts", "events", "source_spans"):
        items = pack.get(key)
        if not isinstance(items, list):
            continue
        for item in items[:8]:
            if isinstance(item, dict):
                text = item.get("text") or item.get("summary") or item.get("content") or json.dumps(item, ensure_ascii=False)
            else:
                text = str(item)
            lines.append(f"- {text}")
    if not lines:
        return ""
    return "Fusion Memory context:\n" + "\n".join(lines)


def _safe_failure(exc: Exception | None = None) -> Dict[str, Any]:
    if isinstance(exc, FusionMemoryHTTPError):
        return {
            "ok": False,
            "error": exc.error,
            "cause": exc.cause,
            "message": exc.message,
        }
    return {
        "ok": False,
        "error": "service_unavailable",
        "cause": "connection_failed",
        "message": "Fusion Memory service is not reachable. Run fusion-memory status or fusion-memory start.",
    }


def _timeout_seconds(value: str | None) -> float:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    if parsed <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return min(MAX_TIMEOUT_SECONDS, max(MIN_TIMEOUT_SECONDS, parsed))
