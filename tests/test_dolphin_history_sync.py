from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fusion_memory.dolphin_history_sync import DolphinHistorySyncConfig, DolphinHistoryTurn, _add_payload, load_gateway_history, load_history_file, sync_once


class DolphinHistorySyncTests(unittest.TestCase):
    def test_load_history_file_filters_agent_turns_and_keeps_filtered_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "session-a.jsonl"
            history_file.write_text(
                "\n".join(
                    [
                        json.dumps({"role": "system", "content": "ignore"}),
                        json.dumps({"role": "user", "content": "Remember I use DashScope."}),
                        "{bad json",
                        json.dumps({"role": "tool", "content": "ignore tool"}),
                        json.dumps({"role": "assistant", "content": [{"text": "Noted."}]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            turns = load_history_file(history_file)

        self.assertEqual([(turn.index, turn.role, turn.text) for turn in turns], [(0, "user", "Remember I use DashScope."), (1, "assistant", "Noted.")])

    def test_sync_once_posts_unsynced_turns_and_persists_state(self) -> None:
        posted: list[dict] = []

        def fake_post(memory_url: str, payload: dict, timeout_seconds: float) -> dict:
            posted.append({"memory_url": memory_url, "payload": payload, "timeout_seconds": timeout_seconds})
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            histories = workspace / "histories"
            histories.mkdir(parents=True)
            history_file = histories / "session-a.jsonl"
            history_file.write_text(
                "\n".join(
                    [
                        json.dumps({"role": "user", "content": "My project codename is river."}),
                        json.dumps({"role": "assistant", "content": "I will remember that."}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = DolphinHistorySyncConfig(
                memory_url="http://memory.local:8700",
                session_id="session-a",
                workspace=workspace,
                scope={
                    "workspace_id": "dolphin",
                    "user_id": "alice",
                    "agent_id": "dolphin",
                    "session_id": "session-a",
                    "app_id": "dolphin",
                },
            )

            first = sync_once(config, post_add=fake_post)
            second = sync_once(config, post_add=fake_post)

            state_file = histories / "session-a.fusion-memory-sync.json"
            state = json.loads(state_file.read_text(encoding="utf-8"))

        self.assertTrue(first["ok"])
        self.assertEqual(first["added"], 2)
        self.assertEqual(first["skipped"], 0)
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(len(posted), 2)
        self.assertEqual(posted[0]["memory_url"], "http://memory.local:8700")
        self.assertEqual(posted[0]["timeout_seconds"], 5.0)
        self.assertEqual(posted[0]["payload"]["input"]["role"], "user")
        self.assertEqual(posted[0]["payload"]["input"]["content"], "My project codename is river.")
        self.assertEqual(posted[0]["payload"]["input"]["turn_id"], "dolphin_history_0")
        self.assertEqual(posted[0]["payload"]["scope"]["session_id"], "session-a")
        self.assertTrue(posted[0]["payload"]["metadata"]["auto_persisted"])
        self.assertEqual(posted[0]["payload"]["metadata"]["source"], "dolphin_history_sync")
        self.assertEqual(posted[0]["payload"]["metadata"]["write_mode"], "history_sync")
        self.assertEqual(posted[0]["payload"]["input"]["metadata"]["write_mode"], "history_sync")
        self.assertEqual(posted[0]["payload"]["metadata"]["history_index"], 0)
        self.assertEqual(len(state["synced"]), 2)

    def test_sync_once_uses_gateway_history_when_configured(self) -> None:
        posted: list[dict] = []
        fetched: list[str | None] = []

        def fake_fetch(config: DolphinHistorySyncConfig):
            fetched.append(config.gateway_url)
            return load_gateway_history_response([{"role": "user", "text": "Use gateway history."}])

        def fake_post(memory_url: str, payload: dict, timeout_seconds: float) -> dict:
            posted.append(payload)
            return {"ok": True}

        with tempfile.TemporaryDirectory() as tmp:
            config = DolphinHistorySyncConfig(
                memory_url="http://memory.local:8700",
                gateway_url="http://gateway.local:8080",
                session_id="gateway-session",
                workspace=Path(tmp),
            )

            result = sync_once(config, fetch_history=fake_fetch, post_add=fake_post)

        self.assertTrue(result["ok"])
        self.assertEqual(fetched, ["http://gateway.local:8080"])
        self.assertEqual(posted[0]["input"]["content"], "Use gateway history.")
        self.assertEqual(posted[0]["scope"]["session_id"], "gateway-session")

    def test_load_gateway_history_reads_gateway_endpoint(self) -> None:
        opened: list[str] = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    [
                        {"role": "user", "text": "Gateway user turn."},
                        {"role": "system", "text": "ignore"},
                        {"role": "assistant", "text": "Gateway assistant turn."},
                    ]
                ).encode("utf-8")

        def fake_urlopen(req, timeout: float) -> Response:
            opened.append(req.full_url)
            self.assertEqual(timeout, 5.0)
            return Response()

        config = DolphinHistorySyncConfig(gateway_url="http://gateway.local:8080/", session_id="session-a")

        with patch("fusion_memory.dolphin_history_sync.request.urlopen", fake_urlopen):
            turns = load_gateway_history(config)

        self.assertEqual(opened, ["http://gateway.local:8080/sessions/session-a/history"])
        self.assertEqual([(turn.index, turn.role, turn.text) for turn in turns], [(0, "user", "Gateway user turn."), (1, "assistant", "Gateway assistant turn.")])

    def test_sync_once_reports_add_errors_without_marking_turn_synced(self) -> None:
        def failing_post(memory_url: str, payload: dict, timeout_seconds: float) -> dict:
            return {"ok": False, "message": "not ready"}

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            histories = workspace / "histories"
            histories.mkdir(parents=True)
            (histories / "session-a.jsonl").write_text(json.dumps({"role": "user", "content": "Persist later."}) + "\n", encoding="utf-8")
            config = DolphinHistorySyncConfig(session_id="session-a", workspace=workspace)

            result = sync_once(config, post_add=failing_post)
            state = json.loads((histories / "session-a.fusion-memory-sync.json").read_text(encoding="utf-8"))

        self.assertFalse(result["ok"])
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["errors"][0]["message"], "not ready")
        self.assertEqual(state["synced"], [])

    def test_payload_keeps_run_scope_and_overrides_session_scope(self) -> None:
        config = DolphinHistorySyncConfig(
            session_id="actual-session",
            scope={"workspace_id": "ws", "agent_id": "agent", "run_id": "run-1", "session_id": "stale"},
        )

        payload = _add_payload(config, DolphinHistoryTurn(index=0, role="user", text="hello"), "hash-a")

        self.assertEqual(payload["scope"]["run_id"], "run-1")
        self.assertEqual(payload["scope"]["session_id"], "actual-session")


def load_gateway_history_response(items: list[dict]) -> list:
    turns = []
    for item in items:
        role = item.get("role")
        text = item.get("text")
        if role and text:
            turns.append(DolphinHistoryTurn(index=len(turns), role=role, text=text))
    return turns


if __name__ == "__main__":
    unittest.main()
