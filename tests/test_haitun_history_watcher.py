from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from fusion_memory.adapters.haitun_history_watcher import config_from_workspace, load_checkpoint, sync_history_once


class HaitunHistoryWatcherTests(unittest.TestCase):
    def test_config_from_workspace_uses_expected_history_and_checkpoint_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cfg = config_from_workspace(
                workspace=workspace,
                session_id="session-1",
                db_path=workspace / "memory.sqlite3",
                env={
                    "PSI_MEMORY_BASE_URL": "http://127.0.0.1:8700",
                    "PSI_MEMORY_WORKSPACE_ID": "ws",
                    "PSI_MEMORY_USER_ID": "u",
                    "PSI_MEMORY_AGENT_ID": "haitun",
                    "PSI_MEMORY_TIMEOUT_SECONDS": "3",
                },
            )

            self.assertEqual(cfg.history_path, workspace / "histories" / "session-1.jsonl")
            self.assertEqual(cfg.checkpoint_path, workspace / ".fusion-memory" / "haitun-history-watcher" / "session-1.json")
            self.assertEqual(cfg.base_url, "http://127.0.0.1:8700")
            self.assertEqual(cfg.workspace_id, "ws")
            self.assertEqual(cfg.timeout_seconds, 3.0)

    def test_sync_history_once_posts_new_jsonl_turns_to_http_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            history.parent.mkdir()
            submitted: list[dict] = []
            history.write_text(
                json.dumps({"role": "user", "content": "我现在更喜欢用 PostgreSQL 做报表。"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"role": "assistant", "content": "已记录。"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            cfg = config_from_workspace(
                workspace=workspace,
                session_id="session-1",
                env={"PSI_MEMORY_WORKSPACE_ID": "ws", "PSI_MEMORY_USER_ID": "u", "PSI_MEMORY_AGENT_ID": "haitun"},
            )

            result = sync_history_once(cfg, submit_add=submitted.append)
            duplicate = sync_history_once(cfg, submit_add=submitted.append)

            self.assertEqual(result["submitted_count"], 1)
            self.assertEqual(duplicate["submitted_count"], 0)
            self.assertEqual(len(submitted), 1)
            self.assertEqual(submitted[0]["scope"]["workspace_id"], "ws")
            self.assertEqual(submitted[0]["scope"]["user_id"], "u")
            self.assertEqual(submitted[0]["scope"]["agent_id"], "haitun")
            self.assertEqual(submitted[0]["scope"]["session_id"], "session-1")
            self.assertEqual(submitted[0]["input"]["messages"][0]["role"], "user")
            self.assertIn("PostgreSQL", submitted[0]["input"]["messages"][0]["content"])
            self.assertEqual(submitted[0]["metadata"]["source"], "haitun-history-watcher")
            self.assertIn("session_time", submitted[0])
            checkpoint = load_checkpoint(cfg.checkpoint_path)
            self.assertEqual(len(checkpoint["submitted_batches"]), 1)

    def test_sync_history_once_keeps_identical_turns_on_different_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            submitted: list[dict] = []
            history.parent.mkdir()
            line = json.dumps({"role": "user", "content": "以后默认用中文回复我。"}, ensure_ascii=False)
            history.write_text(f"{line}\n{line}\n", encoding="utf-8")
            cfg = config_from_workspace(
                workspace=workspace,
                session_id="session-1",
                env={"PSI_MEMORY_WORKSPACE_ID": "ws", "PSI_MEMORY_USER_ID": "u", "PSI_MEMORY_AGENT_ID": "haitun"},
            )

            result = sync_history_once(cfg, submit_add=submitted.append)

            self.assertEqual(result["submitted_count"], 2)
            self.assertEqual(len(submitted), 2)
            self.assertNotEqual(
                submitted[0]["metadata"]["batch_hash"],
                submitted[1]["metadata"]["batch_hash"],
            )
            checkpoint = load_checkpoint(cfg.checkpoint_path)
            self.assertEqual(len(checkpoint["submitted_batches"]), 2)

    def test_cli_sync_haitun_history_once_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            db = workspace / "memory.sqlite3"
            server, thread, received = _start_fake_memory_server()
            history.parent.mkdir()
            history.write_text(json.dumps({"role": "user", "content": "以后默认用中文回复我。"}, ensure_ascii=False) + "\n", encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "fusion_memory.cli",
                        "--db",
                        str(db),
                        "sync-haitun-history",
                        "--workspace",
                        str(workspace),
                        "--session-id",
                        "session-1",
                        "--once",
                        "--json",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={
                        **os.environ,
                        "PSI_MEMORY_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                    },
                    check=True,
                    text=True,
                    capture_output=True,
                )
            finally:
                server.shutdown()
                thread.join(timeout=2)

            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["submitted_count"], 1)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0]["scope"]["session_id"], "session-1")
            self.assertEqual(received[0]["metadata"]["source"], "haitun-history-watcher")

    def test_cli_sync_dolphin_history_alias_still_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            db = workspace / "memory.sqlite3"
            server, thread, received = _start_fake_memory_server()
            history.parent.mkdir()
            history.write_text(json.dumps({"role": "user", "content": "以后默认用中文回复我。"}, ensure_ascii=False) + "\n", encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "fusion_memory.cli",
                        "--db",
                        str(db),
                        "sync-dolphin-history",
                        "--workspace",
                        str(workspace),
                        "--session-id",
                        "session-1",
                        "--once",
                        "--json",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={
                        **os.environ,
                        "PSI_MEMORY_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                    },
                    check=True,
                    text=True,
                    capture_output=True,
                )
            finally:
                server.shutdown()
                thread.join(timeout=2)

            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["submitted_count"], 1)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0]["metadata"]["source"], "haitun-history-watcher")


def _start_fake_memory_server() -> tuple[HTTPServer, threading.Thread, list[dict]]:
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            received.append(json.loads(body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, received


if __name__ == "__main__":
    unittest.main()
