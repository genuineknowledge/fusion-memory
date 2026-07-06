from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from fusion_memory.adapters import haitun_history_watcher as watcher
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

    def test_config_from_workspace_defaults_and_caps_timeout_for_local_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            default_cfg = config_from_workspace(
                workspace=workspace,
                session_id="session-1",
                env={},
            )
            capped_cfg = config_from_workspace(
                workspace=workspace,
                session_id="session-1",
                env={"PSI_MEMORY_TIMEOUT_SECONDS": "999"},
            )

            self.assertEqual(default_cfg.timeout_seconds, 30.0)
            self.assertEqual(capped_cfg.timeout_seconds, 120.0)

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

    def test_start_history_watcher_daemon_spawns_python_without_shell_and_writes_pid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cfg = config_from_workspace(
                workspace=workspace,
                session_id="session-1",
                db_path=workspace / "memory.sqlite3",
                base_url="http://127.0.0.1:9876",
                env={"PATH": os.environ.get("PATH", "")},
            )
            fake_process = _FakeProcess(pid=24680)

            with (
                patch.object(watcher, "_process_exists", return_value=False),
                patch.object(watcher.subprocess, "Popen", return_value=fake_process) as popen,
            ):
                result = watcher.start_history_watcher_daemon(
                    cfg,
                    poll_interval_seconds=0.5,
                )
            pid_text = Path(result["pid_file"]).read_text(encoding="utf-8")

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["pid"], 24680)
        self.assertEqual(pid_text, "24680")
        self.assertTrue(str(result["log_file"]).endswith("session-1.log"))

        command = popen.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "fusion_memory.cli"])
        self.assertIn("--db", command)
        self.assertIn(str(workspace / "memory.sqlite3"), command)
        self.assertIn("sync-haitun-history", command)
        self.assertIn("--workspace", command)
        self.assertIn(str(workspace), command)
        self.assertIn("--session-id", command)
        self.assertIn("session-1", command)
        self.assertIn("--poll-interval-seconds", command)
        self.assertIn("0.5", command)
        self.assertNotIn("--once", command)

        kwargs = popen.call_args.kwargs
        self.assertNotEqual(kwargs.get("shell"), True)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["env"]["PSI_MEMORY_BASE_URL"], "http://127.0.0.1:9876")

    def test_start_history_watcher_daemon_reports_already_running_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cfg = config_from_workspace(workspace=workspace, session_id="session-1")
            pid_file = watcher.history_watcher_pid_file(cfg)
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text("13579", encoding="utf-8")

            with (
                patch.object(watcher, "_process_exists", return_value=True),
                patch.object(watcher.subprocess, "Popen") as popen,
            ):
                result = watcher.start_history_watcher_daemon(cfg)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["already_running"])
        self.assertEqual(result["pid"], 13579)
        popen.assert_not_called()

    def test_history_watcher_status_and_stop_use_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cfg = config_from_workspace(workspace=workspace, session_id="session-1")
            pid_file = watcher.history_watcher_pid_file(cfg)
            pid_file.parent.mkdir(parents=True)
            pid_file.write_text("13579", encoding="utf-8")

            with patch.object(watcher, "_process_exists", return_value=True):
                status = watcher.status_history_watcher_daemon(cfg)

            with (
                patch.object(watcher, "_process_exists", side_effect=[True, False]),
                patch.object(watcher.os, "kill") as kill,
            ):
                stopped = watcher.stop_history_watcher_daemon(cfg, wait_seconds=0.01)

        self.assertTrue(status["running"])
        self.assertEqual(status["pid"], 13579)
        self.assertTrue(stopped["ok"], stopped)
        self.assertTrue(stopped["stopped"])
        self.assertFalse(pid_file.exists())
        kill.assert_called()

    def test_history_watcher_status_is_not_ok_when_pid_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cfg = config_from_workspace(workspace=workspace, session_id="session-1")

            status = watcher.status_history_watcher_daemon(cfg)

        self.assertFalse(status["ok"])
        self.assertFalse(status["running"])
        self.assertIsNone(status["pid"])

    def test_windows_history_watcher_env_dedupes_path_case(self) -> None:
        env = watcher._daemon_env(
            {
                "Path": r"C:\Windows\System32",
                "PATH": r"C:\msys64\ucrt64\bin",
                "OTHER": "value",
            },
            memory_url="http://127.0.0.1:8700",
            os_name="nt",
        )

        path_keys = [key for key in env if key.lower() == "path"]
        self.assertEqual(path_keys, ["Path"])
        self.assertEqual(env["Path"], r"C:\Windows\System32")
        self.assertEqual(env["PSI_MEMORY_BASE_URL"], "http://127.0.0.1:8700")

    def test_windows_history_watcher_daemon_starts_without_console_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "watcher.log"
            with (
                log_path.open("ab") as handle,
                patch.object(watcher.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, create=True),
                patch.object(watcher.subprocess, "DETACHED_PROCESS", 0x00000008, create=True),
                patch.object(watcher.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
            ):
                kwargs = watcher._daemon_popen_kwargs(
                    handle,
                    cwd=tmp,
                    env={
                        "Path": r"C:\Windows\System32",
                        "PATH": r"C:\msys64\ucrt64\bin",
                    },
                    os_name="nt",
                )

        self.assertEqual(kwargs["creationflags"] & 0x08000000, 0x08000000)
        self.assertEqual(kwargs["creationflags"] & 0x00000008, 0x00000008)
        self.assertEqual(kwargs["creationflags"] & 0x00000200, 0x00000200)
        path_keys = [key for key in kwargs["env"] if key.lower() == "path"]
        self.assertEqual(path_keys, ["Path"])

    def test_cli_exposes_cross_platform_history_watcher_daemon_commands(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "fusion_memory.cli", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("start-haitun-history-watcher", proc.stdout)
        self.assertIn("status-haitun-history-watcher", proc.stdout)
        self.assertIn("stop-haitun-history-watcher", proc.stdout)

        sync_help = subprocess.run(
            [sys.executable, "-m", "fusion_memory.cli", "sync-haitun-history", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--background", sync_help.stdout)

    def test_cli_sync_haitun_history_background_outputs_daemon_json(self) -> None:
        from fusion_memory import cli

        old_argv = sys.argv
        old_stdout = sys.stdout
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            try:
                sys.argv = [
                    "fusion-memory",
                    "sync-haitun-history",
                    "--workspace",
                    str(workspace),
                    "--session-id",
                    "session-1",
                    "--memory-url",
                    "http://127.0.0.1:9876",
                    "--background",
                    "--json",
                ]
                sys.stdout = StringIO()
                with patch(
                    "fusion_memory.cli.start_history_watcher_daemon",
                    return_value={
                        "ok": True,
                        "running": True,
                        "pid": 24680,
                        "pid_file": str(workspace / ".fusion-memory" / "haitun-history-watcher" / "session-1.pid"),
                        "log_file": str(workspace / ".fusion-memory" / "haitun-history-watcher" / "session-1.log"),
                    },
                ) as start:
                    code = cli.main()
                payload = json.loads(sys.stdout.getvalue())
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout

        self.assertEqual(code, 0)
        self.assertEqual(payload["pid"], 24680)
        self.assertTrue(payload["running"])
        self.assertIn("pid_file", payload)
        self.assertIn("log_file", payload)
        start.assert_called_once()


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


class _FakeProcess:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid


if __name__ == "__main__":
    unittest.main()
