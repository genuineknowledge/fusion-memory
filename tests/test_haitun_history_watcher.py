from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fusion_memory.adapters.haitun_history_watcher import config_from_workspace, load_checkpoint, sync_history_once
from fusion_memory.api.service import MemoryService
from fusion_memory.core.models import Scope


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

    def test_sync_history_once_persists_new_jsonl_turns_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            history.parent.mkdir()
            db = workspace / "memory.sqlite3"
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
                db_path=db,
                env={"PSI_MEMORY_WORKSPACE_ID": "ws", "PSI_MEMORY_USER_ID": "u", "PSI_MEMORY_AGENT_ID": "haitun"},
            )

            result = sync_history_once(cfg)
            duplicate = sync_history_once(cfg)

            self.assertEqual(result["submitted_count"], 1)
            self.assertEqual(duplicate["submitted_count"], 0)
            checkpoint = load_checkpoint(cfg.checkpoint_path)
            self.assertEqual(len(checkpoint["submitted_batches"]), 1)

            memory = MemoryService(db_path=db)
            try:
                facts = memory.store.list_facts(Scope(workspace_id="ws", user_id="u", agent_id="haitun", session_id="session-1"))
            finally:
                memory.close()
            self.assertTrue(any(fact.category == "preference" and "PostgreSQL" in fact.object for fact in facts))

    def test_sync_history_once_keeps_identical_turns_on_different_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            db = workspace / "memory.sqlite3"
            history.parent.mkdir()
            line = json.dumps({"role": "user", "content": "以后默认用中文回复我。"}, ensure_ascii=False)
            history.write_text(f"{line}\n{line}\n", encoding="utf-8")
            cfg = config_from_workspace(
                workspace=workspace,
                session_id="session-1",
                db_path=db,
                env={"PSI_MEMORY_WORKSPACE_ID": "ws", "PSI_MEMORY_USER_ID": "u", "PSI_MEMORY_AGENT_ID": "haitun"},
            )

            result = sync_history_once(cfg)

            self.assertEqual(result["submitted_count"], 2)
            checkpoint = load_checkpoint(cfg.checkpoint_path)
            self.assertEqual(len(checkpoint["submitted_batches"]), 2)

    def test_cli_sync_haitun_history_once_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            db = workspace / "memory.sqlite3"
            history.parent.mkdir()
            history.write_text(json.dumps({"role": "user", "content": "以后默认用中文回复我。"}, ensure_ascii=False) + "\n", encoding="utf-8")

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
                check=True,
                text=True,
                capture_output=True,
            )

            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["submitted_count"], 1)

    def test_cli_sync_dolphin_history_alias_still_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            history = workspace / "histories" / "session-1.jsonl"
            db = workspace / "memory.sqlite3"
            history.parent.mkdir()
            history.write_text(json.dumps({"role": "user", "content": "以后默认用中文回复我。"}, ensure_ascii=False) + "\n", encoding="utf-8")

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
                check=True,
                text=True,
                capture_output=True,
            )

            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["submitted_count"], 1)


if __name__ == "__main__":
    unittest.main()
