from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fusion_memory.agent_installer import _action_for, _fusion_agent_root, install_agent


class AgentInstallerTests(unittest.TestCase):
    def test_install_all_dry_run_lists_three_targets(self) -> None:
        result = install_agent("all", dry_run=True)
        self.assertTrue(result["ok"])
        self.assertTrue(result["dry_run"])
        self.assertIn("estimated_time", result)
        self.assertIn("check_first", result)
        self.assertEqual(
            [item["target"] for item in result["actions"]],
            ["dolphin", "openclaw", "hermes", "fusion-agent"],
        )
        for action in result["actions"]:
            self.assertIn("estimated_time", action)
            self.assertIn("skip_if", action)

    def test_dolphin_action_points_to_workspace_skill_and_sync_command(self) -> None:
        action = _action_for("dolphin")

        self.assertEqual(action["target"], "dolphin")
        self.assertTrue(Path(action["workspace"]).exists())
        self.assertTrue(Path(action["skill"]).exists())
        self.assertIn("sync-haitun-history", action["sync_command"])
        self.assertNotIn("--db", action["sync_command"])
        self.assertIn("memory_add", action["message"])

    def test_unknown_target_is_beginner_safe(self) -> None:
        result = install_agent("bad-agent", dry_run=True)
        self.assertFalse(result["ok"])
        self.assertIn("Choose one of", result["message"])
        self.assertNotIn("Traceback", result["message"])

    def test_openclaw_command_error_returns_beginner_safe_result(self) -> None:
        with patch("fusion_memory.agent_installer.subprocess.run", side_effect=FileNotFoundError("openclaw missing")):
            result = install_agent("openclaw")

        self.assertFalse(result["ok"])
        self.assertFalse(result["results"][0]["ok"])
        self.assertIn("OpenClaw", result["results"][0]["message"])
        self.assertNotIn("Traceback", result["results"][0]["message"])
        self.assertNotIn("openclaw missing", result["results"][0]["message"])

    def test_actions_include_runtime_smoke_command(self) -> None:
        action = _action_for("openclaw")

        self.assertEqual(action["smoke_command"][:2], ["python3", "tools/agent_runtime_smoke.py"])
        self.assertIn("--target", action["smoke_command"])
        self.assertIn("openclaw", action["smoke_command"])

    def test_hermes_copy_failure_preserves_existing_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            destination = home / "plugins" / "fusion_memory"
            destination.mkdir(parents=True)
            marker = destination / "existing.txt"
            marker.write_text("keep", encoding="utf-8")

            with patch("fusion_memory.agent_installer.shutil.copytree", side_effect=OSError("copy exploded")):
                result = install_agent("hermes", home=home)

            self.assertFalse(result["ok"])
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertTrue(destination.exists())
            self.assertNotIn("copy exploded", result["results"][0]["message"])

    def test_fusion_agent_root_uses_home_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            action = _action_for("fusion-agent", home=home)

        self.assertEqual(action["path"], str(home / "Fusion-Agent"))

    def test_fusion_agent_root_uses_environment_when_home_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"FUSION_AGENT_ROOT": tmp}):
            action = _action_for("fusion-agent")

        self.assertEqual(action["path"], tmp)

    def test_fusion_agent_default_root_is_user_home_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"HOME": tmp}, clear=True):
            root = _fusion_agent_root()

        self.assertEqual(root, Path(tmp) / "Fusion-Agent")
        self.assertNotIn("/public/home/wwb", str(root))


if __name__ == "__main__":
    unittest.main()
