from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fusion_memory.agent_checks import check_agent


class AgentChecksTests(unittest.TestCase):
    def test_unknown_target_is_beginner_safe(self) -> None:
        report = check_agent("missing")
        self.assertFalse(report["ok"])
        self.assertIn("Choose one of", report["message"])

    def test_fusion_agent_check_has_actionable_message(self) -> None:
        report = check_agent("fusion-agent")
        self.assertIn("target", report)
        self.assertIn("message", report)
        self.assertNotIn("Traceback", report["message"])

    def test_dolphin_check_reports_workspace_skill_and_sync_command(self) -> None:
        report = check_agent("dolphin")

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["target"], "dolphin")
        self.assertIn("workspace", report)
        self.assertIn("skill", report)
        self.assertIn("sync-dolphin-history", report["message"])

    def test_hermes_check_uses_installed_destination_under_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            destination = home / "plugins" / "fusion_memory"
            destination.mkdir(parents=True)
            (destination / "__init__.py").write_text("", encoding="utf-8")

            report = check_agent("hermes", home=home)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["path"], str(destination))
        self.assertIn("installed", report["message"])

    def test_hermes_check_reports_missing_destination_even_when_source_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = check_agent("hermes", home=Path(tmp))

        self.assertFalse(report["ok"], report)
        self.assertIn("not installed", report["message"])

    def test_fusion_agent_check_uses_home_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            root = home / "Fusion-Agent"
            root.mkdir()

            report = check_agent("fusion-agent", home=home)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["path"], str(root))

    def test_fusion_agent_check_uses_environment_when_home_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"FUSION_AGENT_ROOT": tmp}):
            report = check_agent("fusion-agent")

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["path"], tmp)


if __name__ == "__main__":
    unittest.main()
