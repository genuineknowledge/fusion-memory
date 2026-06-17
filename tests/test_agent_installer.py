from __future__ import annotations

import unittest

from fusion_memory.agent_installer import install_agent


class AgentInstallerTests(unittest.TestCase):
    def test_install_all_dry_run_lists_three_targets(self) -> None:
        result = install_agent("all", dry_run=True)
        self.assertTrue(result["ok"])
        self.assertTrue(result["dry_run"])
        self.assertEqual(
            [item["target"] for item in result["actions"]],
            ["openclaw", "hermes", "fusion-agent"],
        )

    def test_unknown_target_is_beginner_safe(self) -> None:
        result = install_agent("bad-agent", dry_run=True)
        self.assertFalse(result["ok"])
        self.assertIn("Choose one of", result["message"])
        self.assertNotIn("Traceback", result["message"])


if __name__ == "__main__":
    unittest.main()
