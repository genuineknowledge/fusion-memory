from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
