from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fusion_memory.alpha_beta import run_alpha, run_beta


class AlphaBetaHarnessTests(unittest.TestCase):
    def test_alpha_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "alpha.json"
            result = run_alpha(report_path=report)
            self.assertTrue(result["ok"])
            self.assertTrue(report.exists())
            self.assertGreaterEqual(len(result["checks"]), 5)

    def test_beta_dry_simulation_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "beta.json"
            result = run_beta(report_path=report)
            self.assertTrue(result["ok"])
            self.assertTrue(report.exists())
            self.assertGreaterEqual(len(result["checks"]), 5)

    def test_alpha_returns_safe_result_when_service_fails(self) -> None:
        with patch("fusion_memory.alpha_beta.MemoryService", side_effect=RuntimeError("internal_table secret_dsn")):
            result = run_alpha()

        serialized = str(result)
        self.assertFalse(result["ok"])
        self.assertIn("fusion-memory doctor", result["message"])
        self.assertNotIn("RuntimeError", serialized)
        self.assertNotIn("internal_table", serialized)
        self.assertNotIn("secret_dsn", serialized)

    def test_alpha_returns_safe_result_when_report_write_fails(self) -> None:
        with patch("fusion_memory.alpha_beta._write_report", side_effect=OSError("permission denied internal_path")):
            result = run_alpha(report_path=Path("/tmp/alpha.json"))

        serialized = str(result)
        self.assertFalse(result["ok"])
        self.assertIn("fusion-memory doctor", result["message"])
        self.assertNotIn("OSError", serialized)
        self.assertNotIn("permission denied", serialized)
        self.assertNotIn("internal_path", serialized)

    def test_beta_returns_safe_result_when_check_fails(self) -> None:
        with patch("fusion_memory.alpha_beta.check_agent", side_effect=RuntimeError("agent raw traceback")):
            result = run_beta()

        serialized = str(result)
        self.assertFalse(result["ok"])
        self.assertIn("fusion-memory doctor", result["message"])
        self.assertNotIn("RuntimeError", serialized)
        self.assertNotIn("raw traceback", serialized)

    def test_beta_dry_simulation_does_not_require_fusion_agent_checkout(self) -> None:
        def fake_check(target: str):
            if target == "fusion-agent":
                return {"ok": False, "message": "Fusion-Agent checkout was not found."}
            return {"ok": True, "message": f"{target} files are present."}

        with patch("fusion_memory.alpha_beta.check_agent", side_effect=fake_check):
            result = run_beta()

        self.assertTrue(result["ok"])
        checkout = next(item for item in result["checks"] if item["name"] == "fusion_agent_checkout")
        self.assertTrue(checkout["ok"])
        self.assertIn("optional", checkout["detail"])

    def test_beta_returns_safe_result_when_report_write_fails(self) -> None:
        with patch("fusion_memory.alpha_beta._write_report", side_effect=OSError("report internal failure")):
            result = run_beta(report_path=Path("/tmp/beta.json"))

        serialized = str(result)
        self.assertFalse(result["ok"])
        self.assertIn("fusion-memory doctor", result["message"])
        self.assertNotIn("OSError", serialized)
        self.assertNotIn("internal failure", serialized)


if __name__ == "__main__":
    unittest.main()
