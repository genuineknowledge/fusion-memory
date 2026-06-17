from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
