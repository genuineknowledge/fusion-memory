from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import tools.agent_runtime_smoke as smoke


class AgentRuntimeSmokeTests(unittest.TestCase):
    def test_missing_openclaw_host_is_beginner_safe(self) -> None:
        with patch("tools.agent_runtime_smoke.shutil.which", return_value=None):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertFalse(report["ok"])
        self.assertFalse(report["host_available"])
        self.assertIn("OpenClaw", report["message"])
        self.assertNotIn("Traceback", json.dumps(report))

    def test_cli_writes_output_json(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tools.agent_runtime_smoke.run_smoke", return_value={"ok": True, "target": "hermes"}),
        ):
            out = Path(tmp) / "smoke.json"
            code = smoke.main(["--target", "hermes", "--memory-url", "http://127.0.0.1:8765", "--output", str(out)])

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["target"], "hermes")

    def test_script_invocation_writes_beginner_safe_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke.json"
            missing_checkout = Path(tmp) / "missing-fusion-agent"
            env = {**os.environ, "FUSION_AGENT_ROOT": str(missing_checkout)}
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/agent_runtime_smoke.py",
                    "--target",
                    "fusion-agent",
                    "--memory-url",
                    "http://127.0.0.1:8765",
                    "--output",
                    str(out),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertTrue(out.exists(), completed.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertIn("Fusion-Agent", report["message"])
            self.assertNotIn("Traceback", completed.stderr + json.dumps(report))

    def test_openclaw_builtin_smoke_missing_node_is_beginner_safe(self) -> None:
        with (
            patch("tools.agent_runtime_smoke._host_available", return_value=(True, "host ok")),
            patch("tools.agent_runtime_smoke._plugin_available", return_value=(True, "plugin ok")),
            patch("tools.agent_runtime_smoke.shutil.which", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertFalse(report["ok"])
        self.assertTrue(report["host_available"])
        self.assertFalse(report["write_smoke"])
        self.assertFalse(report["retrieve_smoke"])
        self.assertIn("Install Node.js", report["message"])
        self.assertNotIn("Traceback", json.dumps(report))

    def test_openclaw_host_and_plugin_present_uses_builtin_write_retrieve_smoke(self) -> None:
        with (
            patch("tools.agent_runtime_smoke._host_available", return_value=(True, "host ok")),
            patch("tools.agent_runtime_smoke._plugin_available", return_value=(True, "plugin ok")),
            patch(
                "tools.agent_runtime_smoke._run_builtin_adapter_smoke",
                return_value={
                    "write_smoke": True,
                    "retrieve_smoke": True,
                    "ok": True,
                    "message": "OpenClaw adapter runtime smoke completed.",
                },
            ) as builtin,
            patch.dict(os.environ, {}, clear=True),
        ):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertTrue(report["ok"])
        self.assertTrue(report["write_smoke"])
        self.assertTrue(report["retrieve_smoke"])
        builtin.assert_called_once()
        self.assertEqual(builtin.call_args.args[0], "openclaw")

    def test_openclaw_builtin_smoke_requires_runtime_tool_inspection(self) -> None:
        with (
            patch("tools.agent_runtime_smoke._openclaw_runtime_tools_available", return_value=(False, "runtime missing")),
            patch("tools.agent_runtime_smoke.shutil.which", return_value="/usr/bin/node"),
            patch("tools.agent_runtime_smoke._run_command_smoke") as command_smoke,
        ):
            report = smoke._run_builtin_adapter_smoke("openclaw", memory_url="http://127.0.0.1:8765", timeout=5)

        self.assertFalse(report.get("ok", False))
        self.assertEqual(report["message"], "runtime missing")
        command_smoke.assert_not_called()

    def test_openclaw_runtime_tool_inspection_requires_store_and_search_tools(self) -> None:
        completed = CompletedProcess(
            args=["openclaw"],
            returncode=0,
            stdout='{"tools":["fusion_memory_store","fusion_memory_search"]}',
            stderr="",
        )
        with patch("subprocess.run", return_value=completed):
            ok, message = smoke._openclaw_runtime_tools_available(timeout=5)

        self.assertTrue(ok)
        self.assertIn("visible", message)

    def test_openclaw_runtime_tool_inspection_fails_when_tools_are_absent(self) -> None:
        completed = CompletedProcess(args=["openclaw"], returncode=0, stdout='{"tools":[]}', stderr="Traceback detail")
        with patch("subprocess.run", return_value=completed):
            ok, message = smoke._openclaw_runtime_tools_available(timeout=5)

        self.assertFalse(ok)
        self.assertIn("tools were not visible", message)
        self.assertNotIn("Traceback", message)

    def test_hermes_host_and_plugin_present_uses_builtin_provider_write_retrieve_smoke(self) -> None:
        with (
            patch("tools.agent_runtime_smoke._host_available", return_value=(True, "host ok")),
            patch("tools.agent_runtime_smoke._plugin_available", return_value=(True, "plugin ok")),
            patch(
                "tools.agent_runtime_smoke._run_builtin_adapter_smoke",
                return_value={
                    "write_smoke": True,
                    "retrieve_smoke": True,
                    "ok": True,
                    "message": "Hermes adapter runtime smoke completed.",
                },
            ) as builtin,
            patch.dict(os.environ, {}, clear=True),
        ):
            report = smoke.run_smoke("hermes", memory_url="http://127.0.0.1:8765")

        self.assertTrue(report["ok"])
        self.assertTrue(report["write_smoke"])
        self.assertTrue(report["retrieve_smoke"])
        builtin.assert_called_once()
        self.assertEqual(builtin.call_args.args[0], "hermes")

    def test_cli_output_includes_required_fields_for_partial_mocked_report(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tools.agent_runtime_smoke.run_smoke", return_value={"ok": True, "target": "hermes"}),
        ):
            out = Path(tmp) / "smoke.json"
            code = smoke.main(["--target", "hermes", "--memory-url", "http://127.0.0.1:8765", "--output", str(out)])
            report = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            sorted(report),
            ["host_available", "message", "ok", "plugin_available", "retrieve_smoke", "target", "write_smoke"],
        )
        self.assertEqual(report["target"], "hermes")

    def test_configured_adapter_smoke_command_json_controls_success(self) -> None:
        completed = CompletedProcess(
            args=["fake-smoke"],
            returncode=0,
            stdout='{"write_smoke": true, "retrieve_smoke": true, "message": "adapter ok"}',
            stderr="",
        )
        with (
            patch("tools.agent_runtime_smoke._host_available", return_value=(True, "host ok")),
            patch("tools.agent_runtime_smoke._plugin_available", return_value=(True, "plugin ok")),
            patch("subprocess.run", return_value=completed) as run,
            patch.dict(os.environ, {"FUSION_MEMORY_OPENCLAW_SMOKE_COMMAND": "fake-smoke --json"}, clear=True),
        ):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertTrue(report["ok"])
        self.assertTrue(report["write_smoke"])
        self.assertTrue(report["retrieve_smoke"])
        self.assertEqual(report["message"], "adapter ok")
        self.assertEqual(run.call_args.kwargs["env"]["FUSION_MEMORY_SMOKE_MEMORY_URL"], "http://127.0.0.1:8765")

    def test_configured_adapter_smoke_command_exit_zero_without_json_is_unverified(self) -> None:
        completed = CompletedProcess(args=["fake-smoke"], returncode=0, stdout="ok\n", stderr="")
        with (
            patch("tools.agent_runtime_smoke._host_available", return_value=(True, "host ok")),
            patch("tools.agent_runtime_smoke._plugin_available", return_value=(True, "plugin ok")),
            patch("subprocess.run", return_value=completed),
            patch.dict(os.environ, {"FUSION_MEMORY_OPENCLAW_SMOKE_COMMAND": "fake-smoke"}, clear=True),
        ):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertFalse(report["ok"])
        self.assertFalse(report["write_smoke"])
        self.assertFalse(report["retrieve_smoke"])
        self.assertIn("did not print JSON", report["message"])

    def test_configured_adapter_smoke_command_nonzero_json_preserves_safe_message(self) -> None:
        completed = CompletedProcess(
            args=["fake-smoke"],
            returncode=1,
            stdout='{"write_smoke": false, "retrieve_smoke": false, "message": "Run fusion-memory doctor."}',
            stderr="Traceback secret detail",
        )
        with (
            patch("tools.agent_runtime_smoke._host_available", return_value=(True, "host ok")),
            patch("tools.agent_runtime_smoke._plugin_available", return_value=(True, "plugin ok")),
            patch("subprocess.run", return_value=completed),
            patch.dict(os.environ, {"FUSION_MEMORY_OPENCLAW_SMOKE_COMMAND": "fake-smoke"}, clear=True),
        ):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertFalse(report["ok"])
        self.assertFalse(report["write_smoke"])
        self.assertFalse(report["retrieve_smoke"])
        self.assertEqual(report["message"], "Run fusion-memory doctor.")
        self.assertNotIn("Traceback", json.dumps(report))

    def test_configured_adapter_smoke_command_nonzero_success_json_is_not_ok(self) -> None:
        completed = CompletedProcess(
            args=["fake-smoke"],
            returncode=1,
            stdout='{"write_smoke": true, "retrieve_smoke": true, "message": "adapter says ok"}',
            stderr="",
        )
        with (
            patch("tools.agent_runtime_smoke._host_available", return_value=(True, "host ok")),
            patch("tools.agent_runtime_smoke._plugin_available", return_value=(True, "plugin ok")),
            patch("subprocess.run", return_value=completed),
            patch.dict(os.environ, {"FUSION_MEMORY_OPENCLAW_SMOKE_COMMAND": "fake-smoke"}, clear=True),
        ):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertFalse(report["ok"])
        self.assertFalse(report["write_smoke"])
        self.assertFalse(report["retrieve_smoke"])
        self.assertEqual(report["message"], "adapter says ok")

    def test_hermes_repo_source_alone_is_not_runtime_plugin_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HERMES_HOME": tmp}, clear=True):
            available, message = smoke._plugin_available("hermes")

        self.assertFalse(available)
        self.assertIn("installed", message)


if __name__ == "__main__":
    unittest.main()
