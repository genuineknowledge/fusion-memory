from __future__ import annotations

import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from fusion_memory import windows_installer


class WindowsInstallerTests(unittest.TestCase):
    def test_install_plan_separates_base_install_from_qwen_wheel_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_python = root / ".fusion-memory-venv" / "Scripts" / "python.exe"

            plan = windows_installer.build_install_plan(root, venv_python)

        commands = [step.command for step in plan]
        self.assertIn([str(venv_python), "-m", "pip", "install", "-e", str(root)], commands)
        self.assertIn("--only-binary=:all:", commands[2])
        self.assertIn("torch>=2.5", commands[2])
        self.assertIn("safetensors", commands[2])
        self.assertIn("tokenizers", commands[2])
        self.assertIn("hf-xet", commands[2])
        self.assertNotIn(f"{root}[postgres,qwen]", " ".join(" ".join(command) for command in commands))

    def test_run_logged_kills_process_tree_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "install.log"
            process = _FakeProcess(returncode=None)

            with (
                patch("fusion_memory.windows_installer.subprocess.Popen", return_value=process),
                patch("fusion_memory.windows_installer._terminate_process_tree") as terminate,
            ):
                result = windows_installer.run_logged(
                    ["python", "-m", "pip", "install", "slow-package"],
                    log_path=log_path,
                    timeout_seconds=0.01,
                    step_name="qwen wheel preflight",
                )
            log_text = log_path.read_text(encoding="utf-8")

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "timeout")
        terminate.assert_called_once_with(process)
        self.assertIn("qwen wheel preflight timed out", log_text)

    def test_main_prints_concise_failure_with_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_dir = root / ".fusion-memory-venv"
            log_dir = root / ".fusion-memory-logs"
            stdout = StringIO()

            with (
                patch("fusion_memory.windows_installer.ensure_venv", return_value=venv_dir / "Scripts" / "python.exe"),
                patch(
                    "fusion_memory.windows_installer.run_install_plan",
                    return_value=windows_installer.StepResult(
                        ok=False,
                        step_name="local qwen runtime",
                        error="failed",
                        log_path=log_dir / "install.log",
                    ),
                ),
                redirect_stdout(stdout),
            ):
                exit_code = windows_installer.main(
                    [
                        "--python-command",
                        "py",
                        "--python-arg",
                        "-3.12",
                        "--script-dir",
                        str(root),
                        "--venv-dir",
                        str(venv_dir),
                        "--log-dir",
                        str(log_dir),
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("Fusion Memory installation needs attention.", output)
        self.assertIn(str(log_dir / "install.log"), output)
        self.assertNotIn("safetensors", output)
        self.assertNotIn("tokenizers", output)
        self.assertNotIn("hf-xet", output)

    def test_normalize_process_env_deduplicates_windows_path(self) -> None:
        with patch("fusion_memory.windows_installer.os.name", "nt"):
            env = windows_installer._normalize_process_env(
                {
                    "Path": r"C:\Windows\System32",
                    "PATH": r"C:\msys64\ucrt64\bin",
                    "OTHER": "value",
                }
            )

        path_keys = [key for key in env if key.lower() == "path"]
        self.assertEqual(path_keys, ["Path"])
        self.assertEqual(env["Path"], r"C:\Windows\System32")


class _FakeProcess:
    def __init__(self, *, returncode: int | None) -> None:
        self.returncode = returncode
        self.pid = 12345

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout or 0)


if __name__ == "__main__":
    unittest.main()
