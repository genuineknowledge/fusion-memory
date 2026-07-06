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

            plan = windows_installer.build_install_plan(
                root,
                venv_python,
                log_dir=root / ".fusion-memory-logs",
            )

        commands = [step.command for step in plan]
        self.assertIn([str(venv_python), "-m", "pip", "install", "-e", str(root)], commands)
        qwen_runtime = next(command for command in commands if "torch>=2.5" in command)
        self.assertIn("--only-binary=:all:", qwen_runtime)
        self.assertIn("safetensors", qwen_runtime)
        self.assertIn("tokenizers", qwen_runtime)
        self.assertIn("hf-xet", qwen_runtime)
        self.assertNotIn(f"{root}[postgres,qwen]", " ".join(" ".join(command) for command in commands))

    def test_install_plan_downloads_qwen_models_from_modelscope_before_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_python = root / ".fusion-memory-venv" / "Scripts" / "python.exe"

            plan = windows_installer.build_install_plan(
                root,
                venv_python,
                log_dir=root / ".fusion-memory-logs",
            )

        step_names = [step.step_name for step in plan]
        commands = [step.command for step in plan]
        downloader = next(command for command in commands if "modelscope-hub>=0.1.6" in command)
        self.assertIn("--only-binary=:all:", downloader)
        model_step = next(command for command in commands if "--download-models-only" in command)
        self.assertIn(str(root), model_step)
        self.assertLess(step_names.index("local qwen models"), step_names.index("install readiness"))
        self.assertNotIn("git lfs", " ".join(" ".join(command).lower() for command in commands))

    def test_download_qwen_models_uses_modelscope_model_ids_and_local_model_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[dict[str, object]] = []

            class FakeHubApi:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    calls.append({"init": kwargs})

                def download_repo(self, *args: object, **kwargs: object) -> Path:
                    calls.append({"args": args, "kwargs": kwargs})
                    local_dir = Path(kwargs["local_dir"])
                    _write_ready_model_dir(local_dir)
                    return local_dir

            with patch.dict("sys.modules", {"modelscope_hub": _fake_modelscope_hub(FakeHubApi)}):
                result = windows_installer.download_qwen_models(
                    root,
                    log_dir=root / ".fusion-memory-logs",
                )

        self.assertTrue(result.ok)
        download_calls = [call for call in calls if "args" in call]
        self.assertEqual(
            [call["args"][0] for call in download_calls],
            ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Reranker-0.6B"],
        )
        self.assertEqual([call["args"][1] for call in download_calls], ["model", "model"])
        self.assertEqual(
            [Path(call["kwargs"]["local_dir"]).name for call in download_calls],
            ["Qwen3-Embedding-0.6B", "Qwen3-Reranker-0.6B"],
        )
        for call in download_calls:
            self.assertIn("*.safetensors", call["kwargs"]["allow_patterns"])
            self.assertIn("*.json", call["kwargs"]["allow_patterns"])

    def test_download_qwen_models_fails_when_modelscope_result_is_still_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class FakeHubApi:
                def download_repo(self, *args: object, **kwargs: object) -> Path:
                    local_dir = Path(kwargs["local_dir"])
                    local_dir.mkdir(parents=True, exist_ok=True)
                    (local_dir / "config.json").write_text("{}", encoding="utf-8")
                    return local_dir

            with patch.dict("sys.modules", {"modelscope_hub": _fake_modelscope_hub(FakeHubApi)}):
                result = windows_installer.download_qwen_models(
                    root,
                    log_dir=root / ".fusion-memory-logs",
                )

        self.assertFalse(result.ok)
        self.assertEqual(result.step_name, "local qwen models")
        self.assertEqual(result.error, "model_incomplete")
        self.assertTrue(result.log_path)

    def test_download_qwen_models_removes_incomplete_pointer_directory_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pointer_dir = root / "models" / "Qwen3-Embedding-0.6B"
            pointer_dir.mkdir(parents=True)
            (pointer_dir / "model.safetensors").write_bytes(
                b"version https://git-lfs.github.com/spec/v1\n"
            )
            (pointer_dir / "stale.txt").write_text("stale", encoding="utf-8")
            outer = self

            class FakeHubApi:
                def download_repo(self, *args: object, **kwargs: object) -> Path:
                    local_dir = Path(kwargs["local_dir"])
                    outer.assertFalse((local_dir / "stale.txt").exists())
                    _write_ready_model_dir(local_dir)
                    return local_dir

            with patch.dict("sys.modules", {"modelscope_hub": _fake_modelscope_hub(FakeHubApi)}):
                result = windows_installer.download_qwen_models(
                    root,
                    log_dir=root / ".fusion-memory-logs",
                )

        self.assertTrue(result.ok)

    def test_ensure_venv_uses_uv_managed_python_when_bootstrap_python_is_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_dir = root / ".fusion-memory-venv"
            uv = root / ".fusion-memory-tools" / "uv.exe"

            def fake_run(command: list[str], **kwargs: object) -> windows_installer.StepResult:
                if command[:3] == ["python", "-c", windows_installer.COMPATIBLE_PYTHON_PROBE]:
                    return windows_installer.StepResult(ok=False, step_name="probe")
                if command[0] == str(uv) and command[1:3] == ["python", "install"]:
                    return windows_installer.StepResult(ok=True, step_name="managed python")
                if command[0] == str(uv) and command[1] == "venv":
                    (venv_dir / "Scripts").mkdir(parents=True, exist_ok=True)
                    (venv_dir / "Scripts" / "python.exe").write_text("", encoding="utf-8")
                    return windows_installer.StepResult(ok=True, step_name="memory environment")
                return windows_installer.StepResult(ok=False, step_name="unexpected")

            with (
                patch("fusion_memory.windows_installer._is_windows_host", return_value=True),
                patch("fusion_memory.windows_installer.ensure_uv", return_value=uv),
                patch("fusion_memory.windows_installer.run_logged", side_effect=fake_run) as run_logged,
            ):
                python = windows_installer.ensure_venv(
                    "python",
                    [],
                    venv_dir,
                    log_dir=root / ".fusion-memory-logs",
                )

        self.assertEqual(python, venv_dir / "Scripts" / "python.exe")
        commands = [call.args[0] for call in run_logged.call_args_list]
        self.assertIn([str(uv), "python", "install", "3.12", "--managed-python", "--no-progress"], commands)
        self.assertTrue(any(command[:3] == [str(uv), "venv", "--python"] for command in commands))

    def test_msys_python_on_windows_host_uses_windows_venv_and_uv_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_dir = root / ".fusion-memory-venv"
            uv = root / ".fusion-memory-tools" / "uv.exe"

            def fake_run(command: list[str], **kwargs: object) -> windows_installer.StepResult:
                if command[:3] == ["python", "-c", windows_installer.COMPATIBLE_PYTHON_PROBE]:
                    return windows_installer.StepResult(ok=False, step_name="probe")
                if command[0] == str(uv) and command[1:3] == ["python", "install"]:
                    return windows_installer.StepResult(ok=True, step_name="managed python")
                if command[0] == str(uv) and command[1] == "venv":
                    (venv_dir / "Scripts").mkdir(parents=True, exist_ok=True)
                    (venv_dir / "Scripts" / "python.exe").write_text("", encoding="utf-8")
                    return windows_installer.StepResult(ok=True, step_name="memory environment")
                return windows_installer.StepResult(ok=False, step_name="unexpected")

            with (
                patch("fusion_memory.windows_installer.os.name", "posix"),
                patch("fusion_memory.windows_installer.platform.system", return_value="MSYS_NT-10.0-22631"),
                patch("fusion_memory.windows_installer.ensure_uv", return_value=uv),
                patch("fusion_memory.windows_installer.run_logged", side_effect=fake_run),
            ):
                python = windows_installer.ensure_venv(
                    "python",
                    [],
                    venv_dir,
                    log_dir=root / ".fusion-memory-logs",
                )

        self.assertEqual(python, venv_dir / "Scripts" / "python.exe")

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


def _fake_modelscope_hub(hub_api: object) -> object:
    class Module:
        HubApi = hub_api

    return Module()


def _write_ready_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"0" * (windows_installer.MODEL_SAFETENSORS_MIN_BYTES + 1))
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_bytes(b"0" * (windows_installer.TOKENIZER_JSON_MIN_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
