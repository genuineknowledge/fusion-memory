from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


WINDOWS_RUNTIME_WHEEL_DEPENDENCIES = [
    "psycopg2-binary>=2.9",
    "torch>=2.5",
    "transformers>=4.51",
    "sentence-transformers>=3.4",
    "safetensors",
    "tokenizers",
    "hf-xet",
    "click",
    "typer",
]


@dataclass(frozen=True)
class InstallStep:
    step_name: str
    command: list[str]
    timeout_seconds: float = 900.0


@dataclass(frozen=True)
class StepResult:
    ok: bool
    step_name: str
    error: str = ""
    returncode: int = 0
    log_path: Path | None = None


def build_install_plan(script_dir: str | Path, venv_python: str | Path) -> list[InstallStep]:
    root = Path(script_dir).resolve()
    python = str(Path(venv_python))
    return [
        InstallStep(
            "installer bootstrap",
            [python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            timeout_seconds=300.0,
        ),
        InstallStep(
            "fusion memory base",
            [python, "-m", "pip", "install", "-e", str(root)],
            timeout_seconds=600.0,
        ),
        InstallStep(
            "local qwen runtime",
            [
                python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                *WINDOWS_RUNTIME_WHEEL_DEPENDENCIES,
            ],
            timeout_seconds=1800.0,
        ),
        InstallStep(
            "install readiness",
            [python, "-m", "fusion_memory.cli", "install-check", "--force"],
            timeout_seconds=300.0,
        ),
        InstallStep("doctor", [python, "-m", "fusion_memory.cli", "doctor"], timeout_seconds=180.0),
    ]


def ensure_venv(
    python_command: str,
    python_args: Sequence[str],
    venv_dir: str | Path,
    *,
    log_dir: str | Path,
) -> Path:
    venv = Path(venv_dir)
    python = _venv_python(venv)
    if python.exists():
        return python
    log_path = Path(log_dir) / "install.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_logged(
        [python_command, *python_args, "-m", "venv", str(venv)],
        log_path=log_path,
        timeout_seconds=300.0,
        step_name="memory environment",
    )
    if not result.ok:
        raise InstallerError(result)
    return python


def run_install_plan(
    script_dir: str | Path,
    venv_python: str | Path,
    *,
    log_dir: str | Path,
) -> StepResult:
    log_path = Path(log_dir) / "install.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    plan = build_install_plan(script_dir, venv_python)
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {step.step_name}...")
        result = run_logged(
            step.command,
            log_path=log_path,
            timeout_seconds=step.timeout_seconds,
            step_name=step.step_name,
            cwd=Path(script_dir),
        )
        if not result.ok:
            return result
    return StepResult(ok=True, step_name="complete", log_path=log_path)


def run_logged(
    command: Sequence[str],
    *,
    log_path: str | Path,
    timeout_seconds: float,
    step_name: str,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> StepResult:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    process_env = _normalize_process_env(env or os.environ)
    kwargs = _popen_kwargs(cwd=cwd, env=process_env)
    with path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {step_name} ===\n")
        log.write(_display_command(command) + "\n")
        try:
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **kwargs,
            )
            stdout, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if "process" in locals():
                _terminate_process_tree(process)
            log.write(f"{step_name} timed out after {timeout_seconds:g} seconds.\n")
            return StepResult(ok=False, step_name=step_name, error="timeout", log_path=path)
        except (OSError, subprocess.SubprocessError) as exc:
            log.write(f"{step_name} failed to start: {exc}\n")
            return StepResult(ok=False, step_name=step_name, error="failed_to_start", log_path=path)
        if stdout:
            log.write(stdout)
        if process.returncode != 0:
            log.write(f"{step_name} exited with code {process.returncode}.\n")
            return StepResult(
                ok=False,
                step_name=step_name,
                error="failed",
                returncode=int(process.returncode or 1),
                log_path=path,
            )
    return StepResult(ok=True, step_name=step_name, log_path=path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Fusion Memory on Windows")
    parser.add_argument("--python-command", required=True)
    parser.add_argument("--python-arg", action="append", default=[])
    parser.add_argument("--script-dir", required=True)
    parser.add_argument("--venv-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    script_dir = Path(args.script_dir).resolve()
    venv_dir = Path(args.venv_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    print("Installing Fusion Memory for Windows...")
    try:
        venv_python = ensure_venv(
            args.python_command,
            args.python_arg,
            venv_dir,
            log_dir=log_dir,
        )
        result = run_install_plan(script_dir, venv_python, log_dir=log_dir)
    except InstallerError as exc:
        result = exc.result

    if args.json:
        print(json.dumps(_result_payload(result, venv_dir), ensure_ascii=False, indent=2))
    elif result.ok:
        print("Fusion Memory is installed.")
        print(f"Environment: {venv_dir}")
        print(f"Log: {result.log_path}")
    else:
        print("Fusion Memory installation needs attention.")
        print(f"Step: {result.step_name}")
        print(f"Log: {result.log_path}")
    return 0 if result.ok else 1


class InstallerError(Exception):
    def __init__(self, result: StepResult) -> None:
        super().__init__(result.error)
        self.result = result


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _popen_kwargs(*, cwd: str | Path | None, env: dict[str, str]) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "cwd": str(cwd) if cwd is not None else None,
        "env": env,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        creationflags = 0
        for name in ("CREATE_NO_WINDOW", "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= int(getattr(subprocess, name, 0))
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    pid = getattr(process, "pid", None)
    if os.name == "nt" and pid:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                process.kill()
        return
    process.kill()


def _normalize_process_env(env: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in env.items():
        if value is None:
            continue
        out_key = "Path" if os.name == "nt" and key.lower() == "path" else str(key)
        dedupe_key = out_key.lower() if os.name == "nt" else out_key
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized[out_key] = str(value)
    return normalized


def _display_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _result_payload(result: StepResult, venv_dir: Path) -> dict[str, object]:
    return {
        "ok": result.ok,
        "step": result.step_name,
        "error": result.error,
        "returncode": result.returncode,
        "venv_dir": str(venv_dir),
        "log": str(result.log_path) if result.log_path else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
