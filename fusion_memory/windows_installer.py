from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import shutil
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib import request


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
MODELSCOPE_HUB_DEPENDENCY = "modelscope-hub>=0.1.6"
MODELSCOPE_MODEL_SPECS = [
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B"),
    ("Qwen/Qwen3-Reranker-0.6B", "Qwen3-Reranker-0.6B"),
]
MODELSCOPE_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.py",
    "README.md",
]
MODEL_SAFETENSORS_MIN_BYTES = 100 * 1024 * 1024
TOKENIZER_JSON_MIN_BYTES = 100 * 1024
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
COMPATIBLE_PYTHON_PROBE = (
    "import sys, sysconfig; "
    "text = ' '.join(str(x) for x in (sys.version, sys.executable, sysconfig.get_platform())).lower(); "
    "compatible = sys.version_info[:2] in ((3, 11), (3, 12)) and not any(token in text for token in ('msys', 'mingw', 'ucrt64')); "
    "sys.exit(0 if compatible else 1)"
)


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


def build_install_plan(
    script_dir: str | Path,
    venv_python: str | Path,
    *,
    log_dir: str | Path | None = None,
) -> list[InstallStep]:
    root = Path(script_dir).resolve()
    logs = Path(log_dir).resolve() if log_dir is not None else root / ".fusion-memory-logs"
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
            "modelscope downloader",
            [
                python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                MODELSCOPE_HUB_DEPENDENCY,
            ],
            timeout_seconds=300.0,
        ),
        InstallStep(
            "local qwen models",
            [
                python,
                "-m",
                "fusion_memory.windows_installer",
                "--download-models-only",
                "--script-dir",
                str(root),
                "--log-dir",
                str(logs),
            ],
            timeout_seconds=3600.0,
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
    if _is_windows_host() and not _python_is_compatible(
        python_command,
        python_args,
        log_path=log_path,
    ):
        uv = ensure_uv(Path(venv_dir).resolve().parent, log_dir=log_dir)
        result = run_logged(
            [str(uv), "python", "install", "3.12", "--managed-python", "--no-progress"],
            log_path=log_path,
            timeout_seconds=900.0,
            step_name="managed python",
        )
        if not result.ok:
            raise InstallerError(result)
        result = run_logged(
            [
                str(uv),
                "venv",
                "--python",
                "3.12",
                "--managed-python",
                "--seed",
                str(venv),
            ],
            log_path=log_path,
            timeout_seconds=300.0,
            step_name="memory environment",
        )
        if not result.ok:
            raise InstallerError(result)
    else:
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
    plan = build_install_plan(script_dir, venv_python, log_dir=log_dir)
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
    parser.add_argument("--download-models-only", action="store_true")
    parser.add_argument("--python-command")
    parser.add_argument("--python-arg", action="append", default=[])
    parser.add_argument("--script-dir", required=True)
    parser.add_argument("--venv-dir")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    script_dir = Path(args.script_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    if args.download_models_only:
        result = download_qwen_models(script_dir, log_dir=log_dir)
        if args.json:
            print(json.dumps(_download_result_payload(result), ensure_ascii=False, indent=2))
        elif result.ok:
            print("Fusion Memory local Qwen models are ready.")
        else:
            print("Fusion Memory model download needs attention.")
            print(f"Step: {result.step_name}")
            print(f"Log: {result.log_path}")
        return 0 if result.ok else 1

    if not args.python_command or not args.venv_dir:
        parser.error("--python-command and --venv-dir are required unless --download-models-only is used")

    venv_dir = Path(args.venv_dir).resolve()
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


def ensure_uv(script_dir: str | Path, *, log_dir: str | Path) -> Path:
    env_uv = os.getenv("FUSION_MEMORY_UV_BIN")
    if env_uv and Path(env_uv).is_file():
        return Path(env_uv)
    tools_dir = Path(script_dir) / ".fusion-memory-tools"
    local_uv = tools_dir / ("uv.exe" if _is_windows_host() else "uv")
    if local_uv.is_file():
        return local_uv
    found = shutil.which("uv")
    if found and (not _is_windows_host() or Path(found).name.lower() == "uv.exe"):
        return Path(found)
    if not _is_windows_host():
        raise InstallerError(
            StepResult(
                ok=False,
                step_name="uv bootstrap",
                error="uv_unavailable",
                log_path=Path(log_dir) / "install.log",
            )
        )
    return _download_uv(tools_dir, log_dir=log_dir)


def download_qwen_models(script_dir: str | Path, *, log_dir: str | Path) -> StepResult:
    root = Path(script_dir).resolve()
    log_path = Path(log_dir) / "install.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope_hub import HubApi  # type: ignore[import-not-found]
    except Exception as exc:
        _append_log(log_path, "local qwen models", f"ModelScope Hub is unavailable: {exc}\n")
        return StepResult(
            ok=False,
            step_name="local qwen models",
            error="modelscope_missing",
            log_path=log_path,
        )

    api = HubApi()
    models_root = root / "models"
    for repo_id, local_name in MODELSCOPE_MODEL_SPECS:
        local_dir = models_root / local_name
        status = _model_dir_status(local_dir)
        if status["ok"]:
            _append_log(log_path, "local qwen models", f"{local_name} already ready.\n")
            continue
        if local_dir.exists():
            try:
                shutil.rmtree(local_dir)
            except OSError as exc:
                _append_log(
                    log_path,
                    "local qwen models",
                    f"{local_name} cleanup failed before download: {exc}\n",
                )
                return StepResult(
                    ok=False,
                    step_name="local qwen models",
                    error="model_cleanup_failed",
                    log_path=log_path,
                )
        _append_log(log_path, "local qwen models", f"Downloading {repo_id} from ModelScope...\n")
        try:
            api.download_repo(
                repo_id,
                "model",
                local_dir=str(local_dir),
                allow_patterns=list(MODELSCOPE_ALLOW_PATTERNS),
            )
        except Exception as exc:
            _append_log(log_path, "local qwen models", f"{repo_id} download failed: {exc}\n")
            return StepResult(
                ok=False,
                step_name="local qwen models",
                error="modelscope_download_failed",
                log_path=log_path,
            )
        status = _model_dir_status(local_dir)
        if not status["ok"]:
            _append_log(
                log_path,
                "local qwen models",
                f"{local_name} is still incomplete: {status['message']}\n",
            )
            return StepResult(
                ok=False,
                step_name="local qwen models",
                error="model_incomplete",
                log_path=log_path,
            )
    return StepResult(ok=True, step_name="local qwen models", log_path=log_path)


class InstallerError(Exception):
    def __init__(self, result: StepResult) -> None:
        super().__init__(result.error)
        self.result = result


def _venv_python(venv_dir: Path) -> Path:
    if _is_windows_host():
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
    if _is_windows_host() and pid:
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


def _python_is_compatible(
    python_command: str,
    python_args: Sequence[str],
    *,
    log_path: Path,
) -> bool:
    result = run_logged(
        [python_command, *python_args, "-c", COMPATIBLE_PYTHON_PROBE],
        log_path=log_path,
        timeout_seconds=30.0,
        step_name="python compatibility",
    )
    return result.ok


def _download_uv(tools_dir: Path, *, log_dir: str | Path) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "install.log"
    archive = tools_dir / "uv.zip"
    local_uv = tools_dir / "uv.exe"
    url = os.getenv("FUSION_MEMORY_UV_DOWNLOAD_URL") or _default_uv_download_url()
    _append_log(log_path, "uv bootstrap", f"Downloading uv from {url}\n")
    try:
        with request.urlopen(url, timeout=120.0) as response:
            archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tools_dir)
        for candidate in tools_dir.rglob("uv.exe"):
            if candidate != local_uv:
                shutil.copy2(candidate, local_uv)
            break
    except Exception as exc:
        _append_log(log_path, "uv bootstrap", f"uv download failed: {exc}\n")
        raise InstallerError(
            StepResult(
                ok=False,
                step_name="uv bootstrap",
                error="uv_unavailable",
                log_path=log_path,
            )
        ) from exc
    if not local_uv.is_file():
        _append_log(log_path, "uv bootstrap", "uv.exe was not found in the downloaded archive.\n")
        raise InstallerError(
            StepResult(
                ok=False,
                step_name="uv bootstrap",
                error="uv_unavailable",
                log_path=log_path,
            )
        )
    return local_uv


def _default_uv_download_url() -> str:
    machine = platform.machine().lower()
    target = "aarch64-pc-windows-msvc" if machine in {"arm64", "aarch64"} else "x86_64-pc-windows-msvc"
    return f"https://github.com/astral-sh/uv/releases/latest/download/uv-{target}.zip"


def _is_windows_host() -> bool:
    if os.name == "nt":
        return True
    system = platform.system().lower()
    if system.startswith(("windows", "msys", "mingw", "cygwin")):
        return True
    return os.environ.get("OS", "").lower() == "windows_nt"


def _model_dir_status(path: Path) -> dict[str, object]:
    path = path.expanduser()
    if not path.is_dir():
        return {"ok": False, "message": f"{path.name} directory is missing"}
    checks = [
        _model_file_status(path / "model.safetensors", min_bytes=MODEL_SAFETENSORS_MIN_BYTES, parse_json=False),
        _model_file_status(path / "config.json", min_bytes=2, parse_json=True),
        _model_file_status(path / "tokenizer.json", min_bytes=TOKENIZER_JSON_MIN_BYTES, parse_json=False),
    ]
    failed = [check for check in checks if not check["ok"]]
    if failed:
        return {
            "ok": False,
            "message": "; ".join(str(check["message"]) for check in failed),
            "files": checks,
        }
    return {"ok": True, "message": "ready", "files": checks}


def _model_file_status(path: Path, *, min_bytes: int, parse_json: bool) -> dict[str, object]:
    if not path.is_file():
        return {"ok": False, "path": str(path), "message": f"{path.name} is missing"}
    if _is_lfs_pointer(path):
        return {"ok": False, "path": str(path), "message": f"{path.name} is a Git LFS pointer"}
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {"ok": False, "path": str(path), "message": f"{path.name} cannot be inspected: {exc}"}
    if size < min_bytes:
        return {"ok": False, "path": str(path), "size": size, "message": f"{path.name} is too small ({size} bytes)"}
    if parse_json:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {"ok": False, "path": str(path), "size": size, "message": f"{path.name} is not valid JSON: {exc.__class__.__name__}"}
    return {"ok": True, "path": str(path), "size": size, "message": "ready"}


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(256).startswith(LFS_POINTER_PREFIX)
    except OSError:
        return False


def _append_log(log_path: Path, step_name: str, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {step_name} ===\n")
        log.write(message)


def _normalize_process_env(env: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in env.items():
        if value is None:
            continue
        out_key = "Path" if _is_windows_host() and key.lower() == "path" else str(key)
        dedupe_key = out_key.lower() if _is_windows_host() else out_key
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


def _download_result_payload(result: StepResult) -> dict[str, object]:
    return {
        "ok": result.ok,
        "step": result.step_name,
        "error": result.error,
        "returncode": result.returncode,
        "log": str(result.log_path) if result.log_path else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
