from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_memory.agent_installer import VALID_TARGETS, _action_for  # noqa: E402


DEFAULT_TIMEOUT_SECONDS = 5
SMOKE_COMMAND_ENV = {
    "openclaw": "FUSION_MEMORY_OPENCLAW_SMOKE_COMMAND",
    "hermes": "FUSION_MEMORY_HERMES_SMOKE_COMMAND",
    "fusion-agent": "FUSION_MEMORY_FUSION_AGENT_SMOKE_COMMAND",
}
REQUIRED_REPORT_FIELDS = (
    "target",
    "host_available",
    "plugin_available",
    "write_smoke",
    "retrieve_smoke",
    "ok",
    "message",
)


def run_smoke(target: str, *, memory_url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    report = _base_report(target)
    if target not in VALID_TARGETS:
        report["message"] = "Unknown Agent target. Choose one of: openclaw, hermes, fusion-agent."
        return report

    host_available, host_message = _host_available(target)
    report["host_available"] = host_available

    if not host_available:
        report["message"] = host_message
        return report

    plugin_available, plugin_message = _plugin_available(target)
    report["plugin_available"] = plugin_available
    if not plugin_available:
        report["message"] = plugin_message
        return report

    command = _smoke_command_from_env(target)
    if command is not None:
        report.update(_run_command_smoke(target, command, memory_url=memory_url, timeout=timeout))
        return _normalize_report(target, report)

    if target == "fusion-agent":
        report.update(_run_fusion_agent_adapter_smoke(memory_url=memory_url, timeout=timeout))
        return _normalize_report(target, report)

    report["message"] = (
        f"{_display_name(target)} host and installed plugin were found, but runtime smoke is not configured or verified. "
        f"Set {SMOKE_COMMAND_ENV[target]} to an adapter-level smoke command that prints JSON with "
        "write_smoke and retrieve_smoke."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a beginner-safe Fusion Memory Agent runtime smoke check")
    parser.add_argument("--target", required=True, choices=VALID_TARGETS)
    parser.add_argument("--memory-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    report = _normalize_report(args.target, run_smoke(args.target, memory_url=args.memory_url, timeout=args.timeout))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report.get("ok") else 1


def _base_report(target: str) -> dict[str, Any]:
    return {
        "target": target,
        "host_available": False,
        "plugin_available": False,
        "write_smoke": False,
        "retrieve_smoke": False,
        "ok": False,
        "message": "",
    }


def _host_available(target: str) -> tuple[bool, str]:
    if target == "openclaw":
        if shutil.which("openclaw"):
            return True, "OpenClaw host is available."
        return False, "OpenClaw was not found on PATH. Install OpenClaw, then run fusion-memory install-agent --target openclaw."
    if target == "hermes":
        if shutil.which("hermes"):
            return True, "Hermes host is available."
        return False, "Hermes was not found on PATH. Install Hermes, then run fusion-memory install-agent --target hermes."

    root = Path(_action_for("fusion-agent")["path"])
    if root.exists():
        return True, "Fusion-Agent checkout is available."
    return False, f"Fusion-Agent checkout was not found at {root}. Set FUSION_AGENT_ROOT or clone Fusion-Agent before running the smoke."


def _plugin_available(target: str) -> tuple[bool, str]:
    if target == "openclaw":
        return _openclaw_plugin_available()
    if target == "hermes":
        hermes_plugin = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")).expanduser() / "plugins" / "fusion_memory"
        ok = (hermes_plugin / "__init__.py").exists()
        return (
            ok,
            "Hermes Fusion Memory provider is installed in the runtime plugin directory."
            if ok
            else "Hermes Fusion Memory provider is not installed in the runtime plugin directory. Run fusion-memory install-agent --target hermes.",
        )

    root = Path(_action_for("fusion-agent")["path"])
    ok = (root / "src" / "psi_agent" / "memory" / "tool_api.py").exists()
    return (
        ok,
        "Fusion-Agent memory integration can be checked in the Fusion-Agent checkout."
        if ok
        else "Fusion-Agent memory integration files are missing. Set FUSION_AGENT_ROOT or clone Fusion-Agent.",
    )


def _openclaw_plugin_available() -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["openclaw", "plugins", "list"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return (
            False,
            "OpenClaw Fusion Memory plugin could not be verified in the runtime. Run fusion-memory install-agent --target openclaw.",
        )
    output = completed.stdout.lower()
    ok = completed.returncode == 0 and "fusion" in output and "memory" in output
    return (
        ok,
        "OpenClaw Fusion Memory plugin is visible to the OpenClaw runtime."
        if ok
        else "OpenClaw Fusion Memory plugin is not visible to the OpenClaw runtime. Run fusion-memory install-agent --target openclaw.",
    )


def _smoke_command_from_env(target: str) -> list[str] | None:
    value = os.getenv(SMOKE_COMMAND_ENV[target], "").strip()
    if not value:
        return None
    try:
        command = shlex.split(value)
    except ValueError:
        return []
    return command


def _run_command_smoke(target: str, command: list[str], *, memory_url: str, timeout: int) -> dict[str, Any]:
    if not command:
        return {
            "message": f"{SMOKE_COMMAND_ENV[target]} is set but could not be parsed. Use a plain command line without shell-only syntax.",
        }
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={**os.environ, "FUSION_MEMORY_SMOKE_MEMORY_URL": memory_url},
        )
    except subprocess.TimeoutExpired:
        return {"message": f"{_display_name(target)} adapter runtime smoke timed out. Run fusion-memory doctor."}
    except (OSError, subprocess.SubprocessError):
        return {"message": f"{_display_name(target)} adapter runtime smoke could not be started. Check the command and run fusion-memory doctor."}

    parsed = _parse_command_report(completed.stdout)
    if completed.returncode != 0:
        return {"message": f"{_display_name(target)} adapter runtime smoke failed. Run fusion-memory doctor."}
    if not parsed:
        return {
            "message": (
                f"{_display_name(target)} adapter runtime smoke exited successfully, but did not print JSON "
                "with explicit write_smoke and retrieve_smoke values."
            )
        }

    write_smoke = parsed.get("write_smoke") is True
    retrieve_smoke = parsed.get("retrieve_smoke") is True
    return {
        "write_smoke": write_smoke,
        "retrieve_smoke": retrieve_smoke,
        "ok": write_smoke and retrieve_smoke,
        "message": (
            str(parsed.get("message") or f"{_display_name(target)} adapter runtime smoke completed.")
            if write_smoke and retrieve_smoke
            else str(
                parsed.get("message")
                or f"{_display_name(target)} adapter runtime smoke did not explicitly verify write and retrieve."
            )
        ),
    }


def _parse_command_report(stdout: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_fusion_agent_adapter_smoke(*, memory_url: str, timeout: int) -> dict[str, Any]:
    root = Path(_action_for("fusion-agent")["path"])
    source = root / "src"
    token = f"fusion-agent-smoke-{uuid.uuid4().hex}"
    previous_path = list(sys.path)
    old_env = {key: os.environ.get(key) for key in _fusion_agent_smoke_env(memory_url, timeout)}
    try:
        sys.path.insert(0, str(source))
        os.environ.update(_fusion_agent_smoke_env(memory_url, timeout))
        return asyncio.run(_run_fusion_agent_adapter_smoke_async(token))
    except Exception:
        return {
            "message": (
                "Fusion-Agent adapter runtime smoke could not verify memory through the adapter. "
                "Start Fusion Memory and run fusion-memory doctor."
            )
        }
    finally:
        sys.path[:] = previous_path
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _run_fusion_agent_adapter_smoke_async(token: str) -> dict[str, Any]:
    from psi_agent.memory.tool_api import memory_read, memory_write

    write_result = await memory_write(f"Fusion-Agent runtime smoke token: {token}")
    read_result = await memory_read(f"Find Fusion-Agent runtime smoke token {token}", limit=3)
    write_smoke = "Fusion Memory saved." in write_result
    retrieve_smoke = token in read_result
    return {
        "write_smoke": write_smoke,
        "retrieve_smoke": retrieve_smoke,
        "ok": write_smoke and retrieve_smoke,
        "message": (
            "Fusion-Agent adapter runtime smoke completed."
            if write_smoke and retrieve_smoke
            else "Fusion-Agent adapter runtime smoke did not verify write and retrieve. Run fusion-memory doctor."
        ),
    }


def _fusion_agent_smoke_env(memory_url: str, timeout: int) -> dict[str, str]:
    return {
        "PSI_MEMORY_BASE_URL": memory_url,
        "PSI_MEMORY_TIMEOUT_SECONDS": str(timeout),
        "PSI_MEMORY_WORKSPACE_ID": "agent-runtime-smoke",
        "PSI_MEMORY_USER_ID": "smoke-user",
        "PSI_MEMORY_AGENT_ID": "fusion-agent",
    }


def _normalize_report(target: str, report: dict[str, Any]) -> dict[str, Any]:
    normalized = _base_report(str(report.get("target") or target))
    normalized.update({key: report[key] for key in REQUIRED_REPORT_FIELDS if key in report})
    normalized["host_available"] = normalized["host_available"] is True
    normalized["plugin_available"] = normalized["plugin_available"] is True
    normalized["write_smoke"] = normalized["write_smoke"] is True
    normalized["retrieve_smoke"] = normalized["retrieve_smoke"] is True
    normalized["ok"] = normalized["ok"] is True
    normalized["message"] = str(normalized["message"])
    return normalized


def _display_name(target: str) -> str:
    return {
        "openclaw": "OpenClaw",
        "hermes": "Hermes",
        "fusion-agent": "Fusion-Agent",
    }[target]


if __name__ == "__main__":
    raise SystemExit(main())
