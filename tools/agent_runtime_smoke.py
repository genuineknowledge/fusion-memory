from __future__ import annotations

import argparse
import asyncio
import importlib.util
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

    report.update(_run_builtin_adapter_smoke(target, memory_url=memory_url, timeout=timeout))
    return _normalize_report(target, report)


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
    if not parsed:
        if completed.returncode != 0:
            return {"message": f"{_display_name(target)} adapter runtime smoke failed. Run fusion-memory doctor."}
        return {
            "message": (
                f"{_display_name(target)} adapter runtime smoke exited successfully, but did not print JSON "
                "with explicit write_smoke and retrieve_smoke values."
            )
        }

    write_smoke = parsed.get("write_smoke") is True
    retrieve_smoke = parsed.get("retrieve_smoke") is True
    process_ok = completed.returncode == 0
    fallback_message = (
        f"{_display_name(target)} adapter runtime smoke completed."
        if process_ok and write_smoke and retrieve_smoke
        else f"{_display_name(target)} adapter runtime smoke did not explicitly verify write and retrieve."
    )
    return {
        "write_smoke": process_ok and write_smoke,
        "retrieve_smoke": process_ok and retrieve_smoke,
        "ok": process_ok and write_smoke and retrieve_smoke,
        "message": _safe_adapter_message(parsed.get("message"), fallback_message),
    }


def _parse_command_report(stdout: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_adapter_message(value: Any, fallback: str) -> str:
    message = str(value or fallback).strip()
    unsafe_markers = ("traceback", "exception", "error:", "errno", "econnrefused", "secret", "\n")
    if not message or any(marker in message.lower() for marker in unsafe_markers):
        return fallback
    return message[:240]


def _run_builtin_adapter_smoke(target: str, *, memory_url: str, timeout: int) -> dict[str, Any]:
    if target == "openclaw":
        return _run_openclaw_plugin_smoke(memory_url=memory_url, timeout=timeout)
    if target == "hermes":
        return _run_hermes_provider_smoke(memory_url=memory_url, timeout=timeout)
    return {"message": f"{_display_name(target)} adapter runtime smoke is not available. Run fusion-memory doctor."}


def _run_openclaw_plugin_smoke(*, memory_url: str, timeout: int) -> dict[str, Any]:
    node = shutil.which("node")
    smoke_script = REPO_ROOT / "integrations" / "openclaw-fusion-memory" / "smoke.mjs"
    if node is None or not smoke_script.exists():
        return {
            "message": (
                "OpenClaw Fusion Memory plugin smoke could not run. Install Node.js, reinstall the plugin, "
                "then run fusion-memory doctor."
            )
        }
    runtime_ok, runtime_message = _openclaw_runtime_tools_available(timeout=timeout)
    if not runtime_ok:
        return {"message": runtime_message}
    return _run_command_smoke("openclaw", [node, str(smoke_script)], memory_url=memory_url, timeout=timeout)


def _openclaw_runtime_tools_available(*, timeout: int) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["openclaw", "plugins", "inspect", "fusion-memory", "--runtime", "--json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(max(timeout, 1), 10),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return (
            False,
            "OpenClaw could not inspect the Fusion Memory plugin runtime. Reinstall the plugin, restart OpenClaw, then run fusion-memory doctor.",
        )
    if completed.returncode != 0:
        return (
            False,
            "OpenClaw did not load the Fusion Memory plugin runtime. Reinstall the plugin, restart OpenClaw, then run fusion-memory doctor.",
        )
    output = completed.stdout.lower()
    required = ("fusion_memory_store", "fusion_memory_search")
    if all(name in output for name in required):
        return True, "OpenClaw Fusion Memory plugin runtime tools are visible."
    return (
        False,
        "OpenClaw loaded the plugin, but Fusion Memory tools were not visible. Reinstall the plugin, restart OpenClaw, then run fusion-memory doctor.",
    )


def _run_hermes_provider_smoke(*, memory_url: str, timeout: int) -> dict[str, Any]:
    plugin_dir = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")).expanduser() / "plugins" / "fusion_memory"
    plugin_file = plugin_dir / "__init__.py"
    if not plugin_file.exists():
        return {
            "message": "Hermes Fusion Memory provider is not installed. Run fusion-memory install-agent --target hermes."
        }
    token = f"hermes-smoke-{uuid.uuid4().hex}"
    previous_env = {
        "FUSION_MEMORY_BASE_URL": os.environ.get("FUSION_MEMORY_BASE_URL"),
        "FUSION_MEMORY_TIMEOUT_SECONDS": os.environ.get("FUSION_MEMORY_TIMEOUT_SECONDS"),
    }
    try:
        os.environ["FUSION_MEMORY_BASE_URL"] = memory_url
        os.environ["FUSION_MEMORY_TIMEOUT_SECONDS"] = str(timeout)
        provider_cls = _load_hermes_provider_class(plugin_file)
        provider = provider_cls()
        provider.initialize("agent-runtime-smoke", agent_workspace="agent-runtime-smoke", user_id="smoke-user")
        write_payload = json.loads(provider.handle_tool_call("fusion_memory_store", {"content": f"Hermes runtime smoke token: {token}"}))
        retrieve_payload = json.loads(
            provider.handle_tool_call("fusion_memory_search", {"query": f"Find Hermes runtime smoke token {token}", "limit": 3})
        )
        write_smoke = write_payload.get("ok") is True and write_payload.get("saved") is True
        retrieve_smoke = token in json.dumps(retrieve_payload, ensure_ascii=False)
        return {
            "write_smoke": write_smoke,
            "retrieve_smoke": retrieve_smoke,
            "ok": write_smoke and retrieve_smoke,
            "message": (
                "Hermes adapter runtime smoke completed."
                if write_smoke and retrieve_smoke
                else "Hermes adapter runtime smoke did not verify write and retrieve. Run fusion-memory doctor."
            ),
        }
    except Exception:
        return {
            "message": (
                "Hermes adapter runtime smoke could not verify memory through the provider. "
                "Start Fusion Memory and run fusion-memory doctor."
            )
        }
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_hermes_provider_class(plugin_file: Path) -> Any:
    spec = importlib.util.spec_from_file_location("fusion_memory_hermes_smoke_provider", plugin_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("hermes provider could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FusionMemoryProvider


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
