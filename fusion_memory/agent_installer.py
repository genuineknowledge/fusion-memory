from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

VALID_TARGETS = ("dolphin", "openclaw", "hermes", "fusion-agent")
ROOT = Path(__file__).resolve().parents[1]
DOLPHIN_WORKSPACE = ROOT / "integrations" / "dolphin-fusion-memory" / "workspace"
DOLPHIN_SKILL = DOLPHIN_WORKSPACE / "skills" / "fusion-memory-setup" / "SKILL.md"
OPENCLAW_PLUGIN = ROOT / "integrations" / "openclaw-fusion-memory"
HERMES_PROVIDER = ROOT / "integrations" / "hermes-fusion-memory"
_INSTALL_ERROR = "Install failed. Check permissions and run fusion-memory doctor."


def install_agent(target: str, *, dry_run: bool = False, home: str | Path | None = None) -> dict[str, Any]:
    try:
        targets = list(VALID_TARGETS) if target == "all" else [target]
        invalid = [item for item in targets if item not in VALID_TARGETS]
        if invalid:
            return {
                "ok": False,
                "message": "Unknown Agent target. Choose one of: all, dolphin, openclaw, hermes, fusion-agent.",
            }
        actions = [_action_for(item, home=home) for item in targets]
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "estimated_time": "1-3 minutes for adapter checks; 2-5 minutes if install actions are needed.",
                "check_first": "Run dry-run/status checks before installing; skip actions whose skip_if condition is already true.",
                "actions": actions,
            }
        results = []
        for action in actions:
            results.append(_run_action(action))
        return {"ok": all(item["ok"] for item in results), "actions": actions, "results": results}
    except (OSError, subprocess.SubprocessError, shutil.Error):
        return {"ok": False, "message": _INSTALL_ERROR}


def _action_for(target: str, *, home: str | Path | None = None) -> dict[str, Any]:
    smoke_command = _smoke_command(target)
    if target == "dolphin":
        return {
            "target": "dolphin",
            "workspace": str(DOLPHIN_WORKSPACE),
            "skill": str(DOLPHIN_SKILL),
            "sync_command": [
                "fusion-memory",
                "sync-haitun-history",
                "--workspace",
                str(DOLPHIN_WORKSPACE),
                "--session-id",
                "<session-id>",
                "--background",
                "--json",
            ],
            "smoke_command": smoke_command,
            "estimated_time": "under 1 minute",
            "skip_if": "The Haitun workspace already exposes memory tools and the fusion-memory-setup skill.",
            "message": "Use the Haitun workspace with memory_add, memory_search, memory_answer_context, and sync-haitun-history --background.",
        }
    if target == "openclaw":
        return {
            "target": "openclaw",
            "command": ["openclaw", "plugins", "install", "--link", str(OPENCLAW_PLUGIN)],
            "smoke_command": smoke_command,
            "estimated_time": "1-2 minutes",
            "skip_if": "OpenClaw reports the fusion-memory plugin is already linked to this path.",
            "message": "Install the external OpenClaw Fusion Memory plugin.",
        }
    if target == "hermes":
        hermes_home = _hermes_home(home)
        return {
            "target": "hermes",
            "source": str(HERMES_PROVIDER),
            "destination": str(hermes_home / "plugins" / "fusion_memory"),
            "smoke_command": smoke_command,
            "estimated_time": "under 1 minute",
            "skip_if": "The Hermes fusion_memory provider destination already matches this source.",
            "message": "Install the external Hermes Fusion Memory provider.",
        }
    return {
        "target": "fusion-agent",
        "path": str(_fusion_agent_root(home)),
        "smoke_command": smoke_command,
        "estimated_time": "under 1 minute",
        "skip_if": "Fusion-Agent already points PSI_MEMORY_BASE_URL at the running Fusion Memory service.",
        "message": "Fusion-Agent memory integration is in-repo; verify env PSI_MEMORY_BASE_URL and --memory-enabled.",
    }


def _run_action(action: dict[str, Any]) -> dict[str, Any]:
    target = action["target"]
    if target == "dolphin":
        workspace = Path(action["workspace"])
        skill = Path(action["skill"])
        ok = (workspace / "tools" / "memory_add.py").exists() and skill.exists()
        return {
            "target": target,
            "ok": ok,
            "message": (
                    "Haitun Fusion Memory workspace is ready. Start sync-haitun-history --background beside the Haitun session."
                if ok
                else "Haitun Fusion Memory workspace is incomplete. Reinstall Fusion Memory."
            ),
        }
    if target == "openclaw":
        try:
            command = action["command"]
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            return {
                "target": target,
                "ok": completed.returncode == 0,
                "message": "OpenClaw plugin installed." if completed.returncode == 0 else "OpenClaw plugin install failed. Run fusion-memory doctor.",
            }
        except (OSError, subprocess.SubprocessError):
            return {
                "target": target,
                "ok": False,
                "message": "OpenClaw plugin install failed. Confirm OpenClaw is installed and run fusion-memory doctor.",
            }
    if target == "hermes":
        return _install_hermes(action)
    try:
        return {"target": target, "ok": Path(action["path"]).exists(), "message": action["message"]}
    except OSError:
        return {
            "target": target,
            "ok": False,
            "message": "Fusion-Agent checkout could not be checked. Confirm the path and permissions.",
        }


def _hermes_home(home: str | Path | None = None) -> Path:
    return Path(home or os.getenv("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def _fusion_agent_root(home: str | Path | None = None) -> Path:
    if home is not None:
        return Path(home).expanduser() / "Fusion-Agent"
    env_root = os.getenv("FUSION_AGENT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / "Fusion-Agent"


def _smoke_command(target: str) -> list[str]:
    return [
        "python3",
        "tools/agent_runtime_smoke.py",
        "--target",
        target,
        "--memory-url",
        "http://127.0.0.1:8700",
        "--output",
        f".runtime/agent-smoke-{target}.json",
    ]


def _install_hermes(action: dict[str, Any]) -> dict[str, Any]:
    source = Path(action["source"])
    destination = Path(action["destination"])
    temporary = destination.with_name(f".{destination.name}.tmp")
    backup = destination.with_name(f".{destination.name}.backup")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _remove_path(temporary)
        _remove_path(backup)
        shutil.copytree(source, temporary)
        if destination.exists():
            destination.replace(backup)
        temporary.replace(destination)
        _remove_path(backup)
        return {"target": "hermes", "ok": True, "message": "Hermes provider installed."}
    except (OSError, shutil.Error):
        _remove_path_quietly(temporary)
        if backup.exists() and not destination.exists():
            try:
                backup.replace(destination)
            except OSError:
                pass
        return {
            "target": "hermes",
            "ok": False,
            "message": "Hermes provider install failed. Existing install was left in place.",
        }


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _remove_path_quietly(path: Path) -> None:
    try:
        _remove_path(path)
    except OSError:
        pass
