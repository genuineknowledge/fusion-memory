from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

VALID_TARGETS = ("openclaw", "hermes", "fusion-agent")
ROOT = Path(__file__).resolve().parents[1]
OPENCLAW_PLUGIN = ROOT / "integrations" / "openclaw-fusion-memory"
HERMES_PROVIDER = ROOT / "integrations" / "hermes-fusion-memory"
FUSION_AGENT_ROOT = Path("/public/home/wwb/Fusion-Agent")


def install_agent(target: str, *, dry_run: bool = False, home: str | Path | None = None) -> dict[str, Any]:
    targets = list(VALID_TARGETS) if target == "all" else [target]
    invalid = [item for item in targets if item not in VALID_TARGETS]
    if invalid:
        return {
            "ok": False,
            "message": "Unknown Agent target. Choose one of: all, openclaw, hermes, fusion-agent.",
        }
    actions = [_action_for(item, home=home) for item in targets]
    if dry_run:
        return {"ok": True, "dry_run": True, "actions": actions}
    results = []
    for action in actions:
        results.append(_run_action(action))
    return {"ok": all(item["ok"] for item in results), "actions": actions, "results": results}


def _action_for(target: str, *, home: str | Path | None = None) -> dict[str, Any]:
    if target == "openclaw":
        return {
            "target": "openclaw",
            "command": ["openclaw", "plugins", "install", "--link", str(OPENCLAW_PLUGIN)],
            "message": "Install the external OpenClaw Fusion Memory plugin.",
        }
    if target == "hermes":
        hermes_home = Path(home or os.getenv("HERMES_HOME") or Path.home() / ".hermes")
        return {
            "target": "hermes",
            "source": str(HERMES_PROVIDER),
            "destination": str(hermes_home / "plugins" / "fusion_memory"),
            "message": "Install the external Hermes Fusion Memory provider.",
        }
    return {
        "target": "fusion-agent",
        "path": str(FUSION_AGENT_ROOT),
        "message": "Fusion-Agent memory integration is in-repo; verify env PSI_MEMORY_BASE_URL and --memory-enabled.",
    }


def _run_action(action: dict[str, Any]) -> dict[str, Any]:
    target = action["target"]
    if target == "openclaw":
        command = action["command"]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return {
            "target": target,
            "ok": completed.returncode == 0,
            "message": "OpenClaw plugin installed." if completed.returncode == 0 else "OpenClaw plugin install failed. Run fusion-memory doctor.",
        }
    if target == "hermes":
        source = Path(action["source"])
        destination = Path(action["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            else:
                shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return {"target": target, "ok": True, "message": "Hermes provider installed."}
    return {"target": target, "ok": Path(action["path"]).exists(), "message": action["message"]}
