from __future__ import annotations

from pathlib import Path
from typing import Any

from fusion_memory.agent_installer import DOLPHIN_SKILL, DOLPHIN_WORKSPACE, OPENCLAW_PLUGIN, VALID_TARGETS, _action_for


def check_agent(target: str, *, home: str | Path | None = None) -> dict[str, Any]:
    if target not in VALID_TARGETS:
        return {"ok": False, "message": "Unknown Agent target. Choose one of: dolphin, openclaw, hermes, fusion-agent."}
    if target == "dolphin":
        ok = (DOLPHIN_WORKSPACE / "tools" / "memory_add.py").exists() and DOLPHIN_SKILL.exists()
        return {
            "target": target,
            "ok": ok,
            "workspace": str(DOLPHIN_WORKSPACE),
            "skill": str(DOLPHIN_SKILL),
            "message": (
                "Haitun Fusion Memory workspace is present. Run sync-haitun-history --background beside the Haitun session for passive persistence."
                if ok
                else "Haitun Fusion Memory workspace is incomplete. Run fusion-memory install-agent --target dolphin."
            ),
        }
    if target == "openclaw":
        ok = (OPENCLAW_PLUGIN / "openclaw.plugin.json").exists()
        return {
            "target": target,
            "ok": ok,
            "path": str(OPENCLAW_PLUGIN),
            "message": "OpenClaw Fusion Memory plugin files are present." if ok else "OpenClaw plugin files are missing. Reinstall Fusion Memory.",
        }
    if target == "hermes":
        destination = Path(_action_for("hermes", home=home)["destination"])
        ok = (destination / "__init__.py").exists()
        return {
            "target": target,
            "ok": ok,
            "path": str(destination),
            "message": "Hermes Fusion Memory provider is installed." if ok else "Hermes provider is not installed. Run fusion-memory install-agent --target hermes.",
        }
    root = Path(_action_for("fusion-agent", home=home)["path"])
    ok = root.exists()
    return {
        "target": target,
        "ok": ok,
        "path": str(root),
        "message": "Fusion-Agent checkout is present. Start psi-agent session with --memory-enabled." if ok else "Fusion-Agent checkout was not found.",
    }
