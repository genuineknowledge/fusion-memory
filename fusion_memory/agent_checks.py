from __future__ import annotations

from pathlib import Path
from typing import Any

from fusion_memory.agent_installer import FUSION_AGENT_ROOT, HERMES_PROVIDER, OPENCLAW_PLUGIN, VALID_TARGETS


def check_agent(target: str, *, home: str | Path | None = None) -> dict[str, Any]:
    if target not in VALID_TARGETS:
        return {"ok": False, "message": "Unknown Agent target. Choose one of: openclaw, hermes, fusion-agent."}
    if target == "openclaw":
        ok = (OPENCLAW_PLUGIN / "openclaw.plugin.json").exists()
        return {
            "target": target,
            "ok": ok,
            "message": "OpenClaw Fusion Memory plugin files are present." if ok else "OpenClaw plugin files are missing. Reinstall Fusion Memory.",
        }
    if target == "hermes":
        ok = (HERMES_PROVIDER / "__init__.py").exists()
        return {
            "target": target,
            "ok": ok,
            "message": "Hermes Fusion Memory provider files are present." if ok else "Hermes provider files are missing. Reinstall Fusion Memory.",
        }
    ok = FUSION_AGENT_ROOT.exists()
    return {
        "target": target,
        "ok": ok,
        "message": "Fusion-Agent checkout is present. Start psi-agent session with --memory-enabled." if ok else "Fusion-Agent checkout was not found.",
    }
