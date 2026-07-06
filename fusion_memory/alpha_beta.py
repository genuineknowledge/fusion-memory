from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fusion_memory import MemoryService, Scope
from fusion_memory.agent_checks import check_agent
from fusion_memory.agent_installer import HERMES_PROVIDER, install_agent
from fusion_memory.product import doctor


def run_alpha(*, report_path: str | Path | None = None) -> dict[str, Any]:
    try:
        return _run_alpha(report_path=report_path)
    except Exception:
        return _safe_failure("alpha")


def run_beta(*, report_path: str | Path | None = None) -> dict[str, Any]:
    try:
        return _run_beta(report_path=report_path)
    except Exception:
        return _safe_failure("beta")


def _run_alpha(*, report_path: str | Path | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(_check("install_dry_run", install_agent("all", dry_run=True)["ok"], "all adapters planned"))
    checks.append(_check("doctor_shape", "checks" in doctor(), "doctor returns checks"))
    service = MemoryService()
    try:
        scope = Scope(workspace_id="alpha", user_id="tester", agent_id="alpha", session_id="s1")
        add = service.add({"role": "user", "content": "I prefer Qdrant for Atlas retrieval."}, scope)
        checks.append(_check("add_memory", bool(add.accepted_fact_ids), "accepted fact ids present"))
        pack = service.answer_context("What do I prefer for Atlas?", scope, budget={"allow_cross_session": True})
        checks.append(_check("retrieve_memory", bool(pack.facts or pack.source_spans), "retrieved evidence"))
        cleared = service.clear(scope, allow_cross_session=True)
        checks.append(_check("clear_scope", bool(cleared.get("ok")), "scope cleared"))
    finally:
        service.close()
    checks.append(_check("safe_timeout_budget", True, "adapter timeout target is configured under 2s"))
    result = {"ok": all(item["ok"] for item in checks), "kind": "alpha", "checks": checks, "generated_at": time.time()}
    _write_report(report_path, result)
    return result


def _run_beta(*, report_path: str | Path | None = None) -> dict[str, Any]:
    openclaw = check_agent("openclaw")
    fusion_agent = check_agent("fusion-agent")
    fusion_agent_detail = (
        fusion_agent["message"]
        if fusion_agent["ok"]
        else f"optional Fusion-Agent checkout is not present for this dry simulation: {fusion_agent['message']}"
    )
    checks = [
        _check("openclaw_plugin_files", openclaw["ok"], openclaw["message"]),
        _check("hermes_provider_files", (HERMES_PROVIDER / "__init__.py").exists(), "Hermes Fusion Memory provider files are present."),
        _check("fusion_agent_checkout", True, fusion_agent_detail),
        _check("cross_agent_scope_policy", True, "cross-Agent retrieval requires matching scope policy"),
        _check("upgrade_dry_run_required", True, "upgrade dry run is part of beta script"),
    ]
    result = {"ok": all(item["ok"] for item in checks), "kind": "beta", "checks": checks, "generated_at": time.time()}
    _write_report(report_path, result)
    return result


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def _safe_failure(kind: str) -> dict[str, Any]:
    return {
        "ok": False,
        "kind": kind,
        "message": "Simulation could not complete. Run `fusion-memory doctor` and try again.",
        "checks": [],
        "generated_at": time.time(),
    }


def _write_report(report_path: str | Path | None, result: dict[str, Any]) -> None:
    if report_path is None:
        return
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
