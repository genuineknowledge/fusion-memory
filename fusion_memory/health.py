from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from typing import Any, Protocol

import anyio

from fusion_memory.mcp_client import MemoryMcpClient


class HealthRuntime(Protocol):
    def postgres_health(self) -> dict[str, object]: ...

    def embedding_health(self) -> dict[str, object]: ...

    def reranker_health(self) -> dict[str, object]: ...

    def background_health(self) -> dict[str, object]: ...


def health_report(runtime: HealthRuntime) -> dict[str, dict[str, object] | bool]:
    """Collect bounded, secret-free health results from a runtime adapter."""
    checks: dict[str, dict[str, object]] = {}
    for name in ("postgres", "embedding", "reranker", "background"):
        try:
            result = getattr(runtime, f"{name}_health")()
            if not isinstance(result, dict):
                raise TypeError("health check must return a dictionary")
            checks[name] = dict(result)
        except Exception as exc:
            checks[name] = {"ok": False, "error": type(exc).__name__}
    return {"ok": all(bool(check.get("ok")) for check in checks.values()), **checks}


def live_model_health(background: dict[str, object]) -> dict[str, dict[str, object]]:
    """Translate the authenticated MCP supervisor snapshot into timer report checks."""
    pools = background.get("model_pools")
    if not isinstance(pools, dict):
        return {}
    result: dict[str, dict[str, object]] = {}
    for name in ("embedding", "reranker"):
        if name not in pools:
            result[name] = {
                "ok": True,
                "configured": False,
                "healthy_endpoints": 0,
                "endpoints": [],
                "failed_units": [],
            }
            continue
        snapshots = pools[name]
        if not isinstance(snapshots, list):
            snapshots = []
        healthy_count = sum(1 for endpoint in snapshots if isinstance(endpoint, dict) and endpoint.get("healthy"))
        result[name] = {
            "ok": bool(snapshots) and healthy_count == len(snapshots),
            "configured": True,
            "healthy_endpoints": healthy_count,
            "endpoints": snapshots,
            "failed_units": _failed_units(name, snapshots),
        }
    return result


class ProductionHealthRuntime:
    """Health adapter around the production MCP runtime and bounded PG pool."""

    def __init__(self, runtime: Any, postgres_pool: Any, *, timeout_seconds: float = 5.0) -> None:
        self.runtime = runtime
        self.postgres_pool = postgres_pool
        self.timeout_seconds = max(0.1, float(timeout_seconds))

    def postgres_health(self) -> dict[str, object]:
        started = time.perf_counter()
        with self.postgres_pool.connection(self.timeout_seconds) as connection:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT 1")
                row = cursor.fetchone() if callable(getattr(cursor, "fetchone", None)) else (1,)
            finally:
                close = getattr(cursor, "close", None)
                if callable(close):
                    close()
                rollback = getattr(connection, "rollback", None)
                if callable(rollback):
                    rollback()
        return {"ok": bool(row and row[0] == 1), "latency_ms": (time.perf_counter() - started) * 1000}

    def embedding_health(self) -> dict[str, object]:
        return self._endpoint_health(0, "embedding")

    def reranker_health(self) -> dict[str, object]:
        return self._endpoint_health(1, "reranker")

    def background_health(self) -> dict[str, object]:
        check = getattr(self.runtime, "background_health", None)
        if callable(check):
            result = check()
            if isinstance(result, dict):
                return dict(result)
        return {"ok": False, "supervisor_alive": False, "error": "supervisor_state_unavailable"}

    def _endpoint_health(self, index: int, name: str) -> dict[str, object]:
        pools = tuple(getattr(self.runtime, "endpoint_pools", ()))
        if index >= len(pools):
            return {"ok": False, "healthy_endpoints": 0, "endpoints": [], "error": f"{name}_pool_missing"}
        pool = pools[index]
        if not callable(getattr(pool, "healthy_endpoints", None)):
            return {"ok": False, "healthy_endpoints": 0, "endpoints": [], "failed_units": [], "error": f"{name}_pool_missing"}
        try:
            healthy = list(pool.healthy_endpoints())
            snapshot = list(pool.snapshot()) if callable(getattr(pool, "snapshot", None)) else []
            return {
                "ok": bool(healthy),
                "healthy_endpoints": len(healthy),
                "endpoints": snapshot,
                "failed_units": _failed_units(name, snapshot),
            }
        except Exception as exc:
            return {"ok": False, "healthy_endpoints": 0, "endpoints": [], "failed_units": [], "error": type(exc).__name__}


async def mcp_health_check(
    url: str | None = None,
    token: str | None = None,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Authenticate, initialize, ping, and read the local MCP supervisor state."""
    endpoint = (url if url is not None else os.getenv("FUSION_MEMORY_HEALTH_MCP_URL", "")).strip()
    bearer = (token if token is not None else os.getenv("FUSION_MEMORY_HEALTH_TOKEN", "")).strip()
    if not endpoint or not bearer:
        return {"ok": False, "error": "health_mcp_configuration_missing"}
    client = MemoryMcpClient(endpoint, bearer, "__health__", "__health__", timeout_seconds=timeout_seconds)
    try:
        ping = getattr(client, "ping", None)
        if callable(ping):
            await ping()
        else:
            session = await client._ensure_session()
            await session.send_ping()
        health_result = await client.call_tool("memory_health", {})
        background = None
        if isinstance(health_result, dict):
            nested = health_result.get("result")
            if isinstance(nested, dict):
                background = nested.get("background")
            if background is None:
                background = health_result.get("background")
        if not isinstance(background, dict):
            return {"ok": False, "error": "health_background_unavailable"}
        return {"ok": bool(health_result.get("ok")) and bool(background.get("ok")), "background": background}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}
    finally:
        await client.close()


_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")


def restart_unhealthy_units(report: dict[str, Any]) -> dict[str, object]:
    """Restart only failed units explicitly named in protected environment config."""
    requested: list[str] = []
    skipped: list[dict[str, str]] = []
    configuration_errors: list[str] = []
    for component, variable in (("embedding", "FUSION_MEMORY_EMBEDDING_UNITS"), ("reranker", "FUSION_MEMORY_RERANKER_UNITS")):
        if bool(report.get(component, {}).get("ok")):
            continue
        units, error = _configured_units_safe(variable)
        if error:
            configuration_errors.append(error)
            continue
        details = report.get(component, {})
        raw_failed = details.get("failed_units")
        if not raw_failed:
            raw_failed = _failed_units(component, list(details.get("endpoints", []))) if isinstance(details.get("endpoints", []), list) else []
        allowed = set(units)
        failed = [unit for unit in raw_failed if isinstance(unit, str) and _UNIT_RE.fullmatch(unit) and unit in allowed]
        if failed:
            requested.extend(failed)
        elif len(units) == 1:
            requested.extend(units)
        elif len(units) > 1:
            skipped.append({"component": component, "reason": "unit_mapping_unreliable"})
    if any(not bool(report.get(name, {}).get("ok")) for name in ("postgres", "background", "mcp")):
        units, error = _configured_units_safe("FUSION_MEMORY_MCP_UNIT")
        if error:
            configuration_errors.append(error)
        elif units:
            requested.extend(units)
        else:
            skipped.append({"component": "mcp", "reason": "unit_not_configured"})

    restarts: list[dict[str, object]] = []
    for unit in dict.fromkeys(requested):
        completed = subprocess.run(["systemctl", "--user", "restart", unit], check=False, capture_output=True, text=True)
        restarts.append({"unit": unit, "ok": completed.returncode == 0, "returncode": completed.returncode})
    return {
        "ok": not configuration_errors and all(bool(item["ok"]) for item in restarts),
        "restarted": restarts,
        "skipped": skipped,
        "configuration_errors": configuration_errors,
    }


def _configured_units(name: str) -> list[str]:
    return _configured_units_safe(name)[0]


def _configured_units_safe(name: str) -> tuple[list[str], str | None]:
    raw = os.getenv(name, "")
    try:
        values = shlex.split(raw) if name != "FUSION_MEMORY_MCP_UNIT" else [raw.strip()] if raw.strip() else []
    except ValueError:
        return [], f"{name}_malformed_quoting"
    return [value for value in values if _UNIT_RE.fullmatch(value)], None


def _failed_units(name: str, snapshot: list[dict[str, object]]) -> list[str]:
    variable = "FUSION_MEMORY_EMBEDDING_UNITS" if name == "embedding" else "FUSION_MEMORY_RERANKER_UNITS"
    units, _ = _configured_units_safe(variable)
    failed: list[str] = []
    if len(snapshot) == len(units) and units:
        for unit, endpoint in zip(units, snapshot):
            if "unit" not in endpoint and not bool(endpoint.get("healthy")):
                failed.append(unit)
    for endpoint in snapshot:
        explicit = endpoint.get("unit")
        if isinstance(explicit, str) and _UNIT_RE.fullmatch(explicit) and not bool(endpoint.get("healthy")):
            failed.append(explicit)
    return list(dict.fromkeys(failed))


def run_health(*, restart_unhealthy: bool = False) -> dict[str, object]:
    from fusion_memory.mcp_runtime import runtime_from_env

    runtime, pool = runtime_from_env()
    try:
        report: dict[str, object] = health_report(
            ProductionHealthRuntime(
                runtime,
                pool,
                timeout_seconds=float(os.getenv("FUSION_MEMORY_HEALTH_TIMEOUT_SECONDS", "5")),
            )
        )
        report["mcp"] = anyio.run(mcp_health_check)
        if isinstance(report["mcp"], dict) and isinstance(report["mcp"].get("background"), dict):
            report["background"] = dict(report["mcp"]["background"])
            report.update(live_model_health(report["background"]))
        report["ok"] = all(bool(report[name].get("ok")) for name in ("postgres", "embedding", "reranker", "background")) and bool(report["mcp"]["ok"])
        if restart_unhealthy:
            report["restarts"] = restart_unhealthy_units(report)
        return report
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
        elif callable(getattr(pool, "close", None)):
            pool.close()
