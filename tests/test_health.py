from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from fusion_memory.health import health_report

ROOT = Path(__file__).resolve().parents[1]


class FakeHealthRuntime:
    def postgres_health(self) -> dict[str, object]:
        return {"ok": True, "latency_ms": 1.0}

    def embedding_health(self) -> dict[str, object]:
        return {"ok": True, "healthy_endpoints": 2, "endpoints": []}

    def reranker_health(self) -> dict[str, object]:
        return {"ok": True, "healthy_endpoints": 1, "endpoints": []}

    def background_health(self) -> dict[str, object]:
        return {"ok": True, "tasks": []}


def test_health_reports_postgres_and_model_pool_status():
    result = health_report(FakeHealthRuntime())
    assert result["ok"] is True
    assert result["postgres"]["ok"] is True
    assert result["embedding"]["healthy_endpoints"] == 2
    assert result["reranker"]["healthy_endpoints"] == 1
    assert result["background"]["ok"] is True


def test_health_report_marks_failed_component_unhealthy():
    runtime = FakeHealthRuntime()
    runtime.reranker_health = lambda: {"ok": False, "healthy_endpoints": 0, "endpoints": []}
    result = health_report(runtime)
    assert result["ok"] is False
    assert result["reranker"]["ok"] is False


def test_cli_health_json_pings_mcp_without_printing_token(capsys, monkeypatch):
    from fusion_memory import cli

    token = "health-secret-token"
    monkeypatch.setenv("FUSION_MEMORY_HEALTH_MCP_URL", "http://127.0.0.1:8700/mcp")
    monkeypatch.setenv("FUSION_MEMORY_HEALTH_TOKEN", token)
    runtime = SimpleNamespace(close=lambda: None)
    pool = SimpleNamespace()
    with (
        patch("fusion_memory.cli.runtime_from_env", return_value=(runtime, pool)),
        patch("fusion_memory.cli.ProductionHealthRuntime", return_value=FakeHealthRuntime()),
        patch("fusion_memory.cli.mcp_health_check", new=AsyncMock(return_value={"ok": True})),
        patch("fusion_memory.cli.sys.argv", ["fusion-memory", "health", "--json"]),
    ):
        assert cli.main() == 0
    output = capsys.readouterr().out
    assert token not in output
    assert json.loads(output)["mcp"]["ok"] is True


def test_restart_unhealthy_restarts_only_failed_configured_units(monkeypatch):
    from fusion_memory.health import restart_unhealthy_units

    monkeypatch.setenv("FUSION_MEMORY_MCP_UNIT", "fusion-memory-mcp.service")
    monkeypatch.setenv("FUSION_MEMORY_EMBEDDING_UNITS", "fusion-memory-embedding@a.service fusion-memory-embedding@b.service")
    monkeypatch.setenv("FUSION_MEMORY_RERANKER_UNITS", "fusion-memory-reranker@a.service")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    with patch("fusion_memory.health.subprocess.run", side_effect=fake_run):
        result = restart_unhealthy_units(
            {"postgres": {"ok": True}, "embedding": {"ok": False}, "reranker": {"ok": True}, "background": {"ok": False}}
        )
    assert calls == [["systemctl", "--user", "restart", "fusion-memory-mcp.service"]]
    assert result["ok"] is True


def test_restart_unhealthy_maps_failed_endpoint_snapshot_to_unit(monkeypatch):
    from fusion_memory.health import restart_unhealthy_units

    monkeypatch.setenv("FUSION_MEMORY_EMBEDDING_UNITS", "fusion-memory-embedding@a.service fusion-memory-embedding@b.service")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    report = {
        "embedding": {
            "ok": False,
            "endpoints": [{"healthy": True}, {"healthy": False}],
            "failed_units": ["fusion-memory-embedding@b.service"],
        },
        "reranker": {"ok": True},
        "postgres": {"ok": True},
        "background": {"ok": True},
    }
    with patch("fusion_memory.health.subprocess.run", side_effect=fake_run):
        result = restart_unhealthy_units(report)
    assert calls == [["systemctl", "--user", "restart", "fusion-memory-embedding@b.service"]]
    assert result["ok"] is True


def test_restart_unhealthy_skips_unreliable_multi_unit_mapping(monkeypatch):
    from fusion_memory.health import restart_unhealthy_units

    monkeypatch.setenv("FUSION_MEMORY_EMBEDDING_UNITS", "fusion-memory-embedding@a.service fusion-memory-embedding@b.service")
    with patch("fusion_memory.health.subprocess.run") as run:
        result = restart_unhealthy_units(
            {
                "embedding": {"ok": False, "endpoints": [{"healthy": False}]},
                "reranker": {"ok": True},
                "postgres": {"ok": True},
                "background": {"ok": True},
            }
        )
    run.assert_not_called()
    assert result["restarted"] == []
    assert result["skipped"]


def test_restart_unhealthy_rejects_malformed_allowlist_quoting(monkeypatch):
    from fusion_memory.health import restart_unhealthy_units

    monkeypatch.setenv("FUSION_MEMORY_EMBEDDING_UNITS", "'unterminated")
    with patch("fusion_memory.health.subprocess.run") as run:
        result = restart_unhealthy_units(
            {
                "embedding": {"ok": False},
                "reranker": {"ok": True},
                "postgres": {"ok": True},
                "background": {"ok": True},
            }
        )
    run.assert_not_called()
    assert result["ok"] is False
    assert result["configuration_errors"]


def test_mcp_health_check_requires_background_liveness(monkeypatch):
    from fusion_memory import health

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def ping(self):
            return None

        async def call_tool(self, name, arguments):
            assert name == "memory_health"
            return {"ok": True, "result": {"background": {"ok": False}}}

        async def close(self):
            pass

    monkeypatch.setattr(health, "MemoryMcpClient", Client)
    result = __import__("anyio").run(health.mcp_health_check, "http://test/mcp", "token")
    assert result["ok"] is False
    assert result["background"]["ok"] is False


@pytest.mark.parametrize("name", ["fusion-memory-mcp.service", "fusion-memory-health.service"])
def test_units_do_not_contain_plaintext_secret(name):
    text = (ROOT / "deploy" / "systemd" / name).read_text(encoding="utf-8")
    assert "FUSION_MEMORY_HEALTH_TOKEN=" not in text
    assert "FUSION_MEMORY_TOKEN=" not in text
