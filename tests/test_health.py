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
    with (
        patch(
            "fusion_memory.cli.run_health",
            return_value={"ok": True, "mcp": {"ok": True}},
            create=True,
        ) as run_health,
        patch("fusion_memory.cli.sys.argv", ["fusion-memory", "health", "--json"]),
    ):
        assert cli.main() == 0
    run_health.assert_called_once_with(restart_unhealthy=False)
    output = capsys.readouterr().out
    assert token not in output
    assert json.loads(output)["mcp"]["ok"] is True


def test_run_health_uses_live_mcp_model_snapshots(monkeypatch):
    from fusion_memory import health

    monkeypatch.setenv("FUSION_MEMORY_PG_DSN", "postgresql://memory")
    live_background = {
        "ok": False,
        "supervisor_alive": True,
        "model_pools": {
            "embedding": [{"healthy": False}, {"healthy": True}],
            "reranker": [{"healthy": True}],
        },
    }
    pool = _FakePostgresPool()
    monkeypatch.setattr(health, "PostgresConnectionPool", lambda *args, **kwargs: pool, raising=False)
    monkeypatch.setattr(
        health,
        "mcp_health_check",
        AsyncMock(return_value={"ok": False, "background": live_background}),
    )

    report = health.run_health()

    assert report["embedding"]["ok"] is False
    assert report["embedding"]["healthy_endpoints"] == 1
    assert report["reranker"]["ok"] is True
    assert pool.closed is True


class _FakeCursor:
    def execute(self, statement: str) -> None:
        assert statement == "SELECT 1"

    def fetchone(self):
        return (1,)

    def close(self) -> None:
        pass


class _FakeConnection:
    def cursor(self) -> _FakeCursor:
        return _FakeCursor()

    def rollback(self) -> None:
        pass


class _FakePostgresPool:
    def __init__(self) -> None:
        self.closed = False

    def connection(self, timeout_seconds: float):
        from contextlib import nullcontext

        return nullcontext(_FakeConnection())

    def close(self) -> None:
        self.closed = True


def test_run_health_never_builds_runtime_or_model_adapters(monkeypatch):
    from fusion_memory import health
    from fusion_memory import mcp_runtime
    from fusion_memory.core import runtime_config

    pool = _FakePostgresPool()

    def forbidden(*args, **kwargs):
        raise AssertionError("model runtime builder called")

    monkeypatch.setenv("FUSION_MEMORY_PG_DSN", "postgresql://memory")
    monkeypatch.setattr(health, "PostgresConnectionPool", lambda *args, **kwargs: pool, raising=False)
    monkeypatch.setattr(health, "mcp_health_check", AsyncMock(return_value={"ok": False, "error": "offline"}))
    monkeypatch.setattr(mcp_runtime, "runtime_from_env", forbidden)
    monkeypatch.setattr(runtime_config, "_build_embedder", forbidden)
    monkeypatch.setattr(runtime_config, "_build_reranker", forbidden)

    report = health.run_health()

    assert report["postgres"]["ok"] is True
    assert pool.closed is True


def test_run_health_queries_mcp_when_postgres_pool_construction_fails(monkeypatch):
    from fusion_memory import health

    mcp_check = AsyncMock(return_value={"ok": False, "error": "offline"})

    def fail_pool(*args, **kwargs):
        raise RuntimeError("contains-sensitive-dsn")

    monkeypatch.setenv("FUSION_MEMORY_PG_DSN", "postgresql://secret@memory")
    monkeypatch.setattr(health, "PostgresConnectionPool", fail_pool, raising=False)
    monkeypatch.setattr(health, "mcp_health_check", mcp_check)

    report = health.run_health()

    mcp_check.assert_awaited_once()
    assert report["postgres"] == {"ok": False, "error": "RuntimeError"}
    assert "secret" not in json.dumps(report)


@pytest.mark.parametrize(
    "mcp_result",
    [
        {"ok": False, "error": "offline"},
        {"ok": False, "background": {"ok": False, "model_pools": []}},
        {
            "ok": False,
            "background": {
                "ok": False,
                "model_pools": {"embedding": "invalid", "reranker": []},
            },
        },
        {
            "ok": False,
            "background": {
                "ok": False,
                "model_pools": {"embedding": [{"healthy": True}], "reranker": ["invalid"]},
            },
        },
    ],
)
def test_run_health_reports_unknown_models_without_valid_live_snapshot(monkeypatch, mcp_result):
    from fusion_memory import health

    pool = _FakePostgresPool()
    monkeypatch.setenv("FUSION_MEMORY_PG_DSN", "postgresql://memory")
    monkeypatch.setattr(health, "PostgresConnectionPool", lambda *args, **kwargs: pool, raising=False)
    monkeypatch.setattr(health, "mcp_health_check", AsyncMock(return_value=mcp_result))

    report = health.run_health()

    unknown = {
        "ok": False,
        "configured": None,
        "healthy_endpoints": 0,
        "endpoints": [],
        "failed_units": [],
        "error": "live_mcp_snapshot_unavailable",
    }
    assert report["embedding"] == unknown
    assert report["reranker"] == unknown


def test_restart_unhealthy_does_not_guess_model_unit_without_live_snapshot(monkeypatch):
    from fusion_memory.health import restart_unhealthy_units

    monkeypatch.setenv("FUSION_MEMORY_EMBEDDING_UNITS", "fusion-memory-embedding.service")
    with patch("fusion_memory.health.subprocess.run") as run:
        result = restart_unhealthy_units(
            {
                "embedding": {
                    "ok": False,
                    "configured": None,
                    "healthy_endpoints": 0,
                    "endpoints": [],
                    "failed_units": [],
                    "error": "live_mcp_snapshot_unavailable",
                },
                "reranker": {"ok": True},
                "postgres": {"ok": True},
                "background": {"ok": True},
                "mcp": {"ok": True},
            }
        )

    run.assert_not_called()
    assert result["restarted"] == []
    assert {"component": "embedding", "reason": "live_mcp_snapshot_unavailable"} in result["skipped"]


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


def test_restart_unhealthy_keeps_failed_units_inside_allowlist(monkeypatch):
    from fusion_memory.health import restart_unhealthy_units

    monkeypatch.setenv("FUSION_MEMORY_EMBEDDING_UNITS", "fusion-memory-embedding@a.service fusion-memory-embedding@b.service")
    with patch("fusion_memory.health.subprocess.run") as run:
        result = restart_unhealthy_units(
            {
                "embedding": {"ok": False, "failed_units": ["untrusted.service"]},
                "reranker": {"ok": True},
                "postgres": {"ok": True},
                "background": {"ok": True},
            }
        )
    run.assert_not_called()
    assert result["restarted"] == []
    assert result["skipped"]


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
