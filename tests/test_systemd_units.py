from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"


def test_mcp_unit_is_restartable():
    text = (SYSTEMD / "fusion-memory-mcp.service").read_text(encoding="utf-8")
    assert "Restart=on-failure" in text
    assert "RestartSec=5" in text
    assert "WantedBy=default.target" in text
    assert "EnvironmentFile=" in text
    assert "StandardOutput=journal" in text


def test_model_units_are_restartable_and_load_instance_environment():
    for name, command in (
        ("fusion-memory-embedding@.service", "embedding-server"),
        ("fusion-memory-reranker@.service", "reranker-server"),
    ):
        text = (SYSTEMD / name).read_text(encoding="utf-8")
        assert "EnvironmentFile=%h/.config/fusion-memory/" in text
        assert command in text
        assert "Restart=on-failure" in text
        assert "RestartSec=5" in text
        assert "WantedBy=default.target" in text


def test_history_sync_unit_uses_environment_file_not_token_argument():
    text = (SYSTEMD / "fusion-memory-history-sync@.service").read_text(encoding="utf-8")
    assert "EnvironmentFile=%h/.config/fusion-memory/history-sync-%i.env" in text
    assert "--token" not in text
    assert "--workspace ${FUSION_MEMORY_HAITUN_WORKSPACE}" in text
    assert "--session-id ${FUSION_MEMORY_SESSION_ID}" in text
    assert "Restart=on-failure" in text
    assert "WantedBy=default.target" in text
    assert "Wants=fusion-memory-mcp.service" in text
    assert "Requires=fusion-memory-mcp.service" not in text


def test_readme_labels_legacy_local_paths_and_avoids_plaintext_dsns():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "legacy compatibility/development only" in text
    assert "not production" in text
    assert "postgresql://user:pass@" not in text
    assert "$FUSION_MEMORY_PG_DSN" in text


def test_health_timer_restarts_after_mcp_and_models():
    service = (SYSTEMD / "fusion-memory-health.service").read_text(encoding="utf-8")
    timer = (SYSTEMD / "fusion-memory-health.timer").read_text(encoding="utf-8")
    assert "ExecStart=%h/.local/bin/fusion-memory health --restart-unhealthy" in service
    assert "After=fusion-memory-mcp.service" in service
    assert "OnUnitActiveSec=" in timer
    assert "Persistent=true" in timer
    assert "WantedBy=timers.target" in timer
