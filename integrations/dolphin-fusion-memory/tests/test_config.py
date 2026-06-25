from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "workspace" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from _config import build_memory_config


def test_build_memory_config_defaults() -> None:
    cfg = build_memory_config({})
    assert cfg.base_url == "http://127.0.0.1:8700"
    assert cfg.workspace_id == "dolphin"
    assert cfg.agent_id == "dolphin"
    assert cfg.app_id == "dolphin"
    assert cfg.timeout_seconds == 2.0
    assert cfg.allow_cross_session is True
    assert cfg.scope["session_id"] is None


def test_build_memory_config_clamps_timeout_and_scope(monkeypatch) -> None:
    monkeypatch.setenv("PSI_MEMORY_BASE_URL", "http://example.invalid:1234")
    monkeypatch.setenv("PSI_MEMORY_WORKSPACE_ID", "ws-a")
    monkeypatch.setenv("PSI_MEMORY_USER_ID", "webo")
    monkeypatch.setenv("PSI_MEMORY_AGENT_ID", "dolphin")
    monkeypatch.setenv("PSI_MEMORY_SESSION_ID", "session-7")
    monkeypatch.setenv("PSI_MEMORY_TIMEOUT_SECONDS", "9.5")

    cfg = build_memory_config()
    assert cfg.base_url == "http://example.invalid:1234"
    assert cfg.timeout_seconds == 5.0
    assert cfg.allow_cross_session is False
    assert cfg.scope == {
        "workspace_id": "ws-a",
        "user_id": "webo",
        "agent_id": "dolphin",
        "session_id": "session-7",
        "app_id": "dolphin",
    }
