from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_WORKSPACE = ROOT / "integrations" / "dolphin-fusion-memory" / "workspace"
SYNCED_FILES = [
    "systems/system.py",
    "tools/_client.py",
    "tools/_config.py",
    "tools/memory_add.py",
    "tools/memory_search.py",
    "tools/memory_answer_context.py",
    "skills/fusion-memory-setup/SKILL.md",
]
STRICT_SYNCED_FILES = [
    "systems/system.py",
    "tools/memory_add.py",
    "tools/memory_search.py",
    "tools/memory_answer_context.py",
    "skills/fusion-memory-setup/SKILL.md",
]
COPY_USAGE_PHRASES = [
    "Copy Into Another Workspace",
    "memory-only workspace",
    "memory_*.py",
    "tools/_client.py",
    "tools/_config.py",
    "skills/fusion-memory-setup",
]
AUTO_PERSISTENCE_PHRASES = [
    "Automatic History Persistence",
    "sync-haitun-history",
    "--workspace /path/to/haitun-workspace",
    "only when the agent calls",
    "without changing",
]
FIRST_USE_COMMANDS = [
    "git clone https://github.com/genuineknowledge/fusion-memory.git",
    "sh install.sh",
    "fusion-memory init --json",
    "fusion-memory start --json",
    "fusion-memory doctor --json",
    "PSI_MEMORY_BASE_URL=http://127.0.0.1:8700",
]
PORT_FALLBACK_PHRASE = "tries the next available local port"


def test_psi_agent_fusion_memory_example_matches_canonical_adapter() -> None:
    agent_root = os.environ.get("PSI_AGENT_ROOT")
    if not agent_root:
        pytest.skip(
            "Set PSI_AGENT_ROOT to the psi-agent checkout that contains examples/fusion-memory-workspace"
        )

    agent_workspace = Path(agent_root) / "examples" / "fusion-memory-workspace"
    missing = [path for path in SYNCED_FILES if not (agent_workspace / path).exists()]
    assert missing == []

    for relative_path in STRICT_SYNCED_FILES:
        canonical = (CANONICAL_WORKSPACE / relative_path).read_text()
        synced = (agent_workspace / relative_path).read_text()
        assert synced == canonical, (
            f"{relative_path} is out of sync with the canonical Haitun adapter"
        )

    for relative_path in ("tools/_client.py", "tools/_config.py"):
        synced = (agent_workspace / relative_path).read_text()
        assert "PSI_MEMORY_BASE_URL" in synced or "post_json" in synced


def test_psi_agent_fusion_memory_example_documents_copy_paste_usage() -> None:
    agent_root = os.environ.get("PSI_AGENT_ROOT")
    if not agent_root:
        pytest.skip(
            "Set PSI_AGENT_ROOT to the psi-agent checkout that contains examples/fusion-memory-workspace"
        )

    agent_readme = (
        Path(agent_root) / "examples" / "fusion-memory-workspace" / "README.md"
    )
    memory_readme = ROOT / "integrations" / "dolphin-fusion-memory" / "README.md"

    for phrase in COPY_USAGE_PHRASES:
        assert phrase in agent_readme.read_text()
        assert phrase in memory_readme.read_text()

    for phrase in AUTO_PERSISTENCE_PHRASES:
        assert phrase in agent_readme.read_text()
        assert phrase in memory_readme.read_text()


def test_fusion_memory_docs_match_workspace_setup_skill() -> None:
    quickstart = (ROOT / "docs" / "quickstart.md").read_text()
    memory_readme = (ROOT / "README.md").read_text()
    integration_readme = (
        ROOT / "integrations" / "dolphin-fusion-memory" / "README.md"
    ).read_text()

    for command in FIRST_USE_COMMANDS:
        assert command in quickstart
        assert command in memory_readme
        assert command in integration_readme

    for text in (quickstart, memory_readme, integration_readme):
        assert "wey-bo/fusion-memory" not in text
        assert "git@" not in text

    assert PORT_FALLBACK_PHRASE in quickstart
    assert PORT_FALLBACK_PHRASE in memory_readme
    assert PORT_FALLBACK_PHRASE in integration_readme

    assert "--memory-enabled" not in quickstart
    assert "--memory-enabled" not in integration_readme

    for text in (quickstart, memory_readme, integration_readme):
        assert "sync-haitun-history" in text
        assert "--gateway-url http://127.0.0.1:8080" not in text
        assert "--session-id <session-id>" in text


def test_first_use_setup_skill_uses_public_repository_and_documents_compromised_fallback() -> (
    None
):
    skill = (
        CANONICAL_WORKSPACE / "skills" / "fusion-memory-setup" / "SKILL.md"
    ).read_text()

    assert 'AGENT_DIR="/path/to/current-agent-directory"' in skill
    assert "not the workspace directory" in skill
    assert 'git clone https://github.com/genuineknowledge/fusion-memory.git "$AGENT_DIR/fusion-memory"' in skill
    assert "wey-bo" not in skill
    assert "git@" not in skill
    assert "identity" not in skill.lower()
    assert "authentication" not in skill.lower()
    assert "compromised" in skill
    assert "DASHSCOPE_API_KEY" in skill
    assert "SQLite plus bundled local Qwen vector models" in skill
    assert "Postgres/pgvector is optional" in skill
    assert "PASSIVE_SYNC_PID" in skill
    assert "kill -0" in skill
    assert "HTTP /add" in skill
    assert 'fusion-memory sync-haitun-history' in skill
    assert 'fusion-memory --db "$FM_DB" sync-haitun-history' not in skill
    assert "Persistence (Required After Start)" in skill
    assert "After verifying, immediately start the passive sync process" in skill
    start_section = skill.split("## Start And Verify", 1)[1].split(
        "## Persistence (Required After Start)", 1
    )[0]
    assert 'fusion-memory sync-haitun-history' in start_section
    assert "Git LFS pointer" in skill
    assert "git lfs pull" in skill
    assert "dependency installation failed" not in skill
    assert "Qwen runtime dependencies are unavailable" in skill
    assert "Postgres/pgvector is unavailable" not in skill


@pytest.mark.anyio
async def test_canonical_memory_workspace_prompt_asks_before_enabling_persistence() -> None:
    path = CANONICAL_WORKSPACE / "systems" / "system.py"
    spec = importlib.util.spec_from_file_location("canonical_memory_system", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    _install_psi_agent_yaml_stub()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("psi_agent._yaml", None)
        sys.modules.pop("psi_agent", None)

    prompt = await module.system_prompt_builder()

    assert "At the start of each new interactive session" in prompt
    assert "service is reachable" in prompt
    assert "current session passive sync process is running" in prompt
    assert "do not ask again" in prompt
    assert "ask the user whether to enable Fusion Memory persistent memory" in prompt
    assert "Do not wait for the first memory tool call" in prompt
    assert "cannot remember across sessions" in prompt
    assert "start and verify passive sync" in prompt
    assert "If the user declines" in prompt


def _install_psi_agent_yaml_stub() -> None:
    psi_agent = types.ModuleType("psi_agent")
    yaml_module = types.ModuleType("psi_agent._yaml")

    def parse_yaml_header(_text: str):
        return ({}, "")

    yaml_module.parse_yaml_header = parse_yaml_header
    sys.modules["psi_agent"] = psi_agent
    sys.modules["psi_agent._yaml"] = yaml_module
