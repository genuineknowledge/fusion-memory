from __future__ import annotations

import os
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
COPY_USAGE_PHRASES = [
    "Copy Into Another Workspace",
    "memory-only workspace",
    "memory_*.py",
    "tools/_client.py",
    "tools/_config.py",
    "skills/fusion-memory-setup",
]
FIRST_USE_COMMANDS = [
    "git clone https://github.com/genuineknowledge/fusion-memory.git",
    "sh install.sh",
    "fusion-memory init --local-test --json",
    "fusion-memory start --json",
    "fusion-memory doctor --json",
    "PSI_MEMORY_BASE_URL=http://127.0.0.1:8700",
]
PORT_FALLBACK_PHRASE = "tries the next available local port"


def test_psi_agent_fusion_memory_example_matches_canonical_adapter() -> None:
    agent_root = os.environ.get("PSI_AGENT_ROOT")
    if not agent_root:
        pytest.skip("Set PSI_AGENT_ROOT to the psi-agent checkout that contains examples/fusion-memory-workspace")

    agent_workspace = Path(agent_root) / "examples" / "fusion-memory-workspace"
    missing = [path for path in SYNCED_FILES if not (agent_workspace / path).exists()]
    assert missing == []

    for relative_path in SYNCED_FILES:
        canonical = (CANONICAL_WORKSPACE / relative_path).read_text()
        synced = (agent_workspace / relative_path).read_text()
        assert synced == canonical, f"{relative_path} is out of sync with the canonical Dolphin adapter"


def test_psi_agent_fusion_memory_example_documents_copy_paste_usage() -> None:
    agent_root = os.environ.get("PSI_AGENT_ROOT")
    if not agent_root:
        pytest.skip("Set PSI_AGENT_ROOT to the psi-agent checkout that contains examples/fusion-memory-workspace")

    agent_readme = Path(agent_root) / "examples" / "fusion-memory-workspace" / "README.md"
    memory_readme = ROOT / "integrations" / "dolphin-fusion-memory" / "README.md"

    for phrase in COPY_USAGE_PHRASES:
        assert phrase in agent_readme.read_text()
        assert phrase in memory_readme.read_text()


def test_fusion_memory_docs_match_workspace_setup_skill() -> None:
    quickstart = (ROOT / "docs" / "quickstart.md").read_text()
    memory_readme = (ROOT / "README.md").read_text()
    integration_readme = (ROOT / "integrations" / "dolphin-fusion-memory" / "README.md").read_text()

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


def test_first_use_setup_skill_uses_public_repository_and_documents_compromised_fallback() -> None:
    skill = (CANONICAL_WORKSPACE / "skills" / "fusion-memory-setup" / "SKILL.md").read_text()

    assert "git clone https://github.com/genuineknowledge/fusion-memory.git" in skill
    assert "wey-bo" not in skill
    assert "git@" not in skill
    assert "identity" not in skill.lower()
    assert "authentication" not in skill.lower()
    assert "compromised" in skill
    assert "DASHSCOPE_API_KEY" in skill
