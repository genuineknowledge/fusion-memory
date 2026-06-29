from __future__ import annotations

from pathlib import Path

import pytest

from psi_agent.session.tool_registry import ToolRegistry


@pytest.mark.anyio
async def test_dolphin_loader_discovers_only_public_tools() -> None:
    workspace = Path(__file__).resolve().parents[1] / "workspace"
    registry = await ToolRegistry.load(workspace / "tools", "fusion-memory-test")
    tools = registry.tools

    assert set(tools) == {"memory_add", "memory_search", "memory_answer_context"}
    assert {name for name in tools if registry.get(name) is not None} == set(tools)
    assert "source" not in tools["memory_add"].parameters.get("required", [])
    assert "source" in tools["memory_add"].parameters["properties"]
    assert "_config" not in tools
    assert "_client" not in tools
