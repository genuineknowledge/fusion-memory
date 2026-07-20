from fusion_memory.api.service import MemoryService
from fusion_memory.core.auth import AllowAllAuthorizer, AuthorizationError, Authorizer
from fusion_memory.core.config import MemoryConfig
from fusion_memory.core.models import Scope

__all__ = [
    "MemoryService",
    "MemoryConfig",
    "Scope",
    "Authorizer",
    "AllowAllAuthorizer",
    "AuthorizationError",
    "FusionMemoryRuntime",
    "create_mcp_app",
    "create_mcp_server",
    "run_mcp_server",
]


def __getattr__(name: str):
    """Keep optional MCP imports out of the base package initialization path."""
    if name == "FusionMemoryRuntime":
        from fusion_memory.mcp_runtime import FusionMemoryRuntime

        return FusionMemoryRuntime
    if name in {"create_mcp_app", "create_mcp_server", "run_mcp_server"}:
        from fusion_memory import mcp_server

        return getattr(mcp_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
