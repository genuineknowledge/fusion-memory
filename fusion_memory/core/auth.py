from __future__ import annotations

from typing import Any, Protocol

from fusion_memory.core.models import Scope


class AuthorizationError(PermissionError):
    """Raised when a product-level authorizer denies a memory operation."""


class Authorizer(Protocol):
    def authorize(self, operation: str, scope: Scope, context: dict[str, Any] | None = None) -> None:
        """Allow an operation or raise AuthorizationError/PermissionError."""


class AllowAllAuthorizer:
    def authorize(self, operation: str, scope: Scope, context: dict[str, Any] | None = None) -> None:
        return None
