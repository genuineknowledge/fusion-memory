from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Protocol

import anyio
from mcp.server.auth.provider import AccessToken, TokenVerifier

from fusion_memory.storage.token_store import TokenRecord


class TokenStore(Protocol):
    def verify_digest(self, digest: str) -> TokenRecord | None: ...


def token_digest(token: str, pepper: str) -> str:
    if not pepper:
        raise ValueError("FUSION_MEMORY_TOKEN_PEPPER is required")
    return hmac.new(pepper.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()


class FusionMemoryTokenVerifier(TokenVerifier):
    def __init__(self, store: TokenStore, *, pepper: str) -> None:
        if not pepper:
            raise ValueError("FUSION_MEMORY_TOKEN_PEPPER is required")
        self._store = store
        self._pepper = pepper

    async def verify_token(self, token: str) -> AccessToken | None:
        digest = token_digest(token, self._pepper)
        record = await anyio.to_thread.run_sync(self._store.verify_digest, digest)
        if (
            record is None
            or record.revoked_at is not None
            or (record.expires_at is not None and record.expires_at <= datetime.now(timezone.utc))
            or not hmac.compare_digest(record.token_hash, digest)
        ):
            return None
        return AccessToken(
            token=token,
            subject=record.user_id,
            client_id=record.token_id,
            scopes=list(record.scopes),
        )
