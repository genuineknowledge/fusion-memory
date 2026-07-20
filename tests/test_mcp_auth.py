from datetime import datetime, timezone

import pytest

from fusion_memory.mcp_auth import FusionMemoryTokenVerifier, token_digest
from fusion_memory.storage.token_store import TokenRecord


class FakeTokenStore:
    def __init__(self, record: TokenRecord) -> None:
        self.record = record

    def verify_digest(self, digest: str) -> TokenRecord | None:
        return self.record if self.record.token_hash == digest else None


@pytest.mark.anyio
async def test_token_verifier_maps_token_to_user_and_scopes():
    record = TokenRecord(
        token_id="token-1",
        token_hash=token_digest("secret", "pepper"),
        user_id="user-a",
        scopes=("memory:read", "memory:write", "memory:sync"),
        expires_at=None,
        revoked_at=None,
        created_at=datetime.now(timezone.utc),
        last_used_at=None,
    )
    verifier = FusionMemoryTokenVerifier(FakeTokenStore(record), pepper="pepper")

    access = await verifier.verify_token("secret")

    assert access is not None
    assert access.subject == "user-a"
    assert "memory:read" in access.scopes


@pytest.mark.anyio
async def test_revoked_token_is_rejected():
    record = TokenRecord(
        token_id="token-1",
        token_hash=token_digest("secret", "pepper"),
        user_id="user-a",
        scopes=("memory:read",),
        expires_at=None,
        revoked_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        last_used_at=None,
    )
    verifier = FusionMemoryTokenVerifier(FakeTokenStore(record), pepper="pepper")

    assert await verifier.verify_token("secret") is None
