import json
from datetime import datetime, timezone

import pytest

from fusion_memory.cli import token_create, token_list, token_revoke, validate_token_scopes
from fusion_memory.storage.token_store import TokenRecord


class FakeCliStore:
    def __init__(self, record: TokenRecord) -> None:
        self.record = record
        self.created: tuple[str, tuple[str, ...], datetime | None] | None = None
        self.revoked: list[str] = []

    def create_token(
        self, user_id: str, scopes: tuple[str, ...], expires_at: datetime | None
    ) -> tuple[str, TokenRecord]:
        self.created = (user_id, scopes, expires_at)
        return "secret", self.record

    def list_tokens(self, user_id: str) -> list[TokenRecord]:
        return [self.record] if self.record.user_id == user_id else []

    def revoke_token(self, token_id: str) -> bool:
        self.revoked.append(token_id)
        return token_id == self.record.token_id


@pytest.fixture
def store() -> FakeCliStore:
    return FakeCliStore(
        TokenRecord(
            token_id="token-1",
            token_hash="hash-only",
            user_id="user-a",
            scopes=("memory:read",),
            expires_at=None,
            revoked_at=None,
            created_at=datetime.now(timezone.utc),
            last_used_at=None,
        )
    )


def test_token_list_never_exposes_plaintext(capsys, store: FakeCliStore):
    result = token_list(store, "user-a")
    print(json.dumps(result))

    assert "secret" not in capsys.readouterr().out
    assert result[0]["token_id"] == "token-1"
    assert "token_hash" not in result[0]


def test_token_handlers_create_and_revoke(store: FakeCliStore):
    token = token_create(store, "user-a", ("memory:read",), None)

    assert token == "secret"
    assert store.created == ("user-a", ("memory:read",), None)
    assert token_revoke(store, "token-1") is True
    assert store.revoked == ["token-1"]


def test_sync_scope_requires_write_scope():
    with pytest.raises(ValueError, match="memory:write"):
        validate_token_scopes(("memory:read", "memory:sync"))
