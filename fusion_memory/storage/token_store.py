from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TokenRecord:
    token_id: str
    token_hash: str
    user_id: str
    scopes: tuple[str, ...]
    expires_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime
    last_used_at: datetime | None


class PostgresTokenStore:
    """Persist bearer-token digests without retaining plaintext tokens."""

    def __init__(self, connection_factory: Callable[[], Any], *, pepper: str) -> None:
        if not pepper:
            raise ValueError("FUSION_MEMORY_TOKEN_PEPPER is required")
        self._connection_factory = connection_factory
        self._pepper = pepper

    def create_token(
        self, user_id: str, scopes: Sequence[str], expires_at: datetime | None
    ) -> tuple[str, TokenRecord]:
        from fusion_memory.mcp_auth import token_digest

        token = secrets.token_urlsafe(32)
        token_id = secrets.token_urlsafe(16)
        digest = token_digest(token, self._pepper)
        conn = self._connection_factory()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                insert into memory_api_tokens (token_id, token_hash, user_id, scopes, expires_at)
                values (%s, %s, %s, %s::jsonb, %s)
                returning token_id, token_hash, user_id, scopes, expires_at, revoked_at, created_at, last_used_at
                """,
                (token_id, digest, user_id, json.dumps(list(scopes)), expires_at),
            )
            record = _record_from_row(cursor.fetchone())
            conn.commit()
            return token, record
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                conn.close()

    def verify_digest(self, digest: str) -> TokenRecord | None:
        conn = self._connection_factory()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                select token_id, token_hash, user_id, scopes, expires_at, revoked_at, created_at, last_used_at
                from memory_api_tokens
                where token_hash = %s and revoked_at is null and (expires_at is null or expires_at > now())
                limit 1
                """,
                (digest,),
            )
            row = cursor.fetchone()
            if row is None:
                conn.commit()
                return None
            cursor.execute(
                "update memory_api_tokens set last_used_at = now() where token_id = %s",
                (_row_value(row, "token_id", 0),),
            )
            conn.commit()
            return _record_from_row(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                conn.close()

    def list_tokens(self, user_id: str) -> list[TokenRecord]:
        conn = self._connection_factory()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                select token_id, token_hash, user_id, scopes, expires_at, revoked_at, created_at, last_used_at
                from memory_api_tokens where user_id = %s order by created_at desc
                """,
                (user_id,),
            )
            records = [_record_from_row(row) for row in cursor.fetchall()]
            conn.commit()
            return records
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                conn.close()

    def revoke_token(self, token_id: str) -> bool:
        conn = self._connection_factory()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "update memory_api_tokens set revoked_at = now() where token_id = %s and revoked_at is null",
                (token_id,),
            )
            revoked = cursor.rowcount > 0
            conn.commit()
            return revoked
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                conn.close()


def _record_from_row(row: Any) -> TokenRecord:
    scopes = _row_value(row, "scopes", 3)
    if isinstance(scopes, str):
        scopes = json.loads(scopes)
    return TokenRecord(
        token_id=_row_value(row, "token_id", 0),
        token_hash=_row_value(row, "token_hash", 1),
        user_id=_row_value(row, "user_id", 2),
        scopes=tuple(scopes),
        expires_at=_row_value(row, "expires_at", 4),
        revoked_at=_row_value(row, "revoked_at", 5),
        created_at=_row_value(row, "created_at", 6),
        last_used_at=_row_value(row, "last_used_at", 7),
    )


def _row_value(row: Any, name: str, index: int) -> Any:
    return row[name] if isinstance(row, dict) else row[index]
