from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BatchClaim:
    completed: bool
    result: dict[str, Any] | None


class BatchIdConflictError(ValueError):
    """The same user reused a batch id for a different request body."""

    code = "batch_id_conflict"


class BatchLedger:
    """Connection-bound batch ledger; callers own the surrounding transaction."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def claim_or_get(self, user_id: str, batch_id: str, request_hash: str) -> BatchClaim:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                insert into fusion_memory_mcp_batches
                    (user_id, batch_id, request_hash, status)
                values (%s, %s, %s, 'pending')
                on conflict (user_id, batch_id) do nothing
                """,
                (user_id, batch_id, request_hash),
            )
            cursor.execute(
                """
                select request_hash, status, result
                from fusion_memory_mcp_batches
                where user_id = %s and batch_id = %s
                for update
                """,
                (user_id, batch_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("batch ledger claim was not persisted")
            existing_hash, status, result = _row_values(row)
            if existing_hash != request_hash:
                raise BatchIdConflictError("batch_id_conflict")
            return BatchClaim(completed=str(status) == "completed", result=_json_result(result))
        finally:
            cursor.close()

    def complete(self, user_id: str, batch_id: str, result: dict[str, Any]) -> None:
        cursor = self.connection.cursor()
        try:
            serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            cursor.execute(
                """
                update fusion_memory_mcp_batches
                set status = 'completed', result = %s::jsonb,
                    trace_id = %s, completed_at = now(), updated_at = now()
                where user_id = %s and batch_id = %s
                """,
                (serialized, result.get("trace_id"), user_id, batch_id),
            )
            if getattr(cursor, "rowcount", 1) == 0:
                raise RuntimeError("batch ledger completion target was not found")
        finally:
            cursor.close()


class BatchIngestor:
    """Claim, write, and complete a batch without owning transaction boundaries."""

    def __init__(self, *, ledger: Any, write_messages: Callable[[list[dict[str, Any]], dict[str, Any] | None], Any]) -> None:
        self.ledger = ledger
        self.write_messages = write_messages

    def ingest(
        self,
        *,
        user_id: str,
        batch_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request_hash = _request_hash(messages, metadata)
        claim = self.ledger.claim_or_get(user_id, batch_id, request_hash)
        if claim.completed:
            if claim.result is None:
                raise RuntimeError("completed batch has no result")
            return claim.result
        result = _as_dict(self.write_messages(messages, metadata))
        self.ledger.complete(user_id, batch_id, result)
        return result


def _request_hash(messages: list[dict[str, Any]], metadata: dict[str, Any] | None) -> str:
    body = json.dumps(
        {"messages": messages, "metadata": metadata},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        raise TypeError("batch writer must return a dictionary")
    return value


def _row_values(row: Any) -> tuple[Any, Any, Any]:
    if isinstance(row, dict):
        return row.get("request_hash"), row.get("status"), row.get("result")
    return row[0], row[1], row[2]


def _json_result(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
