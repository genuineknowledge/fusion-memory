from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fusion_memory.adapters.haitun_history_watcher import (
    WatcherConfig,
    config_from_workspace,
    load_checkpoint,
    sync_history_once_async,
)
from fusion_memory.storage.batch_ledger import BatchClaim, BatchIngestor, BatchLedger
from fusion_memory.mcp_client import MemoryMcpClient, _tool_result
from fusion_memory.core.models import Scope
from fusion_memory.mcp_runtime import FusionMemoryRuntime


def test_replaying_batch_returns_original_result_without_second_add():
    ledger = InMemoryBatchLedger()
    writer = RecordingBatchWriter()
    ingestor = BatchIngestor(ledger=ledger, write_messages=writer)
    messages = [{"role": "user", "content": "x"}]

    first = ingestor.ingest(user_id="user-a", batch_id="batch-1", messages=messages, metadata=None)
    second = ingestor.ingest(user_id="user-a", batch_id="batch-1", messages=messages, metadata=None)

    assert second == first
    assert writer.calls == 1


def test_failed_batch_is_not_marked_complete():
    ledger = InMemoryBatchLedger()
    ingestor = BatchIngestor(ledger=ledger, write_messages=FailingBatchWriter())

    with pytest.raises(RuntimeError):
        ingestor.ingest(
            user_id="user-a",
            batch_id="batch-2",
            messages=[{"role": "user", "content": "x"}],
            metadata=None,
        )

    assert ledger.is_complete("user-a", "batch-2") is False


def test_batch_id_conflict_is_non_retryable():
    ledger = InMemoryBatchLedger()
    ingestor = BatchIngestor(ledger=ledger, write_messages=RecordingBatchWriter())
    ingestor.ingest(user_id="user-a", batch_id="batch-3", messages=[{"content": "x"}], metadata=None)

    with pytest.raises(ValueError, match="batch_id_conflict"):
        ingestor.ingest(user_id="user-a", batch_id="batch-3", messages=[{"content": "different"}], metadata=None)


@pytest.mark.anyio
async def test_checkpoint_advances_only_after_confirmed_mcp_success(tmp_path: Path):
    config = watcher_config(tmp_path)
    client = RecordingMcpClient(results=[{"ok": False}, {"ok": True, "result": {"batch_id": "b1"}}])

    with pytest.raises(RuntimeError):
        await sync_history_once_async(config, client=client)
    assert load_checkpoint(config.checkpoint_path).get("submitted_batches", []) == []

    result = await sync_history_once_async(config, client=client)
    assert result["submitted_count"] == 1
    assert len(load_checkpoint(config.checkpoint_path)["submitted_batches"]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("response", [{"isError": True}, {"ok": False}, {}])
async def test_checkpoint_does_not_advance_for_unconfirmed_mcp_results(tmp_path: Path, response: dict[str, Any]):
    config = watcher_config(tmp_path)

    with pytest.raises(RuntimeError):
        await sync_history_once_async(config, client=RecordingMcpClient(results=[response]))

    assert load_checkpoint(config.checkpoint_path).get("submitted_batches", []) == []


@pytest.mark.anyio
async def test_checkpoint_does_not_advance_for_transport_failure(tmp_path: Path):
    config = watcher_config(tmp_path)

    with pytest.raises(TimeoutError):
        await sync_history_once_async(config, client=FailingMcpClient())

    assert load_checkpoint(config.checkpoint_path).get("submitted_batches", []) == []


def test_unstructured_mcp_result_is_not_treated_as_success():
    result = SimpleNamespace(isError=False, structuredContent=None, content=[SimpleNamespace(text="not json")])

    assert _tool_result(result)["ok"] is False


def test_legacy_memory_base_url_gets_mcp_path(tmp_path: Path):
    config = config_from_workspace(
        workspace=tmp_path,
        session_id="s1",
        base_url="http://memory.test:8700",
        env={"FUSION_MEMORY_TOKEN": "secret"},
    )

    assert config.mcp_url == "http://memory.test:8700/mcp"
    assert "secret" not in repr(config)


@pytest.mark.anyio
async def test_runtime_batch_replay_returns_stored_add_result():
    service = RecordingRuntimeService()
    runtime = FusionMemoryRuntime(InlineExecutor(), lambda _store: service)

    first = await runtime.add_batch(Scope(user_id="user-a"), [{"content": "x"}], "batch-1", None)
    second = await runtime.add_batch(Scope(user_id="user-a"), [{"content": "x"}], "batch-1", None)

    assert second == first
    assert service.calls == 1


def test_connection_bound_ledger_never_commits_its_own_sql():
    connection = LedgerConnection()
    ledger = BatchLedger(connection)

    claim = ledger.claim_or_get("user-a", "batch-1", "hash-1")
    ledger.complete("user-a", "batch-1", {"trace_id": "trace-1"})

    assert claim == BatchClaim(completed=False, result=None)
    assert any("insert into fusion_memory_mcp_batches" in statement for statement in connection.statements)
    assert any("for update" in statement for statement in connection.statements)
    assert any("set status = 'completed'" in statement for statement in connection.statements)
    assert connection.commits == 0
    assert connection.rollbacks == 0


def test_batch_migration_defines_user_scoped_idempotency_ledger():
    migration = Path("fusion_memory/storage/migrations/postgres/003_mcp_batches.sql").read_text(encoding="utf-8")

    assert "primary key (user_id, batch_id)" in migration
    assert "request_hash" in migration
    assert "result jsonb" in migration
    assert "status text" in migration


@pytest.mark.anyio
async def test_runtime_memory_write_and_ledger_completion_share_operation_commit():
    connection = LedgerConnection()
    executor = TransactionExecutor(connection)
    runtime = FusionMemoryRuntime(executor, lambda store: TransactionRuntimeService(store, connection))

    result = await runtime.add_batch(Scope(user_id="user-a"), [{"content": "x"}], "batch-1", None)

    assert result["add_result"]["trace_id"] == "trace-1"
    assert connection.events == ["ledger_insert", "ledger_select", "memory_write", "ledger_complete", "commit"]


@pytest.mark.anyio
async def test_memory_mcp_client_reuses_session_until_transport_failure():
    client = MemoryMcpClient("http://memory.test/mcp", "secret", "ws-1", "s1")
    session = RecordingSession([{"ok": True}, {"ok": True}])
    client._session = session

    await client.call_tool("memory_add_batch", {"batch_id": "b1"})
    await client.call_tool("memory_add_batch", {"batch_id": "b2"})

    assert session.calls == 2
    assert client._session is session


@pytest.mark.anyio
async def test_memory_mcp_client_initializes_one_streamable_session(monkeypatch):
    import fusion_memory.mcp_client as client_module

    class FakeHttpClient:
        instances = 0

        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            FakeHttpClient.instances += 1

        async def __aenter__(self) -> "FakeHttpClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class FakeTransport:
        async def __aenter__(self):
            return object(), object(), lambda: "s1"

        async def __aexit__(self, *args: Any) -> None:
            pass

    class FakeSession:
        initialized = 0
        calls = 0

        def __init__(self, read_stream: Any, write_stream: Any) -> None:
            del read_stream, write_stream

        async def __aenter__(self) -> "FakeSession":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def initialize(self) -> None:
            FakeSession.initialized += 1

        async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
            del name, arguments, kwargs
            FakeSession.calls += 1
            return SimpleNamespace(isError=False, structuredContent={"ok": True}, content=[])

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeHttpClient)
    monkeypatch.setattr(client_module, "streamable_http_client", lambda *args, **kwargs: FakeTransport())
    monkeypatch.setattr(client_module, "ClientSession", FakeSession)

    client = MemoryMcpClient("http://memory.test/mcp", "secret", "ws-1", "s1")
    await client.call_tool("memory_add_batch", {"batch_id": "b1"})
    await client.call_tool("memory_add_batch", {"batch_id": "b2"})
    await client.close()

    assert FakeHttpClient.instances == 1
    assert FakeSession.initialized == 1
    assert FakeSession.calls == 2


@pytest.mark.anyio
async def test_memory_mcp_client_resets_session_after_transport_failure():
    client = MemoryMcpClient("http://memory.test/mcp", "secret", "ws-1", "s1")
    stack = RecordingExitStack()
    client._stack = stack
    client._session = FailingSession()

    with pytest.raises(ConnectionError):
        await client.call_tool("memory_add_batch", {"batch_id": "b1"})

    assert client._session is None
    assert stack.closed is True


class InMemoryBatchLedger:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    def claim_or_get(self, user_id: str, batch_id: str, request_hash: str) -> BatchClaim:
        row = self.rows.setdefault(
            (user_id, batch_id),
            {"request_hash": request_hash, "completed": False, "result": None},
        )
        if row["request_hash"] != request_hash:
            raise ValueError("batch_id_conflict")
        return BatchClaim(completed=bool(row["completed"]), result=row["result"])

    def complete(self, user_id: str, batch_id: str, result: dict[str, Any]) -> None:
        self.rows[(user_id, batch_id)].update(completed=True, result=result)

    def is_complete(self, user_id: str, batch_id: str) -> bool:
        return bool(self.rows.get((user_id, batch_id), {}).get("completed"))


class RecordingBatchWriter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: list[dict[str, Any]], metadata: dict[str, Any] | None) -> dict[str, Any]:
        self.calls += 1
        return {"message_count": len(messages), "metadata": metadata or {}}


class FailingBatchWriter:
    def __call__(self, messages: list[dict[str, Any]], metadata: dict[str, Any] | None) -> dict[str, Any]:
        del messages, metadata
        raise RuntimeError("write failed")


class RecordingMcpClient:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = list(results)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert name == "memory_add_batch"
        assert arguments["batch_id"]
        return self.results.pop(0)


class FailingMcpClient:
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del name, arguments
        raise TimeoutError("timed out")


class RecordingSession:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        del name, arguments, kwargs
        self.calls += 1
        return SimpleNamespace(isError=False, structuredContent=self.results.pop(0), content=[])


class FailingSession:
    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        del name, arguments, kwargs
        raise ConnectionError("disconnected")


class RecordingExitStack:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class InlineExecutor:
    def run(self, callback: Any, **kwargs: Any) -> Any:
        del kwargs
        return callback(object())


class RecordingRuntimeService:
    def __init__(self) -> None:
        self.calls = 0

    def add(self, input_data: Any, scope: Scope, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        del input_data, scope, metadata
        self.calls += 1
        return {"trace_id": "trace-1"}

    def close(self) -> None:
        pass


class LedgerConnection:
    def __init__(self) -> None:
        self.row: tuple[str, str, dict[str, Any] | None] | None = None
        self.statements: list[str] = []
        self.events: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> "LedgerCursor":
        return LedgerCursor(self)

    def commit(self) -> None:
        self.commits += 1
        self.events.append("commit")

    def rollback(self) -> None:
        self.rollbacks += 1


class LedgerCursor:
    rowcount = 1

    def __init__(self, connection: LedgerConnection) -> None:
        self.connection = connection

    def execute(self, statement: str, params: Any = None) -> None:
        normalized = " ".join(statement.split()).lower()
        self.connection.statements.append(normalized)
        if normalized.startswith("insert into fusion_memory_mcp_batches"):
            self.connection.row = (params[2], "pending", None)
            self.connection.events.append("ledger_insert")
        elif normalized.startswith("select request_hash"):
            self.connection.events.append("ledger_select")
        elif normalized.startswith("update fusion_memory_mcp_batches"):
            self.connection.events.append("ledger_complete")

    def fetchone(self) -> tuple[str, str, dict[str, Any] | None] | None:
        return self.connection.row

    def close(self) -> None:
        pass


class TransactionExecutor:
    def __init__(self, connection: LedgerConnection) -> None:
        self.connection = connection

    def run(self, callback: Any, **kwargs: Any) -> Any:
        del kwargs
        result = callback(SimpleNamespace(conn=self.connection))
        self.connection.commit()
        return result


class TransactionRuntimeService:
    def __init__(self, store: Any, connection: LedgerConnection) -> None:
        self.store = store
        self.connection = connection

    def add(self, input_data: Any, scope: Scope, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        del input_data, scope, metadata
        assert self.connection.commits == 0
        self.connection.events.append("memory_write")
        return {"trace_id": "trace-1"}

    def close(self) -> None:
        pass


def watcher_config(tmp_path: Path) -> WatcherConfig:
    history_path = tmp_path / "histories" / "s1.jsonl"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps({"role": "user", "content": "remember me"}) + "\n",
        encoding="utf-8",
    )
    return WatcherConfig(
        workspace=tmp_path,
        history_path=history_path,
        checkpoint_path=tmp_path / ".fusion-memory" / "s1.json",
        mcp_url="http://memory.test/mcp",
        token="test-token",
        workspace_id="ws-1",
        agent_id="haitun",
        session_id="s1",
        db_path="unused",
        timeout_seconds=1.0,
    )
