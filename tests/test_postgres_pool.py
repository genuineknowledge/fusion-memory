from __future__ import annotations

import contextlib

import pytest

from fusion_memory.core.runtime_config import postgres_pool_settings_from_env
from fusion_memory.storage.postgres_pool import PoolAcquireTimeout, PostgresConnectionPool
from fusion_memory.storage.postgres_store import PostgresMemoryStore


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection

    def execute(self, statement: str, params: object = None) -> None:
        self.connection.executed.append((statement, params))

    def close(self) -> None:
        pass


class FakeConnection:
    def __init__(self, *, closed: int = 0, transaction_status: int = 0) -> None:
        self.closed = closed
        self.transaction_status = transaction_status
        self.commits = 0
        self.rollbacks = 0
        self.executed: list[tuple[str, object]] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1
        self.transaction_status = 0

    def rollback(self) -> None:
        self.rollbacks += 1
        self.transaction_status = 0

    def get_transaction_status(self) -> int:
        return self.transaction_status


class FakeDriverPool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.returned: list[tuple[FakeConnection, bool]] = []

    def getconn(self) -> FakeConnection:
        return self.connection

    def putconn(self, connection: FakeConnection, close: bool = False) -> None:
        self.returned.append((connection, close))

    def closeall(self) -> None:
        pass


def test_operation_commits_once_and_bound_facade_suppresses_inner_commits() -> None:
    connection = FakeConnection()
    pool = FakeDriverPool(connection)
    store = PostgresMemoryStore("dsn", pool=PostgresConnectionPool("dsn", pool=pool, max_connections=1))

    with store.operation() as bound:
        bound.evidence._commit_if_owner()
        assert connection.commits == 0

    assert connection.commits == 1
    assert pool.returned == [(connection, False)]


def test_failed_operation_rolls_back_and_returns_connection() -> None:
    connection = FakeConnection()
    pool = FakeDriverPool(connection)
    store = PostgresMemoryStore("dsn", pool=PostgresConnectionPool("dsn", pool=pool, max_connections=1))

    with pytest.raises(RuntimeError, match="boom"):
        with store.operation():
            raise RuntimeError("boom")

    assert connection.rollbacks == 1
    assert pool.returned == [(connection, False)]


def test_write_operation_uses_authenticated_user_advisory_lock() -> None:
    connection = FakeConnection()
    store = PostgresMemoryStore(
        "dsn",
        pool=PostgresConnectionPool("dsn", pool=FakeDriverPool(connection), max_connections=1),
        lock_timeout_seconds=0.25,
    )

    with store.operation(user_id="authenticated-user", write=True):
        pass

    assert connection.executed == [
        ("set local lock_timeout = %s", ("250ms",)),
        ("select pg_advisory_xact_lock(hashtextextended(%s, 0))", ("authenticated-user",)),
    ]


def test_pool_timeout_is_retryable() -> None:
    pool = PostgresConnectionPool("dsn", pool=FakeDriverPool(FakeConnection()), max_connections=1)
    assert pool._semaphore.acquire(blocking=False)
    try:
        with pytest.raises(PoolAcquireTimeout):
            with pool.connection(timeout_seconds=0.01):
                pass
    finally:
        pool._semaphore.release()


def test_broken_connection_is_discarded() -> None:
    connection = FakeConnection(closed=1)
    driver_pool = FakeDriverPool(connection)
    pool = PostgresConnectionPool("dsn", pool=driver_pool, max_connections=1)

    with pool.connection(timeout_seconds=0.01):
        pass

    assert driver_pool.returned == [(connection, True)]


def test_postgres_pool_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUSION_MEMORY_PG_MIN_CONNECTIONS", "2")
    monkeypatch.setenv("FUSION_MEMORY_PG_MAX_CONNECTIONS", "6")
    monkeypatch.setenv("FUSION_MEMORY_PG_ACQUIRE_TIMEOUT_SECONDS", "1.5")

    settings = postgres_pool_settings_from_env()

    assert (settings.min_connections, settings.max_connections, settings.acquire_timeout_seconds) == (2, 6, 1.5)
