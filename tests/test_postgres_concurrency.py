from __future__ import annotations

import concurrent.futures
import threading

from fusion_memory.storage.postgres_pool import PostgresConnectionPool, PostgresOperationExecutor
from fusion_memory.storage.postgres_store import PostgresMemoryStore


class LockingCursor:
    def __init__(self, connection: "LockingConnection") -> None:
        self.connection = connection

    def execute(self, statement: str, params: object = None) -> None:
        self.connection.executed.append((statement, params))
        if "pg_advisory_xact_lock" in statement:
            user_id = params[0]
            lock = self.connection.locks.setdefault(user_id, threading.Lock())
            lock.acquire()
            self.connection.locked_user_id = user_id

    def close(self) -> None:
        pass


class LockingConnection:
    def __init__(self, locks: dict[str, threading.Lock]) -> None:
        self.locks = locks
        self.executed: list[tuple[str, object]] = []
        self.locked_user_id: str | None = None
        self.closed = 0

    def cursor(self) -> LockingCursor:
        return LockingCursor(self)

    def commit(self) -> None:
        self._release_lock()

    def rollback(self) -> None:
        self._release_lock()

    def _release_lock(self) -> None:
        if self.locked_user_id is not None:
            self.locks[self.locked_user_id].release()
            self.locked_user_id = None


class LockingDriverPool:
    def __init__(self, connections: list[LockingConnection]) -> None:
        self.connections = connections
        self.available = list(connections)
        self.lock = threading.Lock()

    def getconn(self) -> LockingConnection:
        with self.lock:
            return self.available.pop()

    def putconn(self, connection: LockingConnection, close: bool = False) -> None:
        assert not close
        with self.lock:
            self.available.append(connection)

    def closeall(self) -> None:
        pass


def make_executor() -> tuple[PostgresOperationExecutor, list[LockingConnection]]:
    locks: dict[str, threading.Lock] = {}
    connections = [LockingConnection(locks), LockingConnection(locks)]
    store = PostgresMemoryStore(
        "dsn",
        pool=PostgresConnectionPool("dsn", pool=LockingDriverPool(connections), max_connections=2),
    )
    return PostgresOperationExecutor(store, max_workers=2), connections


def test_executor_runs_different_user_operations_in_parallel() -> None:
    executor, connections = make_executor()
    active: set[str] = set()
    started = threading.Event()
    release = threading.Event()

    def callback(bound_store: PostgresMemoryStore, user_id: str) -> None:
        assert bound_store is not None
        active.add(user_id)
        if len(active) == 2:
            started.set()
        release.wait(timeout=1)
        active.remove(user_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(executor.run, lambda store: callback(store, "user-a"), user_id="user-a", write=True)
        second = callers.submit(executor.run, lambda store: callback(store, "user-b"), user_id="user-b", write=True)
        assert started.wait(timeout=1)
        assert active == {"user-a", "user-b"}
        release.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert all(any("pg_advisory_xact_lock" in sql for sql, _ in connection.executed) for connection in connections)
    executor.close()


def test_executor_serializes_same_user_writes_at_the_advisory_lock_boundary() -> None:
    executor, connections = make_executor()
    first_started = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def callback(bound_store: PostgresMemoryStore) -> None:
        nonlocal active, max_active
        assert bound_store is not None
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            first_started.set()
        release.wait(timeout=1)
        with state_lock:
            active -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(executor.run, callback, user_id="user-a", write=True)
        assert first_started.wait(timeout=1)
        second = callers.submit(executor.run, callback, user_id="user-a", write=True)
        assert not second.done()
        release.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert max_active == 1
    assert sum("pg_advisory_xact_lock" in sql for connection in connections for sql, _ in connection.executed) == 2
    executor.close()


class BlockingUserService:
    def __init__(self) -> None:
        self.active_users: set[str] = set()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.release = threading.Event()

    def run_for_user(self, user_id: str) -> None:
        self.active_users.add(user_id)
        (self.first_started if user_id == "user-a" else self.second_started).set()
        self.release.wait(timeout=1)
        self.active_users.remove(user_id)


def test_different_users_can_execute_in_parallel() -> None:
    service = BlockingUserService()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.run_for_user, "user-a")
        service.first_started.wait(timeout=1)
        second = executor.submit(service.run_for_user, "user-b")
        service.second_started.wait(timeout=1)
        assert service.active_users == {"user-a", "user-b"}
        service.release.set()
        first.result(timeout=1)
        second.result(timeout=1)
