from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Callable, Iterator, TypeVar


T = TypeVar("T")


class PoolAcquireTimeout(RuntimeError):
    """A bounded pool acquisition timed out and may be retried."""

    code = "pool_acquire_timeout"


class RetryableOperationError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class PostgresConnectionPool:
    """Bounded synchronous wrapper around psycopg2's threaded connection pool."""

    def __init__(
        self,
        dsn: str,
        *,
        min_connections: int = 1,
        max_connections: int = 8,
        pool: Any | None = None,
    ) -> None:
        if max_connections < 1 or min_connections < 0 or min_connections > max_connections:
            raise ValueError("invalid Postgres connection pool bounds")
        self.dsn = dsn
        self.max_connections = max_connections
        self._semaphore = threading.BoundedSemaphore(max_connections)
        if pool is None:
            try:
                from psycopg2.pool import ThreadedConnectionPool
            except ImportError as exc:
                raise RuntimeError("Postgres pooling requires psycopg2") from exc
            pool = ThreadedConnectionPool(min_connections, max_connections, dsn)
        self._pool = pool

    @contextmanager
    def connection(self, timeout_seconds: float) -> Iterator[Any]:
        if not self._semaphore.acquire(timeout=timeout_seconds):
            raise PoolAcquireTimeout("Postgres connection pool is exhausted")
        connection: Any | None = None
        try:
            connection = self._pool.getconn()
            yield connection
        finally:
            try:
                if connection is not None:
                    self.release(connection)
            finally:
                self._semaphore.release()

    def release(self, connection: Any) -> None:
        self._pool.putconn(connection, close=self._connection_is_broken(connection))

    def close(self) -> None:
        self._pool.closeall()

    @staticmethod
    def _connection_is_broken(connection: Any) -> bool:
        if bool(getattr(connection, "closed", False)):
            return True
        get_status = getattr(connection, "get_transaction_status", None)
        # psycopg2.extensions.TRANSACTION_STATUS_INERROR is 3. Avoid importing
        # psycopg2 here so injected fake connections remain supported.
        return bool(get_status and get_status() == 3)


class PostgresOperationExecutor:
    """Run complete Postgres operations in a bounded synchronous worker pool.

    Task 5 should create a fresh ``MemoryService(store=bound_store, ...)`` in
    ``callback``. The callback then runs wholly inside one pooled transaction.
    """

    def __init__(self, store: Any, *, max_workers: int = 8, acquire_timeout_seconds: float = 5.0) -> None:
        if max_workers < 1 or acquire_timeout_seconds <= 0:
            raise ValueError("invalid Postgres operation executor bounds")
        self._store = store
        self._acquire_timeout_seconds = acquire_timeout_seconds
        self._slots = threading.BoundedSemaphore(max_workers)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fusion-memory-pg")

    def run(
        self,
        callback: Callable[[Any], T],
        *,
        user_id: str | None = None,
        write: bool = False,
    ) -> T:
        """Execute ``callback(bound_store)`` after bounded worker acquisition."""
        if not self._slots.acquire(timeout=self._acquire_timeout_seconds):
            raise PoolAcquireTimeout("Postgres operation worker pool is exhausted")
        try:
            future = self._executor.submit(self._run_operation, callback, user_id, write)
            return future.result()
        finally:
            self._slots.release()

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _run_operation(self, callback: Callable[[Any], T], user_id: str | None, write: bool) -> T:
        with self._store.operation(user_id=user_id, write=write) as bound_store:
            return callback(bound_store)
