from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Iterator


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
