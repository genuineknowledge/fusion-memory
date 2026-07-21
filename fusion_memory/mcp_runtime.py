from __future__ import annotations

import os
import threading
from dataclasses import asdict, is_dataclass
from contextlib import asynccontextmanager
from typing import Any, Callable

import anyio

from fusion_memory.api.service import MemoryService
from fusion_memory.core.config import MemoryConfig
from fusion_memory.core.models import Scope
from fusion_memory.core.runtime_config import (
    _build_async_extractor,
    _build_embedder,
    _build_extractor,
    _build_query_intent_refiner,
    _build_reranker,
    build_runtime_retrieval_flags,
    postgres_pool_settings_from_env,
)
from fusion_memory.storage.postgres_pool import PostgresConnectionPool, PostgresOperationExecutor
from fusion_memory.storage.postgres_store import PostgresMemoryStore
from fusion_memory.storage.batch_ledger import BatchIngestor, BatchLedger


class FusionMemoryRuntime:
    """Async MCP boundary around one bounded Postgres operation at a time."""

    def __init__(
        self,
        operation_executor: PostgresOperationExecutor,
        service_factory: Callable[[Any], MemoryService],
        *,
        worker_limit: int = 8,
        closers: tuple[Callable[[], None], ...] = (),
        endpoint_pools: tuple[Any, ...] = (),
        health_check_interval_seconds: float = 5.0,
        search_mode: str = "fast",
    ) -> None:
        if worker_limit < 1:
            raise ValueError("FUSION_MEMORY_MCP_WORKER_LIMIT must be positive")
        if search_mode not in {"fast", "balanced", "benchmark"}:
            raise ValueError("FUSION_MEMORY_MCP_SEARCH_MODE must be fast, balanced, or benchmark")
        self._operation_executor = operation_executor
        self._service_factory = service_factory
        self._worker_limiter = anyio.CapacityLimiter(worker_limit)
        self._closers = closers
        self.endpoint_pools = endpoint_pools
        self.health_check_interval_seconds = health_check_interval_seconds
        self.search_mode = search_mode
        self._close_lock = threading.Lock()
        self._supervisor_lock = threading.Lock()
        self._supervisor_alive = False
        self._supervisor_error: str | None = None
        self._closed = False
        self._fallback_batch_ledger = _MemoryBatchLedger()

    def mark_supervisor_alive(self) -> None:
        with self._supervisor_lock:
            self._supervisor_alive = True
            self._supervisor_error = None

    def mark_supervisor_unhealthy(self, error: BaseException | str | None = None) -> None:
        with self._supervisor_lock:
            self._supervisor_alive = False
            self._supervisor_error = type(error).__name__ if isinstance(error, BaseException) else (str(error) if error else None)

    def background_health(self) -> dict[str, object]:
        model_pools: dict[str, list[dict[str, object]]] = {}
        configured_pools_healthy = True
        for label, pool in zip(("embedding", "reranker"), self.endpoint_pools):
            if pool is None:
                continue
            snapshot = getattr(pool, "snapshot", None)
            try:
                value = snapshot() if callable(snapshot) else []
            except Exception:
                value = []
            model_pools[label] = value if isinstance(value, list) else []
            configured_pools_healthy = configured_pools_healthy and any(
                bool(endpoint.get("healthy")) for endpoint in model_pools[label] if isinstance(endpoint, dict)
            )
        with self._supervisor_lock:
            result: dict[str, object] = {
                "ok": self._supervisor_alive and configured_pools_healthy,
                "supervisor_alive": self._supervisor_alive,
                "model_pools": model_pools,
            }
            if self._supervisor_error:
                result["error"] = self._supervisor_error
            return result

    supervisor_health = background_health

    async def add(self, scope: Scope, content: str, source: str | None) -> Any:
        return await self._run(
            scope,
            write=True,
            operation=lambda service: service.add(
                {"role": "user", "content": content, "source_uri": source}, scope
            ),
        )

    async def search(self, scope: Scope, query: str, limit: int) -> Any:
        return await self._run(
            scope,
            write=False,
            operation=lambda service: service.search(
                query,
                scope,
                options={"limit": limit, "allow_cross_session": True, "mode": self.search_mode},
            ),
        )

    async def answer_context(self, scope: Scope, query: str, limit: int) -> Any:
        return await self._run(
            scope,
            write=False,
            operation=lambda service: service.answer_context(
                query, scope, budget={"limit": limit, "allow_cross_session": True}
            ),
        )

    async def add_batch(
        self,
        scope: Scope,
        messages: list[dict[str, Any]],
        batch_id: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        provenance = dict(metadata or {})
        # The server, rather than an arbitrary message payload, owns the
        # idempotency/provenance identifier recorded with this operation.
        provenance["batch_id"] = batch_id
        def operation(service: MemoryService) -> Any:
            store = getattr(service, "store", None)
            connection = getattr(store, "conn", None)
            if connection is None and store is not None:
                connect = getattr(store, "connect", None)
                if callable(connect):
                    connection = connect()
            ledger = BatchLedger(connection) if connection is not None else self._fallback_batch_ledger

            def write_batch(batch_messages: list[dict[str, Any]], batch_metadata: dict[str, Any] | None) -> dict[str, Any]:
                result = service.add({"messages": batch_messages}, scope, metadata=batch_metadata)
                if is_dataclass(result):
                    return asdict(result)
                if not isinstance(result, dict):
                    raise TypeError("memory add batch result must be a dictionary")
                return result

            ingestor = BatchIngestor(ledger=ledger, write_messages=write_batch)
            result = ingestor.ingest(
                user_id=scope.user_id or "",
                batch_id=batch_id,
                messages=messages,
                metadata=provenance,
            )
            return {"batch_id": batch_id, "message_count": len(messages), "add_result": result}

        return await self._run(scope, write=True, operation=operation)

    async def _run(self, scope: Scope, *, write: bool, operation: Callable[[MemoryService], Any]) -> Any:
        def execute() -> Any:
            def callback(bound_store: Any) -> Any:
                service = self._service_factory(bound_store)
                try:
                    return operation(service)
                finally:
                    service.close()

            return self._operation_executor.run(callback, user_id=scope.user_id, write=write)

        return await anyio.to_thread.run_sync(execute, limiter=self._worker_limiter)

    @asynccontextmanager
    async def lifespan(self):
        try:
            yield self
        finally:
            # Shutdown must return pooled resources even when server cancellation is in flight.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self.close)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        close_error: BaseException | None = None
        for closer in self._closers:
            try:
                closer()
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            raise close_error


def runtime_from_env() -> tuple[FusionMemoryRuntime, PostgresConnectionPool]:
    """Build the production MCP runtime from environment-only configuration."""
    dsn = _postgres_dsn_from_env()
    settings = postgres_pool_settings_from_env()
    pool = PostgresConnectionPool(
        dsn,
        min_connections=settings.min_connections,
        max_connections=settings.max_connections,
    )
    store: PostgresMemoryStore | None = None
    executor: PostgresOperationExecutor | None = None
    try:
        store = PostgresMemoryStore(dsn, pool=pool, acquire_timeout_seconds=settings.acquire_timeout_seconds)
        config = MemoryConfig(storage_backend="postgres")
        shared_embedder = _build_embedder()
        shared_reranker = _build_reranker()
        retrieval_flags = build_runtime_retrieval_flags()

        def make_service(bound_store: Any) -> MemoryService:
            request_embedder = _request_local_adapter(shared_embedder, _build_embedder)
            set_embedder = getattr(bound_store, "set_embedder", None)
            if request_embedder is not None and callable(set_embedder):
                set_embedder(request_embedder)
            return MemoryService(
                store=bound_store,
                storage_backend="postgres",
                # MemoryService owns mutable planners and traces; its configuration must
                # not be shared with another request either.
                config=MemoryConfig(**config.snapshot()),
                embedder=request_embedder,
                reranker=_request_local_adapter(shared_reranker, _build_reranker),
                extractor=_build_extractor(),
                async_extractor=_build_async_extractor(),
                query_intent_refiner=_build_query_intent_refiner(),
                query_intent_refiner_mode=os.getenv("FUSION_MEMORY_QUERY_INTENT_MODE", "off"),
                retrieval_flags=retrieval_flags,
            )

        worker_limit = _positive_int_env("FUSION_MEMORY_MCP_WORKER_LIMIT", settings.max_connections)
        executor = PostgresOperationExecutor(
            store,
            max_workers=worker_limit,
            acquire_timeout_seconds=settings.acquire_timeout_seconds,
        )
        # Preserve the semantic embedding/reranker positions even when one
        # adapter has no endpoint pool. The MCP supervisor filters None itself.
        endpoint_pools = (
            getattr(shared_embedder, "pool", None),
            getattr(shared_reranker, "pool", None),
        )
        return FusionMemoryRuntime(
            executor,
            make_service,
            worker_limit=worker_limit,
            closers=(executor.close, store.close),
            endpoint_pools=endpoint_pools,
            health_check_interval_seconds=_positive_float_env("FUSION_MEMORY_MCP_HEALTH_INTERVAL_SECONDS", 5.0),
            search_mode=os.getenv("FUSION_MEMORY_MCP_SEARCH_MODE", "fast").strip().lower(),
        ), pool
    except BaseException:
        if executor is not None:
            try:
                executor.close()
            except BaseException:
                pass
        if store is not None:
            try:
                store.close()
            except BaseException:
                pass
        else:
            try:
                pool.close()
            except BaseException:
                pass
        raise


def _postgres_dsn_from_env() -> str:
    """Use the documented MCP DSN name, retaining the historical alias."""
    return _required_env("FUSION_MEMORY_PG_DSN") if os.getenv("FUSION_MEMORY_PG_DSN", "").strip() else _required_env(
        "FUSION_MEMORY_POSTGRES_DSN"
    )


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _request_local_adapter(shared: Any, factory: Callable[[], Any]) -> Any:
    request_local = getattr(shared, "request_local", None)
    if callable(request_local):
        return request_local()
    return factory()


class _MemoryBatchLedger:
    """Fallback ledger for database-free unit fakes; Postgres uses BatchLedger."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], tuple[str, dict[str, Any] | None, bool]] = {}
        self._lock = threading.Lock()

    def claim_or_get(self, user_id: str, batch_id: str, request_hash: str):
        from fusion_memory.storage.batch_ledger import BatchClaim, BatchIdConflictError

        with self._lock:
            row = self._rows.get((user_id, batch_id))
            if row is None:
                self._rows[(user_id, batch_id)] = (request_hash, None, False)
                return BatchClaim(False, None)
            if row[0] != request_hash:
                raise BatchIdConflictError("batch_id_conflict")
            return BatchClaim(row[2], row[1])

    def complete(self, user_id: str, batch_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            request_hash, _, _ = self._rows[(user_id, batch_id)]
            self._rows[(user_id, batch_id)] = (request_hash, result, True)
