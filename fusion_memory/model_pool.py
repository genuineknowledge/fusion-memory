from __future__ import annotations

import re
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, TypeVar
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit

from fusion_memory.core.embedding import HTTPEmbeddingClient
from fusion_memory.retrieval.reranker import HTTPReranker


class EndpointUnavailable(RuntimeError):
    """No healthy endpoint can accept a model request."""


@dataclass
class _EndpointState:
    endpoint: str
    semaphore: threading.BoundedSemaphore
    healthy: bool = True
    in_flight: int = 0
    failures: int = 0
    ejected_at: float | None = None
    next_probe_at: float | None = None
    probing: bool = False
    latency_ms: float | None = None
    last_error: str | None = None


class EndpointPool:
    """Thread-safe, bounded endpoint selection with health-based recovery."""

    def __init__(
        self,
        endpoints: list[str],
        *,
        max_in_flight: int = 1,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        health_probe: Callable[[str], bool] | None = None,
    ) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        for endpoint in endpoints:
            cleaned = endpoint.strip() if endpoint else ""
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        if not normalized:
            raise ValueError("endpoint pool must contain at least one endpoint")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if recovery_seconds < 0:
            raise ValueError("recovery_seconds must not be negative")
        self.endpoints = normalized
        self.max_in_flight = max_in_flight
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._health_probe = health_probe or _health_probe
        self._states = {
            endpoint: _EndpointState(endpoint=endpoint, semaphore=threading.BoundedSemaphore(max_in_flight))
            for endpoint in normalized
        }
        self._lock = threading.RLock()
        self._availability = threading.Condition(self._lock)
        self._cursor = 0

    def choose(self) -> str:
        """Return the next healthy endpoint in round-robin order."""
        self._recover_due_endpoints()
        with self._lock:
            return self._choose_locked()

    @contextmanager
    def lease(self, timeout_seconds: float) -> Iterator[str]:
        """Reserve one endpoint slot, always returning it when the operation ends."""
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        deadline = time.monotonic() + timeout_seconds
        while True:
            candidates = self._healthy_candidates()
            if not candidates:
                # There may be an ejected endpoint whose recovery window is due.
                self._recover_due_endpoints()
                if time.monotonic() >= deadline:
                    raise TimeoutError("model endpoint pool is exhausted")
                candidates = self._healthy_candidates()
                if not candidates:
                    raise EndpointUnavailable("no healthy model endpoints are available")
            # Try every healthy endpoint without waiting before blocking for a release.
            # A busy round-robin choice must not hide an idle peer.
            for endpoint in candidates:
                if self._try_acquire(endpoint):
                    try:
                        yield endpoint
                        return
                    finally:
                        self._release(endpoint)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model endpoint pool is exhausted")
            # Health probes execute outside the lock, but still consume this lease's
            # one shared deadline before we wait for a capacity notification.
            self._recover_due_endpoints()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("model endpoint pool is exhausted")
            with self._availability:
                self._availability.wait(timeout=remaining)

    def mark_failure(self, endpoint: str, error_message: str | None = None, *, latency_ms: float | None = None) -> None:
        with self._availability:
            state = self._state(endpoint)
            state.failures += 1
            state.latency_ms = latency_ms
            state.last_error = _sanitize_error(error_message) if error_message else None
            if state.failures >= self.failure_threshold:
                state.healthy = False
                state.ejected_at = time.monotonic()
                state.next_probe_at = state.ejected_at + self.recovery_seconds
            self._availability.notify_all()

    def mark_success(self, endpoint: str, *, latency_ms: float | None = None) -> None:
        with self._availability:
            state = self._state(endpoint)
            state.healthy = True
            state.failures = 0
            state.ejected_at = None
            state.next_probe_at = None
            state.probing = False
            state.latency_ms = latency_ms
            self._availability.notify_all()

    def healthy_endpoints(self) -> list[str]:
        self._recover_due_endpoints()
        with self._lock:
            return [endpoint for endpoint in self.endpoints if self._states[endpoint].healthy]

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "endpoint": _safe_endpoint(state.endpoint),
                    "healthy": state.healthy,
                    "in_flight": state.in_flight,
                    "failure_count": state.failures,
                    "latency_ms": state.latency_ms,
                    "last_error": state.last_error,
                }
                for state in (self._states[endpoint] for endpoint in self.endpoints)
            ]

    def _choose_locked(self) -> str:
        for offset in range(len(self.endpoints)):
            index = (self._cursor + offset) % len(self.endpoints)
            endpoint = self.endpoints[index]
            if self._states[endpoint].healthy:
                self._cursor = (index + 1) % len(self.endpoints)
                return endpoint
        raise EndpointUnavailable("no healthy model endpoints are available")

    def _healthy_candidates(self) -> list[str]:
        with self._lock:
            return [
                self.endpoints[(self._cursor + offset) % len(self.endpoints)]
                for offset in range(len(self.endpoints))
                if self._states[self.endpoints[(self._cursor + offset) % len(self.endpoints)]].healthy
            ]

    def _try_acquire(self, endpoint: str) -> bool:
        state = self._states[endpoint]
        if not state.semaphore.acquire(blocking=False):
            return False
        with self._availability:
            if not state.healthy:
                state.semaphore.release()
                return False
            state.in_flight += 1
            self._cursor = (self.endpoints.index(endpoint) + 1) % len(self.endpoints)
            return True

    def _release(self, endpoint: str) -> None:
        state = self._states[endpoint]
        with self._availability:
            state.in_flight -= 1
            state.semaphore.release()
            self._availability.notify_all()

    def _recover_due_endpoints(self) -> None:
        due: list[str] = []
        with self._availability:
            now = time.monotonic()
            for state in self._states.values():
                if (
                    not state.healthy
                    and not state.probing
                    and state.next_probe_at is not None
                    and now >= state.next_probe_at
                ):
                    state.probing = True
                    # Failed probes are not retried on every selection, even when
                    # recovery_seconds is configured as zero for tests or fast recovery.
                    state.next_probe_at = now + max(1.0, self.recovery_seconds)
                    due.append(state.endpoint)
        for endpoint in due:
            probe_error: str | None = None
            try:
                recovered = self._health_probe(endpoint)
            except Exception as exc:  # A failed probe must leave the circuit open.
                recovered = False
                probe_error = str(exc)
            with self._availability:
                state = self._states[endpoint]
                # A concurrent explicit success invalidates this stale probe result.
                if not state.probing:
                    continue
                state.probing = False
                if recovered:
                    state.healthy = True
                    state.failures = 0
                    state.ejected_at = None
                    state.next_probe_at = None
                else:
                    if probe_error:
                        state.last_error = _sanitize_error(probe_error)
                    state.next_probe_at = time.monotonic() + max(1.0, self.recovery_seconds)
                self._availability.notify_all()

    def _state(self, endpoint: str) -> _EndpointState:
        try:
            return self._states[endpoint]
        except KeyError as exc:
            raise ValueError(f"unknown model endpoint: {_safe_endpoint(endpoint)}") from exc


class _EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class _RerankerClient(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]:
        ...


Client = TypeVar("Client")


class _PooledClient:
    def __init__(self, pool: EndpointPool) -> None:
        self.pool = pool

    def _call(self, operation: Callable[[str], Client], invoke: Callable[[Client], object]) -> object:
        last_error: BaseException | None = None
        attempted: set[str] = set()
        while len(attempted) < len(self.pool.endpoints):
            try:
                with self.pool.lease(timeout_seconds=self.timeout_seconds) as endpoint:
                    if endpoint in attempted:
                        break
                    attempted.add(endpoint)
                    started = time.perf_counter()
                    try:
                        result = invoke(operation(endpoint))
                    except Exception as exc:
                        latency_ms = (time.perf_counter() - started) * 1000
                        if not _is_transport_failure(exc):
                            raise
                        self.pool.mark_failure(endpoint, str(exc), latency_ms=latency_ms)
                        last_error = exc
                        continue
                    self.pool.mark_success(endpoint, latency_ms=(time.perf_counter() - started) * 1000)
                    return result
            except EndpointUnavailable as exc:
                last_error = exc
                break
            except TimeoutError as exc:
                last_error = exc
                break
        if last_error is not None:
            raise last_error
        raise EndpointUnavailable("no healthy model endpoints are available")


class PooledEmbedder(_PooledClient):
    """Embedder protocol adapter that routes synchronous HTTP calls through a pool."""

    def __init__(
        self,
        endpoints: list[str],
        *,
        api_key: str | None = None,
        model: str = "local-embedding",
        timeout_seconds: float = 30.0,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        max_in_flight: int = 1,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        client_factory: Callable[[str], _EmbeddingClient] | None = None,
    ) -> None:
        super().__init__(
            EndpointPool(
                endpoints,
                max_in_flight=max_in_flight,
                failure_threshold=failure_threshold,
                recovery_seconds=recovery_seconds,
            )
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.calls: list[dict[str, object]] = []
        self.version = f"pooled-http-embedding:{model}"
        self._clients: dict[str, _EmbeddingClient] = {}
        self._client_factory = client_factory or (
            lambda endpoint: HTTPEmbeddingClient(
                endpoint,
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
                dimensions=dimensions,
                encoding_format=encoding_format,
            )
        )

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = self._call(self._client, lambda client: client.embed_texts(texts))
        if not isinstance(result, list):
            raise ValueError("embedding endpoint returned invalid embeddings")
        self.calls.append({"model": self.model, "text_count": len(texts)})
        return result

    def _client(self, endpoint: str) -> _EmbeddingClient:
        if endpoint not in self._clients:
            self._clients[endpoint] = self._client_factory(endpoint)
        return self._clients[endpoint]


class PooledReranker(_PooledClient):
    """Reranker protocol adapter that routes synchronous HTTP calls through a pool."""

    def __init__(
        self,
        endpoints: list[str],
        *,
        api_key: str | None = None,
        model: str = "local-reranker",
        timeout_seconds: float = 30.0,
        top_n: int | None = None,
        instruct: str | None = None,
        max_in_flight: int = 1,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        client_factory: Callable[[str], _RerankerClient] | None = None,
    ) -> None:
        super().__init__(
            EndpointPool(
                endpoints,
                max_in_flight=max_in_flight,
                failure_threshold=failure_threshold,
                recovery_seconds=recovery_seconds,
            )
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.calls: list[dict[str, object]] = []
        self.version = f"pooled-http-reranker:{model}"
        self._clients: dict[str, _RerankerClient] = {}
        self._client_factory = client_factory or (
            lambda endpoint: HTTPReranker(
                endpoint,
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
                top_n=top_n,
                instruct=instruct,
            )
        )

    def score(self, query: str, docs: list[str]) -> list[float]:
        result = self._call(self._client, lambda client: client.score(query, docs))
        if not isinstance(result, list):
            raise ValueError("reranker endpoint returned invalid scores")
        self.calls.append({"model": self.model, "doc_count": len(docs)})
        return [float(score) for score in result]

    def _client(self, endpoint: str) -> _RerankerClient:
        if endpoint not in self._clients:
            self._clients[endpoint] = self._client_factory(endpoint)
        return self._clients[endpoint]


def _health_probe(endpoint: str) -> bool:
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return False
    health_url = urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
    with request.urlopen(health_url, timeout=2.0) as response:
        return 200 <= response.status < 300


def _is_transport_failure(exc: BaseException) -> bool:
    if isinstance(exc, error.HTTPError):
        return False
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError, error.URLError)):
        return True
    return isinstance(exc, OSError)


def _safe_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|client[_-]?secret|password|passwd|credential|credentials)\s*([=:])\s*[^\s,;]+"
)
_AUTH_SCHEME_RE = re.compile(r"(?i)\b(authorization\s*:\s*)?(basic|bearer)\s+[^\s,;]+")


def _sanitize_error(value: str) -> str:
    value = _CREDENTIAL_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", value)
    return _AUTH_SCHEME_RE.sub(lambda match: f"{match.group(1) or ''}{match.group(2)} [redacted]", value)
