from __future__ import annotations

import threading
import time
from urllib import error

import pytest

from fusion_memory.model_pool import EndpointPool, PooledEmbedder, PooledReranker


def test_pool_round_robins_healthy_endpoints() -> None:
    pool = EndpointPool(["a", "b"])

    assert [pool.choose() for _ in range(4)] == ["a", "b", "a", "b"]


def test_failed_endpoint_is_ejected_then_recovers_after_health_check() -> None:
    pool = EndpointPool(["a", "b"], failure_threshold=1, recovery_seconds=0)

    pool.mark_failure("a", "connection timed out")
    assert pool.choose() == "b"
    pool.mark_success("a")

    assert set(pool.healthy_endpoints()) == {"a", "b"}
    assert pool.snapshot()[0]["last_error"] == "connection timed out"


def test_lease_enforces_per_endpoint_bound_and_releases_after_context() -> None:
    pool = EndpointPool(["a"], max_in_flight=1)

    with pool.lease(timeout_seconds=0.01) as endpoint:
        assert endpoint == "a"
        with pytest.raises(TimeoutError, match="exhausted"):
            with pool.lease(timeout_seconds=0.0):
                pass

    with pool.lease(timeout_seconds=0.01) as endpoint:
        assert endpoint == "a"


def test_pooled_embedder_retries_transport_failure_on_next_endpoint() -> None:
    class Client:
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            if self.endpoint == "a":
                raise TimeoutError("timed out")
            return [[1.0] for _ in texts]

    embedder = PooledEmbedder(["a", "b"], client_factory=Client, failure_threshold=1)

    assert embedder.embed_texts(["alpha"]) == [[1.0]]
    assert embedder.pool.healthy_endpoints() == ["b"]


def test_pooled_reranker_preserves_score_contract() -> None:
    class Client:
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint

        def score(self, query: str, docs: list[str]) -> list[float]:
            return [float(len(query) + len(doc)) for doc in docs]

    reranker = PooledReranker(["a"], client_factory=Client)

    assert reranker.score("q", ["one", "two"]) == [4.0, 4.0]


def test_pool_deduplicates_endpoints_before_creating_state() -> None:
    pool = EndpointPool(["a", "a", "b", "b"])

    assert pool.endpoints == ["a", "b"]
    assert [pool.choose() for _ in range(4)] == ["a", "b", "a", "b"]


def test_lease_fails_over_immediately_when_round_robin_endpoint_is_busy() -> None:
    pool = EndpointPool(["a", "b"], max_in_flight=1)
    pool._states["a"].semaphore.acquire()
    try:
        started = time.monotonic()
        with pool.lease(timeout_seconds=0.2) as endpoint:
            elapsed = time.monotonic() - started
            assert endpoint == "b"
    finally:
        pool._states["a"].semaphore.release()

    assert elapsed < 0.05


def test_lease_deadline_includes_time_spent_in_a_due_health_probe() -> None:
    def slow_successful_probe(_endpoint: str) -> bool:
        time.sleep(0.05)
        return True

    pool = EndpointPool(["a"], failure_threshold=1, recovery_seconds=0, health_probe=slow_successful_probe)
    pool.mark_failure("a", "offline")

    with pytest.raises(TimeoutError, match="exhausted"):
        with pool.lease(timeout_seconds=0.01):
            pass


def test_lease_rescans_for_a_peer_recovered_while_healthy_endpoint_is_saturated() -> None:
    pool = EndpointPool(["a", "b"], failure_threshold=1, recovery_seconds=0, health_probe=lambda _endpoint: True)
    acquired = threading.Event()
    result: list[str] = []

    with pool.lease(timeout_seconds=1.0) as endpoint:
        assert endpoint == "a"
        pool.mark_failure("b", "offline")

        def acquire_recovered_peer() -> None:
            with pool.lease(timeout_seconds=0.2) as leased_endpoint:
                result.append(leased_endpoint)
                acquired.set()

        worker = threading.Thread(target=acquire_recovered_peer)
        worker.start()
        assert acquired.wait(timeout=0.05)
        assert result == ["b"]
        worker.join(timeout=1.0)


def test_slow_health_probe_does_not_block_healthy_endpoint_selection() -> None:
    probe_started = threading.Event()
    release_probe = threading.Event()

    def health_probe(_endpoint: str) -> bool:
        probe_started.set()
        release_probe.wait(timeout=1.0)
        return False

    pool = EndpointPool(["a", "b"], failure_threshold=1, recovery_seconds=0, health_probe=health_probe)
    pool.mark_failure("a", "offline")
    background = threading.Thread(target=pool.choose)
    background.start()
    assert probe_started.wait(timeout=0.2)
    try:
        started = time.monotonic()
        assert pool.choose() == "b"
        assert time.monotonic() - started < 0.05
    finally:
        release_probe.set()
        background.join(timeout=1.0)


def test_failed_health_probe_is_backed_off_before_the_next_selection() -> None:
    probes: list[str] = []
    pool = EndpointPool(
        ["a", "b"],
        failure_threshold=1,
        recovery_seconds=0,
        health_probe=lambda endpoint: probes.append(endpoint) and False,
    )
    pool.mark_failure("a", "offline")

    assert [pool.choose() for _ in range(3)] == ["b", "b", "b"]
    assert probes == ["a"]


def test_http_error_is_not_retried_or_ejected_as_a_transport_failure() -> None:
    calls: list[str] = []

    class Client:
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            calls.append(self.endpoint)
            if self.endpoint == "a":
                raise error.HTTPError("https://a", 500, "server error", {}, None)
            return [[1.0] for _ in texts]

    embedder = PooledEmbedder(["a", "b"], client_factory=Client, failure_threshold=1)

    with pytest.raises(error.HTTPError):
        embedder.embed_texts(["alpha"])
    assert calls == ["a"]
    assert embedder.pool.healthy_endpoints() == ["a", "b"]


def test_snapshot_redacts_common_credential_forms_from_last_error() -> None:
    pool = EndpointPool(["https://user:pass@example.test/v1/embeddings?token=url-token"])
    pool.mark_failure(
        "https://user:pass@example.test/v1/embeddings?token=url-token",
        "request to https://url-user:url-pass@models.example.test/v1/embeddings"
        "?X-Amz-Signature=signed-query#private-fragment failed; "
        "fallback http://backup-user:backup-pass@backup.example.test/health"
        "?arbitrary=opaque#hidden; secret=super-secret password: letmein "
        "Authorization: Basic dXNlcjpwYXNz Bearer bearer-token api_key=api-secret",
    )

    snapshot = pool.snapshot()[0]
    assert snapshot["endpoint"] == "https://example.test/v1/embeddings"
    last_error = str(snapshot["last_error"])
    assert "https://models.example.test/v1/embeddings" in last_error
    assert "http://backup.example.test/health" in last_error
    assert all(
        value not in last_error
        for value in (
            "url-user",
            "url-pass",
            "signed-query",
            "private-fragment",
            "backup-user",
            "backup-pass",
            "opaque",
            "hidden",
            "super-secret",
            "letmein",
            "dXNlcjpwYXNz",
            "bearer-token",
            "api-secret",
        )
    )


@pytest.mark.parametrize(
    ("endpoint", "safe_endpoint", "secrets"),
    [
        pytest.param(
            "https://ipv6-user:ipv6-pass@[2001:db8::1]:8443/v1/embed"
            "?X-Amz-Signature=ipv6-signature#ipv6-fragment",
            "https://[2001:db8::1]:8443/v1/embed",
            ("ipv6-user", "ipv6-pass", "ipv6-signature", "ipv6-fragment"),
            id="bracketed-ipv6",
        ),
        pytest.param(
            "https://delim-user:delim-pass@example.test/v1/embed"
            "?opaque=comma-secret,semicolon-secret;tail#delim-fragment",
            "https://example.test/v1/embed",
            ("delim-user", "delim-pass", "comma-secret", "semicolon-secret", "delim-fragment"),
            id="query-sub-delimiters",
        ),
        pytest.param(
            "https://paren-user:pa(ss)word-secret@example.test/v1/embed"
            "?token=paren-query#paren-fragment",
            "https://example.test/v1/embed",
            ("paren-user", "pa(ss)word-secret", "paren-query", "paren-fragment"),
            id="parenthesized-userinfo",
        ),
        pytest.param(
            "https://port-user:port-pass@example.test:notaport/v1/embed"
            "?token=port-query#port-fragment",
            "[redacted-url]",
            ("port-user", "port-pass", "notaport", "port-query", "port-fragment"),
            id="invalid-port",
        ),
        pytest.param(
            "https://broken-user:broken-pass@[2001:db8::1/v1/embed"
            "?token=broken-query#broken-fragment",
            "[redacted-url]",
            ("broken-user", "broken-pass", "broken-query", "broken-fragment"),
            id="malformed-brackets",
        ),
    ],
)
def test_failure_snapshot_sanitizes_odd_urls_without_masking_transport_failure(
    endpoint: str,
    safe_endpoint: str,
    secrets: tuple[str, ...],
) -> None:
    pool = EndpointPool([endpoint], failure_threshold=1)

    pool.mark_failure(endpoint, f"transport failed ({endpoint}), retry.")

    snapshot = pool.snapshot()[0]
    assert snapshot["healthy"] is False
    assert snapshot["failure_count"] == 1
    assert snapshot["endpoint"] == safe_endpoint
    assert safe_endpoint in str(snapshot["last_error"])
    assert all(secret not in str(snapshot) for secret in secrets)
