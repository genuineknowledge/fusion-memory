from __future__ import annotations

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
