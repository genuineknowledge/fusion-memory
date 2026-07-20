from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from fusion_memory.model_server import create_embedding_app, create_reranker_app


class FakeEmbeddingModel:
    model = "fake-embedding"
    device = "cpu"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeRerankerModel:
    model = "fake-reranker"
    device = "cpu"

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(len(query) + len(doc)) for doc in docs]


@pytest.mark.anyio
async def test_embedding_server_exposes_contract_and_health() -> None:
    app = create_embedding_app(FakeEmbeddingModel(), max_concurrency=1)
    async with TestClient(TestServer(app)) as client:
        health = await client.get("/health")
        health_body = await health.json()
        body = await (await client.post("/v1/embeddings", json={"model": "ignored", "input": ["alpha"]})).json()

    assert health.status == 200
    assert health_body == {"ok": True, "kind": "embedding", "model": "fake-embedding", "device": "cpu"}
    assert body["embeddings"] == [[1.0, 0.0]]


@pytest.mark.anyio
async def test_embedding_server_rejects_invalid_request_shape() -> None:
    app = create_embedding_app(FakeEmbeddingModel(), max_concurrency=1)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/embeddings", json={"model": "ignored", "input": "alpha"})
        body = await response.json()

    assert response.status == 400
    assert "input" in body["error"]


@pytest.mark.anyio
async def test_model_servers_require_the_existing_model_field() -> None:
    embedding_app = create_embedding_app(FakeEmbeddingModel(), max_concurrency=1)
    reranker_app = create_reranker_app(FakeRerankerModel(), max_concurrency=1)
    async with TestClient(TestServer(embedding_app)) as embedding_client:
        embedding_response = await embedding_client.post("/v1/embeddings", json={"input": ["alpha"]})
        embedding_body = await embedding_response.json()
    async with TestClient(TestServer(reranker_app)) as reranker_client:
        reranker_response = await reranker_client.post("/v1/rerank", json={"query": "q", "documents": ["one"]})
        reranker_body = await reranker_response.json()

    assert embedding_response.status == 400
    assert embedding_body["error"] == "model must be a string"
    assert reranker_response.status == 400
    assert reranker_body["error"] == "model must be a string"


@pytest.mark.anyio
async def test_reranker_server_exposes_existing_contract() -> None:
    app = create_reranker_app(FakeRerankerModel(), max_concurrency=1)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/rerank",
            json={"model": "ignored", "query": "q", "documents": ["one", "two"]},
        )
        body = await response.json()

    assert response.status == 200
    assert body == {"scores": [4.0, 4.0]}


def test_cli_dispatches_embedding_server_with_one_model_configuration() -> None:
    from fusion_memory.cli import main

    with patch("fusion_memory.model_server.run_embedding_server") as run_server, patch.object(
        sys,
        "argv",
        [
            "fusion-memory",
            "embedding-server",
            "--host",
            "127.0.0.1",
            "--port",
            "9101",
            "--model",
            "/models/embed",
            "--device",
            "cuda:0",
            "--max-concurrency",
            "2",
        ],
    ):
        main()

    run_server.assert_called_once_with("127.0.0.1", 9101, "/models/embed", "cuda:0", 2)
