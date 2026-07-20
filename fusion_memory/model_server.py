from __future__ import annotations

from typing import Any, Protocol

import anyio
from aiohttp import web

from fusion_memory.core.embedding import Qwen3EmbeddingClient
from fusion_memory.retrieval.reranker import Qwen3Reranker


MODEL_SEMAPHORE: web.AppKey[anyio.Semaphore] = web.AppKey("model_semaphore")


class EmbeddingModel(Protocol):
    model: str
    device: str | None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class RerankerModel(Protocol):
    model: str
    device: str | None

    def score(self, query: str, docs: list[str]) -> list[float]:
        ...


def create_embedding_app(model: EmbeddingModel, max_concurrency: int) -> web.Application:
    """Create an embedding process app around one already-loaded local model."""
    app = _base_app(model, max_concurrency=max_concurrency, kind="embedding")

    async def embeddings(request: web.Request) -> web.Response:
        payload = await _request_object(request)
        _require_model(payload)
        texts = payload.get("input")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise web.HTTPBadRequest(text="input must be a list of strings")
        semaphore = request.app[MODEL_SEMAPHORE]
        async with semaphore:
            vectors = await anyio.to_thread.run_sync(model.embed_texts, texts)
        return web.json_response({"embeddings": vectors})

    app.router.add_post("/v1/embeddings", embeddings)
    return app


def create_reranker_app(model: RerankerModel, max_concurrency: int) -> web.Application:
    """Create a reranker process app around one already-loaded local model."""
    app = _base_app(model, max_concurrency=max_concurrency, kind="reranker")

    async def rerank(request: web.Request) -> web.Response:
        payload = await _request_object(request)
        _require_model(payload)
        query = payload.get("query")
        documents = payload.get("documents")
        if not isinstance(query, str):
            raise web.HTTPBadRequest(text="query must be a string")
        if not isinstance(documents, list) or not all(isinstance(document, str) for document in documents):
            raise web.HTTPBadRequest(text="documents must be a list of strings")
        semaphore = request.app[MODEL_SEMAPHORE]
        async with semaphore:
            scores = await anyio.to_thread.run_sync(model.score, query, documents)
        return web.json_response({"scores": scores})

    app.router.add_post("/v1/rerank", rerank)
    return app


def run_embedding_server(host: str, port: int, model: str, device: str | None, max_concurrency: int) -> None:
    """Load a single Qwen embedding model and serve it until process shutdown."""
    loaded_model = Qwen3EmbeddingClient(model=model, device=device)
    web.run_app(create_embedding_app(loaded_model, max_concurrency), host=host, port=port)


def run_reranker_server(host: str, port: int, model: str, device: str | None, max_concurrency: int) -> None:
    """Load a single Qwen reranker model and serve it until process shutdown."""
    loaded_model = Qwen3Reranker(model=model, device=device)
    web.run_app(create_reranker_app(loaded_model, max_concurrency), host=host, port=port)


def _base_app(model: Any, *, max_concurrency: int, kind: str) -> web.Application:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    app = web.Application()
    app[MODEL_SEMAPHORE] = anyio.Semaphore(max_concurrency)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "kind": kind,
                "model": str(getattr(model, "model", "local-model")),
                "device": getattr(model, "device", None),
            }
        )

    @web.middleware
    async def errors(_request: web.Request, handler):
        try:
            return await handler(_request)
        except web.HTTPException as exc:
            if exc.status == 400:
                return web.json_response({"error": exc.text}, status=400)
            raise
        except Exception:
            return web.json_response({"error": "model inference failed"}, status=500)

    app.middlewares.append(errors)
    app.router.add_get("/health", health)
    return app


async def _request_object(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(text="request body must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    return payload


def _require_model(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        raise web.HTTPBadRequest(text="model must be a string")
