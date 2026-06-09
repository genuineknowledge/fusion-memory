from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from typing import Protocol
from urllib import request

from fusion_memory.core.config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL
from fusion_memory.core.text import tokenize


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...


class DeterministicEmbedder:
    """Small dependency-free embedder for local tests and MVP behavior.

    It is not meant to replace production embeddings. It gives stable vectors so
    retrieval, scoring, and tests can run without external model services.
    """

    def __init__(self, dimensions: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign * count
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class HTTPEmbeddingClient:
    """Dependency-free embedding adapter for local or hosted JSON endpoints.

    Supported response shapes:
    - `{"embeddings": [[...], ...]}`
    - OpenAI style `{"data": [{"embedding": [...]}, ...]}`
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        model: str = "local-embedding",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.calls: list[dict[str, object]] = []
        self.version = f"http-embedding:{model}"

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        payload = {"model": self.model, "input": texts}
        data = _post_json(self.endpoint, payload, api_key=self.api_key, timeout_seconds=self.timeout_seconds)
        embeddings = _extract_embeddings(data)
        if len(embeddings) != len(texts):
            raise ValueError("embedding endpoint returned a different number of vectors than requested texts")
        self.calls.append(
            {
                "model": self.model,
                "text_count": len(texts),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": data.get("usage", {}),
            }
        )
        return embeddings


class Qwen3EmbeddingClient:
    """Optional local Qwen3 embedding adapter.

    This class keeps the core package dependency-free by importing
    `sentence_transformers` only when the adapter is instantiated.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        output_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        device: str | None = None,
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        model_kwargs: dict | None = None,
        encode_kwargs: dict | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3EmbeddingClient requires optional ML dependencies. "
                "Install the qwen extra or provide an HTTPEmbeddingClient endpoint."
            ) from exc
        self.model = model
        self.output_dimension = output_dimension
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.encode_kwargs = dict(encode_kwargs or {})
        self.calls: list[dict[str, object]] = []
        self.version = f"qwen3-embedding:{model}:{output_dimension}"
        self._model = SentenceTransformer(
            model,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs or {},
        )

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        kwargs = {
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "convert_to_numpy": True,
            **self.encode_kwargs,
        }
        embeddings = self._model.encode(texts, **kwargs)
        vectors = [_fit_dimension(_as_float_vector(list(vector)), self.output_dimension) for vector in embeddings]
        self.calls.append(
            {
                "model": self.model,
                "text_count": len(texts),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": {},
            }
        )
        return vectors


def cosine_dense(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _post_json(endpoint: str, payload: dict, *, api_key: str | None, timeout_seconds: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("embedding endpoint must return a JSON object")
    return data


def _extract_embeddings(data: dict) -> list[list[float]]:
    if isinstance(data.get("embeddings"), list):
        return [_as_float_vector(item) for item in data["embeddings"]]
    if isinstance(data.get("data"), list):
        return [_as_float_vector(item["embedding"]) for item in data["data"] if isinstance(item, dict) and "embedding" in item]
    raise ValueError("embedding endpoint did not return embeddings")


def _as_float_vector(value) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("embedding vector must be a list")
    return [float(item) for item in value]


def _fit_dimension(vector: list[float], dimension: int) -> list[float]:
    if len(vector) == dimension:
        return vector
    if len(vector) > dimension:
        fitted = vector[:dimension]
        norm = math.sqrt(sum(value * value for value in fitted))
        return [value / norm for value in fitted] if norm else fitted
    return vector + [0.0] * (dimension - len(vector))
