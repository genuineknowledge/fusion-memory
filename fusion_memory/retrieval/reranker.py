from __future__ import annotations

import json
import math
import time
from typing import Protocol
from urllib import request

from fusion_memory.core.config import DEFAULT_RERANKER_MODEL
from fusion_memory.core.models import Candidate
from fusion_memory.core.text import jaccard, tokenize


class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]:
        ...


class LexicalCrossEncoderReranker:
    """Dependency-free local reranker.

    This is a deterministic stand-in for a production cross-encoder. It scores
    each query/document pair using token overlap, phrase containment, and a small
    length prior. The interface is intentionally compatible with a real reranker.
    """

    version = "lexical-cross-encoder-v0"

    def score(self, query: str, docs: list[str]) -> list[float]:
        q_tokens = set(tokenize(query))
        q_lower = query.lower()
        scores: list[float] = []
        for doc in docs:
            d_tokens = set(tokenize(doc))
            d_lower = doc.lower()
            overlap = jaccard(q_tokens, d_tokens)
            phrase = 0.15 if q_lower and q_lower in d_lower else 0.0
            length_prior = min(0.08, len(d_tokens) / 1000)
            exact_hits = sum(1 for token in q_tokens if token in d_tokens)
            exact_prior = min(0.20, exact_hits * 0.03)
            scores.append(overlap + phrase + length_prior + exact_prior)
        return scores


class HTTPReranker:
    """Dependency-free reranker adapter for cross-encoder style endpoints.

    Supported response shapes:
    - `{"scores": [0.1, ...]}`
    - `{"data": [{"score": 0.1}, ...]}`
    - DashScope rerank style `{"results": [{"index": 0, "relevance_score": 0.1}, ...]}`
    - DashScope output wrapper `{"output": {"results": [{"index": 0, "relevance_score": 0.1}, ...]}}`
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        model: str = "local-reranker",
        timeout_seconds: float = 30.0,
        top_n: int | None = None,
        instruct: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.top_n = top_n
        self.instruct = instruct
        self.calls: list[dict[str, object]] = []
        self.version = f"http-reranker:{model}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        started = time.perf_counter()
        payload = {"model": self.model, "query": query, "documents": docs}
        if self.top_n is not None:
            payload["top_n"] = self.top_n
        if self.instruct:
            payload["instruct"] = self.instruct
        data = _post_json(self.endpoint, payload, api_key=self.api_key, timeout_seconds=self.timeout_seconds)
        scores = _extract_scores(data, expected_count=len(docs))
        if len(scores) != len(docs):
            raise ValueError("reranker endpoint returned a different number of scores than requested docs")
        self.calls.append(
            {
                "model": self.model,
                "doc_count": len(docs),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": data.get("usage", {}),
            }
        )
        return scores


class Qwen3Reranker:
    """Optional local Qwen3 cross-encoder reranker."""

    def __init__(
        self,
        model: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 8,
        apply_sigmoid: bool = False,
        model_kwargs: dict | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3Reranker requires optional ML dependencies. "
                "Install the qwen extra or provide an HTTPReranker endpoint."
            ) from exc
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.apply_sigmoid = apply_sigmoid
        self.calls: list[dict[str, object]] = []
        self.version = f"qwen3-reranker:{model}"
        self._model = CrossEncoder(
            model,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs or {},
        )

    def score(self, query: str, docs: list[str]) -> list[float]:
        started = time.perf_counter()
        raw_scores = self._model.predict([(query, doc) for doc in docs], batch_size=self.batch_size)
        scores = [float(score) for score in list(raw_scores)]
        if self.apply_sigmoid:
            scores = [1.0 / (1.0 + math.exp(-score)) for score in scores]
        self.calls.append(
            {
                "model": self.model,
                "doc_count": len(docs),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": {},
            }
        )
        return scores


def rerank_candidates(query: str, candidates: list[Candidate], reranker: Reranker) -> list[Candidate]:
    rerank_scores = reranker.score(query, [candidate.text for candidate in candidates])
    normalized_scores = _normalize_scores(rerank_scores)
    out: list[Candidate] = []
    for candidate, rerank_score, rerank_score_normalized in zip(candidates, rerank_scores, normalized_scores):
        scores = dict(candidate.scores)
        scores["rerank_score"] = rerank_score
        scores["rerank_score_normalized"] = rerank_score_normalized
        scores["utility_score"] = 0.70 * scores.get("utility_score", 0.0) + 0.30 * rerank_score_normalized
        out.append(
            Candidate(
                id=candidate.id,
                type=candidate.type,
                text=candidate.text,
                source=candidate.source,
                scores=scores,
                source_span_ids=candidate.source_span_ids,
                metadata=candidate.metadata,
            )
        )
    out.sort(
        key=lambda candidate: (
            candidate.scores.get("utility_score", 0.0),
            candidate.scores.get("rerank_score_normalized", 0.0),
            candidate.scores.get("rerank_score", 0.0),
        ),
        reverse=True,
    )
    return out


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high - low < 1e-9:
        return [0.5 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


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
        raise ValueError("reranker endpoint must return a JSON object")
    return data


def _extract_scores(data: dict, *, expected_count: int | None = None) -> list[float]:
    if isinstance(data.get("scores"), list):
        return [float(item) for item in data["scores"]]
    if isinstance(data.get("data"), list):
        return [float(item["score"]) for item in data["data"] if isinstance(item, dict) and "score" in item]
    if isinstance(data.get("output"), dict) and isinstance(data["output"].get("results"), list):
        return _scores_from_indexed_results(data["output"]["results"], expected_count=expected_count)
    if isinstance(data.get("results"), list):
        return _scores_from_indexed_results(data["results"], expected_count=expected_count)
    raise ValueError("reranker endpoint did not return scores")


def _scores_from_indexed_results(results: list, *, expected_count: int | None = None) -> list[float]:
    count = expected_count
    if count is None:
        indexes = [int(item["index"]) for item in results if isinstance(item, dict) and "index" in item]
        count = max(indexes) + 1 if indexes else 0
    scores = [0.0] * count
    for item in results:
        if not isinstance(item, dict) or "index" not in item:
            continue
        if "relevance_score" in item:
            score = item["relevance_score"]
        elif "score" in item:
            score = item["score"]
        else:
            continue
        index = int(item["index"])
        if 0 <= index < count:
            scores[index] = float(score)
    return scores
