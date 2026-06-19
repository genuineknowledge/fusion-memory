from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fusion_memory.api.service import MemoryService
from fusion_memory.core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, MemoryConfig
from fusion_memory.core.embedding import HTTPEmbeddingClient, Qwen3EmbeddingClient
from fusion_memory.core.llm import OpenAICompatibleLLMClient
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor
from fusion_memory.retrieval.reranker import HTTPReranker, Qwen3Reranker


@dataclass(frozen=True)
class RuntimeRetrievalFlags:
    dual_event_ordering_shadow: bool = False
    production_selector: str = "legacy"


def memory_service_from_env(
    db_path: str | Path = ":memory:",
    *,
    config: MemoryConfig | None = None,
    storage_backend: str | None = None,
) -> MemoryService:
    """Build MemoryService from environment-backed runtime model config.

    Defaults preserve dependency-free local behavior. Production or smoke runs can
    opt into Qwen/HTTP adapters without hard-coding endpoints or secrets.
    """

    return MemoryService(
        db_path,
        config=config,
        storage_backend=storage_backend or os.getenv("FUSION_MEMORY_STORAGE_BACKEND", "sqlite"),
        embedder=_build_embedder(),
        reranker=_build_reranker(),
        extractor=_build_extractor(),
        async_extractor=_build_async_extractor(),
        query_intent_refiner=_build_query_intent_refiner(),
        query_intent_refiner_min_confidence=_float_env("FUSION_MEMORY_QUERY_INTENT_MIN_CONFIDENCE", 0.70),
        query_intent_refiner_mode=os.getenv("FUSION_MEMORY_QUERY_INTENT_MODE", "off"),
        retrieval_flags=build_runtime_retrieval_flags(),
    )


def build_runtime_retrieval_flags() -> RuntimeRetrievalFlags:
    selector = os.getenv("FUSION_MEMORY_EVENT_ORDERING_SELECTOR", "legacy").strip().lower()
    if selector != "legacy":
        raise ValueError(f"unsupported event ordering selector: {selector}")
    return RuntimeRetrievalFlags(
        dual_event_ordering_shadow=_bool_env("FUSION_MEMORY_DUAL_EVENT_ORDERING_SHADOW", False),
        production_selector=selector,
    )


def _build_embedder() -> Any | None:
    provider = os.getenv("FUSION_MEMORY_EMBEDDING_PROVIDER", "").strip().lower()
    if not provider or provider == "deterministic":
        return None
    if provider == "qwen":
        return Qwen3EmbeddingClient(
            model=os.getenv("FUSION_MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            output_dimension=_int_env("FUSION_MEMORY_EMBEDDING_DIMENSION", 1024),
            device=_optional_env("FUSION_MEMORY_EMBEDDING_DEVICE"),
            batch_size=_int_env("FUSION_MEMORY_EMBEDDING_BATCH_SIZE", 8),
            model_kwargs=_model_kwargs(),
        )
    if provider == "http":
        endpoint = _required_env("FUSION_MEMORY_EMBEDDING_ENDPOINT")
        return HTTPEmbeddingClient(
            endpoint,
            api_key=_optional_env("FUSION_MEMORY_EMBEDDING_API_KEY"),
            model=os.getenv("FUSION_MEMORY_EMBEDDING_MODEL", "local-embedding"),
            timeout_seconds=_float_env("FUSION_MEMORY_EMBEDDING_TIMEOUT_SECONDS", 30.0),
        )
    raise ValueError(f"unsupported FUSION_MEMORY_EMBEDDING_PROVIDER: {provider}")


def _build_reranker() -> Any | None:
    provider = os.getenv("FUSION_MEMORY_RERANKER_PROVIDER", "").strip().lower()
    if not provider or provider == "lexical":
        return None
    if provider == "qwen":
        return Qwen3Reranker(
            model=os.getenv("FUSION_MEMORY_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            device=_optional_env("FUSION_MEMORY_RERANKER_DEVICE"),
            batch_size=_int_env("FUSION_MEMORY_RERANKER_BATCH_SIZE", 8),
            model_kwargs=_model_kwargs(),
        )
    if provider == "http":
        endpoint = _required_env("FUSION_MEMORY_RERANKER_ENDPOINT")
        return HTTPReranker(
            endpoint,
            api_key=_optional_env("FUSION_MEMORY_RERANKER_API_KEY"),
            model=os.getenv("FUSION_MEMORY_RERANKER_MODEL", "local-reranker"),
            timeout_seconds=_float_env("FUSION_MEMORY_RERANKER_TIMEOUT_SECONDS", 30.0),
        )
    raise ValueError(f"unsupported FUSION_MEMORY_RERANKER_PROVIDER: {provider}")


def _build_extractor() -> Any | None:
    mode = os.getenv("FUSION_MEMORY_EXTRACTOR_MODE", "off").strip().lower()
    if mode not in {"off", "sync"}:
        return None
    if mode != "sync":
        return None
    endpoint = _optional_env("FUSION_MEMORY_EXTRACTOR_ENDPOINT") or _endpoint_from_base_url(
        "FUSION_MEMORY_EXTRACTOR_BASE_URL",
        "chat/completions",
    )
    if not endpoint:
        return None
    client = OpenAICompatibleLLMClient(
        endpoint,
        api_key=_optional_env("FUSION_MEMORY_EXTRACTOR_API_KEY"),
        model=os.getenv("FUSION_MEMORY_EXTRACTOR_MODEL", "local-structured-extractor"),
        timeout_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_TIMEOUT_SECONDS", 30.0),
        retry_attempts=_int_env("FUSION_MEMORY_EXTRACTOR_RETRY_ATTEMPTS", 3),
        retry_backoff_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_RETRY_BACKOFF_SECONDS", 2.0),
        retry_max_backoff_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_RETRY_MAX_BACKOFF_SECONDS", 60.0),
        min_interval_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_MIN_INTERVAL_SECONDS", 0.0),
    )
    return StructuredLLMExtractor(
        client,
        prompt_version=os.getenv("FUSION_MEMORY_EXTRACTOR_PROMPT_VERSION", "llm-extractor-v0"),
    )


def _build_async_extractor() -> Any | None:
    mode = os.getenv("FUSION_MEMORY_EXTRACTOR_MODE", "off").strip().lower()
    if mode not in {"async"}:
        return None
    endpoint = _optional_env("FUSION_MEMORY_EXTRACTOR_ENDPOINT") or _endpoint_from_base_url(
        "FUSION_MEMORY_EXTRACTOR_BASE_URL",
        "chat/completions",
    )
    if not endpoint:
        return None
    client = OpenAICompatibleLLMClient(
        endpoint,
        api_key=_optional_env("FUSION_MEMORY_EXTRACTOR_API_KEY"),
        model=os.getenv("FUSION_MEMORY_EXTRACTOR_MODEL", "local-structured-extractor"),
        timeout_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_TIMEOUT_SECONDS", 30.0),
        retry_attempts=_int_env("FUSION_MEMORY_EXTRACTOR_RETRY_ATTEMPTS", 3),
        retry_backoff_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_RETRY_BACKOFF_SECONDS", 2.0),
        retry_max_backoff_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_RETRY_MAX_BACKOFF_SECONDS", 60.0),
        min_interval_seconds=_float_env("FUSION_MEMORY_EXTRACTOR_MIN_INTERVAL_SECONDS", 0.0),
    )
    return StructuredLLMExtractor(
        client,
        prompt_version=os.getenv("FUSION_MEMORY_EXTRACTOR_PROMPT_VERSION", "llm-extractor-v0"),
    )


def _build_query_intent_refiner() -> Any | None:
    mode = os.getenv("FUSION_MEMORY_QUERY_INTENT_MODE", "off").strip().lower()
    if mode not in {"auto", "always"}:
        return None
    endpoint = _optional_env("FUSION_MEMORY_QUERY_INTENT_ENDPOINT") or _endpoint_from_base_url(
        "FUSION_MEMORY_QUERY_INTENT_BASE_URL",
        "chat/completions",
    )
    if not endpoint:
        return None
    return OpenAICompatibleLLMClient(
        endpoint,
        api_key=_optional_env("FUSION_MEMORY_QUERY_INTENT_API_KEY"),
        model=os.getenv("FUSION_MEMORY_QUERY_INTENT_MODEL", "local-query-intent"),
        timeout_seconds=_float_env("FUSION_MEMORY_QUERY_INTENT_TIMEOUT_SECONDS", 20.0),
        retry_attempts=_int_env("FUSION_MEMORY_QUERY_INTENT_RETRY_ATTEMPTS", 3),
        retry_backoff_seconds=_float_env("FUSION_MEMORY_QUERY_INTENT_RETRY_BACKOFF_SECONDS", 1.0),
        retry_max_backoff_seconds=_float_env("FUSION_MEMORY_QUERY_INTENT_RETRY_MAX_BACKOFF_SECONDS", 30.0),
        min_interval_seconds=_float_env("FUSION_MEMORY_QUERY_INTENT_MIN_INTERVAL_SECONDS", 0.0),
    )


def _model_kwargs() -> dict[str, Any]:
    cache_dir = _optional_env("FUSION_MEMORY_MODEL_CACHE")
    return {"cache_dir": cache_dir} if cache_dir else {}


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _endpoint_from_base_url(name: str, suffix: str) -> str | None:
    base_url = _optional_env(name)
    if not base_url:
        return None
    if base_url.rstrip("/").endswith(f"/{suffix}"):
        return base_url.rstrip("/")
    return f"{base_url.rstrip('/')}/{suffix}"


def _required_env(name: str) -> str:
    value = _optional_env(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _int_env(name: str, default: int) -> int:
    value = _optional_env(name)
    return int(value) if value is not None else default


def _float_env(name: str, default: float) -> float:
    value = _optional_env(name)
    return float(value) if value is not None else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
