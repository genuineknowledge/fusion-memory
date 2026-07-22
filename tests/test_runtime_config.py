from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fusion_memory.core.config import MemoryConfig
from fusion_memory.core.runtime_config import build_runtime_retrieval_flags, memory_service_from_env
from fusion_memory.model_pool import PooledEmbedder, PooledReranker


class RuntimeRetrievalFlagTests(unittest.TestCase):
    def test_runtime_config_snapshot_contains_only_product_rerank_setting(self) -> None:
        rerank_settings = {
            name: value
            for name, value in MemoryConfig().snapshot().items()
            if name.endswith("rerank_top_n")
        }

        self.assertEqual(rerank_settings, {"balanced_mode_rerank_top_n": 50})

    def test_event_ordering_flags_keep_only_legacy_selector(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            flags = build_runtime_retrieval_flags()

        self.assertFalse(hasattr(flags, "dual_event_ordering_shadow"))
        self.assertEqual(flags.production_selector, "legacy")

    def test_dual_event_ordering_shadow_env_is_ignored(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_DUAL_EVENT_ORDERING_SHADOW": "1",
                "FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "legacy",
            },
            clear=True,
        ):
            flags = build_runtime_retrieval_flags()

        self.assertFalse(hasattr(flags, "dual_event_ordering_shadow"))
        self.assertEqual(flags.production_selector, "legacy")

    def test_event_ordering_selector_rejects_unapproved_values(self) -> None:
        with patch.dict(os.environ, {"FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "graph"}, clear=True):
            with self.assertRaisesRegex(ValueError, "unsupported event ordering selector"):
                build_runtime_retrieval_flags()

    def test_non_legacy_event_ordering_selector_is_rejected_for_product_default(self) -> None:
        with patch.dict(os.environ, {"FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "dual"}, clear=True):
            with self.assertRaisesRegex(ValueError, "unsupported event ordering selector"):
                build_runtime_retrieval_flags()

    def test_omitted_query_intent_mode_keeps_router_off_on_memory_service(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class DummyMemoryService:
            def __init__(self, *args, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        with patch.dict(os.environ, {}, clear=True), patch(
            "fusion_memory.core.runtime_config.MemoryService", DummyMemoryService
        ):
            memory_service_from_env()

        self.assertEqual(captured_kwargs["query_intent_refiner_mode"], "off")
        self.assertIsNone(captured_kwargs["query_intent_refiner"])

    def test_query_intent_environment_is_passed_to_memory_service(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class DummyMemoryService:
            def __init__(self, *args, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_QUERY_INTENT_MODE": " ALWAYS ",
                "FUSION_MEMORY_QUERY_INTENT_ENDPOINT": "http://intent/v1/chat/completions",
                "FUSION_MEMORY_QUERY_INTENT_MIN_CONFIDENCE": "0.84",
            },
            clear=True,
        ), patch("fusion_memory.core.runtime_config.MemoryService", DummyMemoryService):
            memory_service_from_env()

        self.assertEqual(captured_kwargs["query_intent_refiner_mode"], "always")
        self.assertEqual(captured_kwargs["query_intent_refiner_min_confidence"], 0.84)
        self.assertIsNotNone(captured_kwargs["query_intent_refiner"])

    def test_invalid_query_intent_min_confidence_fails_during_service_construction(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_QUERY_INTENT_MODE": "off",
                "FUSION_MEMORY_QUERY_INTENT_MIN_CONFIDENCE": "nan",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "intent_refiner_min_confidence must be a finite number between 0.0 and 1.0",
            ):
                memory_service_from_env()

    def test_memory_service_from_env_raises_for_invalid_selector(self) -> None:
        with patch.dict(os.environ, {"FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "graph"}, clear=True):
            with self.assertRaisesRegex(ValueError, "unsupported event ordering selector"):
                memory_service_from_env()

    def test_embedding_endpoint_list_builds_a_pooled_adapter_and_takes_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_EMBEDDING_PROVIDER": "http",
                "FUSION_MEMORY_EMBEDDING_ENDPOINT": "http://legacy/v1/embeddings",
                "FUSION_MEMORY_EMBEDDING_ENDPOINTS": "http://one/v1/embeddings, http://two/v1/embeddings",
            },
            clear=True,
        ):
            with patch("fusion_memory.core.runtime_config.MemoryService") as service:
                memory_service_from_env()

        assert isinstance(service.call_args.kwargs["embedder"], PooledEmbedder)
        assert service.call_args.kwargs["embedder"].pool.endpoints == [
            "http://one/v1/embeddings",
            "http://two/v1/embeddings",
        ]

    def test_reranker_singular_endpoint_remains_a_one_element_pool(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_RERANKER_PROVIDER": "http",
                "FUSION_MEMORY_RERANKER_ENDPOINT": "http://one/v1/rerank",
            },
            clear=True,
        ):
            with patch("fusion_memory.core.runtime_config.MemoryService") as service:
                memory_service_from_env()

        assert isinstance(service.call_args.kwargs["reranker"], PooledReranker)
        assert service.call_args.kwargs["reranker"].pool.endpoints == ["http://one/v1/rerank"]

    def test_empty_explicit_endpoint_list_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_EMBEDDING_PROVIDER": "http",
                "FUSION_MEMORY_EMBEDDING_ENDPOINTS": " , ",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "FUSION_MEMORY_EMBEDDING_ENDPOINTS must contain at least one endpoint"):
                memory_service_from_env()
