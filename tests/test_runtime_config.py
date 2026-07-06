from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fusion_memory.core.runtime_config import build_runtime_retrieval_flags, memory_service_from_env


class RuntimeRetrievalFlagTests(unittest.TestCase):
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

    def test_memory_service_from_env_raises_for_invalid_selector(self) -> None:
        with patch.dict(os.environ, {"FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "graph"}, clear=True):
            with self.assertRaisesRegex(ValueError, "unsupported event ordering selector"):
                memory_service_from_env()
