from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fusion_memory.core.runtime_config import build_runtime_retrieval_flags


class RuntimeRetrievalFlagTests(unittest.TestCase):
    def test_dual_event_ordering_shadow_defaults_off_and_legacy_selector(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            flags = build_runtime_retrieval_flags()

        self.assertFalse(flags.dual_event_ordering_shadow)
        self.assertEqual(flags.production_selector, "legacy")

    def test_dual_event_ordering_shadow_can_be_enabled_without_changing_selector(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_DUAL_EVENT_ORDERING_SHADOW": "1",
                "FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "legacy",
            },
            clear=True,
        ):
            flags = build_runtime_retrieval_flags()

        self.assertTrue(flags.dual_event_ordering_shadow)
        self.assertEqual(flags.production_selector, "legacy")

    def test_event_ordering_selector_rejects_unapproved_values(self) -> None:
        with patch.dict(os.environ, {"FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "graph"}, clear=True):
            with self.assertRaisesRegex(ValueError, "unsupported event ordering selector"):
                build_runtime_retrieval_flags()
