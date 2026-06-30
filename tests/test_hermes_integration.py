from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path(os.getenv("HERMES_AGENT_ROOT", str(Path.home() / "GitHub" / "hermes-agent"))).expanduser()
PROVIDER_PATH = ROOT / "integrations" / "hermes-fusion-memory" / "__init__.py"


def load_provider_module():
    if str(HERMES_ROOT) not in sys.path:
        sys.path.insert(0, str(HERMES_ROOT))
    spec = importlib.util.spec_from_file_location("fusion_memory_hermes_provider", PROVIDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HermesFusionMemoryProviderTests(unittest.TestCase):
    def test_provider_loads_and_exposes_tools(self) -> None:
        module = load_provider_module()
        provider = module.FusionMemoryProvider()
        self.assertEqual(provider.name, "fusion_memory")
        schemas = provider.get_tool_schemas()
        names = {schema["name"] for schema in schemas}
        self.assertEqual(
            names,
            {"fusion_memory_search", "fusion_memory_store", "fusion_memory_clear"},
        )

    def test_tool_failure_is_beginner_safe(self) -> None:
        module = load_provider_module()
        provider = module.FusionMemoryProvider()
        with patch.object(provider, "_post_json", side_effect=TimeoutError("socket timeout")):
            result = provider.handle_tool_call("fusion_memory_search", {"query": "preference"})
        payload = json.loads(result)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "service_unavailable")
        self.assertEqual(payload["cause"], "connection_failed")
        self.assertIn("fusion-memory status", payload["message"])
        self.assertNotIn("socket timeout", payload["message"])

    def test_tool_failure_preserves_structured_server_cause(self) -> None:
        module = load_provider_module()
        provider = module.FusionMemoryProvider()
        error = module.FusionMemoryHTTPError("bad_request", "missing_scope", "Request must include scope.")
        with patch.object(provider, "_post_json", side_effect=error):
            result = provider.handle_tool_call("fusion_memory_search", {"query": "preference"})

        payload = json.loads(result)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "bad_request")
        self.assertEqual(payload["cause"], "missing_scope")
        self.assertEqual(payload["message"], "Request must include scope.")

    def test_invalid_timeout_uses_default(self) -> None:
        module = load_provider_module()
        with patch.dict("os.environ", {"FUSION_MEMORY_TIMEOUT_SECONDS": "not-a-number"}):
            provider = module.FusionMemoryProvider()
        self.assertEqual(provider.timeout_seconds, 1.5)

    def test_timeout_is_clamped_to_safe_bounds(self) -> None:
        module = load_provider_module()
        with patch.dict("os.environ", {"FUSION_MEMORY_TIMEOUT_SECONDS": "0.05"}):
            provider = module.FusionMemoryProvider()
        self.assertEqual(provider.timeout_seconds, 0.1)

        with patch.dict("os.environ", {"FUSION_MEMORY_TIMEOUT_SECONDS": "9"}):
            provider = module.FusionMemoryProvider()
        self.assertEqual(provider.timeout_seconds, 2.0)

        with patch.dict("os.environ", {"FUSION_MEMORY_TIMEOUT_SECONDS": "-1"}):
            provider = module.FusionMemoryProvider()
        self.assertEqual(provider.timeout_seconds, 1.5)


if __name__ == "__main__":
    unittest.main()
