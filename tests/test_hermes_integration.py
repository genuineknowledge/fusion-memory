from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path("/public/home/wwb/GitHub/hermes-agent")
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
        self.assertIn("fusion-memory doctor", payload["message"])
        self.assertNotIn("socket timeout", payload["message"])


if __name__ == "__main__":
    unittest.main()
