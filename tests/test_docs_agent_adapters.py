from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AgentAdapterDocsTests(unittest.TestCase):
    def test_docs_include_beginner_commands_and_no_secret_values(self) -> None:
        docs = [
            ROOT / "README.md",
            ROOT / "docs" / "quickstart.md",
            ROOT / "docs" / "agent-adapters.md",
            ROOT / "docs" / "errors.md",
            ROOT / "docs" / "deployment-qwen-postgres.md",
            ROOT / "integrations" / "openclaw-fusion-memory" / "README.md",
            ROOT / "integrations" / "hermes-fusion-memory" / "README.md",
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in docs)
        self.assertIn("fusion-memory install-agent --target all", text)
        self.assertIn("fusion-memory install-agent --target dolphin", text)
        self.assertIn("sync-haitun-history", text)
        self.assertIn("Haitun recovery", text)
        self.assertIn("fusion-memory doctor", text)
        self.assertIn("OpenClaw recovery", text)
        self.assertIn("Hermes recovery", text)
        self.assertIn("Fusion-Agent recovery", text)
        self.assertIn("$env:PSI_MEMORY_BASE_URL", text)
        self.assertIn("set PSI_MEMORY_BASE_URL=", text)
        self.assertIn("MODEL_CONFIG_FILE", text)
        self.assertNotIn("sk-", text)
        self.assertNotIn("/public/home/wwb", text)
        self.assertNotIn("/home/wwb", text)
        self.assertNotIn("Traceback", text)
        self.assertNotIn("Exception:", text)
        self.assertNotIn("HTTP 500", text)
        self.assertNotIn("psql:", text)
        self.assertNotIn("psi_memories", text)

    def test_fusion_memory_setup_skill_documents_update_install_and_passive_persistence(self) -> None:
        skill = ROOT / "integrations" / "dolphin-fusion-memory" / "workspace" / "skills" / "fusion-memory-setup" / "SKILL.md"
        text = skill.read_text(encoding="utf-8")

        self.assertIn("default tracks fusion-memory main", text)
        self.assertIn("check before installing or upgrading", text)
        self.assertIn("estimate", text)
        self.assertIn("10-20 minutes", text)
        self.assertIn("passive persistence is on by default", text)
        self.assertIn("automatic turn sync", text)
        self.assertIn("fusion-memory --db fusion-memory.sqlite3 sync-haitun-history", text)
        self.assertNotIn("postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory", text)
        self.assertIn("memory_add is for explicit durable facts", text)
        self.assertNotIn("LLM extractor", text)


if __name__ == "__main__":
    unittest.main()
