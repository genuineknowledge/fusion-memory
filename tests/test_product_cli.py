from __future__ import annotations

import json
import os
import tempfile
import unittest
import socket
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import fusion_memory.product as product
from fusion_memory.product import (
    backup_data,
    configure_interactive,
    doctor,
    init_home,
    load_config,
    product_paths,
    render_human,
    service_status,
    start_service,
    stop_service,
    upgrade,
    _service_env,
)


class ProductCliTests(unittest.TestCase):
    def test_install_agent_dry_run_cli_json(self) -> None:
        from fusion_memory.cli import main
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        try:
            sys.argv = ["fusion-memory", "install-agent", "--target", "all", "--dry-run", "--json"]
            sys.stdout = StringIO()
            main()
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])

    def test_init_doctor_backup_and_upgrade_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init = init_home(home, port=0)
            self.assertTrue(init["ok"])
            self.assertTrue((home / "config.json").exists())
            config = load_config(home)
            self.assertEqual(init["db"], "postgresql://***:***@127.0.0.1:55433/fusion_memory")
            self.assertEqual(config["storage_backend"], "postgres")
            self.assertEqual(config["embedding"]["provider"], "qwen")
            self.assertIn("Qwen3-Embedding-0.6B", config["embedding"]["model"])
            self.assertEqual(config["reranker"]["provider"], "qwen")
            self.assertIn("Qwen3-Reranker-0.6B", config["reranker"]["model"])
            self.assertEqual(config["extractor"]["provider"], "rule")
            self.assertEqual(config["query_intent"]["provider"], "off")
            self.assertTrue(hasattr(product, "default_product_settings"))
            defaults = product.default_product_settings(product_paths(home))
            self.assertEqual(defaults["storage_backend"], "postgres")

            report = doctor(home)
            self.assertTrue(report["ok"])
            self.assertTrue(report["checks"])

            (home / "fusion-memory.sqlite3").write_text("seed", encoding="utf-8")
            backup = backup_data(home)
            self.assertTrue(backup["ok"])
            self.assertGreaterEqual(len(backup["files"]), 2)

            plan = upgrade(home, dry_run=True)
            self.assertTrue(plan["ok"])
            self.assertTrue(plan["dry_run"])
            self.assertIn("command", plan)

    def test_interactive_configures_models_without_storing_secret(self) -> None:
        answers = iter(
            [
                "",  # host
                "18766",  # port
                "",  # database sqlite
                "",  # sqlite path
                "http",  # embedding
                "http://embed.example/v1/embeddings",
                "embed-model",
                "FUSION_MEMORY_MODEL_API_KEY",
                "qwen",  # reranker
                "/tmp/qwen-reranker",
                "cpu",
                "api",  # extractor
                "http://llm.example/v1",
                "extractor-model",
                "FUSION_MEMORY_MODEL_API_KEY",
                "",  # query router off
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, patch("builtins.input", lambda _prompt="": next(answers)):
            home = Path(tmp)
            result = configure_interactive(home)
            self.assertTrue(result["ok"])
            raw = (home / "config.json").read_text(encoding="utf-8")
            self.assertNotIn("secret-value", raw)
            config = json.loads(raw)
            self.assertEqual(config["embedding"]["provider"], "http")
            self.assertEqual(config["reranker"]["provider"], "qwen")
            self.assertEqual(config["extractor"]["provider"], "api")
            self.assertEqual(config["query_intent"]["provider"], "off")

            with patch.dict(os.environ, {"FUSION_MEMORY_MODEL_API_KEY": "secret-value"}):
                env = _service_env(config)
            self.assertEqual(env["FUSION_MEMORY_EMBEDDING_PROVIDER"], "http")
            self.assertEqual(env["FUSION_MEMORY_EMBEDDING_API_KEY"], "secret-value")
            self.assertEqual(env["FUSION_MEMORY_RERANKER_PROVIDER"], "qwen")
            self.assertEqual(env["FUSION_MEMORY_EXTRACTOR_API_KEY"], "secret-value")

    def test_interactive_and_human_output_redact_default_postgres_credentials(self) -> None:
        answers = iter(
            [
                "",  # host
                "18767",  # port
                "",  # default postgres database
                "",  # default postgres DSN
                "",  # default qwen embedding
                "",  # default qwen embedding model
                "",  # default qwen embedding device
                "",  # default qwen reranker
                "",  # default qwen reranker model
                "",  # default qwen reranker device
                "",  # default rule extractor
                "",  # default off query router
            ]
        )
        output = StringIO()

        def fake_input(prompt: str = "") -> str:
            print(prompt, end="")
            return next(answers)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("builtins.input", fake_input),
            redirect_stdout(output),
        ):
            result = configure_interactive(Path(tmp))

        self.assertEqual(result["db"], "postgresql://***:***@127.0.0.1:55433/fusion_memory")
        self.assertNotIn("fusion:fusion", json.dumps(result))
        rendered = render_human(result)
        self.assertIn("postgresql://***:***@127.0.0.1:55433/fusion_memory", rendered)
        self.assertNotIn("fusion:fusion", rendered)
        wizard_text = output.getvalue()
        self.assertIn("Postgres / pgvector (recommended)", wizard_text)
        self.assertIn("Qwen3 embedding (recommended)", wizard_text)
        self.assertIn("Qwen3 reranker (recommended)", wizard_text)
        self.assertNotIn("fusion:fusion", wizard_text)
        self.assertNotIn("SQLite local file (recommended)", wizard_text)
        self.assertNotIn("Built-in lightweight embedding (recommended)", wizard_text)
        self.assertNotIn("Built-in lexical reranker (recommended)", wizard_text)

    def test_start_status_and_stop_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            port = _free_port()
            init_home(
                home,
                port=port,
                settings={
                    "db": str(home / "fusion-memory.sqlite3"),
                    "storage_backend": "sqlite",
                    "embedding": {"provider": "deterministic"},
                    "reranker": {"provider": "lexical"},
                },
            )

            started = start_service(home, wait_seconds=10)
            try:
                self.assertTrue(started["ok"], started)
                status = service_status(home)
                self.assertTrue(status["running"], status)
            finally:
                stopped = stop_service(home)
                self.assertTrue(stopped["ok"], stopped)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
