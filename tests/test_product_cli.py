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
    safe_product_error,
    render_human,
    service_status,
    start_service,
    stop_service,
    upgrade,
    _service_env,
)


class ProductCliTests(unittest.TestCase):
    def test_render_human_uses_safe_fallback_for_failed_payloads_without_checks(self) -> None:
        rendered = render_human({"ok": False, "message": "Could not connect", "next_step": "Run fusion-memory doctor"})

        self.assertIn("Could not connect", rendered)
        self.assertIn("Run fusion-memory doctor", rendered)
        self.assertNotIn("Traceback", rendered)

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

    def test_install_agent_invalid_target_cli_json_is_beginner_safe(self) -> None:
        from fusion_memory.cli import main
        import sys

        old_argv = sys.argv
        stdout = StringIO()
        stderr = StringIO()
        try:
            sys.argv = ["fusion-memory", "install-agent", "--target", "bad-agent", "--json"]
            with redirect_stdout(stdout), patch("sys.stderr", stderr):
                main()
            payload = json.loads(stdout.getvalue())
        finally:
            sys.argv = old_argv

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "unexpected_error")
        self.assertIn("Choose one of", payload["message"])
        self.assertIn("Run fusion-memory doctor", payload["next_step"])
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn("usage:", combined)
        self.assertNotIn("invalid choice", combined)
        self.assertNotIn("Traceback", combined)

    def test_parser_error_json_includes_normalized_failure_keys(self) -> None:
        from fusion_memory.cli import FusionMemoryArgumentParser
        import sys

        parser = FusionMemoryArgumentParser(prog="fusion-memory")
        stdout = StringIO()
        old_argv = sys.argv
        try:
            sys.argv = ["fusion-memory", "--json"]
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as ctx:
                    parser.error("invalid choice: 'bad-agent'")
        finally:
            sys.argv = old_argv

        payload = json.loads(stdout.getvalue())
        self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "invalid_command")
        self.assertIn("Fusion Memory", payload["message"])
        self.assertIn("next_step", payload)
        self.assertIn("doctor", payload["next_step"])

    def test_cli_routes_command_errors_through_safe_product_error(self) -> None:
        from fusion_memory.cli import main
        import sys

        old_argv = sys.argv
        stdout = StringIO()
        try:
            sys.argv = ["fusion-memory", "doctor", "--json"]
            with redirect_stdout(stdout), patch("fusion_memory.cli.doctor", side_effect=RuntimeError("Traceback (most recent call last): secret stack")):
                exit_code = main()
        finally:
            sys.argv = old_argv

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "unexpected_error")
        self.assertNotIn("Traceback", payload["message"])
        self.assertNotIn("secret stack", payload["message"])

    def test_install_agent_invalid_target_cli_json_includes_failure_schema(self) -> None:
        from fusion_memory.cli import main
        import sys

        old_argv = sys.argv
        stdout = StringIO()
        try:
            sys.argv = ["fusion-memory", "install-agent", "--target", "bad-agent", "--json"]
            with redirect_stdout(stdout):
                main()
            payload = json.loads(stdout.getvalue())
        finally:
            sys.argv = old_argv

        self.assertFalse(payload["ok"])
        self.assertIn("error", payload)
        self.assertIn("next_step", payload)
        self.assertIn("Choose one of", payload["message"])

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
            self.assertTrue(report["checks"])
            self.assertIn("postgres_connection", {item["name"] for item in report["checks"]})
            self.assertIn("embedding_dependency", {item["name"] for item in report["checks"]})

            (home / "fusion-memory.sqlite3").write_text("seed", encoding="utf-8")
            backup = backup_data(home)
            self.assertTrue(backup["ok"])
            self.assertGreaterEqual(len(backup["files"]), 2)

            plan = upgrade(home, dry_run=True)
            self.assertTrue(plan["ok"])
            self.assertTrue(plan["dry_run"])
            self.assertIn("command", plan)

    def test_local_test_init_is_explicit_fallback_not_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            result = init_home(home, port=0, local_test=True)
            config = load_config(home)

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "local_test")
        self.assertEqual(config["storage_backend"], "sqlite")
        self.assertEqual(config["embedding"]["provider"], "deterministic")
        self.assertEqual(config["reranker"]["provider"], "lexical")
        self.assertIn("not production", result["message"])

    def test_doctor_checks_postgres_pgvector_and_qwen_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init_home(home, port=0)

            report = doctor(home)

        names = {item["name"]: item for item in report["checks"]}
        self.assertIn("postgres_connection", names)
        self.assertIn("pgvector", names)
        self.assertIn("embedding_dependency", names)
        self.assertIn("embedding_model_readiness", names)
        self.assertIn("reranker_dependency", names)
        self.assertIn("reranker_model_readiness", names)
        if not report["ok"]:
            self.assertIn("Fix failed checks", report["next_step"])
        serialized = json.dumps(report)
        self.assertNotIn("Traceback", serialized)
        self.assertNotIn("fusion:fusion", serialized)

    def test_start_failure_maps_qwen_traceback_to_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            paths = product_paths(home)
            init_home(home, port=0)
            paths.log.write_text(
                "Traceback (most recent call last):\n"
                "ModuleNotFoundError: No module named 'sentence_transformers'\n"
                "RuntimeError: Qwen3EmbeddingClient requires optional ML dependencies\n",
                encoding="utf-8",
            )

            with patch("fusion_memory.product.subprocess.Popen") as popen:
                process = popen.return_value
                process.pid = 12345
                process.poll.return_value = 1
                result = start_service(home, wait_seconds=0.01)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "model_not_ready")
        self.assertIn("Qwen", result["message"])
        self.assertIn("fusion-memory doctor", result["message"])
        self.assertNotIn("Traceback", result["message"])
        self.assertNotIn("sentence_transformers", result["message"])

    def test_safe_product_error_maps_connection_failure_to_database_guidance(self) -> None:
        error = safe_product_error(ConnectionError("connection refused"))

        self.assertEqual(error["error"], "database_not_ready")
        self.assertIn("Postgres", error["message"])
        self.assertIn("fusion-memory doctor", error["next_step"])

    def test_safe_product_error_hides_traceback_details(self) -> None:
        error = safe_product_error(RuntimeError("Traceback (most recent call last): secret stack"))

        self.assertNotIn("Traceback", error["message"])
        self.assertNotIn("secret stack", error["message"])

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
            self.assertEqual(env["FUSION_MEMORY_EXTRACTOR_MODE"], "async")
            self.assertEqual(env["FUSION_MEMORY_EXTRACTOR_API_KEY"], "secret-value")
            self.assertEqual(env["FUSION_MEMORY_QUERY_INTENT_MODE"], "off")

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

    def test_status_redacts_postgres_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init_home(
                home,
                settings={
                    "db": "postgresql://fusion:secret@127.0.0.1:55433/fusion_memory",
                    "storage_backend": "postgres",
                },
            )

            status = service_status(home)

        self.assertEqual(status["db"], "postgresql://***:***@127.0.0.1:55433/fusion_memory")
        self.assertNotIn("fusion:secret", json.dumps(status))

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
