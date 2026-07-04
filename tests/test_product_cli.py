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
    default_local_embedding_model_path,
    default_local_reranker_model_path,
    doctor,
    init_home,
    install_readiness,
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
    def test_render_human_uses_safe_fallback_for_failed_payloads_without_checks(
        self,
    ) -> None:
        rendered = render_human(
            {
                "ok": False,
                "message": "Could not connect",
                "next_step": "Run fusion-memory doctor",
            }
        )

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
            sys.argv = [
                "fusion-memory",
                "install-agent",
                "--target",
                "all",
                "--dry-run",
                "--json",
            ]
            sys.stdout = StringIO()
            main()
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])

    def test_install_check_cli_json(self) -> None:
        from fusion_memory.cli import main
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            embedding = home / "models" / "Qwen3-Embedding-0.6B"
            reranker = home / "models" / "Qwen3-Reranker-0.6B"
            _write_ready_model_dir(embedding)
            _write_ready_model_dir(reranker)
            try:
                sys.argv = [
                    "fusion-memory",
                    "install-check",
                    "--home",
                    tmp,
                    "--force",
                    "--json",
                ]
                sys.stdout = StringIO()
                with (
                    patch(
                        "fusion_memory.product.default_local_embedding_model_path",
                        return_value=embedding,
                    ),
                    patch(
                        "fusion_memory.product.default_local_reranker_model_path",
                        return_value=reranker,
                    ),
                    patch(
                        "fusion_memory.product._qwen_dependency_available",
                        return_value=True,
                    ),
                    patch(
                        "fusion_memory.product._qwen_runtime_smoke",
                        return_value={"ok": True, "message": "ready"},
                    ),
                    patch(
                        "fusion_memory.product._postgres_readiness",
                        return_value={
                            "connection": {
                                "name": "postgres_connection",
                                "ok": True,
                                "detail": "Postgres is reachable.",
                            },
                            "pgvector": {
                                "name": "pgvector",
                                "ok": True,
                                "detail": "pgvector is installed.",
                            },
                        },
                    ),
                ):
                    main()
                payload = json.loads(sys.stdout.getvalue())
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "local_full")
        self.assertEqual(payload["db"], str(Path(tmp) / "fusion-memory.sqlite3"))

    def test_install_check_cli_returns_nonzero_when_install_is_not_ready(self) -> None:
        from fusion_memory.cli import main
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        try:
            sys.argv = ["fusion-memory", "install-check", "--json"]
            sys.stdout = StringIO()
            with patch(
                "fusion_memory.cli.install_readiness",
                return_value={
                    "ok": False,
                    "mode": "not_ready",
                    "message": "model.safetensors is a Git LFS pointer",
                    "next_step": "Run git lfs pull.",
                },
            ):
                code = main()
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertIn("Git LFS pointer", payload["message"])

    def test_sync_haitun_history_cli_json_runs_once(self) -> None:
        from fusion_memory.cli import main
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        try:
            sys.argv = [
                "fusion-memory",
                "sync-haitun-history",
                "--session-id",
                "session-a",
                "--workspace",
                "/tmp/workspace",
                "--once",
                "--json",
            ]
            sys.stdout = StringIO()
            with patch(
                "fusion_memory.cli.sync_haitun_history_once",
                return_value={"ok": True, "added": 1, "skipped": 0},
            ):
                main()
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["added"], 1)

    def test_sync_dolphin_history_cli_json_alias_runs_once(self) -> None:
        from fusion_memory.cli import main
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        try:
            sys.argv = [
                "fusion-memory",
                "sync-dolphin-history",
                "--session-id",
                "session-a",
                "--workspace",
                "/tmp/workspace",
                "--once",
                "--json",
            ]
            sys.stdout = StringIO()
            with patch(
                "fusion_memory.cli.sync_haitun_history_once",
                return_value={"ok": True, "added": 1, "skipped": 0},
            ):
                main()
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["added"], 1)

    def test_cli_help_hides_dolphin_history_alias(self) -> None:
        from fusion_memory.cli import main
        import sys

        old_argv = sys.argv
        stdout = StringIO()
        try:
            sys.argv = ["fusion-memory", "--help"]
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as ctx:
                    main()
        finally:
            sys.argv = old_argv

        self.assertEqual(ctx.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("sync-haitun-history", help_text)
        self.assertNotIn("sync-dolphin-history", help_text)

    def test_install_agent_invalid_target_cli_json_is_beginner_safe(self) -> None:
        from fusion_memory.cli import main
        import sys

        old_argv = sys.argv
        stdout = StringIO()
        stderr = StringIO()
        try:
            sys.argv = [
                "fusion-memory",
                "install-agent",
                "--target",
                "bad-agent",
                "--json",
            ]
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
            with (
                redirect_stdout(stdout),
                patch(
                    "fusion_memory.cli.doctor",
                    side_effect=RuntimeError(
                        "Traceback (most recent call last): secret stack"
                    ),
                ),
            ):
                exit_code = main()
        finally:
            sys.argv = old_argv

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "unexpected_error")
        self.assertNotIn("Traceback", payload["message"])
        self.assertNotIn("secret stack", payload["message"])

    def test_install_agent_invalid_target_cli_json_includes_failure_schema(
        self,
    ) -> None:
        from fusion_memory.cli import main
        import sys

        old_argv = sys.argv
        stdout = StringIO()
        try:
            sys.argv = [
                "fusion-memory",
                "install-agent",
                "--target",
                "bad-agent",
                "--json",
            ]
            with redirect_stdout(stdout):
                main()
            payload = json.loads(stdout.getvalue())
        finally:
            sys.argv = old_argv

        self.assertFalse(payload["ok"])
        self.assertIn("error", payload)
        self.assertIn("next_step", payload)
        self.assertIn("Choose one of", payload["message"])

    def test_doctor_cli_json_preserves_failed_check_report_without_unexpected_error(self) -> None:
        from fusion_memory.cli import main
        import sys

        old_argv = sys.argv
        stdout = StringIO()
        report = {
            "ok": False,
            "checks": [{"name": "postgres_connection", "ok": False, "detail": "Postgres driver is missing."}],
            "next_step": "Install psycopg2-binary, then run fusion-memory doctor again.",
        }
        try:
            sys.argv = ["fusion-memory", "doctor", "--json"]
            with redirect_stdout(stdout), patch("fusion_memory.cli.doctor", return_value=report):
                main()
            payload = json.loads(stdout.getvalue())
        finally:
            sys.argv = old_argv

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["checks"][0]["name"], "postgres_connection")
        self.assertIn("psycopg2-binary", payload["next_step"])
        self.assertNotIn("error", payload)
        self.assertNotIn("message", payload)

    def test_init_doctor_backup_and_upgrade_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init = init_home(home, port=0)
            self.assertTrue(init["ok"])
            self.assertTrue((home / "config.json").exists())
            config = load_config(home)
            self.assertEqual(init["db"], str(home / "fusion-memory.sqlite3"))
            self.assertEqual(init["mode"], "local_full")
            self.assertEqual(config["mode"], "local_full")
            self.assertEqual(config["storage_backend"], "sqlite")
            self.assertEqual(config["embedding"]["provider"], "qwen")
            self.assertIn("Qwen3-Embedding-0.6B", config["embedding"]["model"])
            self.assertEqual(config["reranker"]["provider"], "qwen")
            self.assertIn("Qwen3-Reranker-0.6B", config["reranker"]["model"])
            self.assertEqual(config["extractor"]["provider"], "rule")
            self.assertEqual(config["query_intent"]["provider"], "off")
            self.assertTrue(hasattr(product, "default_product_settings"))
            defaults = product.default_product_settings(product_paths(home))
            self.assertEqual(defaults["mode"], "local_full")
            self.assertEqual(defaults["storage_backend"], "sqlite")

            report = doctor(home)
            self.assertTrue(report["checks"])
            self.assertIn("database_directory", {item["name"] for item in report["checks"]})
            self.assertNotIn("postgres_connection", {item["name"] for item in report["checks"]})
            self.assertIn(
                "embedding_dependency", {item["name"] for item in report["checks"]}
            )

            (home / "fusion-memory.sqlite3").write_text("seed", encoding="utf-8")
            backup = backup_data(home)
            self.assertTrue(backup["ok"])
            self.assertGreaterEqual(len(backup["files"]), 2)

            plan = upgrade(home, dry_run=True)
            self.assertTrue(plan["ok"])
            self.assertTrue(plan["dry_run"])
            self.assertIn("command", plan)

    def test_init_home_defaults_to_sqlite_and_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            result = init_home(home)
            config = load_config(home)

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "local_full")
        self.assertEqual(config["mode"], "local_full")
        self.assertEqual(config["storage_backend"], "sqlite")
        self.assertEqual(config["db"], str(home / "fusion-memory.sqlite3"))
        self.assertEqual(config["embedding"]["provider"], "qwen")
        self.assertEqual(
            config["embedding"]["model"], str(default_local_embedding_model_path())
        )
        self.assertEqual(config["reranker"]["provider"], "qwen")
        self.assertEqual(
            config["reranker"]["model"], str(default_local_reranker_model_path())
        )
        self.assertEqual(config["extractor"]["provider"], "rule")
        self.assertEqual(config["query_intent"]["provider"], "off")

    def test_default_qwen_models_are_repository_local_paths(self) -> None:
        repo_root = Path(product.__file__).resolve().parents[1]

        embedding_model = default_local_embedding_model_path()
        reranker_model = default_local_reranker_model_path()

        self.assertEqual(embedding_model, repo_root / "models" / "Qwen3-Embedding-0.6B")
        self.assertEqual(reranker_model, repo_root / "models" / "Qwen3-Reranker-0.6B")

    def test_install_readiness_reports_not_ready_when_models_or_deps_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch(
                    "fusion_memory.product.default_local_embedding_model_path",
                    return_value=home / "missing-embedding",
                ),
                patch(
                    "fusion_memory.product.default_local_reranker_model_path",
                    return_value=home / "missing-reranker",
                ),
                patch(
                    "fusion_memory.product._qwen_dependency_available",
                    return_value=False,
                ),
            ):
                result = install_readiness(home, force=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["mode"], "not_ready")
        self.assertFalse(result["compromised"])
        self.assertFalse((home / "config.json").exists())
        self.assertIn("Install the full runtime dependencies", result["message"])
        self.assertIn(".[postgres,qwen]", result["message"])

    def test_install_readiness_reports_not_ready_when_qwen_dependencies_are_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            embedding = home / "models" / "Qwen3-Embedding-0.6B"
            reranker = home / "models" / "Qwen3-Reranker-0.6B"
            _write_ready_model_dir(embedding)
            _write_ready_model_dir(reranker)
            with (
                patch(
                    "fusion_memory.product.default_local_embedding_model_path",
                    return_value=embedding,
                ),
                patch(
                    "fusion_memory.product.default_local_reranker_model_path",
                    return_value=reranker,
                ),
                patch(
                    "fusion_memory.product._qwen_dependency_available",
                    return_value=False,
                ),
            ):
                result = install_readiness(home, force=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["mode"], "not_ready")
        self.assertFalse(result["compromised"])
        self.assertFalse((home / "config.json").exists())
        self.assertIn("Qwen runtime dependencies", result["message"])
        self.assertIn(".[postgres,qwen]", result["next_step"])

    def test_install_readiness_uses_local_full_when_qwen_is_ready_without_postgres(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            embedding = home / "models" / "Qwen3-Embedding-0.6B"
            reranker = home / "models" / "Qwen3-Reranker-0.6B"
            _write_ready_model_dir(embedding)
            _write_ready_model_dir(reranker)
            postgres_report = {
                "connection": {
                    "name": "postgres_connection",
                    "ok": False,
                    "detail": "Postgres is not reachable.",
                },
                "pgvector": {
                    "name": "pgvector",
                    "ok": False,
                    "detail": "Postgres is not reachable.",
                },
            }
            with (
                patch(
                    "fusion_memory.product.default_local_embedding_model_path",
                    return_value=embedding,
                ),
                patch(
                    "fusion_memory.product.default_local_reranker_model_path",
                    return_value=reranker,
                ),
                patch(
                    "fusion_memory.product._qwen_dependency_available",
                    return_value=True,
                ),
                patch(
                    "fusion_memory.product._qwen_runtime_smoke",
                    return_value={"ok": True, "message": "ready"},
                ),
                patch(
                    "fusion_memory.product._postgres_readiness",
                    return_value=postgres_report,
                ),
            ):
                result = install_readiness(home, force=True)
            config = load_config(home)

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "local_full")
        self.assertFalse(result["compromised"])
        self.assertEqual(config["mode"], "local_full")
        self.assertEqual(config["storage_backend"], "sqlite")
        self.assertEqual(config["embedding"]["provider"], "qwen")
        self.assertEqual(config["reranker"]["provider"], "qwen")
        self.assertIn("SQLite", result["message"])
        self.assertIn("Qwen", result["message"])

    def test_install_readiness_rejects_git_lfs_pointer_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            embedding = home / "models" / "Qwen3-Embedding-0.6B"
            reranker = home / "models" / "Qwen3-Reranker-0.6B"
            _write_lfs_pointer_model_dir(embedding)
            _write_ready_model_dir(reranker)
            with (
                patch(
                    "fusion_memory.product.default_local_embedding_model_path",
                    return_value=embedding,
                ),
                patch(
                    "fusion_memory.product.default_local_reranker_model_path",
                    return_value=reranker,
                ),
                patch(
                    "fusion_memory.product._qwen_dependency_available",
                    return_value=True,
                ),
            ):
                result = install_readiness(home, force=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["mode"], "not_ready")
        self.assertFalse(result["compromised"])
        self.assertIn("Git LFS pointer", result["message"])
        self.assertIn("git lfs pull", result["next_step"])

    def test_install_readiness_rejects_tiny_stub_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            embedding = home / "models" / "Qwen3-Embedding-0.6B"
            reranker = home / "models" / "Qwen3-Reranker-0.6B"
            _write_stub_model_dir(embedding)
            _write_ready_model_dir(reranker)
            with (
                patch(
                    "fusion_memory.product.default_local_embedding_model_path",
                    return_value=embedding,
                ),
                patch(
                    "fusion_memory.product.default_local_reranker_model_path",
                    return_value=reranker,
                ),
                patch(
                    "fusion_memory.product._qwen_dependency_available",
                    return_value=True,
                ),
            ):
                result = install_readiness(home, force=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["mode"], "not_ready")
        self.assertIn("too small", result["message"])

    def test_install_readiness_falls_back_to_compromised_when_model_smoke_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            embedding = home / "models" / "Qwen3-Embedding-0.6B"
            reranker = home / "models" / "Qwen3-Reranker-0.6B"
            _write_ready_model_dir(embedding)
            _write_ready_model_dir(reranker)
            with (
                patch(
                    "fusion_memory.product.default_local_embedding_model_path",
                    return_value=embedding,
                ),
                patch(
                    "fusion_memory.product.default_local_reranker_model_path",
                    return_value=reranker,
                ),
                patch(
                    "fusion_memory.product._qwen_dependency_available",
                    return_value=True,
                ),
                patch(
                    "fusion_memory.product._qwen_runtime_smoke",
                    return_value={
                        "ok": False,
                        "message": "CPU lacks required runtime memory",
                    },
                ),
            ):
                result = install_readiness(home, force=True)
            config = load_config(home)

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "compromised")
        self.assertTrue(result["compromised"])
        self.assertEqual(config["mode"], "compromised")
        self.assertEqual(config["storage_backend"], "sqlite")
        self.assertEqual(config["embedding"]["provider"], "deterministic")
        self.assertEqual(config["reranker"]["provider"], "lexical")
        self.assertIn("hardware or runtime environment", result["message"])
        self.assertIn("DASHSCOPE_API_KEY", result["message"])
        self.assertIn("Aliyun", result["message"])
        self.assertIn("compromised", render_human(result))

    def test_install_readiness_compromised_result_includes_hardware_probe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            embedding = home / "models" / "Qwen3-Embedding-0.6B"
            reranker = home / "models" / "Qwen3-Reranker-0.6B"
            _write_ready_model_dir(embedding)
            _write_ready_model_dir(reranker)
            probe = {
                "platform": {"system": "Windows"},
                "torch": {"available": False},
            }
            with (
                patch(
                    "fusion_memory.product.default_local_embedding_model_path",
                    return_value=embedding,
                ),
                patch(
                    "fusion_memory.product.default_local_reranker_model_path",
                    return_value=reranker,
                ),
                patch(
                    "fusion_memory.product._qwen_dependency_available",
                    return_value=True,
                ),
                patch(
                    "fusion_memory.product._qwen_runtime_smoke",
                    return_value={"ok": False, "message": "runtime failed"},
                ),
                patch(
                    "fusion_memory.product._hardware_runtime_probe",
                    return_value=probe,
                ),
            ):
                result = install_readiness(home, force=True)

        self.assertTrue(result["compromised"])
        self.assertEqual(result["hardware_probe"], probe)
        self.assertEqual(result["runtime_smoke"]["hardware_probe"], probe)

    def test_init_home_local_test_fallback_uses_dependency_free_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            result = init_home(home, local_test=True)
            config = load_config(home)

        self.assertTrue(result["ok"])
        self.assertEqual(config["mode"], "local_test")
        self.assertEqual(config["storage_backend"], "sqlite")
        self.assertEqual(config["embedding"]["provider"], "deterministic")
        self.assertEqual(config["reranker"]["provider"], "lexical")

    def test_doctor_local_test_reports_model_dependency_and_readiness_checks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init_home(home, local_test=True)

            report = doctor(home)

        names = {item["name"]: item for item in report["checks"]}
        self.assertIn("embedding_dependency", names)
        self.assertIn("embedding_readiness", names)
        self.assertIn("reranker_dependency", names)
        self.assertIn("reranker_readiness", names)
        self.assertTrue(names["embedding_dependency"]["ok"])
        self.assertTrue(names["embedding_readiness"]["ok"])
        self.assertTrue(names["reranker_dependency"]["ok"])
        self.assertTrue(names["reranker_readiness"]["ok"])
        self.assertNotIn("embedding", names)
        self.assertNotIn("reranker", names)
        self.assertNotIn("Traceback", json.dumps(report))

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

    def test_doctor_checks_sqlite_and_qwen_readiness_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init_home(home, port=0)

            report = doctor(home)

        names = {item["name"]: item for item in report["checks"]}
        self.assertIn("database_directory", names)
        self.assertNotIn("postgres_connection", names)
        self.assertNotIn("pgvector", names)
        self.assertIn("embedding_dependency", names)
        self.assertIn("embedding_model_readiness", names)
        self.assertIn("reranker_dependency", names)
        self.assertIn("reranker_model_readiness", names)
        if not report["ok"]:
            self.assertIn("Fix failed checks", report["next_step"])
        serialized = json.dumps(report)
        self.assertNotIn("Traceback", serialized)
        self.assertNotIn("fusion:fusion", serialized)

    def test_doctor_reports_port_and_model_readiness_with_next_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init_home(home, port=0)
            report = doctor(home)

        names = {item["name"] for item in report["checks"]}
        self.assertIn("database_directory", names)
        self.assertNotIn("postgres_connection", names)
        self.assertNotIn("pgvector", names)
        self.assertIn("embedding_readiness", names)
        self.assertIn("reranker_readiness", names)
        self.assertIn("port", names)
        self.assertIn("next_step", report)
        self.assertNotIn("Traceback", json.dumps(report))

    def test_upgrade_dry_run_reports_backup_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init_home(home, local_test=True)
            plan = upgrade(home, dry_run=True)

        self.assertTrue(plan["ok"])
        self.assertTrue(plan["dry_run"])
        self.assertIn("backup", plan)
        self.assertIn("rollback", plan)

    def test_upgrade_failure_json_is_beginner_safe_without_raw_subprocess_output(
        self,
    ) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            init_home(home, local_test=True)
            raw_output = (
                "Traceback (most recent call last):\n"
                'File "/tmp/pip.py", line 1, in <module>\n'
                "RuntimeError: secret internal pip failure\n"
            )
            with patch(
                "fusion_memory.product.subprocess.run",
                return_value=subprocess.CompletedProcess(["pip"], 1, stdout=raw_output),
            ):
                result = upgrade(home, package="fusion-memory-test")

        serialized = json.dumps(result)
        self.assertFalse(result["ok"])
        self.assertIn("message", result)
        self.assertIn("next_step", result)
        self.assertNotIn("output", result)
        self.assertNotIn("Traceback", serialized)
        self.assertNotIn("secret internal pip failure", serialized)

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

            with (
                patch("fusion_memory.product._port_available", return_value=True),
                patch("fusion_memory.product.subprocess.Popen") as popen,
            ):
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

    def test_safe_product_error_maps_connection_failure_to_database_guidance(
        self,
    ) -> None:
        error = safe_product_error(ConnectionError("connection refused"))

        self.assertEqual(error["error"], "database_not_ready")
        self.assertIn("Postgres", error["message"])
        self.assertIn("fusion-memory doctor", error["next_step"])

    def test_safe_product_error_hides_traceback_details(self) -> None:
        error = safe_product_error(
            RuntimeError("Traceback (most recent call last): secret stack")
        )

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
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("builtins.input", lambda _prompt="": next(answers)),
            redirect_stdout(StringIO()),
        ):
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

            with patch.dict(
                os.environ, {"FUSION_MEMORY_MODEL_API_KEY": "secret-value"}
            ):
                env = _service_env(config)
            self.assertEqual(env["FUSION_MEMORY_EMBEDDING_PROVIDER"], "http")
            self.assertEqual(env["FUSION_MEMORY_EMBEDDING_API_KEY"], "secret-value")
            self.assertEqual(env["FUSION_MEMORY_RERANKER_PROVIDER"], "qwen")
            self.assertEqual(env["FUSION_MEMORY_EXTRACTOR_MODE"], "async")
            self.assertEqual(env["FUSION_MEMORY_EXTRACTOR_API_KEY"], "secret-value")
            self.assertEqual(env["FUSION_MEMORY_QUERY_INTENT_MODE"], "off")

    def test_interactive_defaults_to_sqlite_and_qwen(
        self,
    ) -> None:
        answers = iter(
            [
                "",  # host
                "18767",  # port
                "",  # default sqlite database
                "",  # default sqlite path
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

        self.assertTrue(result["db"].endswith("fusion-memory.sqlite3"))
        self.assertNotIn("fusion:fusion", json.dumps(result))
        rendered = render_human(result)
        self.assertIn("fusion-memory.sqlite3", rendered)
        self.assertNotIn("fusion:fusion", rendered)
        wizard_text = output.getvalue()
        self.assertIn("SQLite local file (recommended)", wizard_text)
        self.assertIn("Postgres / pgvector", wizard_text)
        self.assertIn("Qwen3 embedding (recommended)", wizard_text)
        self.assertIn("Qwen3 reranker (recommended)", wizard_text)
        self.assertNotIn("fusion:fusion", wizard_text)
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

        self.assertEqual(
            status["db"], "postgresql://***:***@127.0.0.1:55433/fusion_memory"
        )
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

    def test_start_service_tries_next_port_when_configured_port_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            busy_port = _free_port()
            init_home(
                home,
                port=busy_port,
                settings={
                    "db": str(home / "fusion-memory.sqlite3"),
                    "storage_backend": "sqlite",
                    "embedding": {"provider": "deterministic"},
                    "reranker": {"provider": "lexical"},
                },
            )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy_socket:
                busy_socket.bind(("127.0.0.1", busy_port))
                busy_socket.listen(1)
                fake_process = _FakeProcess(pid=4321)

                def fake_health(host: str, port: int, *, timeout: float = 1.0):
                    return {
                        "ok": port != busy_port,
                        "url": f"http://{host}:{port}/health",
                    }

                with (
                    patch(
                        "fusion_memory.product.subprocess.Popen",
                        return_value=fake_process,
                    ) as popen,
                    patch(
                        "fusion_memory.product.service_health", side_effect=fake_health
                    ),
                ):
                    started = start_service(home, wait_seconds=0.1)

            self.assertTrue(started["ok"], started)
            self.assertEqual(started["port"], busy_port + 1)
            self.assertEqual(started["url"], f"http://127.0.0.1:{busy_port + 1}")
            self.assertEqual(load_config(home)["port"], busy_port + 1)
            self.assertIn("--port", popen.call_args.args[0])
            self.assertIn(str(busy_port + 1), popen.call_args.args[0])

    def test_daemon_popen_kwargs_uses_unix_session_detach(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "service.log"
            with (
                log_path.open("ab") as handle,
                patch("fusion_memory.product.os.name", "posix"),
            ):
                kwargs = product._daemon_popen_kwargs(
                    handle, cwd=tmp, env={"A": "B"}
                )

        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], product.subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], product.subprocess.STDOUT)

    def test_daemon_popen_kwargs_uses_windows_no_window_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "service.log"
            with (
                log_path.open("ab") as handle,
                patch("fusion_memory.product.os.name", "nt"),
                patch.object(
                    product.subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0x00000200,
                    create=True,
                ),
                patch.object(
                    product.subprocess,
                    "DETACHED_PROCESS",
                    0x00000008,
                    create=True,
                ),
                patch.object(
                    product.subprocess,
                    "CREATE_NO_WINDOW",
                    0x08000000,
                    create=True,
                ),
            ):
                kwargs = product._daemon_popen_kwargs(
                    handle, cwd=tmp, env={"A": "B"}
                )

        self.assertEqual(kwargs["creationflags"] & 0x08000000, 0x08000000)
        self.assertEqual(kwargs["creationflags"] & 0x00000008, 0x00000008)
        self.assertEqual(kwargs["creationflags"] & 0x00000200, 0x00000200)

    def test_daemon_popen_kwargs_dedupes_windows_path_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "service.log"
            with (
                log_path.open("ab") as handle,
                patch("fusion_memory.product.os.name", "nt"),
            ):
                kwargs = product._daemon_popen_kwargs(
                    handle,
                    cwd=tmp,
                    env={
                        "Path": r"C:\Windows\System32",
                        "PATH": r"C:\msys64\ucrt64\bin",
                        "OTHER": "value",
                    },
                )

        path_keys = [key for key in kwargs["env"] if key.lower() == "path"]
        self.assertEqual(path_keys, ["Path"])
        self.assertEqual(kwargs["env"]["Path"], r"C:\Windows\System32")

    def test_daemon_python_executable_uses_pythonw_on_windows(self) -> None:
        with patch("fusion_memory.product.os.name", "nt"):
            executable = product._daemon_python_executable(
                r"C:\Users\alice\AppData\Local\Programs\Python\Python314\python.exe"
            )

        self.assertEqual(
            executable,
            r"C:\Users\alice\AppData\Local\Programs\Python\Python314\pythonw.exe",
        )

    def test_port_available_returns_false_when_socket_creation_is_blocked(self) -> None:
        with patch(
            "fusion_memory.product.socket.socket",
            side_effect=PermissionError("socket blocked"),
        ):
            self.assertFalse(product._port_available("127.0.0.1", 8700))

    def test_install_scripts_install_full_runtime_dependencies(self) -> None:
        root = Path(product.__file__).resolve().parents[1]

        install_sh = (root / "install.sh").read_text(encoding="utf-8")
        install_ps1 = (root / "install.ps1").read_text(encoding="utf-8")

        self.assertIn('-e "$SCRIPT_DIR"', install_sh)
        self.assertIn('-e "$SCRIPT_DIR[postgres,qwen]"', install_sh)
        self.assertIn("Optional Postgres/Qwen dependencies", install_sh)
        self.assertIn('-e "$ScriptDir"', install_ps1)
        self.assertIn('-e "$ScriptDir[postgres,qwen]"', install_ps1)
        self.assertIn("Optional Postgres/Qwen dependencies", install_ps1)
        self.assertIn("Normalize-ProcessPathEnvironment", install_ps1)
        self.assertLess(
            install_ps1.index("Normalize-ProcessPathEnvironment"),
            install_ps1.index("& $Python -m pip install --upgrade pip"),
        )
        self.assertNotIn("Start-Process", install_ps1)
        self.assertIn("Remove-Item Env:PATH", install_ps1)
        self.assertIn("install-check --force", install_ps1)
        self.assertIn("doctor", install_ps1)
        self.assertGreaterEqual(install_ps1.count("$LASTEXITCODE -ne 0"), 5)

    def test_qwen_extra_does_not_block_python_314(self) -> None:
        import tomllib

        root = Path(product.__file__).resolve().parents[1]
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        qwen_deps = pyproject["project"]["optional-dependencies"]["qwen"]

        self.assertTrue(qwen_deps)
        self.assertFalse(any('python_version < "3.14"' in dep for dep in qwen_deps))

    def test_qwen_dependency_available_requires_full_runtime_dependencies(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            return object() if name == "sentence_transformers" else None

        with patch("fusion_memory.product.find_spec", side_effect=fake_find_spec):
            self.assertFalse(product._qwen_dependency_available())


def _write_ready_model_dir(path: Path) -> None:
    path.mkdir(parents=True)
    with (path / "model.safetensors").open("wb") as handle:
        handle.truncate(product.MODEL_SAFETENSORS_MIN_BYTES)
    (path / "config.json").write_text("{}", encoding="utf-8")
    with (path / "tokenizer.json").open("wb") as handle:
        handle.truncate(product.TOKENIZER_JSON_MIN_BYTES)


def _write_stub_model_dir(path: Path) -> None:
    path.mkdir(parents=True)
    for filename in ("model.safetensors", "config.json", "tokenizer.json"):
        (path / filename).write_text("stub", encoding="utf-8")


def _write_lfs_pointer_model_dir(path: Path) -> None:
    path.mkdir(parents=True)
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 1191586416\n"
    )
    (path / "model.safetensors").write_text(pointer, encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")
    with (path / "tokenizer.json").open("wb") as handle:
        handle.truncate(product.TOKENIZER_JSON_MIN_BYTES)


def _free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError as exc:
        raise unittest.SkipTest("socket creation is not permitted") from exc


class _FakeProcess:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid

    def poll(self):
        return None


if __name__ == "__main__":
    unittest.main()
