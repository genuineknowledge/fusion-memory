from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request


APP_NAME = "Fusion Memory"
CONFIG_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8700
PORT_FALLBACK_ATTEMPTS = 20
DEFAULT_SERVICE_START_WAIT_SECONDS = 120.0
DEFAULT_POSTGRES_DSN = "postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory"
DEFAULT_QWEN_EMBEDDING_MODEL_NAME = "Qwen3-Embedding-0.6B"
DEFAULT_QWEN_RERANKER_MODEL_NAME = "Qwen3-Reranker-0.6B"
MODEL_SAFETENSORS_MIN_BYTES = 100 * 1024 * 1024
TOKENIZER_JSON_MIN_BYTES = 100 * 1024
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_STARTED_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


@dataclass
class ProductPaths:
    home: Path
    config: Path
    db: Path
    models: Path
    log: Path
    pid: Path
    backup_dir: Path


def runtime_status_payload(*, storage_backend: str = "sqlite") -> dict[str, Any]:
    return {
        "ok": True,
        "service": "running",
        "database": {"ok": True, "backend": storage_backend or "sqlite"},
        "models": {"ok": True},
        "version": CONFIG_VERSION,
    }


def _qwen_dependency_available() -> bool:
    return all(
        find_spec(name) is not None
        for name in ("sentence_transformers", "transformers", "torch")
    )


def product_paths(home: str | Path | None = None) -> ProductPaths:
    root = Path(home).expanduser() if home else _default_home()
    return ProductPaths(
        home=root,
        config=root / "config.json",
        db=root / "fusion-memory.sqlite3",
        models=root / "models",
        log=root / "fusion-memory.log",
        pid=root / "fusion-memory.pid",
        backup_dir=root / "backups",
    )


def default_local_embedding_model_path(home: str | Path | None = None) -> Path:
    return product_paths(home).models / DEFAULT_QWEN_EMBEDDING_MODEL_NAME


def default_local_reranker_model_path(home: str | Path | None = None) -> Path:
    return product_paths(home).models / DEFAULT_QWEN_RERANKER_MODEL_NAME


def install_readiness(
    home: str | Path | None = None, *, force: bool = False
) -> dict[str, Any]:
    """Initialize the product after install using local Qwen models when possible.

    Production mode is selected only when the Fusion Memory home-local
    embedding/reranker directories are present,
    optional ML dependencies are installed by the installer, and the current
    machine can load/run the two local vector models. The install scripts
    download the model directories from ModelScope before running this check.
    The local default uses SQLite plus Qwen. Missing files and missing Qwen
    runtime dependencies are reported as installation errors; compromised local
    mode is reserved for complete files and dependencies that fail model
    runtime/hardware smoke.
    """

    embedding_status = _repository_model_status(
        default_local_embedding_model_path(home),
        label="local Qwen3 embedding model",
    )
    reranker_status = _repository_model_status(
        default_local_reranker_model_path(home),
        label="local Qwen3 reranker model",
    )
    embedding_ready = bool(embedding_status["ok"])
    reranker_ready = bool(reranker_status["ok"])
    qwen_dependency_ready = _qwen_dependency_available()
    missing = []
    if not embedding_ready:
        missing.append(str(embedding_status["message"]))
    if not reranker_ready:
        missing.append(str(reranker_status["message"]))
    if not qwen_dependency_ready and not (embedding_ready and reranker_ready):
        missing.append("full Qwen runtime Python dependencies")
    if missing:
        model_messages = " ".join(missing)
        return {
            "ok": False,
            "mode": "not_ready",
            "compromised": False,
            "missing": missing,
            "model_readiness": {
                "embedding": embedding_status,
                "reranker": reranker_status,
            },
            "message": (
                "Fusion Memory install is not ready because "
                + ", ".join(missing)
                + " are missing. Rerun the Fusion Memory installer so it can "
                "download the Qwen model weights from ModelScope and install "
                "the full runtime dependencies."
            ),
            "next_step": (
                "Rerun the Fusion Memory installer to download the Qwen model weights from ModelScope, then rerun fusion-memory install-check --force."
                if "Git LFS pointer" in model_messages
                else "Rerun the Fusion Memory installer, then rerun fusion-memory install-check --force."
            ),
        }

    if not qwen_dependency_ready:
        hardware_probe = _hardware_runtime_probe()
        return {
            "ok": False,
            "mode": "not_ready",
            "compromised": False,
            "missing": ["full Qwen runtime Python dependencies"],
            "model_readiness": {
                "embedding": embedding_status,
                "reranker": reranker_status,
            },
            "hardware_probe": hardware_probe,
            "message": (
                "Fusion Memory install is not ready because the full Qwen runtime "
                "dependencies are not importable. Rerun the Fusion Memory installer "
                "before reporting installation success."
            ),
            "next_step": "Rerun the Fusion Memory installer, then rerun fusion-memory install-check --force.",
        }

    smoke = _qwen_runtime_smoke(home)
    if smoke["ok"]:
        result = init_home(home, force=force)
        result.update(
            {
                "mode": "local_full",
                "compromised": False,
                "message": (
                    "installed in local_full mode with SQLite and "
                    "home-local Qwen embedding and reranker models."
                ),
            }
        )
        return result

    hardware_probe = _hardware_runtime_probe()
    smoke = dict(smoke)
    smoke["hardware_probe"] = hardware_probe
    return _compromised_install_result(
        home,
        force=force,
        reason=(
            "the current hardware or runtime environment could not run the local "
            "Qwen embedding/reranker models: "
            + str(smoke.get("message") or "model runtime smoke failed")
        ),
        runtime_smoke=smoke,
        hardware_probe=hardware_probe,
    )


def init_home(
    home: str | Path | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    local_test: bool = False,
) -> dict[str, Any]:
    paths = product_paths(home)
    paths.home.mkdir(parents=True, exist_ok=True)
    paths.backup_dir.mkdir(parents=True, exist_ok=True)
    if not paths.config.exists() or force:
        config = (
            _local_test_config(paths, host=host, port=port)
            if local_test
            else _default_config(paths, host=host, port=port)
        )
        if settings:
            config.update(settings)
        _write_json(paths.config, config)
    loaded = load_config(home)
    return {
        "ok": True,
        "home": str(paths.home),
        "config": str(paths.config),
        "db": _redact_dsn(str(loaded["db"])),
        "log": str(paths.log),
        "mode": loaded.get("mode", "production"),
        "message": "initialized (local test mode; not production)"
        if loaded.get("mode") == "local_test"
        else "initialized",
    }


def configure_interactive(
    home: str | Path | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    force: bool = False,
) -> dict[str, Any]:
    paths = product_paths(home)
    existing = (
        load_config(home)
        if paths.config.exists()
        else _default_config(paths, host=host, port=port)
    )

    print("Fusion Memory setup")
    print("Press Enter to accept the recommended default.")
    print()

    host = _ask("Service host", str(existing.get("host") or host))
    port = int(_ask("Service port", str(existing.get("port") or port)))

    storage_choice = _ask_choice(
        "Database",
        [
            ("sqlite", "SQLite local file (recommended)"),
            ("postgres", "Postgres / pgvector"),
        ],
        str(existing.get("storage_backend") or "sqlite"),
    )
    if storage_choice == "postgres":
        default_dsn = str(
            existing.get("db")
            if str(existing.get("db", "")).startswith("postgres")
            else DEFAULT_POSTGRES_DSN
        )
        db_answer = _ask("Postgres DSN", _redact_dsn(default_dsn))
        db = default_dsn if db_answer == _redact_dsn(default_dsn) else db_answer
    else:
        db = _ask("SQLite database file", str(existing.get("db") or paths.db))

    embedding = _configure_model(
        "Embedding model",
        existing.get("embedding")
        if isinstance(existing.get("embedding"), dict)
        else {},
        default_provider="qwen",
        choices=[
            ("qwen", "Qwen3 embedding (recommended)"),
            ("deterministic", "Built-in lightweight embedding"),
            ("http", "API embedding service"),
        ],
    )
    reranker = _configure_model(
        "Reranker model",
        existing.get("reranker") if isinstance(existing.get("reranker"), dict) else {},
        default_provider="qwen",
        choices=[
            ("qwen", "Qwen3 reranker (recommended)"),
            ("lexical", "Built-in lexical reranker"),
            ("http", "API reranker service"),
        ],
    )
    extractor = _configure_llm(
        "Extractor/router model",
        existing.get("extractor")
        if isinstance(existing.get("extractor"), dict)
        else {},
        default_provider="rule",
    )
    query_intent = _configure_llm(
        "Query router model",
        existing.get("query_intent")
        if isinstance(existing.get("query_intent"), dict)
        else {},
        default_provider="off",
        allow_off=True,
    )

    config = _default_config(paths, host=host, port=port)
    config.update(
        {
            "host": host,
            "port": port,
            "db": db,
            "storage_backend": storage_choice,
            "embedding": embedding,
            "reranker": reranker,
            "extractor": extractor,
            "query_intent": query_intent,
        }
    )
    paths.home.mkdir(parents=True, exist_ok=True)
    paths.backup_dir.mkdir(parents=True, exist_ok=True)
    if paths.config.exists() and not force:
        backup_data(home)
    _write_json(paths.config, config)
    result = load_config(home)
    return {
        "ok": True,
        "home": str(paths.home),
        "config": str(paths.config),
        "db": _redact_dsn(str(result["db"])),
        "log": str(paths.log),
        "message": "configured",
        "providers": {
            "database": result.get("storage_backend"),
            "embedding": result.get("embedding", {}).get("provider"),
            "reranker": result.get("reranker", {}).get("provider"),
            "extractor": result.get("extractor", {}).get("provider"),
            "query_intent": result.get("query_intent", {}).get("provider"),
        },
    }


def load_config(home: str | Path | None = None) -> dict[str, Any]:
    paths = product_paths(home)
    if not paths.config.exists():
        init_home(home)
    data = json.loads(paths.config.read_text(encoding="utf-8"))
    data.setdefault("host", DEFAULT_HOST)
    data.setdefault("port", DEFAULT_PORT)
    data.setdefault("db", str(paths.db))
    data.setdefault("storage_backend", "sqlite")
    data.setdefault(
        "mode",
        "production" if data.get("storage_backend") == "postgres" else "local_full",
    )
    data.setdefault("log", str(paths.log))
    data.setdefault(
        "embedding", {"provider": "qwen", "model": str(default_local_embedding_model_path(paths.home))}
    )
    data.setdefault(
        "reranker", {"provider": "qwen", "model": str(default_local_reranker_model_path(paths.home))}
    )
    data.setdefault("extractor", {"provider": "rule"})
    data.setdefault("query_intent", {"provider": "off"})
    return data


def doctor(home: str | Path | None = None) -> dict[str, Any]:
    paths = product_paths(home)
    checks: list[dict[str, Any]] = []

    checks.append(
        _check(
            "python",
            sys.version_info >= (3, 11),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    try:
        paths.home.mkdir(parents=True, exist_ok=True)
        probe = paths.home / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append(_check("home_writable", True, str(paths.home)))
    except OSError as exc:
        checks.append(_check("home_writable", False, _friendly_os_error(exc)))

    config = load_config(home)
    if bool(config.get("compromised")) or str(config.get("mode")) == "compromised":
        checks.append(
            _check(
                "compromised_mode",
                True,
                "Running with built-in lightweight retrieval; provide DASHSCOPE_API_KEY for the recommended Aliyun DashScope API path or restore local Qwen models for full quality.",
            )
        )
    if str(config.get("storage_backend")) == "postgres":
        db = str(config.get("db", ""))
        checks.append(
            _check(
                "postgres_dsn",
                db.startswith("postgres"),
                _redact_dsn(db)
                if db.startswith("postgres")
                else "missing Postgres DSN",
            )
        )
        postgres = _postgres_readiness(db)
        checks.append(postgres["connection"])
        checks.append(postgres["pgvector"])
    else:
        db_path = Path(str(config["db"])).expanduser()
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            checks.append(_check("database_directory", True, str(db_path.parent)))
        except OSError as exc:
            checks.append(_check("database_directory", False, _friendly_os_error(exc)))

    checks.extend(_model_checks(config))

    health = service_health(config["host"], int(config["port"]))
    available = _port_available(config["host"], int(config["port"]))
    if health["ok"]:
        checks.append(
            _check("service", True, f"http://{config['host']}:{config['port']}")
        )
    else:
        checks.append(
            _check(
                "service",
                available,
                "ready to start"
                if available
                else f"port {config['port']} is already in use",
            )
        )
    checks.append(
        _check(
            "port",
            available or bool(health["ok"]),
            "service responding"
            if health["ok"]
            else ("available" if available else "already in use"),
        )
    )

    ok = all(item["ok"] for item in checks)
    return {
        "ok": ok,
        "home": str(paths.home),
        "config": str(paths.config),
        "checks": checks,
        "next_step": _doctor_next_step(ok=ok, service_running=bool(health["ok"])),
    }


def start_service(
    home: str | Path | None = None, *, wait_seconds: float = DEFAULT_SERVICE_START_WAIT_SECONDS
) -> dict[str, Any]:
    paths = product_paths(home)
    init_home(home)
    config = load_config(home)
    health = service_health(config["host"], int(config["port"]))
    if health["ok"]:
        return {
            "ok": True,
            "already_running": True,
            "url": _base_url(config),
            "pid": _read_pid(paths.pid),
        }

    recorded_pid = _read_pid(paths.pid)
    if recorded_pid is not None:
        stopped = stop_service(home, wait_seconds=10.0)
        if not stopped.get("ok"):
            return {
                "ok": False,
                "error": "stale_service_stop_failed",
                "message": (
                    f"{APP_NAME} found a recorded service pid but could not stop it. "
                    "Run fusion-memory stop, then retry fusion-memory start."
                ),
                "pid": recorded_pid,
                "stop": stopped,
                "log": str(paths.log),
            }
        health = service_health(config["host"], int(config["port"]))
        if health["ok"]:
            return {
                "ok": True,
                "already_running": True,
                "recovered": True,
                "url": _base_url(config),
                "pid": _read_pid(paths.pid),
            }

    if not _port_available(config["host"], int(config["port"])):
        original_port = int(config["port"])
        fallback_port = _next_available_port(str(config["host"]), original_port + 1)
        if fallback_port is None:
            return {
                "ok": False,
                "error": "port_in_use",
                "message": f"Port {config['port']} is already in use. Change the port in {paths.config}.",
            }
        config["port"] = fallback_port
        _write_json(paths.config, config)

    log_handle = paths.log.open("ab")
    project_root = _local_project_root()
    cmd = [
        _daemon_python_executable(),
        "-m",
        "fusion_memory.server",
        "--host",
        str(config["host"]),
        "--port",
        str(config["port"]),
        "--db",
        str(config["db"]),
        "--storage-backend",
        str(config["storage_backend"]),
    ]
    kwargs = _daemon_popen_kwargs(
        log_handle,
        cwd=project_root or str(paths.home),
        env=_service_env(config),
    )
    process = subprocess.Popen(cmd, **kwargs)
    paths.pid.write_text(str(process.pid), encoding="utf-8")
    _STARTED_PROCESSES[process.pid] = process
    log_handle.close()

    deadline = time.time() + wait_seconds
    last_health: dict[str, Any] = {"ok": False}
    while time.time() < deadline:
        last_health = service_health(config["host"], int(config["port"]))
        if last_health["ok"]:
            return {
                "ok": True,
                "url": _base_url(config),
                "port": int(config["port"]),
                "pid": process.pid,
                "log": str(paths.log),
            }
        if process.poll() is not None:
            _STARTED_PROCESSES.pop(process.pid, None)
            return _startup_failure_result(
                paths,
                process.pid,
                fallback={
                    "ok": False,
                    "error": "service_exited",
                    "message": f"{APP_NAME} could not start. See {paths.log}.",
                    "pid": process.pid,
                    "log": str(paths.log),
                },
            )
        time.sleep(0.2)

    _terminate_process_tree(process)
    _STARTED_PROCESSES.pop(process.pid, None)
    paths.pid.unlink(missing_ok=True)
    return {
        "ok": False,
        "error": "startup_timeout",
        "message": (
            f"{APP_NAME} did not become ready within {wait_seconds:.1f}s and the "
            "startup process was stopped. Check the local log before retrying."
        ),
        "pid": process.pid,
        "log": str(paths.log),
        "health": last_health,
        "terminated": True,
    }


def stop_service(
    home: str | Path | None = None, *, wait_seconds: float = 5.0
) -> dict[str, Any]:
    paths = product_paths(home)
    config = load_config(home)
    pid = _read_pid(paths.pid)
    if pid is None:
        return {"ok": True, "already_stopped": True, "url": _base_url(config)}
    process = _STARTED_PROCESSES.get(pid)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _STARTED_PROCESSES.pop(pid, None)
        paths.pid.unlink(missing_ok=True)
        return {"ok": True, "already_stopped": True, "url": _base_url(config)}
    except OSError as exc:
        return {
            "ok": False,
            "error": "stop_failed",
            "message": _friendly_os_error(exc),
            "pid": pid,
        }

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            _STARTED_PROCESSES.pop(pid, None)
            paths.pid.unlink(missing_ok=True)
            return {"ok": True, "stopped": True, "pid": pid}
        if not _process_exists(pid):
            if process is not None:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            _STARTED_PROCESSES.pop(pid, None)
            paths.pid.unlink(missing_ok=True)
            return {"ok": True, "stopped": True, "pid": pid}
        time.sleep(0.2)
    if os.name != "nt":
        try:
            os.kill(pid, signal.SIGKILL)
            if process is not None:
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            _STARTED_PROCESSES.pop(pid, None)
            paths.pid.unlink(missing_ok=True)
            return {"ok": True, "stopped": True, "forced": True, "pid": pid}
        except OSError as exc:
            return {
                "ok": False,
                "error": "stop_timeout",
                "message": _friendly_os_error(exc),
                "pid": pid,
            }
    return {
        "ok": False,
        "error": "stop_timeout",
        "message": f"Service did not stop within {wait_seconds:.1f}s.",
        "pid": pid,
    }


def service_status(home: str | Path | None = None) -> dict[str, Any]:
    paths = product_paths(home)
    config = load_config(home)
    pid = _read_pid(paths.pid)
    health = service_health(config["host"], int(config["port"]))
    return {
        "ok": health["ok"],
        "running": health["ok"],
        "url": _base_url(config),
        "pid": pid,
        "home": str(paths.home),
        "db": _redact_dsn(str(config["db"])),
        "log": str(config["log"]),
        "message": "running" if health["ok"] else "not running",
    }


def upgrade(
    home: str | Path | None = None, *, package: str | None = None, dry_run: bool = False
) -> dict[str, Any]:
    paths = product_paths(home)
    init_home(home)
    target = package or _local_project_root() or "fusion-memory"
    command = [sys.executable, "-m", "pip", "install", "--upgrade", str(target)]
    backup_plan = {"required": True, "directory": str(paths.backup_dir)}
    rollback = {
        "available": True,
        "step": "Restore the latest backup from the backups directory.",
    }
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "backup": backup_plan,
            "rollback": rollback,
            "command": command,
        }
    backup = backup_data(home)
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        paths.log.parent.mkdir(parents=True, exist_ok=True)
        with paths.log.open("a", encoding="utf-8") as handle:
            handle.write("\n[fusion-memory upgrade]\n")
            handle.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                handle.write("\n")
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": "upgrade_failed",
            "message": "Upgrade did not finish. Your existing memory data was backed up and the previous version is still available.",
            "next_step": "Run fusion-memory doctor, then retry fusion-memory upgrade. If it still fails, share the local log with support.",
            "backup": backup,
            "rollback": rollback,
            "command": command,
            "returncode": completed.returncode,
            "log": str(paths.log),
        }
    return {
        "ok": True,
        "backup": backup,
        "rollback": rollback,
        "command": command,
        "returncode": completed.returncode,
        "message": "Upgrade finished.",
        "log": str(paths.log),
    }


def backup_data(home: str | Path | None = None) -> dict[str, Any]:
    paths = product_paths(home)
    paths.backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    copied: list[str] = []
    for src in (paths.config, paths.db):
        if src.exists():
            dst = paths.backup_dir / f"{src.name}.{stamp}.bak"
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return {"ok": True, "files": copied}


def service_health(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, timeout: float = 1.0
) -> dict[str, Any]:
    url = f"http://{host}:{port}/health"
    try:
        with request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"ok": bool(payload.get("ok")), "url": url}
    except (OSError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "url": url, "message": str(exc)}


def render_human(result: dict[str, Any]) -> str:
    if "checks" in result:
        lines = [f"{APP_NAME} doctor"]
        for item in result["checks"]:
            marker = "OK" if item["ok"] else "FAIL"
            lines.append(f"- {marker} {item['name']}: {item['detail']}")
        lines.append(f"Next: {result['next_step']}")
        return "\n".join(lines)
    if result.get("home") and result.get("config") and result.get("db"):
        lines = [
            f"{APP_NAME}: OK",
            f"- Home: {result['home']}",
            f"- Config: {result['config']}",
            f"- Database: {_redact_dsn(str(result['db']))}",
        ]
        if result.get("message"):
            lines.append(f"- Message: {result['message']}")
        if result.get("compromised"):
            lines.append("- Mode: compromised")
        return "\n".join(lines)
    if result.get("ok"):
        if result.get("url"):
            state = result.get("message") or (
                "running" if result.get("running") else "ready"
            )
            return f"{APP_NAME}: OK ({result['url']}, {state})"
        if result.get("files") is not None:
            return f"{APP_NAME}: backup OK ({len(result['files'])} file(s))"
        return f"{APP_NAME}: OK"
    message = result.get("message") or result.get("error") or "failed"
    next_step = result.get("next_step")
    lines = [f"{APP_NAME}: {message}"]
    if next_step:
        lines.append(f"Next: {next_step}")
    return "\n".join(lines)


def safe_product_error(exc: BaseException) -> dict[str, str]:
    message = str(exc).lower()
    if (
        isinstance(exc, ConnectionError)
        or "connection refused" in message
        or "could not connect" in message
    ):
        return {
            "error": "database_not_ready",
            "message": "Postgres is not ready or cannot be reached.",
            "next_step": "Run fusion-memory doctor, then start Postgres or switch to local test mode.",
        }
    if "address already in use" in message or ("port" in message and "use" in message):
        return {
            "error": "port_in_use",
            "message": "The configured service port is already in use.",
            "next_step": "Run fusion-memory doctor and choose another port in the config file.",
        }
    if (
        "transformers" in message
        or "sentence_transformers" in message
        or "model" in message
    ):
        return {
            "error": "model_dependency_missing",
            "message": "The configured model dependency is not ready.",
            "next_step": "Run fusion-memory doctor to check Qwen embedding and reranker readiness.",
        }
    return {
        "error": "unexpected_error",
        "message": "Fusion Memory could not complete the request.",
        "next_step": "Run fusion-memory doctor and check the local log file.",
    }


def _compromised_install_result(
    home: str | Path | None,
    *,
    force: bool,
    reason: str,
    runtime_smoke: dict[str, Any],
    hardware_probe: dict[str, Any],
) -> dict[str, Any]:
    result = init_home(
        home, force=force, settings=compromised_local_settings(product_paths(home))
    )
    result.update(
        {
            "mode": "compromised",
            "compromised": True,
            "runtime_smoke": runtime_smoke,
            "hardware_probe": hardware_probe,
            "message": (
                "installed in compromised local mode because "
                + reason
                + ". Fusion Memory will use SQLite plus built-in lightweight embedding/reranker. "
                + _cpu_runtime_note(hardware_probe)
                + "To restore full memory quality, use the local Qwen models with the qwen "
                "runtime dependencies, or provide API model settings. Postgres/pgvector is "
                "optional for production storage. Recommended API option: Aliyun "
                "DashScope; set DASHSCOPE_API_KEY or map it through FUSION_MEMORY_MODEL_API_KEY "
                "before enabling API providers."
            ),
        }
    )
    return result


def _cpu_runtime_note(hardware_probe: dict[str, Any]) -> str:
    torch_probe = hardware_probe.get("torch")
    if not isinstance(torch_probe, dict):
        return ""
    if (
        bool(torch_probe.get("available"))
        and not bool(torch_probe.get("cuda_available"))
        and not bool(torch_probe.get("mps_available"))
    ):
        return (
            "CPU-only systems are supported, but local Qwen startup and inference "
            "can be slow; this fallback means the model smoke check did not finish "
            "successfully in the current runtime. "
        )
    return ""


def _postgres_readiness_ok(report: dict[str, dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in report.values())


def _default_config(paths: ProductPaths, *, host: str, port: int) -> dict[str, Any]:
    return default_product_settings(paths) | {
        "host": host,
        "port": port,
    }


def default_product_settings(paths: ProductPaths) -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "mode": "local_full",
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "db": str(paths.db),
        "storage_backend": "sqlite",
        "log": str(paths.log),
        "embedding": {"provider": "qwen", "model": str(default_local_embedding_model_path(paths.home))},
        "reranker": {"provider": "qwen", "model": str(default_local_reranker_model_path(paths.home))},
        "extractor": {"provider": "rule"},
        "query_intent": {"provider": "off"},
    }


def compromised_local_settings(paths: ProductPaths) -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "mode": "compromised",
        "compromised": True,
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "db": str(paths.db),
        "storage_backend": "sqlite",
        "log": str(paths.log),
        "embedding": {"provider": "deterministic"},
        "reranker": {"provider": "lexical"},
        "extractor": {"provider": "rule"},
        "query_intent": {"provider": "off"},
        "message": (
            "Compromised local mode uses built-in lightweight retrieval. "
            "Set DASHSCOPE_API_KEY for the recommended Aliyun DashScope API path "
            "or restore local Qwen models for full quality."
        ),
    }


def local_test_settings(paths: ProductPaths) -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "mode": "local_test",
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "db": str(paths.db),
        "storage_backend": "sqlite",
        "log": str(paths.log),
        "embedding": {"provider": "deterministic"},
        "reranker": {"provider": "lexical"},
        "extractor": {"provider": "rule"},
        "query_intent": {"provider": "off"},
    }


def _local_test_config(paths: ProductPaths, *, host: str, port: int) -> dict[str, Any]:
    return local_test_settings(paths) | {
        "host": host,
        "port": port,
    }


def _redact_dsn(value: str) -> str:
    if "@" not in value:
        return value
    prefix, suffix = value.rsplit("@", 1)
    scheme = prefix.split("://", 1)[0] if "://" in prefix else "postgresql"
    return f"{scheme}://***:***@{suffix}"


def _configure_model(
    title: str,
    existing: dict[str, Any],
    *,
    default_provider: str,
    choices: list[tuple[str, str]],
) -> dict[str, Any]:
    provider = _ask_choice(
        title, choices, str(existing.get("provider") or default_provider)
    )
    config: dict[str, Any] = {"provider": provider}
    if provider == "qwen":
        default_model = str(existing.get("model") or _default_qwen_model(title))
        config["model"] = _ask(f"{title} local model path/name", default_model)
        device = _ask(f"{title} device", str(existing.get("device") or "auto"))
        if device and device != "auto":
            config["device"] = device
    elif provider == "http":
        config["endpoint"] = _ask(
            f"{title} API endpoint", str(existing.get("endpoint") or "")
        )
        config["model"] = _ask(f"{title} API model", str(existing.get("model") or ""))
        config["api_key_env"] = _ask(
            f"{title} API key env var",
            str(existing.get("api_key_env") or "FUSION_MEMORY_MODEL_API_KEY"),
        )
    return config


def _default_qwen_model(title: str) -> str:
    if "Reranker" in title:
        return str(default_local_reranker_model_path())
    return str(default_local_embedding_model_path())


def _configure_llm(
    title: str,
    existing: dict[str, Any],
    *,
    default_provider: str,
    allow_off: bool = False,
) -> dict[str, Any]:
    choices = []
    if allow_off:
        choices.append(("off", "Disabled (recommended)"))
    choices.extend(
        [
            (
                "rule",
                "Built-in rules (recommended)" if not allow_off else "Built-in rules",
            ),
            ("api", "OpenAI-compatible API"),
        ]
    )
    provider = _ask_choice(
        title, choices, str(existing.get("provider") or default_provider)
    )
    config: dict[str, Any] = {"provider": provider}
    if provider == "api":
        config["base_url"] = _ask(
            f"{title} API base URL", str(existing.get("base_url") or "")
        )
        config["model"] = _ask(f"{title} API model", str(existing.get("model") or ""))
        config["api_key_env"] = _ask(
            f"{title} API key env var",
            str(existing.get("api_key_env") or "FUSION_MEMORY_MODEL_API_KEY"),
        )
    return config


def _ask(label: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _ask_choice(label: str, choices: list[tuple[str, str]], default: str) -> str:
    valid = {key for key, _description in choices}
    print(label + ":")
    for index, (key, description) in enumerate(choices, start=1):
        marker = " (default)" if key == default else ""
        print(f"  {index}. {description} [{key}]{marker}")
    while True:
        raw = input(f"Choose {label} [{default}]: ").strip()
        if not raw:
            return default
        if raw in valid:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][0]
        print(
            "Please choose one of: " + ", ".join(key for key, _description in choices)
        )


def _service_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["FUSION_MEMORY_DB"] = str(config.get("db") or "")
    env["FUSION_MEMORY_STORAGE_BACKEND"] = str(
        config.get("storage_backend") or "sqlite"
    )
    _apply_embedding_env(
        env,
        config.get("embedding") if isinstance(config.get("embedding"), dict) else {},
    )
    _apply_reranker_env(
        env, config.get("reranker") if isinstance(config.get("reranker"), dict) else {}
    )
    _apply_extractor_env(
        env,
        config.get("extractor") if isinstance(config.get("extractor"), dict) else {},
    )
    _apply_query_intent_env(
        env,
        config.get("query_intent")
        if isinstance(config.get("query_intent"), dict)
        else {},
    )
    return env


def _apply_embedding_env(env: dict[str, str], config: dict[str, Any]) -> None:
    provider = str(config.get("provider") or "deterministic")
    if provider == "deterministic":
        env.pop("FUSION_MEMORY_EMBEDDING_PROVIDER", None)
        return
    env["FUSION_MEMORY_EMBEDDING_PROVIDER"] = provider
    _set_if_present(env, "FUSION_MEMORY_EMBEDDING_MODEL", config.get("model"))
    _set_if_present(env, "FUSION_MEMORY_EMBEDDING_ENDPOINT", config.get("endpoint"))
    _set_if_present(env, "FUSION_MEMORY_EMBEDDING_DEVICE", config.get("device"))
    _copy_secret_env(env, "FUSION_MEMORY_EMBEDDING_API_KEY", config.get("api_key_env"))


def _apply_reranker_env(env: dict[str, str], config: dict[str, Any]) -> None:
    provider = str(config.get("provider") or "lexical")
    if provider == "lexical":
        env.pop("FUSION_MEMORY_RERANKER_PROVIDER", None)
        return
    env["FUSION_MEMORY_RERANKER_PROVIDER"] = provider
    _set_if_present(env, "FUSION_MEMORY_RERANKER_MODEL", config.get("model"))
    _set_if_present(env, "FUSION_MEMORY_RERANKER_ENDPOINT", config.get("endpoint"))
    _set_if_present(env, "FUSION_MEMORY_RERANKER_DEVICE", config.get("device"))
    _copy_secret_env(env, "FUSION_MEMORY_RERANKER_API_KEY", config.get("api_key_env"))


def _apply_extractor_env(env: dict[str, str], config: dict[str, Any]) -> None:
    provider = str(config.get("provider") or "rule")
    if provider != "api":
        env.pop("FUSION_MEMORY_EXTRACTOR_MODE", None)
        env.pop("FUSION_MEMORY_EXTRACTOR_BASE_URL", None)
        env.pop("FUSION_MEMORY_EXTRACTOR_ENDPOINT", None)
        return
    env["FUSION_MEMORY_EXTRACTOR_MODE"] = str(config.get("mode") or "async")
    _set_if_present(env, "FUSION_MEMORY_EXTRACTOR_BASE_URL", config.get("base_url"))
    _set_if_present(env, "FUSION_MEMORY_EXTRACTOR_MODEL", config.get("model"))
    _copy_secret_env(env, "FUSION_MEMORY_EXTRACTOR_API_KEY", config.get("api_key_env"))


def _apply_query_intent_env(env: dict[str, str], config: dict[str, Any]) -> None:
    provider = str(config.get("provider") or "off")
    if provider != "api":
        env["FUSION_MEMORY_QUERY_INTENT_MODE"] = "off"
        env.pop("FUSION_MEMORY_QUERY_INTENT_BASE_URL", None)
        env.pop("FUSION_MEMORY_QUERY_INTENT_ENDPOINT", None)
        return
    env["FUSION_MEMORY_QUERY_INTENT_MODE"] = str(config.get("mode") or "off")
    _set_if_present(env, "FUSION_MEMORY_QUERY_INTENT_BASE_URL", config.get("base_url"))
    _set_if_present(env, "FUSION_MEMORY_QUERY_INTENT_MODEL", config.get("model"))
    _copy_secret_env(
        env, "FUSION_MEMORY_QUERY_INTENT_API_KEY", config.get("api_key_env")
    )


def _set_if_present(env: dict[str, str], name: str, value: Any) -> None:
    if value is not None and str(value).strip():
        env[name] = str(value).strip()


def _copy_secret_env(env: dict[str, str], target: str, source_name: Any) -> None:
    if not source_name:
        return
    source = str(source_name).strip()
    if source and os.getenv(source):
        env[target] = os.environ[source]


def _model_checks(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for label in ("embedding", "reranker", "extractor", "query_intent"):
        raw = config.get(label) if isinstance(config.get(label), dict) else {}
        provider = str(raw.get("provider") or "")
        if label in {"embedding", "reranker"}:
            checks.extend(_retrieval_model_checks(label, provider, raw))
            continue
        if provider in {"", "rule", "off"}:
            checks.append(_check(label, True, provider or "default"))
            continue
        if provider in {"deterministic", "lexical"}:
            checks.append(_check(label, True, f"{provider} fallback is built in."))
            continue
        if provider in {"http", "api"}:
            checks.append(_http_model_check(label, provider, raw))
            continue
        checks.append(_check(label, False, f"unsupported provider: {provider}"))
    return checks


def _retrieval_model_checks(
    label: str, provider: str, raw: dict[str, Any]
) -> list[dict[str, Any]]:
    if provider in {"", "deterministic", "lexical"}:
        detail = f"{provider or 'default'} fallback is built in and requires no external dependency."
        return [
            _check(f"{label}_dependency", True, detail),
            _check(f"{label}_readiness", True, detail),
        ]
    if provider == "qwen":
        model = str(raw.get("model") or "")
        dependency = _qwen_dependency_check(label)
        readiness = _qwen_model_readiness_check(label, model, dependency["ok"])
        return [dependency, {**readiness, "name": f"{label}_readiness"}, readiness]
    if provider in {"http", "api"}:
        check = _http_model_check(label, provider, raw)
        return [
            {**check, "name": f"{label}_dependency"},
            {**check, "name": f"{label}_readiness"},
        ]
    detail = f"unsupported provider: {provider}"
    return [
        _check(f"{label}_dependency", False, detail),
        _check(f"{label}_readiness", False, detail),
    ]


def _http_model_check(label: str, provider: str, raw: dict[str, Any]) -> dict[str, Any]:
    endpoint = str(raw.get("endpoint") or raw.get("base_url") or "")
    env_name = str(raw.get("api_key_env") or "")
    secret_ok = not env_name or bool(os.getenv(env_name))
    return _check(
        label,
        bool(endpoint) and secret_ok,
        f"{provider} endpoint={'set' if endpoint else 'missing'}, key_env={env_name or 'none'}{' set' if secret_ok else ' missing'}",
    )


def _postgres_readiness(dsn: str) -> dict[str, dict[str, Any]]:
    if not dsn.startswith("postgres"):
        missing = _check("postgres_connection", False, "Postgres DSN is missing.")
        return {
            "connection": missing,
            "pgvector": _check("pgvector", False, "Postgres is not ready yet."),
        }
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError:
        return {
            "connection": _check(
                "postgres_connection",
                False,
                "Postgres driver is missing. Install psycopg2-binary.",
            ),
            "pgvector": _check("pgvector", False, "Postgres driver is missing."),
        }
    try:
        conn = psycopg2.connect(dsn, connect_timeout=2)
    except Exception:
        return {
            "connection": _check(
                "postgres_connection",
                False,
                "Postgres is not reachable. Start Postgres or check the DSN.",
            ),
            "pgvector": _check("pgvector", False, "Postgres is not reachable."),
        }
    try:
        with conn.cursor() as cursor:
            cursor.execute("select 1")
        pgvector_ok = False
        try:
            with conn.cursor() as cursor:
                cursor.execute("select 1 from pg_extension where extname = 'vector'")
                pgvector_ok = cursor.fetchone() is not None
        except Exception:
            pgvector_ok = False
        return {
            "connection": _check("postgres_connection", True, "Postgres is reachable."),
            "pgvector": _check(
                "pgvector",
                pgvector_ok,
                "pgvector is installed."
                if pgvector_ok
                else "pgvector extension is not installed.",
            ),
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _qwen_dependency_check(label: str) -> dict[str, Any]:
    ok = _qwen_dependency_available()
    name = f"{label}_dependency"
    detail = (
        "Qwen ML dependencies are installed."
        if ok
        else "Qwen ML dependencies are missing. Install the qwen extra, configure an API provider, or use compromised/local-test mode temporarily."
    )
    return _check(name, ok, detail)


def _qwen_model_readiness_check(
    label: str, model: str, dependency_ok: bool
) -> dict[str, Any]:
    name = f"{label}_model_readiness"
    if not model:
        return _check(name, False, "Qwen model is not configured.")
    if _looks_like_path(model):
        path = Path(model).expanduser()
        status = _repository_model_status(path, label=f"{label} Qwen model")
        exists = bool(status["ok"])
        return _check(
            name,
            exists and dependency_ok,
            "Fusion Memory home-local Qwen model path is ready."
            if exists and dependency_ok
            else f"Fusion Memory home-local Qwen model path or dependencies are not ready: {status['message']}",
        )
    return _check(
        name,
        False,
        "Qwen model must be a local path under the Fusion Memory models directory. Rerun the installer to download model weights from ModelScope.",
    )


def _qwen_runtime_smoke(home: str | Path | None = None) -> dict[str, Any]:
    try:
        from fusion_memory.core.embedding import Qwen3EmbeddingClient
        from fusion_memory.retrieval.reranker import Qwen3Reranker

        embedding = Qwen3EmbeddingClient(
            model=str(default_local_embedding_model_path(home)), batch_size=1
        )
        vector = embedding.embed_text("fusion memory install check")
        if not vector:
            return {
                "ok": False,
                "message": "Qwen embedding model returned an empty vector.",
            }
        reranker = Qwen3Reranker(
            model=str(default_local_reranker_model_path(home)), batch_size=1
        )
        scores = reranker.score("fusion memory", ["fusion memory install check"])
        if not scores:
            return {"ok": False, "message": "Qwen reranker model returned no scores."}
        return {
            "ok": True,
            "message": "Home-local Qwen models loaded and ran a minimal smoke test.",
        }
    except Exception as exc:
        return {"ok": False, "message": _friendly_runtime_smoke_message(exc)}


def _friendly_runtime_smoke_message(exc: BaseException) -> str:
    text = str(exc).strip()
    lowered = text.lower()
    if (
        "cannot reshape tensor" in lowered
        or "shape [1, 0" in lowered
        or "0 elements" in lowered
    ):
        return (
            "Qwen runtime smoke failed while loading or scoring the local "
            "embedding/reranker models."
        )
    if (
        "out of memory" in lowered
        or "cuda" in lowered
        or "mps" in lowered
        or "meta tensor" in lowered
    ):
        return "The local Qwen models could not run on this hardware/runtime."
    if (
        "sentence_transformers" in lowered
        or "transformers" in lowered
        or "torch" in lowered
    ):
        return (
            "The full Qwen runtime dependencies are not importable after installation."
        )
    return text[:240] if text else exc.__class__.__name__


def _hardware_runtime_probe() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "cpu": {"count": os.cpu_count()},
        "memory": {},
        "torch": {"available": False},
    }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(page_count, int):
            probe["memory"]["total_bytes"] = page_size * page_count
    except (AttributeError, OSError, ValueError):
        pass
    try:
        import torch  # type: ignore[import-not-found]

        cuda_available = bool(torch.cuda.is_available())
        probe["torch"] = {
            "available": True,
            "version": str(getattr(torch, "__version__", "")),
            "cuda_available": cuda_available,
            "cuda_device_count": int(torch.cuda.device_count())
            if cuda_available
            else 0,
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except Exception as exc:
        probe["torch"] = {"available": False, "error": exc.__class__.__name__}
    return probe


def _repository_model_ready(path: Path) -> bool:
    return bool(_repository_model_status(path)["ok"])


def _repository_model_status(
    path: Path, *, label: str = "local Qwen model"
) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_dir():
        return {
            "ok": False,
            "path": str(path),
            "message": f"{label} directory is missing",
        }
    checks = [
        _model_file_status(
            path / "model.safetensors",
            min_bytes=MODEL_SAFETENSORS_MIN_BYTES,
            parse_json=False,
        ),
        _model_file_status(path / "config.json", min_bytes=2, parse_json=True),
        _model_file_status(
            path / "tokenizer.json",
            min_bytes=TOKENIZER_JSON_MIN_BYTES,
            parse_json=False,
        ),
    ]
    failed = [check for check in checks if not check["ok"]]
    if failed:
        details = "; ".join(str(check["message"]) for check in failed)
        return {
            "ok": False,
            "path": str(path),
            "files": checks,
            "message": f"{label} is incomplete: {details}",
        }
    return {
        "ok": True,
        "path": str(path),
        "files": checks,
        "message": f"{label} is ready",
    }


def _model_file_status(
    path: Path, *, min_bytes: int, parse_json: bool
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "ok": False,
            "path": str(path),
            "message": f"{path.name} is missing",
        }
    if _is_lfs_pointer(path):
        return {
            "ok": False,
            "path": str(path),
            "message": f"{path.name} is a Git LFS pointer, not the real model file",
        }
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {
            "ok": False,
            "path": str(path),
            "message": f"{path.name} cannot be inspected: {_friendly_os_error(exc)}",
        }
    if size < min_bytes:
        return {
            "ok": False,
            "path": str(path),
            "size": size,
            "message": f"{path.name} is too small ({size} bytes)",
        }
    if parse_json:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "path": str(path),
                "size": size,
                "message": f"{path.name} is not valid JSON: {exc.__class__.__name__}",
            }
    return {"ok": True, "path": str(path), "size": size, "message": "ready"}


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(256)
    except OSError:
        return False
    return head.startswith(LFS_POINTER_PREFIX)


def _daemon_popen_kwargs(
    log_handle: Any, *, cwd: str, env: dict[str, str]
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "cwd": cwd,
        "env": _normalize_process_env(env),
    }
    if os.name == "nt":
        flags = 0
        for name in (
            "CREATE_NEW_PROCESS_GROUP",
            "DETACHED_PROCESS",
            "CREATE_NO_WINDOW",
        ):
            flags |= int(getattr(subprocess, name, 0))
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    pid = getattr(process, "pid", None)
    if pid is None:
        try:
            process.kill()
        except OSError:
            pass
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:
                pass


def _normalize_process_env(env: Mapping[str, str] | dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    seen: set[str] = set()
    for key, value in env.items():
        if value is None:
            continue
        out_key = "Path" if os.name == "nt" and key.lower() == "path" else str(key)
        dedupe_key = out_key.lower() if os.name == "nt" else out_key
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized[out_key] = str(value)
    return normalized


def _daemon_python_executable(executable: str | None = None) -> str:
    resolved = executable or sys.executable
    if os.name != "nt":
        return resolved
    lower = resolved.lower()
    if lower.endswith("pythonw.exe"):
        return resolved
    if lower.endswith("python.exe"):
        return resolved[: -len("python.exe")] + "pythonw.exe"
    return resolved


def _looks_like_path(value: str) -> bool:
    return value.startswith(("~", "/", ".")) or ":\\" in value or "\\" in value


def _startup_failure_result(
    paths: ProductPaths, pid: int, *, fallback: dict[str, Any]
) -> dict[str, Any]:
    log_tail = _read_log_tail(paths.log)
    classified = _classify_startup_failure(log_tail)
    if not classified:
        return fallback
    return {
        "ok": False,
        "error": classified["error"],
        "message": classified["message"],
        "pid": pid,
        "log": str(paths.log),
    }


def _read_log_tail(path: Path, *, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _classify_startup_failure(log_tail: str) -> dict[str, str] | None:
    lower = log_tail.lower()
    if (
        "sentence_transformers" in lower
        or "qwen3embeddingclient requires optional ml dependencies" in lower
        or "qwen3reranker requires optional ml dependencies" in lower
    ):
        return {
            "error": "model_not_ready",
            "message": "Qwen models are not ready. Run fusion-memory doctor, install the Qwen dependencies, or initialize with --local-test for a temporary local mode.",
        }
    if (
        "connection refused" in lower
        or "could not connect" in lower
        or "postgres" in lower
        and "not reachable" in lower
    ):
        return {
            "error": "database_not_ready",
            "message": "Postgres is not ready. Start Postgres, check the DSN, then run fusion-memory doctor.",
        }
    if "address already in use" in lower or "port" in lower and "in use" in lower:
        return {
            "error": "port_in_use",
            "message": "The Fusion Memory port is already in use. Change the configured port or stop the other service.",
        }
    return None


def _default_home() -> Path:
    env_home = os.getenv("FUSION_MEMORY_HOME")
    if env_home:
        return Path(env_home).expanduser()
    system = platform.system().lower()
    if system == "windows":
        return (
            Path(os.getenv("APPDATA", str(Path.home() / "AppData" / "Roaming")))
            / "FusionMemory"
        )
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / "FusionMemory"
    return (
        Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        / "fusion-memory"
    )


def _base_url(config: dict[str, Any]) -> str:
    return f"http://{config['host']}:{config['port']}"


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def _doctor_next_step(*, ok: bool, service_running: bool) -> str:
    if not ok:
        return "Fix failed checks, then run fusion-memory doctor again. For a temporary local test mode, run fusion-memory init --local-test --force."
    return "fusion-memory status" if service_running else "fusion-memory start"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def _next_available_port(
    host: str, start_port: int, *, attempts: int = PORT_FALLBACK_ATTEMPTS
) -> int | None:
    for port in range(start_port, start_port + attempts):
        if _port_available(host, port):
            return port
    return None


def _friendly_os_error(exc: OSError) -> str:
    return str(exc) or exc.__class__.__name__


def _local_project_root() -> str | None:
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists():
        return str(root)
    return None
