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
from typing import Any
from urllib import error, request


APP_NAME = "Fusion Memory"
CONFIG_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_POSTGRES_DSN = "postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory"
DEFAULT_QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QWEN_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
_STARTED_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


@dataclass
class ProductPaths:
    home: Path
    config: Path
    db: Path
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


def product_paths(home: str | Path | None = None) -> ProductPaths:
    root = Path(home).expanduser() if home else _default_home()
    return ProductPaths(
        home=root,
        config=root / "config.json",
        db=root / "fusion-memory.sqlite3",
        log=root / "fusion-memory.log",
        pid=root / "fusion-memory.pid",
        backup_dir=root / "backups",
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
        config = _local_test_config(paths, host=host, port=port) if local_test else _default_config(paths, host=host, port=port)
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
        "message": "initialized (local test mode; not production)" if loaded.get("mode") == "local_test" else "initialized",
    }


def configure_interactive(
    home: str | Path | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    force: bool = False,
) -> dict[str, Any]:
    paths = product_paths(home)
    existing = load_config(home) if paths.config.exists() else _default_config(paths, host=host, port=port)

    print("Fusion Memory setup")
    print("Press Enter to accept the recommended default.")
    print()

    host = _ask("Service host", str(existing.get("host") or host))
    port = int(_ask("Service port", str(existing.get("port") or port)))

    storage_choice = _ask_choice(
        "Database",
        [
            ("postgres", "Postgres / pgvector (recommended)"),
            ("sqlite", "SQLite local file"),
        ],
        str(existing.get("storage_backend") or "postgres"),
    )
    if storage_choice == "postgres":
        default_dsn = str(existing.get("db") if str(existing.get("db", "")).startswith("postgres") else DEFAULT_POSTGRES_DSN)
        db_answer = _ask("Postgres DSN", _redact_dsn(default_dsn))
        db = default_dsn if db_answer == _redact_dsn(default_dsn) else db_answer
    else:
        db = _ask("SQLite database file", str(existing.get("db") or paths.db))

    embedding = _configure_model(
        "Embedding model",
        existing.get("embedding") if isinstance(existing.get("embedding"), dict) else {},
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
        existing.get("extractor") if isinstance(existing.get("extractor"), dict) else {},
        default_provider="rule",
    )
    query_intent = _configure_llm(
        "Query router model",
        existing.get("query_intent") if isinstance(existing.get("query_intent"), dict) else {},
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
    data.setdefault("db", DEFAULT_POSTGRES_DSN)
    data.setdefault("storage_backend", "postgres")
    data.setdefault("log", str(paths.log))
    data.setdefault("embedding", {"provider": "qwen", "model": DEFAULT_QWEN_EMBEDDING_MODEL})
    data.setdefault("reranker", {"provider": "qwen", "model": DEFAULT_QWEN_RERANKER_MODEL})
    data.setdefault("extractor", {"provider": "rule"})
    data.setdefault("query_intent", {"provider": "off"})
    return data


def doctor(home: str | Path | None = None) -> dict[str, Any]:
    paths = product_paths(home)
    checks: list[dict[str, Any]] = []

    checks.append(_check("python", sys.version_info >= (3, 11), f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))

    try:
        paths.home.mkdir(parents=True, exist_ok=True)
        probe = paths.home / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append(_check("home_writable", True, str(paths.home)))
    except OSError as exc:
        checks.append(_check("home_writable", False, _friendly_os_error(exc)))

    config = load_config(home)
    if str(config.get("storage_backend")) == "postgres":
        db = str(config.get("db", ""))
        checks.append(_check("postgres_dsn", db.startswith("postgres"), _redact_dsn(db) if db.startswith("postgres") else "missing Postgres DSN"))
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
        checks.append(_check("service", True, f"http://{config['host']}:{config['port']}"))
    else:
        checks.append(
            _check(
                "service",
                available,
                "ready to start" if available else f"port {config['port']} is already in use",
            )
        )
    checks.append(
        _check(
            "port",
            available or bool(health["ok"]),
            "service responding" if health["ok"] else ("available" if available else "already in use"),
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


def start_service(home: str | Path | None = None, *, wait_seconds: float = 10.0) -> dict[str, Any]:
    paths = product_paths(home)
    init_home(home)
    config = load_config(home)
    health = service_health(config["host"], int(config["port"]))
    if health["ok"]:
        return {"ok": True, "already_running": True, "url": _base_url(config), "pid": _read_pid(paths.pid)}

    if not _port_available(config["host"], int(config["port"])):
        return {
            "ok": False,
            "error": "port_in_use",
            "message": f"Port {config['port']} is already in use. Change the port in {paths.config}.",
        }

    log_handle = paths.log.open("ab")
    project_root = _local_project_root()
    cmd = [
        sys.executable,
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
    kwargs: dict[str, Any] = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "cwd": project_root or str(paths.home),
        "env": _service_env(config),
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(cmd, **kwargs)
    paths.pid.write_text(str(process.pid), encoding="utf-8")
    _STARTED_PROCESSES[process.pid] = process
    log_handle.close()

    deadline = time.time() + wait_seconds
    last_health: dict[str, Any] = {"ok": False}
    while time.time() < deadline:
        last_health = service_health(config["host"], int(config["port"]))
        if last_health["ok"]:
            return {"ok": True, "url": _base_url(config), "pid": process.pid, "log": str(paths.log)}
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

    return {
        "ok": False,
        "error": "startup_timeout",
        "message": f"{APP_NAME} is still starting. Run fusion-memory status or check {paths.log}.",
        "pid": process.pid,
        "log": str(paths.log),
        "health": last_health,
    }


def stop_service(home: str | Path | None = None, *, wait_seconds: float = 5.0) -> dict[str, Any]:
    paths = product_paths(home)
    config = load_config(home)
    pid = _read_pid(paths.pid)
    if pid is None:
        return {"ok": True, "already_stopped": True, "url": _base_url(config)}
    process = _STARTED_PROCESSES.get(pid)
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _STARTED_PROCESSES.pop(pid, None)
        paths.pid.unlink(missing_ok=True)
        return {"ok": True, "already_stopped": True, "url": _base_url(config)}
    except OSError as exc:
        return {"ok": False, "error": "stop_failed", "message": _friendly_os_error(exc), "pid": pid}

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
            return {"ok": False, "error": "stop_timeout", "message": _friendly_os_error(exc), "pid": pid}
    return {"ok": False, "error": "stop_timeout", "message": f"Service did not stop within {wait_seconds:.1f}s.", "pid": pid}


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


def upgrade(home: str | Path | None = None, *, package: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    paths = product_paths(home)
    init_home(home)
    target = package or _local_project_root() or "fusion-memory"
    command = [sys.executable, "-m", "pip", "install", "--upgrade", str(target)]
    backup_plan = {"required": True, "directory": str(paths.backup_dir)}
    rollback = {"available": True, "step": "Restore the latest backup from the backups directory."}
    if dry_run:
        return {"ok": True, "dry_run": True, "backup": backup_plan, "rollback": rollback, "command": command}
    backup = backup_data(home)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {
        "ok": completed.returncode == 0,
        "backup": backup,
        "rollback": rollback,
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout[-4000:],
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


def service_health(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, timeout: float = 1.0) -> dict[str, Any]:
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
        return "\n".join(
            [
                f"{APP_NAME}: OK",
                f"- Home: {result['home']}",
                f"- Config: {result['config']}",
                f"- Database: {_redact_dsn(str(result['db']))}",
            ]
        )
    if result.get("ok"):
        if result.get("url"):
            state = result.get("message") or ("running" if result.get("running") else "ready")
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
    if isinstance(exc, ConnectionError) or "connection refused" in message or "could not connect" in message:
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
    if "transformers" in message or "sentence_transformers" in message or "model" in message:
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


def _default_config(paths: ProductPaths, *, host: str, port: int) -> dict[str, Any]:
    return default_product_settings(paths) | {
        "host": host,
        "port": port,
    }


def default_product_settings(paths: ProductPaths) -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "db": DEFAULT_POSTGRES_DSN,
        "storage_backend": "postgres",
        "log": str(paths.log),
        "embedding": {"provider": "qwen", "model": DEFAULT_QWEN_EMBEDDING_MODEL},
        "reranker": {"provider": "qwen", "model": DEFAULT_QWEN_RERANKER_MODEL},
        "extractor": {"provider": "rule"},
        "query_intent": {"provider": "off"},
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
    provider = _ask_choice(title, choices, str(existing.get("provider") or default_provider))
    config: dict[str, Any] = {"provider": provider}
    if provider == "qwen":
        default_model = str(existing.get("model") or _default_qwen_model(title))
        config["model"] = _ask(f"{title} local model path/name", default_model)
        device = _ask(f"{title} device", str(existing.get("device") or "auto"))
        if device and device != "auto":
            config["device"] = device
    elif provider == "http":
        config["endpoint"] = _ask(f"{title} API endpoint", str(existing.get("endpoint") or ""))
        config["model"] = _ask(f"{title} API model", str(existing.get("model") or ""))
        config["api_key_env"] = _ask(f"{title} API key env var", str(existing.get("api_key_env") or "FUSION_MEMORY_MODEL_API_KEY"))
    return config


def _default_qwen_model(title: str) -> str:
    if "Reranker" in title:
        return DEFAULT_QWEN_RERANKER_MODEL
    return DEFAULT_QWEN_EMBEDDING_MODEL


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
            ("rule", "Built-in rules (recommended)" if not allow_off else "Built-in rules"),
            ("api", "OpenAI-compatible API"),
        ]
    )
    provider = _ask_choice(title, choices, str(existing.get("provider") or default_provider))
    config: dict[str, Any] = {"provider": provider}
    if provider == "api":
        config["base_url"] = _ask(f"{title} API base URL", str(existing.get("base_url") or ""))
        config["model"] = _ask(f"{title} API model", str(existing.get("model") or ""))
        config["api_key_env"] = _ask(f"{title} API key env var", str(existing.get("api_key_env") or "FUSION_MEMORY_MODEL_API_KEY"))
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
        print("Please choose one of: " + ", ".join(key for key, _description in choices))


def _service_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["FUSION_MEMORY_DB"] = str(config.get("db") or "")
    env["FUSION_MEMORY_STORAGE_BACKEND"] = str(config.get("storage_backend") or "sqlite")
    _apply_embedding_env(env, config.get("embedding") if isinstance(config.get("embedding"), dict) else {})
    _apply_reranker_env(env, config.get("reranker") if isinstance(config.get("reranker"), dict) else {})
    _apply_extractor_env(env, config.get("extractor") if isinstance(config.get("extractor"), dict) else {})
    _apply_query_intent_env(env, config.get("query_intent") if isinstance(config.get("query_intent"), dict) else {})
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
    _copy_secret_env(env, "FUSION_MEMORY_QUERY_INTENT_API_KEY", config.get("api_key_env"))


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


def _retrieval_model_checks(label: str, provider: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
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
        return {"connection": missing, "pgvector": _check("pgvector", False, "Postgres is not ready yet.")}
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError:
        return {
            "connection": _check("postgres_connection", False, "Postgres driver is missing. Install psycopg2-binary."),
            "pgvector": _check("pgvector", False, "Postgres driver is missing."),
        }
    try:
        conn = psycopg2.connect(dsn, connect_timeout=2)
    except Exception:
        return {
            "connection": _check("postgres_connection", False, "Postgres is not reachable. Start Postgres or check the DSN."),
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
            "pgvector": _check("pgvector", pgvector_ok, "pgvector is installed." if pgvector_ok else "pgvector extension is not installed."),
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _qwen_dependency_check(label: str) -> dict[str, Any]:
    ok = find_spec("sentence_transformers") is not None
    name = f"{label}_dependency"
    detail = "Qwen ML dependencies are installed." if ok else "Qwen ML dependencies are missing. Install the qwen extra or use --local-test for a temporary local mode."
    return _check(name, ok, detail)


def _qwen_model_readiness_check(label: str, model: str, dependency_ok: bool) -> dict[str, Any]:
    name = f"{label}_model_readiness"
    if not model:
        return _check(name, False, "Qwen model is not configured.")
    if _looks_like_path(model):
        exists = Path(model).expanduser().exists()
        return _check(name, exists and dependency_ok, "Qwen model path is ready." if exists and dependency_ok else "Qwen model path or dependencies are not ready.")
    return _check(
        name,
        dependency_ok,
        "Qwen model can be loaded by the configured model id." if dependency_ok else "Qwen model cannot load until ML dependencies are installed.",
    )


def _looks_like_path(value: str) -> bool:
    return value.startswith(("~", "/", ".")) or ":\\" in value or "\\" in value


def _startup_failure_result(paths: ProductPaths, pid: int, *, fallback: dict[str, Any]) -> dict[str, Any]:
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
    if "sentence_transformers" in lower or "qwen3embeddingclient requires optional ml dependencies" in lower or "qwen3reranker requires optional ml dependencies" in lower:
        return {
            "error": "model_not_ready",
            "message": "Qwen models are not ready. Run fusion-memory doctor, install the Qwen dependencies, or initialize with --local-test for a temporary local mode.",
        }
    if "connection refused" in lower or "could not connect" in lower or "postgres" in lower and "not reachable" in lower:
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
        return Path(os.getenv("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "FusionMemory"
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / "FusionMemory"
    return Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "fusion-memory"


def _base_url(config: dict[str, Any]) -> str:
    return f"http://{config['host']}:{config['port']}"


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def _doctor_next_step(*, ok: bool, service_running: bool) -> str:
    if not ok:
        return "Fix failed checks, then run fusion-memory doctor again. For a temporary local test mode, run fusion-memory init --local-test --force."
    return "fusion-memory status" if service_running else "fusion-memory start"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _friendly_os_error(exc: OSError) -> str:
    return str(exc) or exc.__class__.__name__


def _local_project_root() -> str | None:
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists():
        return str(root)
    return None
