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
from pathlib import Path
from typing import Any
from urllib import error, request


APP_NAME = "Fusion Memory"
CONFIG_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
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
) -> dict[str, Any]:
    paths = product_paths(home)
    paths.home.mkdir(parents=True, exist_ok=True)
    paths.backup_dir.mkdir(parents=True, exist_ok=True)
    if not paths.config.exists() or force:
        config = _default_config(paths, host=host, port=port)
        if settings:
            config.update(settings)
        _write_json(paths.config, config)
    return {
        "ok": True,
        "home": str(paths.home),
        "config": str(paths.config),
        "db": str(paths.db),
        "log": str(paths.log),
        "message": "initialized",
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
            ("sqlite", "SQLite local file (recommended)"),
            ("postgres", "Postgres / pgvector"),
        ],
        str(existing.get("storage_backend") or "sqlite"),
    )
    if storage_choice == "postgres":
        db = _ask("Postgres DSN", str(existing.get("db") if str(existing.get("db", "")).startswith("postgres") else "postgresql://user:pass@127.0.0.1:5432/fusion_memory"))
    else:
        db = _ask("SQLite database file", str(existing.get("db") or paths.db))

    embedding = _configure_model(
        "Embedding model",
        existing.get("embedding") if isinstance(existing.get("embedding"), dict) else {},
        default_provider="deterministic",
        choices=[
            ("deterministic", "Built-in lightweight embedding (recommended)"),
            ("qwen", "Local Qwen3 embedding"),
            ("http", "API embedding service"),
        ],
    )
    reranker = _configure_model(
        "Reranker model",
        existing.get("reranker") if isinstance(existing.get("reranker"), dict) else {},
        default_provider="lexical",
        choices=[
            ("lexical", "Built-in lexical reranker (recommended)"),
            ("qwen", "Local Qwen3 reranker"),
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
        "db": str(result["db"]),
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
    data.setdefault("log", str(paths.log))
    data.setdefault("embedding", {"provider": "deterministic"})
    data.setdefault("reranker", {"provider": "lexical"})
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
    db_path = Path(str(config["db"])).expanduser()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checks.append(_check("database_directory", True, str(db_path.parent)))
    except OSError as exc:
        checks.append(_check("database_directory", False, _friendly_os_error(exc)))
    if str(config.get("storage_backend")) == "postgres":
        checks.append(_check("postgres_dsn", str(config.get("db", "")).startswith("postgres"), "configured" if str(config.get("db", "")).startswith("postgres") else "missing Postgres DSN"))

    checks.extend(_model_checks(config))

    health = service_health(config["host"], int(config["port"]))
    if health["ok"]:
        checks.append(_check("service", True, f"http://{config['host']}:{config['port']}"))
    else:
        available = _port_available(config["host"], int(config["port"]))
        checks.append(
            _check(
                "service",
                available,
                "ready to start" if available else f"port {config['port']} is already in use",
            )
        )

    ok = all(item["ok"] for item in checks)
    return {
        "ok": ok,
        "home": str(paths.home),
        "config": str(paths.config),
        "checks": checks,
        "next_step": "fusion-memory start" if ok and not health["ok"] else "fusion-memory status",
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
            return {
                "ok": False,
                "error": "service_exited",
                "message": f"{APP_NAME} could not start. See {paths.log}.",
                "pid": process.pid,
                "log": str(paths.log),
            }
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
        "db": str(config["db"]),
        "log": str(config["log"]),
        "message": "running" if health["ok"] else "not running",
    }


def upgrade(home: str | Path | None = None, *, package: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    paths = product_paths(home)
    init_home(home)
    backup = backup_data(home)
    target = package or _local_project_root() or "fusion-memory"
    command = [sys.executable, "-m", "pip", "install", "--upgrade", str(target)]
    if dry_run:
        return {"ok": True, "dry_run": True, "backup": backup, "command": command}
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {
        "ok": completed.returncode == 0,
        "backup": backup,
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
                f"- Database: {result['db']}",
            ]
        )
    if result.get("ok"):
        if result.get("url"):
            state = result.get("message") or ("running" if result.get("running") else "ready")
            return f"{APP_NAME}: OK ({result['url']}, {state})"
        if result.get("files") is not None:
            return f"{APP_NAME}: backup OK ({len(result['files'])} file(s))"
        return f"{APP_NAME}: OK"
    return f"{APP_NAME}: {result.get('message') or result.get('error') or 'failed'}"


def _default_config(paths: ProductPaths, *, host: str, port: int) -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "host": host,
        "port": port,
        "db": str(paths.db),
        "storage_backend": "sqlite",
        "log": str(paths.log),
        "embedding": {"provider": "deterministic"},
        "reranker": {"provider": "lexical"},
        "extractor": {"provider": "rule"},
        "query_intent": {"provider": "off"},
    }


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
        default_model = str(existing.get("model") or "")
        config["model"] = _ask(f"{title} local model path/name", default_model)
        device = _ask(f"{title} device", str(existing.get("device") or "auto"))
        if device and device != "auto":
            config["device"] = device
    elif provider == "http":
        config["endpoint"] = _ask(f"{title} API endpoint", str(existing.get("endpoint") or ""))
        config["model"] = _ask(f"{title} API model", str(existing.get("model") or ""))
        config["api_key_env"] = _ask(f"{title} API key env var", str(existing.get("api_key_env") or "FUSION_MEMORY_MODEL_API_KEY"))
    return config


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
        env.pop("FUSION_MEMORY_EXTRACTOR_BASE_URL", None)
        env.pop("FUSION_MEMORY_EXTRACTOR_ENDPOINT", None)
        return
    _set_if_present(env, "FUSION_MEMORY_EXTRACTOR_BASE_URL", config.get("base_url"))
    _set_if_present(env, "FUSION_MEMORY_EXTRACTOR_MODEL", config.get("model"))
    _copy_secret_env(env, "FUSION_MEMORY_EXTRACTOR_API_KEY", config.get("api_key_env"))


def _apply_query_intent_env(env: dict[str, str], config: dict[str, Any]) -> None:
    provider = str(config.get("provider") or "off")
    if provider != "api":
        env.pop("FUSION_MEMORY_QUERY_INTENT_BASE_URL", None)
        env.pop("FUSION_MEMORY_QUERY_INTENT_ENDPOINT", None)
        return
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
        if provider in {"", "deterministic", "lexical", "rule", "off"}:
            checks.append(_check(label, True, provider or "default"))
            continue
        if provider == "qwen":
            model = str(raw.get("model") or "")
            ok = bool(model)
            if model and ("/" in model or "\\" in model):
                ok = Path(model).expanduser().exists()
            checks.append(_check(label, ok, f"qwen model={model or 'missing'}"))
            continue
        if provider in {"http", "api"}:
            endpoint = str(raw.get("endpoint") or raw.get("base_url") or "")
            env_name = str(raw.get("api_key_env") or "")
            secret_ok = not env_name or bool(os.getenv(env_name))
            checks.append(
                _check(
                    label,
                    bool(endpoint) and secret_ok,
                    f"{provider} endpoint={'set' if endpoint else 'missing'}, key_env={env_name or 'none'}{' set' if secret_ok else ' missing'}",
                )
            )
            continue
        checks.append(_check(label, False, f"unsupported provider: {provider}"))
    return checks


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
