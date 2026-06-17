# Fusion Memory Agent Adapters Design

## Goal

First-stage productization adapts Fusion Memory to three Agent systems:

- Fusion-Agent at `/public/home/wwb/Fusion-Agent`
- real OpenClaw, pulled to `/public/home/wwb/GitHub/openclaw`
- real Hermes, pulled to `/public/home/wwb/GitHub/hermes-agent`

The first stage must not modify real OpenClaw or Hermes source code. OpenClaw and Hermes integration ships as external lightweight client plugins that talk to one local Fusion Memory HTTP service. Fusion-Agent may continue from its current in-repo partial integration.

## Product Targets

The target user is a beginner. The default path should avoid choices unless they are necessary.

Acceptance targets:

- Beginner install finishes in 30 seconds when prerequisites and local model/cache resources already exist.
- Install failure rate target: 1% after prerequisites are satisfied.
- One-command upgrade backs up data first and should not break a working install; failure rate target: 1%.
- Beginner guide should be understandable by 80% of non-technical users.
- Windows, Linux, and macOS are supported.
- Normal use should not crash the host Agent; crash/exit rate target: 1%.
- Core user flows are closed loop: install, initialize, start service, connect Agent, auto-recall, manual read/write, auto-write, doctor, upgrade, rollback guidance.
- Tests cover 80% of main scenarios and expected exception paths.
- Beginner-facing surfaces never expose Python tracebacks, Node stack traces, DSNs with secrets, raw HTTP errors, or internal table names.
- First token response impact target is under 2 seconds. Memory retrieval must fail open and use bounded timeouts.
- Help docs and error guidance exist for install, doctor, upgrade, service unavailable, Postgres unavailable, model unavailable, and plugin not enabled.
- Default configuration covers 90% of beginner use: local service, Postgres storage, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, rule extractor unless an LLM extractor endpoint is explicitly configured.

Testing model key resources are referenced by path only: `/public/home/wwb/test_key/key.txt`. The implementation and docs must not print or copy secrets from this file.

## Architecture

Fusion Memory remains the single memory runtime:

```text
Agent process
  -> lightweight Agent adapter/plugin/provider
  -> http://127.0.0.1:8765
  -> Fusion Memory service
  -> Postgres/pgvector
  -> Qwen3 embedding/reranker
```

The Agent adapters are clients only. They do not embed Fusion Memory internals, do not manage Postgres directly, and do not load Qwen models in the Agent process. This keeps Agent startup fast, isolates failures, and gives upgrade/rollback one owner.

## Shared HTTP Contract

Fusion Memory service exposes the existing endpoints and adds product-grade behavior around them:

- `GET /health`: fast liveness check.
- `GET /status`: readiness detail for service, database, models, and migrations.
- `POST /add`: write user/assistant turns or manual memory entries.
- `POST /answer-context`: retrieve context for a turn.
- `POST /search`: manual search.
- `POST /clear`: scoped clear for tests and user-requested reset.

All endpoints return JSON objects. User-facing clients convert errors into short guidance:

- "Fusion Memory is not running. Run `fusion-memory start` or `fusion-memory doctor`."
- "Fusion Memory is starting. Try again in a moment."
- "The database is not ready. Run `fusion-memory doctor`."
- "The model is not ready. Run `fusion-memory doctor --models`."

Adapters must not surface stack traces or raw exception text to the user.

## Scope Model

Every request uses a stable scope:

```json
{
  "workspace_id": "...",
  "user_id": "...",
  "agent_id": "fusion-agent | openclaw | hermes",
  "session_id": "...",
  "app_id": "fusion-memory"
}
```

Default scope derivation:

- `workspace_id`: workspace/profile/agent root name unless overridden.
- `user_id`: platform user id when available, otherwise OS user.
- `agent_id`: fixed per integration.
- `session_id`: Agent session id when available; otherwise a process/session generated id.
- `app_id`: `fusion-memory`.

Manual reads default to cross-session within the same workspace/user/agent. Automatic writes include session id so history remains attributable.

## Fusion Memory Product Runtime

The memory repository owns:

- `fusion-memory init`: creates config and runs migrations.
- `fusion-memory start|stop|status|doctor`: manages the local service.
- `fusion-memory upgrade`: backs up config/data, upgrades package, runs compatibility checks, and leaves the previous config/data intact on failure.
- `fusion-memory install-agent`: installs/configures Agent adapters.
- `fusion-memory alpha-test` and `fusion-memory beta-test`: simulation harnesses.

Default initialized config:

```json
{
  "storage_backend": "postgres",
  "embedding": {"provider": "qwen", "model": "Qwen/Qwen3-Embedding-0.6B"},
  "reranker": {"provider": "qwen", "model": "Qwen/Qwen3-Reranker-0.6B"},
  "extractor": {"provider": "rule"},
  "query_intent": {"provider": "off"}
}
```

The installer should support a beginner default and an escape hatch:

- Beginner default: Postgres + Qwen, using existing local resources or managed local service config.
- Fallback only when explicitly selected: SQLite + deterministic/lexical, marked as test/local fallback rather than the product default.

Postgres must be verified before the service is reported ready. If a local Postgres bootstrap is provided, it must be idempotent and must not overwrite an existing database without confirmation.

## OpenClaw External Plugin

OpenClaw source is not modified in stage one.

Plugin location in this project:

```text
memory/integrations/openclaw-fusion-memory/
```

Install path:

```bash
openclaw plugins install --link /public/home/wwb/memory/integrations/openclaw-fusion-memory
```

The plugin is a native OpenClaw plugin with:

- `openclaw.plugin.json`
- `package.json`
- TypeScript runtime entry
- tests
- beginner README

It registers tools:

- `fusion_memory_search`
- `fusion_memory_get`
- `fusion_memory_store`
- `fusion_memory_clear`

It also registers a CLI/status surface if supported by the external plugin API:

- `openclaw fusion-memory status`
- or plugin-owned command through OpenClaw's plugin CLI registration.

Tool behavior:

- `fusion_memory_search` calls `/answer-context` or `/search`.
- `fusion_memory_store` calls `/add`.
- `fusion_memory_get` returns exact retrieved source snippets when possible.
- `fusion_memory_clear` is available but guarded by clear wording and scoped defaults.

Active recall:

- OpenClaw `active-memory` can use configured memory tools. The installer configures the active-memory allowlist to include Fusion Memory tools when possible.
- If active-memory is unavailable or disabled, manual tools still work and doctor explains how to enable automatic recall.

Failure behavior:

- Service unavailable returns a short tool result telling the agent to continue without memory and optionally tell the user to run doctor.
- Timeout returns a short result within the configured memory timeout.
- The plugin never throws uncaught errors into OpenClaw runtime.

## Hermes External Provider

Hermes source is not modified in stage one.

Provider location in this project:

```text
memory/integrations/hermes-fusion-memory/
```

Install path:

```text
$HERMES_HOME/plugins/fusion_memory/
```

The installer copies or symlinks the provider directory and writes Hermes config:

```yaml
memory:
  provider: fusion_memory
```

Provider responsibilities:

- Implements `agent.memory_provider.MemoryProvider`.
- `is_available()` checks config/env only and does not make slow network calls.
- `initialize()` stores session/workspace/platform identity.
- `system_prompt_block()` gives short static guidance that memory may be available.
- `prefetch(query)` calls Fusion Memory with a strict timeout and returns fenced context.
- `queue_prefetch(query)` may optionally prefetch in background for the next turn.
- `sync_turn(user_content, assistant_content, messages=...)` writes completed turns asynchronously or quickly.
- `on_memory_write(...)` mirrors Hermes built-in memory writes into Fusion Memory.
- `get_tool_schemas()` exposes manual read/write/clear tools.
- `handle_tool_call()` maps tool calls to Fusion Memory HTTP endpoints.
- `get_config_schema()` lets `hermes memory setup` discover service URL and timeout if used.

Failure behavior:

- Provider failures are logged at debug/warning level and return empty context.
- Tool calls return beginner-safe messages.
- Shutdown drains or abandons background work within Hermes' existing timeout expectations.

## Fusion-Agent Integration

Fusion-Agent may be modified in repo, continuing from the current partial state:

- `src/psi_agent/memory/client.py`
- `src/psi_agent/memory/adapter.py`
- `src/psi_agent/memory/tool_api.py`
- `src/psi_agent/memory/scope.py`
- `src/psi_agent/memory/formatting.py`
- session CLI/config/runner integration
- openclaw-style and hermes-style example workspace tools

Required stage-one improvements:

- Tool API catches `FusionMemoryError`, timeout, and JSON errors, returning beginner-safe messages.
- Session automatic recall/write remains fail-open.
- CLI options and env vars are documented.
- Workspace tools expose both compatibility style (`memory(action=...)`) and direct style (`memory_read`, `memory_write`) where useful.
- Tests cover auto recall, auto write, manual tools, service down, non-JSON response, HTTP 400, timeout, and scope derivation.

## Unified Installer

The installer lives in the memory repo and owns the full beginner path:

```bash
fusion-memory install
fusion-memory install-agent --target all
fusion-memory install-agent --target openclaw
fusion-memory install-agent --target hermes
fusion-memory install-agent --target fusion-agent
```

Expected flow:

1. Check Python version and package install.
2. Initialize Fusion Memory config.
3. Verify Postgres and run migrations.
4. Verify Qwen embedding/reranker can load or report actionable guidance.
5. Start Fusion Memory service or confirm it is already running.
6. Install selected Agent adapters.
7. Write only minimal Agent config changes, backing up files first.
8. Run doctor checks for service and each selected Agent.
9. Print a short success message with next command.

Installers must be idempotent. Re-running should update config if needed and preserve user data.

Upgrade flow:

1. Stop or leave service running based on compatibility needs.
2. Back up Fusion Memory config, database pointers, and Agent config files touched by the installer.
3. Upgrade package/plugin files.
4. Run migrations.
5. Restart or health-check service.
6. Verify each Agent plugin still loads.
7. On failure, leave previous config/data in place and print rollback guidance.

## Documentation

Docs to add/update:

- `docs/quickstart.md`: beginner path with one command and default choices.
- `docs/agent-adapters.md`: OpenClaw, Hermes, Fusion-Agent setup and troubleshooting.
- `docs/errors.md`: user-facing error guide.
- plugin READMEs under both integration directories.
- alpha/beta test reports under `docs/alpha-beta/`.

Docs must not ask beginners to understand Postgres internals. They should say what to run next.

## Alpha Test

Alpha is a local simulation suite for Fusion Memory before broader beta usage.

It should run against a disposable scope and test:

- Fresh install dry run.
- Config init with default Postgres/Qwen settings.
- Doctor with service stopped.
- Service start and health.
- Add one preference, retrieve it, and answer-context includes it.
- Multi-turn auto-write shape.
- Cross-session retrieval within same user/workspace.
- Scope isolation across users/agents.
- Service unavailable behavior for all three adapter clients.
- HTTP 400 and non-JSON response handling.
- Timeout handling under 2 seconds for adapter calls.
- Upgrade dry run creates backup and command plan.
- Clear/reset only deletes scoped data.

Alpha passes when all critical checks pass locally and all failures are user-friendly.

## Beta Test

Beta simulates beginner and cross-Agent workflows:

- Install all adapters on Linux.
- Install selected adapter on macOS and Windows where available.
- OpenClaw manual store/search.
- OpenClaw automatic recall through active-memory when configured.
- Hermes prefetch/sync turn.
- Hermes manual tools.
- Fusion-Agent session auto recall/write.
- One fact written by one Agent can be retrieved by another only when scope policy permits.
- Postgres restart recovery.
- Model unavailable guidance.
- Agent plugin disabled/uninstalled recovery.
- Upgrade from previous adapter version.

Beta report records:

- install time
- first token impact
- tool latency
- service memory/model load time
- failure messages
- pass/fail per scenario

## Non-Goals For Stage One

- No upstream changes to real OpenClaw or Hermes source.
- No bundled OpenClaw plugin in `/public/home/wwb/GitHub/openclaw/extensions`.
- No bundled Hermes provider in `/public/home/wwb/GitHub/hermes-agent/plugins`.
- No GUI installer.
- No hidden automatic reading of `/public/home/wwb/test_key/key.txt`.
- No replacement of OpenClaw's built-in memory files or Hermes' built-in memory tool. Fusion Memory mirrors and augments them.

## Open Questions Resolved

- Agent integration mode: one shared local Fusion Memory HTTP service.
- OpenClaw/Hermes source changes: none in stage one.
- Fusion-Agent source changes: allowed, continuing current partial adaptation.
- Test model key file: use `/public/home/wwb/test_key/key.txt` by path only.

