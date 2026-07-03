# Error Guide

## Fusion Memory is not available

Run:

```bash
fusion-memory doctor
fusion-memory start
```

The Agent should continue without memory.

## Database is not ready

Run:

```bash
fusion-memory doctor --json
```

Check that Postgres is running and that the configured database exists.
The JSON report includes `postgres_connection` and `pgvector` checks.

## Model is not ready

Run:

```bash
fusion-memory doctor --json
```

The default models are Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B.
The JSON report includes `embedding_dependency`, `embedding_readiness`,
`reranker_dependency`, and `reranker_readiness` checks.

## Port is not available

Run:

```bash
fusion-memory doctor --json
```

Check the `port` and `service` entries. If the port is already in use, edit the
configured port or stop the other process, then run `fusion-memory doctor`
again.

## Upgrade rollback

Run:

```bash
fusion-memory upgrade --dry-run --json
```

The dry-run report shows the backup directory and rollback step before changing
the installed package.

## Adapter is not enabled

Run:

```bash
fusion-memory install-agent --target all
```

For one adapter, use the matching recovery command:

```bash
fusion-memory install-agent --target dolphin
fusion-memory install-agent --target openclaw
fusion-memory install-agent --target hermes
fusion-memory install-agent --target fusion-agent
fusion-memory doctor
```

Haitun recovery: verify the Haitun workspace, confirm `PSI_MEMORY_BASE_URL`
points at the local Fusion Memory daemon, then run
`fusion-memory sync-haitun-history --workspace <workspace> --session-id <session-id>`
beside the Haitun session for passive persistence.

OpenClaw recovery: reinstall the external OpenClaw plugin, restart OpenClaw, and
keep the OpenClaw source checkout unchanged.

Hermes recovery: reinstall the external Hermes provider, restart Hermes, and
keep the Hermes source checkout unchanged.

Fusion-Agent recovery: set `PSI_MEMORY_BASE_URL` in the current shell, start the
session with `--memory-enabled`, and allow the agent to continue without memory
if the local service is unavailable.

For test model configuration, set `MODEL_CONFIG_FILE` to your local key file
path and pass that path to benchmark commands. Never paste key contents into an
issue, log, or chat.
