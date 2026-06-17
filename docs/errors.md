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
fusion-memory doctor
```

Check that Postgres is running and that the configured database exists.

## Model is not ready

Run:

```bash
fusion-memory doctor
```

The default models are Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B.

## Adapter is not enabled

Run:

```bash
fusion-memory install-agent --target all
```

For one adapter, use the matching recovery command:

```bash
fusion-memory install-agent --target openclaw
fusion-memory install-agent --target hermes
fusion-memory install-agent --target fusion-agent
fusion-memory doctor
```

OpenClaw recovery: reinstall the external OpenClaw plugin, restart OpenClaw, and
keep the OpenClaw source checkout unchanged.

Hermes recovery: reinstall the external Hermes provider, restart Hermes, and
keep the Hermes source checkout unchanged.

Fusion-Agent recovery: set `PSI_MEMORY_BASE_URL` in the current shell, start the
session with `--memory-enabled`, and allow the agent to continue without memory
if the local service is unavailable.

For test model configuration, pass `/public/home/wwb/test_key/key.txt` by path.
Never paste key contents into an issue, log, or chat.
