# Agent Adapters

Fusion Memory uses one local service for all Agent integrations.

Install all adapters:

```bash
fusion-memory install-agent --target all
```

Install one adapter:

```bash
fusion-memory install-agent --target dolphin
fusion-memory install-agent --target openclaw
fusion-memory install-agent --target hermes
fusion-memory install-agent --target fusion-agent
```

Run a runtime smoke after installing an adapter and starting the local Fusion
Memory service:

```bash
python3 tools/agent_runtime_smoke.py --target dolphin --memory-url http://127.0.0.1:8765 --output .runtime/agent-smoke-dolphin.json
python3 tools/agent_runtime_smoke.py --target openclaw --memory-url http://127.0.0.1:8765 --output .runtime/agent-smoke-openclaw.json
python3 tools/agent_runtime_smoke.py --target hermes --memory-url http://127.0.0.1:8765 --output .runtime/agent-smoke-hermes.json
python3 tools/agent_runtime_smoke.py --target fusion-agent --memory-url http://127.0.0.1:8765 --output .runtime/agent-smoke-fusion-agent.json
```

The smoke report contains `target`, `host_available`, `plugin_available`,
`write_smoke`, `retrieve_smoke`, `ok`, and `message`. Missing host binaries or
checkouts return `ok=false` with recovery guidance instead of raw runtime logs.
`plugin_available` means the adapter is visible to the target runtime where the
runtime exposes that information. Repository source files are not enough to make
the runtime smoke pass.

Dolphin, OpenClaw, Hermes, and Fusion-Agent include built-in write/retrieve smoke paths.
Dolphin runs the workspace smoke against `memory_add`, `memory_search`, and
`memory_answer_context`, and passive persistence is handled by
`sync-dolphin-history` reading saved JSONL history beside the Dolphin session.
OpenClaw first asks the host to inspect the `fusion-memory` plugin with
`openclaw plugins inspect fusion-memory --runtime --json`, then runs the
external plugin's `smoke.mjs` script, which executes the same registered
`fusion_memory_store` and `fusion_memory_search` tool handlers. Hermes loads the
installed `fusion_memory` provider from the runtime plugin directory and calls
its store/search tool methods. Fusion-Agent imports its in-repo memory adapter.
A passing report means both `write_smoke` and `retrieve_smoke` were explicitly
verified through the adapter path.

For custom host deployments, override the built-in smoke with an adapter-level
command that exercises the target adapter and prints JSON with explicit
`write_smoke` and `retrieve_smoke` boolean fields:

```bash
export FUSION_MEMORY_DOLPHIN_SMOKE_COMMAND="python3 integrations/dolphin-fusion-memory/smoke.py"
export FUSION_MEMORY_OPENCLAW_SMOKE_COMMAND="openclaw fusion-memory-smoke"
export FUSION_MEMORY_HERMES_SMOKE_COMMAND="hermes fusion-memory-smoke"
export FUSION_MEMORY_FUSION_AGENT_SMOKE_COMMAND="python3 path/to/fusion-agent-smoke.py"
```

The smoke harness passes the selected service URL to an override command as
`FUSION_MEMORY_SMOKE_MEMORY_URL`. If a built-in or override smoke cannot run,
the report returns `ok=false`, `write_smoke=false`, and `retrieve_smoke=false`
with a beginner-safe recovery message.

Dolphin uses the repository workspace under `integrations/dolphin-fusion-memory/workspace`.
OpenClaw and Hermes are installed as external plugins. Their source checkouts
are not modified in stage one.

## Dolphin recovery

If Dolphin memory tools are missing or passive persistence is not running,
verify the workspace and start the history sync beside the Dolphin session:

```bash
fusion-memory install-agent --target dolphin
fusion-memory doctor
fusion-memory --db fusion-memory.sqlite3 sync-dolphin-history \
  --workspace integrations/dolphin-fusion-memory/workspace \
  --session-id <session-id>
```

The sync process reads Dolphin's saved `histories/<session-id>.jsonl` file and
does not patch Dolphin source code. Continue the session if memory is offline;
restart the sync process after the local service is back.

Fusion-Agent uses its in-repo adapter. Start a session with memory enabled and:

Linux / macOS:

```bash
export PSI_MEMORY_BASE_URL=http://127.0.0.1:8765
```

Windows PowerShell:

```powershell
$env:PSI_MEMORY_BASE_URL = "http://127.0.0.1:8765"
```

Windows cmd:

```bat
set PSI_MEMORY_BASE_URL=http://127.0.0.1:8765
```

Then pass the memory flag to the agent session:

```bash
psi-agent session --memory-enabled
```

## OpenClaw recovery

If OpenClaw cannot find Fusion Memory tools, leave the OpenClaw checkout as-is
and reinstall the external plugin:

```bash
fusion-memory install-agent --target openclaw
fusion-memory doctor
```

Restart OpenClaw after reinstalling the plugin. If memory is still unavailable,
OpenClaw should continue without memory; use `fusion-memory doctor` output as
the support summary instead of pasting raw runtime logs.

## Hermes recovery

If Hermes does not show the Fusion Memory provider, reinstall the external
provider and restart Hermes:

```bash
fusion-memory install-agent --target hermes
fusion-memory doctor
```

Do not edit the Hermes source checkout for first-stage recovery. If the provider
is disabled or missing after reinstall, run `fusion-memory install-agent --target
all` and check that the local Fusion Memory service is running.

## Fusion-Agent recovery

If Fusion-Agent starts without memory, confirm the service URL is set for the
current shell and that the session was started with memory enabled:

```bash
fusion-memory doctor
psi-agent session --memory-enabled
```

On Windows, set `PSI_MEMORY_BASE_URL` with the PowerShell or cmd examples above
before starting the session. If Fusion Memory is unavailable, Fusion-Agent should
continue the session without memory and show beginner-safe recovery guidance.

For test model configuration, set `MODEL_CONFIG_FILE` to a local key file path
and pass it to commands that accept a model config file. Do not paste key
content into logs or docs.
