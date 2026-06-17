# Agent Adapters

Fusion Memory uses one local service for all Agent integrations.

Install all adapters:

```bash
fusion-memory install-agent --target all
```

Install one adapter:

```bash
fusion-memory install-agent --target openclaw
fusion-memory install-agent --target hermes
fusion-memory install-agent --target fusion-agent
```

OpenClaw and Hermes are installed as external plugins. Their source checkouts
are not modified in stage one.

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

For test model configuration, pass `/public/home/wwb/test_key/key.txt` as a file
path to test commands that accept a model config file. Do not paste key content
into logs or docs.
