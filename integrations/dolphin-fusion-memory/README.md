# Dolphin-Agent Fusion Memory Integration

This workspace lets Dolphin-Agent use Fusion Memory through HTTP-only workspace tools.
It does not load `MemoryService`, model code, or database code inside the Dolphin session
process.

Workspace path:

```text
/public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace
```

## Tools

- `memory_add`: store durable user preferences, project facts, or stable decisions.
- `memory_search`: retrieve raw evidence by keyword.
- `memory_answer_context`: retrieve a query-grounded context pack, preferred for
  answering questions about prior user history, preferences, and project context.

## Environment

- `PSI_MEMORY_BASE_URL`: Fusion Memory HTTP server URL. Defaults to
  `http://127.0.0.1:8700`.
- `PSI_MEMORY_WORKSPACE_ID`: memory workspace scope. Defaults to `dolphin`.
- `PSI_MEMORY_USER_ID`: user scope. Defaults to the current OS user or `user`.
- `PSI_MEMORY_AGENT_ID`: agent scope. Defaults to `dolphin`.
- `PSI_MEMORY_SESSION_ID`: optional session scope. When unset, reads allow
  cross-session retrieval.
- `PSI_MEMORY_TIMEOUT_SECONDS`: request timeout in seconds. Defaults to `2.0` and is
  clamped to `0.1..5.0`.
- `FUSION_MEMORY_SMOKE_MEMORY_URL`: smoke-script-only override for the Fusion Memory
  URL.

## Run

Start Fusion Memory on the Dolphin default port:

```bash
cd /public/home/wwb/memory
python -m fusion_memory.server --port 8700
```

Run the live adapter smoke:

```bash
cd /public/home/wwb/memory
cd integrations/dolphin-fusion-memory
python smoke.py
```

The smoke writes a unique token with `memory_add`, retrieves it with
`memory_search`, and asks for `memory_answer_context`. It exits with code `0` only
when all three steps confirm the token against a live Fusion Memory server.

Start a Dolphin-Agent session with this workspace:

```bash
PSI_MEMORY_BASE_URL=http://127.0.0.1:8700 \
PSI_MEMORY_SESSION_ID=<session-id> \
uv run psi-agent session \
  --workspace /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace \
  --session-id <session-id> \
  --channel-socket ./channel.sock \
  --ai-socket ./ai.sock
```

## Offline Behavior

The tools never raise Fusion Memory connection failures into the agent loop. If
Fusion Memory is offline or returns an error, they return the shared fallback:

```json
{"ok": false, "message": "Fusion Memory is not available. Continue without memory, then run fusion-memory doctor."}
```

The Dolphin session can continue without memory and retry after the Fusion Memory
server is available.
