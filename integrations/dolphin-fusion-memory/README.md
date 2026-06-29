# Dolphin-Agent Fusion Memory Integration

This workspace lets Dolphin-Agent use Fusion Memory through HTTP-only workspace tools.
It does not load `MemoryService`, model code, or database code inside the Dolphin session
process.

Workspace path:

```text
/public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace
```

The workspace files under this directory are the canonical Dolphin/Fusion
Memory adapter. The `psi-agent` repository also carries a beginner-facing copy
at `examples/fusion-memory-workspace`; keep the Python adapter files in sync so
the agent example and the memory integration exercise the same HTTP contract.

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
- `PSI_AGENT_GATEWAY_URL`: optional Dolphin/psi-agent gateway URL used by
  automatic history persistence. Example: `http://127.0.0.1:8080`.
- `PSI_AGENT_WORKSPACE`: optional Dolphin/psi-agent workspace path used by
  automatic history persistence when no gateway URL is configured.
- `FUSION_MEMORY_SMOKE_MEMORY_URL`: smoke-script-only override for the Fusion Memory
  URL.

## First Use Setup

Before using `memory_add`, `memory_search`, or `memory_answer_context` for the
first time, initialize and start Fusion Memory. The workspace includes
`skills/fusion-memory-setup/SKILL.md` with the full beginner workflow.

Minimal local setup:

```bash
git clone https://github.com/genuineknowledge/fusion-memory.git
cd fusion-memory
sh install.sh
fusion-memory init --local-test --json
fusion-memory start --json
fusion-memory doctor --json
export PSI_MEMORY_BASE_URL=http://127.0.0.1:8700
```

If port `8700` is already in use, `fusion-memory start --json` tries the next available local port and returns the actual `url`; set `PSI_MEMORY_BASE_URL` to that returned URL before starting this workspace.

The Fusion Memory repository includes `models/Qwen3-Embedding-0.6B` and
`models/Qwen3-Reranker-0.6B`. The installer does not download model weights from
other locations. It installs full runtime dependencies (`.[postgres,qwen]`),
including Postgres adapter, local Qwen adapter, PyTorch, and Transformers. If
model files are missing or dependency installation failed, install-check reports
not ready and asks you to rerun `pip install -e ".[postgres,qwen]"`. Only when
model files and dependencies are present but this hardware/runtime cannot load or
run both bundled vector models does it fall back to a compromised local mode with
built-in lightweight retrieval and print the API-key next step. Recommended API
provider: Aliyun DashScope; set `DASHSCOPE_API_KEY` before configuring API-backed
providers.

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

## Automatic History Persistence

The workspace tools do not automatically write every conversation turn by
themselves. By default, memory is written only when the agent calls
`memory_add`. Explicit tool writes are marked with
`metadata.write_mode="explicit_tool"` and `metadata.auto_persisted=false`.

For continuous persistence without changing Dolphin-Agent core, run the Fusion
Memory history sync process beside the agent session. It reads Dolphin history
and posts new user/assistant turns to Fusion Memory `/add`. Re-running it is
safe because it stores a local deduplication state file. Synced writes are
marked with `metadata.write_mode="history_sync"` and
`metadata.auto_persisted=true`.

If you use the psi-agent gateway, start the sync against the gateway history API:

```bash
fusion-memory sync-dolphin-history \
  --gateway-url http://127.0.0.1:8080 \
  --session-id <session-id>
```

If you run a plain workspace session, sync directly from the workspace history
file:

```bash
fusion-memory sync-dolphin-history \
  --workspace /path/to/fusion-memory-workspace \
  --session-id <session-id>
```

For a one-time backfill, add `--once --json`. The gateway mode requires the
gateway process to be running and the session to be registered in that gateway.
The workspace-file mode reads `histories/<session-id>.jsonl` and works after the
session has saved history to disk.

## Copy Into Another Workspace

This workspace is self-contained. To create a new memory-only workspace, copy the
whole directory:

```bash
cp -R /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace ./my-memory-workspace
```

Then start Dolphin-Agent with `--workspace ./my-memory-workspace`.

To add Fusion Memory to an existing workspace, copy the memory tools and merge
the system prompt instructions:

```bash
mkdir -p ./my-workspace/tools ./my-workspace/systems ./my-workspace/skills
cp /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace/tools/_client.py ./my-workspace/tools/
cp /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace/tools/_config.py ./my-workspace/tools/
cp /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace/tools/memory_*.py ./my-workspace/tools/
cp -R /public/home/wwb/memory/integrations/dolphin-fusion-memory/workspace/skills/fusion-memory-setup ./my-workspace/skills/
```

If the target workspace already has `systems/system.py`, add the Fusion Memory
tool guidance from this workspace's `systems/system.py` into the existing prompt.
If it does not, copy `workspace/systems/system.py` as-is.

## Offline Behavior

The tools never raise Fusion Memory connection failures into the agent loop. If
Fusion Memory is offline or returns an error, they return the shared fallback:

```json
{"ok": false, "message": "Fusion Memory is not available. Continue without memory, then run fusion-memory doctor."}
```

The Dolphin session can continue without memory and retry after the Fusion Memory
server is available.
