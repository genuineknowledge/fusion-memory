# Fusion Memory Install, Daemon, And Agent Consent Design

## Goal

Fix three productization gaps in Fusion Memory:

- The agent must proactively ask whether the user wants persistent Fusion Memory before first use, and explain that without installation/enabling it cannot remember across sessions.
- Install readiness must reject incomplete Git LFS pointer model files and must report hardware/runtime evidence before falling back to compromised mode.
- `fusion-memory start` must launch the service as a background daemon without opening a Windows terminal window.

## Scope

Memory-side changes are made in the Fusion Memory repository and pushed directly to `genuineknowledge/main`.

Agent-side changes are made in the Dolphin/psi-agent repository from the latest `origin/main`, pushed to a new branch, and opened as a pull request.

The installer will not download model weights. If the repository contains Git LFS pointer files instead of real weights, readiness will report `not_ready` with a Git LFS pull/install next step.

## Architecture

### Agent Consent Prompt

The canonical memory workspace prompt in `integrations/dolphin-fusion-memory/workspace/systems/system.py` will add a first-use policy:

- Before first using Fusion Memory tools, ask the user whether to enable Fusion Memory persistent memory.
- The prompt must state that without installing/enabling Fusion Memory, the agent cannot remember across sessions and can only use current-session context.
- If the user agrees, read and follow `skills/fusion-memory-setup/SKILL.md`.
- If the user declines, continue without memory and do not call memory tools.

The integrated Haitun workspace in Dolphin uses `examples/haitun-workspace/systems/prompt_sections.py`; the same policy will be mirrored there. The standalone Dolphin memory example uses `examples/fusion-memory-workspace/systems/system.py`; it will match the canonical memory workspace.

### Model Readiness

Fusion Memory will replace `_repository_model_ready(path) -> bool` with a structured model directory check. The check will validate:

- Required files exist.
- Required files are not Git LFS pointer files.
- `model.safetensors` is larger than a conservative minimum for Qwen 0.6B weights.
- `tokenizer.json` is larger than a conservative tokenizer minimum.
- `config.json` exists, is not a pointer, and parses as JSON.

`install_readiness()` will continue to return `not_ready` when files/dependencies are missing or incomplete. Compromised mode will remain reserved for cases where files and dependencies are present but the current hardware/runtime cannot run the bundled Qwen models.

### Hardware/Runtime Probe

Fusion Memory will add a small `_hardware_runtime_probe()` helper. It will avoid heavy model loads and report:

- Platform system, machine, processor, Python version.
- CPU count.
- Memory bytes when detectable through `os.sysconf`.
- Torch importability, version, CUDA availability, CUDA device count, and MPS availability when supported.

`install_readiness()` will include this probe in compromised-mode results and in the runtime smoke payload. This makes the fallback decision auditable without requiring users to inspect raw tracebacks.

### Daemon Launch

Fusion Memory will extract daemon `subprocess.Popen` options into a helper:

- Always route stdout/stderr to the service log and stdin to `DEVNULL`.
- On Unix-like systems, use `start_new_session=True`.
- On Windows, set `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW` when those constants exist.

`start_service()` will use this helper so the behavior is testable and platform-specific details are centralized.

## Error Handling

Model directory failures will be surfaced as beginner-safe `not_ready` messages. Git LFS pointer files will have an actionable next step: install Git LFS and pull the real model weights.

Runtime smoke failures will not expose raw tracebacks in user-facing messages. The structured `runtime_smoke` and `hardware_probe` payloads are machine-readable diagnostics.

## Testing

Memory-side focused tests:

- Git LFS pointer files are rejected by install readiness.
- Stub/small model files are rejected by install readiness.
- Compromised fallback includes hardware probe data when runtime smoke fails.
- Windows daemon options include `CREATE_NO_WINDOW` and detached flags.
- Unix daemon options include `start_new_session=True`.
- Canonical workspace system prompt includes the first-use consent and cross-session warning.

Dolphin-side focused tests:

- Standalone Fusion Memory workspace prompt includes first-use consent and cross-session warning.
- Integrated Haitun Fusion Memory section includes first-use consent and cross-session warning.

## Out Of Scope

- No automatic `git lfs pull`.
- No Hugging Face or DashScope model download flow.
- No systemd, launchd, or Windows Service registration.
- No changes to Fusion Memory server HTTP APIs.
