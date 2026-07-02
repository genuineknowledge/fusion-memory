# Fusion Memory Install Daemon Agent Consent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement agent first-use consent, robust repo-local model validation, hardware/runtime fallback diagnostics, and daemon service launch behavior.

**Architecture:** Memory-side product logic stays in `fusion_memory/product.py`, with focused unit tests in `tests/test_product_cli.py` and prompt-sync tests in `tests/test_dolphin_agent_example_sync.py`. Agent-side prompt copies are updated in Dolphin from latest `origin/main` and verified by new prompt tests.

**Tech Stack:** Python 3.11+ for Fusion Memory, Python 3.14+ style in Dolphin, `unittest`, `pytest`, standard-library subprocess/platform/os helpers.

## Global Constraints

- Memory repository changes are committed and pushed directly to `genuineknowledge/main`.
- Dolphin repository must track latest `origin/main` before branching.
- Dolphin changes are pushed to a new branch and opened as a PR.
- The installer must not download model weights.
- Missing model files, small stub files, and Git LFS pointer files must return `not_ready`, not `compromised`.
- `compromised` mode is only for complete model files and dependencies that fail runtime/hardware smoke.
- Windows service start must not open a terminal window.

---

### Task 1: Memory Prompt Consent Tests

**Files:**
- Modify: `tests/test_dolphin_agent_example_sync.py`
- Modify: `integrations/dolphin-fusion-memory/workspace/systems/system.py`
- Modify: `integrations/dolphin-fusion-memory/workspace/skills/fusion-memory-setup/SKILL.md`

**Interfaces:**
- Consumes: `system_prompt_builder() -> str`
- Produces: prompt text containing `ask the user whether to enable Fusion Memory persistent memory` and `cannot remember across sessions`

- [ ] **Step 1: Write failing tests**

Add assertions to `test_first_use_setup_skill_uses_public_repository_and_documents_compromised_fallback()`:

```python
assert "Git LFS pointer" in skill
assert "git lfs pull" in skill
```

Add a new async test:

```python
@pytest.mark.anyio
async def test_canonical_memory_workspace_prompt_asks_before_enabling_persistence() -> None:
    import importlib.util

    path = CANONICAL_WORKSPACE / "systems" / "system.py"
    spec = importlib.util.spec_from_file_location("canonical_memory_system", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    prompt = await module.system_prompt_builder()

    assert "ask the user whether to enable Fusion Memory persistent memory" in prompt
    assert "cannot remember across sessions" in prompt
    assert "If the user declines" in prompt
```

- [ ] **Step 2: Verify red**

Run:

```bash
python3 -m pytest tests/test_dolphin_agent_example_sync.py -q
```

Expected: fails because the prompt and setup skill do not contain the new consent/LFS wording.

- [ ] **Step 3: Implement prompt text**

Update `integrations/dolphin-fusion-memory/workspace/systems/system.py` so the first-use section says:

```text
Before the first use of Fusion Memory, ask the user whether to enable Fusion Memory persistent memory. Explain that without installing and enabling Fusion Memory, you cannot remember across sessions and can only use current-session context. If the user agrees, use the fusion-memory-setup skill to initialize, start, and check the Fusion Memory service. If the user declines, continue without memory and do not call Fusion Memory tools.
```

Update the setup skill to say Git LFS pointers are rejected and users should run `git lfs pull` to fetch real weights.

- [ ] **Step 4: Verify green**

Run:

```bash
python3 -m pytest tests/test_dolphin_agent_example_sync.py -q
```

Expected: passes or skips only external `PSI_AGENT_ROOT` sync tests.

### Task 2: Model Directory Readiness

**Files:**
- Modify: `fusion_memory/product.py`
- Modify: `tests/test_product_cli.py`

**Interfaces:**
- Produces: `_repository_model_status(path: Path, *, label: str = "Qwen model") -> dict[str, Any]`
- Produces: `_repository_model_ready(path: Path) -> bool`

- [ ] **Step 1: Write failing tests**

Add tests to `tests/test_product_cli.py`:

```python
def test_install_readiness_rejects_git_lfs_pointer_model_files(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        embedding = home / "models" / "Qwen3-Embedding-0.6B"
        reranker = home / "models" / "Qwen3-Reranker-0.6B"
        _write_lfs_pointer_model_dir(embedding)
        _write_ready_model_dir(reranker)
        with (
            patch("fusion_memory.product.default_local_embedding_model_path", return_value=embedding),
            patch("fusion_memory.product.default_local_reranker_model_path", return_value=reranker),
            patch("fusion_memory.product._qwen_dependency_available", return_value=True),
        ):
            result = install_readiness(home, force=True)

    self.assertFalse(result["ok"])
    self.assertEqual(result["mode"], "not_ready")
    self.assertFalse(result["compromised"])
    self.assertIn("Git LFS pointer", result["message"])
    self.assertIn("git lfs pull", result["next_step"])
```

```python
def test_install_readiness_rejects_tiny_stub_model_files(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        embedding = home / "models" / "Qwen3-Embedding-0.6B"
        reranker = home / "models" / "Qwen3-Reranker-0.6B"
        _write_stub_model_dir(embedding)
        _write_ready_model_dir(reranker)
        with (
            patch("fusion_memory.product.default_local_embedding_model_path", return_value=embedding),
            patch("fusion_memory.product.default_local_reranker_model_path", return_value=reranker),
            patch("fusion_memory.product._qwen_dependency_available", return_value=True),
        ):
            result = install_readiness(home, force=True)

    self.assertFalse(result["ok"])
    self.assertEqual(result["mode"], "not_ready")
    self.assertIn("too small", result["message"])
```

- [ ] **Step 2: Verify red**

Run:

```bash
python3 -m pytest tests/test_product_cli.py -k "install_readiness" -q
```

Expected: fails because pointer/stub files are currently treated as ready.

- [ ] **Step 3: Implement structured model status**

Add constants:

```python
MODEL_SAFETENSORS_MIN_BYTES = 100 * 1024 * 1024
TOKENIZER_JSON_MIN_BYTES = 100 * 1024
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
```

Add `_repository_model_status()`, `_file_status()`, and `_is_lfs_pointer()` helpers. Make `_repository_model_ready()` return `_repository_model_status(path)["ok"]`.

Update `install_readiness()` missing messages to include model status details and a Git LFS next step when any detail mentions a pointer.

- [ ] **Step 4: Verify green**

Run:

```bash
python3 -m pytest tests/test_product_cli.py -k "install_readiness" -q
```

Expected: passes.

### Task 3: Hardware Probe And Compromised Diagnostics

**Files:**
- Modify: `fusion_memory/product.py`
- Modify: `tests/test_product_cli.py`

**Interfaces:**
- Produces: `_hardware_runtime_probe() -> dict[str, Any]`
- Produces: `install_readiness(...): result["hardware_probe"]`

- [ ] **Step 1: Write failing test**

Add:

```python
def test_install_readiness_compromised_result_includes_hardware_probe(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        embedding = home / "models" / "Qwen3-Embedding-0.6B"
        reranker = home / "models" / "Qwen3-Reranker-0.6B"
        _write_ready_model_dir(embedding)
        _write_ready_model_dir(reranker)
        probe = {"platform": {"system": "Windows"}, "torch": {"available": False}}
        with (
            patch("fusion_memory.product.default_local_embedding_model_path", return_value=embedding),
            patch("fusion_memory.product.default_local_reranker_model_path", return_value=reranker),
            patch("fusion_memory.product._qwen_dependency_available", return_value=True),
            patch("fusion_memory.product._qwen_runtime_smoke", return_value={"ok": False, "message": "runtime failed"}),
            patch("fusion_memory.product._hardware_runtime_probe", return_value=probe),
        ):
            result = install_readiness(home, force=True)

    self.assertTrue(result["compromised"])
    self.assertEqual(result["hardware_probe"], probe)
    self.assertEqual(result["runtime_smoke"]["hardware_probe"], probe)
```

- [ ] **Step 2: Verify red**

Run:

```bash
python3 -m pytest tests/test_product_cli.py -k "compromised_result_includes_hardware_probe" -q
```

Expected: fails because hardware probe is absent.

- [ ] **Step 3: Implement probe**

Add `_hardware_runtime_probe()` using `platform`, `os.cpu_count()`, optional `os.sysconf`, and optional `torch` import. Add the probe to compromised `result` and nested `runtime_smoke`.

- [ ] **Step 4: Verify green**

Run:

```bash
python3 -m pytest tests/test_product_cli.py -k "compromised_result_includes_hardware_probe or falls_back_to_compromised" -q
```

Expected: passes.

### Task 4: Daemon Popen Options

**Files:**
- Modify: `fusion_memory/product.py`
- Modify: `tests/test_product_cli.py`

**Interfaces:**
- Produces: `_daemon_popen_kwargs(log_handle: Any, cwd: str, env: dict[str, str]) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_daemon_popen_kwargs_uses_unix_session_detach(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "service.log"
        with log_path.open("ab") as handle, patch("fusion_memory.product.os.name", "posix"):
            kwargs = product._daemon_popen_kwargs(handle, cwd=tmp, env={"A": "B"})

    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["stdin"], product.subprocess.DEVNULL)
    self.assertEqual(kwargs["stderr"], product.subprocess.STDOUT)
```

```python
def test_daemon_popen_kwargs_uses_windows_no_window_flag(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "service.log"
        with (
            log_path.open("ab") as handle,
            patch("fusion_memory.product.os.name", "nt"),
            patch.object(product.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, create=True),
            patch.object(product.subprocess, "DETACHED_PROCESS", 0x00000008, create=True),
            patch.object(product.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
        ):
            kwargs = product._daemon_popen_kwargs(handle, cwd=tmp, env={"A": "B"})

    self.assertEqual(kwargs["creationflags"] & 0x08000000, 0x08000000)
    self.assertEqual(kwargs["creationflags"] & 0x00000008, 0x00000008)
    self.assertEqual(kwargs["creationflags"] & 0x00000200, 0x00000200)
```

- [ ] **Step 2: Verify red**

Run:

```bash
python3 -m pytest tests/test_product_cli.py -k "daemon_popen_kwargs" -q
```

Expected: fails because helper does not exist.

- [ ] **Step 3: Implement daemon helper**

Move current `Popen` kwargs construction into `_daemon_popen_kwargs()`. Windows flags must include `CREATE_NO_WINDOW` when available. Update `start_service()` to call it.

- [ ] **Step 4: Verify green**

Run:

```bash
python3 -m pytest tests/test_product_cli.py -k "daemon_popen_kwargs or start_service_tries_next_port" -q
```

Expected: passes.

### Task 5: Dolphin Agent Branch Sync

**Files:**
- Modify: `/public/home/wwb/Dolphin-Agent/examples/fusion-memory-workspace/systems/system.py`
- Modify: `/public/home/wwb/Dolphin-Agent/examples/fusion-memory-workspace/skills/fusion-memory-setup/SKILL.md`
- Modify: `/public/home/wwb/Dolphin-Agent/examples/haitun-workspace/systems/prompt_sections.py`
- Modify: `/public/home/wwb/Dolphin-Agent/examples/haitun-workspace/skills/fusion-memory-setup/SKILL.md`
- Modify or create: `/public/home/wwb/Dolphin-Agent/tests/integration/test_fusion_memory_prompt.py`

**Interfaces:**
- Consumes: `FUSION_MEMORY_SECTION`
- Consumes: `system_prompt_builder() -> str`

- [ ] **Step 1: Create branch from latest main**

Run:

```bash
git -C /public/home/wwb/Dolphin-Agent checkout -b fusion-memory-consent-prompt
```

- [ ] **Step 2: Write failing tests**

Create `tests/integration/test_fusion_memory_prompt.py`:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.anyio
async def test_fusion_memory_workspace_prompt_requires_first_use_consent() -> None:
    module = _load_module(ROOT / "examples" / "fusion-memory-workspace" / "systems" / "system.py", "fusion_memory_workspace_system")
    prompt = await module.system_prompt_builder()

    assert "ask the user whether to enable Fusion Memory persistent memory" in prompt
    assert "cannot remember across sessions" in prompt
    assert "If the user declines" in prompt


def test_haitun_fusion_memory_section_requires_first_use_consent() -> None:
    module = _load_module(ROOT / "examples" / "haitun-workspace" / "systems" / "prompt_sections.py", "haitun_prompt_sections")
    section = module.FUSION_MEMORY_SECTION

    assert "ask the user whether to enable Fusion Memory persistent memory" in section
    assert "cannot remember across sessions" in section
    assert "If the user declines" in section


def _load_module(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(path.parent))
        except ValueError:
            pass
```

- [ ] **Step 3: Verify red**

Run:

```bash
uv run pytest tests/integration/test_fusion_memory_prompt.py -q
```

Expected: fails because Dolphin prompt copies do not contain the consent wording.

- [ ] **Step 4: Implement Dolphin prompt sync**

Copy the new first-use wording and Git LFS setup skill wording from the memory repository into the two Dolphin workspace copies.

- [ ] **Step 5: Verify green**

Run:

```bash
uv run pytest tests/integration/test_fusion_memory_prompt.py -q
```

Expected: passes.

### Task 6: Final Verification And Publishing

**Files:**
- No new files beyond prior tasks.

**Interfaces:**
- Produces: pushed Fusion Memory `main`
- Produces: pushed Dolphin PR branch

- [ ] **Step 1: Run memory focused tests**

Run:

```bash
python3 -m pytest tests/test_product_cli.py tests/test_dolphin_agent_example_sync.py tests/test_docs_agent_adapters.py -q
```

Expected: all non-environment-gated tests pass.

- [ ] **Step 2: Run memory compile check**

Run:

```bash
python3 -m compileall -q fusion_memory tests
```

Expected: exit code 0.

- [ ] **Step 3: Run Dolphin focused tests**

Run:

```bash
uv run pytest tests/integration/test_fusion_memory_prompt.py -q
```

Expected: exit code 0.

- [ ] **Step 4: Commit memory changes on main**

Run:

```bash
git -C /public/home/wwb/memory status --short
git -C /public/home/wwb/memory add fusion_memory/product.py tests/test_product_cli.py tests/test_dolphin_agent_example_sync.py integrations/dolphin-fusion-memory/workspace/systems/system.py integrations/dolphin-fusion-memory/workspace/skills/fusion-memory-setup/SKILL.md docs/superpowers/specs/2026-07-02-fusion-memory-install-daemon-agent-consent-design.md docs/superpowers/plans/2026-07-02-fusion-memory-install-daemon-agent-consent.md
git -C /public/home/wwb/memory commit -m "fix: harden install readiness and daemon startup"
git -C /public/home/wwb/memory push genuineknowledge main
```

- [ ] **Step 5: Commit Dolphin branch**

Run:

```bash
git -C /public/home/wwb/Dolphin-Agent status --short
git -C /public/home/wwb/Dolphin-Agent add examples/fusion-memory-workspace/systems/system.py examples/fusion-memory-workspace/skills/fusion-memory-setup/SKILL.md examples/haitun-workspace/systems/prompt_sections.py examples/haitun-workspace/skills/fusion-memory-setup/SKILL.md tests/integration/test_fusion_memory_prompt.py
git -C /public/home/wwb/Dolphin-Agent commit -m "fix: ask before enabling Fusion Memory persistence"
git -C /public/home/wwb/Dolphin-Agent push -u origin fusion-memory-consent-prompt
```

- [ ] **Step 6: Open Dolphin PR**

Run:

```bash
gh pr create --repo genuineknowledge/psi-agent --base main --head fusion-memory-consent-prompt --title "Ask before enabling Fusion Memory persistence" --body "Adds first-use Fusion Memory persistence consent wording and documents that memory cannot persist across sessions unless Fusion Memory is installed and enabled."
```

Expected: PR URL is printed.
