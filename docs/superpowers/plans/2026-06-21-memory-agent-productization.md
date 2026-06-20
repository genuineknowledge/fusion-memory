# Memory Agent Productization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current BEAM-driven memory prototype into a product-ready Agent memory module by keeping legacy retrieval stable, expanding replay coverage, making rules auditable, introducing a real retrieval pipeline execution layer, hardening product operations, and validating real Agent adapter smoke paths.

**Architecture:** Legacy event_ordering stays the production default. Graph, dual, and hybrid paths remain shadow/experimental until replay proves parity across event_ordering, current_value, multi-condition, and Chinese recall. Product readiness work is separated from retrieval behavior: CLI/doctor/installer changes must expose beginner-safe JSON/human messages without changing retrieval defaults.

**Tech Stack:** Python 3.11+/3.12, `unittest`, PostgreSQL + pgvector, local Qwen3 0.6B embedding/reranker, existing `MemoryService`, existing BEAM replay tooling, external OpenClaw/Hermes client plugins, Fusion-Agent checkout at `/public/home/wwb/Fusion-Agent`.

## Global Constraints

- Do not delete legacy event_ordering code in this phase.
- Do not make graph, dual, or hybrid the default production selector in this phase.
- Dual/hybrid paths are shadow or replay-only unless a feature flag explicitly enables telemetry collection.
- Every retrieval behavior change must be measurable with replay artifacts.
- Graph is a sorting/structure layer; raw chronology and legacy recall remain the recall backbone.
- LLM extractor/router stay out of the real-time main path.
- Do not add project-specific or software-specific regex rescue branches.
- Rule cleanup must be evidence-driven: first-pass deletion may only remove no-hit, no-contribution, or duplicate rules after audit evidence.
- No raw user text may be stored in rule-hit telemetry, rule audit, or pipeline trace.
- User-facing CLI/product paths must return safe actionable errors, not raw tracebacks.
- Default product configuration targets PostgreSQL + pgvector, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, and explicit local-test fallback when production dependencies are unavailable.
- Do not modify real OpenClaw or Hermes source trees; only external Fusion Memory plugins/adapters may change.

---

## File Structure

- Modify: `fusion_memory/core/runtime_config.py`
  - Keep legacy selector default; expose shadow flags only.
- Modify: `fusion_memory/api/service.py`
  - Route retrieval through a small pipeline execution layer while preserving current output.
  - Collect dual/hybrid shadow telemetry without changing selected candidates.
- Create: `fusion_memory/retrieval/pipeline.py`
  - Own `QueryUnderstanding`, `CandidateRecall`, `CandidateFusion`, and `EvidencePackBuilder` execution records.
- Modify: `fusion_memory/retrieval/retrieval_trace.py`
  - Serialize execution-layer traces with no raw user text.
- Modify: `tools/beam_event_ordering_replay.py`
  - Keep four-path event_ordering replay as the baseline gate.
- Create: `tools/beam_retrieval_replay.py`
  - Replay `current_value`, `multi_condition`, and Chinese recall probes through comparable retrieval paths.
- Modify: `fusion_memory/retrieval/rule_registry.py`
  - Ensure all registered hits have ability/category/stage and sanitized metadata.
- Modify: `fusion_memory/retrieval/rule_audit.py`
  - Add global audit merge support and first-pass deletion candidate classification.
- Modify: `tools/rule_audit.py`
  - Accept multiple replay inputs and emit a single audit table.
- Modify: `fusion_memory/product.py`
  - Harden `doctor`, safe error mapping, readiness checks, local-test fallback messaging, backup/upgrade checks.
- Modify: `fusion_memory/cli.py`
  - Ensure all product commands return beginner-safe JSON on parser/runtime errors.
- Modify: `fusion_memory/agent_installer.py`
  - Add runtime smoke commands for OpenClaw, Hermes, and Fusion-Agent adapters without modifying host source.
- Create: `tools/agent_runtime_smoke.py`
  - Run install/load/write/search smoke for OpenClaw, Hermes, and Fusion-Agent where host binaries/checkouts exist.
- Tests:
  - `tests/test_retrieval_pipeline.py`
  - `tests/test_retrieval_trace.py`
  - `tests/test_beam_retrieval_replay.py`
  - `tests/test_rule_audit.py`
  - `tests/test_product_cli.py`
  - `tests/test_agent_installer.py`
  - `tests/test_agent_runtime_smoke.py`

---

### Task 1: Solidify Dual/Hybrid Shadow Path While Keeping Legacy Default

**Files:**
- Modify: `fusion_memory/core/runtime_config.py`
- Modify: `fusion_memory/api/service.py`
- Modify: `tests/test_runtime_config.py`
- Modify: `tests/test_fusion_memory.py`

**Interfaces:**
- Consumes: `FUSION_MEMORY_DUAL_EVENT_ORDERING_SHADOW`
- Consumes: `FUSION_MEMORY_EVENT_ORDERING_SELECTOR`
- Produces: `coverage["event_ordering_dual_shadow"]` only when shadow is enabled.
- Preserves: production selector accepts only `legacy`.

- [ ] **Step 1: Add a failing test that graph/dual/hybrid cannot become default**

Add to `tests/test_runtime_config.py`:

```python
def test_non_legacy_event_ordering_selector_is_rejected_for_product_default(self) -> None:
    with patch.dict(os.environ, {"FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "dual"}, clear=True):
        with self.assertRaisesRegex(ValueError, "unsupported event ordering selector"):
            build_runtime_retrieval_flags()
```

- [ ] **Step 2: Add a failing service test that shadow does not replace selected candidates**

Add to `tests/test_fusion_memory.py`:

```python
def test_dual_shadow_does_not_replace_event_ordering_selected_candidates(self) -> None:
    class Flags:
        dual_event_ordering_shadow = True
        production_selector = "legacy"

    service = MemoryService(retrieval_flags=Flags())
    scope = Scope(workspace_id="shadow-default", user_id="u", agent_id="a")
    try:
        service.add({"role": "user", "content": "First I created the schema. Then I added CRUD. Finally I tested errors."}, scope)
        result = service.search(
            "What order did I describe the work?",
            scope,
            {"query_type_hint": "event_ordering", "limit": 5},
        )
    finally:
        service.close()

    self.assertIn("event_ordering_dual_shadow", result.coverage)
    self.assertNotEqual(result.coverage["event_ordering_dual_shadow"].get("selected_driver"), "production")
    self.assertTrue(result.candidates)
```

- [ ] **Step 3: Run red tests**

Run:

```bash
python3 -m unittest \
  tests.test_runtime_config.RuntimeRetrievalFlagTests.test_non_legacy_event_ordering_selector_is_rejected_for_product_default \
  tests.test_fusion_memory.FusionMemoryTests.test_dual_shadow_does_not_replace_event_ordering_selected_candidates
```

Expected: runtime flag test may already pass; service test fails if coverage is missing or mutates selected candidates.

- [ ] **Step 4: Implement minimal shadow-only behavior**

In `fusion_memory/api/service.py`, ensure event_ordering search keeps current selected candidates from the legacy path and only appends a `coverage["event_ordering_dual_shadow"]` object with:

```python
{
    "enabled": True,
    "candidate_count": len(deduped_shadow_candidates),
    "sources": sorted(set(shadow_sources)),
    "selected_driver": "shadow",
}
```

Do not change `result.candidates` ordering or IDs.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_runtime_config tests.test_fusion_memory.FusionMemoryTests.test_dual_shadow_does_not_replace_event_ordering_selected_candidates
```

Expected: PASS.

---

### Task 2: Expand Replay Beyond Event Ordering

**Files:**
- Create: `tools/beam_retrieval_replay.py`
- Create: `tests/test_beam_retrieval_replay.py`
- Modify: `docs/superpowers/plans/2026-06-21-memory-agent-productization.md`

**Interfaces:**
- Produces CLI:
  - `python3 tools/beam_retrieval_replay.py --dataset /public/home/wwb/datasets/BEAM --split 100k --workspace <workspace> --categories current_value,multi_condition,zh_recall --output <path>`
- Produces JSON fields:
  - `summary.categories[category].query_count`
  - `summary.categories[category].coverage_insufficient_rate`
  - `summary.categories[category].mean_source_span_count`
  - `records[].pipeline_trace`

- [ ] **Step 1: Write replay parser/unit tests**

Create `tests/test_beam_retrieval_replay.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.beam_retrieval_replay as replay


class BeamRetrievalReplayTests(unittest.TestCase):
    def test_category_filter_parses_current_multi_and_zh_aliases(self) -> None:
        self.assertEqual(
            replay._parse_categories("current_value,multi_condition,zh_recall"),
            {"current_value", "multi_condition", "zh_recall"},
        )

    def test_record_summary_counts_coverage_and_source_spans(self) -> None:
        records = [
            {"category": "current_value", "source_span_count": 2, "coverage_insufficient": False},
            {"category": "current_value", "source_span_count": 0, "coverage_insufficient": True},
            {"category": "zh_recall", "source_span_count": 3, "coverage_insufficient": False},
        ]

        summary = replay._summarize_records(records)

        self.assertEqual(summary["categories"]["current_value"]["query_count"], 2)
        self.assertEqual(summary["categories"]["current_value"]["coverage_insufficient_rate"], 0.5)
        self.assertEqual(summary["categories"]["current_value"]["mean_source_span_count"], 1.0)
        self.assertEqual(summary["categories"]["zh_recall"]["query_count"], 1)

    def test_run_replay_writes_records_with_pipeline_trace(self) -> None:
        fake_query = SimpleNamespace(id="q1", query="What is my current IDE?", category="knowledge_update")
        fake_pack = SimpleNamespace(
            source_spans=[{"span_id": "s1"}],
            coverage={"coverage_insufficient": False},
            debug_trace=[],
        )
        service = MagicMock()
        service.answer_context.return_value = fake_pack

        with tempfile.TemporaryDirectory() as tmp, patch.object(replay, "_load_queries", return_value=[fake_query]):
            out = Path(tmp) / "replay.json"
            report = replay.run_replay(
                service,
                base_scope=replay.Scope(workspace_id="w", user_id="u", agent_id="a"),
                categories={"current_value"},
                output_path=out,
                query_limit=None,
            )

        self.assertEqual(report["summary"]["categories"]["current_value"]["query_count"], 1)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("pipeline_trace", payload["records"][0])
```

- [ ] **Step 2: Run red tests**

Run:

```bash
python3 -m unittest tests.test_beam_retrieval_replay -v
```

Expected: FAIL because `tools.beam_retrieval_replay` is missing.

- [ ] **Step 3: Implement replay script**

Create `tools/beam_retrieval_replay.py` with helpers:

```python
def _parse_categories(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}
```

Map BEAM categories:

```python
CATEGORY_ALIASES = {
    "current_value": {"knowledge_update", "preference_following", "instruction_following"},
    "multi_condition": {"multi_session_reasoning", "temporal_reasoning"},
    "zh_recall": {"zh_recall"},
}
```

For Chinese recall, include deterministic local probes if BEAM has no Chinese category:

```python
ZH_PROBES = [
    SimpleNamespace(id="zh:1", category="zh_recall", query="我现在使用的数据库是什么？"),
    SimpleNamespace(id="zh:2", category="zh_recall", query="我之前说过偏好的模型是什么？"),
]
```

Each record must contain `query_id`, `category`, `source_span_count`, `coverage_insufficient`, `pipeline_trace`, and `rule_hits` if present in coverage.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_beam_retrieval_replay -v
```

Expected: PASS.

---

### Task 3: Global Rule Audit And First-Pass Cleanup Gate

**Files:**
- Modify: `tools/rule_audit.py`
- Modify: `fusion_memory/retrieval/rule_audit.py`
- Modify: `tests/test_rule_audit.py`

**Interfaces:**
- CLI accepts repeated `--input`.
- Output row includes:
  - `rule_id`
  - `ability`
  - `hit_count`
  - `contribution_count`
  - `negative_impact_count`
  - `cleanup_action`
  - `safe_to_delete`
  - `evidence_inputs`
- Legacy event_ordering rules must always be `safe_to_delete=false`.

- [ ] **Step 1: Add failing multi-input audit test**

Add to `tests/test_rule_audit.py`:

```python
def test_cli_merges_multiple_replay_inputs_and_marks_evidence_inputs(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        first = tmp / "event.json"
        second = tmp / "current.json"
        out = tmp / "audit.json"
        first.write_text(json.dumps({"records": [{"query_id": "q1", "rule_hits": [{"rule_id": "rule.keep", "contributed_candidate_id": "c1"}]}]}), encoding="utf-8")
        second.write_text(json.dumps({"records": [{"query_id": "q2", "rule_hits": [{"rule_id": "rule.drop", "contributed_candidate_id": None}]}]}), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "tools/rule_audit.py", "--input", str(first), "--input", str(second), "--output", str(out)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        rows = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual({row["rule_id"] for row in rows}, {"rule.keep", "rule.drop"})
        self.assertTrue(all(row["evidence_inputs"] for row in rows))
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_rule_audit.RuleAuditTests.test_cli_merges_multiple_replay_inputs_and_marks_evidence_inputs
```

Expected: FAIL because CLI accepts only one `--input`.

- [ ] **Step 3: Implement global audit merge**

In `tools/rule_audit.py`:

```python
parser.add_argument("--input", action="append", required=True, help="Replay JSON input. May be repeated.")
```

Load each input, annotate each record with:

```python
record["_audit_input"] = str(input_path)
```

Add `evidence_inputs` to each row by collecting `_audit_input` from records containing that rule.

- [ ] **Step 4: Preserve legacy shadow protection**

Ensure `_cleanup_classification()` keeps:

```python
if rule_id.startswith("event_ordering.legacy"):
    cleanup_action = "keep_shadow"
    safe_to_delete = False
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_rule_audit -v
```

Expected: PASS.

---

### Task 4: Retrieval Pipeline Execution Layer

**Files:**
- Create: `fusion_memory/retrieval/pipeline.py`
- Modify: `fusion_memory/retrieval/retrieval_trace.py`
- Modify: `fusion_memory/api/service.py`
- Create: `tests/test_retrieval_pipeline.py`
- Modify: `tests/test_retrieval_trace.py`

**Interfaces:**
- Produces dataclasses:
  - `QueryUnderstandingRecord(language: str, intent: str, features: tuple[str, ...])`
  - `CandidateRecallRecord(source_counts: dict[str, int])`
  - `CandidateFusionRecord(selected_sources: tuple[str, ...], dropped_count: int)`
  - `EvidencePackBuilderRecord(source_span_count: int, coverage_insufficient: bool)`
  - `RetrievalPipelineRecord`
- Produces function:
  - `build_pipeline_record(query_type: str, mode: str, *, language: str, intent: str, features: list[str], recalled: list[Candidate], selected: list[Candidate], dropped_count: int, source_span_count: int, coverage_insufficient: bool) -> RetrievalPipelineRecord`

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/test_retrieval_pipeline.py`:

```python
from __future__ import annotations

import unittest

from fusion_memory.core.models import Candidate
from fusion_memory.retrieval.pipeline import build_pipeline_record


class RetrievalPipelineTests(unittest.TestCase):
    def test_build_pipeline_record_counts_sources_without_raw_text(self) -> None:
        recalled = [
            Candidate("c1", "span", "raw secret text", "l0_raw", {"utility_score": 0.8}, ["s1"], {}),
            Candidate("c2", "fact", "another secret", "l3_current_view", {"utility_score": 0.7}, ["s2"], {}),
        ]
        selected = [recalled[1]]

        record = build_pipeline_record(
            "current_value",
            "benchmark",
            language="en",
            intent="current_value",
            features=["current_value"],
            recalled=recalled,
            selected=selected,
            dropped_count=1,
            source_span_count=1,
            coverage_insufficient=False,
        )
        payload = record.to_dict()

        self.assertEqual(payload["pipeline_layers"]["CandidateRecall"]["source_counts"]["l0_raw"], 1)
        self.assertEqual(payload["pipeline_layers"]["CandidateFusion"]["selected_sources"], ["l3_current_view"])
        self.assertNotIn("raw secret text", repr(payload))
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_retrieval_pipeline -v
```

Expected: FAIL because `fusion_memory.retrieval.pipeline` is missing.

- [ ] **Step 3: Implement pipeline dataclasses and serializer**

Create `fusion_memory/retrieval/pipeline.py` with immutable records and `to_dict()` that delegates to `RetrievalTraceBuilder` or returns the same `pipeline_layers` shape. Do not include candidate text, query text, prompt text, or source span content.

- [ ] **Step 4: Wire service trace without changing retrieval output**

In `MemoryService.search()` and `MemoryService.answer_context()`, build a `RetrievalPipelineRecord` after candidate fusion and attach its dictionary into coverage/debug trace under `pipeline_trace`. Preserve existing `RetrievalTraceBuilder` output keys for backward compatibility.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_retrieval_pipeline tests.test_retrieval_trace -v
```

Expected: PASS.

---

### Task 5: Product Installer, Doctor, Error Layer, Defaults, Upgrade

**Files:**
- Modify: `fusion_memory/product.py`
- Modify: `fusion_memory/cli.py`
- Modify: `tests/test_product_cli.py`
- Modify: `docs/quickstart.md`
- Modify: `docs/errors.md`

**Interfaces:**
- `fusion-memory init --json` defaults to Postgres + Qwen.
- `fusion-memory init --local-test --json` explicitly downgrades to SQLite + deterministic/lexical.
- `fusion-memory doctor --json` checks:
  - `postgres_connection`
  - `pgvector`
  - `embedding_dependency`
  - `embedding_readiness`
  - `reranker_dependency`
  - `reranker_readiness`
  - `service`
  - `port`
- `fusion-memory upgrade --dry-run --json` reports backup plan and rollback step.

- [ ] **Step 1: Add failing doctor readiness schema test**

Add to `tests/test_product_cli.py`:

```python
def test_doctor_reports_port_and_model_readiness_with_next_steps(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        init_home(home, port=0)
        report = doctor(home)

    names = {item["name"] for item in report["checks"]}
    self.assertIn("postgres_connection", names)
    self.assertIn("pgvector", names)
    self.assertIn("embedding_readiness", names)
    self.assertIn("reranker_readiness", names)
    self.assertIn("port", names)
    self.assertIn("next_step", report)
    self.assertNotIn("Traceback", json.dumps(report))
```

- [ ] **Step 2: Add failing upgrade rollback test**

Add to `tests/test_product_cli.py`:

```python
def test_upgrade_dry_run_reports_backup_and_rollback(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        init_home(home, local_test=True)
        plan = upgrade(home, dry_run=True)

    self.assertTrue(plan["ok"])
    self.assertTrue(plan["dry_run"])
    self.assertIn("backup", plan)
    self.assertIn("rollback", plan)
```

- [ ] **Step 3: Run red tests**

Run:

```bash
python3 -m unittest \
  tests.test_product_cli.ProductCliTests.test_doctor_reports_port_and_model_readiness_with_next_steps \
  tests.test_product_cli.ProductCliTests.test_upgrade_dry_run_reports_backup_and_rollback
```

Expected: FAIL if `port`, `backup`, or `rollback` fields are missing.

- [ ] **Step 4: Implement product readiness fields**

In `doctor()`, append a separate `port` check:

```python
checks.append(_check("port", _port_available(config["host"], int(config["port"])) or health["ok"], "available" if available else "already in use"))
```

Ensure `_model_checks()` emits both dependency and readiness checks for embedding/reranker.

In `upgrade()`, include:

```python
"backup": {"required": True, "directory": str(paths.backup_dir)},
"rollback": {"available": True, "step": "Restore the latest backup from the backups directory."}
```

- [ ] **Step 5: Run focused product tests**

Run:

```bash
python3 -m unittest tests.test_product_cli -v
```

Expected: PASS.

---

### Task 6: Real Agent Runtime Smoke Harness

**Files:**
- Modify: `fusion_memory/agent_installer.py`
- Create: `tools/agent_runtime_smoke.py`
- Create: `tests/test_agent_runtime_smoke.py`
- Modify: `tests/test_agent_installer.py`
- Modify: `docs/agent-adapters.md`

**Interfaces:**
- CLI:
  - `python3 tools/agent_runtime_smoke.py --target openclaw --memory-url http://127.0.0.1:8765 --output <path>`
  - `python3 tools/agent_runtime_smoke.py --target hermes --memory-url http://127.0.0.1:8765 --output <path>`
  - `python3 tools/agent_runtime_smoke.py --target fusion-agent --memory-url http://127.0.0.1:8765 --output <path>`
- JSON fields:
  - `target`
  - `host_available`
  - `plugin_available`
  - `write_smoke`
  - `retrieve_smoke`
  - `ok`
  - `message`
- Missing host binary/checkouts return `ok=false` with safe actionable message, not traceback.

- [ ] **Step 1: Write failing smoke report test**

Create `tests/test_agent_runtime_smoke.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools.agent_runtime_smoke as smoke


class AgentRuntimeSmokeTests(unittest.TestCase):
    def test_missing_openclaw_host_is_beginner_safe(self) -> None:
        with patch("tools.agent_runtime_smoke.shutil.which", return_value=None):
            report = smoke.run_smoke("openclaw", memory_url="http://127.0.0.1:8765")

        self.assertFalse(report["ok"])
        self.assertFalse(report["host_available"])
        self.assertIn("OpenClaw", report["message"])
        self.assertNotIn("Traceback", json.dumps(report))

    def test_cli_writes_output_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("tools.agent_runtime_smoke.run_smoke", return_value={"ok": True, "target": "hermes"}):
            out = Path(tmp) / "smoke.json"
            code = smoke.main(["--target", "hermes", "--memory-url", "http://127.0.0.1:8765", "--output", str(out)])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["target"], "hermes")
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_agent_runtime_smoke -v
```

Expected: FAIL because `tools.agent_runtime_smoke` is missing.

- [ ] **Step 3: Implement runtime smoke harness**

Create `tools/agent_runtime_smoke.py` with:

```python
VALID_TARGETS = {"openclaw", "hermes", "fusion-agent"}
```

For `openclaw`, check `shutil.which("openclaw")` and plugin directory exists. For `hermes`, check `$HERMES_HOME/plugins/fusion_memory` or repo integration directory. For `fusion-agent`, check `/public/home/wwb/Fusion-Agent` or `$FUSION_AGENT_ROOT`.

If host is unavailable, return safe failure. If available, run only documented smoke commands with timeout and redact stderr to a safe message. Do not modify OpenClaw/Hermes source.

- [ ] **Step 4: Run focused smoke tests**

Run:

```bash
python3 -m unittest tests.test_agent_runtime_smoke tests.test_agent_installer -v
```

Expected: PASS.

---

## Final Verification

After all tasks:

```bash
python3 -m unittest \
  tests.test_runtime_config \
  tests.test_fusion_memory \
  tests.test_retrieval_pipeline \
  tests.test_retrieval_trace \
  tests.test_beam_event_ordering_replay \
  tests.test_beam_retrieval_replay \
  tests.test_rule_registry \
  tests.test_rule_audit \
  tests.test_product_cli \
  tests.test_agent_installer \
  tests.test_agent_runtime_smoke
```

Run current live replay/audit commands when Postgres is available:

```bash
/public/home/wwb/anaconda3/envs/fusion-memory-qwen/bin/python tools/beam_event_ordering_replay.py \
  --dataset /public/home/wwb/datasets/BEAM \
  --split 100k \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --user-id beam_user \
  --agent-id fusion_memory \
  --run-id beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --db postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory \
  --mode all \
  --hybrid-source source_spans \
  --gate \
  --output /public/home/wwb/memory/.runtime/beam-runs/event_ordering_four_path_productization_20260621.json

/public/home/wwb/anaconda3/envs/fusion-memory-qwen/bin/python tools/rule_audit.py \
  --input /public/home/wwb/memory/.runtime/beam-runs/event_ordering_four_path_productization_20260621.json \
  --output /public/home/wwb/memory/.runtime/beam-runs/rule_audit_productization_20260621.json \
  --csv /public/home/wwb/memory/.runtime/beam-runs/rule_audit_productization_20260621.csv
```

Expected:

- Legacy remains production default.
- Graph-only replay is allowed to fail parity.
- Dual/hybrid replay must be present and comparable.
- Rule audit marks legacy event_ordering rules as `safe_to_delete=false`.
- Product CLI JSON errors contain no `Traceback`.
- Agent smoke reports missing hosts as beginner-safe failures.
