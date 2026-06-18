# Graph Replay And Rule Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make graph-vs-legacy event-ordering replay reliable, add rule-hit observability across retrieval, produce an auditable rule table, and only then prune or migrate low-risk regex rules.

**Architecture:** First stabilize the replay environment so event-ordering metrics are trustworthy. Then add a lightweight rule registry and telemetry layer that records regex/rule hits, selected evidence contribution, and final drops. Rule cleanup is staged: remove only no-hit/no-contribution/duplicate rules first, migrate domain labels to taxonomy second, and delete legacy event-ordering rules only after graph parity is proven by replay.

**Tech Stack:** Python dataclasses, JSON/CSV audit reports, existing `MemoryService` traces, `tools/beam_event_ordering_replay.py`, `unittest`, Postgres migration runner, existing taxonomy config.

## Global Constraints

- Do not delete legacy event-ordering rules until graph parity is demonstrated by replay.
- Do not add new project-specific or software-specific regex branches.
- Keep graph-vs-legacy replay deterministic and reproducible from a single command.
- Rule telemetry must be low overhead and safe in normal user flows.
- Rule cleanup must be evidence-driven: no-hit, no-contribution, or duplicate rules only in the first pruning pass.
- Domain/tool labels must migrate to `fusion_memory/config/default_taxonomy.json`, not private regex branches.
- Existing dirty worktree changes must not be reverted or silently bundled.
- Every implementation task must use TDD: failing test, red run, minimal implementation, green run, commit.

---

## File Structure

Create:

- `fusion_memory/retrieval/rule_registry.py`: registry dataclasses, decorator/helper APIs, and in-memory telemetry collector.
- `tools/rule_audit.py`: CLI that reads traces/replay output and writes rule audit JSON/CSV.
- `tests/test_rule_registry.py`: registry and telemetry unit tests.
- `tests/test_rule_audit.py`: audit report tests.

Modify:

- `tools/beam_event_ordering_replay.py`: add replay preflight, model-pack isolation, full-run output fields, and stable graph-vs-legacy modes.
- `fusion_memory/retrieval/evidence_pack.py`: register high-volume pack rules and emit rule-hit telemetry for audited regex helpers.
- `fusion_memory/api/service_helpers.py`: register condition/current/exact recall rules and emit telemetry.
- `fusion_memory/retrieval/event_graph_selection.py`: register legacy event-ordering rescue rules and taxonomy hits.
- `fusion_memory/retrieval/event_ordering_pack.py`: register legacy pack rules, without expanding behavior.
- `fusion_memory/retrieval/taxonomy.py`: add helper metadata needed by audit/migration.
- `fusion_memory/config/default_taxonomy.json`: add domain labels migrated during Task 6.
- Focused tests in `tests/test_fusion_memory.py`, `tests/test_model_adapters.py`, and replay tests where needed.

---

### Task 1: Replay Environment Preflight And Isolation

**Files:**
- Modify: `tools/beam_event_ordering_replay.py`
- Test: `tests/test_beam_event_ordering_replay.py`

**Interfaces:**
- Produces:
  - `preflight_replay_environment(args: argparse.Namespace) -> dict[str, object]`
  - CLI flag `--preflight-only`
  - CLI flag `--hybrid-source` with values `model_pack` and `source_spans`
  - Report field `preflight`

- [ ] **Step 1: Write failing preflight test**

Append to `tests/test_beam_event_ordering_replay.py`:

```python
class BeamReplayPreflightTests(unittest.TestCase):
    def test_preflight_reports_postgres_chronology_migration_status(self) -> None:
        class Store:
            def list_chronology_topics(self, scope, include_session=False):
                raise RuntimeError('relation "chronology_topics" does not exist')

        report = preflight_replay_environment_from_store(Store())

        self.assertFalse(report["chronology_tables_ready"])
        self.assertEqual(report["chronology_error"], "missing_chronology_tables")
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_beam_event_ordering_replay.BeamReplayPreflightTests -v
```

Expected: FAIL because `preflight_replay_environment_from_store` is missing.

- [ ] **Step 3: Implement preflight helper**

Add to `tools/beam_event_ordering_replay.py`:

```python
def preflight_replay_environment_from_store(store: Any) -> dict[str, object]:
    try:
        store.list_chronology_topics(Scope(workspace_id="preflight", user_id="preflight", agent_id="preflight"), include_session=True)
    except Exception as exc:
        message = str(exc).lower()
        if "chronology_" in message and ("does not exist" in message or "no such table" in message):
            return {"chronology_tables_ready": False, "chronology_error": "missing_chronology_tables"}
        return {"chronology_tables_ready": False, "chronology_error": type(exc).__name__}
    return {"chronology_tables_ready": True, "chronology_error": None}
```

Add `--preflight-only` and `--hybrid-source` parser options. When `--preflight-only` is passed, write a report containing only `preflight` and exit 0. When `--hybrid-source source_spans` is used, `_hybrid_items()` must not call `_pack_for_model`; it should derive sequence text from `pack.source_spans` and coverage only.

- [ ] **Step 4: Run green tests**

Run:

```bash
python3 -m unittest tests.test_beam_event_ordering_replay -v
python3 -m py_compile tools/beam_event_ordering_replay.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/beam_event_ordering_replay.py tests/test_beam_event_ordering_replay.py
git commit -m "test: add beam replay preflight"
```

---

### Task 2: Full Graph-Vs-Legacy Replay Outputs

**Files:**
- Modify: `tools/beam_event_ordering_replay.py`
- Test: `tests/test_beam_event_ordering_replay.py`

**Interfaces:**
- Consumes: Task 1 `--hybrid-source source_spans`.
- Produces:
  - CLI flag `--mode` with values `graph_only`, `legacy_only`, `hybrid`, `all`
  - Report fields `bucket_summary`, `route_summary`, `replay_config`
  - Per-record field `bucket`

- [ ] **Step 1: Write failing bucket summary test**

Append:

```python
class BeamReplayBucketTests(unittest.TestCase):
    def test_bucket_summary_groups_event_ordering_cases(self) -> None:
        records = [
            {"bucket": "explicit_order", "paths": {"graph": {"metrics": {"f1": 1.0, "kendall_tau_norm": 1.0}}}},
            {"bucket": "explicit_order", "paths": {"graph": {"metrics": {"f1": 0.0, "kendall_tau_norm": 0.5}}}},
            {"bucket": "long_mixed_topic", "paths": {"graph": {"metrics": {"f1": 0.5, "kendall_tau_norm": 0.75}}}},
        ]

        summary = _bucket_summary(records, path="graph")

        self.assertEqual(summary["explicit_order"]["count"], 2)
        self.assertAlmostEqual(summary["explicit_order"]["f1"], 0.5)
        self.assertEqual(summary["long_mixed_topic"]["count"], 1)
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_beam_event_ordering_replay.BeamReplayBucketTests -v
```

Expected: FAIL because `_bucket_summary` is missing.

- [ ] **Step 3: Implement bucket classification and summary**

Add:

```python
def _event_ordering_bucket(query: str, reference: list[str]) -> str:
    lower = query.lower()
    joined = " ".join(reference).lower()
    if re.search(r"\b(?:first|then|after|before|next|later)\b|首先|然后|之后|之前", joined):
        return "explicit_order"
    if re.search(r"\b20\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|月\\s*\\d+\\s*日", joined):
        return "dated"
    if len(reference) >= 5 or re.search(r"\b(?:across|throughout|different aspects|long timeline)\b", lower):
        return "long_mixed_topic"
    if re.search(r"[\u4e00-\u9fff]", query):
        return "chinese"
    return "implicit_order"
```

Add `_bucket_summary(records, path)` that averages `precision`, `recall`, `f1`, and `kendall_tau_norm` by `record["bucket"]`.

- [ ] **Step 4: Wire report output**

Each record should include:

```python
"bucket": _event_ordering_bucket(query.query, reference)
```

The final report should include:

```python
"bucket_summary": {
    "graph": _bucket_summary(records, "graph"),
    "legacy": _bucket_summary(records, "legacy"),
    "hybrid": _bucket_summary(records, "hybrid"),
},
"replay_config": {
    "mode": args.mode,
    "hybrid_source": args.hybrid_source,
    "limit": args.limit,
},
```

- [ ] **Step 5: Run green tests**

Run:

```bash
python3 -m unittest tests.test_beam_event_ordering_replay -v
python3 -m py_compile tools/beam_event_ordering_replay.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/beam_event_ordering_replay.py tests/test_beam_event_ordering_replay.py
git commit -m "feat: add event ordering replay buckets"
```

---

### Task 3: Rule Registry And Hit Telemetry

**Files:**
- Create: `fusion_memory/retrieval/rule_registry.py`
- Test: `tests/test_rule_registry.py`

**Interfaces:**
- Produces:
  - `RuleDefinition(rule_id: str, module: str, purpose: str, category: str, pattern: str | None = None, owner: str = "retrieval")`
  - `RuleHit(rule_id: str, query: str, text_hash: str, contributed_candidate_id: str | None, stage: str, metadata: dict[str, object])`
  - `register_rule(rule: RuleDefinition) -> RuleDefinition`
  - `record_rule_hit(rule_id: str, query: str, text: str, stage: str, contributed_candidate_id: str | None = None, metadata: dict[str, object] | None = None) -> RuleHit`
  - `drain_rule_hits() -> list[RuleHit]`
  - `registered_rules() -> list[RuleDefinition]`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_rule_registry.py`:

```python
from __future__ import annotations

import unittest

from fusion_memory.retrieval.rule_registry import (
    RuleDefinition,
    drain_rule_hits,
    record_rule_hit,
    register_rule,
    registered_rules,
)


class RuleRegistryTests(unittest.TestCase):
    def test_register_rule_and_record_hit_without_raw_text(self) -> None:
        drain_rule_hits()
        rule = register_rule(
            RuleDefinition(
                rule_id="current_value.stale_history_marker",
                module="fusion_memory.retrieval.evidence_pack",
                purpose="avoid stale current-value evidence",
                category="generic",
                pattern="initially|previously",
            )
        )

        hit = record_rule_hit(
            rule.rule_id,
            query="What is current?",
            text="I initially used SQLite.",
            stage="evidence_pack_filter",
            contributed_candidate_id="span_1",
        )

        self.assertIn(rule, registered_rules())
        self.assertEqual(hit.rule_id, rule.rule_id)
        self.assertNotIn("SQLite", hit.text_hash)
        self.assertEqual(drain_rule_hits()[0].contributed_candidate_id, "span_1")
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_rule_registry -v
```

Expected: FAIL because `rule_registry.py` is missing.

- [ ] **Step 3: Implement registry**

Create `fusion_memory/retrieval/rule_registry.py` with dataclasses and module-level registries. Use `hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]` for `text_hash`; never store raw text in `RuleHit`.

- [ ] **Step 4: Run green tests**

Run:

```bash
python3 -m unittest tests.test_rule_registry -v
python3 -m py_compile fusion_memory/retrieval/rule_registry.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/rule_registry.py tests/test_rule_registry.py
git commit -m "feat: add retrieval rule registry"
```

---

### Task 4: Instrument High-Risk Rule Families

**Files:**
- Modify: `fusion_memory/retrieval/evidence_pack.py`
- Modify: `fusion_memory/api/service_helpers.py`
- Modify: `fusion_memory/retrieval/event_graph_selection.py`
- Modify: `fusion_memory/retrieval/event_ordering_pack.py`
- Test: `tests/test_rule_registry.py`

**Interfaces:**
- Consumes: `record_rule_hit(...)`, `register_rule(...)`.
- Produces:
  - Rule IDs for current-value, Chinese exact match, multi-condition, event-ordering legacy rescue, and taxonomy hits.

- [ ] **Step 1: Write failing instrumentation test**

Append:

```python
class RuleInstrumentationTests(unittest.TestCase):
    def test_current_value_stale_filter_records_rule_hit(self) -> None:
        from fusion_memory.retrieval.evidence_pack import _is_stale_historical_current_value_span

        drain_rule_hits()

        self.assertTrue(_is_stale_historical_current_value_span("I initially used SQLite."))
        hits = drain_rule_hits()

        self.assertTrue(any(hit.rule_id == "current_value.stale_history_marker" for hit in hits))
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_rule_registry.RuleInstrumentationTests -v
```

Expected: FAIL because no hit is recorded.

- [ ] **Step 3: Register and emit high-risk hits**

In each target module, register rules at import time. Emit hits only when the rule changes behavior or contributes metadata:

```python
record_rule_hit(
    "current_value.stale_history_marker",
    query="",
    text=text,
    stage="evidence_pack_filter",
    metadata={"decision": "drop_stale_history"},
)
```

For query-dependent helpers, pass the real query. For module-local helpers without query, pass `""`.

- [ ] **Step 4: Attach rule hits to search trace**

In `MemoryService.search()`, after candidate selection and before `save_trace`, drain rule hits and add:

```python
trace["rule_hits"] = [hit.__dict__ for hit in drain_rule_hits()]
```

Do the same in `answer_context()` by preserving rule hits already present in the search trace; do not duplicate raw text.

- [ ] **Step 5: Run green tests**

Run:

```bash
python3 -m unittest tests.test_rule_registry tests.test_retrieval_preservation tests.test_fusion_memory.FusionMemoryTests.test_current_value_query_prioritizes_latest_correction_over_historical_value -v
python3 -m py_compile fusion_memory/retrieval/rule_registry.py fusion_memory/retrieval/evidence_pack.py fusion_memory/api/service_helpers.py fusion_memory/retrieval/event_graph_selection.py fusion_memory/retrieval/event_ordering_pack.py fusion_memory/api/service.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/retrieval/rule_registry.py fusion_memory/retrieval/evidence_pack.py fusion_memory/api/service_helpers.py fusion_memory/retrieval/event_graph_selection.py fusion_memory/retrieval/event_ordering_pack.py fusion_memory/api/service.py tests/test_rule_registry.py
git commit -m "feat: trace retrieval rule hits"
```

---

### Task 5: Rule Audit Report CLI

**Files:**
- Create: `tools/rule_audit.py`
- Test: `tests/test_rule_audit.py`

**Interfaces:**
- Consumes: replay JSON and trace `rule_hits`.
- Produces:
  - `build_rule_audit(records: list[dict[str, object]]) -> list[dict[str, object]]`
  - CLI: `python3 tools/rule_audit.py --input replay.json --output audit.json --csv audit.csv`

- [ ] **Step 1: Write failing audit test**

Create `tests/test_rule_audit.py`:

```python
from __future__ import annotations

import unittest

from tools.rule_audit import build_rule_audit


class RuleAuditTests(unittest.TestCase):
    def test_build_rule_audit_counts_hits_contribution_and_drops(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {"rule_id": "current_value.stale_history_marker", "contributed_candidate_id": "c1", "stage": "filter"}
                ],
                "paths": {"hybrid": {"sources": ["l3_current_view"]}},
                "coverage": {"dropped_high_signal_candidates": [{"candidate_id": "c1"}]},
            },
            {
                "query_id": "q2",
                "rule_hits": [
                    {"rule_id": "current_value.stale_history_marker", "contributed_candidate_id": None, "stage": "filter"}
                ],
                "paths": {"hybrid": {"sources": []}},
                "coverage": {},
            },
        ]

        audit = build_rule_audit(records)

        row = next(item for item in audit if item["rule_id"] == "current_value.stale_history_marker")
        self.assertEqual(row["hit_count"], 2)
        self.assertEqual(row["contribution_count"], 1)
        self.assertEqual(row["dropped_count"], 1)
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_rule_audit -v
```

Expected: FAIL because `tools.rule_audit` is missing.

- [ ] **Step 3: Implement audit builder and CLI**

Implement `build_rule_audit()` to output rows sorted by `rule_id`:

```python
{
    "rule_id": rule_id,
    "hit_count": hit_count,
    "query_count": len(query_ids),
    "contribution_count": contribution_count,
    "dropped_count": dropped_count,
    "candidate_sources": sorted(candidate_sources),
    "recommendation": "keep" | "delete_candidate" | "migrate_to_taxonomy" | "legacy_shadow",
}
```

First-pass recommendation logic:

- `delete_candidate` when `hit_count == 0` or `contribution_count == 0`.
- `migrate_to_taxonomy` when `rule_id` contains `.domain_label` or metadata category is `taxonomy_candidate`.
- `legacy_shadow` when `rule_id` starts with `event_ordering.legacy`.
- `keep` otherwise.

- [ ] **Step 4: Run green tests**

Run:

```bash
python3 -m unittest tests.test_rule_audit -v
python3 -m py_compile tools/rule_audit.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/rule_audit.py tests/test_rule_audit.py
git commit -m "feat: add retrieval rule audit report"
```

---

### Task 6: Produce Full Replay And Rule Audit Artifacts

**Files:**
- Modify: `.git/sdd/progress.md` only for durable run notes
- Output: `.runtime/beam-runs/event_ordering_graph_vs_legacy_full.json`
- Output: `.runtime/beam-runs/rule_audit_event_ordering.json`
- Output: `.runtime/beam-runs/rule_audit_event_ordering.csv`

**Interfaces:**
- Consumes: Tasks 1-5 CLIs.
- Produces: reviewable replay and audit artifacts.

- [ ] **Step 1: Run replay preflight**

Run:

```bash
.runtime/beam-venv/bin/python tools/beam_event_ordering_replay.py \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --split 100k \
  --dataset /public/home/wwb/datasets/BEAM \
  --db postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory \
  --preflight-only \
  --output .runtime/beam-runs/event_ordering_preflight.json
```

Expected: JSON output includes `preflight`.

- [ ] **Step 2: Run full graph-vs-legacy replay**

Run:

```bash
.runtime/beam-venv/bin/python tools/beam_event_ordering_replay.py \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --split 100k \
  --dataset /public/home/wwb/datasets/BEAM \
  --db postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory \
  --mode all \
  --hybrid-source source_spans \
  --gate \
  --output .runtime/beam-runs/event_ordering_graph_vs_legacy_full.json
```

Expected: Command exits 0 and writes `summary`, `bucket_summary`, `records`, and `gate`.

- [ ] **Step 3: Run rule audit**

Run:

```bash
python3 tools/rule_audit.py \
  --input .runtime/beam-runs/event_ordering_graph_vs_legacy_full.json \
  --output .runtime/beam-runs/rule_audit_event_ordering.json \
  --csv .runtime/beam-runs/rule_audit_event_ordering.csv
```

Expected: JSON and CSV exist and include `rule_id`, `hit_count`, `contribution_count`, `dropped_count`, and `recommendation`.

- [ ] **Step 4: Record run summary**

Append to `.git/sdd/progress.md`:

```text
Graph replay full run: date=YYYY-MM-DD; graph_f1=NUMBER_FROM_REPORT; legacy_f1=NUMBER_FROM_REPORT; hybrid_f1=NUMBER_FROM_REPORT; graph_tau=NUMBER_FROM_REPORT; legacy_tau=NUMBER_FROM_REPORT; gate_passed=true_or_false; audit_rows=INTEGER_FROM_AUDIT.
```

- [ ] **Step 5: Commit code only**

Do not commit `.runtime` artifacts. Commit only code/test changes already made in Tasks 1-5 and `.git/sdd/progress.md` if desired by the controller.

---

### Task 7: First-Pass Rule Pruning

**Files:**
- Modify only modules named by `rule_audit_event_ordering.json` with `recommendation == "delete_candidate"`
- Test: focused tests for each touched module

**Interfaces:**
- Consumes: `.runtime/beam-runs/rule_audit_event_ordering.json`.
- Produces: no-hit/no-contribution/duplicate rules removed.

- [ ] **Step 1: Create prune candidate list**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path(".runtime/beam-runs/rule_audit_event_ordering.json").read_text())
for row in rows:
    if row.get("recommendation") == "delete_candidate":
        print(row["rule_id"], row["hit_count"], row["contribution_count"], row["dropped_count"])
PY
```

Expected: A concrete list of rule IDs. If the list is empty, commit no pruning changes and record that no first-pass prune is safe.

- [ ] **Step 2: Remove only safe candidates**

For each candidate, delete the specific registered rule and its behavior only when:

- `hit_count == 0`, or
- `hit_count > 0` and `contribution_count == 0` and `dropped_count == 0`, or
- it is an exact duplicate of another rule with the same hit set and lower contribution.

Do not delete event-ordering legacy fallback rules in this task even if low contribution; mark them `legacy_shadow`.

- [ ] **Step 3: Run focused tests and replay smoke**

Run:

```bash
python3 -m unittest tests.test_rule_registry tests.test_rule_audit tests.test_retrieval_preservation tests.test_chronology_selector -v
.runtime/beam-venv/bin/python tools/beam_event_ordering_replay.py \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --split 100k \
  --dataset /public/home/wwb/datasets/BEAM \
  --db postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory \
  --max-queries 20 \
  --mode all \
  --hybrid-source source_spans \
  --gate \
  --output .runtime/beam-runs/event_ordering_after_prune_smoke.json
```

Expected: Unit tests PASS. Replay smoke writes JSON. Gate may fail, but graph/legacy/hybrid metrics must be present.

- [ ] **Step 4: Commit**

```bash
git add PATHS_PRINTED_BY_GIT_STATUS tests/test_rule_registry.py tests/test_rule_audit.py
git commit -m "chore: prune unused retrieval rules"
```

---

### Task 8: Domain Label Taxonomy Migration

**Files:**
- Modify: `fusion_memory/config/default_taxonomy.json`
- Modify: rule modules listed by audit as `migrate_to_taxonomy`
- Test: `tests/test_chronology_selector.py`
- Test: `tests/test_rule_registry.py`

**Interfaces:**
- Consumes: audit rows with `recommendation == "migrate_to_taxonomy"`.
- Produces: domain label aliases represented by taxonomy entries and referenced by `taxonomy_alias_hits`.

- [ ] **Step 1: Add taxonomy migration test**

Append to `tests/test_chronology_selector.py`:

```python
class TaxonomyMigrationTests(unittest.TestCase):
    def test_taxonomy_covers_domain_labels_used_by_event_ordering_rules(self) -> None:
        entries = load_default_taxonomy()
        hits = taxonomy_alias_hits("Gunicorn worker ports and SQLite schema migrations", entries)

        self.assertIn("gunicorn", hits)
        self.assertIn("sqlite", hits)
        self.assertIn("schema migration", hits)
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_chronology_selector.TaxonomyMigrationTests -v
```

Expected: FAIL until entries are added.

- [ ] **Step 3: Add taxonomy entries**

Add entries like:

```json
{"label": "gunicorn", "aliases": ["Gunicorn", "gunicorn worker"], "tags": ["deployment"], "language": "en"},
{"label": "sqlite", "aliases": ["SQLite", "sqlite"], "tags": ["database"], "language": "en"},
{"label": "schema migration", "aliases": ["schema migration", "schema migrations", "database schema"], "tags": ["database"], "language": "en"}
```

- [ ] **Step 4: Replace migrated domain label checks**

Where audit identifies domain-label regex checks, replace local phrase lists with `taxonomy_alias_hits(text)`. Keep behavior equivalent by checking taxonomy labels:

```python
hits = taxonomy_alias_hits(text)
if {"gunicorn", "sqlite", "schema migration"} & hits:
    ...
```

- [ ] **Step 5: Run green tests**

Run:

```bash
python3 -m unittest tests.test_chronology_selector tests.test_rule_registry -v
python3 -m py_compile fusion_memory/retrieval/taxonomy.py fusion_memory/retrieval/event_graph_selection.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/config/default_taxonomy.json fusion_memory/retrieval/taxonomy.py fusion_memory/retrieval/event_graph_selection.py tests/test_chronology_selector.py tests/test_rule_registry.py
git commit -m "chore: migrate domain labels to taxonomy"
```

---

### Task 9: Graph Parity Gate And Legacy Deletion Decision

**Files:**
- Modify: `.git/sdd/progress.md`
- Optional modify: `fusion_memory/retrieval/event_graph_selection.py`
- Optional modify: `fusion_memory/retrieval/event_ordering_pack.py`

**Interfaces:**
- Consumes: full replay reports before and after pruning/taxonomy migration.
- Produces: explicit decision record: keep legacy, shadow only, or delete selected legacy rules.

- [ ] **Step 1: Compare replay reports**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
before = json.loads(Path(".runtime/beam-runs/event_ordering_graph_vs_legacy_full.json").read_text())
after = json.loads(Path(".runtime/beam-runs/event_ordering_after_prune_smoke.json").read_text())
for name, report in [("before", before), ("after", after)]:
    s = report["summary"]
    print(name, {
        "graph_f1": s["graph"]["f1"],
        "legacy_f1": s["legacy"]["f1"],
        "hybrid_f1": s["hybrid"]["f1"],
        "graph_tau": s["graph"]["kendall_tau_norm"],
        "legacy_tau": s["legacy"]["kendall_tau_norm"],
        "hybrid_tau": s["hybrid"]["kendall_tau_norm"],
        "graph_fallback_rate": s.get("graph_fallback_rate"),
    })
PY
```

- [ ] **Step 2: Apply deletion criteria**

Do not delete legacy rules unless all are true:

- `hybrid.f1 >= legacy.f1`
- `hybrid.kendall_tau_norm >= legacy.kendall_tau_norm`
- graph bucket summary does not regress `long_mixed_topic` or `chinese` by more than `0.02` F1
- graph fallback reasons are explainable and not dominated by `graph_unavailable`
- no increase in `dropped_high_signal_candidate_count`

- [ ] **Step 3: Record decision**

Append one of:

```text
Legacy event_ordering deletion decision: keep legacy fallback; graph parity not reached because REASON_FROM_REPLAY_METRICS.
```

or:

```text
Legacy event_ordering deletion decision: delete selected legacy rules RULE_ID_LIST_FROM_AUDIT; graph parity reached with graph_f1=NUMBER_FROM_REPORT, legacy_f1=NUMBER_FROM_REPORT, hybrid_f1=NUMBER_FROM_REPORT.
```

- [ ] **Step 4: Delete only approved legacy rules**

If deletion criteria pass, remove only the selected legacy rules named in the decision record. If criteria fail, make no deletion commit.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m unittest tests.test_chronology_selector tests.test_event_ordering_graph tests.test_beam_event_ordering_replay -v
python3 -m py_compile fusion_memory/retrieval/event_graph_selection.py fusion_memory/retrieval/event_ordering_pack.py
```

Expected: PASS.

- [ ] **Step 6: Commit decision**

```bash
git add .git/sdd/progress.md fusion_memory/retrieval/event_graph_selection.py fusion_memory/retrieval/event_ordering_pack.py
git commit -m "chore: record event ordering graph parity decision"
```

---

## Final Verification

- [ ] Run focused unit suite:

```bash
python3 -m unittest \
  tests.test_beam_event_ordering_replay \
  tests.test_chronology_selector \
  tests.test_chronology_normalizer \
  tests.test_rule_registry \
  tests.test_rule_audit \
  tests.test_retrieval_preservation \
  -v
```

- [ ] Run compile check:

```bash
python3 -m py_compile \
  tools/beam_event_ordering_replay.py \
  tools/rule_audit.py \
  fusion_memory/retrieval/rule_registry.py \
  fusion_memory/retrieval/taxonomy.py \
  fusion_memory/retrieval/event_graph_selection.py \
  fusion_memory/retrieval/event_ordering_pack.py \
  fusion_memory/retrieval/evidence_pack.py \
  fusion_memory/api/service.py
```

- [ ] Run full replay:

```bash
.runtime/beam-venv/bin/python tools/beam_event_ordering_replay.py \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --split 100k \
  --dataset /public/home/wwb/datasets/BEAM \
  --db postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory \
  --mode all \
  --hybrid-source source_spans \
  --gate \
  --output .runtime/beam-runs/event_ordering_graph_vs_legacy_final.json
```

- [ ] Run audit:

```bash
python3 tools/rule_audit.py \
  --input .runtime/beam-runs/event_ordering_graph_vs_legacy_final.json \
  --output .runtime/beam-runs/rule_audit_final.json \
  --csv .runtime/beam-runs/rule_audit_final.csv
```

- [ ] Run diff check:

```bash
git diff --check
```
