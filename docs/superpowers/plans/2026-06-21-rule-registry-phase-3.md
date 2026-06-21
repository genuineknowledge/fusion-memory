# Rule Registry Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make retrieval rules auditable by protected status, duplicate relationship, provider/lifecycle dimensions, and first-pass cleanup safety without changing production retrieval behavior.

**Architecture:** Keep runtime retrieval unchanged. Extend `RuleDefinition` and audit rows with structural governance fields, derive provider/lifecycle dimensions from sanitized replay records, and add a cleanup gate that classifies only evidence-backed, unprotected rules as first-pass deletion candidates. No raw query text or candidate text may enter telemetry or audit output.

**Tech Stack:** Python 3.11+/3.12, `unittest`, existing `RuleDefinition` / `RuleHit`, existing `tools/rule_audit.py`, Phase 1 candidate lifecycle, Phase 2 recall provider registry.

## Global Constraints

- Legacy event ordering remains the production default.
- Graph, dual, and hybrid paths remain shadow/replay/flag-only until replay proves parity.
- Do not delete legacy event ordering code in this refactor.
- LLM extractor and LLM router remain out of the realtime main retrieval path.
- No raw user text may be stored in pipeline trace, rule-hit telemetry, replay artifacts, or rule audit.
- Every retrieval behavior change must be measurable with replay artifacts.
- Product-facing errors remain beginner-safe and must not expose tracebacks.
- Existing OpenClaw/Hermes integration remains external; do not modify host source trees.
- First-pass cleanup may only mark zero-hit, zero-contribution, or exact duplicate unprotected rules as safe to delete.
- Domain label regex rules must be marked `migrate_to_taxonomy`, not safe to delete.

---

## File Structure

- Modify: `fusion_memory/retrieval/rule_registry.py`
  - Add protected governance fields to `RuleDefinition`.
  - Add optional provider/lifecycle fields to sanitized `RuleHit`.
- Modify: `fusion_memory/retrieval/rule_audit.py`
  - Respect protected and duplicate declarations when classifying cleanup.
- Modify: `tools/rule_audit.py`
  - Merge top-level, coverage, pipeline trace, and candidate lifecycle dimensions into audit rows.
  - Emit provider/lifecycle columns in JSON and CSV.
- Modify: rule registrations in:
  - `fusion_memory/api/service.py`
  - `fusion_memory/api/service_helpers.py`
  - `fusion_memory/retrieval/evidence_pack.py`
  - `fusion_memory/retrieval/event_ordering_pack.py`
- Modify: tests:
  - `tests/test_rule_registry.py`
  - `tests/test_rule_audit.py`
  - `tests/test_beam_retrieval_replay.py` only if replay sanitization needs a regression guard.

---

### Task 1: Rule Governance Declarations

**Files:**
- Modify: `fusion_memory/retrieval/rule_registry.py`
- Modify: `fusion_memory/api/service.py`
- Modify: `fusion_memory/api/service_helpers.py`
- Modify: `fusion_memory/retrieval/evidence_pack.py`
- Modify: `fusion_memory/retrieval/event_ordering_pack.py`
- Modify: `tests/test_rule_registry.py`

**Interfaces:**
- Extends: `RuleDefinition(rule_id, module, purpose, category, pattern=None, owner="retrieval", ability="general", protected=False, protected_reason="", duplicate_of=None)`
- Extends: `RuleHit` with optional sanitized `provider_id`, `lifecycle_stage`, and `lifecycle_reason`.
- Preserves: existing positional `RuleHit(...)` and `RuleDefinition(...)` constructor compatibility.

- [ ] **Step 1: Write failing governance tests**

Add to `tests/test_rule_registry.py`:

```python
def test_rule_definition_declares_protection_and_duplicates(self) -> None:
    protected = RuleDefinition(
        rule_id="current_value.stale_history_marker",
        module="m",
        purpose="drop stale current-value history",
        category="high_risk",
        ability="current_value",
        protected=True,
        protected_reason="high_precision_current_value",
    )
    duplicate = RuleDefinition(
        rule_id="current_value.stale_history_marker.cn_alias",
        module="m",
        purpose="duplicate Chinese alias",
        category="current_value",
        duplicate_of="current_value.stale_history_marker",
    )

    self.assertTrue(protected.protected)
    self.assertEqual(protected.protected_reason, "high_precision_current_value")
    self.assertEqual(duplicate.duplicate_of, "current_value.stale_history_marker")
```

Add:

```python
def test_record_rule_hit_accepts_sanitized_provider_and_lifecycle_dimensions(self) -> None:
    hit = record_rule_hit(
        "current_value.stale_history_marker",
        query="What is my current database?",
        text="I now use PostgreSQL.",
        stage="evidence_pack_filter",
        provider_id="l3_current_view",
        lifecycle_stage="selected",
        lifecycle_reason="views",
        metadata={"note": "I now use PostgreSQL."},
    )

    self.assertEqual(hit.provider_id, "l3_current_view")
    self.assertEqual(hit.lifecycle_stage, "selected")
    self.assertEqual(hit.lifecycle_reason, "views")
    self.assertRegex(str(hit.metadata["note"]), r"^[0-9a-f]{12}$")
    self.assertNotIn("PostgreSQL", repr(hit.metadata))
```

- [ ] **Step 2: Run red tests**

Run:

```bash
python3 -m unittest \
  tests.test_rule_registry.RuleRegistryTests.test_rule_definition_declares_protection_and_duplicates \
  tests.test_rule_registry.RuleRegistryTests.test_record_rule_hit_accepts_sanitized_provider_and_lifecycle_dimensions \
  -v
```

Expected: FAIL because the new dataclass fields and `record_rule_hit()` keyword arguments do not exist.

- [ ] **Step 3: Implement governance fields**

In `fusion_memory/retrieval/rule_registry.py`, extend dataclasses:

```python
@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    module: str
    purpose: str
    category: str
    pattern: str | None = None
    owner: str = "retrieval"
    ability: str = "general"
    protected: bool = False
    protected_reason: str = ""
    duplicate_of: str | None = None
```

```python
@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    query: str
    text_hash: str
    contributed_candidate_id: str | None
    stage: str
    metadata: dict[str, object] = field(default_factory=dict)
    contributed: bool | None = None
    impact: str = "observed"
    provider_id: str | None = None
    lifecycle_stage: str | None = None
    lifecycle_reason: str | None = None
```

Extend `record_rule_hit()` with keyword-only arguments:

```python
    provider_id: str | None = None,
    lifecycle_stage: str | None = None,
    lifecycle_reason: str | None = None,
```

Sanitize these fields with a helper that only allows compact identifiers:

```python
def _safe_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if _is_safe_metadata_string(text) else _hash_metadata_value(text)
```

- [ ] **Step 4: Mark protected rules**

Update existing `RuleDefinition(...)` registrations:

```python
RuleDefinition(
    rule_id="event_ordering.legacy_rescue",
    ...,
    ability="event_ordering",
    protected=True,
    protected_reason="legacy_event_ordering_fallback",
)
```

```python
RuleDefinition(
    rule_id="current_value.stale_history_marker",
    ...,
    ability="current_value",
    protected=True,
    protected_reason="high_precision_current_value",
)
```

```python
RuleDefinition(
    rule_id="exact_match.cjk_phrase",
    ...,
    ability="zh_recall",
    protected=True,
    protected_reason="chinese_recall_precision",
)
```

- [ ] **Step 5: Run green tests and commit**

Run:

```bash
python3 -m unittest tests.test_rule_registry -v
```

Expected: PASS.

Commit:

```bash
git add fusion_memory/retrieval/rule_registry.py fusion_memory/api/service.py fusion_memory/api/service_helpers.py fusion_memory/retrieval/evidence_pack.py fusion_memory/retrieval/event_ordering_pack.py tests/test_rule_registry.py
git commit -m "feat: add rule governance declarations"
```

---

### Task 2: Provider And Lifecycle Audit Dimensions

**Files:**
- Modify: `tools/rule_audit.py`
- Modify: `fusion_memory/retrieval/rule_audit.py`
- Modify: `tests/test_rule_audit.py`

**Interfaces:**
- Produces audit row fields:
  - `provider_ids: list[str]`
  - `lifecycle_stages: list[str]`
  - `lifecycle_reasons: list[str]`
  - `protected: bool`
  - `protected_reason: str`
- Consumes replay record fields:
  - `rule_hits`
  - `coverage.rule_hits`
  - `candidate_lifecycle.records`
  - `coverage.candidate_lifecycle.records`
  - `pipeline_trace.candidate_lifecycle.records`

- [ ] **Step 1: Write failing audit dimension tests**

Add to `tests/test_rule_audit.py`:

```python
def test_build_rule_audit_includes_provider_and_lifecycle_dimensions(self) -> None:
    records = [
        {
            "query_id": "q1",
            "rule_hits": [
                {
                    "rule_id": "current_value.stale_history_marker",
                    "contributed_candidate_id": "c1",
                    "provider_id": "views",
                    "lifecycle_stage": "selected",
                    "lifecycle_reason": "views",
                    "impact": "selected",
                }
            ],
            "candidate_lifecycle": {
                "records": [
                    {
                        "candidate_id": "c1",
                        "candidate_source": "l3_current_view",
                        "stage": "selected",
                        "reason_code": "views",
                    }
                ]
            },
        }
    ]

    audit = build_rule_audit(records)
    row = next(item for item in audit if item["rule_id"] == "current_value.stale_history_marker")

    self.assertEqual(row["provider_ids"], ["views"])
    self.assertEqual(row["lifecycle_stages"], ["selected"])
    self.assertEqual(row["lifecycle_reasons"], ["views"])
```

Add:

```python
def test_cli_csv_includes_rule_governance_columns(self) -> None:
    payload = {
        "records": [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "event_ordering.legacy_rescue",
                        "provider_id": "event_ordering_coverage",
                        "lifecycle_stage": "rescued",
                        "lifecycle_reason": "event_ordering_coverage",
                    }
                ],
            }
        ]
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "replay.json"
        output_path = tmp / "audit.json"
        csv_path = tmp / "audit.csv"
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                "tools/rule_audit.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--csv",
                str(csv_path),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertIn("provider_ids", reader.fieldnames)
            self.assertIn("lifecycle_stages", reader.fieldnames)
            self.assertIn("lifecycle_reasons", reader.fieldnames)
            self.assertIn("protected", reader.fieldnames)
            self.assertIn("protected_reason", reader.fieldnames)
```

- [ ] **Step 2: Run red tests**

Run:

```bash
python3 -m unittest \
  tests.test_rule_audit.RuleAuditTests.test_build_rule_audit_includes_provider_and_lifecycle_dimensions \
  tests.test_rule_audit.RuleAuditTests.test_cli_csv_includes_rule_governance_columns \
  -v
```

Expected: FAIL because audit rows and CSV do not expose these fields.

- [ ] **Step 3: Implement dimension extraction**

In `tools/rule_audit.py`, add helpers:

```python
def _lifecycle_records_for_record(record: dict[str, object]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for container in (
        _as_dict(record.get("candidate_lifecycle")),
        _as_dict(_as_dict(record.get("coverage")).get("candidate_lifecycle")),
        _as_dict(_as_dict(record.get("pipeline_trace")).get("candidate_lifecycle")),
    ):
        for item in _as_list(container.get("records")):
            if isinstance(item, dict):
                records.append(item)
    return records
```

During row accumulation:

```python
row["provider_ids"] = set()
row["lifecycle_stages"] = set()
row["lifecycle_reasons"] = set()
```

For every hit, add safe string values from hit keys `provider_id`, `lifecycle_stage`, and `lifecycle_reason`. If `contributed_candidate_id` matches a lifecycle `candidate_id`, also add that lifecycle record's `stage` and `reason_code`.

- [ ] **Step 4: Add governance definitions to audit classification**

In `fusion_memory/retrieval/rule_audit.py`, include `protected`, `protected_reason`, and `duplicate_of` from `RuleDefinition` in each row. A protected rule must never be `safe_to_delete=True`.

- [ ] **Step 5: Run green tests and commit**

Run:

```bash
python3 -m unittest tests.test_rule_audit tests.test_rule_registry -v
```

Expected: PASS.

Commit:

```bash
git add tools/rule_audit.py fusion_memory/retrieval/rule_audit.py tests/test_rule_audit.py
git commit -m "feat: add rule audit provider lifecycle dimensions"
```

---

### Task 3: First-Pass Cleanup Gate

**Files:**
- Modify: `fusion_memory/retrieval/rule_audit.py`
- Modify: `tools/rule_audit.py`
- Modify: `tests/test_rule_audit.py`

**Interfaces:**
- Produces cleanup fields:
  - `cleanup_phase`
  - `cleanup_action`
  - `safe_to_delete`
  - `cleanup_blockers: list[str]`
- Preserves protected rule behavior:
  - legacy event ordering fallback rules are not safe to delete
  - high-precision current-value stale-history rules are not safe to delete
  - high-precision explicit temporal marker rules are not safe to delete
  - safety/error guidance rules are not safe to delete
  - domain label regex rules are `migrate_to_taxonomy`, not delete

- [ ] **Step 1: Write failing cleanup gate tests**

Add to `tests/test_rule_audit.py`:

```python
def test_cleanup_gate_blocks_protected_zero_contribution_rules(self) -> None:
    records = [
        {
            "query_id": "q1",
            "rule_hits": [
                {
                    "rule_id": "current_value.stale_history_marker",
                    "contributed_candidate_id": None,
                    "protected": True,
                    "protected_reason": "high_precision_current_value",
                }
            ],
        }
    ]

    audit = build_rule_audit(records)
    row = next(item for item in audit if item["rule_id"] == "current_value.stale_history_marker")

    self.assertFalse(row["safe_to_delete"])
    self.assertEqual(row["cleanup_action"], "keep_protected")
    self.assertEqual(row["cleanup_blockers"], ["protected:high_precision_current_value"])
```

Add:

```python
def test_cleanup_gate_marks_exact_duplicate_unprotected_rule_safe_to_delete(self) -> None:
    records = [
        {
            "query_id": "q1",
            "rule_hits": [
                {
                    "rule_id": "rule.alpha_duplicate",
                    "contributed_candidate_id": None,
                    "metadata": {"duplicate_of": "rule.alpha"},
                }
            ],
        }
    ]

    audit = build_rule_audit(records)
    row = next(item for item in audit if item["rule_id"] == "rule.alpha_duplicate")

    self.assertEqual(row["cleanup_action"], "delete_duplicate")
    self.assertEqual(row["cleanup_phase"], "first_pass")
    self.assertTrue(row["safe_to_delete"])
```

- [ ] **Step 2: Run red tests**

Run:

```bash
python3 -m unittest \
  tests.test_rule_audit.RuleAuditTests.test_cleanup_gate_blocks_protected_zero_contribution_rules \
  tests.test_rule_audit.RuleAuditTests.test_cleanup_gate_marks_exact_duplicate_unprotected_rule_safe_to_delete \
  -v
```

Expected: first test FAIL because protected cleanup blockers are not present yet.

- [ ] **Step 3: Implement cleanup blockers**

In both audit paths, classify in this order:

```python
if protected:
    cleanup_action = "keep_protected"
elif rule_id.startswith("event_ordering.legacy"):
    cleanup_action = "keep_shadow"
elif duplicate_of is not None:
    cleanup_action = "delete_duplicate"
elif domain_label_or_taxonomy:
    cleanup_action = "migrate_to_taxonomy"
elif hit_count == 0:
    cleanup_action = "delete_no_hits"
elif contribution_count == 0:
    cleanup_action = "delete_no_contribution"
else:
    cleanup_action = "keep"
```

`safe_to_delete` is true only for `delete_*`. `cleanup_blockers` contains `protected:<protected_reason>` for protected rows and `domain_label_taxonomy_migration_required` for taxonomy migrations.

- [ ] **Step 4: Run cleanup/audit tests**

Run:

```bash
python3 -m unittest tests.test_rule_audit tests.test_rule_registry tests.test_beam_retrieval_replay -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/rule_audit.py tools/rule_audit.py tests/test_rule_audit.py
git commit -m "feat: enforce first pass rule cleanup gate"
```

---

### Task 4: Phase 3 Verification Gate

**Files:**
- Modify only if tests expose issues.

**Interfaces:**
- Verifies rule audit enforcement does not change retrieval defaults.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m unittest \
  tests.test_rule_registry \
  tests.test_rule_audit \
  tests.test_beam_retrieval_replay \
  tests.test_retrieval_pipeline \
  tests.test_fusion_memory.FusionMemoryTests.test_current_value_query_prioritizes_latest_correction_over_historical_value \
  tests.test_fusion_memory.FusionMemoryTests.test_chinese_error_query_recalls_traceback_guidance \
  tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_default_search_does_not_select_graph_candidates \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run broad regression gate**

Run:

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
  tests.test_config_and_reporting \
  tests.test_authorizer \
  tests.test_product_cli \
  tests.test_agent_installer \
  tests.test_agent_runtime_smoke \
  tests.test_event_ordering_graph \
  tests.test_chronology_selector \
  -v
```

Expected: PASS.

- [ ] **Step 3: Commit only if verification required fixes**

If no files changed, do not create an empty commit. Record verification in the SDD progress ledger.

---

## Later Phase Notes

After Phase 3 passes:

- Phase 4 should introduce shared temporal relation objects in shadow mode.
- Phase 5 should continue graph topic clustering and dual graph-order + legacy-recall shadow evaluation.
- Phase 6 should move `MemoryService.search()` orchestration into real pipeline execution units after provider registry and audit governance have stabilized.
