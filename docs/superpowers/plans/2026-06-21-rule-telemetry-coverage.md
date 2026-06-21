# Rule Telemetry Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand sanitized rule/provider audit visibility so regex cleanup decisions are evidence-based, without changing retrieval ranking or default graph behavior.

**Architecture:** Keep `RuleRegistry` as the single telemetry contract. Add only structural, sanitized observations for query-token, taxonomy-alias, and provider dimensions so replay/audit can show which rule families are active and whether they contribute.

**Tech Stack:** Python 3 unittest; existing `fusion_memory.retrieval.rule_registry`, `fusion_memory.retrieval.rule_audit`, `fusion_memory.retrieval.pipeline`, BEAM replay tools.

## Global Constraints

- Legacy `event_ordering` remains production default and fallback.
- Graph, dual, and hybrid paths remain shadow/replay/flag-only.
- Do not tune ranking, scoring, quotas, MMR, reranking, preservation, filtering, or evidence packing.
- LLM extractor and LLM router remain out of realtime retrieval.
- Do not modify real OpenClaw/Hermes source trees.
- No raw user text may be stored in provider registry telemetry, lifecycle trace, rule telemetry, replay artifacts, or audit outputs.
- Product-facing errors must remain beginner-safe and must not expose tracebacks.
- This phase may improve audit/readability and add observation-only rule hits; it must not delete behavior rules yet.

---

### Task 1: Safe Provider And Rule Audit Dimensions

**Files:**
- Modify: `fusion_memory/retrieval/rule_registry.py`
- Modify: `fusion_memory/retrieval/rule_audit.py`
- Modify: `tests/test_rule_registry.py`
- Modify: `tests/test_rule_audit.py`

**Interfaces:**
- Consumes: `_is_safe_metadata_string(value, key=None)`, `_is_safe_identifier(value)`, `record_rule_hit(...)`, `build_provider_audit(...)`.
- Produces: readable safe dimensions for existing provider ids/source families such as `raw`, `l0_raw`, `topic_scope_raw`, and `graph`.

- [ ] **Step 1: Write failing tests**

Add tests that prove these existing structural identifiers are not hashed in rule hits and provider audit rows:

```python
def test_record_rule_hit_keeps_raw_and_graph_structural_dimensions(self) -> None:
    hit = record_rule_hit(
        "current_value.stale_history_marker",
        query="user asks current value",
        text="candidate text",
        stage="filter",
        provider_id="raw_span",
        lifecycle_stage="recalled",
        lifecycle_reason="topic_scope_raw",
        metadata={"source_family": "raw", "graph_policy": "graph"},
    )
    self.assertEqual(hit.provider_id, "raw_span")
    self.assertEqual(hit.lifecycle_stage, "recalled")
    self.assertEqual(hit.lifecycle_reason, "topic_scope_raw")
    self.assertEqual(hit.metadata["source_family"], "raw")
    self.assertEqual(hit.metadata["graph_policy"], "graph")
```

Add a provider audit test that verifies `source_family="raw"` remains readable.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m unittest tests.test_rule_registry tests.test_rule_audit -v
```

Expected: at least one new assertion fails because one or more safe structural values are currently hashed.

- [ ] **Step 3: Implement safe allowlist expansion**

Update the safe identifier allowlists in `rule_registry.py` and `rule_audit.py` with only structural tokens already emitted by the retrieval pipeline/provider registry. Do not allow arbitrary strings, whitespace, CJK, raw query text, raw candidate text, source span content, prompts, or metadata values not listed explicitly.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_rule_registry tests.test_rule_audit tests.test_retrieval_pipeline -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/rule_registry.py fusion_memory/retrieval/rule_audit.py tests/test_rule_registry.py tests/test_rule_audit.py
git commit -m "Improve rule audit structural dimensions"
```

### Task 2: Observation-Only Rule Hits For Existing Match Families

**Files:**
- Modify: `fusion_memory/api/service.py`
- Modify: `fusion_memory/api/service_helpers.py`
- Modify: `fusion_memory/retrieval/rule_registry.py`
- Modify: `tests/test_rule_registry.py`

**Interfaces:**
- Consumes: `record_rule_hit(...)` and existing search/answer-context rule hit collection.
- Produces: registered observation-only rule ids for Chinese exact matching, multi-condition matching, and taxonomy alias matching.

- [ ] **Step 1: Write failing tests**

Change the existing tests that currently assert no rule hit for multi-condition and taxonomy alias matches. New expected behavior:

```python
self.assertTrue(any(hit.rule_id == "multi_condition.query_token_match" for hit in hits))
self.assertTrue(any(hit.rule_id == "taxonomy.alias_match" for hit in hits))
```

Keep assertions that no raw Chinese text, raw query text, or raw candidate text appears in emitted hit dictionaries.

- [ ] **Step 2: Register observation-only rules**

Register these rule definitions with `protected=False` and clear categories/abilities:

```python
RuleDefinition(
    rule_id="multi_condition.query_token_match",
    module="fusion_memory.api.service",
    purpose="Observe multi-condition query token matching without changing retrieval behavior.",
    category="multi_condition",
    ability="multi_condition",
)
RuleDefinition(
    rule_id="taxonomy.alias_match",
    module="fusion_memory.api.service_helpers",
    purpose="Observe taxonomy alias matching without changing retrieval behavior.",
    category="taxonomy_candidate",
    ability="zh_recall",
)
RuleDefinition(
    rule_id="zh_recall.cjk_exact_match",
    module="fusion_memory.api.service_helpers",
    purpose="Observe CJK exact phrase preservation without storing the phrase.",
    category="zh_recall",
    ability="zh_recall",
)
```

- [ ] **Step 3: Emit sanitized observation hits**

At the existing match points only, call `record_rule_hit(...)` with:

- `contributed=None` unless the code can identify a selected/contributed candidate without new ranking logic.
- `impact="observed"`.
- structural metadata only, such as `{"decision": "observed", "source": "taxonomy"}`.
- no raw query text in metadata; pass the query through the `query` parameter only so `RuleRegistry` hashes it.
- no raw matched phrase in metadata; if needed, use counts or already-safe labels.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_rule_registry tests.test_rule_audit tests.test_retrieval_pipeline -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/api/service.py fusion_memory/api/service_helpers.py fusion_memory/retrieval/rule_registry.py tests/test_rule_registry.py
git commit -m "Add observation rule telemetry for recall matches"
```

### Task 3: Replay Audit Smoke With Expanded Rule Coverage

**Files:**
- Modify: `docs/superpowers/plans/2026-06-21-rule-telemetry-coverage.md`
- May modify: `tests/test_beam_retrieval_replay.py`
- May modify: `tests/test_beam_event_ordering_replay.py`

**Interfaces:**
- Consumes: replay artifact format already preserving `coverage.rule_hits` and `CandidateRecall.provider_summary`.
- Produces: documented audit smoke commands and a small test if replay sanitization drops the new rule ids.

- [x] **Step 1: Run focused replay-artifact tests**

Run:

```bash
python3 -m unittest tests.test_beam_retrieval_replay tests.test_beam_event_ordering_replay tests.test_rule_audit -v
```

Result: PASS via `python3 -m unittest tests.test_beam_retrieval_replay tests.test_beam_event_ordering_replay tests.test_rule_audit -v` (76 tests).

- [x] **Step 2: Add tests only if needed**

Result: no replay test changes were needed. Existing replay sanitization preserved the new observation-only rule ids and structural metadata well enough for combined rule/provider audit output.

- [x] **Step 3: Run bounded replay audit smoke**

Run the existing bounded replay set for:

- event_ordering, `--mode all --max-queries 10 --gate`
- current_value, `--max-queries 10`
- multi_condition, `--max-queries 10`
- zh_recall, `--max-queries 10`

Then run `tools/rule_audit.py` with `--provider-output`.

Commands run:

```bash
/public/home/wwb/anaconda3/envs/fusion-memory-qwen/bin/python tools/beam_event_ordering_replay.py --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 --split 100k --dataset /public/home/wwb/datasets/BEAM --db postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory --mode all --max-queries 10 --gate --output /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/event_ordering.json
/public/home/wwb/anaconda3/envs/fusion-memory-qwen/bin/python tools/beam_retrieval_replay.py --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 --split 100k --dataset /public/home/wwb/datasets/BEAM --db postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory --categories current_value --max-queries 10 --output /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/current_value.json
/public/home/wwb/anaconda3/envs/fusion-memory-qwen/bin/python tools/beam_retrieval_replay.py --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 --split 100k --dataset /public/home/wwb/datasets/BEAM --db postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory --categories multi_condition --max-queries 10 --output /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/multi_condition.json
/public/home/wwb/anaconda3/envs/fusion-memory-qwen/bin/python tools/beam_retrieval_replay.py --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 --split 100k --dataset /public/home/wwb/datasets/BEAM --db postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory --categories zh_recall --max-queries 10 --output /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/zh_recall.json
/public/home/wwb/anaconda3/envs/fusion-memory-qwen/bin/python tools/rule_audit.py --input /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/event_ordering.json --input /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/current_value.json --input /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/multi_condition.json --input /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/zh_recall.json --output /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/rule_audit.json --csv /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/rule_audit.csv --provider-output /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/provider_audit.json --provider-csv /public/home/wwb/memory/.worktrees/rule-telemetry-coverage/.runtime/task3-smoke/provider_audit.csv
```

Results:

- provider audit row count remains greater than zero.
- rule audit includes the existing protected rules and any newly observed match-family rules that appear in the bounded replay.
- no replay/audit JSON or CSV contains raw query text, raw candidate text, source span content, prompt text, or traceback content.

Observed:

- Provider audit row count was `15`.
- Combined rule audit included `current_value.stale_history_marker`, `event_ordering.legacy_rescue`, `multi_condition.query_token_match`, and `taxonomy.alias_match`.
- `zh_recall.cjk_exact_match` did not appear in this bounded smoke because the replay output for the two built-in `zh_recall` probes did not emit that rule id in these runs.
- Initial `event_ordering.json` smoke exposed plaintext `reference`, `paths.*.items`, and `metrics.aligned` strings.
- Fixed `tools/beam_event_ordering_replay.py` so persisted replay records keep `reference_count`, `reference_hashes`, `item_count`, `item_hashes`, `aligned_count`, and `aligned_hashes` instead of raw ordering text. Metrics and audit-relevant coverage are preserved.
- Re-ran event_ordering smoke to `.runtime/task3-smoke/event_ordering_sanitized_v2.json`; `reference`, path `items`, and `metrics.aligned` are absent from persisted records.
- Checked plaintext markers including `I'm Craig`, `personal budget tracker`, `Flask 2.3.1`, `SQLite 3.39`, `Core functionality`, and `Transaction error handling`; none appeared in the sanitized v2 event_ordering replay artifact.
- Re-ran `tools/rule_audit.py` against sanitized v2 event_ordering plus the bounded retrieval replay artifacts. Provider audit remained at `15` rows and combined rule audit included `current_value.stale_history_marker`, `event_ordering.legacy_rescue`, `multi_condition.query_token_match`, and `taxonomy.alias_match`.

- [x] **Step 4: Commit**

If only documentation changed:

```bash
git add docs/superpowers/plans/2026-06-21-rule-telemetry-coverage.md
git commit -m "Document expanded rule telemetry replay smoke"
```

If test changes were needed, include them in the same commit.
