# Provider Telemetry Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make recall provider telemetry visible in sanitized retrieval traces and rule audit artifacts so later rule cleanup can be based on provider contribution evidence.

**Architecture:** Keep the provider registry behavior-preserving. Extend `CandidateRecall` trace records with sanitized `provider_summary`, then add provider-level audit rows derived from replay `pipeline_trace.CandidateRecall.provider_summary` records. Do not change ranking, provider order, event ordering defaults, candidate lifecycle semantics, or replay raw-text safety.

**Tech Stack:** Python 3.11+/3.12, `unittest`, existing retrieval pipeline, existing replay JSON format, existing `tools/rule_audit.py` CLI.

## Global Constraints

- Preserve `build_candidate_lists()` public signature and return shape.
- Preserve provider execution order and candidate ordering.
- Legacy event ordering remains the production default.
- Graph, dual, and hybrid paths remain shadow/replay/flag-only.
- Do not delete legacy event ordering code.
- Do not tune ranking, scoring, quotas, MMR, reranking, preservation, filtering, or evidence packing.
- LLM extractor and LLM router remain out of realtime retrieval.
- No raw user text may be stored in provider telemetry, lifecycle trace, rule telemetry, replay artifacts, or audit outputs.
- Product-facing errors remain beginner-safe and must not expose tracebacks.

---

## Task 1: Surface Provider Summary In RetrievalTrace

**Files:**
- Modify: `fusion_memory/retrieval/pipeline.py`
- Modify: `fusion_memory/api/service.py`
- Modify: `tests/test_retrieval_pipeline.py`
- Modify: `tests/test_fusion_memory.py`

**Interfaces:**
- Extend `CandidateRecallRecord` with `provider_summary: tuple[dict[str, object], ...]`.
- Extend `build_pipeline_record(..., provider_summary: list[dict[str, object]] | None = None)`.
- `RetrievalPipelineRecord.pipeline_layers()["CandidateRecall"]` must include `provider_summary` only when provider summary exists.

- [ ] **Step 1: Write failing tests**

Add test coverage proving `build_pipeline_record()` emits provider summary without raw text and `MemoryService.search()` exposes provider summary under `coverage["pipeline_trace"]["pipeline_layers"]["CandidateRecall"]`.

- [ ] **Step 2: Implement trace plumbing**

Pass `recall_result.provider_summary` from `MemoryService.search()` into `build_pipeline_record()`. Sanitize via the existing `_safe_provider_summary()` path before serializing.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_retrieval_pipeline tests.test_fusion_memory.FusionMemoryTests.test_search_trace_contains_retrieval_pipeline_sections -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add fusion_memory/retrieval/pipeline.py fusion_memory/api/service.py tests/test_retrieval_pipeline.py tests/test_fusion_memory.py
git commit -m "feat: surface provider summary in retrieval trace"
```

---

## Task 2: Add Provider Audit Output To Rule Audit CLI

**Files:**
- Modify: `tools/rule_audit.py`
- Modify: `tests/test_rule_audit.py`

**Interfaces:**
- Add `build_provider_audit(records: list[dict[str, object]]) -> list[dict[str, object]]`.
- Provider audit rows must include `provider_id`, `source_family`, `hit_count`, `query_count`, `output_count`, `output_source_counts`, `evidence_inputs`, `production_default`, `shadow_only`, and `graph_related`.
- Add CLI flag `--provider-output PATH` and optional `--provider-csv PATH`.
- Existing `--output` and `--csv` rule audit behavior must remain unchanged.

- [ ] **Step 1: Write failing tests**

Add unit tests for `build_provider_audit()` from sanitized replay records, including nested `coverage.pipeline_trace` and top-level `pipeline_trace` locations. Assert unsafe provider/source strings are hashed and raw query/candidate text is absent.

- [ ] **Step 2: Implement provider audit aggregation**

Read `pipeline_trace.pipeline_layers.CandidateRecall.provider_summary` and aggregate by safe `provider_id`. Sum output counts, merge output source counts, count distinct query ids, and preserve evidence input paths.

- [ ] **Step 3: Implement CLI output flags**

When `--provider-output` is supplied, write provider audit JSON. When `--provider-csv` is supplied, write provider audit CSV. Keep safe beginner CLI error behavior.

- [ ] **Step 4: Verify**

Run:

```bash
python3 -m unittest tests.test_rule_audit -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/rule_audit.py tests/test_rule_audit.py
git commit -m "feat: add provider audit report"
```

---

## Task 3: Phase Gate

**Files:**
- No implementation files expected unless tests reveal a gap.

- [ ] **Step 1: Run focused gate**

```bash
python3 -m unittest \
  tests.test_retrieval_pipeline \
  tests.test_rule_audit \
  tests.test_beam_retrieval_replay \
  tests.test_recall_provider_registry \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run broad adjacent gate**

```bash
python3 -m unittest \
  tests.test_retrieval_pipeline \
  tests.test_rule_audit \
  tests.test_rule_registry \
  tests.test_beam_retrieval_replay \
  tests.test_fusion_memory \
  tests.test_recall_provider_registry \
  -v
```

Expected: PASS.

- [ ] **Step 3: Commit docs if needed**

```bash
git add docs/superpowers/plans/2026-06-21-provider-telemetry-audit.md
git commit -m "docs: plan provider telemetry audit"
```
