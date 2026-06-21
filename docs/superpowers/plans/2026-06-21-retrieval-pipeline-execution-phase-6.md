# Retrieval Pipeline Execution Phase 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current trace-oriented retrieval pipeline into a real execution layer while preserving current search behavior.

**Architecture:** Add execution units around the existing `MemoryService.search()` logic first, then move orchestration through those units without changing candidate selection. The new layer owns structured query understanding, provider recall orchestration, fusion lifecycle bookkeeping, and sanitized trace flushing.

**Tech Stack:** Python dataclasses, existing `unittest` tests, existing retrieval provider registry, existing candidate lifecycle and pipeline trace structures.

## Global Constraints

- Legacy event ordering remains the production default.
- Graph, dual, and hybrid paths remain shadow/replay/flag-only until replay proves parity.
- Do not delete legacy event ordering code in this refactor.
- LLM extractor and LLM router remain out of the realtime main retrieval path.
- No raw user text may be stored in pipeline trace, rule-hit telemetry, replay artifacts, or rule audit.
- Every retrieval behavior change must be measurable with replay artifacts.
- Product-facing errors remain beginner-safe and must not expose tracebacks.
- Existing OpenClaw/Hermes integration remains external; do not modify host source trees.
- `MemoryService.search()` public API, trace ids, and default selected candidates must not change.

---

### Task 1: Query Understanding Execution Unit

**Files:**
- Modify: `fusion_memory/retrieval/pipeline.py`
- Modify: `tests/test_retrieval_pipeline.py`

**Interfaces:**
- Produces: `QueryUnderstandingResult`, `QueryUnderstandingEngine.run(query: str, scope: Scope, options: dict[str, Any], planner: Any) -> QueryUnderstandingResult`
- Consumes: existing planner `plan(query, query_type_hint=...)` and `planner.last_intent_telemetry`

- [ ] **Step 1: Add failing tests**

Add tests that assert `QueryUnderstandingEngine` returns the existing plan, language, features, intent telemetry, and never stores the raw query in its safe dict.

- [ ] **Step 2: Run focused test**

Run: `python3 -m unittest tests.test_retrieval_pipeline.RetrievalPipelineTests.test_query_understanding_engine_sanitizes_raw_query -v`

Expected: FAIL before implementation.

- [ ] **Step 3: Implement minimal engine**

Add frozen dataclass `QueryUnderstandingResult` with fields `plan`, `language`, `intent`, `features`, `intent_telemetry`, `precomputed`. Add `safe_record()` returning only `language`, `intent`, and `features`. Add `QueryUnderstandingEngine.run(...)` that honors `_plan`, `_intent_telemetry`, and `query_type_hint`.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest tests.test_retrieval_pipeline tests.test_retrieval_trace -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add fusion_memory/retrieval/pipeline.py tests/test_retrieval_pipeline.py
git commit -m "feat: add query understanding execution unit"
```

### Task 2: Recall Orchestrator Execution Unit

**Files:**
- Modify: `fusion_memory/retrieval/pipeline.py`
- Modify: `fusion_memory/retrieval/candidate_provider.py`
- Modify: `tests/test_retrieval_pipeline.py`

**Interfaces:**
- Consumes: `QueryUnderstandingResult`
- Produces: `RecallResult(candidate_lists: list[list[Candidate]], recalled_candidates: list[Candidate], provider_summary: list[dict[str, Any]])`
- Produces: `RecallOrchestrator.run(context: RetrievalExecutionContext) -> RecallResult`
- `build_candidate_lists(...)` must continue returning `list[list[Candidate]]` for compatibility.

- [ ] **Step 1: Add failing tests**

Add tests for `RecallOrchestrator.run()` using a fake service/provider path. Assert source counts are structural and raw candidate text is absent from `safe_record()`.

- [ ] **Step 2: Run focused test**

Run: `python3 -m unittest tests.test_retrieval_pipeline.RetrievalPipelineTests.test_recall_orchestrator_returns_sanitized_result -v`

Expected: FAIL before implementation.

- [ ] **Step 3: Implement recall context/result**

Add dataclass `RetrievalExecutionContext` with service, query, scope, options, query_understanding, include_session, per_source_limit, enabled_sources, mode, limit, rerank_top_n. Add `RecallResult.safe_record()` with `source_counts` and optional provider summary that contains no candidate text.

- [ ] **Step 4: Wire candidate provider compatibility**

Refactor `build_candidate_lists()` to instantiate `RecallOrchestrator` and return `result.candidate_lists`. Preserve current enabled-source and include-session behavior.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest tests.test_retrieval_pipeline tests.test_retrieval_trace -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add fusion_memory/retrieval/pipeline.py fusion_memory/retrieval/candidate_provider.py tests/test_retrieval_pipeline.py
git commit -m "feat: add recall orchestrator execution unit"
```

### Task 3: Candidate Fusion Execution Unit

**Files:**
- Modify: `fusion_memory/retrieval/pipeline.py`
- Modify: `fusion_memory/api/service.py`
- Modify: `tests/test_retrieval_pipeline.py`

**Interfaces:**
- Consumes: `RetrievalExecutionContext`, `RecallResult`
- Produces: `FusionResult(fused, scored, quota_result, marked, scored_again, rerank_top_n, mode, limit)`
- Produces: `CandidateFusionEngine.run(context: RetrievalExecutionContext, recall_result: RecallResult) -> FusionResult`

- [ ] **Step 1: Add failing tests**

Add a small deterministic test that verifies `CandidateFusionEngine` preserves existing reciprocal-rank-fusion and quota flow using a fake service with config/quota.

- [ ] **Step 2: Run focused test**

Run: `python3 -m unittest tests.test_retrieval_pipeline.RetrievalPipelineTests.test_candidate_fusion_engine_matches_existing_scoring_shape -v`

Expected: FAIL before implementation.

- [ ] **Step 3: Implement fusion engine**

Move the initial fusion/scoring/quota/marked/scored-again calculation from `MemoryService._search_with_rule_hits()` into `CandidateFusionEngine.run()`. Do not move preservation, topic filters, stale-current filters, selected candidate persistence, or trace saving in this task.

- [ ] **Step 4: Wire search through engine**

Replace the equivalent block in `_search_with_rule_hits()` with `CandidateFusionEngine().run(...)`. Keep variable names compatible with the existing downstream code.

- [ ] **Step 5: Run focused behavior tests**

Run:

```bash
python3 -m unittest \
  tests.test_retrieval_pipeline \
  tests.test_fusion_memory.FusionMemoryTests.test_current_value_query_prioritizes_latest_correction_over_historical_value \
  tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_default_search_does_not_select_graph_candidates \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add fusion_memory/retrieval/pipeline.py fusion_memory/api/service.py tests/test_retrieval_pipeline.py
git commit -m "feat: add candidate fusion execution unit"
```

### Task 4: Trace Recorder And Evidence Pack Assembler

**Files:**
- Modify: `fusion_memory/retrieval/pipeline.py`
- Modify: `fusion_memory/api/service.py`
- Modify: `tests/test_retrieval_pipeline.py`
- Modify: `tests/test_retrieval_trace.py`

**Interfaces:**
- Consumes: `RetrievalExecutionContext`, selected candidates, quota result, lifecycle summary, temporal relation summary
- Produces: `RetrievalTraceRecorder.flush() -> dict[str, Any]`
- Produces: `EvidencePackAssembler.update_pipeline_output(value: Any, source_span_count: int, coverage_insufficient: bool) -> dict[str, Any]`

- [ ] **Step 1: Add failing tests**

Add tests that `RetrievalTraceRecorder.flush()` emits the same top-level keys as `build_pipeline_record(...).to_dict()`, includes `TemporalRelations` when supplied, and excludes raw query/candidate text.

- [ ] **Step 2: Run focused test**

Run: `python3 -m unittest tests.test_retrieval_pipeline.RetrievalPipelineTests.test_retrieval_trace_recorder_flushes_sanitized_pipeline_layers -v`

Expected: FAIL before implementation.

- [ ] **Step 3: Implement recorder/assembler**

Implement `RetrievalTraceRecorder` as a thin wrapper around `RetrievalPipelineRecord`. Implement `EvidencePackAssembler.update_pipeline_output(...)` by delegating to existing `update_pipeline_evidence_output(...)`.

- [ ] **Step 4: Wire search/answer_context**

Replace direct `build_pipeline_record(...)` use in `MemoryService._search_with_rule_hits()` with `RetrievalTraceRecorder`. Replace direct `update_pipeline_evidence_output(...)` use in `answer_context()` with `EvidencePackAssembler`.

- [ ] **Step 5: Run focused trace tests**

Run: `python3 -m unittest tests.test_retrieval_pipeline tests.test_retrieval_trace -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add fusion_memory/retrieval/pipeline.py fusion_memory/api/service.py tests/test_retrieval_pipeline.py tests/test_retrieval_trace.py
git commit -m "feat: add retrieval trace recorder"
```

### Task 5: Search Orchestrator Wiring And Replay Gates

**Files:**
- Modify: `fusion_memory/api/service.py`
- Modify: `tests/test_fusion_memory.py`
- Modify: `tests/test_beam_retrieval_replay.py`
- Modify: `tests/test_beam_event_ordering_replay.py`

**Interfaces:**
- Consumes: all Phase 6 execution units
- Produces: `MemoryService.search()` as a thinner orchestrator that delegates query understanding, recall, initial fusion, trace flushing, and pack trace update.

- [ ] **Step 1: Add focused regression assertions**

Add or update tests that assert default event ordering still excludes graph candidates, dual shadow remains coverage-only, current value latest state wins, Chinese recall retains selected evidence, and pipeline trace sections remain present.

- [ ] **Step 2: Refactor orchestration**

Update `_search_with_rule_hits()` so query understanding, recall, initial fusion, and trace construction use Phase 6 execution units. Keep selection preservation/filtering code in service unless moving it would require broad semantic edits.

- [ ] **Step 3: Run focused Phase 6 gate**

Run:

```bash
python3 -m unittest \
  tests.test_retrieval_pipeline \
  tests.test_retrieval_trace \
  tests.test_fusion_memory.FusionMemoryTests.test_current_value_query_prioritizes_latest_correction_over_historical_value \
  tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_default_search_does_not_select_graph_candidates \
  tests.test_fusion_memory.FusionMemoryTests.test_search_trace_contains_retrieval_pipeline_sections \
  tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_search_pipeline_trace_includes_temporal_relations_layer_for_graph_candidate \
  tests.test_beam_retrieval_replay \
  tests.test_beam_event_ordering_replay \
  -v
```

Expected: PASS.

- [ ] **Step 4: Run broad gate**

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

- [ ] **Step 5: Commit**

Run:

```bash
git add fusion_memory/api/service.py tests/test_fusion_memory.py tests/test_beam_retrieval_replay.py tests/test_beam_event_ordering_replay.py
git commit -m "refactor: route search through retrieval execution units"
```
