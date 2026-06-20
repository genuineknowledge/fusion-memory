# Event Ordering Graph-First Chronology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace heuristic-first event ordering with a graph-first chronology layer while preserving the current BEAM baseline during shadow evaluation.

**Architecture:** Build a dedicated chronology graph from stored spans and events, then rank event-ordering candidates from graph structure before any legacy heuristic path. Keep the current `event_ordering_*` modules as fallback and comparison paths until replay shows the graph path is at least as good.

**Tech Stack:** Python, `unittest`, existing `MemoryService`, current retrieval/eval modules, no new runtime dependencies.

## Global Constraints

- Default behavior must remain no-LLM and must not require the real extractor/router.
- Do not expand event-ordering regex families while the graph path is being introduced.
- Preserve the current BEAM baseline; graph-first behavior is additive until shadow replay proves parity or improvement.
- Keep `MemoryService.search()` and `MemoryService.answer_context()` backward compatible.
- Do not modify real OpenClaw/Hermes source repositories.

---

### Task 1: Lock the graph contract with failing tests

**Files:**
- Modify: `tests/test_fusion_memory.py`
- Create: `tests/test_event_ordering_graph.py`

**Interfaces:**
- Consumes: current event-ordering candidate helpers in `fusion_memory/api/service.py` and `fusion_memory/retrieval/event_graph_selection.py`
- Produces: failing tests that define the graph-first contract for chronology nodes, edges, and shadow metadata

- [ ] **Step 1: Write the failing tests**

Add tests that expect:

```python
def test_build_event_chronology_graph_emits_nodes_and_edges():
    graph = build_event_chronology_graph(query, spans, events)
    assert graph.nodes
    assert graph.edges
    assert any(edge.kind in {"before", "after", "updates", "replaces"} for edge in graph.edges)

def test_graph_first_event_selection_prefers_causal_chain_over_label_noise():
    candidates = select_graph_first_event_ordering_candidates(query, spans, events, limit=4)
    assert candidates[0].source.startswith("event_ordering_graph")

def test_event_ordering_search_exposes_shadow_graph_coverage():
    pack = memory.answer_context(query, scope, budget={"limit": 6, "mode": "benchmark"})
    assert "event_ordering_graph" in pack.coverage or "event_ordering_shadow" in pack.coverage
```

Also add a model-adapter test that confirms event-ordering packs can still be built while the graph path is shadowed, not replacing the legacy path yet.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
python3 -m unittest tests.test_event_ordering_graph tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_pack_preserves_event_timeline_graph_candidates -v
```

Expected: failures that show the graph-first API or shadow metadata does not exist yet.

- [ ] **Step 3: Commit the test intent mentally only**

Do not change production code in this task. The only deliverable is the failing contract.

### Task 2: Implement the chronology graph builder and selector

**Files:**
- Create: `fusion_memory/retrieval/event_chronology_graph.py`
- Modify: `fusion_memory/retrieval/event_graph_selection.py`
- Modify: `fusion_memory/retrieval/event_ordering_common.py`
- Modify: `fusion_memory/retrieval/event_ordering_records.py`

**Interfaces:**
- Produces:
  - `ChronologyNode`
  - `ChronologyEdge`
  - `ChronologyGraph`
  - `build_event_chronology_graph(query: str, spans: list[Any], events: list[MemoryEvent]) -> ChronologyGraph`
  - `select_graph_first_event_ordering_candidates(query: str, spans: list[Any], events: list[MemoryEvent], limit: int) -> list[Candidate]`
- Consumes: `MemoryEvent`, `EvidenceSpan`, existing topic/phase helpers, and lexical overlap utilities

- [ ] **Step 1: Write the minimal implementation**

Implement a graph builder that:

```python
@dataclass(frozen=True)
class ChronologyNode:
    node_id: str
    kind: str
    label: str
    timestamp: datetime | None
    source_span_id: str | None
    topic: str | None
    confidence: float

@dataclass(frozen=True)
class ChronologyEdge:
    source_id: str
    target_id: str
    kind: str
    confidence: float

@dataclass(frozen=True)
class ChronologyGraph:
    nodes: list[ChronologyNode]
    edges: list[ChronologyEdge]
    phases: list[str]
    topics: list[str]
```

The first version should infer only high-confidence structure:
- explicit before/after/then markers
- timestamp order
- update/replaces relations
- phase buckets like setup / decision / implementation / debug / validation / release

The selector should rank graph-connected events before legacy heuristic candidates, but keep the old `event_ordering_*` path available as fallback when the graph is sparse.

- [ ] **Step 2: Run the focused tests again**

Run:

```bash
python3 -m unittest tests.test_event_ordering_graph -v
```

Expected: graph contract tests pass; no regression in existing event-ordering tests.

- [ ] **Step 3: Keep the old heuristics as fallback only**

Do not remove any legacy event-ordering modules in this task. The graph path must be additive only.

### Task 3: Wire graph-first chronology into retrieval and model packs

**Files:**
- Modify: `fusion_memory/api/service.py`
- Modify: `fusion_memory/retrieval/candidate_provider.py`
- Modify: `fusion_memory/eval/model_adapters.py`
- Modify: `fusion_memory/retrieval/structured_annotations.py`

**Interfaces:**
- Consumes: `select_graph_first_event_ordering_candidates(...)`
- Produces: event-ordering candidate lists and coverage metadata that expose both graph-first and legacy outputs

- [ ] **Step 1: Add the graph-first branch to candidate construction**

Update the event-ordering branch so the new graph candidates are inserted before the existing episode / timeline / raw facet heuristics. Preserve the old branch as fallback when the graph returns too few nodes.

- [ ] **Step 2: Expose shadow metrics in coverage**

Add metadata that records:
- graph candidate count
- legacy candidate count
- whether the graph path or legacy fallback produced the final selected spans
- whether any graph candidates were dropped by later filters

- [ ] **Step 3: Update model-pack generation**

Make `build_event_ordering_model_pack(...)` consume graph-first chronology data without losing the legacy pack shape expected by the evaluator.

- [ ] **Step 4: Verify the integration tests**

Run:

```bash
python3 -m unittest tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_pack_preserves_event_timeline_graph_candidates tests.test_model_adapters.ModelAdapterTests.test_event_ordering_model_pack_adds_referenceable_episodes -v
```

Expected: both pass with the new graph path visible in coverage.

### Task 4: Shadow eval, replay, and pruning

**Files:**
- Modify: `tests/test_fusion_memory.py`
- Modify: `tests/test_model_adapters.py`
- Modify: `docs/beam-100k-final-evaluation-report-20260617.md` only if a short note is needed
- Modify later only if justified by replay: `fusion_memory/retrieval/event_ordering_labels.py`, `fusion_memory/retrieval/event_ordering_milestones.py`, `fusion_memory/retrieval/event_ordering_sequence.py`, `fusion_memory/retrieval/event_ordering_anchors.py`

**Interfaces:**
- Consumes: graph-first and legacy event-ordering outputs from Task 3
- Produces: a replayable comparison harness and a deletion list for clearly redundant regex helpers

- [ ] **Step 1: Add a shadow replay test**

Create a replay test that runs the same BEAM-style event-ordering cases through both paths and asserts:
- graph path returns the right chronology for the known weak cases
- legacy path still remains available as a fallback
- graph path does not reduce non-event-ordering retrieval quality

- [ ] **Step 2: Run the full retrieval suite**

Run:

```bash
python3 -m unittest tests.test_fusion_memory tests.test_model_adapters tests.test_temporal_normalizer -v
```

Expected: all tests pass, including the new shadow replay coverage.

- [ ] **Step 3: Prune only after parity is proven**

If the shadow replay shows graph parity on the event-ordering BEAM subset, remove only truly redundant project-specific regex helpers and leave the high-precision generic rules:
- explicit first/then/after/before
- date and deadline parsing
- phase markers
- current state / replacement markers

Do not delete the baseline graph fallback until the replay data is stable.

## Self-Review

- Spec coverage: event-ordering graph builder, retrieval integration, model-adapter integration, shadow eval, and safe pruning are each mapped to a task.
- Placeholder scan: no TBD/TODO/implement later markers.
- Type consistency: graph dataclasses and selector signatures are defined before later tasks reference them.
