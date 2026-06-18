# Graph-First Chronology And Retrieval Quality Design

## Goal

Move Fusion Memory from a BEAM-shaped, heuristic-heavy event-ordering path to a persistent graph-first chronology layer. The new layer must build structured graph records at write time, use those records for event-ordering queries before legacy selectors, and expose shadow evaluation that proves graph parity before legacy rule pruning.

This work also addresses a broader retrieval quality issue: correct evidence is often present in an early candidate pool but is later dropped by filtering, ranking, or pack compaction. The design adds preservation contracts and dropped-evidence telemetry across event ordering, current values, Chinese recall, and multi-condition recall.

## Current Evidence

The current runtime graph selector is not strong enough to replace legacy event-ordering code.

Recent 40-query BEAM 100k event-ordering replay:

```text
graph   precision=0.0281 recall=0.0467 f1=0.0348 tau_norm=0.1957
legacy  precision=0.0938 recall=0.1507 f1=0.1136 tau_norm=0.2276
hybrid  precision=0.0283 recall=0.0283 f1=0.0283 tau_norm=0.1653
```

The replay output is retrieval-level and does not use an LLM judge. It is still enough to show that the current graph path is infrastructure, not a replacement. The graph is built at query time from `MemoryEvent.description`, which makes actor/action/object, topic, phase, and order evidence too weak.

## Non-Goals

- Do not delete existing `event_ordering_*` modules until shadow replay proves graph parity or improvement.
- Do not add more project-specific or software-specific regex branches to rescue individual BEAM cases.
- Do not make real LLM extractor/router part of the synchronous write/read path.
- Do not solve all retrieval categories in one rewrite. This design adds shared contracts and telemetry, then moves the highest-risk paths onto them incrementally.

## Architecture

The target architecture separates raw evidence, extracted facts, and graph chronology:

```text
add(span)
  -> rule extractor
  -> fact/current-view/profile writers
  -> chronology graph normalizer
  -> graph repository

search(event_ordering)
  -> query planner
  -> graph topic resolver
  -> graph chronology selector
  -> legacy event_ordering fallback and shadow path
  -> preservation/rerank/filter contract
  -> evidence pack
```

The graph layer is persistent. Query-time graph building remains only as a compatibility fallback during migration.

## Data Model

### EventNode

`EventNode` is the first-class chronology unit.

Fields:

- `node_id`
- `scope`: workspace, user, agent, run, session, app
- `actor`: normalized actor, usually `user`, `assistant`, or a named entity
- `action`: normalized verb or action phrase
- `object`: target object or topic object
- `topic_id`
- `phase_id`
- `timestamp`
- `source_span_id`
- `source_turn_id`
- `text`: concise evidence text
- `language`
- `confidence`
- `explicit_order_marker`: normalized marker such as `first`, `then`, `after`, `before`, `later`, `finally`
- `created_at`

### EventEdge

`EventEdge` stores order and lifecycle relations between nodes.

Fields:

- `edge_id`
- `from_node_id`
- `to_node_id`
- `edge_type`: `before`, `after`, `updates`, `replaces`, `causes`, `same_topic_next`
- `evidence_type`: `explicit_marker`, `timestamp`, `source_order`, `task_status`, `phase_order`
- `source_span_ids`
- `confidence`
- `created_at`

### Topic

`Topic` groups nodes that belong to the same user task, project, preference area, or conversation theme.

Fields:

- `topic_id`
- `scope`
- `canonical_label`
- `aliases`
- `language`
- `taxonomy_tags`
- `source_span_ids`
- `confidence`
- `created_at`

### Phase

`Phase` is a topic-scoped chronology grouping.

Fields:

- `phase_id`
- `topic_id`
- `phase_type`: `setup`, `decision`, `implementation`, `debug`, `validation`, `release`, `preference`, `current_state`, `unknown`
- `order_hint`
- `source_span_ids`
- `confidence`
- `created_at`

## Write Path

The write path creates graph structure synchronously with bounded deterministic logic. Any LLM extractor work remains asynchronous and shadow-only until separately validated.

Steps:

1. Insert source span as today.
2. Run existing rule extractor and EncodingGate for facts/events.
3. Normalize each accepted event and high-signal user turn into zero or more `EventNode` rows.
4. Resolve or create topic using lexical overlap, source continuity, explicit entity/topic mentions, and configured taxonomy aliases.
5. Resolve phase using high-precision phase markers and task-state markers.
6. Create edges:
   - explicit order markers produce high-confidence `before`/`after` edges
   - source order within the same topic can produce `same_topic_next`
   - update/replacement language produces `updates`/`replaces`
   - phase order can produce lower-confidence `before` edges when topic is stable
7. Record normalization telemetry for dropped or low-confidence graph candidates.

Graph write failure must not fail user memory writes. It should return a safe warning in trace/debug metadata and continue with existing fact/event storage.

## Query Path

For `event_ordering`, the selector uses graph first:

1. Resolve query topic candidates.
2. Traverse graph nodes by topic and related aliases.
3. Build a candidate timeline with source order, edge order, phase coverage, and deduplication.
4. Generate answer-ready labels from `phase_type + action + object`, not from broad raw snippets.
5. Preserve raw user source spans next to graph labels.
6. If graph coverage is weak, use legacy `event_ordering_*` as fallback.
7. Always run legacy in shadow mode while graph is under evaluation.

Weak graph coverage reasons:

- no topic resolved
- no high-confidence nodes
- no usable edges
- too few source spans
- graph labels do not cover requested item count
- selector confidence below threshold

## Shadow Evaluation

Shadow eval is a gate, not a report-only feature.

The existing replay script should become a maintained evaluator:

```text
tools/beam_event_ordering_replay.py
```

Required outputs:

- graph-only metrics
- legacy-only metrics
- hybrid metrics
- precision, recall, F1, Kendall tau, normalized tau
- per-query winner
- graph empty rate
- graph fallback rate
- dropped high-signal candidate count
- topic drift count
- duplicate label count
- over-abstract label count

Acceptance gate before pruning legacy rules:

- graph F1 >= legacy F1 on BEAM event_ordering replay
- graph normalized Kendall tau >= legacy normalized Kendall tau
- hybrid path >= legacy path
- graph empty rate <= legacy empty rate
- no regression on current-value, Chinese recall, and multi-condition recall test sets

## Retrieval Preservation Contract

All retrieval paths should carry preservation metadata. This is the shared fix for "the right evidence exists in the pool but later disappears."

Candidate metadata fields:

- `must_preserve_reason`: one or more of:
  - `current_value`
  - `explicit_user_preference`
  - `graph_chronology_anchor`
  - `multi_condition_match`
  - `language_exact_match`
  - `source_order_anchor`
- `drop_allowed`: boolean
- `drop_reason`: set only when a high-signal candidate is dropped
- `matched_conditions`: query constraints satisfied by this candidate
- `language_match`: `exact`, `translated`, `unknown`
- `evidence_role`: `answer`, `support`, `context`, `fallback`

Filtering and reranking must respect this contract:

- A must-preserve candidate cannot be removed silently.
- If a budget forces removal, trace must record why and what replaced it.
- Evidence pack must expose `dropped_high_signal_candidates` in debug/coverage metadata.
- Tests should assert preservation behavior for current values, Chinese exact recall, multi-condition recall, and event-ordering graph anchors.

## Chinese Recall

Chinese support should not depend on scattered regex additions.

Design requirements:

- tokenizer and lexical overlap must handle Chinese character n-grams and mixed Chinese/English technical terms
- topic aliases should support Chinese and English labels in the same taxonomy entry
- language match should influence preservation
- Chinese exact phrases from the query should survive topic filters and pack compaction

The first graph implementation only needs deterministic Chinese token handling and exact phrase preservation. LLM translation/query expansion remains optional and off by default.

## Current Value Recall

Current-value queries must prefer current views and valid facts over stale history.

Design requirements:

- current-value candidates get `must_preserve_reason=current_value`
- stale facts can remain as history but cannot outrank the current view for "current/latest/now" queries
- value-history pack should expose superseded evidence separately
- filters must record when stale candidates are removed

## Multi-Condition Recall

Multi-condition recall fails when independent filters each drop part of the answer.

Design requirements:

- planner extracts required conditions into `QueryPlan.conditions`
- candidates record `matched_conditions`
- selector preserves a small set that jointly covers all conditions
- pack coverage reports missing conditions
- tests include cases where no single span matches all conditions but a small evidence set does

## Rule Governance

Rules are allowed only when their ownership is clear.

Keep in code:

- explicit order markers: `first`, `then`, `next`, `after`, `before`, `later`, `finally`, and Chinese equivalents
- date/deadline parsing
- update/replacement markers
- task status markers
- generic phase markers

Move to taxonomy config:

- framework/tool/domain terms such as Flask, Render, CRUD, deployment, probability, permutations, combinations
- project labels and aliases
- common bilingual topic aliases

Delete after parity:

- one-off BEAM rescue rules
- project/software-specific label cleanup that duplicates taxonomy
- duplicate private regex branches across `event_ordering_*`, model pack formatting, and service preserve helpers

## Migration Plan

Phase 1: Schema and repositories.

- Add graph dataclasses.
- Add SQLite migrations.
- Add Postgres migrations.
- Add repositories for topics, phases, event nodes, and event edges.
- Add tests for round-trip storage and migration compatibility.

Phase 2: Write-time graph construction.

- Add deterministic graph normalizer.
- Integrate after accepted event creation.
- Add graph write telemetry.
- Keep failures non-fatal.

Phase 3: Graph selector.

- Replace query-time graph construction with persisted graph reads.
- Return graph candidates with source spans and answer-ready labels.
- Keep legacy fallback and shadow outputs.

Phase 4: Evaluation gate.

- Extend BEAM replay to include graph diagnostics.
- Add Chinese, current-value, and multi-condition replay fixtures.
- Add pass/fail thresholds.

Phase 5: Preservation contract.

- Add candidate preservation metadata.
- Add dropped-candidate telemetry.
- Update filters/rerank/pack compaction to respect the contract.

Phase 6: Rule migration and pruning.

- Add taxonomy config.
- Move domain rules into taxonomy.
- Delete rules only after graph and hybrid meet gates.

## Test Strategy

Required tests:

- Graph schema migration tests for SQLite and Postgres.
- Graph repository round-trip tests.
- Write-path tests for actor/action/object/topic/phase/order marker extraction.
- Graph selector tests for topic-scoped chronology.
- Shadow replay tests for graph/legacy/hybrid comparison output.
- Preservation tests that prove high-signal candidates are not silently dropped.
- Chinese recall tests with exact Chinese phrases and mixed Chinese/English terms.
- Current-value tests where stale historical facts exist.
- Multi-condition tests with distributed evidence.

## Rollout

The feature should ship behind internal config flags:

```json
{
  "chronology_graph": {
    "write_enabled": true,
    "read_enabled": false,
    "shadow_enabled": true
  }
}
```

Rollout order:

1. Write graph in shadow mode.
2. Compare graph and legacy on replay.
3. Enable graph read for event-ordering shadow-only output.
4. Enable graph-first with legacy fallback.
5. Prune legacy rules after gates pass.

## Open Questions

- Should taxonomy config live in repo defaults, user config, or both?
- Should graph topics merge across sessions by default or require explicit high confidence?
- Should graph selector use only deterministic labels first, or allow asynchronous LLM label proposals after quality evaluation?
- What exact non-BEAM long-term conversation set should become the product regression suite?
