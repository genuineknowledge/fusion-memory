# BEAM 100K Final Evaluation Report

Date: 2026-06-17

This report records the final two full BEAM 100K validations kept after cleanup, compares them with True Memory Pro, and summarizes the current operator state for later project review.

## Result Summary

Target baseline: True Memory Pro on BEAM 100K = `0.766`.

Both retained full-run validations exceed the target. The observed current-code band is `0.7676505254168324` to `0.7751916960517531`. The second run is closer to the target, so future work should preserve this baseline before making broad retrieval or answer-layer changes.

| System / Run | BEAM 100K Accuracy | Delta vs True Memory Pro 0.766 | Answer Match Rate | Answer Failures | Judge Failures | Status |
|---|---:|---:|---:|---:|---:|---|
| True Memory Pro baseline | `0.7660000000` | `0.0000000000` | N/A | N/A | N/A | Reference |
| Fusion Memory full run 1: `full_after_temporal_gate_w24_20260617` | `0.7751916961` | `+0.0091916961` | `0.8325` | `0` | `0` | Pass |
| Fusion Memory full run 2: `full_after_temporal_gate_repeat1_w24_20260617` | `0.7676505254` | `+0.0016505254` | `0.8200` | `0` | `0` | Pass |

Retained result files:

- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_w24_20260617.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_w24_20260617.diagnostic.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_w24_20260617.partials/`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_repeat1_w24_20260617.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_repeat1_w24_20260617.diagnostic.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_repeat1_w24_20260617.partials/`

## Category Scores

| Category | Run 1 Accuracy | Run 2 Accuracy | Direction / Notes |
|---|---:|---:|---|
| abstention | `1.0000` | `0.9875` | Strong; no current priority. |
| contradiction_resolution | `0.76875` | `0.75625` | Near target but still below `0.8`; remaining work is claim-slot grounding, not answer wording. |
| event_ordering | `0.2076758891` | `0.2067433494` | Main unresolved hard category. |
| information_extraction | `0.7828125` | `0.8041666667` | Borderline to strong; exact answer operators are promising but should stay narrowly gated. |
| instruction_following | `0.9125` | `0.8875` | Strong; typed requirements helped. |
| knowledge_update | `0.8125` | `0.8000` | Above `0.8` band; value/state operators are useful. |
| multi_session_reasoning | `0.7869345238` | `0.7719345238` | Slightly below `0.8`; role/scope count/list operators remain next-step candidates. |
| preference_following | `0.9375` | `0.95625` | Strong; keep current constraint/checklist direction. |
| summarization | `0.7182440476` | `0.7061607143` | Still low; coverage matrix helped but broad summaries miss rubric facets. |
| temporal_reasoning | `0.8250` | `0.8000` | Restored from instability; deterministic temporal consumer is validated but still variance-sensitive. |

## Runtime Cleanup

Historical benchmark artifacts were removed from `.runtime/beam-runs` after recording this report. The runtime directory now keeps only the two final full-run outputs, diagnostics, and partial worker logs listed above.

Old probes, targeted runs, category runs, stale full-runs, and experimental partials were intentionally deleted to reduce noise and prevent future reviews from treating obsolete runs as current evidence.

## Current Operator State

The current high-level direction is acceptable: the final gains came from moving repeated diagnostics into typed pack/model-view structures rather than adding direct qid/gold-answer branches. There is still architecture debt in the answer adapter, and future work should avoid expanding answer templates or named scenario regexes.

### Validated / Useful Operators

| Operator Family | Main Files / Entry Points | Current State |
|---|---|---|
| Temporal endpoint pairs and deterministic temporal consumer | `fusion_memory/retrieval/temporal_pack.py`, `fusion_memory/eval/model_adapters.py` | Validated. The final temporal gate added score-margin tolerance, generic direct duration pair handling, and event-slot ambiguity protection. Full-run temporal was `0.825` then `0.800`. |
| Summary coverage matrix / must-mention points | `fusion_memory/eval/model_adapters.py` | Useful but incomplete. Query-focused summary points improved final full-run summarization into the `0.706-0.718` band, but broad summaries still miss rubric-specific facets. |
| Value history and slot-state transition | `fusion_memory/retrieval/value_history_pack.py`, `fusion_memory/retrieval/slot_state_transition.py` | Useful. Knowledge update reached `0.8125` then `0.800`. Remaining failures need stronger same-topic value separation and state transition modeling, not local regex additions. |
| Aggregation answer candidates for multi-session | `fusion_memory/retrieval/aggregation_answers.py`, `fusion_memory/retrieval/aggregation_pack.py` | Directionally correct. Distinct counts, value deltas, deadline candidates, and grouped count prototypes helped, but role/scope over-counting remains. |
| Preference constraints and answer requirements | `fusion_memory/retrieval/aggregation_preferences.py`, `fusion_memory/retrieval/answer_requirements.py` | Strong. Preference stayed `0.9375-0.95625`; instruction stayed `0.8875-0.9125`. Preserve typed constraints/checklists and avoid scenario-specific prompt examples. |
| Contradiction claim pairs | `fusion_memory/retrieval/contradiction_claims.py`, `fusion_memory/eval/model_adapters.py` | Partially useful. Category is around `0.756-0.769`; remaining work is query-slot-grounded positive/negative/current claim selection. |
| Exact answer operators | `fusion_memory/retrieval/exact_answer_operators.py` | Promising but not fully validated as a broad category gain. Keep high-confidence gating; avoid frontloading exact candidates into source spans without category replay. |

### Heuristic / Architecture Risk

- `fusion_memory/eval/model_adapters.py` is doing too much: model-pack projection, prompt construction, deterministic answer shortcuts, temporal/summarization/contradiction logic, and benchmark-specific instructions. Further growth here risks turning typed operators back into answer-layer heuristics.
- Named scenario recognizers should not be expanded. Historical diagnostics showed warning signs around book-store/live-chat/patent/Excel-like cases.
- The acceptable path is to promote useful behavior into reusable typed contracts: role/scope, slot state, endpoint pairs, conflict claims, and summary facets.
- The current code has no known qid/gold-answer branch, but it is close enough to the heuristic threshold that future additions should require full-category validation.

## Event Ordering Status

`event_ordering` remains the main structural weakness:

- Final full-run scores: `0.2076758891` and `0.2067433494`.
- Historical category attempts around scoped episodes / raw-facet preservation stayed near `0.196-0.200` and did not produce a real category gain.
- Failures are mostly not simple evidence absence. Historical diagnostics identified `event_order_or_label_mismatch`: the pack often has relevant spans, but the selected sequence abstraction and labels do not align with the BEAM reference ordering.

Current event-ordering files to review:

- `fusion_memory/retrieval/event_ordering_pack.py`
- `fusion_memory/retrieval/event_ordering_sequence.py`
- `fusion_memory/retrieval/event_ordering_typed.py`
- `fusion_memory/retrieval/event_ordering_common.py`
- `fusion_memory/retrieval/event_ordering_episodes.py`
- `fusion_memory/retrieval/event_ordering_records.py`
- `fusion_memory/retrieval/event_ordering_labels.py`
- `fusion_memory/retrieval/event_graph_selection.py`
- `fusion_memory/eval/model_adapters.py`

Recommended future direction:

1. Do not keep adding event-ordering label phrase lists or answer templates.
2. Build a graph-first chronology layer with first-class aspect/topic nodes.
3. Model typed edges such as `NEXT`, `SAME_TOPIC`, `REFINES`, `SUPERSEDES`, `STARTS`, and `COMPLETES`.
4. Keep heuristics only as fallback scoring and labeling, not as the primary event-ordering mechanism.
5. Validate event-ordering by full category runs, because targeted probes repeatedly failed to predict full-category movement.

## Review Entry Points

Recommended first files for project review:

| Area | Entry Point |
|---|---|
| Final benchmark evidence | `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_w24_20260617.json` and repeat file |
| Current handoff state | `AGENTS.md` |
| Answer/model-pack projection and deterministic consumers | `fusion_memory/eval/model_adapters.py` |
| Pack construction | `fusion_memory/retrieval/evidence_pack.py` |
| Temporal operators | `fusion_memory/retrieval/temporal_pack.py` |
| Multi-session aggregation | `fusion_memory/retrieval/aggregation_answers.py`, `fusion_memory/retrieval/aggregation_pack.py` |
| Knowledge-update state | `fusion_memory/retrieval/value_history_pack.py`, `fusion_memory/retrieval/slot_state_transition.py` |
| Preferences/instructions | `fusion_memory/retrieval/aggregation_preferences.py`, `fusion_memory/retrieval/answer_requirements.py` |
| Event ordering | `fusion_memory/retrieval/event_ordering_pack.py` and related event-ordering modules |

## Recommendation

Stop optimization for now and preserve the current baseline. The system passed BEAM 100K twice against the `0.766` target, but the second pass margin is narrow. If work resumes, start with an architecture review and event-ordering design, or with full-category typed-operator loops for summarization and multi-session. Do not continue by adding single-case answer templates.

## 2026-07-23 Retrieval Refactor Note

The product retrieval engine was subsequently refactored on
`refactor/retrieval-engine-debeam`. At product-gate commit
`faf37d91598b1e2f6d4dd9c358eb7f9b1f673c04`:

- `MemoryService` is a 1175-line facade over the product retrieval engine;
- production modes are only `fast` and `balanced`;
- five repository-backed providers own product recall;
- BEAM planning, category instructions, and deterministic answer behavior are
  isolated under `fusion_memory/eval/beam/`;
- the full automated pytest suite reported `994 passed, 9 skipped, 12 subtests passed` and included ordinary BEAM adapter/profile unit tests.

Comparison report:
`docs/retrieval-engine-debeam-comparison-20260722.md`.

No separate BEAM smoke command, benchmark runner, or full benchmark scoring was
run for this refactor, by user direction. No new BEAM artifacts or category
deltas exist. The two scores and artifact paths recorded earlier in this report
remain historical baselines and were not overwritten.
