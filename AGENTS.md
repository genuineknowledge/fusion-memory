# Fusion Memory BEAM Handoff

This file is the handoff anchor for retained BEAM evidence and the current product retrieval boundary.

## 2026-07-23 Retrieval Refactor

- Branch: `refactor/retrieval-engine-debeam`
- Product-gate commit before documentation: `faf37d91598b1e2f6d4dd9c358eb7f9b1f673c04`
- Comparison report: `docs/retrieval-engine-debeam-comparison-20260722.md`
- Active architecture: `docs/fusion-memory-architecture.md`
- Production retrieval supports only `fast` and `balanced` and does not import `fusion_memory.eval`.
- BEAM is now an eval-only profile under `fusion_memory/eval/beam/`; category prompts and deterministic answer behavior live in `fusion_memory/eval/beam/model_adapters.py`.
- No separately invoked BEAM smoke command, benchmark runner, full scoring run, or new run artifact was produced for this refactor, by user direction. The ordinary full pytest suite still includes BEAM adapter/profile unit tests.
- Full automated validation at `faf37d9`: `994 passed, 9 skipped, 12 subtests passed`; deployed MCP/PostgreSQL E2E remained unexecuted because URL/tokens were not configured.

## Current Status

Primary target: beat True Memory Pro `0.766` on BEAM 100K without qid/gold-answer hardcoding.

Status: the historical target was achieved in two consecutive full BEAM 100K runs. Those scores remain retained baselines, not validation of the 2026-07-23 retrieval refactor.

Observed current-code full-run band:
- `0.7751916960517531`
- `0.7676505254168324`

The second run still clears the target but by a narrow margin. Preserve the current baseline before broad refactors.

## Retained Evaluation Artifacts

Historical benchmark artifacts were cleaned from `.runtime/beam-runs`; only the two final full-run outputs, diagnostics, and partial worker logs remain.

Retained files:
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_w24_20260617.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_w24_20260617.diagnostic.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_w24_20260617.partials/`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_repeat1_w24_20260617.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_repeat1_w24_20260617.diagnostic.json`
- `.runtime/beam-runs/current_validation_20260616/full_after_temporal_gate_repeat1_w24_20260617.partials/`

Final report:
- `docs/beam-100k-final-evaluation-report-20260617.md`

## Final Result Summary

| Run | Accuracy | Delta vs `0.766` | Answer Match Rate | Answer Failures | Judge Failures |
|---|---:|---:|---:|---:|---:|
| `full_after_temporal_gate_w24_20260617` | `0.7751916961` | `+0.0091916961` | `0.8325` | `0` | `0` |
| `full_after_temporal_gate_repeat1_w24_20260617` | `0.7676505254` | `+0.0016505254` | `0.8200` | `0` | `0` |

## Category Scores

| Category | Run 1 | Run 2 | Notes |
|---|---:|---:|---|
| abstention | `1.0000` | `0.9875` | Strong. |
| contradiction_resolution | `0.76875` | `0.75625` | Needs query-slot-grounded claim selection if revisited. |
| event_ordering | `0.2076758891` | `0.2067433494` | Main unresolved hard category. |
| information_extraction | `0.7828125` | `0.8041666667` | Exact operators promising, keep gated. |
| instruction_following | `0.9125` | `0.8875` | Strong. |
| knowledge_update | `0.8125` | `0.8000` | Useful value/state operators. |
| multi_session_reasoning | `0.7869345238` | `0.7719345238` | Needs generic role/scope count/list operators if revisited. |
| preference_following | `0.9375` | `0.95625` | Strong; preserve typed constraints/checklists. |
| summarization | `0.7182440476` | `0.7061607143` | Coverage matrix helped but broad summaries still miss facets. |
| temporal_reasoning | `0.8250` | `0.8000` | Restored by deterministic temporal consumer; still variance-sensitive. |

## Runtime Resources

Set these paths locally instead of hard-coding machine-specific directories:

```bash
export FUSION_MEMORY_REPO=/path/to/fusion-memory
export FUSION_MEMORY_PYTHON=python
export BEAM_DATASET=/path/to/BEAM
export MODEL_CONFIG_FILE=/path/to/model-config.txt
```

Workspace:
- `beam_100k_rule_qwenembed_sessionized_20260612_1745`

Postgres DSN:
- `postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory`

Model config:
- `$MODEL_CONFIG_FILE`
- Do not print secrets.

Runner pattern:
```bash
cd "$FUSION_MEMORY_REPO"
"$FUSION_MEMORY_PYTHON" tools/beam_parallel_runner.py \
  --dataset "$BEAM_DATASET" --split 100k \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --db postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory \
  --output .runtime/beam-runs/current_validation_20260616/<run>.json \
  query --workers 24 --progress-every 20 \
  --model-config-file "$MODEL_CONFIG_FILE" --model-timeout-seconds 300 \
  --partial-dir .runtime/beam-runs/current_validation_20260616/<run>.partials \
  --diagnostic-output .runtime/beam-runs/current_validation_20260616/<run>.diagnostic.json \
  --max-consecutive-answer-failures 3 --answer-failure-retries 1
```

## Hard Constraints

- No qid-specific or gold-answer hardcoding.
- Do not expand answer templates or named scenario regexes as the main optimization path.
- Prefer typed pack contracts and category-level operators.
- For future work, run the full affected category after tuning a category or large operator.
- The worktree may be dirty; do not revert unrelated user changes.
- Do not copy AGPL True Memory Pro code. Learning concepts is allowed.

## Operator State

Useful current operator families:
- Temporal endpoint pairs and deterministic temporal consumer: `fusion_memory/retrieval/temporal_pack.py`, `fusion_memory/eval/beam/model_adapters.py`
- Summary coverage matrix / must-mention points: `fusion_memory/eval/model_adapters.py`; deterministic BEAM summary behavior is under `fusion_memory/eval/beam/model_adapters.py`
- Value history and slot-state transition: `fusion_memory/retrieval/value_history_pack.py`, `fusion_memory/retrieval/slot_state_transition.py`
- Multi-session aggregation answer candidates: `fusion_memory/retrieval/aggregation_answers.py`, `fusion_memory/retrieval/aggregation_pack.py`
- Preference constraints and answer requirements: `fusion_memory/retrieval/aggregation_preferences.py`, `fusion_memory/retrieval/answer_requirements.py`
- Contradiction claim pairs: `fusion_memory/retrieval/contradiction_claims.py`
- Exact answer operators: `fusion_memory/retrieval/exact_answer_operators.py`

Architecture risk:
- `fusion_memory/eval/model_adapters.py` still owns a large generic model-pack projection surface. BEAM category routing has moved out; avoid moving category behavior back into this file.
- Existing typed operators are acceptable; further single-case answer shortcuts would risk overfitting.

## Event Ordering

`event_ordering` remains the main structural weakness.

Final full-run scores:
- `0.20767588908895968`
- `0.20674334940642`

Historical conclusion:
- Failures are mostly sequence abstraction / label / order mismatch, not just evidence absence.
- Scoped episode and raw-facet recall helped pack coverage but did not improve full category.
- Future path should be graph-first chronology with aspect/topic nodes and typed edges such as `NEXT`, `SAME_TOPIC`, `REFINES`, `SUPERSEDES`, `STARTS`, and `COMPLETES`.
- Do not keep adding event-ordering phrase lists or answer templates.

Key files:
- `fusion_memory/retrieval/event_ordering_pack.py`
- `fusion_memory/retrieval/event_ordering_sequence.py`
- `fusion_memory/retrieval/event_ordering_typed.py`
- `fusion_memory/retrieval/event_ordering_common.py`
- `fusion_memory/retrieval/event_ordering_episodes.py`
- `fusion_memory/retrieval/event_ordering_records.py`
- `fusion_memory/retrieval/event_ordering_labels.py`
- `fusion_memory/retrieval/event_graph_selection.py`

## Next Step If Work Resumes

Start from `docs/beam-100k-final-evaluation-report-20260617.md`.

Recommended options:
1. Architecture review and event-ordering graph design.
2. Full-category typed-operator loop for `summarization`.
3. Full-category typed-operator loop for `multi_session_reasoning`.

Do not start by editing one-off answer text or adding query-shaped regexes.
