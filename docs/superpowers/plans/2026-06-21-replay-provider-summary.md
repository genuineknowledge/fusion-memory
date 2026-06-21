# Replay Provider Summary Preservation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve sanitized recall provider summaries in replay artifacts so `tools/rule_audit.py --provider-output` can produce non-empty provider audit evidence from BEAM replay records.

**Architecture:** Keep runtime retrieval unchanged. Extend replay sanitizers to carry `pipeline_trace.pipeline_layers.CandidateRecall.provider_summary` through as sanitized structural data. Event-ordering replay should preserve the same provider summary when compacting hybrid coverage.

**Tech Stack:** Python 3.11+/3.12, `unittest`, existing `tools/beam_retrieval_replay.py`, existing `tools/beam_event_ordering_replay.py`, existing `tools/rule_audit.py`.

## Global Constraints

- Replay artifacts must not contain raw query text, candidate text, source span content, prompt text, or unsanitized metadata.
- Existing rule audit output must remain compatible.
- Provider audit output should become non-empty when replay records contain provider summaries.
- Do not change retrieval behavior, ranking, scoring, provider order, event ordering defaults, or graph promotion state.
- Legacy event_ordering remains default; graph/dual/hybrid remain replay/shadow-only.

---

## Task 1: BEAM Retrieval Replay Provider Summary

**Files:**
- Modify: `tools/beam_retrieval_replay.py`
- Modify: `tests/test_beam_retrieval_replay.py`

**Requirements:**
- Add sanitized `provider_summary` under replay record `pipeline_trace[0]["pipeline_layers"]["CandidateRecall"]`.
- Preserve `source_counts` behavior.
- Hash unsafe provider ids/source families/output source keys.
- Coerce invalid/non-finite counts safely.
- Add a test proving `build_provider_audit(payload["records"])` returns provider rows from replay output.
- Add a raw-text safety test for unsafe provider summary dimensions.

**Verification:**

```bash
python3 -m unittest tests.test_beam_retrieval_replay tests.test_rule_audit -v
```

---

## Task 2: Event Ordering Replay Coverage Provider Summary

**Files:**
- Modify: `tools/beam_event_ordering_replay.py`
- Modify: `tests/test_beam_event_ordering_replay.py`

**Requirements:**
- Preserve sanitized provider summary inside compacted hybrid coverage when `coverage["pipeline_trace"]` exists.
- Ensure `tools/rule_audit.py --provider-output` can read event-ordering replay provider summaries from `coverage.pipeline_trace`.
- Keep existing event ordering metrics/path schema unchanged.
- Add tests that compacted coverage keeps safe provider summary and strips unsafe provider strings.

**Verification:**

```bash
python3 -m unittest tests.test_beam_event_ordering_replay tests.test_rule_audit -v
```

---

## Task 3: Gate And Audit Smoke

**Requirements:**
- Run focused tests:

```bash
python3 -m unittest \
  tests.test_beam_retrieval_replay \
  tests.test_beam_event_ordering_replay \
  tests.test_rule_audit \
  tests.test_retrieval_pipeline \
  -v
```

- Run a tiny synthetic audit smoke or one bounded replay to prove `provider_audit` rows are produced.
- Merge back to local `main` after review and rerun the focused gate on `main`.
