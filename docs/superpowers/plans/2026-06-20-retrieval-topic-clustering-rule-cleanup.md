# Retrieval Topic Clustering And Rule Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve event-ordering graph usefulness with topic clustering/merge, keep dual graph-order + legacy-recall as a shadow evaluator, make regex/rule cleanup measurable, and introduce narrow Retrieval Pipeline interfaces without changing production defaults.

**Architecture:** Keep legacy `event_ordering_*` as the production fallback. Add deterministic topic clustering in the chronology graph layer so fragmented same-session user spans can share stable topic clusters. Use four-path replay (`graph`, `legacy`, `dual`, `hybrid`) as the quality gate. Expand rule audit and retrieval trace around existing code before deleting any rules.

**Tech Stack:** Python 3.11+, `unittest`, existing `MemoryService`, existing chronology storage/replay tooling, existing `fusion_memory.retrieval.rule_registry`, existing `fusion_memory.retrieval.retrieval_trace`.

## Global Constraints

- Do not delete legacy event_ordering code in this phase.
- Do not make graph, dual, or hybrid the default production selector in this phase.
- Every retrieval behavior change must be measurable with replay across `legacy`, `graph`, `dual`, and `hybrid`.
- Graph is a sorting/structure layer; raw chronology and legacy recall remain the recall backbone.
- LLM extractor/router stay out of the real-time main path.
- Do not add project-specific or software-specific regex rescue branches.
- Rule cleanup must be evidence-driven: first-pass cleanup may only mark no-hit, no-contribution, or duplicate rules as delete candidates.
- No raw user text may be stored in rule-hit telemetry or pipeline trace.

---

## File Structure

- Create: `fusion_memory/retrieval/topic_clustering.py`
  - Deterministic topic-cluster helpers used by chronology normalization and selector telemetry.
- Modify: `fusion_memory/retrieval/chronology_normalizer.py`
  - Use topic clustering to merge fragmented same-session topics before node/edge construction.
- Modify: `fusion_memory/retrieval/chronology_selector.py`
  - Add cluster-aware topic expansion and telemetry fields without relaxing graph quality gates.
- Modify: `tools/beam_event_ordering_replay.py`
  - Add dual-vs-legacy shadow diagnostics, cluster diagnostics, and audit-friendly rule-hit output.
- Modify: `fusion_memory/retrieval/rule_audit.py`
  - Add duplicate/no-contribution/delete-candidate classification and CSV-ready fields.
- Modify: `fusion_memory/retrieval/retrieval_trace.py`
  - Add trace section helpers that map onto `QueryUnderstanding`, `CandidateRecall`, `CandidateFusion`, and `EvidencePackBuilder`.
- Tests:
  - `tests/test_topic_clustering.py`
  - `tests/test_chronology_normalizer.py`
  - `tests/test_chronology_selector.py`
  - `tests/test_beam_event_ordering_replay.py`
  - `tests/test_rule_audit.py`
  - `tests/test_retrieval_trace.py`

---

### Task 1: Topic Clustering And Write-Time Topic Merge

**Files:**
- Create: `fusion_memory/retrieval/topic_clustering.py`
- Modify: `fusion_memory/retrieval/chronology_normalizer.py`
- Create: `tests/test_topic_clustering.py`
- Modify: `tests/test_chronology_normalizer.py`

**Interfaces:**
- Produces: `TopicClusterDecision(label: str, confidence: float, reasons: tuple[str, ...], aliases: tuple[str, ...])`
- Produces: `cluster_topic_label(text: str, *, session_hint: str | None = None, previous_label: str | None = None) -> TopicClusterDecision`
- Produces: `cluster_topic_telemetry(decisions: list[TopicClusterDecision]) -> dict[str, object]`
- Consumes existing `taxonomy_entry_for_text`, `tokenize`, `EPISODE_TOPIC_RULES`, and `session_topic_hint`.

- [ ] **Step 1: Write failing topic-clustering unit tests**

Create `tests/test_topic_clustering.py`:

```python
from __future__ import annotations

import unittest

from fusion_memory.retrieval.topic_clustering import cluster_topic_label, cluster_topic_telemetry


class TopicClusteringTests(unittest.TestCase):
    def test_cluster_uses_session_hint_for_related_fragment(self) -> None:
        decision = cluster_topic_label(
            "Then I compared median formulas and altitude methods.",
            session_hint="triangle geometry",
            previous_label="triangle classification",
        )

        self.assertEqual(decision.label, "triangle geometry")
        self.assertGreaterEqual(decision.confidence, 0.70)
        self.assertIn("session_hint", decision.reasons)

    def test_cluster_keeps_strong_taxonomy_label(self) -> None:
        decision = cluster_topic_label(
            "I need the OpenClaw memory adapter to use beginner friendly errors.",
            session_hint="triangle geometry",
            previous_label="triangle geometry",
        )

        self.assertNotEqual(decision.label, "triangle geometry")
        self.assertIn("taxonomy", decision.reasons)

    def test_cluster_telemetry_counts_merged_and_taxonomy_decisions(self) -> None:
        decisions = [
            cluster_topic_label("Then I compared median formulas.", session_hint="triangle geometry"),
            cluster_topic_label("I need the OpenClaw adapter.", session_hint="triangle geometry"),
        ]

        telemetry = cluster_topic_telemetry(decisions)

        self.assertEqual(telemetry["decision_count"], 2)
        self.assertGreaterEqual(telemetry["merged_by_session_hint"], 1)
        self.assertGreaterEqual(telemetry["taxonomy_count"], 1)
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_topic_clustering -v
```

Expected: FAIL because `fusion_memory.retrieval.topic_clustering` is missing.

- [ ] **Step 3: Implement topic clustering helper**

Create `fusion_memory/retrieval/topic_clustering.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

from fusion_memory.core.text import tokenize
from fusion_memory.retrieval.taxonomy import taxonomy_entry_for_text


@dataclass(frozen=True)
class TopicClusterDecision:
    label: str
    confidence: float
    reasons: tuple[str, ...]
    aliases: tuple[str, ...] = ()


def cluster_topic_label(
    text: str,
    *,
    session_hint: str | None = None,
    previous_label: str | None = None,
) -> TopicClusterDecision:
    entry = taxonomy_entry_for_text(text)
    if entry is not None:
        return TopicClusterDecision(
            label=entry.label,
            confidence=0.90 if len(entry.label.split()) >= 2 else 0.78,
            reasons=("taxonomy",),
            aliases=tuple(entry.aliases),
        )
    tokens = {token for token in tokenize(text) if len(token) > 2}
    hint_tokens = {token for token in tokenize(session_hint or "") if len(token) > 2}
    previous_tokens = {token for token in tokenize(previous_label or "") if len(token) > 2}
    if session_hint and (tokens & hint_tokens or tokens & previous_tokens or _is_continuation(text)):
        return TopicClusterDecision(
            label=session_hint,
            confidence=0.74,
            reasons=("session_hint",),
            aliases=(previous_label,) if previous_label and previous_label != session_hint else (),
        )
    if previous_label and _is_continuation(text):
        return TopicClusterDecision(label=previous_label, confidence=0.62, reasons=("previous_topic",))
    label = " ".join(list(tokens)[:4]) or "unknown"
    return TopicClusterDecision(label=label, confidence=0.45, reasons=("lexical_fallback",))


def cluster_topic_telemetry(decisions: list[TopicClusterDecision]) -> dict[str, object]:
    return {
        "decision_count": len(decisions),
        "merged_by_session_hint": sum(1 for decision in decisions if "session_hint" in decision.reasons),
        "taxonomy_count": sum(1 for decision in decisions if "taxonomy" in decision.reasons),
        "fallback_count": sum(1 for decision in decisions if "lexical_fallback" in decision.reasons),
        "labels": sorted({decision.label for decision in decisions}),
    }


def _is_continuation(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("then", "next", "later", "after that", "随后", "然后", "接着"))
```

Refine only as needed to pass tests and fit existing taxonomy behavior.

- [ ] **Step 4: Integrate clustering into `chronology_normalizer.py`**

Change `_infer_topic_label()` to call `cluster_topic_label()` for the non-taxonomy/session decision path. Preserve the existing return shape:

```python
topic_label, topic_is_strong, taxonomy_entry = _infer_topic_label(...)
```

Keep taxonomy entries flowing into `_merge_topic_taxonomy()`. Add cluster decisions to write-batch telemetry:

```python
"topic_cluster": cluster_topic_telemetry(cluster_decisions),
```

Do not remove the existing `EPISODE_TOPIC_RULES`; only route their output through the clustering helper.

- [ ] **Step 5: Add normalizer regression test**

Add to `tests/test_chronology_normalizer.py`:

```python
def test_write_batch_telemetry_reports_topic_cluster_merges(self) -> None:
    scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
    base = datetime(2026, 6, 18, 10, 0, tzinfo=timezone.utc)
    spans = [
        EvidenceSpan("s1", scope, "t1", "user", "turn", "First I studied triangle classification with equilateral and scalene examples.", stable_hash("tc1"), base),
        EvidenceSpan("s2", scope, "t2", "user", "turn", "Then I compared median formulas and altitude methods.", stable_hash("tc2"), base + timedelta(minutes=5)),
    ]

    batch = build_chronology_write_batch(scope, spans, [])

    self.assertEqual({topic.canonical_label for topic in batch.topics}, {"triangle geometry"})
    self.assertGreaterEqual(batch.telemetry["topic_cluster"]["merged_by_session_hint"], 1)
```

- [ ] **Step 6: Run green tests**

Run:

```bash
python3 -m unittest tests.test_topic_clustering tests.test_chronology_normalizer -v
python3 -m py_compile fusion_memory/retrieval/topic_clustering.py fusion_memory/retrieval/chronology_normalizer.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add fusion_memory/retrieval/topic_clustering.py fusion_memory/retrieval/chronology_normalizer.py tests/test_topic_clustering.py tests/test_chronology_normalizer.py
git commit -m "feat: cluster chronology topics"
```

---

### Task 2: Cluster-Aware Selector And Dual Shadow Replay Diagnostics

**Files:**
- Modify: `fusion_memory/retrieval/chronology_selector.py`
- Modify: `tools/beam_event_ordering_replay.py`
- Modify: `tests/test_chronology_selector.py`
- Modify: `tests/test_beam_event_ordering_replay.py`

**Interfaces:**
- Consumes: topic cluster labels/aliases from persisted `ChronologyTopic`.
- Produces selector telemetry fields:
  - `cluster_expanded_topic_ids`
  - `selected_topic_count`
  - `graph_ordered_legacy_recall_count`
- Produces replay summary fields:
  - `dual_lift_over_legacy_f1`
  - `dual_lift_over_legacy_tau`
  - `cluster_diagnostics`

- [ ] **Step 1: Write failing selector expansion test**

Add to `tests/test_chronology_selector.py`:

```python
def test_selector_expands_cluster_related_topics_by_alias(self) -> None:
    memory = MemoryService()
    scope = Scope(workspace_id="graph-select-cluster", user_id="u", agent_id="a", session_id="s")
    created_at = ts("2026-06-18T10:00:00+00:00")
    topic_a = ChronologyTopic("topic-tri-a", scope, "triangle classification", ["triangle geometry"], "en", [], [], 0.9, created_at)
    topic_b = ChronologyTopic("topic-tri-b", scope, "triangle area methods", ["triangle geometry"], "en", [], [], 0.9, created_at)
    for topic in (topic_a, topic_b):
        memory.store.upsert_chronology_topic(topic)
        memory.store.upsert_chronology_phase(ChronologyPhase(f"phase-{topic.topic_id}", topic.topic_id, "implementation", 20, [], 0.9, created_at))
    for node_id, topic_id, text, minute, marker in (
        ("node-a", "topic-tri-a", "First I studied triangle classification.", 0, "first"),
        ("node-b", "topic-tri-b", "Then I compared triangle area methods.", 5, "then"),
    ):
        memory.store.upsert_chronology_event_node(
            ChronologyEventNode(
                node_id, scope, "user", "studied", text, topic_id, f"phase-{topic_id}",
                ts(f"2026-06-18T10:0{minute}:00+00:00"), f"span-{node_id}", f"turn-{node_id}",
                text, "en", 0.9, marker, created_at
            )
        )
    memory.store.insert_chronology_event_edge(ChronologyEventEdge("edge-tri", "node-a", "node-b", "before", "explicit_marker", ["span-node-a", "span-node-b"], 0.9, created_at))

    candidates, telemetry = select_persisted_graph_event_ordering_candidates(
        "What order did I study triangle geometry?",
        scope,
        memory.store,
        limit=5,
        include_session=True,
    )

    self.assertEqual(telemetry["selected_driver"], "persisted_graph")
    self.assertEqual(telemetry["selected_topic_count"], 2)
    self.assertEqual({candidate.metadata["graph_topic_id"] for candidate in candidates}, {"topic-tri-a", "topic-tri-b"})
```

- [ ] **Step 2: Run red selector test**

Run:

```bash
python3 -m unittest tests.test_chronology_selector.ChronologySelectorTests.test_selector_expands_cluster_related_topics_by_alias -v
```

Expected: FAIL because selector only uses top scored topic IDs.

- [ ] **Step 3: Implement cluster-aware topic expansion**

In `chronology_selector.py`, add helper:

```python
def _expand_topic_ids_by_cluster_alias(topics: list[Any], topic_ids: list[str]) -> tuple[list[str], list[str]]:
    selected = {topic.topic_id for topic in topics if topic.topic_id in set(topic_ids)}
    selected_aliases = {
        alias.lower()
        for topic in topics
        if topic.topic_id in selected
        for alias in getattr(topic, "aliases", []) or []
    }
    expanded: list[str] = list(topic_ids)
    added: list[str] = []
    for topic in topics:
        if topic.topic_id in selected:
            continue
        aliases = {str(alias).lower() for alias in getattr(topic, "aliases", []) or []}
        if selected_aliases and selected_aliases & aliases:
            expanded.append(topic.topic_id)
            added.append(topic.topic_id)
    return list(dict.fromkeys(expanded)), added
```

Call it after initial `topic_ids` selection and include telemetry keys. Do not expand topics that have no shared alias. Do not remove the existing same-topic safeguards.

- [ ] **Step 4: Add replay summary unit test**

Add to `tests/test_beam_event_ordering_replay.py`:

```python
def test_aggregate_reports_dual_lift_and_cluster_diagnostics(self) -> None:
    records = [
        {
            "paths": {
                "legacy": {"active": True, "metrics": {"f1": 0.25, "kendall_tau_norm": 0.40, "system_count": 2, "matched": 1}},
                "dual": {"active": True, "metrics": {"f1": 0.50, "kendall_tau_norm": 0.60, "system_count": 2, "matched": 2}},
                "graph": {"active": True, "metrics": {"f1": 0.10, "kendall_tau_norm": 0.30, "system_count": 1, "matched": 0}},
                "hybrid": {"active": False},
            },
            "coverage": {"event_ordering_graph": {"selected_topic_count": 2, "cluster_expanded_topic_ids": ["topic-b"]}},
        }
    ]

    summary = _aggregate(records)

    self.assertAlmostEqual(summary["dual_lift_over_legacy_f1"], 0.25)
    self.assertAlmostEqual(summary["dual_lift_over_legacy_tau"], 0.20)
    self.assertEqual(summary["cluster_diagnostics"]["expanded_query_count"], 1)
```

- [ ] **Step 5: Implement replay diagnostics**

In `_aggregate()`, after path summaries are built:

```python
out["dual_lift_over_legacy_f1"] = float(out.get("dual", {}).get("f1") or 0.0) - float(out.get("legacy", {}).get("f1") or 0.0)
out["dual_lift_over_legacy_tau"] = float(out.get("dual", {}).get("kendall_tau_norm") or 0.0) - float(out.get("legacy", {}).get("kendall_tau_norm") or 0.0)
out["cluster_diagnostics"] = _cluster_diagnostics(records)
```

Add `_cluster_diagnostics(records)` that counts records whose graph coverage includes non-empty `cluster_expanded_topic_ids`, and averages `selected_topic_count`.

- [ ] **Step 6: Run green tests**

Run:

```bash
python3 -m unittest tests.test_chronology_selector tests.test_beam_event_ordering_replay -v
python3 -m py_compile fusion_memory/retrieval/chronology_selector.py tools/beam_event_ordering_replay.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add fusion_memory/retrieval/chronology_selector.py tools/beam_event_ordering_replay.py tests/test_chronology_selector.py tests/test_beam_event_ordering_replay.py
git commit -m "feat: add cluster-aware graph replay diagnostics"
```

---

### Task 3: Rule Audit Classification And First-Pass Cleanup Report

**Files:**
- Modify: `fusion_memory/retrieval/rule_audit.py`
- Modify: `tools/rule_audit.py`
- Modify: `tests/test_rule_audit.py`
- Modify: `tests/test_rule_registry.py`

**Interfaces:**
- Produces audit fields:
  - `duplicate_of`
  - `cleanup_phase`
  - `cleanup_action`
  - `safe_to_delete`
- Keeps existing `recommendation` values for compatibility.

- [ ] **Step 1: Write failing duplicate/no-contribution audit tests**

Add to `tests/test_rule_audit.py`:

```python
def test_rule_audit_marks_duplicate_no_contribution_rules_for_first_pass_cleanup(self) -> None:
    records = [
        {"query_id": "q1", "rule_hits": [{"rule_id": "rule.alpha", "contributed_candidate_id": "c1", "stage": "filter"}], "coverage": {}, "paths": {"hybrid": {"sources": ["s"]}}},
        {"query_id": "q2", "rule_hits": [{"rule_id": "rule.alpha_duplicate", "contributed_candidate_id": None, "stage": "filter", "metadata": {"duplicate_of": "rule.alpha"}}], "coverage": {}, "paths": {"hybrid": {"sources": []}}},
    ]

    audit = build_rule_audit(records)
    duplicate = next(row for row in audit if row["rule_id"] == "rule.alpha_duplicate")

    self.assertEqual(duplicate["duplicate_of"], "rule.alpha")
    self.assertEqual(duplicate["cleanup_phase"], "first_pass")
    self.assertEqual(duplicate["cleanup_action"], "delete_duplicate")
    self.assertTrue(duplicate["safe_to_delete"])
```

- [ ] **Step 2: Run red audit test**

Run:

```bash
python3 -m unittest tests.test_rule_audit.RuleAuditTests.test_rule_audit_marks_duplicate_no_contribution_rules_for_first_pass_cleanup -v
```

Expected: FAIL because cleanup fields are missing.

- [ ] **Step 3: Implement audit cleanup classification**

Extend `tools.rule_audit.build_rule_audit()` rows:

```python
"duplicate_of": duplicate_of,
"cleanup_phase": cleanup_phase,
"cleanup_action": cleanup_action,
"safe_to_delete": safe_to_delete,
```

Classification:

- `delete_duplicate` when any hit metadata has `duplicate_of`.
- `delete_no_contribution` when hit_count > 0 and contribution_count == 0, except `event_ordering.legacy.*`.
- `delete_no_hits` when registered-rule audits include zero-hit rules.
- `migrate_to_taxonomy` when existing recommendation is `migrate_to_taxonomy`.
- `keep_shadow` for `event_ordering.legacy.*`.
- `keep` otherwise.

Mirror the same field names in `fusion_memory.retrieval.rule_audit.build_rule_audit()` for registered-rule audits.

- [ ] **Step 4: Update CLI CSV field list**

Ensure `tools/rule_audit.py` writes the new fields in deterministic order:

```python
rule_id,hit_count,query_count,contribution_count,dropped_count,candidate_sources,recommendation,duplicate_of,cleanup_phase,cleanup_action,safe_to_delete
```

- [ ] **Step 5: Run green tests**

Run:

```bash
python3 -m unittest tests.test_rule_audit tests.test_rule_registry -v
python3 -m py_compile fusion_memory/retrieval/rule_audit.py tools/rule_audit.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/retrieval/rule_audit.py tools/rule_audit.py tests/test_rule_audit.py tests/test_rule_registry.py
git commit -m "feat: classify retrieval rule cleanup candidates"
```

---

### Task 4: Retrieval Pipeline Trace Facade

**Files:**
- Modify: `fusion_memory/retrieval/retrieval_trace.py`
- Modify: `tests/test_retrieval_trace.py`
- Modify: `fusion_memory/api/service.py`
- Modify: `tests/test_fusion_memory.py`

**Interfaces:**
- Produces trace section methods:
  - `query_understanding(...)`
  - `candidate_recall(...)`
  - `candidate_fusion(...)`
  - `evidence_output(...)`
  - `pipeline_layers() -> dict[str, object]`
- Produces stable layer names:
  - `QueryUnderstanding`
  - `CandidateRecall`
  - `CandidateFusion`
  - `EvidencePackBuilder`

- [ ] **Step 1: Write failing trace layer test**

Add to `tests/test_retrieval_trace.py`:

```python
def test_pipeline_layers_expose_stable_boundaries(self) -> None:
    builder = RetrievalTraceBuilder(query_type="event_ordering", mode="benchmark")
    builder.query_understanding(language="zh", intent="event_ordering", features=["temporal"])
    builder.candidate_recall(source_counts={"graph": 2})
    builder.candidate_fusion(selected_sources=["graph"], dropped_count=0)
    builder.evidence_output(source_span_count=2, coverage_insufficient=False)

    layers = builder.pipeline_layers()

    self.assertEqual(list(layers), ["QueryUnderstanding", "CandidateRecall", "CandidateFusion", "EvidencePackBuilder"])
    self.assertEqual(layers["QueryUnderstanding"]["language"], "zh")
    self.assertEqual(layers["CandidateRecall"]["source_counts"], {"graph": 2})
```

- [ ] **Step 2: Run red trace test**

Run:

```bash
python3 -m unittest tests.test_retrieval_trace.RetrievalTraceBuilderTests.test_pipeline_layers_expose_stable_boundaries -v
```

Expected: FAIL because `pipeline_layers()` is missing.

- [ ] **Step 3: Implement pipeline layer facade**

Add constants and method in `retrieval_trace.py`:

```python
PIPELINE_LAYER_ORDER = (
    ("QueryUnderstanding", "query_understanding"),
    ("CandidateRecall", "candidate_recall"),
    ("CandidateFusion", "candidate_fusion"),
    ("EvidencePackBuilder", "evidence_output"),
)

def pipeline_layers(self) -> dict[str, object]:
    return {
        layer_name: self._sections.get(section_name, {})
        for layer_name, section_name in PIPELINE_LAYER_ORDER
    }
```

Update `to_dict()` to include:

```python
"pipeline_layers": self.pipeline_layers(),
```

Keep old section keys for compatibility.

- [ ] **Step 4: Add service trace integration assertion**

In the existing search trace test in `tests/test_fusion_memory.py`, assert:

```python
self.assertIn("pipeline_layers", retrieval_trace)
self.assertIn("QueryUnderstanding", retrieval_trace["pipeline_layers"])
self.assertIn("CandidateRecall", retrieval_trace["pipeline_layers"])
```

- [ ] **Step 5: Run green tests**

Run:

```bash
python3 -m unittest tests.test_retrieval_trace tests.test_fusion_memory.FusionMemoryTests.test_search_trace_contains_retrieval_pipeline_sections -v
python3 -m py_compile fusion_memory/retrieval/retrieval_trace.py fusion_memory/api/service.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/retrieval/retrieval_trace.py fusion_memory/api/service.py tests/test_retrieval_trace.py tests/test_fusion_memory.py
git commit -m "feat: expose retrieval pipeline trace layers"
```

---

### Task 5: Verification Replay And Rule Audit Artifact Command

**Files:**
- Modify: `tools/beam_event_ordering_replay.py`
- Modify: `tests/test_beam_event_ordering_replay.py`

**Interfaces:**
- Produces `replay_config["artifact_commands"]` with:
  - `rule_audit_json`
  - `rule_audit_csv`
- Produces compact stdout fields:
  - `dual_vs_legacy_passed`
  - `dual_lift_over_legacy_f1`
  - `cluster_expanded_query_count`

- [ ] **Step 1: Write failing summary/artifact test**

Add to `tests/test_beam_event_ordering_replay.py`:

```python
def test_summary_for_stdout_includes_dual_and_cluster_fields(self) -> None:
    report = {
        "workspace": "w",
        "split": "100k",
        "query_count": 2,
        "summary": {
            "dual_vs_legacy_passed": True,
            "dual_lift_over_legacy_f1": 0.03,
            "cluster_diagnostics": {"expanded_query_count": 1},
        },
    }

    summary = _summary_for_stdout(report)

    self.assertTrue(summary["dual_vs_legacy_passed"])
    self.assertEqual(summary["dual_lift_over_legacy_f1"], 0.03)
    self.assertEqual(summary["cluster_expanded_query_count"], 1)
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_beam_event_ordering_replay.BeamReplaySummaryTests.test_summary_for_stdout_includes_dual_and_cluster_fields -v
```

Expected: FAIL because compact stdout does not expose those fields.

- [ ] **Step 3: Add artifact commands and stdout fields**

In `run_replay()`, add to `replay_config`:

```python
"artifact_commands": {
    "rule_audit_json": f"python3 tools/rule_audit.py --input {args.output} --output artifacts/rule-audit.json",
    "rule_audit_csv": f"python3 tools/rule_audit.py --input {args.output} --output artifacts/rule-audit.json --csv artifacts/rule-audit.csv",
},
```

If `args.output` is not present inside `run_replay()`, use `getattr(args, "output", "replay.json")`.

In `_summary_for_stdout()`, expose:

```python
"dual_vs_legacy_passed": summary.get("dual_vs_legacy_passed"),
"dual_lift_over_legacy_f1": summary.get("dual_lift_over_legacy_f1"),
"cluster_expanded_query_count": summary.get("cluster_diagnostics", {}).get("expanded_query_count"),
```

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
git commit -m "chore: expose replay audit artifact commands"
```

---

### Task 6: Final Verification

**Files:**
- No production file changes unless a verification failure requires a fix.

**Interfaces:**
- Verifies focused suite.
- Runs best-effort graph-vs-legacy replay if BEAM/Postgres workspace is available.

- [ ] **Step 1: Run focused full suite**

Run:

```bash
python3 -m unittest \
  tests.test_beam_event_ordering_replay \
  tests.test_chronology_normalizer \
  tests.test_chronology_selector \
  tests.test_chronology_backfill \
  tests.test_model_adapters \
  tests.test_fusion_memory \
  tests.test_rule_registry \
  tests.test_rule_audit \
  tests.test_retrieval_trace \
  tests.test_product_cli
```

Expected: PASS.

- [ ] **Step 2: Run replay preflight**

Run:

```bash
python3 tools/beam_event_ordering_replay.py \
  --workspace beam-100k \
  --preflight-only \
  --output artifacts/beam-event-ordering-preflight.json
```

Expected: Writes JSON. If Postgres/BEAM data is unavailable, record the safe failure reason in the final report and do not claim replay metrics.

- [ ] **Step 3: Run replay when preflight is ready**

Only if preflight status is `ok`, run:

```bash
python3 tools/beam_event_ordering_replay.py \
  --workspace beam-100k \
  --mode all \
  --hybrid-source source_spans \
  --gate \
  --output artifacts/beam-event-ordering-graph-legacy-dual-hybrid.json
```

Expected: `dual_vs_legacy_passed` should remain true or failures must be reported. `graph_vs_legacy_passed` is not required in this phase.

- [ ] **Step 4: Run rule audit artifact command if replay ran**

Run:

```bash
python3 tools/rule_audit.py \
  --input artifacts/beam-event-ordering-graph-legacy-dual-hybrid.json \
  --output artifacts/rule-audit.json \
  --csv artifacts/rule-audit.csv
```

Expected: Writes deterministic JSON/CSV.

- [ ] **Step 5: Commit verification notes if docs/artifacts are intentionally tracked**

Do not commit large replay artifacts unless they are already tracked or explicitly requested. If only code/tests changed, skip this step.
