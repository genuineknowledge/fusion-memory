# Memory Product Retrieval Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the current working retrieval baseline, productize graph/legacy shadow evaluation, and introduce the first stable layer of RetrievalTrace, RuleRegistry telemetry, and user-safe product readiness checks.

**Architecture:** Keep legacy event_ordering as the production fallback and add dual graph-order + legacy-recall only as a shadow/feature-flag path. Add narrow pipeline primitives around the current monolithic service first: query understanding telemetry, candidate recall/fusion trace, evidence pack trace, and rule contribution reporting. Product readiness work stays parallel: installer/doctor/error layer must expose actionable user messages without raw tracebacks.

**Tech Stack:** Python 3.11+, `unittest`, existing `MemoryService`, existing `fusion_memory.retrieval.rule_registry`, existing BEAM replay tooling, PostgreSQL/pgvector readiness checks, Qwen 0.6B embedding/reranker runtime configuration.

## Global Constraints

- Do not delete legacy event_ordering code in this phase.
- Do not make dual graph-order + legacy-recall the default production selector in this phase.
- Every retrieval behavior change must be measurable with replay across `legacy`, `graph`, `dual`, and `hybrid`.
- Graph is a sorting/structure layer; raw chronology and legacy recall remain the recall backbone.
- LLM extractor/router stay out of the real-time main path; extractor may run only as async background work.
- User-facing CLI/product paths must return safe actionable errors, not raw tracebacks.
- Default product configuration targets PostgreSQL + pgvector, Qwen 0.6B embedding/reranker, and local-test fallback when production dependencies are unavailable.

---

## File Structure

- Modify: `fusion_memory/core/runtime_config.py`
  - Add runtime flags for dual shadow retrieval and future LLM defaults.
- Modify: `fusion_memory/api/service.py`
  - Add production shadow collection for dual event_ordering without changing selected candidates.
  - Add retrieval trace sections around query understanding, candidate recall, fusion, preservation, and evidence output.
- Create: `fusion_memory/retrieval/retrieval_trace.py`
  - Small dataclasses/helpers for trace sections and source/count summaries.
- Modify: `fusion_memory/retrieval/rule_registry.py`
  - Extend rule definitions and hits with contribution/impact fields while keeping sensitive metadata redaction.
- Create: `fusion_memory/retrieval/rule_audit.py`
  - Summarize rule hit telemetry into an audit table.
- Modify: `fusion_memory/product.py`
  - Strengthen doctor readiness output and safe error mapping.
- Modify: `fusion_memory/cli.py`
  - Route product errors through the safe error layer for JSON and human output.
- Modify: `tools/beam_event_ordering_replay.py`
  - Keep four-path replay as the quality gate and include dual gate fields.
- Test: `tests/test_runtime_config.py`, `tests/test_fusion_memory.py`, `tests/test_rule_registry.py`, `tests/test_product_cli.py`, `tests/test_beam_event_ordering_replay.py`

---

### Task 1: Freeze Event Ordering Baseline and Runtime Flags

**Files:**
- Modify: `fusion_memory/core/runtime_config.py`
- Modify: `fusion_memory/api/service.py`
- Create: `tests/test_runtime_config.py`
- Test: `tests/test_fusion_memory.py`

**Interfaces:**
- Produces: `RuntimeRetrievalFlags(dual_event_ordering_shadow: bool, production_selector: str)`
- Produces: `build_runtime_retrieval_flags() -> RuntimeRetrievalFlags`
- Consumes: environment variables:
  - `FUSION_MEMORY_DUAL_EVENT_ORDERING_SHADOW`
  - `FUSION_MEMORY_EVENT_ORDERING_SELECTOR`

- [ ] **Step 1: Write the failing runtime flag tests**

Create `tests/test_runtime_config.py`:

```python
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fusion_memory.core.runtime_config import build_runtime_retrieval_flags


class RuntimeRetrievalFlagTests(unittest.TestCase):
    def test_dual_event_ordering_shadow_defaults_off_and_legacy_selector(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            flags = build_runtime_retrieval_flags()

        self.assertFalse(flags.dual_event_ordering_shadow)
        self.assertEqual(flags.production_selector, "legacy")

    def test_dual_event_ordering_shadow_can_be_enabled_without_changing_selector(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_DUAL_EVENT_ORDERING_SHADOW": "1",
                "FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "legacy",
            },
            clear=True,
        ):
            flags = build_runtime_retrieval_flags()

        self.assertTrue(flags.dual_event_ordering_shadow)
        self.assertEqual(flags.production_selector, "legacy")

    def test_event_ordering_selector_rejects_unapproved_values(self) -> None:
        with patch.dict(os.environ, {"FUSION_MEMORY_EVENT_ORDERING_SELECTOR": "graph"}, clear=True):
            with self.assertRaisesRegex(ValueError, "unsupported event ordering selector"):
                build_runtime_retrieval_flags()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_runtime_config.RuntimeRetrievalFlagTests
```

Expected: FAIL because `build_runtime_retrieval_flags` is missing.

- [ ] **Step 3: Add minimal runtime flag implementation**

Add to `fusion_memory/core/runtime_config.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeRetrievalFlags:
    dual_event_ordering_shadow: bool = False
    production_selector: str = "legacy"


def build_runtime_retrieval_flags() -> RuntimeRetrievalFlags:
    selector = os.getenv("FUSION_MEMORY_EVENT_ORDERING_SELECTOR", "legacy").strip().lower()
    if selector != "legacy":
        raise ValueError(f"unsupported event ordering selector: {selector}")
    return RuntimeRetrievalFlags(
        dual_event_ordering_shadow=_bool_env("FUSION_MEMORY_DUAL_EVENT_ORDERING_SHADOW", False),
        production_selector=selector,
    )


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
```

Update `memory_service_from_env()` to pass `retrieval_flags=build_runtime_retrieval_flags()` into `MemoryService`.

Update `MemoryService.__init__` signature in `fusion_memory/api/service.py`:

```python
retrieval_flags: Any | None = None,
```

and store:

```python
self.retrieval_flags = retrieval_flags
```

- [ ] **Step 4: Run tests to verify flags pass**

Run:

```bash
python3 -m unittest tests.test_runtime_config.RuntimeRetrievalFlagTests
```

Expected: PASS.

- [ ] **Step 5: Add service-level default behavior test**

Add to `tests/test_fusion_memory.py`:

```python
def test_event_ordering_dual_shadow_is_disabled_by_default(self) -> None:
    service = MemoryService()
    try:
        self.assertFalse(getattr(service.retrieval_flags, "dual_event_ordering_shadow", False))
        self.assertEqual(getattr(service.retrieval_flags, "production_selector", "legacy"), "legacy")
    finally:
        service.close()
```

- [ ] **Step 6: Run service behavior test**

Run:

```bash
python3 -m unittest tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_dual_shadow_is_disabled_by_default
```

Expected: PASS.

---

### Task 2: Add Dual Event Ordering Production Shadow Trace

**Files:**
- Modify: `fusion_memory/api/service.py`
- Test: `tests/test_fusion_memory.py`
- Keep replay validation in: `tools/beam_event_ordering_replay.py`

**Interfaces:**
- Consumes: `self.retrieval_flags.dual_event_ordering_shadow`
- Produces coverage field: `coverage["event_ordering_dual_shadow"]`
- The field must not change `SearchResult.candidates`.

- [ ] **Step 1: Write failing shadow trace test**

Add to `tests/test_fusion_memory.py`:

```python
def test_event_ordering_dual_shadow_reports_without_replacing_selected_candidates(self) -> None:
    class Flags:
        dual_event_ordering_shadow = True
        production_selector = "legacy"

    service = MemoryService(retrieval_flags=Flags())
    scope = Scope(workspace_id="ws-dual-shadow", user_id="u", agent_id="a")
    try:
        service.add({"role": "user", "content": "First I set up schema. Then I implemented transaction CRUD."}, scope)
        result = service.search("What order did I discuss the budget tracker work?", scope, {"query_type_hint": "event_ordering", "limit": 5})

        self.assertIn("event_ordering_dual_shadow", result.coverage)
        shadow = result.coverage["event_ordering_dual_shadow"]
        self.assertEqual(shadow["selected_driver"], "dual_shadow")
        self.assertIn("candidate_count", shadow)
        self.assertIn("sources", shadow)
        self.assertGreaterEqual(len(result.candidates), 1)
    finally:
        service.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_dual_shadow_reports_without_replacing_selected_candidates
```

Expected: FAIL because `event_ordering_dual_shadow` is not populated.

- [ ] **Step 3: Implement shadow-only dual summary**

Add helper to `MemoryService`:

```python
def _event_ordering_dual_shadow_coverage(self, query: str, scope: Scope, limit: int, include_session: bool) -> dict[str, Any]:
    graph_candidates = [
        candidate
        for candidate in self._event_ordering_graph_selector_candidates(query, scope, limit=limit, include_session=include_session)
        if candidate.source == "event_ordering_persisted_graph"
    ]
    plan = self.planner.plan(query, query_type_hint="event_ordering")
    legacy_items, legacy_sources = self._event_ordering_legacy_recall_for_shadow(query, scope, plan, limit, include_session)
    return {
        "selected_driver": "dual_shadow",
        "graph_candidate_count": len(graph_candidates),
        "legacy_candidate_count": len(legacy_items),
        "candidate_count": min(limit, len(legacy_items) + len(graph_candidates)),
        "sources": list(dict.fromkeys([candidate.source for candidate in graph_candidates] + legacy_sources)),
        "production_selector": getattr(self.retrieval_flags, "production_selector", "legacy"),
    }
```

Add helper:

```python
def _event_ordering_legacy_recall_for_shadow(self, query: str, scope: Scope, plan: Any, limit: int, include_session: bool) -> tuple[list[str], list[str]]:
    candidates: list[Candidate] = []
    candidates.extend(self._event_ordering_episode_recall_candidates(query, scope, plan, limit=max(limit * 4, limit + 24), include_session=include_session))
    candidates.extend(self._event_ordering_timeline_candidates(query, plan, scope, limit=max(limit * 3, limit + 12), include_session=include_session))
    ordered = _dedupe_event_ordering_candidates_for_shadow(candidates)
    return [candidate.text for candidate in ordered[:limit] if candidate.text], [candidate.source for candidate in ordered[:limit]]
```

Add module-level helper near other event ordering helpers:

```python
def _dedupe_event_ordering_candidates_for_shadow(candidates: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.id, " ".join(re.findall(r"[a-zA-Z0-9]+", candidate.text.lower())))
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out
```

In `_search_with_rule_hits`, after existing `coverage.update(self._event_ordering_shadow_coverage(...))`, add:

```python
if getattr(self.retrieval_flags, "dual_event_ordering_shadow", False):
    coverage["event_ordering_dual_shadow"] = self._event_ordering_dual_shadow_coverage(
        query,
        scope,
        limit,
        include_session,
    )
```

- [ ] **Step 4: Run shadow trace test**

Run:

```bash
python3 -m unittest tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_dual_shadow_reports_without_replacing_selected_candidates
```

Expected: PASS.

- [ ] **Step 5: Run replay regression**

Run:

```bash
python3 -m unittest tests.test_beam_event_ordering_replay tests.test_fusion_memory
```

Expected: PASS.

---

### Task 3: RetrievalTrace Skeleton Around Current Pipeline

**Files:**
- Create: `fusion_memory/retrieval/retrieval_trace.py`
- Modify: `fusion_memory/api/service.py`
- Test: `tests/test_retrieval_trace.py`
- Test: `tests/test_fusion_memory.py`

**Interfaces:**
- Produces: `RetrievalTraceBuilder`
- Produces trace fields:
  - `trace["retrieval_trace"]["query_understanding"]`
  - `trace["retrieval_trace"]["candidate_recall"]`
  - `trace["retrieval_trace"]["candidate_fusion"]`
  - `trace["retrieval_trace"]["evidence_output"]`

- [ ] **Step 1: Write failing unit test for trace builder**

Create `tests/test_retrieval_trace.py`:

```python
from __future__ import annotations

import unittest

from fusion_memory.retrieval.retrieval_trace import RetrievalTraceBuilder


class RetrievalTraceBuilderTests(unittest.TestCase):
    def test_builds_pipeline_sections_without_raw_text(self) -> None:
        builder = RetrievalTraceBuilder(query_type="event_ordering", mode="benchmark")
        builder.query_understanding(language="en", intent="event_ordering", features=["temporal", "multi_condition"])
        builder.candidate_recall(source_counts={"event_ordering_episode_recall": 3, "event_ordering_persisted_graph": 2})
        builder.candidate_fusion(selected_sources=["event_ordering_episode_recall"], dropped_count=1)
        builder.evidence_output(source_span_count=3, coverage_insufficient=False)

        trace = builder.to_dict()

        self.assertEqual(trace["query_understanding"]["intent"], "event_ordering")
        self.assertEqual(trace["candidate_recall"]["source_counts"]["event_ordering_episode_recall"], 3)
        self.assertEqual(trace["candidate_fusion"]["dropped_count"], 1)
        self.assertFalse(trace["evidence_output"]["coverage_insufficient"])
        self.assertNotIn("query", trace)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_retrieval_trace
```

Expected: FAIL because `fusion_memory.retrieval.retrieval_trace` does not exist.

- [ ] **Step 3: Implement trace builder**

Create `fusion_memory/retrieval/retrieval_trace.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalTraceBuilder:
    query_type: str
    mode: str
    _sections: dict[str, Any] = field(default_factory=dict)

    def query_understanding(self, *, language: str, intent: str, features: list[str]) -> None:
        self._sections["query_understanding"] = {
            "language": language,
            "intent": intent,
            "features": list(features),
        }

    def candidate_recall(self, *, source_counts: dict[str, int]) -> None:
        self._sections["candidate_recall"] = {"source_counts": dict(source_counts)}

    def candidate_fusion(self, *, selected_sources: list[str], dropped_count: int) -> None:
        self._sections["candidate_fusion"] = {
            "selected_sources": list(selected_sources),
            "dropped_count": int(dropped_count),
        }

    def evidence_output(self, *, source_span_count: int, coverage_insufficient: bool) -> None:
        self._sections["evidence_output"] = {
            "source_span_count": int(source_span_count),
            "coverage_insufficient": bool(coverage_insufficient),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "mode": self.mode,
            **self._sections,
        }
```

- [ ] **Step 4: Run trace unit test**

Run:

```bash
python3 -m unittest tests.test_retrieval_trace
```

Expected: PASS.

- [ ] **Step 5: Attach trace builder to `MemoryService.search`**

In `fusion_memory/api/service.py`, import:

```python
from fusion_memory.retrieval.retrieval_trace import RetrievalTraceBuilder
```

In `_search_with_rule_hits`, after `mode` and `plan` are known:

```python
retrieval_trace = RetrievalTraceBuilder(query_type=plan.query_type, mode=str(mode))
retrieval_trace.query_understanding(
    language="zh" if re.search(r"[\u4e00-\u9fff]", query) else "en",
    intent=plan.query_type,
    features=[feature for feature, enabled in {
        "current_value": bool(getattr(plan, "current_value", False)),
        "multi_condition": bool(getattr(plan, "constraints", None)),
        "temporal": plan.query_type in {"temporal_lookup", "event_ordering"},
    }.items() if enabled],
)
```

After `candidate_lists`:

```python
retrieval_trace.candidate_recall(
    source_counts={
        items[0].source if items else f"source_{index}": len(items)
        for index, items in enumerate(candidate_lists)
    }
)
```

After `selected` and `dropped_high_signal` are known:

```python
retrieval_trace.candidate_fusion(
    selected_sources=list(dict.fromkeys(candidate.source for candidate in selected)),
    dropped_count=len(dropped_high_signal),
)
retrieval_trace.evidence_output(
    source_span_count=len(quota_result.selected_span_ids),
    coverage_insufficient=quota_result.coverage_insufficient,
)
```

Add to trace dict:

```python
"retrieval_trace": retrieval_trace.to_dict(),
```

- [ ] **Step 6: Add integration test for saved retrieval trace**

Add to `tests/test_fusion_memory.py`:

```python
def test_search_trace_contains_retrieval_pipeline_sections(self) -> None:
    service = MemoryService()
    scope = Scope(workspace_id="ws-trace", user_id="u", agent_id="a")
    try:
        service.add({"role": "user", "content": "I now prefer PostgreSQL for the memory database."}, scope)
        result = service.search("What database do I currently prefer?", scope)
        trace = service.store.get_trace(result.trace_id, scope)

        retrieval_trace = trace["retrieval_trace"]
        self.assertIn("query_understanding", retrieval_trace)
        self.assertIn("candidate_recall", retrieval_trace)
        self.assertIn("candidate_fusion", retrieval_trace)
        self.assertIn("evidence_output", retrieval_trace)
    finally:
        service.close()
```

- [ ] **Step 7: Run integration test**

Run:

```bash
python3 -m unittest tests.test_retrieval_trace tests.test_fusion_memory.FusionMemoryTests.test_search_trace_contains_retrieval_pipeline_sections
```

Expected: PASS.

---

### Task 4: Rule Registry Contribution and Audit Table

**Files:**
- Modify: `fusion_memory/retrieval/rule_registry.py`
- Create: `fusion_memory/retrieval/rule_audit.py`
- Test: `tests/test_rule_registry.py`

**Interfaces:**
- Extends `RuleDefinition` with optional `ability: str`
- Extends `RuleHit` with `contributed: bool | None` and `impact: str`
- Produces: `build_rule_audit(rule_definitions: list[RuleDefinition], hits: list[dict[str, object]]) -> list[dict[str, object]]`

- [ ] **Step 1: Write failing rule audit tests**

Add to `tests/test_rule_registry.py`:

```python
from fusion_memory.retrieval.rule_audit import build_rule_audit
from fusion_memory.retrieval.rule_registry import RuleDefinition


def test_rule_audit_reports_hits_contributions_and_zero_hit_rules(self) -> None:
    rules = [
        RuleDefinition(rule_id="event.order", module="m", purpose="event order", category="event_ordering", ability="event_ordering"),
        RuleDefinition(rule_id="zh.recall", module="m", purpose="Chinese recall", category="retrieval", ability="chinese_recall"),
    ]
    hits = [
        {"rule_id": "event.order", "contributed": True, "impact": "selected"},
        {"rule_id": "event.order", "contributed": False, "impact": "filtered"},
    ]

    audit = build_rule_audit(rules, hits)

    self.assertEqual(audit[0]["rule_id"], "event.order")
    self.assertEqual(audit[0]["hit_count"], 2)
    self.assertEqual(audit[0]["contribution_count"], 1)
    self.assertEqual(audit[0]["negative_impact_count"], 1)
    self.assertEqual(audit[1]["rule_id"], "zh.recall")
    self.assertEqual(audit[1]["hit_count"], 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_rule_registry
```

Expected: FAIL because `rule_audit` or `ability` is missing.

- [ ] **Step 3: Extend registry dataclasses compatibly**

In `fusion_memory/retrieval/rule_registry.py`, update:

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
```

Update `RuleHit`:

```python
contributed: bool | None = None
impact: str = "observed"
```

Update `record_rule_hit` signature:

```python
contributed: bool | None = None,
impact: str = "observed",
```

and pass those fields into `RuleHit`.

- [ ] **Step 4: Add audit implementation**

Create `fusion_memory/retrieval/rule_audit.py`:

```python
from __future__ import annotations

from typing import Any

from fusion_memory.retrieval.rule_registry import RuleDefinition


def build_rule_audit(rule_definitions: list[RuleDefinition], hits: list[dict[str, object]]) -> list[dict[str, object]]:
    by_rule: dict[str, list[dict[str, object]]] = {}
    for hit in hits:
        by_rule.setdefault(str(hit.get("rule_id")), []).append(hit)

    rows: list[dict[str, Any]] = []
    for rule in sorted(rule_definitions, key=lambda item: item.rule_id):
        rule_hits = by_rule.get(rule.rule_id, [])
        contribution_count = sum(1 for hit in rule_hits if hit.get("contributed") is True or hit.get("impact") == "selected")
        negative_impact_count = sum(1 for hit in rule_hits if hit.get("impact") in {"filtered", "dropped", "misranked"})
        rows.append(
            {
                "rule_id": rule.rule_id,
                "ability": rule.ability,
                "category": rule.category,
                "module": rule.module,
                "hit_count": len(rule_hits),
                "contribution_count": contribution_count,
                "negative_impact_count": negative_impact_count,
                "candidate_for_deletion": len(rule_hits) == 0,
            }
        )
    return rows
```

- [ ] **Step 5: Run rule registry tests**

Run:

```bash
python3 -m unittest tests.test_rule_registry
```

Expected: PASS.

---

### Task 5: Product Doctor Safe Error Layer

**Files:**
- Modify: `fusion_memory/product.py`
- Modify: `fusion_memory/cli.py`
- Test: `tests/test_product_cli.py`

**Interfaces:**
- Produces: `safe_product_error(exc: BaseException) -> dict[str, str]`
- Product JSON errors must include `error`, `message`, `next_step`
- Product human output must not include traceback text.

- [ ] **Step 1: Write failing safe error tests**

Add to `tests/test_product_cli.py`:

```python
from fusion_memory.product import safe_product_error


def test_safe_product_error_maps_connection_failure_to_database_guidance() -> None:
    error = safe_product_error(ConnectionError("connection refused"))

    self.assertEqual(error["error"], "database_not_ready")
    self.assertIn("Postgres", error["message"])
    self.assertIn("fusion-memory doctor", error["next_step"])


def test_safe_product_error_hides_traceback_details() -> None:
    error = safe_product_error(RuntimeError("Traceback (most recent call last): secret stack"))

    self.assertNotIn("Traceback", error["message"])
    self.assertNotIn("secret stack", error["message"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_product_cli
```

Expected: FAIL because `safe_product_error` is missing.

- [ ] **Step 3: Implement safe error mapper**

Add to `fusion_memory/product.py`:

```python
def safe_product_error(exc: BaseException) -> dict[str, str]:
    message = str(exc).lower()
    if isinstance(exc, ConnectionError) or "connection refused" in message or "could not connect" in message:
        return {
            "error": "database_not_ready",
            "message": "Postgres is not ready or cannot be reached.",
            "next_step": "Run fusion-memory doctor, then start Postgres or switch to local test mode.",
        }
    if "address already in use" in message or "port" in message and "use" in message:
        return {
            "error": "port_in_use",
            "message": "The configured service port is already in use.",
            "next_step": "Run fusion-memory doctor and choose another port in the config file.",
        }
    if "transformers" in message or "sentence_transformers" in message or "model" in message:
        return {
            "error": "model_dependency_missing",
            "message": "The configured model dependency is not ready.",
            "next_step": "Run fusion-memory doctor to check Qwen embedding and reranker readiness.",
        }
    return {
        "error": "unexpected_error",
        "message": "Fusion Memory could not complete the request.",
        "next_step": "Run fusion-memory doctor and check the local log file.",
    }
```

- [ ] **Step 4: Wire CLI command handler through safe mapper**

In `fusion_memory/cli.py`, import:

```python
from fusion_memory.product import safe_product_error
```

Wrap top-level `main()` command execution:

```python
try:
    ...
except Exception as exc:
    payload = {"ok": False, **safe_product_error(exc)}
    _print_product_result(payload, json_output=getattr(args, "json", False))
    return 1
```

Do not catch `SystemExit` from argparse.

- [ ] **Step 5: Run product CLI tests**

Run:

```bash
python3 -m unittest tests.test_product_cli
```

Expected: PASS.

---

### Task 6: Product Defaults and Local-Test Fallback Documentation

**Files:**
- Modify: `fusion_memory/product.py`
- Modify: `docs/quickstart.md`
- Test: `tests/test_product_cli.py`

**Interfaces:**
- `init_home(local_test=False)` default remains PostgreSQL + Qwen config.
- `init_home(local_test=True)` writes SQLite + deterministic/lexical local test config.
- `doctor()` returns `next_step` that states local-test fallback when production dependencies are not ready.

- [ ] **Step 1: Write failing default config test**

Add to `tests/test_product_cli.py`:

```python
def test_init_home_defaults_to_postgres_and_qwen(tmp_path) -> None:
    result = init_home(tmp_path)
    config = load_config(tmp_path)

    self.assertTrue(result["ok"])
    self.assertEqual(config["storage_backend"], "postgres")
    self.assertEqual(config["embedding"]["provider"], "qwen")
    self.assertEqual(config["reranker"]["provider"], "qwen")
    self.assertEqual(config["extractor"]["provider"], "rule")
    self.assertEqual(config["query_intent"]["provider"], "off")


def test_init_home_local_test_fallback_uses_dependency_free_defaults(tmp_path) -> None:
    result = init_home(tmp_path, local_test=True)
    config = load_config(tmp_path)

    self.assertTrue(result["ok"])
    self.assertEqual(config["mode"], "local_test")
    self.assertEqual(config["storage_backend"], "sqlite")
    self.assertEqual(config["embedding"]["provider"], "deterministic")
    self.assertEqual(config["reranker"]["provider"], "lexical")
```

- [ ] **Step 2: Run tests to verify current behavior**

Run:

```bash
python3 -m unittest tests.test_product_cli
```

Expected: PASS if existing defaults already match; otherwise FAIL on the specific default mismatch.

- [ ] **Step 3: Fix defaults only if the test fails**

If production defaults fail, update `_default_config()` in `fusion_memory/product.py`:

```python
"mode": "production",
"storage_backend": "postgres",
"db": DEFAULT_POSTGRES_DSN,
"embedding": {"provider": "qwen", "model": DEFAULT_QWEN_EMBEDDING_MODEL},
"reranker": {"provider": "qwen", "model": DEFAULT_QWEN_RERANKER_MODEL},
"extractor": {"provider": "rule"},
"query_intent": {"provider": "off"},
```

If local-test defaults fail, update `_local_test_config()`:

```python
"mode": "local_test",
"storage_backend": "sqlite",
"db": str(paths.db),
"embedding": {"provider": "deterministic"},
"reranker": {"provider": "lexical"},
"extractor": {"provider": "rule"},
"query_intent": {"provider": "off"},
```

- [ ] **Step 4: Update quickstart**

Add to `docs/quickstart.md`:

```markdown
### Recommended first run

Run:

```bash
fusion-memory init
fusion-memory doctor --json
```

The default production setup uses PostgreSQL + pgvector and Qwen 0.6B embedding/reranker.

If Postgres or model dependencies are not ready, use local test mode:

```bash
fusion-memory init --local-test
fusion-memory start
```

Local test mode is dependency-free and is intended for trying the product. It is not the recommended production configuration.
```

- [ ] **Step 5: Run product tests**

Run:

```bash
python3 -m unittest tests.test_product_cli
```

Expected: PASS.

---

### Task 7: Full Verification and Replay Gate

**Files:**
- Validate: `tools/beam_event_ordering_replay.py`
- Validate: `.runtime/beam-runs/`

**Interfaces:**
- Produces final replay JSON with active `graph`, `legacy`, `dual`, and inactive/active `hybrid` according to mode.
- Required comparison: `dual_vs_legacy_passed` must be reported.

- [ ] **Step 1: Run focused regression suite**

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
  tests.test_product_cli
```

Expected: PASS.

- [ ] **Step 2: Run four-path event_ordering replay**

Run:

```bash
.runtime/beam-venv/bin/python tools/beam_event_ordering_replay.py \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --db postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory \
  --split 100k \
  --mode all \
  --hybrid-source source_spans \
  --gate \
  --output .runtime/beam-runs/event_ordering_four_path_product_refactor_20260619.json
```

Expected:
- Report file is written.
- `summary.legacy.active == true`
- `summary.graph.active == true`
- `summary.dual.active == true`
- `summary.dual_vs_legacy_passed` is present.

- [ ] **Step 3: Print replay summary**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path(".runtime/beam-runs/event_ordering_four_path_product_refactor_20260619.json")
r = json.loads(p.read_text())
for path in ("graph", "legacy", "dual", "hybrid"):
    s = r["summary"].get(path, {})
    print(path, {k: s.get(k) for k in ("active", "f1", "kendall_tau_norm", "empty_rate", "mean_matched")})
print("dual_vs_legacy_passed", r["summary"].get("dual_vs_legacy_passed"))
print("gate_failures", r["summary"].get("gate_failures"))
PY
```

Expected: printed metrics are included in the final task report.

---

## Self-Review

- Spec coverage:
  - Baseline freeze: Task 1 and Task 7.
  - Dual shadow path: Task 2 and Task 7.
  - Unified retrieval pipeline first layer: Task 3.
  - Rule registry and telemetry: Task 4.
  - Graph as sorting/structure layer: Task 2 keeps production fallback unchanged and Task 7 measures dual.
  - Product install/doctor/error closure: Task 5 and Task 6.
  - LLM extractor/router out of main path: Global Constraints and Task 6 defaults keep extractor `rule`, query intent `off`.
- Placeholder scan: no unfinished markers or unspecified implementation steps remain.
- Type consistency:
  - `RuntimeRetrievalFlags` is consumed by `MemoryService.retrieval_flags`.
  - `RetrievalTraceBuilder.to_dict()` is saved under `trace["retrieval_trace"]`.
  - `build_rule_audit()` consumes `RuleDefinition` plus saved hit dictionaries.

## Execution Options

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, faster iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, with checkpoints after each task.
