# Fusion Memory Product Retrieval Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the BEAM-centered production retrieval path with a small product retrieval engine, reduce `fusion_memory/api/service.py` to an application facade, and keep BEAM runnable only through an evaluation-owned profile.

**Architecture:** Build the product contracts, planner, providers, selector, trace, and evidence pack beside the legacy path first. Inject the new engine into `MemoryService`, move BEAM to an eval-owned composition, switch production and MCP to the product engine, then delete the legacy rescue/preservation path and its category-shaped tests.

**Tech Stack:** Python 3.11+, dataclasses, typing protocols, SQLite/PostgreSQL/pgvector stores, existing embedding/reranker ports, MCP streamable HTTP, pytest/unittest.

## Global Constraints

- Work directly in `/public/home/wwb/memory` on branch `refactor/retrieval-engine-debeam`; do not create another worktree.
- Preserve the existing untracked `/public/home/wwb/memory/uv.lock`.
- Keep public `MemoryService` and MCP product operations compatible: add, search, answer context, delete/clear, history, views, and provenance.
- MCP reads remain authenticated by bearer-token subject and use `Scope(user_id=<subject>, app_id="mcp")`; workspace/session are write provenance, not read isolation boundaries.
- Same-user memory remains visible across sessions/workspaces; different users remain isolated.
- Different users continue to run concurrently; same-user writes continue to use the PostgreSQL advisory lock.
- Production accepts only `fast` and `balanced` retrieval modes; `benchmark` and `query_type_hint` are eval-only concepts.
- Production code must not import `fusion_memory.eval`.
- Do not migrate database data or change persisted schemas in this refactor.
- Do not add qid-specific, gold-answer, named-scenario, or query-shaped benchmark shortcuts.
- Preserve the retained BEAM 100K artifacts and the recorded baselines `0.7751916960517531` and `0.7676505254168324`.
- Every implementation task uses red-green TDD and ends with a focused commit.

---

## Target File Map

### Product retrieval

- `fusion_memory/retrieval/context.py`: immutable request, runtime context, plan, provider report, and result contracts.
- `fusion_memory/retrieval/engine.py`: `RetrievalEngine` protocol and product engine factory.
- `fusion_memory/retrieval/ports.py`: repository protocol used by providers and pack builders.
- `fusion_memory/retrieval/product_planner.py`: capability-based product query planner during migration.
- `fusion_memory/retrieval/providers/product_base.py`: provider protocol and provider outcome.
- `fusion_memory/retrieval/providers/vector.py`: semantic candidate source.
- `fusion_memory/retrieval/providers/lexical.py`: exact/lexical candidate source.
- `fusion_memory/retrieval/providers/temporal.py`: recency and explicit time-range source.
- `fusion_memory/retrieval/providers/entity.py`: entity-linked source.
- `fusion_memory/retrieval/providers/chronology.py`: persisted chronology graph source with event fallback.
- `fusion_memory/retrieval/providers/product_registry.py`: product provider registration and execution.
- `fusion_memory/retrieval/selection.py`: normalization, RRF fusion, optional rerank, diversity, and stable final selection.
- `fusion_memory/retrieval/tracing.py`: sanitized product retrieval trace.
- `fusion_memory/retrieval/product_engine.py`: product pipeline orchestration.
- `fusion_memory/retrieval/product_evidence_pack.py`: category-free product evidence pack builder during migration.

### Evaluation

- `fusion_memory/eval/beam/__init__.py`: BEAM profile exports.
- `fusion_memory/eval/beam/query_planner.py`: BEAM category to product-capability mapping.
- `fusion_memory/eval/beam/engine.py`: eval-owned retrieval and evidence-pack composition.
- `fusion_memory/eval/beam/model_adapters.py`: BEAM-only deterministic answer/category helpers moved out of the generic adapter.

### Service helpers retained after cleanup

- `fusion_memory/api/service_telemetry.py`: model-call sanitization and labeled reporting helpers.
- `fusion_memory/ingestion/order_markers.py`: explicit order-marker parsing used while writing event edges.

### New tests

- `tests/test_retrieval_public_contract.py`
- `tests/test_product_retrieval_contracts.py`
- `tests/test_product_query_planner.py`
- `tests/test_product_retrieval_providers.py`
- `tests/test_product_retrieval_engine.py`
- `tests/test_product_evidence_pack.py`
- `tests/test_memory_service_retrieval_engine.py`
- `tests/test_retrieval_architecture_boundaries.py`
- `tests/test_product_retrieval_cases.py`

### Migration evidence

- `docs/retrieval-heuristic-inventory-20260722.md`: exhaustive disposition of legacy retrieval methods and helpers.
- `docs/retrieval-engine-debeam-comparison-20260722.md`: product and BEAM validation evidence.

---

### Task 0: Capture Public Behavior And Classify Legacy Heuristics

**Files:**
- Create: `docs/retrieval-heuristic-inventory-20260722.md`
- Create: `tests/test_retrieval_public_contract.py`
- Read: `fusion_memory/api/service.py:360-744,1620-3737`
- Read: `fusion_memory/api/service_helpers.py:1-1773`

**Interfaces:**
- Produces: a stable public-contract test suite that must pass before and after the refactor.
- Produces: a complete legacy-method disposition table with exactly one destination per method group: product, eval-only, delete, or non-retrieval helper.

- [ ] **Step 1: Add public characterization tests**

```python
from fusion_memory import MemoryService, Scope
from fusion_memory.core.models import EvidencePack, SearchResult


def test_public_search_and_answer_context_shapes_remain_stable() -> None:
    service = MemoryService()
    write_scope = Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a")
    read_scope = Scope(user_id="user-a")
    try:
        service.add("Atlas uses Qdrant for retrieval.", write_scope)
        result = service.search("What does Atlas use?", read_scope, {"allow_cross_session": True})
        pack = service.answer_context("What does Atlas use?", read_scope, {"allow_cross_session": True})
        assert isinstance(result, SearchResult)
        assert result.trace_id
        assert isinstance(result.coverage, dict)
        assert isinstance(pack, EvidencePack)
        assert any(span["session_id"] == "session-a" for span in pack.source_spans)
    finally:
        service.close()


def test_public_read_is_same_user_cross_session_and_different_user_isolated() -> None:
    service = MemoryService()
    try:
        service.add("Private marker cobalt-key belongs to user A.", Scope(user_id="user-a", workspace_id="a", session_id="s1"))
        same_user = service.search("Which private marker?", Scope(user_id="user-a"), {"allow_cross_session": True})
        other_user = service.search("Which private marker?", Scope(user_id="user-b"), {"allow_cross_session": True})
        assert any("cobalt-key" in candidate.text for candidate in same_user.candidates)
        assert all("cobalt-key" not in candidate.text for candidate in other_user.candidates)
    finally:
        service.close()
```

- [ ] **Step 2: Run characterization tests against the legacy implementation**

Run: `pytest -q tests/test_retrieval_public_contract.py tests/test_postgres_concurrency.py`

Expected: all tests pass before production code changes. If a characterization assertion exposes an existing defect, stop this plan and handle that defect through the systematic-debugging workflow rather than weakening the contract.

- [ ] **Step 3: Write the heuristic disposition inventory**

The document must contain this complete top-level mapping:

| Legacy group | Destination | Reason |
| --- | --- | --- |
| raw/fact/event/view/profile hybrid recall | Product providers | Generic product retrieval capability |
| `_entity_candidates`, simplified `_exact_candidates` | Product entity/lexical providers | Explicit entity and lexical behavior |
| persisted chronology selector | Product chronology provider | Generic ordered-event capability |
| contradiction/aggregation/event-ordering coverage providers | BEAM eval profile or delete | Category-shaped compensation |
| scent-trail, broad-raw, quality-fallback and topic-rescue chains | Delete | Repeated post-recall heuristic compensation |
| `_preserve_*`, post-rerank preservation and runtime required preservation | Delete | Product selector is one-pass |
| category-heavy evidence-pack expansion | BEAM eval profile or delete | Model/benchmark shaping, not product recall |
| `_sanitize_model_call`, `_model_call_summary`, `_labeled_precision` | `api/service_telemetry.py` | Non-retrieval service reporting |
| `_explicit_order_mentions` | `ingestion/order_markers.py` | Write-side event-edge parsing |

Below the table, list every `MemoryService` method from lines 1640-3737 and every top-level `service_helpers.py` function under one of those rows. No method/function may be omitted or assigned to two destinations.

- [ ] **Step 4: Commit the baseline**

```bash
git add -f docs/retrieval-heuristic-inventory-20260722.md
git add tests/test_retrieval_public_contract.py
git commit -m "test(memory): capture retrieval contracts and heuristic inventory"
```

---

### Task 1: Add Product Retrieval Contracts And Ports

**Files:**
- Create: `fusion_memory/retrieval/context.py`
- Create: `fusion_memory/retrieval/engine.py`
- Create: `fusion_memory/retrieval/ports.py`
- Create: `tests/test_product_retrieval_contracts.py`

**Interfaces:**
- Produces: `SearchMode`, `ProviderKind`, `OrderingMode`, `TimeRange`, `ProviderRequest`, `SearchRequest`, `RetrievalContext`, `ProductQueryPlan`, `ProviderFailure`, `ProviderReport`, `RetrievalResult`.
- Produces: `RetrievalEngine.search(context, request, plan=None) -> RetrievalResult` and `RetrievalEngine.build_evidence_pack(context, request, result, token_budget) -> EvidencePack`.
- Produces: structural `MemorySearchRepository` protocol implemented by both existing stores.

- [ ] **Step 1: Write failing contract tests**

```python
from datetime import datetime, timezone

import pytest

from fusion_memory.core.models import Scope
from fusion_memory.retrieval.context import ProviderKind, RetrievalContext, SearchRequest


def test_product_search_request_rejects_benchmark_mode() -> None:
    with pytest.raises(ValueError, match="fast or balanced"):
        SearchRequest(query="where is the decision", limit=12, mode="benchmark")


def test_retrieval_context_rejects_authenticated_user_mismatch() -> None:
    with pytest.raises(ValueError, match="user_id"):
        RetrievalContext(
            scope=Scope(user_id="user-b"),
            user_id="user-a",
            now=datetime.now(timezone.utc),
            trace_id="trace-1",
            deadline=None,
            include_session=False,
        )


def test_search_request_uses_immutable_provider_filter() -> None:
    request = SearchRequest(
        query="latest Atlas database",
        limit=8,
        enabled_providers=frozenset({ProviderKind.LEXICAL, ProviderKind.TEMPORAL}),
    )
    assert request.enabled_providers == frozenset({ProviderKind.LEXICAL, ProviderKind.TEMPORAL})
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `pytest -q tests/test_product_retrieval_contracts.py`

Expected: FAIL during collection because `fusion_memory.retrieval.context` does not exist.

- [ ] **Step 3: Implement the immutable contracts**

Create `context.py` with these exact public shapes:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from fusion_memory.core.models import Candidate, Scope

SearchMode = Literal["fast", "balanced"]


class ProviderKind(str, Enum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    CHRONOLOGY = "chronology"


class OrderingMode(str, Enum):
    RELEVANCE = "relevance"
    RECENCY = "recency"
    CHRONOLOGICAL = "chronological"


@dataclass(frozen=True)
class TimeRange:
    start: datetime | None = None
    end: datetime | None = None

    def contains(self, value: datetime) -> bool:
        return (self.start is None or value >= self.start) and (self.end is None or value <= self.end)


@dataclass(frozen=True)
class ProviderRequest:
    kind: ProviderKind
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("provider limit must be positive")


@dataclass(frozen=True)
class SearchRequest:
    query: str
    limit: int
    mode: SearchMode = "fast"
    time_range: TimeRange | None = None
    include_trace: bool = True
    enabled_providers: frozenset[ProviderKind] | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query is required")
        if self.limit < 1:
            raise ValueError("limit must be positive")
        if self.mode not in {"fast", "balanced"}:
            raise ValueError("mode must be fast or balanced")


@dataclass(frozen=True)
class RetrievalContext:
    scope: Scope
    user_id: str | None
    now: datetime
    trace_id: str
    deadline: datetime | None
    include_session: bool

    def __post_init__(self) -> None:
        if self.user_id is not None and self.scope.user_id != self.user_id:
            raise ValueError("retrieval context user_id must match scope.user_id")

    def check_deadline(self) -> None:
        if self.deadline is not None and datetime.now(timezone.utc) > self.deadline.astimezone(timezone.utc):
            raise TimeoutError("retrieval deadline exceeded")


@dataclass(frozen=True)
class ProductQueryPlan:
    intent: str
    provider_requests: tuple[ProviderRequest, ...]
    time_range: TimeRange | None
    entities: tuple[str, ...]
    speaker: str | None
    ordering: OrderingMode
    use_reranker: bool
    query_intent: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderFailure:
    provider: ProviderKind
    error_code: str
    retryable: bool


@dataclass(frozen=True)
class ProviderReport:
    provider: ProviderKind
    candidate_count: int
    elapsed_ms: float
    failure: ProviderFailure | None = None


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[Candidate, ...]
    coverage: dict[str, Any]
    trace: dict[str, Any]
    plan: ProductQueryPlan
```

Create `engine.py` with the two-method protocol and a `RetrievalUnavailable(RuntimeError)` exception. Create `ports.py` as a `Protocol` declaring the existing store methods used later: `search_spans`, `list_spans`, `search_facts`, `list_facts`, `search_events`, `list_events`, `list_current_views`, `search_entity_profiles`, `search_entities`, `get_span`, `get_fact`, and the four `list_chronology_*` methods. Use the exact existing method signatures, including `scope` and `include_session`.

- [ ] **Step 4: Run contract tests**

Run: `pytest -q tests/test_product_retrieval_contracts.py`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/context.py fusion_memory/retrieval/engine.py fusion_memory/retrieval/ports.py tests/test_product_retrieval_contracts.py
git commit -m "feat(retrieval): add product engine contracts"
```

---

### Task 2: Add A Capability-Based Product Query Planner

**Files:**
- Create: `fusion_memory/retrieval/product_planner.py`
- Create: `tests/test_product_query_planner.py`
- Read: `fusion_memory/retrieval/query_intent.py`

**Interfaces:**
- Consumes: `SearchRequest`, `ProviderRequest`, `ProviderKind`, `OrderingMode`, `ProductQueryPlan`.
- Produces: `ProductQueryPlanner.plan(request: SearchRequest) -> ProductQueryPlan`.
- Does not consume: BEAM category, `query_type_hint`, benchmark metadata, gold answers, or eval rubric.

- [ ] **Step 1: Write planner tests for product capabilities**

```python
from fusion_memory.retrieval.context import OrderingMode, ProviderKind, SearchRequest
from fusion_memory.retrieval.product_planner import ProductQueryPlanner


def _providers(plan):
    return {request.kind for request in plan.provider_requests}


def test_planner_uses_generic_sources_for_fact_query() -> None:
    plan = ProductQueryPlanner().plan(SearchRequest("What database does Atlas use?", 8))
    assert _providers(plan) == {ProviderKind.VECTOR, ProviderKind.LEXICAL, ProviderKind.ENTITY}
    assert plan.intent == "factual"
    assert plan.ordering is OrderingMode.RELEVANCE


def test_planner_routes_explicit_timeline_without_beam_category() -> None:
    plan = ProductQueryPlanner().plan(SearchRequest("按时间顺序列出 Atlas 的部署过程", 10, mode="balanced"))
    assert ProviderKind.CHRONOLOGY in _providers(plan)
    assert ProviderKind.TEMPORAL in _providers(plan)
    assert plan.intent == "chronology"
    assert plan.ordering is OrderingMode.CHRONOLOGICAL
    assert "event_ordering" not in repr(plan)


def test_planner_uses_recency_for_current_state() -> None:
    plan = ProductQueryPlanner().plan(SearchRequest("What is the latest Atlas database?", 6))
    assert ProviderKind.TEMPORAL in _providers(plan)
    assert plan.intent == "current_state"
    assert plan.ordering is OrderingMode.RECENCY
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest -q tests/test_product_query_planner.py`

Expected: FAIL because `ProductQueryPlanner` does not exist.

- [ ] **Step 3: Implement deterministic capability planning**

Use `analyze_query_intent(request.query)` and construct the plan with these rules:

```python
def _intent_label(intent) -> str:
    if intent.temporal.requires_order:
        return "chronology"
    if intent.needs_current_state:
        return "current_state"
    if intent.needs_conflict_check:
        return "conflict"
    if intent.answer_shape == "summary":
        return "summary"
    if intent.aggregation.operation != "none":
        return "aggregation"
    if intent.answer_shape == "instruction":
        return "instruction"
    if intent.temporal.requires_time:
        return "temporal"
    return "factual"


def _provider_requests(intent, limit: int) -> tuple[ProviderRequest, ...]:
    kinds = [ProviderKind.VECTOR, ProviderKind.LEXICAL]
    if intent.entities:
        kinds.append(ProviderKind.ENTITY)
    if intent.temporal.requires_time or intent.needs_current_state:
        kinds.append(ProviderKind.TEMPORAL)
    if intent.temporal.requires_order:
        kinds.append(ProviderKind.CHRONOLOGY)
    if not intent.entities:
        kinds.append(ProviderKind.ENTITY)
    return tuple(ProviderRequest(kind, max(limit * 2, 12)) for kind in dict.fromkeys(kinds))
```

Map speaker scope `any` to `None`; map ordered queries to `CHRONOLOGICAL`, current-state queries to `RECENCY`, and all others to `RELEVANCE`. Set `use_reranker` only for `balanced` mode. Preserve `intent.to_dict()` in `query_intent` for trace and pack metadata.

- [ ] **Step 4: Run planner tests and existing intent tests**

Run: `pytest -q tests/test_product_query_planner.py tests/test_model_adapters.py -k 'query_intent or product'`

Expected: new planner tests pass; existing selected intent tests pass.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/product_planner.py tests/test_product_query_planner.py
git commit -m "feat(retrieval): plan product capabilities without benchmark types"
```

---

### Task 3: Add Vector, Lexical, And Entity Providers Without Service Callbacks

**Files:**
- Create: `fusion_memory/retrieval/providers/product_base.py`
- Create: `fusion_memory/retrieval/providers/vector.py`
- Create: `fusion_memory/retrieval/providers/lexical.py`
- Create: `fusion_memory/retrieval/providers/entity.py`
- Create: `tests/test_product_retrieval_providers.py`

**Interfaces:**
- Consumes: `MemorySearchRepository`, `RetrievalContext`, `SearchRequest`, `ProductQueryPlan`, `ProviderRequest`.
- Produces: `ProviderContext`, `ProviderOutcome`, `ProviderUnavailable`, and three `CandidateProvider` implementations.
- Provider constructors accept repositories/ports, never `MemoryService`.

- [ ] **Step 1: Write provider tests using a repository fake**

```python
from datetime import datetime, timezone

from fusion_memory.core.models import EvidenceSpan, Scope
from fusion_memory.retrieval.context import ProviderKind
from fusion_memory.retrieval.providers.lexical import LexicalProvider
from fusion_memory.retrieval.providers.product_base import ProviderContext


def test_lexical_provider_reads_repository_without_service(product_provider_context, repository_fake) -> None:
    repository_fake.spans = [
        EvidenceSpan(
            span_id="span-1",
            scope=Scope(user_id="user-a"),
            turn_id="turn-1",
            speaker="user",
            span_type="turn",
            content="Atlas uses Qdrant for retrieval.",
            content_hash="hash-1",
            timestamp=datetime.now(timezone.utc),
        )
    ]
    outcome = LexicalProvider(repository_fake).recall(product_provider_context("Atlas Qdrant", ProviderKind.LEXICAL))
    assert [candidate.id for candidate in outcome.candidates] == ["span-1"]
    assert outcome.candidates[0].source == "product_lexical"
    assert not hasattr(outcome, "service")
```

Add companion tests that:

- `VectorProvider` uses `search_spans` and keeps only `semantic_score > 0` candidates.
- `EntityProvider` uses `search_entities`, hydrates source spans with the same scope, and never returns another user's span.
- Every outcome exposes only provider ID, candidates, elapsed time, and optional failure.

- [ ] **Step 2: Verify provider tests fail**

Run: `pytest -q tests/test_product_retrieval_providers.py -k 'vector or lexical or entity'`

Expected: FAIL because the product provider modules do not exist.

- [ ] **Step 3: Implement the common provider contract**

```python
from dataclasses import dataclass

from fusion_memory.core.models import Candidate
from fusion_memory.retrieval.context import ProductQueryPlan, ProviderFailure, ProviderKind, RetrievalContext, SearchRequest
from fusion_memory.retrieval.ports import MemorySearchRepository


class ProviderUnavailable(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ProviderContext:
    runtime: RetrievalContext
    request: SearchRequest
    plan: ProductQueryPlan
    repository: MemorySearchRepository
    provider: ProviderKind
    limit: int


@dataclass(frozen=True)
class ProviderOutcome:
    provider: ProviderKind
    candidates: tuple[Candidate, ...]
    elapsed_ms: float
    failure: ProviderFailure | None = None
```

- [ ] **Step 4: Implement the three providers**

Use these source names and score rules:

```python
VECTOR_SOURCE = "product_vector"
LEXICAL_SOURCE = "product_lexical"
ENTITY_SOURCE = "product_entity"
```

- `VectorProvider`: call `search_spans`; emit candidates when `semantic_score > 0`; copy `semantic_score`, `bm25_score`, `score`, speaker, span type, and timestamp. Convert only `EndpointUnavailable` into `ProviderUnavailable("model_unavailable")`; let storage errors propagate.
- `LexicalProvider`: call `list_spans`, `list_facts`, `list_events`, `list_current_views`, and `search_entity_profiles`; score with `keyword_score`; emit only positive matches; sort by exact phrase containment, score, timestamp, and stable ID.
- `EntityProvider`: call `search_entities`; for each entity source span call `get_span(span_id, context.runtime.scope, include_session=context.runtime.include_session)`; emit a span candidate with entity name and entity ID metadata.

All three providers must enforce `context.limit` and must not import `fusion_memory.api.service`.

- [ ] **Step 5: Run provider tests**

Run: `pytest -q tests/test_product_retrieval_providers.py -k 'vector or lexical or entity'`

Expected: selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/retrieval/providers/product_base.py fusion_memory/retrieval/providers/vector.py fusion_memory/retrieval/providers/lexical.py fusion_memory/retrieval/providers/entity.py tests/test_product_retrieval_providers.py
git commit -m "feat(retrieval): add service-independent product providers"
```

---

### Task 4: Add Temporal And Chronology Providers And Product Registry

**Files:**
- Create: `fusion_memory/retrieval/providers/temporal.py`
- Create: `fusion_memory/retrieval/providers/chronology.py`
- Create: `fusion_memory/retrieval/providers/product_registry.py`
- Modify: `tests/test_product_retrieval_providers.py`
- Read: `fusion_memory/retrieval/chronology_selector.py`

**Interfaces:**
- Produces: `TemporalProvider`, `ChronologyProvider`, and `ProductProviderRegistry.run(context, request, plan) -> tuple[ProviderOutcome, ...]`.
- Registry execution is sequential while repositories are transaction-bound; cross-user request concurrency remains at `PostgresOperationExecutor`.

- [ ] **Step 1: Add failing temporal, chronology, and partial-failure tests**

```python
def test_temporal_provider_honors_explicit_time_range(temporal_context, repository_fake) -> None:
    outcome = TemporalProvider(repository_fake).recall(temporal_context)
    assert [candidate.id for candidate in outcome.candidates] == ["inside-range"]


def test_chronology_provider_uses_persisted_graph_without_preservation_metadata(chronology_context, repository_fake) -> None:
    outcome = ChronologyProvider(repository_fake).recall(chronology_context)
    assert [candidate.source for candidate in outcome.candidates] == ["product_chronology"]
    assert "must_preserve_reason" not in outcome.candidates[0].metadata


def test_registry_records_model_provider_failure_and_keeps_lexical_result(registry_context) -> None:
    registry = ProductProviderRegistry([FailingVectorProvider(), StaticLexicalProvider()])
    outcomes = registry.run(*registry_context)
    assert outcomes[0].failure.error_code == "model_unavailable"
    assert [candidate.id for candidate in outcomes[1].candidates] == ["lexical-1"]
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest -q tests/test_product_retrieval_providers.py -k 'temporal or chronology or registry'`

Expected: FAIL because the new providers and registry do not exist.

- [ ] **Step 3: Implement temporal and chronology behavior**

- `TemporalProvider` reads spans and events, applies `request.time_range` when provided, otherwise keeps query-relevant recent records, calculates `temporal_score`, and emits `product_temporal` candidates in descending recency.
- `ChronologyProvider` calls `select_persisted_graph_event_ordering_candidates`; rename returned source to `product_chronology`, remove preservation metadata, and retain graph topic/phase/timeline metadata. If no persisted graph candidates exist, sort repository events by `time_start`, then ID, and emit at most the provider budget.
- Both providers use only the repository protocol and context.

- [ ] **Step 4: Implement the product registry**

```python
class ProductProviderRegistry:
    def __init__(self, providers) -> None:
        self._providers = {provider.kind: provider for provider in providers}

    def run(self, runtime, request, plan):
        from time import perf_counter

        outcomes = []
        for provider_request in plan.provider_requests:
            runtime.check_deadline()
            if request.enabled_providers is not None and provider_request.kind not in request.enabled_providers:
                continue
            provider = self._providers[provider_request.kind]
            context = ProviderContext(
                runtime=runtime,
                request=request,
                plan=plan,
                repository=provider.repository,
                provider=provider_request.kind,
                limit=provider_request.limit,
            )
            started = perf_counter()
            try:
                outcomes.append(provider.recall(context))
            except ProviderUnavailable as exc:
                outcomes.append(
                    ProviderOutcome(
                        provider=provider_request.kind,
                        candidates=(),
                        elapsed_ms=(perf_counter() - started) * 1000,
                        failure=ProviderFailure(
                            provider=provider_request.kind,
                            error_code=exc.code,
                            retryable=True,
                        ),
                    )
                )
        return tuple(outcomes)
```

Do not catch arbitrary exceptions. PostgreSQL/authorization/visibility errors must reach the service and fail the request.

- [ ] **Step 5: Run all provider tests**

Run: `pytest -q tests/test_product_retrieval_providers.py`

Expected: all provider tests pass.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/retrieval/providers/temporal.py fusion_memory/retrieval/providers/chronology.py fusion_memory/retrieval/providers/product_registry.py tests/test_product_retrieval_providers.py
git commit -m "feat(retrieval): add temporal chronology and provider registry"
```

---

### Task 5: Add Generic Selection, Trace, And Product Engine

**Files:**
- Create: `fusion_memory/retrieval/selection.py`
- Create: `fusion_memory/retrieval/tracing.py`
- Create: `fusion_memory/retrieval/product_engine.py`
- Create: `tests/test_product_retrieval_engine.py`
- Modify: `fusion_memory/retrieval/engine.py`
- Modify: `fusion_memory/retrieval/product_planner.py`

**Interfaces:**
- Consumes: planner, registry, reranker, RRF, MMR, and product contracts.
- Produces: `ProductRetrievalEngine.search`, `ProductRetrievalEngine.search_with_plan`, and sanitized trace records.
- `search_with_plan` is the public eval extension point; it accepts only `ProductQueryPlan`, never a benchmark category.

- [ ] **Step 1: Write failing engine tests**

```python
def test_engine_runs_one_pass_without_post_selection_rescue(engine_fixture) -> None:
    result = engine_fixture.engine.search(engine_fixture.context, engine_fixture.request)
    assert [candidate.id for candidate in result.candidates] == ["exact", "semantic"]
    assert result.trace["stages"] == ["plan", "recall", "fusion", "selection"]
    assert "rescue" not in repr(result.trace).lower()


def test_engine_degrades_when_one_provider_is_unavailable(engine_with_failed_vector) -> None:
    result = engine_with_failed_vector.search(engine_with_failed_vector.context, engine_with_failed_vector.request)
    assert [candidate.id for candidate in result.candidates] == ["lexical"]
    assert result.coverage["degraded"] is True
    assert result.coverage["provider_failures"] == ["model_unavailable"]


def test_engine_raises_when_all_planned_providers_fail(engine_with_all_failures) -> None:
    with pytest.raises(RetrievalUnavailable, match="all planned providers failed"):
        engine_with_all_failures.search(engine_with_all_failures.context, engine_with_all_failures.request)


def test_engine_uses_safe_default_for_invalid_plan(engine_with_invalid_planner) -> None:
    result = engine_with_invalid_planner.search(
        engine_with_invalid_planner.context,
        engine_with_invalid_planner.request,
    )
    assert {request.kind for request in result.plan.provider_requests} == {
        ProviderKind.VECTOR,
        ProviderKind.LEXICAL,
    }
    assert result.coverage["planner_fallback"] == "invalid_plan"


def test_trace_never_contains_query_or_memory_text(engine_fixture) -> None:
    result = engine_fixture.engine.search(engine_fixture.context, engine_fixture.request)
    assert engine_fixture.request.query not in repr(result.trace)
    assert "private-memory-body" not in repr(result.trace)
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest -q tests/test_product_retrieval_engine.py`

Expected: FAIL because selection, tracing, and product engine modules do not exist.

- [ ] **Step 3: Implement one-pass generic selection**

`selection.py` must:

1. Fuse candidate lists using existing `reciprocal_rank_fusion`.
2. Convert fused rank and generic signals into `utility_score` without inspecting intent/category.
3. Run the existing reranker once only when `plan.use_reranker` is true.
4. Run MMR once.
5. Produce a stable final order using selected position and ID as tie breakers.

Use this scoring function:

```python
def generic_utility(candidate, rank: int, total: int) -> float:
    rank_score = 1.0 - ((rank - 1) / max(1, total))
    signal = max(
        float(candidate.scores.get("semantic_score", 0.0)),
        float(candidate.scores.get("bm25_score", 0.0)),
        float(candidate.scores.get("exact_signal", 0.0)),
        float(candidate.scores.get("temporal_score", 0.0)),
        float(candidate.scores.get("graph_proximity", 0.0)),
    )
    return 0.60 * rank_score + 0.40 * signal
```

- [ ] **Step 4: Implement sanitized tracing**

Trace records contain provider kind, count, elapsed milliseconds, failure code, filtered count, selected IDs hashed with `stable_hash`, mode, intent label, and stage duration. They must not contain query text, candidate text, bearer tokens, endpoint URLs, or model credentials.

Add `validate_product_plan(plan)` and `ProductQueryPlanner.safe_default(request)`. The safe default contains only vector and lexical provider requests, relevance ordering, no reranker in fast mode, and intent `factual`. If planner output is not a `ProductQueryPlan` or has no provider requests, the engine uses the safe default and records `planner_fallback="invalid_plan"`. Exceptions raised while computing the plan are not swallowed.

- [ ] **Step 5: Implement product engine orchestration**

```python
class ProductRetrievalEngine:
    def search(self, context, request, plan=None):
        planned = plan or self.planner.plan(request)
        if validate_product_plan(planned):
            return self.search_with_plan(context, request, planned)
        fallback = self.planner.safe_default(request)
        result = self.search_with_plan(context, request, fallback)
        return RetrievalResult(
            candidates=result.candidates,
            coverage={**result.coverage, "planner_fallback": "invalid_plan"},
            trace={**result.trace, "planner_fallback": "invalid_plan"},
            plan=result.plan,
        )

    def search_with_plan(self, context, request, plan):
        context.check_deadline()
        outcomes = self.registry.run(context, request, plan)
        context.check_deadline()
        successful = [outcome for outcome in outcomes if outcome.failure is None]
        if not successful:
            raise RetrievalUnavailable("all planned providers failed")
        candidate_lists = [list(outcome.candidates) for outcome in successful if outcome.candidates]
        selected = select_candidates(
            request.query,
            candidate_lists,
            limit=request.limit,
            use_reranker=plan.use_reranker,
            reranker=self.reranker,
            mmr_lambda=self.mmr_lambda,
        )
        failures = [outcome.failure.error_code for outcome in outcomes if outcome.failure is not None]
        coverage = {
            "intent": plan.intent,
            "degraded": bool(failures),
            "provider_failures": failures,
            "provider_counts": {
                outcome.provider.value: len(outcome.candidates)
                for outcome in outcomes
            },
        }
        trace = build_retrieval_trace(context, request, plan, outcomes, selected)
        return RetrievalResult(tuple(selected), coverage, trace, plan)
```

If a reranker raises `EndpointUnavailable`, record `reranker_unavailable` and keep the pre-rerank selection. Let storage and programming errors propagate.

- [ ] **Step 6: Run engine tests**

Run: `pytest -q tests/test_product_retrieval_engine.py`

Expected: all engine tests pass.

- [ ] **Step 7: Commit**

```bash
git add fusion_memory/retrieval/selection.py fusion_memory/retrieval/tracing.py fusion_memory/retrieval/product_engine.py fusion_memory/retrieval/engine.py tests/test_product_retrieval_engine.py
git commit -m "feat(retrieval): add one-pass product retrieval engine"
```

---

### Task 6: Add A Category-Free Product Evidence Pack

**Files:**
- Create: `fusion_memory/retrieval/product_evidence_pack.py`
- Create: `tests/test_product_evidence_pack.py`
- Modify: `fusion_memory/retrieval/product_engine.py`

**Interfaces:**
- Consumes: repository, config, `SearchRequest`, `RetrievalResult`.
- Produces: `ProductEvidencePackBuilder.build(context, request, result, token_budget) -> EvidencePack`.
- Does not consume: legacy `QueryPlan`, BEAM category, rescue metadata, benchmark mode.

- [ ] **Step 1: Write failing evidence-pack tests**

```python
def test_product_pack_preserves_source_provenance(pack_fixture) -> None:
    pack = pack_fixture.builder.build(
        pack_fixture.context,
        pack_fixture.request,
        pack_fixture.result,
        token_budget=1200,
    )
    assert pack.source_spans[0]["id"] == "span-1"
    assert pack.source_spans[0]["session_id"] == "session-a"
    assert pack.source_spans[0]["candidate_source"] == "product_lexical"


def test_product_pack_orders_chronology_by_timeline_index(chronology_pack_fixture) -> None:
    pack = chronology_pack_fixture.build()
    assert [span["id"] for span in pack.source_spans] == ["span-early", "span-late"]


def test_product_pack_abstains_without_supported_source_evidence(empty_pack_fixture) -> None:
    pack = empty_pack_fixture.build()
    assert pack.answer_policy == "abstain_if_not_supported"
    assert "query_type" not in pack.coverage
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest -q tests/test_product_evidence_pack.py`

Expected: FAIL because `ProductEvidencePackBuilder` does not exist.

- [ ] **Step 3: Implement the minimal product pack**

Build records from selected candidate types and hydrate every `source_span_id` with:

```python
span = repository.get_span(
    span_id,
    context.scope,
    include_session=context.include_session,
)
```

Each source record contains `id`, `session_id`, `turn_id`, `speaker`, `timestamp`, `source_uri`, compacted `content`, `candidate_source`, and `source_span_ids`. Stop before exceeding `token_budget`. For chronology plans sort by `timeline_index`, timestamp, then ID; for recency plans sort by timestamp descending; otherwise retain engine rank.

Coverage contains only product concepts:

```python
coverage = {
    **result.coverage,
    "intent": result.plan.intent,
    "query_intent": result.plan.query_intent,
    "source_span_count": len(source_spans),
    "token_budget": token_budget,
    "estimated_source_tokens": estimated_tokens,
}
```

Set `answer_policy` to `abstain_if_not_supported` when no hydrated source spans exist; otherwise use `answer_with_evidence_or_abstain`.

Add the protocol method to `ProductRetrievalEngine`:

```python
def build_evidence_pack(self, context, request, result, token_budget):
    return self.pack_builder.build(context, request, result, token_budget)
```

- [ ] **Step 4: Run pack tests**

Run: `pytest -q tests/test_product_evidence_pack.py`

Expected: all product pack tests pass.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/product_evidence_pack.py fusion_memory/retrieval/product_engine.py tests/test_product_evidence_pack.py
git commit -m "feat(retrieval): build category-free product evidence packs"
```

---

### Task 7: Inject The Product Engine Into MemoryService Without Switching Defaults

**Files:**
- Modify: `fusion_memory/api/service.py:170-744`
- Modify: `fusion_memory/retrieval/engine.py`
- Create: `tests/test_memory_service_retrieval_engine.py`

**Interfaces:**
- `MemoryService.__init__` gains `retrieval_engine: RetrievalEngine | None = None`.
- A provided engine owns search and answer-context retrieval; `None` temporarily keeps the legacy path for one migration task.
- Service remains responsible for authorization, trace persistence, audit persistence, and public result mapping.

- [ ] **Step 1: Write failing injection tests**

```python
def test_memory_service_delegates_search_to_injected_engine(fake_engine, memory_store) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)
    result = service.search("Atlas database", Scope(user_id="user-a"), {"limit": 5, "mode": "fast"})
    assert fake_engine.search_calls[0].request.query == "Atlas database"
    assert result.candidates[0].id == "candidate-1"
    assert memory_store.saved_traces[-1][0] == result.trace_id


def test_memory_service_builds_pack_with_same_engine_result(fake_engine, memory_store) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)
    pack = service.answer_context("Atlas database", Scope(user_id="user-a"), {"limit": 5})
    assert fake_engine.pack_calls[0].result is fake_engine.last_result
    assert pack.source_spans[0]["id"] == "span-1"
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest -q tests/test_memory_service_retrieval_engine.py`

Expected: FAIL because `MemoryService` does not accept `retrieval_engine`.

- [ ] **Step 3: Add the injected path**

Add a private `_run_retrieval_engine` helper that:

1. Validates read scope and authorizes the operation.
2. Rejects `query_type_hint` and rejects modes outside `fast`/`balanced` only on the injected path.
3. Creates `RetrievalContext(scope, scope.user_id, now, trace_id, deadline, include_session)`.
4. Creates `SearchRequest` from product options. Preserve `enabled_sources` compatibility by translating source families to provider kinds: `raw -> vector+lexical+temporal+chronology`, `exact -> lexical`, `entities -> entity`, `facts -> vector+lexical`, `events -> vector+temporal+chronology`, `views -> lexical`, and `profiles -> lexical+entity`. Also accept the new internal `enabled_providers` option for eval tests.
5. Calls `retrieval_engine.search` once.
6. Saves the sanitized engine trace and a `memory.search` audit event with query hash, query length, intent, mode, candidate count, and model-call summary.
7. Returns both the internal result and public trace ID so search and answer-context share the same execution helper.

`search()` maps to existing `SearchResult`; `answer_context()` calls `build_evidence_pack()` on the same internal result. Do not call public `search()` from `answer_context()` on the new path.

- [ ] **Step 4: Run injection and public contract tests**

Run: `pytest -q tests/test_memory_service_retrieval_engine.py tests/test_fusion_memory.py -k 'scope_isolation or search_trace_contains or chinese_preference_query'`

Expected: injection tests pass and selected legacy contract tests remain green.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/api/service.py fusion_memory/retrieval/engine.py tests/test_memory_service_retrieval_engine.py
git commit -m "refactor(memory): inject retrieval engine behind service facade"
```

---

### Task 8: Build Eval-Owned BEAM Profile And Remove Benchmark Mode From Other Eval Adapters

**Files:**
- Create: `fusion_memory/eval/beam/__init__.py`
- Create: `fusion_memory/eval/beam/query_planner.py`
- Create: `fusion_memory/eval/beam/engine.py`
- Modify: `fusion_memory/eval/beam_adapter.py:35-150`
- Modify: `fusion_memory/eval/adapter.py:89-180`
- Modify: `fusion_memory/eval/longmemeval_adapter.py:70-175`
- Modify: `tests/test_beam_adapter.py`
- Modify: `tests/test_longmemeval_adapter.py`

**Interfaces:**
- Produces: `BeamQueryPlanner.plan(query, category, limit) -> ProductQueryPlan`.
- Produces: `BeamRetrievalEngine.answer_context(query, scope, category, budget) -> EvidencePack`.
- Beam category is passed only inside `fusion_memory/eval/beam/` and to answer/judge models.

- [ ] **Step 1: Replace the old category-hint test with an eval-profile test**

```python
def test_beam_adapter_routes_category_only_to_eval_engine() -> None:
    engine = CaptureBeamEngine()
    adapter = BeamAdapter(
        MemoryService(),
        Scope(workspace_id="w", user_id="u", agent_id="a"),
        retrieval_engine=engine,
    )
    query = EvalQuery(
        id="beam:100k:1:contradiction_resolution:0",
        query="Have I used Excel?",
        gold_answers=["There is contradictory information."],
        category="contradiction_resolution",
    )
    adapter.answer_query(query)
    assert engine.calls[0].category == "contradiction_resolution"
    assert "query_type_hint" not in repr(engine.calls[0])
```

Add tests asserting `BenchmarkAdapter.run_ablation()` and `LongMemEvalAdapter.run_ablation()` use only `fast` and `balanced`, never `benchmark`.

- [ ] **Step 2: Verify eval tests fail**

Run: `pytest -q tests/test_beam_adapter.py tests/test_longmemeval_adapter.py`

Expected: FAIL because the adapters still pass `mode="benchmark"` and `query_type_hint` to `MemoryService`.

- [ ] **Step 3: Implement `BeamQueryPlanner` as an eval decorator**

Start from `ProductQueryPlanner.plan` with a balanced `SearchRequest`, then ensure provider capabilities by category:

```python
CATEGORY_PROVIDERS = {
    "event_ordering": (ProviderKind.CHRONOLOGY, ProviderKind.TEMPORAL),
    "temporal_reasoning": (ProviderKind.TEMPORAL,),
    "contradiction_resolution": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "knowledge_update": (ProviderKind.TEMPORAL, ProviderKind.LEXICAL),
    "multi_session_reasoning": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "preference_following": (ProviderKind.LEXICAL, ProviderKind.ENTITY),
    "instruction_following": (ProviderKind.LEXICAL,),
    "information_extraction": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "summarization": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
    "abstention": (ProviderKind.LEXICAL, ProviderKind.VECTOR),
}
```

The returned object is still a `ProductQueryPlan`; the category remains a local variable in the eval engine and never enters the plan.

- [ ] **Step 4: Implement `BeamRetrievalEngine`**

The engine uses `ProductRetrievalEngine.search_with_plan`, then `ProductEvidencePackBuilder`. Add eval-only coverage fields after the product pack is built:

```python
pack.coverage["benchmark"] = "BEAM"
pack.coverage["benchmark_category"] = category
pack.coverage["query_type"] = category
```

Use the BEAM query scope with `include_session=True` so official queries remain isolated to their chat session. Large eval budgets are local to this engine: minimum retrieval limit 24 for event ordering, 50 for other categories, and minimum token budget 24000.

- [ ] **Step 5: Update adapters**

- `BeamAdapter` accepts optional `retrieval_engine`; default to a `BeamRetrievalEngine` built from the service's store/config/reranker.
- `BeamAdapter.answer_query()` calls the eval engine, not `service.answer_context()`.
- Generic and LongMemEval ablations use `fast` and `balanced` only.
- LongMemEval defaults to `balanced` plus `allow_cross_session=True`; it does not send `benchmark` or `query_type_hint`.
- LongMemEval component ablation may keep its existing `enabled_sources` sets because `MemoryService` translates those stable source-family names to product provider kinds.

- [ ] **Step 6: Run eval tests**

Run: `pytest -q tests/test_beam_adapter.py tests/test_longmemeval_adapter.py tests/test_cli_and_eval.py`

Expected: all selected eval tests pass; the old category-hint assertion is gone.

- [ ] **Step 7: Commit**

```bash
git add fusion_memory/eval/beam fusion_memory/eval/beam_adapter.py fusion_memory/eval/adapter.py fusion_memory/eval/longmemeval_adapter.py tests/test_beam_adapter.py tests/test_longmemeval_adapter.py
git commit -m "refactor(eval): isolate BEAM retrieval profile"
```

---

### Task 9: Switch Production And MCP To The Product Engine

**Files:**
- Modify: `fusion_memory/api/service.py:170-744`
- Modify: `fusion_memory/mcp_runtime.py:28-125,202-300`
- Modify: `fusion_memory/core/config.py:65-90`
- Modify: `tests/test_mcp_server.py`
- Modify: `tests/test_mcp_scope.py`
- Modify: `tests/test_runtime_config.py`
- Modify: `tests/test_memory_service_retrieval_engine.py`

**Interfaces:**
- `MemoryService()` constructs a product engine when no engine is injected.
- `FusionMemoryRuntime.search_mode` accepts only `fast` or `balanced`.
- `FUSION_MEMORY_MCP_SEARCH_MODE=benchmark` fails during startup configuration.

- [ ] **Step 1: Add failing production-mode tests**

```python
def test_memory_service_uses_product_engine_by_default() -> None:
    service = MemoryService()
    try:
        assert service.retrieval_engine.__class__.__name__ == "ProductRetrievalEngine"
    finally:
        service.close()


def test_runtime_rejects_benchmark_search_mode(fake_executor, fake_factory) -> None:
    with pytest.raises(ValueError, match="fast or balanced"):
        FusionMemoryRuntime(fake_executor, fake_factory, search_mode="benchmark")


def test_mcp_read_scope_still_contains_only_authenticated_user() -> None:
    write_scope = Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a", app_id="mcp")
    read_scope = Scope(user_id="user-a", app_id="mcp")
    assert write_scope.workspace_id == "workspace-a"
    assert read_scope.workspace_id is None
    assert read_scope.session_id is None
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest -q tests/test_memory_service_retrieval_engine.py tests/test_mcp_server.py tests/test_mcp_scope.py -k 'product_engine_by_default or benchmark_search_mode or authenticated_user'`

Expected: default-engine and runtime-mode tests fail.

- [ ] **Step 3: Add the product engine factory and switch defaults**

`build_product_retrieval_engine(repository, config, reranker, planner=None)` constructs the five providers, registry, planner, product pack builder, and `ProductRetrievalEngine`. `MemoryService.__init__` uses it when `retrieval_engine` is omitted. Keep the legacy methods physically present but unreachable until Task 10; do not expose a legacy switch through environment, API, or MCP.

- [ ] **Step 4: Remove production benchmark configuration**

- Runtime mode validation becomes `{"fast", "balanced"}`.
- Remove `benchmark_mode_rerank_top_n` from `MemoryConfig` and its snapshot/tests.
- Reject `options["query_type_hint"]` in product service methods.
- Keep MCP tool schemas unchanged; they already expose only query and limit.

- [ ] **Step 5: Run production and MCP tests**

Run: `pytest -q tests/test_memory_service_retrieval_engine.py tests/test_mcp_auth.py tests/test_mcp_scope.py tests/test_mcp_tools.py tests/test_mcp_server.py tests/test_postgres_concurrency.py`

Expected: all selected tests pass, including different-user parallelism and same-user advisory locking.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/api/service.py fusion_memory/retrieval/engine.py fusion_memory/mcp_runtime.py fusion_memory/core/config.py tests/test_memory_service_retrieval_engine.py tests/test_mcp_server.py tests/test_mcp_scope.py tests/test_runtime_config.py
git commit -m "refactor(memory): switch production to product retrieval engine"
```

---

### Task 10: Move Retained Helpers And Slim MemoryService To A Facade

**Files:**
- Modify: `fusion_memory/api/service.py:1-4175`
- Modify: `fusion_memory/api/service_helpers.py`
- Create: `fusion_memory/api/service_telemetry.py`
- Create: `fusion_memory/ingestion/order_markers.py`
- Create: `tests/test_retrieval_architecture_boundaries.py`
- Modify: `tests/test_config_and_reporting.py`
- Modify: `tests/test_fusion_memory.py`

**Interfaces:**
- `MemoryService` retains public application-service methods, authorization, trace/audit persistence, writes, history, views, and product-engine delegation.
- Service telemetry and write-side order parsing move to focused modules before retrieval helpers are deleted.

- [ ] **Step 1: Write failing static architecture tests**

```python
from pathlib import Path


PRODUCTION_ROOTS = [
    Path("fusion_memory/api"),
    Path("fusion_memory/retrieval"),
    Path("fusion_memory/mcp_runtime.py"),
]


def _production_python() -> str:
    files = []
    for root in PRODUCTION_ROOTS:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def test_production_retrieval_has_no_beam_categories_or_benchmark_mode() -> None:
    source = _production_python()
    for forbidden in (
        "query_type_hint",
        'mode == "benchmark"',
        'mode in {"balanced", "benchmark"}',
        '"contradiction_resolution"',
        '"multi_session_reasoning"',
        '"preference_following"',
        '"instruction_following"',
        '"information_extraction"',
    ):
        assert forbidden not in source


def test_product_providers_do_not_import_or_reference_memory_service() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("fusion_memory/retrieval/providers").rglob("*.py"))
    assert "fusion_memory.api.service" not in source
    assert "MemoryService" not in source
    assert "context.service" not in source


def test_memory_service_is_within_facade_size_budget() -> None:
    lines = Path("fusion_memory/api/service.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 1200
```

- [ ] **Step 2: Verify architecture tests fail**

Run: `pytest -q tests/test_retrieval_architecture_boundaries.py`

Expected: all three tests fail against the legacy files.

- [ ] **Step 3: Move the four non-retrieval helpers before deleting `service_helpers.py`**

Move `_sanitize_model_call`, `_model_call_summary`, and `_labeled_precision` to `api/service_telemetry.py`. Move `_explicit_order_mentions` and its compiled order regex to `ingestion/order_markers.py`. Update `MemoryService` imports and run:

Run: `pytest -q tests/test_config_and_reporting.py tests/test_fusion_memory.py -k 'encoding_report or explicit_event or model_call'`

Expected: selected tests pass.

- [ ] **Step 4: Remove legacy search and helper methods from MemoryService**

Delete the current legacy search body at `service.py:364-646`, the legacy answer-context body at `service.py:651-744`, and retrieval methods at `service.py:1620-3737`. Keep thin public methods that authorize, call the product engine, persist trace/audit, and map results. Delete retrieval-only imports and the temporary legacy switch.

- [ ] **Step 5: Run facade and write-side tests**

Run: `pytest -q tests/test_memory_service_retrieval_engine.py tests/test_retrieval_public_contract.py tests/test_config_and_reporting.py tests/test_fusion_memory.py -k 'scope or ingestion or report or event_edge or search_trace or answer_context'`

Expected: selected public/write/facade tests pass; the line-count architecture test passes even though the other static checks still fail on legacy retrieval modules.

- [ ] **Step 6: Commit**

```bash
git add fusion_memory/api/service.py fusion_memory/api/service_helpers.py fusion_memory/api/service_telemetry.py fusion_memory/ingestion/order_markers.py tests/test_retrieval_architecture_boundaries.py tests/test_config_and_reporting.py tests/test_fusion_memory.py
git commit -m "refactor(memory): slim memory service to application facade"
```

---

### Task 11: Canonicalize Product Modules And Delete Legacy Retrieval Files

**Files:**
- Delete: `fusion_memory/api/service_helpers.py`
- Delete: `fusion_memory/retrieval/pipeline.py`
- Delete: `fusion_memory/retrieval/candidate_provider.py`
- Delete: `fusion_memory/retrieval/providers/raw.py`
- Delete: `fusion_memory/retrieval/providers/structured.py`
- Delete: `fusion_memory/retrieval/scoring.py`
- Delete: `fusion_memory/retrieval/preservation.py`
- Delete: `fusion_memory/retrieval/raw_evidence_quota.py`
- Delete: `fusion_memory/retrieval/retrieval_trace.py`
- Delete: `fusion_memory/retrieval/pack_contract.py`
- Replace: `fusion_memory/retrieval/query_planner.py` with product planner implementation
- Replace: `fusion_memory/retrieval/evidence_pack.py` with product evidence-pack implementation
- Replace: `fusion_memory/retrieval/providers/base.py` and `fusion_memory/retrieval/providers/registry.py` with product implementations
- Modify: `fusion_memory/core/models.py:195-219`
- Modify: `fusion_memory/core/config.py`
- Modify: `fusion_memory/retrieval/product_engine.py`
- Modify: `tests/test_retrieval_architecture_boundaries.py`

**Interfaces:**
- Production `QueryPlanner` is the product planner; legacy `core.models.QueryPlan` is removed.
- Canonical `EvidencePackBuilder` is the product builder.
- Canonical provider base and registry accept repositories and product contexts, never `MemoryService`.

- [ ] **Step 1: Replace migration filenames with canonical product filenames**

```bash
git rm fusion_memory/retrieval/query_planner.py fusion_memory/retrieval/evidence_pack.py
git mv fusion_memory/retrieval/product_planner.py fusion_memory/retrieval/query_planner.py
git mv fusion_memory/retrieval/product_evidence_pack.py fusion_memory/retrieval/evidence_pack.py
git rm fusion_memory/retrieval/providers/base.py fusion_memory/retrieval/providers/registry.py
git mv fusion_memory/retrieval/providers/product_base.py fusion_memory/retrieval/providers/base.py
git mv fusion_memory/retrieval/providers/product_registry.py fusion_memory/retrieval/providers/registry.py
```

Update imports in `product_engine.py`, `engine.py`, `MemoryService`, and tests so production refers only to canonical names.

- [ ] **Step 2: Delete legacy modules and category-shaped configuration**

Delete the listed legacy pipeline/provider/preservation/pack-contract files and `service_helpers.py`. Remove `QueryPlan` from `core/models.py`. Remove `raw_evidence_quotas` and its category map if there is no remaining product consumer. A fixed product source-evidence minimum belongs in the product pack/engine config and must not be keyed by query category.

- [ ] **Step 3: Prove that all legacy types and callbacks are gone**

Run:

```bash
! grep -RIn --exclude-dir=__pycache__ -E 'QueryPlan|RetrievalExecutionContext|RecallContext|context\.service|service\._(topic|aggregation|event_ordering|preserve|apply_quality)' fusion_memory/api fusion_memory/retrieval
```

Expected: no matches. Product names such as `ProductQueryPlan` are allowed; refine the expression to `\bQueryPlan\b` if the shell grep treats the substring as a match.

- [ ] **Step 4: Run product and architecture tests**

Run: `pytest -q tests/test_retrieval_architecture_boundaries.py tests/test_product_retrieval_contracts.py tests/test_product_query_planner.py tests/test_product_retrieval_providers.py tests/test_product_retrieval_engine.py tests/test_product_evidence_pack.py tests/test_memory_service_retrieval_engine.py`

Expected: all selected tests pass; provider callback and production-category assertions are green.

- [ ] **Step 5: Commit**

```bash
git add -A fusion_memory/api/service_helpers.py fusion_memory/retrieval fusion_memory/core/models.py fusion_memory/core/config.py tests/test_retrieval_architecture_boundaries.py
git commit -m "refactor(retrieval): delete legacy benchmark pipeline"
```

---

### Task 12: Move BEAM Model Logic And Retire Implementation-Shaped Tests

**Files:**
- Modify: `fusion_memory/eval/model_adapters.py`
- Create: `fusion_memory/eval/beam/model_adapters.py`
- Modify: `fusion_memory/eval/beam_adapter.py`
- Modify: `tests/test_fusion_memory.py`
- Modify: `tests/test_retrieval_pipeline.py`
- Modify: `tests/test_recall_provider_registry.py`
- Delete: `tests/test_retrieval_preservation.py`
- Modify: `tests/test_model_adapters.py`
- Create: `tests/test_beam_profile_regressions.py`
- Modify: `tests/test_retrieval_architecture_boundaries.py`

**Interfaces:**
- Generic eval model adapters contain no BEAM category routing.
- BEAM-only deterministic answer and instruction logic lives under `fusion_memory/eval/beam/`.
- Tests assert public/product behavior or the explicit BEAM profile, not deleted private rescue methods.

- [ ] **Step 1: Add failing eval-boundary tests**

Extend `tests/test_retrieval_architecture_boundaries.py`:

```python
def test_beam_category_literals_are_confined_to_beam_eval_package() -> None:
    generic_eval = Path("fusion_memory/eval/model_adapters.py").read_text(encoding="utf-8")
    for category in (
        "contradiction_resolution",
        "multi_session_reasoning",
        "preference_following",
        "instruction_following",
        "information_extraction",
    ):
        assert category not in generic_eval
```

Run: `pytest -q tests/test_retrieval_architecture_boundaries.py -k beam_category_literals`

Expected: FAIL because category branches remain in the generic eval adapter.

- [ ] **Step 2: Move BEAM-only model helpers into `eval/beam/`**

Move category branches from `_deterministic_model_pack_answer` and `_answer_instruction`, plus their category-specific helper functions, into `eval/beam/model_adapters.py`. Keep generic OpenAI-compatible answer and judge clients in `eval/model_adapters.py`. `BeamAdapter` imports the BEAM wrapper; production and LongMemEval do not.

- [ ] **Step 3: Retire implementation-shaped tests and preserve product behavior tests**

- Delete tests that patch `_preserve_*`, `_apply_topic_scope_filter`, `_quality_fallback_candidates`, legacy provider order, or `query_type_hint`.
- Keep ingestion, scope isolation, source provenance, current view, entity persistence, event graph writing, and public trace tests.
- Replace benchmark-shaped service tests with BEAM adapter/profile tests for four representative categories: chronology, temporal reasoning, contradiction, and aggregation.
- Replace `tests/test_retrieval_pipeline.py` with product engine/trace assertions or remove cases already covered by the new focused files.
- Replace `tests/test_recall_provider_registry.py` with product registry assertions or remove duplicated legacy-equivalence cases.
- Remove `tests/test_retrieval_preservation.py`; the product engine intentionally has no preservation phase.

- [ ] **Step 4: Add representative BEAM profile regressions**

Use one small in-memory case per category group and call `BeamRetrievalEngine.answer_context` directly:

```python
@pytest.mark.parametrize(
    ("category", "memory", "query"),
    [
        ("event_ordering", "Atlas started, then deployment completed.", "List Atlas events in order."),
        ("temporal_reasoning", "Atlas deployment is July 30.", "When is Atlas deployment?"),
        ("contradiction_resolution", "Atlas first used SQLite, then switched to Qdrant.", "Did Atlas change databases?"),
        ("multi_session_reasoning", "Atlas uses Qdrant and reports use PostgreSQL.", "List the databases mentioned."),
    ],
)
def test_beam_profile_runs_without_category_in_product_plan(category, memory, query) -> None:
    service = MemoryService()
    scope = Scope(user_id="beam-user", workspace_id="beam", session_id="beam-session")
    try:
        service.add(memory, scope)
        engine = BeamRetrievalEngine.from_service(service)
        pack = engine.answer_context(query, scope, category, {"limit": 12})
        plan = engine.planner.plan(query, category, 12)
        assert pack.coverage["benchmark"] == "BEAM"
        assert pack.coverage["benchmark_category"] == category
        assert pack.source_spans
        assert not hasattr(plan, "category")
    finally:
        service.close()
```

These tests check that the profile runs; they do not enforce the old candidate order or BEAM score.

- [ ] **Step 5: Run architecture and focused regression tests**

Run: `pytest -q tests/test_retrieval_architecture_boundaries.py tests/test_product_retrieval_contracts.py tests/test_product_query_planner.py tests/test_product_retrieval_providers.py tests/test_product_retrieval_engine.py tests/test_product_evidence_pack.py tests/test_memory_service_retrieval_engine.py tests/test_beam_adapter.py`

Expected: all selected tests pass; `service.py` is at most 1200 lines.

- [ ] **Step 6: Commit**

```bash
git add -A fusion_memory/eval tests/test_fusion_memory.py tests/test_retrieval_pipeline.py tests/test_recall_provider_registry.py tests/test_retrieval_preservation.py tests/test_model_adapters.py tests/test_beam_profile_regressions.py tests/test_retrieval_architecture_boundaries.py
git commit -m "refactor(eval): confine BEAM logic and retire rescue tests"
```

---

### Task 13: Add Product Cases, Fault Injection, And Cross-Session Isolation Coverage

**Files:**
- Create: `tests/test_product_retrieval_cases.py`
- Modify: `tests/integration/test_mcp_postgres_e2e.py`
- Modify: `tests/test_postgres_concurrency.py`
- Modify: `tests/test_product_retrieval_engine.py`

**Interfaces:**
- Validates product release gates rather than old result parity.
- Validates partial model degradation without swallowing storage failures.

- [ ] **Step 1: Add the product case matrix**

Use parameterized cases with real `MemoryService` and deterministic SQLite models:

```python
@pytest.mark.parametrize(
    ("memories", "query", "required_text"),
    [
        (["I prefer Qdrant for Atlas retrieval."], "What database do I prefer for Atlas?", "Qdrant"),
        (["Atlas deployment deadline is July 30."], "When is the Atlas deployment deadline?", "July 30"),
        (["The incident started, then mitigation completed."], "List the incident events in order.", "mitigation"),
        (["The internal project code is ZINC-42."], "What is the internal project code?", "ZINC-42"),
    ],
)
def test_product_queries_retrieve_required_evidence(memories, query, required_text) -> None:
    service = MemoryService()
    scope = Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a")
    try:
        for memory in memories:
            service.add({"role": "user", "content": memory}, scope)
        result = service.search(query, Scope(user_id="user-a"), {"limit": 10, "mode": "balanced"})
        assert any(required_text.lower() in candidate.text.lower() for candidate in result.candidates)
    finally:
        service.close()
```

- [ ] **Step 2: Add same-user and different-user gates**

```python
def test_same_user_reads_across_workspace_and_session() -> None:
    service = MemoryService()
    try:
        service.add("Workspace A remembers cobalt-key.", Scope(user_id="user-a", workspace_id="a", session_id="s1"))
        result = service.search("What key was remembered?", Scope(user_id="user-a"), {"limit": 10})
        assert any("cobalt-key" in candidate.text for candidate in result.candidates)
    finally:
        service.close()


def test_different_user_never_receives_other_users_candidate() -> None:
    service = MemoryService()
    try:
        service.add("User A secret is cobalt-key.", Scope(user_id="user-a", workspace_id="a", session_id="s1"))
        result = service.search("What secret was stored?", Scope(user_id="user-b"), {"limit": 10})
        assert all("cobalt-key" not in candidate.text for candidate in result.candidates)
    finally:
        service.close()
```

- [ ] **Step 3: Add fault-injection gates**

Cover these exact outcomes:

- Vector provider raises `ProviderUnavailable("model_unavailable")`: lexical result is returned with `coverage["degraded"] is True`.
- Reranker raises `EndpointUnavailable`: pre-rerank candidates are returned and trace records `reranker_unavailable`.
- Repository raises `PostgresBackendUnavailable`: request raises and no partial candidates are returned.
- Context deadline is already expired: engine raises `TimeoutError` before provider execution.
- Trace contains counts and error codes but not query/candidate text.

- [ ] **Step 4: Extend MCP PostgreSQL integration coverage**

In `tests/integration/test_mcp_postgres_e2e.py`, write two memories for one token subject using different workspace/session headers, search without provenance headers, and assert both are visible. Create a second token subject and assert neither first-user marker appears. Reuse the existing stack and token helpers; do not add a second integration harness.

- [ ] **Step 5: Run product, integration, and concurrency tests**

Run: `pytest -q tests/test_product_retrieval_cases.py tests/test_product_retrieval_engine.py tests/test_postgres_concurrency.py`

Expected: all unit/concurrency tests pass.

Run when the documented PostgreSQL integration stack is available: `pytest -q -m integration tests/integration/test_mcp_postgres_e2e.py`

Expected: integration tests pass with same-user cross-session visibility and different-user isolation.

- [ ] **Step 6: Commit**

```bash
git add tests/test_product_retrieval_cases.py tests/test_product_retrieval_engine.py tests/test_postgres_concurrency.py tests/integration/test_mcp_postgres_e2e.py
git commit -m "test(memory): gate product retrieval isolation and degradation"
```

---

### Task 14: Update Architecture Docs And Run Final Validation

**Files:**
- Modify: `docs/fusion-memory-architecture.md`
- Modify: `docs/beam-100k-final-evaluation-report-20260617.md`
- Create: `docs/retrieval-engine-debeam-comparison-20260722.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Documents the new production engine and the eval-only BEAM profile.
- Records actual product tests, latency comparison, BEAM total/category deltas, and retained baseline artifact locations.

- [ ] **Step 1: Update active architecture documentation**

Replace the old L4 description with the final dependency graph and list the five product providers. State that `MemoryService` is a facade, production modes are `fast`/`balanced`, and BEAM categories exist only under `fusion_memory/eval/beam/`.

- [ ] **Step 2: Run static and full automated validation**

```bash
git diff --check
pytest -q
test "$(wc -l < fusion_memory/api/service.py)" -le 1200
! grep -RIn --exclude-dir=__pycache__ -E 'query_type_hint|mode == "benchmark"|context\.service|service\._(topic|aggregation|event_ordering|preserve|apply_quality)' fusion_memory/api fusion_memory/retrieval fusion_memory/mcp_runtime.py
```

Expected: diff check exits 0, full pytest reports zero failures, the line-count assertion exits 0, and the forbidden-dependency scan produces no matches.

- [ ] **Step 3: Capture product latency and result comparison**

Run the product case suite against the parent commit and current commit on the same hardware/models using:

```bash
pytest -q tests/test_product_retrieval_cases.py --durations=20
```

Record test pass counts, top-k target evidence coverage, and the slowest 20 durations in `docs/retrieval-engine-debeam-comparison-20260722.md`. The report must state the exact commit hashes and model/runtime configuration; it must not print model credentials or token values.

- [ ] **Step 4: Run the BEAM profile smoke and full comparison**

Smoke command:

```bash
pytest -q tests/test_beam_adapter.py tests/test_beam_parallel_runner.py tests/test_beam_failure_diagnostics.py
```

Full run command, using the paths documented in `AGENTS.md`:

```bash
"${FUSION_MEMORY_PYTHON:-python}" tools/beam_parallel_runner.py \
  --dataset "$BEAM_DATASET" --split 100k \
  --workspace beam_100k_rule_qwenembed_sessionized_20260612_1745 \
  --db postgresql://fusion:fusion@127.0.0.1:55433/fusion_memory \
  --output .runtime/beam-runs/retrieval_engine_debeam_20260722.json \
  query --workers 24 --progress-every 20 \
  --model-config-file "$MODEL_CONFIG_FILE" --model-timeout-seconds 300 \
  --partial-dir .runtime/beam-runs/retrieval_engine_debeam_20260722.partials \
  --diagnostic-output .runtime/beam-runs/retrieval_engine_debeam_20260722.diagnostic.json \
  --max-consecutive-answer-failures 3 --answer-failure-retries 1
```

Expected: runner exits 0 with zero answer/judge infrastructure failures. Record actual total/category scores and deltas from both retained baselines in the comparison document. Score regression is diagnostic; user isolation, product cases, and architecture boundaries remain the merge gates.

- [ ] **Step 5: Update the handoff anchor without overwriting old baselines**

Append the refactor branch commit, new report path, new run artifact paths, and an explicit note that BEAM is now an eval-only profile. Preserve the two historical baseline values and retained artifact list in `AGENTS.md` and the final evaluation report.

- [ ] **Step 6: Commit documentation and validation evidence**

```bash
git add docs/fusion-memory-architecture.md docs/beam-100k-final-evaluation-report-20260617.md docs/retrieval-engine-debeam-comparison-20260722.md AGENTS.md
git commit -m "docs(memory): record product retrieval refactor validation"
```

- [ ] **Step 7: Verify the final branch and push**

```bash
git status --short --branch
git log --oneline genuineknowledge/main..HEAD
git push genuineknowledge refactor/retrieval-engine-debeam
```

Expected: only the preserved untracked `uv.lock` appears in status, the task commits appear above the design/plan commits, and the remote branch advances successfully.
