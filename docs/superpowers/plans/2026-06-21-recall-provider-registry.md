# Recall Provider Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor recall orchestration behind a provider registry while preserving `build_candidate_lists()` behavior and all production retrieval defaults.

**Architecture:** Keep `fusion_memory/retrieval/candidate_provider.py` as the compatibility facade used by `MemoryService`. Add `fusion_memory/retrieval/providers/` with provider declarations and an ordered registry. Migrate providers in batches, with parity tests comparing facade output against the pre-registry behavior.

**Tech Stack:** Python 3.11+/3.12, `unittest`, existing `MemoryService`, existing `Candidate` / `Scope` / `QueryPlan`, existing provider helper methods, existing Phase 1 lifecycle and replay tests.

## Global Constraints

- Preserve `build_candidate_lists()` public signature and return shape.
- Preserve provider execution order and candidate ordering.
- Preserve existing candidate ids, types, sources, scores, source span ids, and metadata.
- Legacy event ordering remains the production default.
- Graph, dual, and hybrid paths remain shadow/replay/flag-only.
- Do not delete legacy event ordering code.
- Do not tune ranking, scoring, quotas, MMR, reranking, preservation, filtering, or evidence packing.
- LLM extractor and LLM router remain out of realtime retrieval.
- No raw user text may be stored in provider registry telemetry, lifecycle trace, rule telemetry, replay artifacts, or audit outputs.
- Product-facing errors remain beginner-safe and must not expose tracebacks.
- Existing OpenClaw/Hermes integration remains external; do not modify host source trees.

---

## File Structure

- Create: `fusion_memory/retrieval/providers/__init__.py`
  - Exports provider interfaces and default registry factory.
- Create: `fusion_memory/retrieval/providers/base.py`
  - Owns `RecallContext`, `RecallProvider`, and `provider_applies_to_query_type()`.
- Create: `fusion_memory/retrieval/providers/registry.py`
  - Owns `ProviderRegistry`, source/query filtering, and ordered execution.
- Create: `fusion_memory/retrieval/providers/raw.py`
  - Owns raw-family providers: raw span, topic-scoped raw, broad raw, scent trail, contradiction, temporal, aggregation, and event-ordering raw/episode/timeline coverage.
- Create: `fusion_memory/retrieval/providers/structured.py`
  - Owns structured providers: facts, events, current views, entity profiles, exact, and entities.
- Modify: `fusion_memory/retrieval/candidate_provider.py`
  - Becomes a facade over `default_provider_registry()` while retaining `_event_ordering_production_candidate()`.
- Create: `tests/test_recall_provider_registry.py`
  - Unit and integration parity tests for provider order, filtering, and facade output.
- Modify: `tests/test_retrieval_pipeline.py` only if provider telemetry needs an existing retrieval smoke assertion.

---

### Task 1: Provider Interfaces And Registry Skeleton

**Files:**
- Create: `fusion_memory/retrieval/providers/__init__.py`
- Create: `fusion_memory/retrieval/providers/base.py`
- Create: `fusion_memory/retrieval/providers/registry.py`
- Create: `tests/test_recall_provider_registry.py`

**Interfaces:**
- Produces: `RecallContext`
- Produces: `RecallProvider`
- Produces: `ProviderRegistry`
- Produces: `ProviderRegistry.enabled_providers(context: RecallContext) -> list[RecallProvider]`
- Produces: `ProviderRegistry.recall(context: RecallContext) -> list[list[Candidate]]`

- [ ] **Step 1: Write failing registry skeleton tests**

Create `tests/test_recall_provider_registry.py`:

```python
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from fusion_memory.core.models import Candidate, QueryPlan, Scope
from fusion_memory.retrieval.providers.base import RecallContext, RecallProvider
from fusion_memory.retrieval.providers.registry import ProviderRegistry


@dataclass(frozen=True)
class DummyProvider:
    provider_id: str
    source_family: str
    output_source: str
    supported_query_types: frozenset[str] | None = None
    production_default: bool = True
    shadow_only: bool = False
    graph_related: bool = False

    @property
    def output_sources(self) -> frozenset[str]:
        return frozenset({self.output_source})

    @property
    def replay_categories(self) -> frozenset[str]:
        return frozenset()

    def recall(self, context: RecallContext) -> list[Candidate]:
        return [
            Candidate(
                id=self.provider_id,
                type="span",
                text=f"candidate text for {self.provider_id}",
                source=self.output_source,
                scores={"score": 1.0},
                source_span_ids=[self.provider_id],
                metadata={},
            )
        ]


class RecallProviderRegistryTests(unittest.TestCase):
    def _context(
        self,
        *,
        query_type: str = "fact_lookup",
        enabled_sources: set[str] | None = None,
    ) -> RecallContext:
        return RecallContext(
            service=object(),
            query="raw private query should not be serialized",
            scope=Scope(workspace_id="w", user_id="u", agent_id="a"),
            plan=QueryPlan(query="q", query_type=query_type, entities=[], time_constraints=[]),
            per_source_limit=5,
            enabled_sources=enabled_sources,
            include_session=False,
            event_milestone_group=lambda event: None,
            prior_candidates=[],
        )

    def test_registry_filters_by_source_family_and_query_type_in_order(self) -> None:
        raw = DummyProvider("raw_span", "raw", "l0_raw_hybrid")
        facts = DummyProvider("facts", "facts", "l1_fact_hybrid")
        temporal = DummyProvider("temporal", "raw", "temporal_coverage", frozenset({"temporal_lookup"}))
        registry = ProviderRegistry([raw, facts, temporal])

        providers = registry.enabled_providers(self._context(enabled_sources={"raw"}))

        self.assertEqual([provider.provider_id for provider in providers], ["raw_span"])

    def test_registry_recall_preserves_provider_order_and_prior_candidates(self) -> None:
        first = DummyProvider("first", "raw", "l0_raw_hybrid")
        second = DummyProvider("second", "raw", "raw_scent_trail")
        registry = ProviderRegistry([first, second])
        context = self._context()

        lists = registry.recall(context)

        self.assertEqual([[candidate.id for candidate in items] for items in lists], [["first"], ["second"]])
        self.assertEqual([candidate.id for candidate in context.prior_candidates], ["first", "second"])

    def test_registry_summary_is_structural_without_query_text(self) -> None:
        registry = ProviderRegistry([DummyProvider("raw_span", "raw", "l0_raw_hybrid")])
        context = self._context()
        registry.recall(context)

        summary = registry.summary(context)

        self.assertEqual(summary[0]["provider_id"], "raw_span")
        self.assertEqual(summary[0]["source_family"], "raw")
        self.assertEqual(summary[0]["output_count"], 1)
        self.assertNotIn("raw private query", repr(summary))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_recall_provider_registry -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_memory.retrieval.providers'`.

- [ ] **Step 3: Implement provider interfaces**

Create `fusion_memory/retrieval/providers/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from fusion_memory.core.models import Candidate, Scope


@dataclass
class RecallContext:
    service: Any
    query: str
    scope: Scope
    plan: Any
    per_source_limit: int
    enabled_sources: set[str] | None
    include_session: bool
    event_milestone_group: Callable[[Any], str | None]
    prior_candidates: list[Candidate] = field(default_factory=list)


class RecallProvider(Protocol):
    provider_id: str
    source_family: str
    production_default: bool
    shadow_only: bool
    graph_related: bool
    supported_query_types: frozenset[str] | None
    output_sources: frozenset[str]
    replay_categories: frozenset[str]

    def recall(self, context: RecallContext) -> list[Candidate]:
        ...


def provider_applies_to_query_type(provider: RecallProvider, query_type: str) -> bool:
    supported = provider.supported_query_types
    return supported is None or query_type in supported
```

Create `fusion_memory/retrieval/providers/registry.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fusion_memory.core.models import Candidate
from fusion_memory.retrieval.providers.base import RecallContext, RecallProvider, provider_applies_to_query_type


class ProviderRegistry:
    def __init__(self, providers: Iterable[RecallProvider]) -> None:
        self._providers = list(providers)
        self._last_summary: dict[int, list[dict[str, Any]]] = {}

    @property
    def providers(self) -> list[RecallProvider]:
        return list(self._providers)

    def enabled_providers(self, context: RecallContext) -> list[RecallProvider]:
        enabled = context.enabled_sources
        query_type = str(getattr(context.plan, "query_type", ""))
        out: list[RecallProvider] = []
        for provider in self._providers:
            if provider.shadow_only:
                continue
            if not provider.production_default:
                continue
            if enabled is not None and provider.source_family not in enabled:
                continue
            if not provider_applies_to_query_type(provider, query_type):
                continue
            out.append(provider)
        return out

    def recall(self, context: RecallContext) -> list[list[Candidate]]:
        candidate_lists: list[list[Candidate]] = []
        summary: list[dict[str, Any]] = []
        for provider in self.enabled_providers(context):
            candidates = provider.recall(context)
            if candidates:
                candidate_lists.append(candidates)
                context.prior_candidates.extend(candidates)
            summary.append(
                {
                    "provider_id": provider.provider_id,
                    "source_family": provider.source_family,
                    "output_count": len(candidates),
                    "output_source_counts": _source_counts(candidates),
                    "production_default": bool(provider.production_default),
                    "shadow_only": bool(provider.shadow_only),
                    "graph_related": bool(provider.graph_related),
                }
            )
        self._last_summary[id(context)] = summary
        return candidate_lists

    def summary(self, context: RecallContext) -> list[dict[str, Any]]:
        return [dict(item) for item in self._last_summary.get(id(context), [])]


def _source_counts(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate.source)
        counts[source] = counts.get(source, 0) + 1
    return counts
```

Create `fusion_memory/retrieval/providers/__init__.py`:

```python
from fusion_memory.retrieval.providers.base import RecallContext, RecallProvider
from fusion_memory.retrieval.providers.registry import ProviderRegistry

__all__ = ["ProviderRegistry", "RecallContext", "RecallProvider"]
```

- [ ] **Step 4: Run green test**

Run:

```bash
python3 -m unittest tests.test_recall_provider_registry -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/providers tests/test_recall_provider_registry.py
git commit -m "feat: add recall provider registry skeleton"
```

---

### Task 2: Structured Providers

**Files:**
- Create: `fusion_memory/retrieval/providers/structured.py`
- Modify: `tests/test_recall_provider_registry.py`

**Interfaces:**
- Consumes: `RecallContext`
- Produces: `FactProvider`
- Produces: `EventProvider`
- Produces: `CurrentViewProvider`
- Produces: `EntityProfileProvider`
- Produces: `ExactProvider`
- Produces: `EntityProvider`

- [ ] **Step 1: Write failing structured provider tests**

Append to `tests/test_recall_provider_registry.py`:

```python
from datetime import datetime, timezone
from types import SimpleNamespace

from fusion_memory.retrieval.providers.structured import (
    CurrentViewProvider,
    EntityProfileProvider,
    EntityProvider,
    EventProvider,
    ExactProvider,
    FactProvider,
)


class StructuredProviderTests(unittest.TestCase):
    def _service(self) -> Any:
        service = SimpleNamespace()
        service._retrieval_query = lambda query, plan, source: f"{source}:{query}"
        service._plan_uses_source = lambda plan, source: True
        service._event_ordering_observed_at = lambda event: event.time_start
        service._exact_candidates = lambda query, scope, limit, plan=None, include_session=False: [
            Candidate("exact-1", "span", "exact text", "exact_answer", {"score": 1.0}, ["s-exact"], {})
        ]
        service._entity_candidates = lambda query, scope, limit, include_session=False: [
            Candidate("entity-1", "entity", "entity text", "entity_graph", {"score": 1.0}, ["s-entity"], {})
        ]
        fact = SimpleNamespace(
            fact_id="fact-1",
            text="fact text",
            source_span_ids=["s-fact"],
            category="preference",
            confidence=0.9,
        )
        event = SimpleNamespace(
            event_id="event-1",
            description="event text",
            source_span_ids=["s-event"],
            event_type="decision",
            time_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        view = SimpleNamespace(
            view_id="view-1",
            text="view text",
            source_span_ids=["s-view"],
            view_type="current",
            confidence=0.8,
        )
        profile = SimpleNamespace(
            profile_id="profile-1",
            text="profile text",
            source_span_ids=["s-profile"],
            profile_type="person",
            support_count=2,
        )
        service.store = SimpleNamespace(
            search_facts=lambda *args, **kwargs: [(fact, {"score": 0.7})],
            search_events=lambda *args, **kwargs: [(event, {"score": 0.6})],
            list_current_views=lambda *args, **kwargs: [view],
            search_entity_profiles=lambda *args, **kwargs: [(profile, {"score": 0.5})],
        )
        return service

    def _context(self, *, query_type: str = "fact_lookup") -> RecallContext:
        return RecallContext(
            service=self._service(),
            query="Atlas retrieval",
            scope=Scope(workspace_id="w", user_id="u", agent_id="a"),
            plan=QueryPlan(query="Atlas retrieval", query_type=query_type, entities=[], time_constraints=[]),
            per_source_limit=3,
            enabled_sources=None,
            include_session=False,
            event_milestone_group=lambda event: "decision",
            prior_candidates=[],
        )

    def test_structured_providers_emit_current_candidate_sources(self) -> None:
        context = self._context()
        providers = [FactProvider(), EventProvider(), CurrentViewProvider(), EntityProfileProvider(), ExactProvider(), EntityProvider()]

        output = [[candidate.source for candidate in provider.recall(context)] for provider in providers]

        self.assertEqual(
            output,
            [["l1_fact_hybrid"], ["l2_event_graph"], ["l3_current_view"], ["l3_entity_profile"], ["exact_answer"], ["entity_graph"]],
        )

    def test_event_provider_preserves_event_ordering_source(self) -> None:
        context = self._context(query_type="event_ordering")
        context.service._event_ordering_event_candidates = lambda *args, **kwargs: [
            (
                SimpleNamespace(
                    event_id="event-order-1",
                    description="event order text",
                    source_span_ids=["s-event-order"],
                    event_type="milestone",
                    time_start=datetime(2026, 6, 2, tzinfo=timezone.utc),
                ),
                {"score": 0.8},
            )
        ]

        candidates = EventProvider().recall(context)

        self.assertEqual(candidates[0].source, "event_timeline_graph")
        self.assertEqual(candidates[0].scores["graph_proximity"], 0.80)
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_recall_provider_registry.StructuredProviderTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_memory.retrieval.providers.structured'`.

- [ ] **Step 3: Implement structured providers**

Create `fusion_memory/retrieval/providers/structured.py` with provider classes that move the existing fact/event/view/profile/exact/entity logic from `candidate_provider.py` without changing candidate construction.

Use this shared base:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fusion_memory.core.models import Candidate
from fusion_memory.core.text import keyword_score
from fusion_memory.retrieval.providers.base import RecallContext


@dataclass(frozen=True)
class _ProviderMeta:
    provider_id: str
    source_family: str
    output_sources: frozenset[str]
    supported_query_types: frozenset[str] | None = None
    production_default: bool = True
    shadow_only: bool = False
    graph_related: bool = False
    replay_categories: frozenset[str] = frozenset()
```

Each concrete class should expose the attributes by assigning `meta = _ProviderMeta(...)` and properties that return `meta` fields, or by class attributes if simpler. Keep code small and direct; do not introduce a framework beyond what Task 1 requires.

Implementation details must match current `candidate_provider.py`:

- `FactProvider`: calls `service.store.search_facts(service._retrieval_query(..., "facts"), ...)`; emits `l1_fact_hybrid`.
- `EventProvider`: for `event_ordering`, calls `service._event_ordering_event_candidates(...)`; otherwise calls `store.search_events(...)`; emits `event_timeline_graph` for event ordering and `l2_event_graph` otherwise.
- `CurrentViewProvider`: calls `store.list_current_views(...)`; computes `keyword_score(query, view.text)` twice as current code does; emits `l3_current_view`.
- `EntityProfileProvider`: calls `store.search_entity_profiles(...)`; emits `l3_entity_profile`.
- `ExactProvider`: calls `service._exact_candidates(service._retrieval_query(..., "exact"), ...)`.
- `EntityProvider`: calls `service._entity_candidates(service._retrieval_query(..., "entities"), ...)`.

- [ ] **Step 4: Run green test**

Run:

```bash
python3 -m unittest tests.test_recall_provider_registry.StructuredProviderTests -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/providers/structured.py tests/test_recall_provider_registry.py
git commit -m "feat: add structured recall providers"
```

---

### Task 3: Raw And Typed Providers

**Files:**
- Create: `fusion_memory/retrieval/providers/raw.py`
- Modify: `tests/test_recall_provider_registry.py`

**Interfaces:**
- Produces raw-family provider classes listed in the spec.
- Consumes: `RecallContext.prior_candidates` for `ScentTrailProvider`.
- Consumes: `_event_ordering_production_candidate()` behavior without importing from `candidate_provider.py`.

- [ ] **Step 1: Write failing raw provider tests**

Append tests that instantiate raw providers with a fake service and assert:

```python
from fusion_memory.retrieval.providers.raw import (
    AggregationCoverageProvider,
    BroadRawProvider,
    ContradictionClaimProvider,
    EventOrderingCoverageProvider,
    EventOrderingEpisodeProvider,
    EventOrderingTimelineProvider,
    RawSpanProvider,
    ScentTrailProvider,
    TemporalCoverageProvider,
    TopicScopedRawProvider,
)


class RawProviderTests(unittest.TestCase):
    def _context(self, query_type: str = "fact_lookup") -> RecallContext:
        service = SimpleNamespace()
        service._retrieval_query = lambda query, plan, source: f"{source}:{query}"
        service.store = SimpleNamespace(
            search_spans=lambda *args, **kwargs: [
                (
                    SimpleNamespace(
                        span_id="span-1",
                        content="raw text",
                        speaker="user",
                        span_type="turn",
                        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
                    ),
                    {"score": 0.9},
                )
            ]
        )
        service._topic_scoped_raw_candidates = lambda *args, **kwargs: [
            Candidate("topic-1", "span", "topic text", "topic_scoped_raw", {"score": 0.8}, ["topic-1"], {})
        ]
        service._broad_raw_recall_candidates = lambda *args, **kwargs: [
            Candidate("broad-1", "span", "broad text", "broad_raw_recall", {"score": 0.7}, ["broad-1"], {})
        ]
        service._raw_scent_trail_candidates = lambda query, scope, plan, prior, **kwargs: [
            Candidate(f"scent-{len(prior)}", "span", "scent text", "raw_scent_trail", {"score": 0.6}, ["scent-1"], {})
        ]
        service._contradiction_claim_candidates = lambda *args, **kwargs: [
            Candidate("claim-1", "claim", "claim text", "contradiction_claim_positive", {"score": 0.5}, ["claim-1"], {})
        ]
        service._temporal_coverage_candidates = lambda *args, **kwargs: [
            Candidate("temporal-1", "span", "temporal text", "temporal_coverage", {"score": 0.5}, ["temporal-1"], {})
        ]
        service._aggregation_coverage_candidates = lambda *args, **kwargs: [
            Candidate("aggregation-1", "span", "aggregation text", "aggregation_coverage", {"score": 0.5}, ["aggregation-1"], {})
        ]
        service._event_ordering_coverage_candidates = lambda *args, **kwargs: [
            Candidate("graph-1", "event", "graph text", "event_ordering_graph_shadow", {"score": 0.9}, ["graph-1"], {}),
            Candidate("legacy-1", "event", "legacy text", "event_ordering_coverage", {"score": 0.8}, ["legacy-1"], {}),
        ]
        service._event_ordering_episode_recall_candidates = lambda *args, **kwargs: [
            Candidate("episode-1", "span", "episode text", "event_ordering_episode_recall", {"score": 0.7}, ["episode-1"], {})
        ]
        service._event_ordering_timeline_candidates = lambda *args, **kwargs: [
            Candidate("timeline-1", "event", "timeline text", "event_ordering_timeline", {"score": 0.6}, ["timeline-1"], {})
        ]
        return RecallContext(
            service=service,
            query="Atlas retrieval",
            scope=Scope(workspace_id="w", user_id="u", agent_id="a"),
            plan=QueryPlan(query="Atlas retrieval", query_type=query_type, entities=[], time_constraints=[]),
            per_source_limit=3,
            enabled_sources=None,
            include_session=False,
            event_milestone_group=lambda event: None,
            prior_candidates=[],
        )

    def test_raw_provider_sources_match_current_behavior(self) -> None:
        context = self._context()
        providers = [RawSpanProvider(), TopicScopedRawProvider(), BroadRawProvider()]

        output = [[candidate.source for candidate in provider.recall(context)] for provider in providers]

        self.assertEqual(output, [["l0_raw_hybrid"], ["topic_scoped_raw"], ["broad_raw_recall"]])

    def test_scent_trail_uses_prior_candidates(self) -> None:
        context = self._context()
        context.prior_candidates.append(Candidate("prior", "span", "prior", "l0_raw_hybrid", {}, ["prior"], {}))

        candidates = ScentTrailProvider().recall(context)

        self.assertEqual(candidates[0].id, "scent-1")

    def test_event_ordering_coverage_filters_graph_candidates_in_production(self) -> None:
        context = self._context(query_type="event_ordering")

        candidates = EventOrderingCoverageProvider().recall(context)

        self.assertEqual([candidate.id for candidate in candidates], ["legacy-1"])
```

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_recall_provider_registry.RawProviderTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fusion_memory.retrieval.providers.raw'`.

- [ ] **Step 3: Implement raw providers**

Create `fusion_memory/retrieval/providers/raw.py` by moving raw-family logic from `candidate_provider.py` into provider classes. Keep helper `_event_ordering_production_candidate()` local in this module or import a shared helper from `candidate_provider.py` only if that does not create a circular import.

Provider query type constraints:

- `ContradictionClaimProvider.supported_query_types = frozenset({"contradiction_resolution"})`
- `TemporalCoverageProvider.supported_query_types = frozenset({"temporal_lookup"})`
- `AggregationCoverageProvider.supported_query_types = frozenset({"multi_session_reasoning"})`
- event-ordering providers support `frozenset({"event_ordering"})`
- raw span, topic-scoped raw, broad raw, scent trail support all query types

- [ ] **Step 4: Run green test**

Run:

```bash
python3 -m unittest tests.test_recall_provider_registry.RawProviderTests -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/providers/raw.py tests/test_recall_provider_registry.py
git commit -m "feat: add raw recall providers"
```

---

### Task 4: Candidate Provider Facade Migration

**Files:**
- Modify: `fusion_memory/retrieval/candidate_provider.py`
- Modify: `fusion_memory/retrieval/providers/__init__.py`
- Modify: `fusion_memory/retrieval/providers/registry.py`
- Modify: `tests/test_recall_provider_registry.py`

**Interfaces:**
- Produces: `default_provider_registry() -> ProviderRegistry`
- Preserves: `build_candidate_lists(...) -> list[list[Candidate]]`

- [ ] **Step 1: Write failing facade parity tests**

Add to `tests/test_recall_provider_registry.py`:

```python
from fusion_memory.api.service import MemoryService
from fusion_memory.retrieval.candidate_provider import build_candidate_lists
from fusion_memory.retrieval.event_graph_selection import _event_milestone_group


def _candidate_signature(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.id,
        candidate.type,
        candidate.source,
        tuple(candidate.source_span_ids),
        tuple(sorted((str(key), repr(value)) for key, value in candidate.scores.items())),
        tuple(sorted((str(key), repr(value)) for key, value in candidate.metadata.items())),
    )


class CandidateProviderFacadeParityTests(unittest.TestCase):
    def test_facade_returns_expected_source_order_for_mixed_memory(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-provider-parity", user_id="u", agent_id="a")
        try:
            memory.add("I prefer PostgreSQL for Atlas retrieval. I also mentioned Qdrant as a backup.", scope)
            plan = memory.planner.plan("What retrieval database do I prefer?")
            lists = build_candidate_lists(
                memory,
                "What retrieval database do I prefer?",
                scope,
                plan,
                per_source_limit=6,
                event_milestone_group=_event_milestone_group,
            )
        finally:
            memory.close()

        sources = [[candidate.source for candidate in items] for items in lists]
        self.assertTrue(any("l0_raw_hybrid" in group for group in sources))
        self.assertTrue(any("l1_fact_hybrid" in group for group in sources))
        self.assertTrue(any("l3_current_view" in group for group in sources))

    def test_facade_enabled_sources_keeps_exact_source_filtering(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-provider-enabled", user_id="u", agent_id="a")
        try:
            memory.add("Atlas retrieval uses PostgreSQL.", scope)
            plan = memory.planner.plan("What does Atlas retrieval use?")
            lists = build_candidate_lists(
                memory,
                "What does Atlas retrieval use?",
                scope,
                plan,
                per_source_limit=6,
                enabled_sources={"facts"},
                event_milestone_group=_event_milestone_group,
            )
        finally:
            memory.close()

        self.assertTrue(lists)
        self.assertTrue(all(candidate.source == "l1_fact_hybrid" for items in lists for candidate in items))

    def test_facade_default_event_ordering_excludes_graph_candidates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-provider-event-ordering", user_id="u", agent_id="a")
        try:
            memory.add("First I planned Atlas. Then I implemented raw recall. Finally I reviewed graph ordering.", scope)
            plan = memory.planner.plan("What happened in order for Atlas?", query_type_hint="event_ordering")
            lists = build_candidate_lists(
                memory,
                "What happened in order for Atlas?",
                scope,
                plan,
                per_source_limit=8,
                event_milestone_group=_event_milestone_group,
            )
        finally:
            memory.close()

        sources = [candidate.source for items in lists for candidate in items]
        self.assertNotIn("event_ordering_persisted_graph", sources)
        self.assertFalse(any(source.startswith("event_ordering_graph") for source in sources))
```

- [ ] **Step 2: Run tests before migration**

Run:

```bash
python3 -m unittest tests.test_recall_provider_registry.CandidateProviderFacadeParityTests -v
```

Expected: PASS before migration. These are characterization tests.

- [ ] **Step 3: Implement default provider registry and facade**

In `providers/registry.py`, add:

```python
from fusion_memory.retrieval.providers.raw import (
    AggregationCoverageProvider,
    BroadRawProvider,
    ContradictionClaimProvider,
    EventOrderingCoverageProvider,
    EventOrderingEpisodeProvider,
    EventOrderingTimelineProvider,
    RawSpanProvider,
    ScentTrailProvider,
    TemporalCoverageProvider,
    TopicScopedRawProvider,
)
from fusion_memory.retrieval.providers.structured import (
    CurrentViewProvider,
    EntityProfileProvider,
    EntityProvider,
    EventProvider,
    ExactProvider,
    FactProvider,
)


def default_provider_registry() -> ProviderRegistry:
    return ProviderRegistry(
        [
            RawSpanProvider(),
            TopicScopedRawProvider(),
            BroadRawProvider(),
            ScentTrailProvider(),
            ContradictionClaimProvider(),
            TemporalCoverageProvider(),
            AggregationCoverageProvider(),
            EventOrderingCoverageProvider(),
            EventOrderingEpisodeProvider(),
            EventOrderingTimelineProvider(),
            FactProvider(),
            EventProvider(),
            CurrentViewProvider(),
            EntityProfileProvider(),
            ExactProvider(),
            EntityProvider(),
        ]
    )
```

In `candidate_provider.py`, replace the function body with:

```python
    enabled = set(enabled_sources) if enabled_sources is not None else None
    context = RecallContext(
        service=service,
        query=query,
        scope=scope,
        plan=plan,
        per_source_limit=per_source_limit,
        enabled_sources=enabled,
        include_session=include_session,
        event_milestone_group=event_milestone_group,
        prior_candidates=[],
    )
    return default_provider_registry().recall(context)
```

Remove imports that are no longer used from `candidate_provider.py`. Keep `_event_ordering_production_candidate()` only if another module imports it; otherwise move it into `providers/raw.py`.

- [ ] **Step 4: Run facade and regression tests**

Run:

```bash
python3 -m unittest \
  tests.test_recall_provider_registry \
  tests.test_retrieval_pipeline \
  tests.test_runtime_config \
  tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_default_search_does_not_select_graph_candidates \
  tests.test_config_and_reporting.ConfigAndReportingTests.test_search_audit_event_does_not_store_raw_query_text \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fusion_memory/retrieval/candidate_provider.py fusion_memory/retrieval/providers tests/test_recall_provider_registry.py
git commit -m "refactor: route candidate provider through registry"
```

---

### Task 5: Phase 2 Verification Gate

**Files:**
- Modify only if tests expose issues.

**Interfaces:**
- Verifies provider registry migration preserves retrieval behavior.

- [ ] **Step 1: Run focused provider/retrieval tests**

Run:

```bash
python3 -m unittest \
  tests.test_recall_provider_registry \
  tests.test_candidate_lifecycle \
  tests.test_retrieval_pipeline \
  tests.test_beam_retrieval_replay \
  tests.test_runtime_config \
  tests.test_config_and_reporting \
  tests.test_rule_registry \
  tests.test_fusion_memory.FusionMemoryTests.test_event_ordering_default_search_does_not_select_graph_candidates \
  tests.test_fusion_memory.FusionMemoryTests.test_dual_shadow_does_not_replace_event_ordering_selected_candidates \
  tests.test_product_cli.ProductCliTests.test_upgrade_failure_json_is_beginner_safe_without_raw_subprocess_output \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run broad suite**

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

- [ ] **Step 3: Commit verification notes only if files changed**

If tests require fixes, commit those fixes with a focused message. If no files changed, do not create an empty commit.

---

## Later Phase Notes

After Phase 2 passes:

- Phase 3 should use provider ids and lifecycle output as rule audit dimensions.
- Phase 4 should introduce temporal relation objects behind shadow selectors.
- Phase 5 should continue graph topic clustering and dual graph-order + legacy-recall shadow evaluation.
- Phase 6 should move `MemoryService.search()` orchestration into real pipeline execution units after provider registry has stabilized.
