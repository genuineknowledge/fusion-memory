# Recall Provider Registry Design

## Goal

Introduce a recall provider registry that decomposes `fusion_memory/retrieval/candidate_provider.py` without changing retrieval behavior. This is Phase 2 of the retrieval-layer refactor and builds on Phase 1 candidate lifecycle telemetry.

## Current State

`candidate_provider.py` exposes one public facade, `build_candidate_lists()`, and currently contains all recall orchestration:

- raw span recall
- topic-scoped raw recall
- broad raw recall
- scent-trail recall
- contradiction, temporal, aggregation, and event-ordering coverage
- fact/event/current-view/entity-profile recall
- exact and entity recall

This works, but the branch structure is too dense for later Rule Audit, Temporal Relation, and Graph Ordering work. There is no provider declaration layer that can explain which recall family ran, whether a provider is production default, whether it is graph-related, or which replay category it supports.

## Non-Negotiable Constraints

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

## Recommended Approach

Use a compatibility facade with a provider registry behind it.

`candidate_provider.build_candidate_lists()` remains the only function called by `MemoryService._candidate_lists()`. Internally, it constructs a `RecallContext`, asks a `ProviderRegistry` for enabled providers, and concatenates provider result lists in the exact same order as today.

This avoids changes in `MemoryService.search()` and makes Phase 2 a structural refactor rather than a retrieval behavior change.

## Core Interfaces

### `RecallContext`

Location: `fusion_memory/retrieval/providers/base.py`

Fields:

- `service: Any`
- `query: str`
- `scope: Scope`
- `plan: Any`
- `per_source_limit: int`
- `enabled_sources: set[str] | None`
- `include_session: bool`
- `event_milestone_group: Callable[[Any], str | None]`
- `prior_candidates: list[Candidate]`

`query` is needed to call existing retrieval helpers, but registry telemetry must never serialize it.

### `RecallProvider`

Location: `fusion_memory/retrieval/providers/base.py`

Required attributes:

- `provider_id: str`
- `source_family: str`
- `production_default: bool`
- `shadow_only: bool`
- `graph_related: bool`
- `supported_query_types: frozenset[str] | None`
- `output_sources: frozenset[str]`
- `replay_categories: frozenset[str]`

Required method:

- `recall(context: RecallContext) -> list[Candidate]`

Provider methods return one candidate list, not a flattened global list. If a provider has no output, it returns `[]`; the registry skips empty lists when appending.

### `ProviderRegistry`

Location: `fusion_memory/retrieval/providers/registry.py`

Responsibilities:

- Own the ordered provider list.
- Filter providers by enabled source family, query type, production/shadow mode, and graph policy.
- Execute providers in order.
- Maintain `prior_candidates` for providers that need earlier recall context, especially scent-trail recall.
- Return `list[list[Candidate]]` matching the current facade output.

The registry does not score, rank, dedupe, filter, or rescue candidates.

## Provider Decomposition

### Batch 1: Low-Risk Provider Extraction

The first implementation batch should extract providers that are mechanically close to the existing branches and have low ordering risk:

- `RawSpanProvider`
- `FactProvider`
- `EventProvider`
- `CurrentViewProvider`
- `EntityProfileProvider`
- `ExactProvider`
- `EntityProvider`

These providers cover the stable source families and establish the registry API.

### Batch 2: Typed And Contextual Provider Extraction

The second implementation batch should migrate provider families that depend on query type, prior candidates, or event-ordering guards:

- `TopicScopedRawProvider`
- `BroadRawProvider`
- `ScentTrailProvider`
- `ContradictionClaimProvider`
- `TemporalCoverageProvider`
- `AggregationCoverageProvider`
- `EventOrderingCoverageProvider`
- `EventOrderingEpisodeProvider`
- `EventOrderingTimelineProvider`

Batch 2 remains behavior-preserving. It should not introduce graph default recall.

### Deferred Graph Provider

`GraphShadowProvider` is not part of the first production registry migration. It should be declared only after graph shadow replay has a stable provider contract.

If added later, it must be:

- `shadow_only=True`
- `graph_related=True`
- excluded from production event-ordering recall unless an explicit shadow/replay option requests it

## Source Family Mapping

Provider filtering must preserve current `enabled_sources` behavior:

- `raw`: raw span, topic-scoped raw, broad raw, scent trail, contradiction claim, temporal coverage, aggregation coverage, event-ordering coverage, event-ordering episode, event-ordering timeline
- `facts`: fact recall
- `events`: event recall
- `views`: current view recall
- `profiles`: entity profile recall
- `exact`: exact answer candidates
- `entities`: entity candidates

`enabled_sources=None` means all production-default providers are eligible, subject to query-type checks.

## Event Ordering Safety

Event ordering is the highest-risk category. The registry must preserve these existing behaviors:

- production event-ordering recall excludes `event_ordering_persisted_graph`
- production event-ordering recall excludes sources starting with `event_ordering_graph`
- legacy event-ordering event, episode, and timeline recall remain enabled
- dual graph-order + legacy-recall remains shadow-only

Provider tests must include a default event-ordering search that proves graph candidates are not selected.

## Telemetry And Lifecycle

Phase 2 does not need to change lifecycle output schema. Provider ids may be used internally and may later become lifecycle `reason_code` values, but only if this does not alter raw-text safety.

Provider registry telemetry, if exposed in tests or traces, may include only structural fields:

- provider id
- source family
- query type
- output count
- output source counts
- production/shadow flags

It must not include query text, candidate text, source span content, prompt text, or raw metadata.

## Testing Strategy

Phase 2 must use parity-focused tests.

Required tests:

- registry returns providers in deterministic order
- disabled source families remove only the same candidate groups as before
- batch 1 provider output matches old `build_candidate_lists()` output for raw/facts/events/views/profiles/exact/entities fixtures
- full `build_candidate_lists()` facade returns the same candidate group count and same candidate ids/sources as the pre-registry implementation for representative fixtures
- default event-ordering search still excludes graph candidates
- candidate lifecycle and replay sanitization tests still pass
- product-safe audit test still passes

Recommended broad gate:

```bash
python3 -m unittest \
  tests.test_recall_provider_registry \
  tests.test_retrieval_pipeline \
  tests.test_runtime_config \
  tests.test_fusion_memory \
  tests.test_beam_retrieval_replay \
  tests.test_rule_registry \
  tests.test_config_and_reporting \
  tests.test_product_cli \
  tests.test_event_ordering_graph \
  tests.test_chronology_selector \
  -v
```

## Success Criteria

Phase 2 succeeds when:

- `candidate_provider.py` is a compatibility facade over `ProviderRegistry`
- provider declarations exist for production recall families
- candidate list output parity is tested for representative source families
- default event-ordering behavior remains legacy-first and graph-safe
- Phase 1 lifecycle and replay tests still pass
- no raw user text is introduced into provider telemetry, replay, audit, or lifecycle outputs

## Out Of Scope

- Removing old event-ordering rules
- Promoting graph recall to default
- Changing ranking, scoring, quotas, MMR, reranking, preservation, or filtering
- Enabling realtime LLM extractor/router
- Product installer or OpenClaw/Hermes changes
