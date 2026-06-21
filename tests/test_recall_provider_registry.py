from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fusion_memory.api.service import MemoryService
from fusion_memory.core.models import Candidate, QueryPlan, Scope
from fusion_memory.core.text import keyword_score
from fusion_memory.retrieval.candidate_provider import build_candidate_lists
from fusion_memory.retrieval.event_graph_selection import _event_milestone_group
from fusion_memory.retrieval.providers.base import RecallContext
from fusion_memory.retrieval.providers.registry import ProviderRegistry
from fusion_memory.retrieval.providers.structured import (
    CurrentViewProvider,
    EntityProfileProvider,
    EntityProvider,
    EventProvider,
    ExactProvider,
    FactProvider,
)
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


def _candidate_signature(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.id,
        candidate.type,
        candidate.source,
        tuple(candidate.source_span_ids),
        tuple(sorted((str(key), repr(value)) for key, value in candidate.scores.items())),
        tuple(sorted((str(key), repr(value)) for key, value in candidate.metadata.items())),
    )


def _legacy_build_candidate_lists(
    service: Any,
    query: str,
    scope: Scope,
    plan: Any,
    per_source_limit: int,
    enabled_sources: list[str] | set[str] | None = None,
    include_session: bool = False,
    *,
    event_milestone_group: Any,
) -> list[list[Candidate]]:
    enabled = set(enabled_sources) if enabled_sources is not None else None
    candidate_lists: list[list[Candidate]] = []
    speaker = plan.speaker_focus if plan.speaker_focus != "any" else None
    if service._source_enabled("raw", enabled):
        raw_span_results = service.store.search_spans(
            service._retrieval_query(query, plan, "raw"),
            scope,
            limit=per_source_limit,
            speaker=speaker,
            include_session=include_session,
        )
        candidate_lists.append(
            [
                Candidate(
                    id=span.span_id,
                    type="span",
                    text=span.content,
                    source="l0_raw_hybrid",
                    scores=scores,
                    source_span_ids=[span.span_id],
                    metadata={"speaker": span.speaker, "span_type": span.span_type, "timestamp": span.timestamp.isoformat()},
                )
                for span, scores in raw_span_results
            ]
        )
        topic_scoped = service._topic_scoped_raw_candidates(
            query,
            scope,
            plan,
            limit=max(per_source_limit * 2, per_source_limit + 12),
            include_session=include_session,
        )
        if topic_scoped:
            candidate_lists.append(topic_scoped)
        broad_raw = service._broad_raw_recall_candidates(
            query,
            scope,
            plan,
            limit=max(per_source_limit * 3, per_source_limit + 24),
            include_session=include_session,
        )
        if broad_raw:
            candidate_lists.append(broad_raw)
        scent_trail = service._raw_scent_trail_candidates(
            query,
            scope,
            plan,
            [candidate for items in candidate_lists for candidate in items],
            limit=max(per_source_limit, 12),
            include_session=include_session,
        )
        if scent_trail:
            candidate_lists.append(scent_trail)
        if plan.query_type == "contradiction_resolution":
            contradiction_claims = service._contradiction_claim_candidates(
                query,
                scope,
                plan,
                limit=max(per_source_limit, 12),
                include_session=include_session,
            )
            if contradiction_claims:
                candidate_lists.append(contradiction_claims)
        if plan.query_type == "temporal_lookup":
            temporal_candidates = service._temporal_coverage_candidates(
                query,
                scope,
                plan,
                limit=max(per_source_limit * 2, per_source_limit + 12),
                include_session=include_session,
            )
            if temporal_candidates:
                candidate_lists.append(temporal_candidates)
        if plan.query_type == "multi_session_reasoning":
            aggregation_candidates = service._aggregation_coverage_candidates(
                query,
                scope,
                plan,
                limit=max(per_source_limit * 2, per_source_limit + 12),
                include_session=include_session,
            )
            if aggregation_candidates:
                candidate_lists.append(aggregation_candidates)
        if plan.query_type == "event_ordering":
            coverage_candidates = service._event_ordering_coverage_candidates(
                query,
                scope,
                limit=max(per_source_limit * 3, per_source_limit + 12),
                include_session=include_session,
            )
            coverage_candidates = [
                candidate
                for candidate in coverage_candidates
                if candidate.source != "event_ordering_persisted_graph"
                and not str(candidate.source).startswith("event_ordering_graph")
            ]
            if coverage_candidates:
                candidate_lists.append(coverage_candidates)
            episode_recall = service._event_ordering_episode_recall_candidates(
                query,
                scope,
                plan,
                limit=max(per_source_limit * 4, per_source_limit + 24),
                include_session=include_session,
            )
            if episode_recall:
                candidate_lists.append(episode_recall)
            candidate_lists.append(
                service._event_ordering_timeline_candidates(
                    query,
                    plan,
                    scope,
                    limit=max(per_source_limit * 3, per_source_limit + 12),
                    include_session=include_session,
                )
            )
    if service._source_enabled("facts", enabled) and service._plan_uses_source(plan, "facts"):
        fact_results = service.store.search_facts(
            service._retrieval_query(query, plan, "facts"),
            scope,
            limit=per_source_limit,
            include_session=include_session,
        )
        candidate_lists.append(
            [
                Candidate(
                    id=fact.fact_id,
                    type="fact",
                    text=fact.text,
                    source="l1_fact_hybrid",
                    scores={**scores, "view_or_profile_prior": 0.0},
                    source_span_ids=fact.source_span_ids,
                    metadata={"category": fact.category, "confidence": fact.confidence},
                )
                for fact, scores in fact_results
            ]
        )
    if service._source_enabled("events", enabled) and service._plan_uses_source(plan, "events"):
        if plan.query_type == "event_ordering":
            event_results = service._event_ordering_event_candidates(
                query,
                scope,
                limit=max(per_source_limit * 2, 12),
                include_session=include_session,
            )
        else:
            event_results = service.store.search_events(
                service._retrieval_query(query, plan, "events"),
                scope,
                limit=per_source_limit,
                include_session=include_session,
            )
        candidate_lists.append(
            [
                Candidate(
                    id=event.event_id,
                    type="event",
                    text=event.description,
                    source="event_timeline_graph" if plan.query_type == "event_ordering" else "l2_event_graph",
                    scores={**scores, "graph_proximity": 0.80 if plan.query_type == "event_ordering" else 0.55},
                    source_span_ids=event.source_span_ids,
                    metadata={
                        "event_type": event.event_type,
                        "time_start": (
                            service._event_ordering_observed_at(event).isoformat()
                            if plan.query_type == "event_ordering" and service._event_ordering_observed_at(event)
                            else event.time_start.isoformat()
                            if event.time_start
                            else None
                        ),
                        "milestone_group": event_milestone_group(event),
                    },
                )
                for event, scores in event_results
            ]
        )
    if service._source_enabled("views", enabled) and service._plan_uses_source(plan, "views"):
        views = service.store.list_current_views(scope, include_session=include_session)
        candidate_lists.append(
            [
                Candidate(
                    id=view.view_id,
                    type="view",
                    text=view.text,
                    source="l3_current_view",
                    scores={
                        "bm25_score": keyword_score(query, view.text),
                        "view_or_profile_prior": 0.85,
                        "score": keyword_score(query, view.text) + 0.85,
                    },
                    source_span_ids=view.source_span_ids,
                    metadata={"view_type": view.view_type, "confidence": view.confidence},
                )
                for view in views
            ]
        )
    if service._source_enabled("profiles", enabled) and service._plan_uses_source(plan, "profiles"):
        profile_results = service.store.search_entity_profiles(
            service._retrieval_query(query, plan, "profiles"),
            scope,
            limit=per_source_limit,
            include_session=include_session,
        )
        candidate_lists.append(
            [
                Candidate(
                    id=profile.profile_id,
                    type="profile",
                    text=profile.text,
                    source="l3_entity_profile",
                    scores={
                        **scores,
                        "view_or_profile_prior": 0.55,
                        "score": scores.get("score", 0.0) + 0.55,
                    },
                    source_span_ids=profile.source_span_ids,
                    metadata={"profile_type": profile.profile_type, "support_count": profile.support_count},
                )
                for profile, scores in profile_results
            ]
        )
    if service._source_enabled("exact", enabled):
        candidate_lists.append(
            service._exact_candidates(
                service._retrieval_query(query, plan, "exact"),
                scope,
                per_source_limit,
                plan=plan,
                include_session=include_session,
            )
        )
    if service._source_enabled("entities", enabled):
        candidate_lists.append(
            service._entity_candidates(
                service._retrieval_query(query, plan, "entities"),
                scope,
                per_source_limit,
                include_session=include_session,
            )
        )
    return candidate_lists


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


@dataclass(frozen=True)
class StaticProvider:
    provider_id: str
    source_family: str
    output_source: str
    candidate_ids: tuple[str, ...]
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
                id=candidate_id,
                type="span",
                text=f"candidate text for {candidate_id}",
                source=self.output_source,
                scores={"score": 1.0},
                source_span_ids=[candidate_id],
                metadata={},
            )
            for candidate_id in self.candidate_ids
        ]


class RecallProviderRegistryTests(unittest.TestCase):
    def _context(
        self,
        *,
        query_type: str = "fact_lookup",
        enabled_sources: set[str] | None = None,
    ) -> RecallContext:
        return RecallContext(
            service=SimpleNamespace(_plan_uses_source=lambda plan, source: True),
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

    def test_registry_recall_keeps_empty_groups_for_enabled_applicable_providers(self) -> None:
        first = StaticProvider("raw_span", "raw", "l0_raw_hybrid", ("first",))
        second = StaticProvider("facts", "facts", "l1_fact_hybrid", ())
        third = StaticProvider("views", "views", "l3_current_view", ("third",))
        registry = ProviderRegistry([first, second, third])
        context = self._context()

        lists = registry.recall(context)

        self.assertEqual([[candidate.id for candidate in items] for items in lists], [["first"], [], ["third"]])

    def test_registry_recall_excludes_empty_groups_from_prior_candidates(self) -> None:
        first = StaticProvider("raw_span", "raw", "l0_raw_hybrid", ("first",))
        second = StaticProvider("facts", "facts", "l1_fact_hybrid", ())
        third = StaticProvider("views", "views", "l3_current_view", ("third",))
        registry = ProviderRegistry([first, second, third])
        context = self._context()

        registry.recall(context)

        self.assertEqual([candidate.id for candidate in context.prior_candidates], ["first", "third"])

    def test_registry_summary_is_structural_without_query_text(self) -> None:
        registry = ProviderRegistry([DummyProvider("raw_span", "raw", "l0_raw_hybrid")])
        context = self._context()
        registry.recall(context)

        summary = registry.summary(context)

        self.assertEqual(summary[0]["provider_id"], "raw_span")
        self.assertEqual(summary[0]["source_family"], "raw")
        self.assertEqual(summary[0]["output_count"], 1)
        self.assertNotIn("raw private query", repr(summary))


class CandidateProviderFacadeParityTests(unittest.TestCase):
    def test_facade_returns_expected_source_order_for_mixed_memory(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-provider-parity", user_id="u", agent_id="a")
        try:
            memory.add("I prefer PostgreSQL for Atlas retrieval. I also mentioned Qdrant as a backup.", scope)
            plan = memory.planner.plan("What retrieval database do I currently prefer?")
            lists = build_candidate_lists(
                memory,
                "What retrieval database do I currently prefer?",
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
            memory.add("I prefer PostgreSQL for Atlas retrieval.", scope)
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

    def test_facade_candidate_signatures_match_legacy_order_for_populated_query(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-provider-signature-parity", user_id="u", agent_id="a")
        query = "What retrieval database do I currently prefer?"
        try:
            memory.add("I prefer PostgreSQL for Atlas retrieval. I also mentioned Qdrant as a backup.", scope)
            plan = memory.planner.plan(query)
            lists = build_candidate_lists(
                memory,
                query,
                scope,
                plan,
                per_source_limit=6,
                event_milestone_group=_event_milestone_group,
            )
            legacy_lists = _legacy_build_candidate_lists(
                memory,
                query,
                scope,
                plan,
                per_source_limit=6,
                event_milestone_group=_event_milestone_group,
            )
        finally:
            memory.close()

        self.assertEqual(
            [[_candidate_signature(candidate) for candidate in items] for items in lists],
            [[_candidate_signature(candidate) for candidate in items] for items in legacy_lists],
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

    def test_structured_providers_respect_plan_source_gates(self) -> None:
        context = self._context()
        context.service._plan_uses_source = lambda plan, source: False
        providers = [FactProvider(), EventProvider(), CurrentViewProvider(), EntityProfileProvider()]

        for provider in providers:
            with self.subTest(provider=provider.provider_id):
                self.assertEqual(provider.recall(context), [])

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
            Candidate("topic-1", "span", "topic text", "topic_scope_raw", {"score": 0.8}, ["topic-1"], {})
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

        self.assertEqual(output, [["l0_raw_hybrid"], ["topic_scope_raw"], ["broad_raw_recall"]])

    def test_raw_provider_metadata_declares_legacy_output_sources(self) -> None:
        expected_sources = {
            "topic_scoped_raw": frozenset({"topic_scope_raw"}),
            "contradiction_claim": frozenset(
                {
                    "contradiction_claim_positive",
                    "contradiction_claim_negative",
                    "contradiction_claim_uncertain",
                }
            ),
            "temporal_coverage": frozenset({"temporal_coverage_raw"}),
            "aggregation_coverage": frozenset({"aggregation_coverage_raw", "aggregation_context_support"}),
            "event_ordering_coverage": frozenset({"event_ordering_coverage", "event_ordering_coverage_support"}),
        }
        providers = [
            TopicScopedRawProvider(),
            ContradictionClaimProvider(),
            TemporalCoverageProvider(),
            AggregationCoverageProvider(),
            EventOrderingCoverageProvider(),
        ]

        for provider in providers:
            with self.subTest(provider=provider.provider_id):
                self.assertEqual(provider.output_sources, expected_sources[provider.provider_id])

    def test_scent_trail_uses_prior_candidates(self) -> None:
        context = self._context()
        context.prior_candidates.append(Candidate("prior", "span", "prior", "l0_raw_hybrid", {}, ["prior"], {}))

        candidates = ScentTrailProvider().recall(context)

        self.assertEqual(candidates[0].id, "scent-1")

    def test_event_ordering_coverage_filters_graph_candidates_in_production(self) -> None:
        context = self._context(query_type="event_ordering")

        candidates = EventOrderingCoverageProvider().recall(context)

        self.assertEqual([candidate.id for candidate in candidates], ["legacy-1"])


if __name__ == "__main__":
    unittest.main()
