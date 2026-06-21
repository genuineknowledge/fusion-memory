from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fusion_memory.core.models import Candidate, QueryPlan, Scope
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
