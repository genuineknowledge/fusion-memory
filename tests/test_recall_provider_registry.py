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


if __name__ == "__main__":
    unittest.main()
