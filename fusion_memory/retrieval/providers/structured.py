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


class _StructuredProvider:
    meta: _ProviderMeta

    @property
    def provider_id(self) -> str:
        return self.meta.provider_id

    @property
    def source_family(self) -> str:
        return self.meta.source_family

    @property
    def output_sources(self) -> frozenset[str]:
        return self.meta.output_sources

    @property
    def supported_query_types(self) -> frozenset[str] | None:
        return self.meta.supported_query_types

    @property
    def production_default(self) -> bool:
        return self.meta.production_default

    @property
    def shadow_only(self) -> bool:
        return self.meta.shadow_only

    @property
    def graph_related(self) -> bool:
        return self.meta.graph_related

    @property
    def replay_categories(self) -> frozenset[str]:
        return self.meta.replay_categories

    def _plan_uses_source(self, context: RecallContext) -> bool:
        return context.service._plan_uses_source(context.plan, self.source_family)


class FactProvider(_StructuredProvider):
    meta = _ProviderMeta(
        provider_id="facts",
        source_family="facts",
        output_sources=frozenset({"l1_fact_hybrid"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        if not self._plan_uses_source(context):
            return []
        service = context.service
        fact_results = service.store.search_facts(
            service._retrieval_query(context.query, context.plan, "facts"),
            context.scope,
            limit=context.per_source_limit,
            include_session=context.include_session,
        )
        return [
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


class EventProvider(_StructuredProvider):
    meta = _ProviderMeta(
        provider_id="events",
        source_family="events",
        output_sources=frozenset({"l2_event_graph", "event_timeline_graph"}),
        graph_related=True,
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        if not self._plan_uses_source(context):
            return []
        service = context.service
        if context.plan.query_type == "event_ordering":
            event_results = service._event_ordering_event_candidates(
                context.query,
                context.scope,
                limit=max(context.per_source_limit * 2, 12),
                include_session=context.include_session,
            )
        else:
            event_results = service.store.search_events(
                service._retrieval_query(context.query, context.plan, "events"),
                context.scope,
                limit=context.per_source_limit,
                include_session=context.include_session,
            )
        return [
            Candidate(
                id=event.event_id,
                type="event",
                text=event.description,
                source="event_timeline_graph" if context.plan.query_type == "event_ordering" else "l2_event_graph",
                scores={**scores, "graph_proximity": 0.80 if context.plan.query_type == "event_ordering" else 0.55},
                source_span_ids=event.source_span_ids,
                metadata={
                    "event_type": event.event_type,
                    "time_start": (
                        service._event_ordering_observed_at(event).isoformat()
                        if context.plan.query_type == "event_ordering" and service._event_ordering_observed_at(event)
                        else event.time_start.isoformat()
                        if event.time_start
                        else None
                    ),
                    "milestone_group": context.event_milestone_group(event),
                },
            )
            for event, scores in event_results
        ]


class CurrentViewProvider(_StructuredProvider):
    meta = _ProviderMeta(
        provider_id="views",
        source_family="views",
        output_sources=frozenset({"l3_current_view"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        if not self._plan_uses_source(context):
            return []
        views = context.service.store.list_current_views(context.scope, include_session=context.include_session)
        return [
            Candidate(
                id=view.view_id,
                type="view",
                text=view.text,
                source="l3_current_view",
                scores={
                    "bm25_score": keyword_score(context.query, view.text),
                    "view_or_profile_prior": 0.85,
                    "score": keyword_score(context.query, view.text) + 0.85,
                },
                source_span_ids=view.source_span_ids,
                metadata={"view_type": view.view_type, "confidence": view.confidence},
            )
            for view in views
        ]


class EntityProfileProvider(_StructuredProvider):
    meta = _ProviderMeta(
        provider_id="profiles",
        source_family="profiles",
        output_sources=frozenset({"l3_entity_profile"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        if not self._plan_uses_source(context):
            return []
        service = context.service
        profile_results = service.store.search_entity_profiles(
            service._retrieval_query(context.query, context.plan, "profiles"),
            context.scope,
            limit=context.per_source_limit,
            include_session=context.include_session,
        )
        return [
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


class ExactProvider(_StructuredProvider):
    meta = _ProviderMeta(
        provider_id="exact",
        source_family="exact",
        output_sources=frozenset({"exact_answer"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        service = context.service
        return service._exact_candidates(
            service._retrieval_query(context.query, context.plan, "exact"),
            context.scope,
            context.per_source_limit,
            plan=context.plan,
            include_session=context.include_session,
        )


class EntityProvider(_StructuredProvider):
    meta = _ProviderMeta(
        provider_id="entities",
        source_family="entities",
        output_sources=frozenset({"entity_graph"}),
        graph_related=True,
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        service = context.service
        return service._entity_candidates(
            service._retrieval_query(context.query, context.plan, "entities"),
            context.scope,
            context.per_source_limit,
            include_session=context.include_session,
        )
