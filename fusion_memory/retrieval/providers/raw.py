from __future__ import annotations

from dataclasses import dataclass

from fusion_memory.core.models import Candidate
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


class _RawProvider:
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


def _event_ordering_production_candidate(candidate: Candidate) -> bool:
    source = str(candidate.source or "")
    if source == "event_ordering_persisted_graph":
        return False
    if source.startswith("event_ordering_graph"):
        return False
    return True


class RawSpanProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="raw_span",
        source_family="raw",
        output_sources=frozenset({"l0_raw_hybrid"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        service = context.service
        speaker = context.plan.speaker_focus if context.plan.speaker_focus != "any" else None
        raw_span_results = service.store.search_spans(
            service._retrieval_query(context.query, context.plan, "raw"),
            context.scope,
            limit=context.per_source_limit,
            speaker=speaker,
            include_session=context.include_session,
        )
        return [
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


class TopicScopedRawProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="topic_scoped_raw",
        source_family="raw",
        output_sources=frozenset({"topic_scope_raw"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._topic_scoped_raw_candidates(
            context.query,
            context.scope,
            context.plan,
            limit=max(context.per_source_limit * 2, context.per_source_limit + 12),
            include_session=context.include_session,
        )


class BroadRawProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="broad_raw",
        source_family="raw",
        output_sources=frozenset({"broad_raw_recall"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._broad_raw_recall_candidates(
            context.query,
            context.scope,
            context.plan,
            limit=max(context.per_source_limit * 3, context.per_source_limit + 24),
            include_session=context.include_session,
        )


class ScentTrailProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="scent_trail",
        source_family="raw",
        output_sources=frozenset({"raw_scent_trail"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._raw_scent_trail_candidates(
            context.query,
            context.scope,
            context.plan,
            list(context.prior_candidates),
            limit=max(context.per_source_limit, 12),
            include_session=context.include_session,
        )


class ContradictionClaimProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="contradiction_claim",
        source_family="raw",
        output_sources=frozenset(
            {
                "contradiction_claim_positive",
                "contradiction_claim_negative",
                "contradiction_claim_uncertain",
            }
        ),
        supported_query_types=frozenset({"contradiction_resolution"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._contradiction_claim_candidates(
            context.query,
            context.scope,
            context.plan,
            limit=max(context.per_source_limit, 12),
            include_session=context.include_session,
        )


class TemporalCoverageProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="temporal_coverage",
        source_family="raw",
        output_sources=frozenset({"temporal_coverage_raw"}),
        supported_query_types=frozenset({"temporal_lookup"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._temporal_coverage_candidates(
            context.query,
            context.scope,
            context.plan,
            limit=max(context.per_source_limit * 2, context.per_source_limit + 12),
            include_session=context.include_session,
        )


class AggregationCoverageProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="aggregation_coverage",
        source_family="raw",
        output_sources=frozenset({"aggregation_coverage_raw", "aggregation_context_support"}),
        supported_query_types=frozenset({"multi_session_reasoning"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._aggregation_coverage_candidates(
            context.query,
            context.scope,
            context.plan,
            limit=max(context.per_source_limit * 2, context.per_source_limit + 12),
            include_session=context.include_session,
        )


class EventOrderingCoverageProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="event_ordering_coverage",
        source_family="raw",
        output_sources=frozenset({"event_ordering_coverage", "event_ordering_coverage_support"}),
        supported_query_types=frozenset({"event_ordering"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        coverage_candidates = context.service._event_ordering_coverage_candidates(
            context.query,
            context.scope,
            limit=max(context.per_source_limit * 3, context.per_source_limit + 12),
            include_session=context.include_session,
        )
        return [
            candidate
            for candidate in coverage_candidates
            if _event_ordering_production_candidate(candidate)
        ]


class EventOrderingEpisodeProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="event_ordering_episode",
        source_family="raw",
        output_sources=frozenset({"event_ordering_episode_recall"}),
        supported_query_types=frozenset({"event_ordering"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._event_ordering_episode_recall_candidates(
            context.query,
            context.scope,
            context.plan,
            limit=max(context.per_source_limit * 4, context.per_source_limit + 24),
            include_session=context.include_session,
        )


class EventOrderingTimelineProvider(_RawProvider):
    meta = _ProviderMeta(
        provider_id="event_ordering_timeline",
        source_family="raw",
        output_sources=frozenset({"event_ordering_timeline"}),
        supported_query_types=frozenset({"event_ordering"}),
    )

    def recall(self, context: RecallContext) -> list[Candidate]:
        return context.service._event_ordering_timeline_candidates(
            context.query,
            context.plan,
            context.scope,
            limit=max(context.per_source_limit * 3, context.per_source_limit + 12),
            include_session=context.include_session,
        )
