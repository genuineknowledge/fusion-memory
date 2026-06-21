from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fusion_memory.core.models import Candidate
from fusion_memory.retrieval.providers.base import RecallContext, RecallProvider, provider_applies_to_query_type
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
