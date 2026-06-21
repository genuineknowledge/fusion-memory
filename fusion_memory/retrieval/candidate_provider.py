from __future__ import annotations

from typing import Any, Callable

from fusion_memory.core.models import Candidate, Scope
from fusion_memory.retrieval.providers import RecallContext, default_provider_registry


def build_candidate_lists(
    service: Any,
    query: str,
    scope: Scope,
    plan: Any,
    per_source_limit: int,
    enabled_sources: list[str] | set[str] | None = None,
    include_session: bool = False,
    *,
    event_milestone_group: Callable[[Any], str | None],
) -> list[list[Candidate]]:
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
