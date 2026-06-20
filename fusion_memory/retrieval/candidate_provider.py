from __future__ import annotations

from typing import Any, Callable

from fusion_memory.core.models import Candidate, Scope
from fusion_memory.core.text import keyword_score


def _event_ordering_production_candidate(candidate: Candidate) -> bool:
    source = str(candidate.source or "")
    if source == "event_ordering_persisted_graph":
        return False
    if source.startswith("event_ordering_graph"):
        return False
    return True


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
                if _event_ordering_production_candidate(candidate)
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
        exact = service._exact_candidates(
            service._retrieval_query(query, plan, "exact"),
            scope,
            per_source_limit,
            plan=plan,
            include_session=include_session,
        )
        candidate_lists.append(exact)
    if service._source_enabled("entities", enabled):
        entity = service._entity_candidates(
            service._retrieval_query(query, plan, "entities"),
            scope,
            per_source_limit,
            include_session=include_session,
        )
        candidate_lists.append(entity)
    return candidate_lists
