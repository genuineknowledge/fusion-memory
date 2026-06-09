from __future__ import annotations

from fusion_memory.core.models import Candidate, QueryPlan


def score_candidate(candidate: Candidate, plan: QueryPlan) -> Candidate:
    scores = dict(candidate.scores)
    semantic = scores.get("semantic_score", 0.0)
    bm25 = scores.get("bm25_score", 0.0)
    entity = scores.get("entity_overlap", 0.0)
    temporal = scores.get("temporal_fit", 0.0)
    graph = scores.get("graph_proximity", 0.0)
    view = scores.get("view_or_profile_prior", 0.0)
    rrf = scores.get("rrf_score", 0.0)
    if candidate.type == "view":
        view = max(view, 0.8)
    if candidate.type == "profile":
        view = max(view, 0.45)
    if candidate.type == "event":
        graph = max(graph, 0.35)
    if candidate.type == "span":
        scores["source_quality"] = 0.8
    weights = {
        "semantic": 0.25,
        "bm25": 0.20,
        "entity": 0.12,
        "temporal": 0.14,
        "graph": 0.10,
        "view": 0.07,
        "rrf": 0.12,
    }
    if plan.query_type in {"temporal_lookup", "event_ordering"}:
        weights.update({"temporal": 0.25, "graph": 0.18, "view": 0.02})
    elif plan.query_type in {"preference", "instruction"}:
        weights.update({"view": 0.20, "temporal": 0.05})
    elif plan.query_type in {"contradiction_resolution", "knowledge_update"}:
        weights.update({"temporal": 0.18, "graph": 0.15})
    elif plan.query_type == "abstention":
        weights.update({"bm25": 0.28, "semantic": 0.18})
    utility = (
        weights["semantic"] * semantic
        + weights["bm25"] * bm25
        + weights["entity"] * entity
        + weights["temporal"] * temporal
        + weights["graph"] * graph
        + weights["view"] * view
        + weights["rrf"] * rrf
    )
    scores["utility_score"] = utility
    return Candidate(
        id=candidate.id,
        type=candidate.type,
        text=candidate.text,
        source=candidate.source,
        scores=scores,
        source_span_ids=candidate.source_span_ids,
        metadata=candidate.metadata,
    )

