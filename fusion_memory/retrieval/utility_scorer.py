from __future__ import annotations

from fusion_memory.core.models import Candidate, new_id
from fusion_memory.retrieval.context import ProductQueryPlan


def feature_vector(
    candidate: Candidate,
    plan: ProductQueryPlan,
) -> dict[str, float | str]:
    scores = candidate.scores
    return {
        "rrf_score": scores.get("rrf_score", 0.0),
        "semantic_score": scores.get("semantic_score", 0.0),
        "bm25_score": scores.get("bm25_score", 0.0),
        "entity_overlap": scores.get("entity_overlap", 0.0),
        "temporal_fit": scores.get("temporal_fit", 0.0),
        "graph_proximity": scores.get("graph_proximity", 0.0),
        "view_or_profile_prior": scores.get("view_or_profile_prior", 0.0),
        "source_quality": scores.get("source_quality", 0.0),
        "utility_score": scores.get("utility_score", 0.0),
        "candidate_type": candidate.type,
        "query_type": plan.intent,
        "quota_selected": 1.0 if candidate.metadata.get("quota_selected") else 0.0,
    }


def weak_label(candidate: Candidate, plan: ProductQueryPlan) -> str:
    if candidate.type == "span" and candidate.scores.get("bm25_score", 0.0) > 0.15:
        return "useful"
    if candidate.type in {"view", "profile"} and plan.intent in {"current_state", "instruction"}:
        return "useful"
    if candidate.scores.get("utility_score", 0.0) <= 0:
        return "not_useful"
    return "unknown"


def utility_example(
    query_id: str,
    query: str,
    plan: ProductQueryPlan,
    candidate: Candidate,
) -> dict:
    return {
        "example_id": new_id("utility"),
        "query_id": query_id,
        "query_text": query,
        "query_type": plan.intent,
        "candidate_id": candidate.id,
        "candidate_type": candidate.type,
        "features": feature_vector(candidate, plan),
        "label": weak_label(candidate, plan),
        "label_source": "weak_rule",
        "answer_correct": None,
    }
