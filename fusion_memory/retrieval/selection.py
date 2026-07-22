from __future__ import annotations

from time import perf_counter
from typing import Any

from fusion_memory.core.models import Candidate
from fusion_memory.model_pool import EndpointUnavailable
from fusion_memory.retrieval.mmr import mmr
from fusion_memory.retrieval.reranker import Reranker, rerank_candidates
from fusion_memory.retrieval.rrf import reciprocal_rank_fusion


def generic_utility(candidate: Candidate, rank: int, total: int) -> float:
    rank_score = 1.0 - ((rank - 1) / max(1, total))
    signal = max(
        float(candidate.scores.get("semantic_score", 0.0)),
        float(candidate.scores.get("bm25_score", 0.0)),
        float(candidate.scores.get("exact_signal", 0.0)),
        float(candidate.scores.get("temporal_score", 0.0)),
        float(candidate.scores.get("graph_proximity", 0.0)),
    )
    return 0.60 * rank_score + 0.40 * signal


def select_candidates(
    query: str,
    candidate_lists: list[list[Candidate]],
    *,
    limit: int,
    use_reranker: bool,
    reranker: Reranker | None,
    mmr_lambda: float,
    status: dict[str, Any] | None = None,
) -> list[Candidate]:
    fusion_started = perf_counter()
    fused = reciprocal_rank_fusion(candidate_lists)
    if status is not None:
        status["fusion_elapsed_ms"] = (perf_counter() - fusion_started) * 1000
        status["fused_count"] = len(fused)

    selection_started = perf_counter()
    total = len(fused)
    scored = [
        _with_utility(candidate, generic_utility(candidate, rank, total))
        for rank, candidate in enumerate(fused, 1)
    ]
    scored.sort(
        key=lambda candidate: (
            -float(candidate.scores.get("utility_score", 0.0)),
            -float(candidate.scores.get("rrf_score", 0.0)),
            candidate.id,
        )
    )

    ranked = scored
    if use_reranker and reranker is not None:
        try:
            ranked = rerank_candidates(query, scored, reranker)
        except EndpointUnavailable:
            if status is not None:
                status["reranker_failure"] = "reranker_unavailable"
            ranked = scored

    selected = mmr(ranked, limit=limit, lambda_=mmr_lambda)
    stable_selected = sorted(enumerate(selected), key=lambda item: (item[0], item[1].id))
    if status is not None:
        status["selection_elapsed_ms"] = (perf_counter() - selection_started) * 1000
    return [candidate for _position, candidate in stable_selected]


def _with_utility(candidate: Candidate, utility_score: float) -> Candidate:
    return Candidate(
        id=candidate.id,
        type=candidate.type,
        text=candidate.text,
        source=candidate.source,
        scores={**candidate.scores, "utility_score": utility_score},
        source_span_ids=list(candidate.source_span_ids),
        metadata=dict(candidate.metadata),
    )
