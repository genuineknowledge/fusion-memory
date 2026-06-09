from __future__ import annotations

from fusion_memory.core.models import Candidate
from fusion_memory.core.text import jaccard, tokenize


def mmr(candidates: list[Candidate], limit: int, lambda_: float = 0.72) -> list[Candidate]:
    remaining = list(candidates)
    selected: list[Candidate] = []
    while remaining and len(selected) < limit:
        best_index = 0
        best_value = float("-inf")
        for index, candidate in enumerate(remaining):
            relevance = candidate.scores.get("utility_score", candidate.scores.get("score", 0.0))
            diversity_penalty = max((_similarity(candidate, chosen) for chosen in selected), default=0.0)
            value = lambda_ * relevance - (1 - lambda_) * diversity_penalty
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _similarity(a: Candidate, b: Candidate) -> float:
    if set(a.source_span_ids) & set(b.source_span_ids):
        return 1.0
    return jaccard(set(tokenize(a.text)), set(tokenize(b.text)))

