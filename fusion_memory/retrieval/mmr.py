from __future__ import annotations

from fusion_memory.core.models import Candidate
from fusion_memory.core.text import tokenize


def mmr(candidates: list[Candidate], limit: int, lambda_: float = 0.72) -> list[Candidate]:
    if limit <= 0 or not candidates:
        return []
    ranked = [
        {
            "candidate": candidate,
            "relevance": candidate.scores.get("utility_score", candidate.scores.get("score", 0.0)),
            "tokens": frozenset(tokenize(candidate.text)),
            "source_span_ids": frozenset(candidate.source_span_ids),
        }
        for candidate in candidates
    ]
    selected: list[dict[str, object]] = []
    while ranked and len(selected) < limit:
        best_index = 0
        best_value = float("-inf")
        for index, candidate in enumerate(ranked):
            relevance = float(candidate["relevance"])
            diversity_penalty = max((_similarity(candidate, chosen) for chosen in selected), default=0.0)
            value = lambda_ * relevance - (1 - lambda_) * diversity_penalty
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(ranked.pop(best_index))
    return [item["candidate"] for item in selected]


def _similarity(a: dict[str, object], b: dict[str, object]) -> float:
    if a["source_span_ids"] & b["source_span_ids"]:
        return 1.0
    left = a["tokens"]
    right = b["tokens"]
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / max(1, len(left) + len(right) - intersection)
