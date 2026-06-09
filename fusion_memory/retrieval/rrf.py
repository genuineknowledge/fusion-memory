from __future__ import annotations

from collections import defaultdict

from fusion_memory.core.models import Candidate


def reciprocal_rank_fusion(candidate_lists: list[list[Candidate]], k: int = 60) -> list[Candidate]:
    by_id: dict[tuple[str, str], Candidate] = {}
    scores: dict[tuple[str, str], float] = defaultdict(float)
    sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    for candidates in candidate_lists:
        for rank, candidate in enumerate(candidates, start=1):
            key = (candidate.type, candidate.id)
            by_id[key] = candidate
            scores[key] += 1.0 / (k + rank)
            sources[key].append(candidate.source)
    fused: list[Candidate] = []
    for key, candidate in by_id.items():
        merged = Candidate(
            id=candidate.id,
            type=candidate.type,
            text=candidate.text,
            source="+".join(sorted(set(sources[key]))),
            scores={**candidate.scores, "rrf_score": scores[key]},
            source_span_ids=candidate.source_span_ids,
            metadata=candidate.metadata,
        )
        fused.append(merged)
    fused.sort(key=lambda c: c.scores.get("rrf_score", 0.0), reverse=True)
    return fused

