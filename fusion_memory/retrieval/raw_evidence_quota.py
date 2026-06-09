from __future__ import annotations

from dataclasses import dataclass

from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import Candidate, QueryPlan, Scope
from fusion_memory.storage.sqlite_store import SQLiteMemoryStore


@dataclass
class QuotaResult:
    candidates: list[Candidate]
    selected_span_ids: list[str]
    required: int
    coverage_insufficient: bool
    backfilled: int


class RawEvidenceQuota:
    def __init__(self, store: SQLiteMemoryStore, config: MemoryConfig | None = None) -> None:
        self.store = store
        self.config = config or DEFAULT_CONFIG

    def enforce(self, plan: QueryPlan, scope: Scope, candidates: list[Candidate], *, include_session: bool = False) -> QuotaResult:
        required = self.config.raw_evidence_quotas.get(plan.query_type, 2)
        span_candidates = [candidate for candidate in candidates if candidate.type == "span"]
        selected_ids = [candidate.id for candidate in span_candidates[:required]]
        backfilled = 0
        if len(selected_ids) < required:
            speaker = plan.speaker_focus if plan.speaker_focus != "any" else None
            for span, scores in self.store.search_spans(plan.query, scope, limit=required * 2, speaker=speaker, include_session=include_session):
                if span.span_id in selected_ids:
                    continue
                candidates.append(
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source="raw_quota_backfill",
                        scores=scores,
                        source_span_ids=[span.span_id],
                        metadata={"speaker": span.speaker, "timestamp": span.timestamp.isoformat()},
                    )
                )
                selected_ids.append(span.span_id)
                backfilled += 1
                if len(selected_ids) >= required:
                    break
        return QuotaResult(
            candidates=candidates,
            selected_span_ids=selected_ids[:required],
            required=required,
            coverage_insufficient=len(selected_ids) < required,
            backfilled=backfilled,
        )
