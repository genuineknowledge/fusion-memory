from __future__ import annotations

from datetime import datetime
from typing import Any

from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import Candidate, EvidencePack, EvidenceSpan
from fusion_memory.core.text import compact_summary, tokenize
from fusion_memory.retrieval.context import (
    OrderingMode,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.ports import MemorySearchRepository


_PRODUCT_PACK_EXCLUDED_KEYS = {"category", "query_type"}


class ProductEvidencePackBuilder:
    def __init__(
        self,
        repository: MemorySearchRepository,
        config: MemoryConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or DEFAULT_CONFIG

    def build(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack:
        source_records: list[tuple[dict[str, Any], int | None, datetime, int]] = []
        seen_span_ids: set[str] = set()
        estimated_tokens = 0
        budget_exceeded = False

        for candidate_rank, candidate in enumerate(result.candidates):
            for span_id in candidate.source_span_ids:
                if not span_id or span_id in seen_span_ids:
                    continue
                seen_span_ids.add(span_id)
                span = self.repository.get_span(
                    span_id,
                    context.scope,
                    include_session=context.include_session,
                )
                if span is None:
                    continue
                record, content_tokens = self._source_record(candidate, span)
                if estimated_tokens + content_tokens > token_budget:
                    budget_exceeded = True
                    break
                estimated_tokens += content_tokens
                source_records.append(
                    (
                        record,
                        _timeline_index(candidate),
                        span.timestamp,
                        candidate_rank,
                    )
                )
            if budget_exceeded:
                break

        source_records = _ordered_source_records(source_records, result.plan.ordering)
        source_spans = [record for record, _, _, _ in source_records]
        coverage = {
            **{
                key: value
                for key, value in result.coverage.items()
                if key not in _PRODUCT_PACK_EXCLUDED_KEYS
            },
            "intent": result.plan.intent,
            "query_intent": result.plan.query_intent,
            "source_span_count": len(source_spans),
            "token_budget": token_budget,
            "estimated_source_tokens": estimated_tokens,
        }
        return EvidencePack(
            query=request.query,
            answer_policy=(
                "answer_with_evidence_or_abstain"
                if source_spans
                else "abstain_if_not_supported"
            ),
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=source_spans,
            conflicts=[],
            coverage=coverage,
            debug_trace=_product_trace(result.trace),
        )

    def _source_record(self, candidate: Candidate, span: EvidenceSpan) -> tuple[dict[str, Any], int]:
        content = compact_summary(span.content, self.config.evidence_span_summary_chars)
        return (
            {
                "id": span.span_id,
                "session_id": span.scope.session_id,
                "turn_id": span.turn_id,
                "speaker": span.speaker,
                "timestamp": span.timestamp.isoformat(),
                "source_uri": span.source_uri,
                "content": content,
                "candidate_source": candidate.source,
                "source_span_ids": list(candidate.source_span_ids),
            },
            len(tokenize(content)),
        )


def _timeline_index(candidate: Candidate) -> int | None:
    value = candidate.metadata.get("timeline_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ordered_source_records(
    records: list[tuple[dict[str, Any], int | None, datetime, int]],
    ordering: OrderingMode,
) -> list[tuple[dict[str, Any], int | None, datetime, int]]:
    if ordering == OrderingMode.CHRONOLOGICAL:
        return sorted(
            records,
            key=lambda item: (
                item[1] if item[1] is not None else 10**9,
                item[2],
                str(item[0]["id"]),
            ),
        )
    if ordering == OrderingMode.RECENCY:
        return sorted(records, key=lambda item: (item[2], -item[3]), reverse=True)
    return records


def _product_trace(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _product_trace(item)
            for key, item in value.items()
            if key not in _PRODUCT_PACK_EXCLUDED_KEYS
        }
    if isinstance(value, list):
        return [_product_trace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_product_trace(item) for item in value)
    return value
