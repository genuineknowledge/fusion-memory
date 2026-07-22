from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import Candidate, EvidencePack, EvidenceSpan
from fusion_memory.core.text import compact_summary, tokenize
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProviderKind,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)
from fusion_memory.retrieval.ports import MemorySearchRepository
from fusion_memory.retrieval.tracing import sanitize_dimension


_PRODUCT_COVERAGE_FIELDS = (
    "degraded",
    "provider_failures",
    "provider_counts",
    "reranker_unavailable",
    "planner_fallback",
)
_PRODUCT_PROVIDER_KINDS = {provider.value for provider in ProviderKind}
_PRODUCT_STAGES = {"plan", "recall", "fusion", "selection"}
_QUERY_INTENT_STRING_FIELDS = (
    "schema_version",
    "language",
    "answer_shape",
    "evidence_scope",
    "speaker_scope",
)
_QUERY_INTENT_STRING_LIST_FIELDS = (
    "entities",
    "target_terms",
    "object_types",
    "route_reasons",
)
_TEMPORAL_STRING_LIST_FIELDS = ("endpoint_roles", "time_expressions")
_TEMPORAL_BOOLEAN_FIELDS = ("requires_time", "requires_order", "requires_duration")
_AGGREGATION_STRING_LIST_FIELDS = ("target_terms", "unit_terms")


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
        selected_span_ids = {str(record["id"]) for record in source_spans}
        current_views, facts = self._structured_records(
            context,
            result.candidates,
            selected_span_ids,
            estimated_tokens,
            token_budget,
        )
        coverage = _product_coverage(
            result,
            source_span_count=len(source_spans),
            token_budget=token_budget,
            estimated_tokens=estimated_tokens,
        )
        return EvidencePack(
            query=request.query,
            answer_policy=(
                "answer_with_evidence_or_abstain"
                if source_spans
                else "abstain_if_not_supported"
            ),
            current_views=current_views,
            entity_profiles=[],
            facts=facts,
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

    def _structured_records(
        self,
        context: RetrievalContext,
        candidates: tuple[Candidate, ...],
        selected_span_ids: set[str],
        estimated_tokens: int,
        token_budget: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        selected_views = {}
        if any(candidate.type == "view" for candidate in candidates):
            selected_views = {
                view.view_id: view
                for view in self.repository.list_current_views(
                    context.scope,
                    include_session=context.include_session,
                )
            }
        current_views: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        seen_ids: set[tuple[str, str]] = set()

        for candidate in candidates:
            record_type = str(candidate.type)
            key = (record_type, candidate.id)
            if key in seen_ids or record_type not in {"fact", "view"}:
                continue
            seen_ids.add(key)

            if record_type == "fact":
                record = self.repository.get_fact(
                    candidate.id,
                    context.scope,
                    include_session=context.include_session,
                )
                record_id = record.fact_id if record is not None else None
            else:
                record = selected_views.get(candidate.id)
                record_id = record.view_id if record is not None else None
            if record is None or record_id is None:
                continue

            source_span_ids = _supported_source_span_ids(
                candidate,
                record.source_span_ids,
                selected_span_ids,
            )
            if not source_span_ids:
                continue
            text = compact_summary(
                record.text,
                self.config.evidence_span_summary_chars,
            )
            record_tokens = len(tokenize(text))
            if estimated_tokens + record_tokens > token_budget:
                break
            estimated_tokens += record_tokens
            output = {
                "id": record_id,
                "text": text,
                "candidate_source": candidate.source,
                "source_span_ids": source_span_ids,
            }
            if record_type == "fact":
                facts.append(output)
            else:
                current_views.append(output)

        return current_views, facts


def _timeline_index(candidate: Candidate) -> int | None:
    value = candidate.metadata.get("timeline_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _supported_source_span_ids(
    candidate: Candidate,
    record_span_ids: list[str],
    selected_span_ids: set[str],
) -> list[str]:
    candidate_span_ids = set(candidate.source_span_ids)
    return list(
        dict.fromkeys(
            span_id
            for span_id in record_span_ids
            if span_id in candidate_span_ids and span_id in selected_span_ids
        )
    )


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


def _product_coverage(
    result: RetrievalResult,
    *,
    source_span_count: int,
    token_budget: int,
    estimated_tokens: int,
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for field in _PRODUCT_COVERAGE_FIELDS:
        if field not in result.coverage:
            continue
        value = result.coverage[field]
        if field == "provider_counts":
            value = _provider_counts(value)
        elif field == "provider_failures":
            value = _string_list(value)
        elif field in {"degraded", "reranker_unavailable"}:
            if not isinstance(value, bool):
                continue
        elif field == "planner_fallback" and not isinstance(value, str):
            continue
        coverage[field] = value
    coverage.update(
        intent=sanitize_dimension(result.plan.intent),
        query_intent=_query_intent(result.plan.query_intent),
        source_span_count=source_span_count,
        token_budget=token_budget,
        estimated_source_tokens=estimated_tokens,
    )
    return coverage


def _product_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    trace: dict[str, Any] = {}
    stages = value.get("stages")
    if isinstance(stages, (list, tuple)):
        trace["stages"] = [
            stage
            for stage in stages
            if isinstance(stage, str) and stage in _PRODUCT_STAGES
        ]
    if value.get("mode") in {"fast", "balanced"}:
        trace["mode"] = value["mode"]
    if isinstance(value.get("intent"), str):
        trace["intent"] = value["intent"]
    providers = _provider_trace(value.get("providers"))
    if providers:
        trace["providers"] = providers
    if isinstance(value.get("filtered_count"), int):
        trace["filtered_count"] = value["filtered_count"]
    selected_ids = _string_list(value.get("selected_ids"))
    if selected_ids:
        trace["selected_ids"] = selected_ids
    durations = _stage_durations(value.get("stage_durations_ms"))
    if durations:
        trace["stage_durations_ms"] = durations
    for field in ("reranker_failure", "planner_fallback"):
        if isinstance(value.get(field), str):
            trace[field] = value[field]
    return [trace] if trace else []


def _provider_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(kind): count
        for kind, count in value.items()
        if kind in _PRODUCT_PROVIDER_KINDS and isinstance(count, int)
    }


def _provider_trace(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    providers: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or item.get("kind") not in _PRODUCT_PROVIDER_KINDS:
            continue
        record: dict[str, Any] = {"kind": item["kind"]}
        if isinstance(item.get("count"), int):
            record["count"] = item["count"]
        if isinstance(item.get("elapsed_ms"), (int, float)):
            record["elapsed_ms"] = item["elapsed_ms"]
        failure_code = item.get("failure_code")
        if failure_code is None or isinstance(failure_code, str):
            record["failure_code"] = failure_code
        providers.append(record)
    return providers


def _stage_durations(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        stage: duration
        for stage, duration in value.items()
        if stage in _PRODUCT_STAGES and isinstance(duration, (int, float))
    }


def _query_intent(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    intent: dict[str, Any] = {}
    for field in _QUERY_INTENT_STRING_FIELDS:
        if isinstance(value.get(field), str):
            intent[field] = value[field]
    for field in _QUERY_INTENT_STRING_LIST_FIELDS:
        items = _string_list(value.get(field))
        if field in value:
            intent[field] = items
    for field in ("needs_current_state", "needs_conflict_check"):
        if isinstance(value.get(field), bool):
            intent[field] = value[field]
    if isinstance(value.get("confidence"), (int, float)):
        intent["confidence"] = value["confidence"]
    temporal = _temporal_intent(value.get("temporal"))
    if temporal:
        intent["temporal"] = temporal
    aggregation = _aggregation_intent(value.get("aggregation"))
    if aggregation:
        intent["aggregation"] = aggregation
    return intent


def _temporal_intent(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    temporal: dict[str, Any] = {}
    for field in _TEMPORAL_BOOLEAN_FIELDS:
        if isinstance(value.get(field), bool):
            temporal[field] = value[field]
    if isinstance(value.get("order_direction"), str):
        temporal["order_direction"] = value["order_direction"]
    for field in _TEMPORAL_STRING_LIST_FIELDS:
        if field in value:
            temporal[field] = _string_list(value[field])
    return temporal


def _aggregation_intent(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    aggregation: dict[str, Any] = {}
    if isinstance(value.get("operation"), str):
        aggregation["operation"] = value["operation"]
    if isinstance(value.get("distinct"), bool):
        aggregation["distinct"] = value["distinct"]
    for field in _AGGREGATION_STRING_LIST_FIELDS:
        if field in value:
            aggregation[field] = _string_list(value[field])
    return aggregation


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str)]
