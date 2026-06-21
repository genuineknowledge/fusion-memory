from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from fusion_memory.core.models import Candidate
from fusion_memory.core.models import Scope
from fusion_memory.retrieval.providers import RecallContext
from fusion_memory.retrieval.providers import default_provider_registry
from fusion_memory.retrieval.rrf import reciprocal_rank_fusion
from fusion_memory.retrieval.scoring import score_candidate
from fusion_memory.retrieval.temporal_relations import temporal_relation_summary_from_safe_records


@dataclass(frozen=True)
class QueryUnderstandingRecord:
    language: str
    intent: str
    features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "intent": self.intent,
            "features": list(self.features),
        }


@dataclass(frozen=True)
class QueryUnderstandingResult:
    plan: Any
    language: str
    intent: str
    features: tuple[str, ...]
    intent_telemetry: dict[str, Any] | None
    precomputed: bool

    def safe_record(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "intent": self.intent,
            "features": list(self.features),
        }


class QueryUnderstandingEngine:
    def run(
        self,
        query: str,
        scope: Scope,
        options: dict[str, Any],
        planner: Any,
    ) -> QueryUnderstandingResult:
        del scope
        options = options or {}
        precomputed_plan = options.get("_plan")
        precomputed = precomputed_plan is not None
        plan = precomputed_plan if precomputed else planner.plan(query, query_type_hint=options.get("query_type_hint"))
        intent_telemetry = options.get("_intent_telemetry") if precomputed else getattr(planner, "last_intent_telemetry", None)
        return query_understanding_result_from_plan(
            plan=plan,
            query=query,
            intent_telemetry=intent_telemetry,
            precomputed=precomputed,
        )


def query_understanding_result_from_plan(
    *,
    plan: Any,
    query: str,
    intent_telemetry: dict[str, Any] | None = None,
    precomputed: bool = True,
) -> QueryUnderstandingResult:
    language = "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"
    features = tuple(
        feature
        for feature, enabled in {
            "current_value": bool(getattr(plan, "current_value", False)),
            "multi_condition": bool(getattr(plan, "constraints", None)),
            "temporal": getattr(plan, "query_type", None) in {"temporal_lookup", "event_ordering"},
        }.items()
        if enabled
    )
    return QueryUnderstandingResult(
        plan=plan,
        language=language,
        intent=str(getattr(plan, "query_type", "")),
        features=features,
        intent_telemetry=intent_telemetry,
        precomputed=precomputed,
    )


@dataclass(frozen=True)
class RetrievalExecutionContext:
    service: Any
    query: str
    scope: Scope
    options: dict[str, Any]
    query_understanding: QueryUnderstandingResult
    include_session: bool
    per_source_limit: int
    enabled_sources: list[str] | set[str] | None
    mode: str
    limit: int
    rerank_top_n: int
    event_milestone_group: Callable[[Any], str | None]


@dataclass(frozen=True)
class RecallResult:
    candidate_lists: list[list[Candidate]]
    recalled_candidates: list[Candidate]
    provider_summary: list[dict[str, Any]] = field(default_factory=list)

    def safe_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"source_counts": _candidate_source_counts(self.recalled_candidates)}
        if self.provider_summary:
            record["provider_summary"] = [_safe_provider_summary(item) for item in self.provider_summary]
        return record


class RecallOrchestrator:
    def __init__(self, registry: Any | None = None) -> None:
        self._registry = registry

    def run(self, context: RetrievalExecutionContext) -> RecallResult:
        enabled = set(context.enabled_sources) if context.enabled_sources is not None else None
        recall_context = RecallContext(
            service=context.service,
            query=context.query,
            scope=context.scope,
            plan=context.query_understanding.plan,
            per_source_limit=context.per_source_limit,
            enabled_sources=enabled,
            include_session=context.include_session,
            event_milestone_group=context.event_milestone_group,
            prior_candidates=[],
        )
        registry = self._registry or default_provider_registry()
        candidate_lists = registry.recall(recall_context)
        recalled_candidates = [candidate for candidates in candidate_lists for candidate in candidates]
        provider_summary = registry.summary(recall_context) if hasattr(registry, "summary") else []
        return RecallResult(
            candidate_lists=candidate_lists,
            recalled_candidates=recalled_candidates,
            provider_summary=provider_summary,
        )


@dataclass(frozen=True)
class FusionResult:
    fused: list[Candidate]
    scored: list[Candidate]
    quota_result: Any
    marked: list[Candidate]
    scored_again: list[Candidate]
    rerank_top_n: int
    mode: str
    limit: int

    def safe_record(self) -> dict[str, Any]:
        return {
            "fused_source_counts": _candidate_source_counts(self.fused),
            "scored_count": len(self.scored),
            "marked_count": len(self.marked),
            "scored_again_count": len(self.scored_again),
            "quota_required": int(getattr(self.quota_result, "required", 0)),
            "quota_selected": len(getattr(self.quota_result, "selected_span_ids", []) or []),
            "coverage_insufficient": bool(getattr(self.quota_result, "coverage_insufficient", False)),
            "backfilled": int(getattr(self.quota_result, "backfilled", 0)),
            "mode": self.mode,
            "limit": int(self.limit),
            "rerank_top_n": int(self.rerank_top_n),
        }


class CandidateFusionEngine:
    def run(self, context: RetrievalExecutionContext, recall_result: RecallResult) -> FusionResult:
        service = context.service
        plan = context.query_understanding.plan
        fused = reciprocal_rank_fusion(recall_result.candidate_lists, k=service.config.rrf_k)
        scored = [score_candidate(candidate, plan) for candidate in fused]
        quota_result = service.quota.enforce(plan, context.scope, scored, include_session=context.include_session)
        marked = self._mark_quota_selected(quota_result.candidates, quota_result.selected_span_ids)
        scored_again = [score_candidate(candidate, plan) for candidate in marked]
        scored_again.sort(key=lambda candidate: candidate.scores.get("utility_score", 0.0), reverse=True)
        mode = context.options.get("mode", context.mode)
        limit = context.options.get("limit", context.limit)
        rerank_top_n = context.options.get("rerank_top_n") or (
            service.config.balanced_mode_rerank_top_n
            if mode == "balanced"
            else service.config.benchmark_mode_rerank_top_n
            if mode == "benchmark"
            else limit
        )
        return FusionResult(
            fused=fused,
            scored=scored,
            quota_result=quota_result,
            marked=marked,
            scored_again=scored_again,
            rerank_top_n=rerank_top_n,
            mode=mode,
            limit=limit,
        )

    def _mark_quota_selected(self, candidates: list[Candidate], span_ids: list[str]) -> list[Candidate]:
        selected = set(span_ids)
        out: list[Candidate] = []
        for candidate in candidates:
            metadata = dict(candidate.metadata)
            if candidate.type == "span" and candidate.id in selected:
                metadata["quota_selected"] = True
            if candidate.type == "view" and candidate.source == "l3_current_view":
                reasons = list(metadata.get("must_preserve_reason") or [])
                if "current_value" not in reasons:
                    reasons.append("current_value")
                metadata["must_preserve_reason"] = reasons
                metadata["evidence_role"] = "answer"
                metadata["current_value"] = True
            out.append(
                Candidate(
                    id=candidate.id,
                    type=candidate.type,
                    text=candidate.text,
                    source=candidate.source,
                    scores=candidate.scores,
                    source_span_ids=candidate.source_span_ids,
                    metadata=metadata,
                )
            )
        return out


def _candidate_source_counts(candidates: list[Candidate]) -> dict[str, int]:
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
    return source_counts


def _safe_provider_summary(summary: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in (
        "provider_id",
        "source_family",
        "output_count",
        "output_source_counts",
        "production_default",
        "shadow_only",
        "graph_related",
    ):
        if key not in summary:
            continue
        value = summary[key]
        if key == "output_source_counts" and isinstance(value, dict):
            safe[key] = {str(source): int(count) for source, count in value.items()}
        elif key == "output_count":
            safe[key] = int(value)
        elif key in {"production_default", "shadow_only", "graph_related"}:
            safe[key] = bool(value)
        else:
            safe[key] = str(value)
    return safe


@dataclass(frozen=True)
class CandidateRecallRecord:
    source_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"source_counts": dict(self.source_counts)}


@dataclass(frozen=True)
class CandidateFusionRecord:
    selected_sources: tuple[str, ...]
    dropped_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_sources": list(self.selected_sources),
            "dropped_count": int(self.dropped_count),
        }


@dataclass(frozen=True)
class EvidencePackBuilderRecord:
    source_span_count: int
    coverage_insufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_span_count": int(self.source_span_count),
            "coverage_insufficient": bool(self.coverage_insufficient),
        }


@dataclass(frozen=True)
class TemporalRelationsRecord:
    relation_count: int
    relation_types: tuple[str, ...]
    role_labels: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    source_span_count: int = 0
    source_span_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TemporalRelationsRecord:
        return cls(
            relation_count=int(value.get("relation_count") or 0),
            relation_types=tuple(str(item) for item in (value.get("relation_types") or []) if item),
            role_labels=tuple(str(item) for item in (value.get("role_labels") or []) if item),
            reason_codes=tuple(str(item) for item in (value.get("reason_codes") or []) if item),
            source_span_count=int(value.get("source_span_count") or 0),
            source_span_ids=tuple(str(item) for item in (value.get("source_span_ids") or []) if item),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_count": int(self.relation_count),
            "relation_types": list(self.relation_types),
            "role_labels": list(self.role_labels),
            "reason_codes": list(self.reason_codes),
            "source_span_count": int(self.source_span_count),
            "source_span_ids": list(self.source_span_ids),
        }


@dataclass(frozen=True)
class RetrievalPipelineRecord:
    query_type: str
    mode: str
    query_understanding: QueryUnderstandingRecord
    candidate_recall: CandidateRecallRecord
    candidate_fusion: CandidateFusionRecord
    evidence_output: EvidencePackBuilderRecord
    temporal_relations: TemporalRelationsRecord | None = None

    def pipeline_layers(self) -> dict[str, Any]:
        layers = {
            "QueryUnderstanding": self.query_understanding.to_dict(),
            "CandidateRecall": self.candidate_recall.to_dict(),
            "CandidateFusion": self.candidate_fusion.to_dict(),
            "EvidencePackBuilder": self.evidence_output.to_dict(),
        }
        if self.temporal_relations is not None:
            layers["TemporalRelations"] = self.temporal_relations.to_dict()
        return layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "mode": self.mode,
            "pipeline_layers": self.pipeline_layers(),
            "query_understanding": self.query_understanding.to_dict(),
            "candidate_recall": self.candidate_recall.to_dict(),
            "candidate_fusion": self.candidate_fusion.to_dict(),
            "evidence_output": self.evidence_output.to_dict(),
        }


class RetrievalTraceRecorder:
    def __init__(self, record: RetrievalPipelineRecord) -> None:
        self._record = record

    def flush(self) -> dict[str, Any]:
        return self._record.to_dict()


def build_pipeline_record(
    query_type: str,
    mode: str,
    *,
    language: str,
    intent: str,
    features: list[str],
    recalled: list[Candidate],
    selected: list[Candidate],
    dropped_count: int,
    source_span_count: int,
    coverage_insufficient: bool,
    temporal_relation_summary: dict[str, object] | None = None,
) -> RetrievalPipelineRecord:
    source_counts: dict[str, int] = {}
    for candidate in recalled:
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
    selected_sources = tuple(dict.fromkeys(candidate.source for candidate in selected))
    return RetrievalPipelineRecord(
        query_type=query_type,
        mode=mode,
        query_understanding=QueryUnderstandingRecord(
            language=language,
            intent=intent,
            features=tuple(features),
        ),
        candidate_recall=CandidateRecallRecord(source_counts=source_counts),
        candidate_fusion=CandidateFusionRecord(
            selected_sources=selected_sources,
            dropped_count=dropped_count,
        ),
        evidence_output=EvidencePackBuilderRecord(
            source_span_count=source_span_count,
            coverage_insufficient=coverage_insufficient,
        ),
        temporal_relations=(
            TemporalRelationsRecord.from_dict(temporal_relation_summary)
            if temporal_relation_summary is not None
            else None
        ),
    )


class EvidencePackAssembler:
    def update_pipeline_output(
        self,
        value: Any,
        *,
        source_span_count: int,
        coverage_insufficient: bool,
    ) -> dict[str, Any]:
        return update_pipeline_evidence_output(
            value,
            source_span_count=source_span_count,
            coverage_insufficient=coverage_insufficient,
        )


def update_pipeline_evidence_output(
    value: Any,
    *,
    source_span_count: int,
    coverage_insufficient: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    updated = dict(value)
    evidence = {
        "source_span_count": int(source_span_count),
        "coverage_insufficient": bool(coverage_insufficient),
    }
    layers = dict(updated.get("pipeline_layers") or {})
    layers["EvidencePackBuilder"] = evidence
    updated["pipeline_layers"] = layers
    if "evidence_output" in updated:
        updated["evidence_output"] = evidence
    return updated


def selected_temporal_relation_summary(candidates: list[Candidate]) -> dict[str, object] | None:
    safe_records: list[dict[str, object]] = []
    summary_relation_count = 0
    summary_relation_types: set[str] = set()
    summary_role_labels: set[str] = set()
    summary_reason_codes: set[str] = set()
    summary_source_span_ids: set[str] = set()

    for candidate in candidates:
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        relations = metadata.get("temporal_relations")
        candidate_safe_records = [item for item in relations if isinstance(item, dict)] if isinstance(relations, list) else []
        if candidate_safe_records:
            safe_records.extend(candidate_safe_records)
            continue

        summary = metadata.get("temporal_relation_summary")
        if isinstance(summary, dict):
            summary_relation_count += int(summary.get("relation_count") or 0)
            summary_relation_types.update(str(item) for item in (summary.get("relation_types") or []) if item)
            summary_role_labels.update(str(item) for item in (summary.get("role_labels") or []) if item)
            summary_reason_codes.update(str(item) for item in (summary.get("reason_codes") or []) if item)
            summary_source_span_ids.update(str(item) for item in (summary.get("source_span_ids") or []) if item)

    if summary_relation_count == 0 and not safe_records:
        return None

    safe_summary = temporal_relation_summary_from_safe_records(safe_records) if safe_records else {
        "relation_count": 0,
        "relation_types": [],
        "role_labels": [],
        "reason_codes": [],
        "source_span_count": 0,
        "source_span_ids": [],
    }

    relation_types = sorted(summary_relation_types | set(str(item) for item in safe_summary["relation_types"]))
    role_labels = sorted(summary_role_labels | set(str(item) for item in safe_summary["role_labels"]))
    reason_codes = sorted(summary_reason_codes | set(str(item) for item in safe_summary["reason_codes"]))
    source_span_ids = sorted(summary_source_span_ids | set(str(item) for item in safe_summary["source_span_ids"]))
    return {
        "relation_count": summary_relation_count + int(safe_summary["relation_count"]),
        "relation_types": relation_types,
        "role_labels": role_labels,
        "reason_codes": reason_codes,
        "source_span_count": len(source_span_ids),
        "source_span_ids": source_span_ids,
    }
