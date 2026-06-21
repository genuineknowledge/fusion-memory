from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fusion_memory.core.models import Candidate
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
