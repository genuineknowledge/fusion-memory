from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


TemporalRelationType = Literal[
    "before",
    "after",
    "supersedes",
    "valid_from",
    "valid_to",
    "changed_from",
    "changed_to",
    "deadline",
    "decision_at",
    "observed_at",
]

_ALLOWED_REASON_CODES = {
    "current_value_marker",
    "date_observed",
    "decision_marker",
    "deadline_marker",
    "explicit_order_marker",
    "previous_value_marker",
    "range_endpoint",
    "update_marker",
}

_ORDER_BEFORE_RE = re.compile(r"\b(?:first|before|earlier|initially|previously|originally)\b", re.I)
_ORDER_AFTER_RE = re.compile(r"\b(?:after|then|later|next|finally|subsequently)\b", re.I)
_UPDATE_RE = re.compile(r"\b(?:updated|update|changed|change|revised|revise|adjusted|adjust|moved|move|rescheduled|reschedule|raised|raise|reduced|reduce|increased|increase|decreased|decrease|now|current|latest|set to)\b", re.I)
_SOURCE_VALUE_RE = re.compile(r"\b(?:from|previous|previously|before|old|original|originally|baseline)\b", re.I)
_DEADLINE_RE = re.compile(r"\b(?:deadline|due|due date|due by|by\s+(?:\w+|\d)|target date|no later than)\b", re.I)
_DECISION_RE = re.compile(r"\b(?:decided|decision|chose|picked|settled|agreed)\b", re.I)
_DATE_RE = re.compile(r"\b(?:20\d{2}-\d{1,2}-\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*20\d{2})?)\b", re.I)


@dataclass(frozen=True)
class TemporalRelation:
    relation_type: TemporalRelationType
    confidence: float
    reason_code: str
    role_labels: tuple[str, ...] = ()
    source_span_ids: tuple[str, ...] = ()
    normalized_date: str | None = None

    def to_safe_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "relation_type": self.relation_type,
            "confidence": round(float(self.confidence), 3),
            "reason_code": self.reason_code if self.reason_code in _ALLOWED_REASON_CODES else "other",
        }
        if self.role_labels:
            record["role_labels"] = list(self.role_labels)
        if self.source_span_ids:
            record["source_span_ids"] = list(self.source_span_ids)
        if self.normalized_date is not None:
            record["normalized_date"] = self.normalized_date
        return record


def temporal_relations_for_text(
    text: str,
    *,
    query: str = "",
    value_text: str = "",
    value_type: str = "",
    normalized_date: str | None = None,
    source_span_id: str | None = None,
) -> list[TemporalRelation]:
    del value_type
    lower_text = text.lower()
    del query
    has_value = bool(value_text)
    source_ids = (source_span_id,) if source_span_id else ()

    relations: list[TemporalRelation] = []

    if _ORDER_BEFORE_RE.search(lower_text):
        relations.append(
            _relation(
                "before",
                confidence=0.72,
                reason_code="explicit_order_marker",
                role_labels=("earlier_event",),
                source_span_ids=source_ids,
            )
        )
    if _ORDER_AFTER_RE.search(lower_text):
        relations.append(
            _relation(
                "after",
                confidence=0.72,
                reason_code="explicit_order_marker",
                role_labels=("later_event",),
                source_span_ids=source_ids,
            )
        )

    if has_value and _SOURCE_VALUE_RE.search(lower_text):
        relations.append(
            _relation(
                "changed_from",
                confidence=0.69,
                reason_code="previous_value_marker",
                role_labels=("previous_value",),
                source_span_ids=source_ids,
            )
        )

    if has_value and _UPDATE_RE.search(lower_text):
        relations.append(
            _relation(
                "changed_to",
                confidence=0.81,
                reason_code="update_marker",
                role_labels=("current_value",),
                source_span_ids=source_ids,
            )
        )
        relations.append(
            _relation(
                "supersedes",
                confidence=0.74,
                reason_code="current_value_marker",
                role_labels=("current_value",),
                source_span_ids=source_ids,
            )
        )

    if _DEADLINE_RE.search(lower_text):
        relations.append(
            _relation(
                "deadline",
                confidence=0.84,
                reason_code="deadline_marker",
                role_labels=("deadline",),
                source_span_ids=source_ids,
            )
        )

    if _DECISION_RE.search(lower_text):
        relations.append(
            _relation(
                "decision_at",
                confidence=0.83,
                reason_code="decision_marker",
                role_labels=("decision_point",),
                source_span_ids=source_ids,
            )
        )

    if normalized_date is not None:
        relations.append(
            _relation(
                "observed_at",
                confidence=0.63,
                reason_code="date_observed",
                role_labels=("normalized_date",),
                source_span_ids=source_ids,
                normalized_date=normalized_date,
            )
        )

    return _dedupe_relations(relations)


def temporal_relation_summary(relations: list[TemporalRelation]) -> dict[str, object]:
    relation_types = sorted({relation.relation_type for relation in relations})
    source_span_ids = sorted({source_id for relation in relations for source_id in relation.source_span_ids})
    role_labels = sorted({role_label for relation in relations for role_label in relation.role_labels})
    reason_codes = sorted({relation.reason_code for relation in relations if relation.reason_code in _ALLOWED_REASON_CODES})
    return {
        "relation_count": len(relations),
        "relation_types": relation_types,
        "role_labels": role_labels,
        "reason_codes": reason_codes,
        "source_span_count": len(source_span_ids),
        "source_span_ids": source_span_ids,
    }


def safe_temporal_relation_records(relations: list[TemporalRelation], *, limit: int = 12) -> list[dict[str, object]]:
    return [relation.to_safe_dict() for relation in relations[: max(0, limit)]]


def _relation(
    relation_type: TemporalRelationType,
    *,
    confidence: float,
    reason_code: str,
    role_labels: tuple[str, ...],
    source_span_ids: tuple[str, ...],
    normalized_date: str | None = None,
) -> TemporalRelation:
    return TemporalRelation(
        relation_type=relation_type,
        confidence=confidence,
        reason_code=reason_code,
        role_labels=role_labels,
        source_span_ids=source_span_ids,
        normalized_date=normalized_date,
    )


def _dedupe_relations(relations: list[TemporalRelation]) -> list[TemporalRelation]:
    deduped: list[TemporalRelation] = []
    seen: set[tuple[object, ...]] = set()
    for relation in relations:
        key = (
            relation.relation_type,
            relation.reason_code,
            relation.role_labels,
            relation.source_span_ids,
            relation.normalized_date,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(relation)
    return deduped
