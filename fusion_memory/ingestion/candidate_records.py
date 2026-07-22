from __future__ import annotations

from datetime import datetime
from typing import Any

from fusion_memory.core.models import (
    FactRelation,
    MemoryEvent,
    MemoryFact,
    Scope,
    new_id,
)
from fusion_memory.core.text import stable_hash
from fusion_memory.storage.sqlite_store import dt_from_str


def candidate_to_fact(
    scope: Scope,
    candidate: Any,
    session_time: datetime,
) -> MemoryFact:
    structured = candidate.structured
    return MemoryFact(
        fact_id=new_id("fact"),
        scope=scope,
        subject=str(structured.get("subject", "user")),
        predicate=str(structured.get("predicate", "said")),
        object=str(structured.get("object", candidate.text)),
        text=candidate.text,
        category=str(structured.get("category", "general_fact")),
        confidence=float(structured.get("confidence", candidate.confidence)),
        salience=float(structured.get("salience", 0.5)),
        observed_at=session_time,
        valid_from=session_time,
        valid_to=None,
        polarity=str(structured.get("polarity", "unknown")),
        source_span_ids=list(dict.fromkeys(candidate.source_span_ids)),
        metadata={
            "hash": stable_hash(candidate.text),
            "candidate_local_id": candidate.local_id,
            **(
                {"value_mentions": structured["value_mentions"]}
                if structured.get("value_mentions")
                else {}
            ),
            **(
                {"topic_terms": structured["topic_terms"]}
                if structured.get("topic_terms")
                else {}
            ),
        },
    )


def candidate_to_event(scope: Scope, candidate: Any) -> MemoryEvent:
    structured = candidate.structured
    return MemoryEvent(
        event_id=new_id("event"),
        scope=scope,
        event_type=str(structured.get("event_type", "user_action")),
        participants=list(structured.get("participants", [])),
        description=str(structured.get("description", candidate.text)),
        time_start=dt_from_str(structured.get("time_start")),
        time_end=dt_from_str(structured.get("time_end")),
        time_granularity=str(structured.get("time_granularity", "unknown")),
        time_source=str(structured.get("time_source", "unknown")),
        source_span_ids=list(dict.fromkeys(candidate.source_span_ids)),
        confidence=float(structured.get("confidence", candidate.confidence)),
    )


def candidate_to_relation(
    candidate: Any,
    local_to_fact: dict[str, str],
) -> FactRelation | None:
    structured = candidate.structured
    from_id = local_to_fact.get(str(structured.get("from_local_id")))
    to_id = structured.get("to_fact_id")
    if not from_id or not to_id:
        return None
    return FactRelation(
        relation_id=new_id("rel"),
        from_fact_id=from_id,
        to_fact_id=str(to_id),
        relation_type=str(structured.get("relation_type", "linked_to")),
        source_span_ids=list(dict.fromkeys(candidate.source_span_ids)),
        confidence=float(structured.get("confidence", candidate.confidence)),
    )
