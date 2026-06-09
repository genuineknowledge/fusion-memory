from __future__ import annotations

from datetime import datetime
from typing import Any

from fusion_memory.core.llm import LLMClient
from fusion_memory.core.models import EvidenceSpan, ExtractedCandidate, MemoryFact, new_id


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {"type": "array"},
        "events": {"type": "array"},
        "relations": {"type": "array"},
    },
}

EXTRACTION_PROMPT = """Extract only durable memory candidates from the input spans.

Return a JSON object with optional arrays: facts, events, relations.
Each fact must be an object with text, subject, predicate, object, category,
confidence, salience, and source_span_ids. source_span_ids must only contain
span_id values present in the input. Do not return plain strings.
Use facts for stable preferences, instructions, profile details, project state,
or other information that should be remembered later. Leave arrays empty when
there is nothing durable to remember.
"""


class StructuredLLMExtractor:
    """Adapter for production structured extraction.

    The LLM is expected to return a dict with optional `facts`, `events`, and
    `relations` arrays. This class validates source attribution and converts
    the payload into Fusion `ExtractedCandidate`s; EncodingGate still decides
    whether candidates may be promoted.
    """

    def __init__(self, client: LLMClient, prompt_version: str = "llm-extractor-v0") -> None:
        self.client = client
        self.prompt_version = prompt_version

    def extract(self, spans: list[EvidenceSpan], existing_facts: list[MemoryFact], session_time: datetime) -> list[ExtractedCandidate]:
        if not spans:
            return []
        response = self.client.structured(
            prompt=f"{self.prompt_version}\n\n{EXTRACTION_PROMPT}",
            schema=EXTRACTION_SCHEMA,
            input={
                "session_time": session_time.isoformat(),
                "spans": [
                    {
                        "span_id": span.span_id,
                        "speaker": span.speaker,
                        "span_type": span.span_type,
                        "timestamp": span.timestamp.isoformat(),
                        "content": span.content,
                    }
                    for span in spans
                ],
                "existing_facts": [
                    {
                        "fact_id": fact.fact_id,
                        "text": fact.text,
                        "category": fact.category,
                        "source_span_ids": fact.source_span_ids,
                    }
                    for fact in existing_facts[-50:]
                ],
            },
        )
        valid_span_ids = {span.span_id for span in spans}
        out: list[ExtractedCandidate] = []
        for fact in response.get("facts", []) or []:
            out.append(self._fact_candidate(fact, valid_span_ids))
        for event in response.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            out.append(self._event_candidate(event, valid_span_ids))
        for relation in response.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            out.append(self._relation_candidate(relation))
        return out

    def _fact_candidate(self, fact: dict[str, Any] | str, valid_span_ids: set[str]) -> ExtractedCandidate:
        if isinstance(fact, str):
            fact = {
                "text": fact,
                "category": "general_fact",
                "confidence": 0.75,
                "salience": 0.6,
                "source_span_ids": list(valid_span_ids) if len(valid_span_ids) == 1 else [],
            }
        elif not isinstance(fact, dict):
            fact = {"text": str(fact), "source_span_ids": []}
        source_span_ids = self._source_span_ids(fact, valid_span_ids)
        structured = {
            "subject": fact.get("subject", "user"),
            "predicate": fact.get("predicate", "said"),
            "object": fact.get("object", fact.get("text", "")),
            "category": fact.get("category", "general_fact"),
            "confidence": float(fact.get("confidence", 0.5)),
            "salience": float(fact.get("salience", 0.5)),
        }
        return ExtractedCandidate(
            local_id=fact.get("local_id") or new_id("cand"),
            candidate_type="fact",
            text=fact.get("text") or f"{structured['subject']} {structured['predicate']} {structured['object']}",
            structured=structured,
            confidence=structured["confidence"],
            source_span_ids=source_span_ids,
            extractor_name="structured_llm_extractor",
            prompt_version=self.prompt_version,
        )

    def _event_candidate(self, event: dict[str, Any], valid_span_ids: set[str]) -> ExtractedCandidate:
        source_span_ids = self._source_span_ids(event, valid_span_ids)
        structured = {
            "event_type": event.get("event_type", "user_action"),
            "participants": event.get("participants", []),
            "description": event.get("description", event.get("text", "")),
            "time_start": event.get("time_start"),
            "time_end": event.get("time_end"),
            "time_granularity": event.get("time_granularity", "unknown"),
            "time_source": event.get("time_source", "unknown"),
            "confidence": float(event.get("confidence", 0.5)),
        }
        return ExtractedCandidate(
            local_id=event.get("local_id") or new_id("cand"),
            candidate_type="event",
            text=event.get("text") or structured["description"],
            structured=structured,
            confidence=structured["confidence"],
            source_span_ids=source_span_ids,
            extractor_name="structured_llm_extractor",
            prompt_version=self.prompt_version,
        )

    def _relation_candidate(self, relation: dict[str, Any]) -> ExtractedCandidate:
        confidence = float(relation.get("confidence", 0.5))
        return ExtractedCandidate(
            local_id=relation.get("local_id") or new_id("cand"),
            candidate_type="relation",
            text=relation.get("text") or f"{relation.get('from_local_id')} {relation.get('relation_type')} {relation.get('to_fact_id')}",
            structured={
                "relation_type": relation.get("relation_type", "linked_to"),
                "from_local_id": relation.get("from_local_id"),
                "to_fact_id": relation.get("to_fact_id"),
                "confidence": confidence,
            },
            confidence=confidence,
            source_span_ids=list(dict.fromkeys(relation.get("source_span_ids", []))),
            extractor_name="structured_llm_extractor",
            prompt_version=self.prompt_version,
        )

    def _source_span_ids(self, item: dict[str, Any], valid_span_ids: set[str]) -> list[str]:
        source_span_ids = [span_id for span_id in item.get("source_span_ids", []) if span_id in valid_span_ids]
        return list(dict.fromkeys(source_span_ids))
