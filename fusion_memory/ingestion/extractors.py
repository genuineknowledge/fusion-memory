from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from fusion_memory.core.models import EvidenceSpan, ExtractedCandidate, MemoryFact, new_id
from fusion_memory.core.text import compact_summary, extract_entities
from fusion_memory.ingestion.temporal_normalizer import TemporalNormalizer


PREFERENCE_RE = re.compile(r"\b(?:i|we)\s+(?:now\s+)?(?:prefer|like|use|want)\s+(.+?)(?:[.!?]|$)", re.I)
NOW_PREFERRED_RE = re.compile(r"\b([A-Z][A-Za-z0-9_\- ]{1,80}?)\s+is\s+now\s+preferred\b", re.I)
SWITCH_RE = re.compile(r"\b(?:switched|moved|changed)\s+(.+?)\s+(?:from\s+(.+?)\s+)?to\s+(.+?)(?:[.!?]|$)", re.I)
INSTRUCTION_RE = re.compile(r"\b(?:remember|always|please|以后|记住|默认|do not|don't)\b", re.I)
EVENT_RE = re.compile(r"\b(?:tested|switched|added|removed|decided|deployed|created|fixed|changed|moved|started|finished)\b", re.I)


class RuleBasedExtractor:
    def __init__(self) -> None:
        self.temporal = TemporalNormalizer()

    def extract(self, spans: list[EvidenceSpan], existing_facts: list[MemoryFact], session_time: datetime) -> list[ExtractedCandidate]:
        candidates: list[ExtractedCandidate] = []
        for span in spans:
            candidates.extend(self._extract_fact_candidates(span))
            candidates.extend(self._extract_event_candidates(span, session_time))
        candidates.extend(self._extract_relation_candidates(candidates, existing_facts))
        return candidates

    def _extract_fact_candidates(self, span: EvidenceSpan) -> list[ExtractedCandidate]:
        text = span.content.strip()
        lower = text.lower()
        out: list[ExtractedCandidate] = []
        if span.speaker == "tool":
            out.append(
                self._candidate(
                    "fact",
                    f"Tool result: {compact_summary(text, 220)}",
                    span,
                    {
                        "subject": "tool",
                        "predicate": "returned",
                        "object": compact_summary(text, 180),
                        "category": "tool_result",
                        "confidence": 0.82,
                        "salience": 0.72,
                    },
                )
            )
            return out
        if span.speaker in {"assistant", "agent"}:
            if any(word in lower for word in ["recommend", "suggest", "plan", "建议", "方案"]):
                out.append(
                    self._candidate(
                        "fact",
                        f"Assistant/agent stated: {compact_summary(text, 220)}",
                        span,
                        {
                            "subject": span.speaker,
                            "predicate": "stated",
                            "object": compact_summary(text, 180),
                            "category": "assistant_statement" if span.speaker == "assistant" else "agent_action",
                            "confidence": 0.76,
                            "salience": 0.55,
                        },
                    )
                )
            return out
        if "don't remember that as my preference" in lower or "do not remember that as my preference" in lower:
            out.append(
                self._candidate(
                    "fact",
                    "User explicitly said not to store the previous suggestion as a preference.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "rejects_memory",
                        "object": "previous suggestion as preference",
                        "category": "instruction",
                        "confidence": 0.85,
                        "salience": 0.68,
                    },
                )
            )
            return out
        switch = SWITCH_RE.search(text)
        if switch:
            target = switch.group(3).strip()
            out.append(
                self._candidate(
                    "fact",
                    f"User switched {switch.group(1).strip()} to {target}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "switched_to",
                        "object": target,
                        "category": "project_state",
                        "confidence": 0.86,
                        "salience": 0.82,
                    },
                )
            )
        pref = PREFERENCE_RE.search(text)
        if pref:
            obj = pref.group(1).strip()
            out.append(
                self._candidate(
                    "fact",
                    f"User prefers {obj}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "prefers",
                        "object": obj,
                        "category": "preference",
                        "confidence": 0.82,
                        "salience": 0.78,
                    },
                )
            )
        now_pref = NOW_PREFERRED_RE.search(text)
        if now_pref:
            obj = now_pref.group(1).strip()
            out.append(
                self._candidate(
                    "fact",
                    f"User prefers {obj}.",
                    span,
                    {
                        "subject": "user",
                        "predicate": "prefers",
                        "object": obj,
                        "category": "preference",
                        "confidence": 0.84,
                        "salience": 0.80,
                    },
                )
            )
        if INSTRUCTION_RE.search(text) and not pref:
            category = "instruction" if any(w in lower for w in ["always", "please", "do not", "don't", "以后", "默认"]) else "general_fact"
            out.append(
                self._candidate(
                    "fact",
                    f"User instruction/fact: {compact_summary(text, 220)}",
                    span,
                    {
                        "subject": "user",
                        "predicate": "said",
                        "object": compact_summary(text, 180),
                        "category": category,
                        "confidence": 0.72,
                        "salience": 0.55 if category == "general_fact" else 0.70,
                    },
                )
            )
        elif not out and len(text) > 20 and span.speaker == "user":
            if any(w in lower for w in ["database", "project", "atlas", "qdrant", "postgres", "kubernetes"]):
                out.append(
                    self._candidate(
                        "fact",
                        f"User said: {compact_summary(text, 220)}",
                        span,
                        {
                            "subject": "user",
                            "predicate": "said",
                            "object": compact_summary(text, 180),
                            "category": "general_fact",
                            "confidence": 0.64,
                            "salience": 0.45,
                        },
                    )
                )
        return out

    def _extract_event_candidates(self, span: EvidenceSpan, session_time: datetime) -> list[ExtractedCandidate]:
        if not EVENT_RE.search(span.content):
            return []
        normalized = self.temporal.normalize(span.content, session_time)
        return [
            self._candidate(
                "event",
                compact_summary(span.content, 240),
                span,
                {
                    "event_type": "state_change" if re.search(r"switched|changed|moved", span.content, re.I) else "user_action",
                    "participants": extract_entities(span.content) or ["user"],
                    "description": compact_summary(span.content, 240),
                    "time_start": normalized.time_start.isoformat() if normalized.time_start else None,
                    "time_end": normalized.time_end.isoformat() if normalized.time_end else None,
                    "time_granularity": normalized.granularity,
                    "time_source": normalized.source,
                    "confidence": 0.74,
                },
            )
        ]

    def _extract_relation_candidates(
        self, candidates: Iterable[ExtractedCandidate], existing_facts: list[MemoryFact]
    ) -> list[ExtractedCandidate]:
        out: list[ExtractedCandidate] = []
        for candidate in candidates:
            if candidate.candidate_type != "fact":
                continue
            structured = candidate.structured
            if structured.get("category") not in {"preference", "project_state", "instruction"}:
                continue
            for fact in existing_facts:
                if fact.category != structured.get("category"):
                    continue
                if fact.object.lower() == str(structured.get("object", "")).lower():
                    continue
                if fact.subject == structured.get("subject"):
                    out.append(
                        ExtractedCandidate(
                            local_id=new_id("cand"),
                            candidate_type="relation",
                            text=f"{candidate.text} supersedes {fact.text}",
                            structured={
                                "relation_type": "supersedes",
                                "from_local_id": candidate.local_id,
                                "to_fact_id": fact.fact_id,
                                "confidence": 0.78,
                            },
                            confidence=0.78,
                            source_span_ids=candidate.source_span_ids + fact.source_span_ids,
                            extractor_name="relation_detector",
                        )
                    )
        return out

    def _candidate(self, candidate_type: str, text: str, span: EvidenceSpan, structured: dict) -> ExtractedCandidate:
        return ExtractedCandidate(
            local_id=new_id("cand"),
            candidate_type=candidate_type,
            text=text,
            structured=structured,
            confidence=float(structured.get("confidence", 0.5)),
            source_span_ids=[span.span_id],
            extractor_name="rule_based_extractor",
        )
