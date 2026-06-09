from __future__ import annotations

from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import EncodingDecision, ExtractedCandidate, MemoryFact, new_id
from fusion_memory.core.text import jaccard, tokenize


LOW_VALUE_UTTERANCES = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "got it",
    "hello",
    "hi",
    "好的",
    "谢谢",
}


class EncodingGate:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def decide(self, candidates: list[ExtractedCandidate], existing_facts: list[MemoryFact]) -> list[EncodingDecision]:
        return [self._decide_one(candidate, existing_facts) for candidate in candidates]

    def _decide_one(self, candidate: ExtractedCandidate, existing_facts: list[MemoryFact]) -> EncodingDecision:
        reasons: list[str] = []
        scores = {
            "confidence": candidate.confidence,
            "salience": float(candidate.structured.get("salience", 0.5)),
            "novelty": 1.0,
            "duplicate_score": 0.0,
            "source_quality": 1.0 if candidate.source_span_ids else 0.0,
        }
        text_lower = candidate.text.strip().lower()
        if not candidate.source_span_ids:
            reasons.append("missing_source")
            return self._decision(candidate, "reject", reasons, scores)
        if text_lower in LOW_VALUE_UTTERANCES:
            reasons.append("low_value_speech_act")
            return self._decision(candidate, "reject", reasons, scores)
        if candidate.candidate_type == "fact":
            category = candidate.structured.get("category")
            if candidate.structured.get("predicate") == "rejects_memory":
                reasons.append("explicit_negative_memory_instruction")
                return self._decision(candidate, "reject", reasons, scores)
            if category == "preference" and str(candidate.structured.get("subject")) != "user":
                reasons.append("speaker_attribution")
                return self._decision(candidate, "reject", reasons, scores)
            if "don't remember that as my preference" in text_lower:
                reasons.append("explicit_negative_memory_instruction")
                return self._decision(candidate, "reject", reasons, scores)
            duplicate_score, duplicate_id = self._duplicate_score(candidate, existing_facts)
            scores["duplicate_score"] = duplicate_score
            scores["novelty"] = 1.0 - duplicate_score
            if duplicate_score >= self.config.duplicate_similarity_threshold:
                reasons.append("duplicate")
                return self._decision(candidate, "merge", reasons, scores, [duplicate_id] if duplicate_id else [])
            if (
                candidate.confidence >= self.config.fact_accept_confidence
                and scores["salience"] >= self.config.salience_threshold
                and scores["novelty"] >= self.config.novelty_threshold
            ):
                reasons.append("accepted_fact")
                return self._decision(candidate, "accept", reasons, scores)
        if candidate.candidate_type == "event":
            if candidate.confidence >= self.config.event_accept_confidence:
                reasons.append("accepted_event")
                return self._decision(candidate, "accept", reasons, scores)
        if candidate.candidate_type == "relation":
            if candidate.confidence >= self.config.relation_accept_confidence:
                reasons.append("accepted_relation")
                return self._decision(candidate, "update_relation", reasons, scores)
            reasons.append("low_confidence_relation")
            return self._decision(candidate, "quarantine", reasons, scores)
        if candidate.confidence >= self.config.fact_accept_confidence:
            reasons.append("accepted_generic")
            return self._decision(candidate, "accept", reasons, scores)
        if candidate.confidence >= self.config.fact_quarantine_confidence:
            reasons.append("low_confidence_quarantine")
            return self._decision(candidate, "quarantine", reasons, scores)
        reasons.append("low_confidence")
        return self._decision(candidate, "reject", reasons, scores)

    def _duplicate_score(self, candidate: ExtractedCandidate, existing_facts: list[MemoryFact]) -> tuple[float, str | None]:
        cand_tokens = set(tokenize(candidate.text))
        best = 0.0
        best_id: str | None = None
        for fact in existing_facts:
            score = jaccard(cand_tokens, set(tokenize(fact.text)))
            if score > best:
                best = score
                best_id = fact.fact_id
        return best, best_id

    def _decision(
        self,
        candidate: ExtractedCandidate,
        decision: str,
        reason_codes: list[str],
        scores: dict[str, float],
        matched_existing_ids: list[str] | None = None,
    ) -> EncodingDecision:
        return EncodingDecision(
            decision_id=new_id("decision"),
            candidate_type=candidate.candidate_type,
            candidate=candidate,
            decision=decision,
            reason_codes=reason_codes,
            scores=scores,
            matched_existing_ids=matched_existing_ids or [],
        )
