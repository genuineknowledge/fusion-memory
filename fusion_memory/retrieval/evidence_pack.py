from __future__ import annotations

from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import Candidate, EvidencePack, QueryPlan
from fusion_memory.core.text import compact_summary, tokenize
from fusion_memory.storage.sqlite_store import SQLiteMemoryStore


class EvidencePackBuilder:
    def __init__(self, store: SQLiteMemoryStore, config: MemoryConfig | None = None) -> None:
        self.store = store
        self.config = config or DEFAULT_CONFIG

    def build(
        self,
        query: str,
        plan: QueryPlan,
        candidates: list[Candidate],
        coverage: dict,
        trace: list[dict],
        token_budget: int | None = None,
    ) -> EvidencePack:
        token_budget = token_budget or self.config.answer_context_budget_tokens
        current_views: list[dict] = []
        profiles: list[dict] = []
        facts: list[dict] = []
        events: list[dict] = []
        spans: list[dict] = []
        conflicts: list[dict] = []
        seen_spans: set[str] = set()
        estimated_tokens = 0
        for candidate in candidates:
            if candidate.type == "view":
                current_views.append({"id": candidate.id, "text": candidate.text, "source_span_ids": candidate.source_span_ids})
            elif candidate.type == "profile":
                profiles.append({"id": candidate.id, "text": candidate.text, "source_span_ids": candidate.source_span_ids})
            elif candidate.type == "fact":
                facts.append({"id": candidate.id, "text": candidate.text, "source_span_ids": candidate.source_span_ids})
            elif candidate.type == "event":
                events.append({"id": candidate.id, "text": candidate.text, "source_span_ids": candidate.source_span_ids})
            for span_id in candidate.source_span_ids:
                if span_id in seen_spans:
                    continue
                span = self.store.get_span(span_id)
                if not span:
                    continue
                seen_spans.add(span_id)
                content = compact_summary(span.content, self.config.evidence_span_summary_chars)
                content_tokens = len(tokenize(content))
                if estimated_tokens + content_tokens > token_budget:
                    remaining = max(0, token_budget - estimated_tokens)
                    if remaining <= 0:
                        continue
                    words = content.split()
                    content = " ".join(words[:remaining])
                    content_tokens = len(tokenize(content))
                estimated_tokens += content_tokens
                spans.append(
                    {
                        "id": span.span_id,
                        "session_id": span.scope.session_id,
                        "turn_id": span.turn_id,
                        "source_uri": span.source_uri,
                        "speaker": span.speaker,
                        "timestamp": span.timestamp.isoformat(),
                        "content": content,
                    }
                )
        if plan.query_type in {"contradiction_resolution", "knowledge_update"}:
            conflicts = [
                {"fact_id": fact["id"], "source_span_ids": fact["source_span_ids"]}
                for fact in facts[:4]
            ]
        answer_policy = "answer_with_evidence_or_abstain"
        if plan.query_type == "abstention" or coverage.get("coverage_insufficient"):
            answer_policy = "abstain_if_not_supported"
        coverage = {**coverage, "token_budget": token_budget, "estimated_source_tokens": estimated_tokens}
        return EvidencePack(
            query=query,
            answer_policy=answer_policy,
            current_views=current_views,
            entity_profiles=profiles,
            facts=facts,
            events=events,
            source_spans=spans,
            conflicts=conflicts,
            coverage=coverage,
            debug_trace=trace,
        )
