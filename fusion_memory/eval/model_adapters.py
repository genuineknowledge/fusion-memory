from __future__ import annotations

from typing import Any

from fusion_memory.core.llm import LLMClient
from fusion_memory.core.models import EvidencePack
from fusion_memory.core.text import compact_summary


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "matched": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["matched"],
}


class OpenAICompatibleAnswerModel:
    """Benchmark answer model backed by any structured OpenAI-compatible client."""

    def __init__(self, client: LLMClient, prompt_version: str = "eval-answer-v0") -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.version = f"llm_answer:{_client_version(client)}:{prompt_version}"

    def answer(self, query: str, pack: EvidencePack) -> str:
        response = self.client.structured(
            prompt=self.prompt_version,
            schema=ANSWER_SCHEMA,
            input={
                "instruction": (
                    "Answer the query using only the provided Fusion Memory evidence pack. "
                    "If the evidence pack does not support an answer, return a concise abstention."
                ),
                "query": query,
                "answer_policy": pack.answer_policy,
                "coverage": pack.coverage,
                "evidence_pack": _pack_for_model(pack),
            },
        )
        answer = response.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return "Not enough supported memory to answer."


class OpenAICompatibleJudgeModel:
    """Semantic answer judge backed by any structured OpenAI-compatible client."""

    def __init__(self, client: LLMClient, prompt_version: str = "eval-judge-v0") -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.version = f"llm_judge:{_client_version(client)}:{prompt_version}"

    def score(self, answer: str, gold_answers: list[str]) -> bool:
        if not gold_answers:
            return False
        response = self.client.structured(
            prompt=self.prompt_version,
            schema=JUDGE_SCHEMA,
            input={
                "instruction": (
                    "Return matched=true when the candidate answer is semantically equivalent "
                    "to at least one gold answer. Be strict about unsupported extra claims."
                ),
                "candidate_answer": answer,
                "gold_answers": gold_answers,
            },
        )
        return bool(response.get("matched", False))


def _pack_for_model(pack: EvidencePack) -> dict[str, Any]:
    return {
        "current_views": _compact_records(pack.current_views, preferred_text_key="text"),
        "entity_profiles": _compact_records(pack.entity_profiles, preferred_text_key="text"),
        "facts": _compact_records(pack.facts, preferred_text_key="text"),
        "events": _compact_records(pack.events, preferred_text_key="description"),
        "source_spans": _compact_records(pack.source_spans, preferred_text_key="content"),
        "conflicts": pack.conflicts[:10],
    }


def _compact_records(records: list[dict[str, Any]], *, preferred_text_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records[:20]:
        compacted: dict[str, Any] = {}
        for key in [
            "id",
            "fact_id",
            "event_id",
            "view_id",
            "profile_id",
            "type",
            "category",
            "subject",
            "predicate",
            "object",
            "entity_id",
            "profile_type",
            "timestamp",
            "time_start",
            "time_end",
            "source_span_ids",
        ]:
            if key in record:
                compacted[key] = record[key]
        text = str(record.get(preferred_text_key) or record.get("text") or record.get("content") or "")
        if text:
            compacted[preferred_text_key] = compact_summary(text, 1200)
        out.append(compacted)
    return out


def _client_version(client: LLMClient) -> str:
    return str(getattr(client, "version", client.__class__.__name__))
