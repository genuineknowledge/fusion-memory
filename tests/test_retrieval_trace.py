from __future__ import annotations

from fusion_memory.core.models import Candidate
from fusion_memory.core.text import stable_hash
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderKind,
    ProviderRequest,
    SearchRequest,
)
from fusion_memory.retrieval.engine import sanitize_retrieval_trace
from fusion_memory.retrieval.providers.base import ProviderOutcome
from fusion_memory.retrieval.tracing import build_retrieval_trace


def _plan() -> ProductQueryPlan:
    return ProductQueryPlan(
        intent="chronology",
        provider_requests=(ProviderRequest(ProviderKind.CHRONOLOGY, 4),),
        time_range=None,
        entities=("Atlas",),
        speaker="user",
        ordering=OrderingMode.CHRONOLOGICAL,
        use_reranker=False,
    )


def test_product_trace_has_stable_stages_without_query_or_candidate_text() -> None:
    request = SearchRequest("private Atlas deployment sequence", 4)
    candidate = Candidate(
        id="event-1",
        type="event",
        text="private deployment details",
        source="product_chronology",
        scores={"chronology_rank": 1.0},
        source_span_ids=["span-1"],
        metadata={},
    )

    trace = build_retrieval_trace(
        object(),
        request,
        _plan(),
        (ProviderOutcome(ProviderKind.CHRONOLOGY, (candidate,), 1.25),),
        [candidate],
        filtered_count=2,
    )

    assert trace["stages"] == ["plan", "recall", "fusion", "selection"]
    assert trace["providers"] == [
        {
            "kind": "chronology",
            "count": 1,
            "elapsed_ms": 1.25,
            "failure_code": None,
        }
    ]
    assert trace["filtered_count"] == 2
    assert trace["selected_ids"] == [stable_hash("event-1")]
    assert request.query not in repr(trace)
    assert candidate.text not in repr(trace)


def test_trace_sanitizer_keeps_only_allowed_query_intent_telemetry() -> None:
    trace = sanitize_retrieval_trace(
        {
            "stages": ["plan", "recall", "private_stage"],
            "mode": "fast",
            "intent": "chronology",
            "query": "private query",
            "query_intent_telemetry": {
                "source": "llm_query_intent",
                "prompt_version": "query-intent-refiner-v0",
                "fallback": True,
                "accepted": False,
                "deterministic_confidence": 0.6,
                "reason": "llm_call_failed",
                "error": "Bearer private-secret",
            },
        }
    )

    assert trace["stages"] == ["plan", "recall"]
    assert trace["query_intent_telemetry"] == {
        "source": "llm_query_intent",
        "prompt_version": "query-intent-refiner-v0",
        "fallback": True,
        "accepted": False,
        "deterministic_confidence": 0.6,
        "reason": "llm_call_failed",
    }
    assert "private" not in repr(trace)
