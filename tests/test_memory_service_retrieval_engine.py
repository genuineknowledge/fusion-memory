from __future__ import annotations

import pytest

from fusion_memory import MemoryService, Scope
from fusion_memory.core.models import Candidate, EvidencePack
from fusion_memory.core.text import stable_hash
from fusion_memory.retrieval.context import (
    OrderingMode,
    ProductQueryPlan,
    ProviderKind,
    ProviderRequest,
    RetrievalContext,
    RetrievalResult,
    SearchRequest,
)


class RecordingStore:
    def __init__(self) -> None:
        self.saved_traces: list[tuple[str, dict[str, object], Scope]] = []
        self.audit_events: list[tuple[Scope, str, dict[str, object]]] = []

    def save_trace(self, trace_id: str, trace: dict[str, object], scope: Scope) -> None:
        self.saved_traces.append((trace_id, trace, scope))

    def insert_audit_event(self, scope: Scope, event_type: str, **kwargs: object) -> str:
        self.audit_events.append((scope, event_type, kwargs))
        return "audit-1"

    def close(self) -> None:
        pass


class RecordingAuthorizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Scope, dict[str, object]]] = []

    def authorize(
        self,
        operation: str,
        scope: Scope,
        context: dict[str, object] | None = None,
    ) -> None:
        self.calls.append((operation, scope, dict(context or {})))


class FakeEngine:
    def __init__(self) -> None:
        self.search_calls: list[tuple[RetrievalContext, SearchRequest]] = []
        self.pack_calls: list[tuple[RetrievalContext, SearchRequest, RetrievalResult, int]] = []
        self.calls: list[dict[str, object]] = []
        self.last_result: RetrievalResult | None = None
        self.intent = "factual"
        self.extra_trace: dict[str, object] = {}
        self.record_raw_model_call = False

    def search(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan | None = None,
    ) -> RetrievalResult:
        assert plan is None
        self.search_calls.append((context, request))
        if self.record_raw_model_call:
            self.calls.append({"prompt": request.query})
        candidate = Candidate(
            id="candidate-1",
            type="span",
            text="Atlas uses a database.",
            source="product_lexical",
            scores={"bm25_score": 1.0},
            source_span_ids=["span-1"],
            metadata={},
        )
        self.last_result = RetrievalResult(
            candidates=(candidate,),
            coverage={"intent": self.intent},
            trace={
                "stages": ["plan", "recall", "fusion", "selection"],
                "intent": self.intent,
                "query": request.query,
                **self.extra_trace,
            },
            plan=ProductQueryPlan(
                intent=self.intent,
                provider_requests=(ProviderRequest(ProviderKind.LEXICAL, request.limit),),
                time_range=None,
                entities=(),
                speaker=None,
                ordering=OrderingMode.RELEVANCE,
                use_reranker=False,
            ),
        )
        return self.last_result

    def build_evidence_pack(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack:
        self.pack_calls.append((context, request, result, token_budget))
        return EvidencePack(
            query=request.query,
            answer_policy="abstain_if_not_supported",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[{"id": "span-1"}],
            conflicts=[],
            coverage=result.coverage,
            debug_trace=[],
        )


@pytest.fixture
def memory_store() -> RecordingStore:
    return RecordingStore()


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


def test_memory_service_delegates_search_to_injected_engine(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    result = service.search(
        "Atlas database",
        Scope(user_id="user-a"),
        {"limit": 5, "mode": "fast"},
    )

    assert fake_engine.search_calls[0][1].query == "Atlas database"
    assert result.candidates[0].id == "candidate-1"
    assert memory_store.saved_traces[-1][0] == result.trace_id


def test_memory_service_builds_pack_with_same_engine_result(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    pack = service.answer_context(
        "Atlas database",
        Scope(user_id="user-a"),
        {"limit": 5},
    )

    assert fake_engine.pack_calls[0][2] is fake_engine.last_result
    assert pack.source_spans[0]["id"] == "span-1"
    assert len(fake_engine.search_calls) == 1


def test_injected_engine_receives_scoped_context_and_translated_source_filter(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    authorizer = RecordingAuthorizer()
    service = MemoryService(
        store=memory_store,
        retrieval_engine=fake_engine,
        authorizer=authorizer,
    )
    scope = Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a")

    service.search(
        "Atlas database",
        scope,
        {"enabled_sources": ["raw", "profiles"], "limit": 3},
    )

    context, request = fake_engine.search_calls[0]
    assert context.scope == scope
    assert context.user_id == "user-a"
    assert context.now.tzinfo is not None
    assert context.include_session is True
    assert request.enabled_providers == frozenset(
        {
            ProviderKind.VECTOR,
            ProviderKind.LEXICAL,
            ProviderKind.TEMPORAL,
            ProviderKind.CHRONOLOGY,
            ProviderKind.ENTITY,
        }
    )
    assert [call[0] for call in authorizer.calls] == ["memory.search"]


def test_injected_engine_accepts_internal_provider_filter(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    service.search(
        "Atlas database",
        Scope(user_id="user-a"),
        {"enabled_providers": ["lexical", ProviderKind.ENTITY]},
    )

    assert fake_engine.search_calls[0][1].enabled_providers == frozenset(
        {ProviderKind.LEXICAL, ProviderKind.ENTITY}
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"query_type_hint": "factual"}, "unsupported retrieval options"),
        ({"mode": "benchmark"}, "fast or balanced"),
    ],
)
def test_injected_engine_rejects_non_product_options(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
    options: dict[str, object],
    message: str,
) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    with pytest.raises(ValueError, match=message):
        service.search("Atlas database", Scope(user_id="user-a"), options)

    assert fake_engine.search_calls == []


def test_injected_engine_persists_sanitized_trace_and_search_audit(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    result = service.search("Atlas database", Scope(user_id="user-a"), {"mode": "balanced"})

    trace_id, trace, trace_scope = memory_store.saved_traces[-1]
    assert trace_id == result.trace_id
    assert trace_scope == Scope(user_id="user-a")
    assert "Atlas database" not in repr(trace)
    assert trace["allow_cross_session"] is False
    assert trace["include_session"] is False
    audit_scope, event_type, audit = memory_store.audit_events[-1]
    assert audit_scope == Scope(user_id="user-a")
    assert event_type == "memory.search"
    assert audit["trace_id"] == result.trace_id
    assert audit["payload"] == {
        "query_hash": stable_hash("Atlas database"),
        "query_length": len("Atlas database"),
        "intent": "factual",
        "mode": "balanced",
        "candidate_count": 1,
        "allow_cross_session": False,
        "include_session": False,
        "model_calls": {
            "count": 0,
            "model_versions": [],
            "total_latency_ms": 0,
            "usage": {},
        },
    }


def test_injected_engine_sanitizes_custom_intent_before_persistence(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    fake_engine.intent = "private custom intent"
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    service.search("Atlas database", Scope(user_id="user-a"))

    persisted_trace = memory_store.saved_traces[-1][1]
    audit_payload = memory_store.audit_events[-1][2]["payload"]
    assert "private custom intent" not in repr(persisted_trace)
    assert "private custom intent" not in repr(audit_payload)


def test_injected_engine_sanitizes_nested_trace_fields(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    private_text = "private nested Atlas trace"
    fake_engine.extra_trace = {
        "stages": ["plan", private_text],
        "providers": [{"kind": {"private": private_text}, "debug": private_text}],
        "selected_ids": [private_text],
        "stage_durations_ms": {private_text: private_text},
        "reranker_failure": private_text,
        "planner_fallback": private_text,
    }
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    service.search("Atlas database", Scope(user_id="user-a"))

    assert private_text not in repr(memory_store.saved_traces[-1][1])


def test_injected_engine_does_not_persist_raw_model_prompt(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    fake_engine.record_raw_model_call = True
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    service.search("private Atlas model prompt", Scope(user_id="user-a"))

    assert "private Atlas model prompt" not in repr(memory_store.saved_traces[-1][1])


def test_answer_context_authorizes_both_public_operations(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    authorizer = RecordingAuthorizer()
    service = MemoryService(
        store=memory_store,
        retrieval_engine=fake_engine,
        authorizer=authorizer,
    )

    service.answer_context(
        "Atlas database",
        Scope(user_id="user-a"),
        {"limit": 5, "token_budget": 321},
    )

    assert [call[0] for call in authorizer.calls] == [
        "memory.answer_context",
        "memory.search",
    ]
    assert fake_engine.pack_calls[0][3] == 321
