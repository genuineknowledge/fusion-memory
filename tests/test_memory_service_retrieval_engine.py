from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fusion_memory import AuthorizationError, MemoryService, Scope
from fusion_memory.core.config import MemoryConfig
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


class UnhashableStr(str):
    __hash__ = None


class RecordingStore:
    def __init__(self) -> None:
        self.saved_traces: list[tuple[str, dict[str, object], Scope]] = []
        self.audit_events: list[tuple[Scope, str, dict[str, object]]] = []
        self.persistence_calls: list[str] = []
        self.trace_error: Exception | None = None
        self.audit_error: Exception | None = None

    def save_trace(self, trace_id: str, trace: dict[str, object], scope: Scope) -> None:
        self.persistence_calls.append("save_trace")
        if self.trace_error is not None:
            raise self.trace_error
        self.saved_traces.append((trace_id, trace, scope))

    def insert_audit_event(self, scope: Scope, event_type: str, **kwargs: object) -> str:
        self.persistence_calls.append("insert_audit_event")
        if self.audit_error is not None:
            raise self.audit_error
        self.audit_events.append((scope, event_type, kwargs))
        return "audit-1"

    def close(self) -> None:
        pass


class RecordingAuthorizer:
    def __init__(self, denied: set[str] | None = None) -> None:
        self.denied = denied or set()
        self.calls: list[tuple[str, Scope, dict[str, object]]] = []

    def authorize(
        self,
        operation: str,
        scope: Scope,
        context: dict[str, object] | None = None,
    ) -> None:
        self.calls.append((operation, scope, dict(context or {})))
        if operation in self.denied:
            raise AuthorizationError(f"denied: {operation}")


class FakeEngine:
    def __init__(self) -> None:
        self.search_calls: list[tuple[RetrievalContext, SearchRequest]] = []
        self.pack_calls: list[tuple[RetrievalContext, SearchRequest, RetrievalResult, int]] = []
        self.calls: list[dict[str, object]] = []
        self.last_result: RetrievalResult | None = None
        self.intent = "factual"
        self.extra_trace: dict[str, object] = {}
        self.model_call: dict[str, object] | None = None

    def search(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan | None = None,
    ) -> RetrievalResult:
        assert plan is None
        self.search_calls.append((context, request))
        if self.model_call is not None:
            self.calls.append(self.model_call)
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


class ScopeFilteringEngine(FakeEngine):
    records = (
        ("user-a", "session-1", "user-a-session-1-marker"),
        ("user-a", "session-2", "user-a-session-2-marker"),
        ("user-b", "session-3", "user-b-session-3-marker"),
    )

    def search(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan | None = None,
    ) -> RetrievalResult:
        base_result = super().search(context, request, plan)
        visible = [
            (owner, session, text)
            for owner, session, text in self.records
            if owner == context.scope.user_id
            and (
                not context.include_session
                or session == context.scope.session_id
            )
        ]
        candidates = tuple(
            Candidate(
                id=f"candidate-{index}",
                type="span",
                text=text,
                source="scope-filtering-engine",
                scores={},
                source_span_ids=[f"span-{index}"],
                metadata={"session_id": session},
            )
            for index, (_, session, text) in enumerate(visible)
        )
        self.last_result = RetrievalResult(
            candidates=candidates,
            coverage=base_result.coverage,
            trace=base_result.trace,
            plan=base_result.plan,
        )
        return self.last_result


@pytest.fixture
def memory_store() -> RecordingStore:
    return RecordingStore()


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


def test_memory_service_uses_product_engine_by_default() -> None:
    service = MemoryService()
    try:
        assert service.retrieval_engine.__class__.__name__ == "ProductRetrievalEngine"
    finally:
        service.close()


def test_product_engine_factory_wires_all_providers_pack_builder_and_falsey_planner(
    memory_store: RecordingStore,
) -> None:
    from fusion_memory.retrieval.engine import build_product_retrieval_engine
    from fusion_memory.retrieval.evidence_pack import ProductEvidencePackBuilder

    class FalseyPlanner:
        def __bool__(self) -> bool:
            return False

    planner = FalseyPlanner()
    reranker = object()

    engine = build_product_retrieval_engine(
        memory_store,
        MemoryConfig(),
        reranker,
        planner=planner,
    )

    assert engine.planner is planner
    assert engine.reranker is reranker
    assert isinstance(engine.pack_builder, ProductEvidencePackBuilder)
    assert set(engine.registry._providers) == set(ProviderKind)
    assert all(
        provider.repository is memory_store
        for provider in engine.registry._providers.values()
    )


@pytest.mark.parametrize("operation", ["search", "answer_context"])
def test_memory_service_never_falls_back_when_product_engine_is_unavailable(
    memory_store: RecordingStore,
    fake_engine: FakeEngine,
    operation: str,
) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)
    service.retrieval_engine = None

    with pytest.raises(RuntimeError, match="retrieval engine is not configured"):
        getattr(service, operation)("Atlas database", Scope(user_id="user-a"))


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


@pytest.mark.parametrize(
    ("source", "expected_providers"),
    [
        (
            "raw",
            {
                ProviderKind.VECTOR,
                ProviderKind.LEXICAL,
                ProviderKind.TEMPORAL,
                ProviderKind.CHRONOLOGY,
            },
        ),
        ("exact", {ProviderKind.LEXICAL}),
        ("entities", {ProviderKind.ENTITY}),
        ("facts", {ProviderKind.VECTOR, ProviderKind.LEXICAL}),
        (
            "events",
            {
                ProviderKind.VECTOR,
                ProviderKind.TEMPORAL,
                ProviderKind.CHRONOLOGY,
            },
        ),
        ("views", {ProviderKind.LEXICAL}),
        ("profiles", {ProviderKind.LEXICAL, ProviderKind.ENTITY}),
    ],
)
def test_injected_engine_translates_each_source_family_exactly(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
    source: str,
    expected_providers: set[ProviderKind],
) -> None:
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    service.search(
        "Atlas database",
        Scope(user_id="user-a"),
        {"enabled_sources": [source]},
    )

    assert fake_engine.search_calls[0][1].enabled_providers == frozenset(
        expected_providers
    )


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


@pytest.mark.parametrize(
    ("options", "message", "secret"),
    [
        ({"mode": ["private-mode-secret"]}, "mode must be fast or balanced", "private-mode-secret"),
        ({"mode": {"private-mode-secret": True}}, "mode must be fast or balanced", "private-mode-secret"),
        (
            {"mode": UnhashableStr("private-mode-secret")},
            "mode must be fast or balanced",
            "private-mode-secret",
        ),
        (
            {"enabled_providers": [["private-provider-secret"]]},
            "enabled_providers contains an unsupported provider",
            "private-provider-secret",
        ),
        (
            {"enabled_providers": UnhashableStr("private-provider-secret")},
            "enabled_providers contains an unsupported provider",
            "private-provider-secret",
        ),
        (
            {"enabled_providers": 42},
            "enabled_providers contains an unsupported provider",
            "42",
        ),
        (
            {"enabled_sources": [["private-source-secret"]]},
            "enabled_sources contains an unsupported source family",
            "private-source-secret",
        ),
        (
            {"enabled_sources": UnhashableStr("private-source-secret")},
            "enabled_sources contains an unsupported source family",
            "private-source-secret",
        ),
        (
            {"enabled_sources": 42},
            "enabled_sources contains an unsupported source family",
            "42",
        ),
    ],
)
@pytest.mark.parametrize("operation", ["search", "answer_context"])
def test_injected_engine_validation_errors_are_stable_unchained_and_non_reflective(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
    options: dict[str, object],
    message: str,
    secret: str,
    operation: str,
) -> None:
    authorizer = RecordingAuthorizer({"memory.search", "memory.answer_context"})
    service = MemoryService(
        store=memory_store,
        retrieval_engine=fake_engine,
        authorizer=authorizer,
    )

    with pytest.raises(ValueError) as error:
        if operation == "search":
            service.search("Atlas database", Scope(user_id="user-a"), options)
        else:
            service.answer_context("Atlas database", Scope(user_id="user-a"), options)

    assert str(error.value) == message
    assert secret not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
    assert authorizer.calls == []
    assert fake_engine.search_calls == []


def test_injected_engine_authorization_denial_precedes_search_and_persistence(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    authorizer = RecordingAuthorizer({"memory.search"})
    service = MemoryService(
        store=memory_store,
        retrieval_engine=fake_engine,
        authorizer=authorizer,
    )

    with pytest.raises(AuthorizationError, match="memory.search"):
        service.search("Atlas database", Scope(user_id="user-a"))

    assert fake_engine.search_calls == []
    assert memory_store.persistence_calls == []


def test_injected_engine_invalid_read_scope_precedes_authorization_and_search(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    authorizer = RecordingAuthorizer()
    service = MemoryService(
        store=memory_store,
        retrieval_engine=fake_engine,
        authorizer=authorizer,
    )

    with pytest.raises(ValueError, match="read requires"):
        service.search("Atlas database", Scope())

    assert authorizer.calls == []
    assert fake_engine.search_calls == []
    assert memory_store.persistence_calls == []


def test_injected_engine_propagates_deadline_trace_id_and_trace_preference(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    deadline = datetime(2030, 1, 2, 3, 4, tzinfo=timezone.utc)
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    result = service.search(
        "Atlas database",
        Scope(user_id="user-a"),
        {"deadline": deadline, "include_trace": False},
    )

    context, request = fake_engine.search_calls[0]
    audit = memory_store.audit_events[0][2]
    assert context.deadline is deadline
    assert request.include_trace is False
    assert context.trace_id == result.trace_id
    assert memory_store.saved_traces[0][0] == result.trace_id
    assert audit["trace_id"] == result.trace_id


def test_injected_engine_boundary_supports_same_user_cross_session_and_user_isolation(
    memory_store: RecordingStore,
) -> None:
    engine = ScopeFilteringEngine()
    service = MemoryService(store=memory_store, retrieval_engine=engine)

    session_result = service.search(
        "marker",
        Scope(user_id="user-a", session_id="session-1"),
    )
    cross_session_result = service.search(
        "marker",
        Scope(user_id="user-a", session_id="session-1"),
        {"allow_cross_session": True},
    )
    other_user_result = service.search(
        "marker",
        Scope(user_id="user-b"),
        {"allow_cross_session": True},
    )

    assert [candidate.text for candidate in session_result.candidates] == [
        "user-a-session-1-marker"
    ]
    assert {candidate.text for candidate in cross_session_result.candidates} == {
        "user-a-session-1-marker",
        "user-a-session-2-marker",
    }
    assert [candidate.text for candidate in other_user_result.candidates] == [
        "user-b-session-3-marker"
    ]


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


def test_injected_engine_structurally_sanitizes_model_calls_in_trace_and_audit(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    secrets = {
        "private Atlas model query",
        "private-model-secret",
        "private-version-secret",
        "private-usage-key-secret",
        "private-usage-value-secret",
        "private-nested-usage-secret",
        "private-endpoint-secret",
        "private-token-secret",
        "private-credential-secret",
    }
    fake_engine.model_call = {
        "prompt": "private Atlas model query",
        "input": "private Atlas model query",
        "model": "private-model-secret",
        "model_version": "private-version-secret",
        "endpoint": "private-endpoint-secret",
        "token": "private-token-secret",
        "credentials": "private-credential-secret",
        "latency_ms": 12.5,
        "usage": {
            "total_tokens": 17,
            "prompt_tokens": 11,
            "completion_tokens": "private-usage-value-secret",
            "private-usage-key-secret": 99,
            "nested": {"value": "private-nested-usage-secret"},
        },
        "arbitrary": {"query": "private Atlas model query"},
    }
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    service.search("private Atlas model query", Scope(user_id="user-a"))

    trace = memory_store.saved_traces[-1][1]
    audit_payload = memory_store.audit_events[-1][2]["payload"]
    persisted = repr((trace, audit_payload))
    for secret in secrets:
        assert secret not in persisted
    assert trace["model_calls"] == [
        {
            "component": "retrieval_engine",
            "model": f"hashed_{stable_hash('private-model-secret')[:16]}",
            "model_version": f"hashed_{stable_hash('private-version-secret')[:16]}",
            "latency_ms": 12.5,
            "usage": {"prompt_tokens": 11.0, "total_tokens": 17.0},
        }
    ]
    assert audit_payload["model_calls"] == {
        "count": 1,
        "model_versions": [
            f"hashed_{stable_hash('private-version-secret')[:16]}"
        ],
        "total_latency_ms": 12.5,
        "usage": {"prompt_tokens": 11.0, "total_tokens": 17.0},
    }


def test_trace_persistence_failure_prevents_audit_write(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    memory_store.trace_error = RuntimeError("trace persistence failed")
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    with pytest.raises(RuntimeError, match="trace persistence failed"):
        service.search("Atlas database", Scope(user_id="user-a"))

    assert len(fake_engine.search_calls) == 1
    assert memory_store.persistence_calls == ["save_trace"]
    assert memory_store.audit_events == []


def test_audit_persistence_failure_occurs_after_trace_write(
    fake_engine: FakeEngine,
    memory_store: RecordingStore,
) -> None:
    memory_store.audit_error = RuntimeError("audit persistence failed")
    service = MemoryService(store=memory_store, retrieval_engine=fake_engine)

    with pytest.raises(RuntimeError, match="audit persistence failed"):
        service.search("Atlas database", Scope(user_id="user-a"))

    assert len(fake_engine.search_calls) == 1
    assert memory_store.persistence_calls == ["save_trace", "insert_audit_event"]
    assert len(memory_store.saved_traces) == 1


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
