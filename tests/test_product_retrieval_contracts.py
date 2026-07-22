from datetime import datetime, timezone

import pytest

from fusion_memory.core.models import Scope
from fusion_memory.retrieval.context import ProviderKind, RetrievalContext, SearchRequest


def test_product_search_request_rejects_benchmark_mode() -> None:
    with pytest.raises(ValueError, match="fast or balanced"):
        SearchRequest(query="where is the decision", limit=12, mode="benchmark")


def test_retrieval_context_rejects_authenticated_user_mismatch() -> None:
    with pytest.raises(ValueError, match="user_id"):
        RetrievalContext(
            scope=Scope(user_id="user-b"),
            user_id="user-a",
            now=datetime.now(timezone.utc),
            trace_id="trace-1",
            deadline=None,
            include_session=False,
        )


def test_search_request_uses_immutable_provider_filter() -> None:
    request = SearchRequest(
        query="latest Atlas database",
        limit=8,
        enabled_providers=frozenset({ProviderKind.LEXICAL, ProviderKind.TEMPORAL}),
    )
    assert request.enabled_providers == frozenset({ProviderKind.LEXICAL, ProviderKind.TEMPORAL})
