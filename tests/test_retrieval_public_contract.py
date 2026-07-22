from fusion_memory import MemoryService, Scope
from fusion_memory.core.models import EvidencePack, SearchResult


def test_public_search_and_answer_context_shapes_remain_stable() -> None:
    service = MemoryService()
    write_scope = Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a")
    read_scope = Scope(user_id="user-a")
    try:
        service.add("Atlas uses Qdrant for retrieval.", write_scope)
        result = service.search("What does Atlas use?", read_scope, {"allow_cross_session": True})
        pack = service.answer_context("What does Atlas use?", read_scope, {"allow_cross_session": True})
        assert isinstance(result, SearchResult)
        assert result.trace_id
        assert isinstance(result.coverage, dict)
        assert isinstance(pack, EvidencePack)
        assert any(span["session_id"] == "session-a" for span in pack.source_spans)
    finally:
        service.close()


def test_public_read_is_same_user_cross_session_and_different_user_isolated() -> None:
    service = MemoryService()
    try:
        service.add("Private marker cobalt-key belongs to user A.", Scope(user_id="user-a", workspace_id="a", session_id="s1"))
        same_user = service.search("Which private marker?", Scope(user_id="user-a"), {"allow_cross_session": True})
        other_user = service.search("Which private marker?", Scope(user_id="user-b"), {"allow_cross_session": True})
        assert any("cobalt-key" in candidate.text for candidate in same_user.candidates)
        assert all("cobalt-key" not in candidate.text for candidate in other_user.candidates)
    finally:
        service.close()
