from __future__ import annotations

import pytest

from fusion_memory import MemoryService, Scope


@pytest.mark.parametrize(
    ("memories", "query", "required_text"),
    [
        (["I prefer Qdrant for Atlas retrieval."], "What database do I prefer for Atlas?", "Qdrant"),
        (["Atlas deployment deadline is July 30."], "When is the Atlas deployment deadline?", "July 30"),
        (["The incident started, then mitigation completed."], "List the incident events in order.", "mitigation"),
        (["The internal project code is ZINC-42."], "What is the internal project code?", "ZINC-42"),
    ],
)
def test_product_queries_retrieve_required_evidence(
    memories: list[str],
    query: str,
    required_text: str,
) -> None:
    service = MemoryService()
    scope = Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a")
    try:
        for memory in memories:
            service.add({"role": "user", "content": memory}, scope)

        result = service.search(
            query,
            Scope(user_id="user-a"),
            {"limit": 10, "mode": "balanced"},
        )

        assert any(
            required_text.lower() in candidate.text.lower()
            for candidate in result.candidates
        )
    finally:
        service.close()


def test_same_user_reads_across_workspace_and_session() -> None:
    service = MemoryService()
    try:
        service.add(
            "Workspace A remembers cobalt-key.",
            Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a"),
        )
        service.add(
            "Workspace B remembers amber-key.",
            Scope(user_id="user-a", workspace_id="workspace-b", session_id="session-b"),
        )

        result = service.search(
            "Which keys were remembered?",
            Scope(user_id="user-a"),
            {"limit": 10},
        )

        texts = [candidate.text for candidate in result.candidates]
        assert any("cobalt-key" in text for text in texts)
        assert any("amber-key" in text for text in texts)
    finally:
        service.close()


def test_different_user_never_receives_other_users_candidate() -> None:
    service = MemoryService()
    try:
        service.add(
            "User A secret is cobalt-key.",
            Scope(user_id="user-a", workspace_id="workspace-a", session_id="session-a"),
        )

        result = service.search(
            "What secret was stored?",
            Scope(user_id="user-b"),
            {"limit": 10},
        )

        assert all("cobalt-key" not in candidate.text for candidate in result.candidates)
    finally:
        service.close()
