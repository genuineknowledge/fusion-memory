from fusion_memory.core.models import Scope
from fusion_memory.mcp_server import RequestProvenance


def test_mcp_read_scope_still_contains_only_authenticated_user():
    provenance = RequestProvenance("workspace-a", "session-a")
    write_scope = Scope(
        user_id="user-a",
        workspace_id=provenance.workspace_id,
        session_id=provenance.session_id,
        app_id="mcp",
    )
    read_scope = Scope(user_id="user-a", app_id="mcp")

    assert write_scope.workspace_id == "workspace-a"
    assert write_scope.session_id == "session-a"
    assert read_scope.workspace_id is None
    assert read_scope.session_id is None
