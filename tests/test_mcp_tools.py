import json

import pytest

from fusion_memory.core.models import Scope
from fusion_memory.mcp_runtime import FusionMemoryRuntime
from fusion_memory.mcp_server import _bounded_messages, _error, _limit


def test_search_limit_is_clamped_to_mcp_boundary():
    assert _limit(0) == 1
    assert _limit(99) == 32


def test_batch_rejects_empty_message_content():
    with pytest.raises(ValueError, match="text must be non-empty"):
        _bounded_messages([{"content": ""}])


@pytest.mark.anyio
async def test_add_batch_reports_batch_id_count_and_add_result():
    class Service:
        def add(self, input_data, scope, metadata=None):
            assert input_data == {"messages": [{"content": "first"}, {"content": "second"}]}
            assert metadata == {"batch_id": "batch-1", "source": "history"}
            return {"span_ids": ["span-1", "span-2"]}

        def close(self):
            pass

    class Executor:
        def run(self, callback, **kwargs):
            return callback(object())

    runtime = FusionMemoryRuntime(Executor(), lambda _store: Service())
    result = await runtime.add_batch(
        scope=Scope(user_id="user-a"),
        messages=[{"content": "first"}, {"content": "second"}],
        batch_id="batch-1",
        metadata={"batch_id": "forged", "source": "history"},
    )

    assert result == {
        "batch_id": "batch-1",
        "message_count": 2,
        "add_result": {"span_ids": ["span-1", "span-2"]},
    }


def test_structured_error_never_echoes_sensitive_exception_text():
    result = _error("invalid_request", retryable=False, message="Bearer token-a")

    assert "token-a" not in json.dumps(result)
