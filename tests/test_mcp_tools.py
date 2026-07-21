import json

import pytest

from fusion_memory.core.models import Scope
from fusion_memory.mcp_runtime import FusionMemoryRuntime
from fusion_memory.mcp_server import _bounded_messages, _check_batch_payload_size, _error, _limit


def test_search_limit_is_clamped_to_mcp_boundary():
    assert _limit(0) == 1
    assert _limit(99) == 32


def test_batch_rejects_empty_message_content():
    with pytest.raises(ValueError, match="text must be non-empty"):
        _bounded_messages([{"content": ""}])


def test_batch_counts_serialized_source_bytes_and_rejects_message_metadata(monkeypatch):
    monkeypatch.setenv("FUSION_MEMORY_MCP_MAX_BATCH_BYTES", "64")

    with pytest.raises(ValueError, match="batch exceeds configured byte limit"):
        _bounded_messages([{"role": "user", "content": "ok", "source_uri": "x" * 100}])
    with pytest.raises(ValueError, match="unsupported message fields"):
        _bounded_messages([{"content": "ok", "metadata": {"x": "y" * 100}}])


def test_batch_rejects_oversized_top_level_metadata(monkeypatch):
    monkeypatch.setenv("FUSION_MEMORY_MCP_MAX_BATCH_BYTES", "64")

    with pytest.raises(ValueError, match="batch exceeds configured byte limit"):
        _check_batch_payload_size([{"content": "ok"}], "batch-1", {"history": "x" * 100})


@pytest.mark.parametrize(
    "message",
    [
        {"content": "ok", "role": ["user"]},
        {"content": "ok", "source": {"path": "history.jsonl"}},
        {"content": "ok", "source": {"path": "history.jsonl"}, "source_uri": "history://session-1"},
        {"content": "ok", "source_uri": 7},
        {"content": "ok", "timestamp": ["2026-07-20T12:00:00+00:00"]},
        {"content": "ok", "turn_id": 3},
        {"content": "ok", "timestamp": "not-a-timestamp"},
    ],
)
def test_batch_rejects_malformed_optional_history_fields(message):
    with pytest.raises(ValueError, match="must be a non-empty string|timestamp must be ISO-8601|only one of source or source_uri"):
        _bounded_messages([message])


def test_batch_normalizes_valid_history_source_alias_and_timestamp():
    assert _bounded_messages(
        [
            {
                "role": "assistant",
                "content": "remember this",
                "source": "history://session-1",
                "timestamp": "2026-07-20T12:00:00+00:00",
                "turn_id": "turn-7",
            }
        ]
    ) == [
        {
            "role": "assistant",
            "content": "remember this",
            "source_uri": "history://session-1",
            "timestamp": "2026-07-20T12:00:00+00:00",
            "turn_id": "turn-7",
        }
    ]


def test_batch_rejects_duplicate_source_aliases_before_byte_accounting(monkeypatch):
    monkeypatch.setenv("FUSION_MEMORY_MCP_MAX_BATCH_BYTES", "64")

    with pytest.raises(ValueError, match="only one of source or source_uri"):
        _bounded_messages([{"content": "ok", "source": "x" * 100, "source_uri": "history://session-1"}])


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
