from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.models import EvidenceSpan, Scope, new_id
from fusion_memory.core.text import extract_entities, stable_hash
from fusion_memory.ingestion.window_builder import build_session_windows, chunk_document_message


def normalize_input(
    input_data: Any,
    scope: Scope,
    session_time: datetime | None,
    metadata: dict[str, Any] | None = None,
    config: MemoryConfig | None = None,
) -> list[EvidenceSpan]:
    scope.validate_for_add()
    metadata = metadata or {}
    config = config or DEFAULT_CONFIG
    timestamp = session_time or datetime.now(timezone.utc)
    messages = _to_messages(input_data)
    spans: list[EvidenceSpan] = []
    for index, message in enumerate(messages):
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        speaker = message.get("speaker") or message.get("role") or "user"
        if speaker == "developer":
            speaker = "system"
        if speaker not in {"user", "assistant", "agent", "tool", "system", "document"}:
            speaker = "user"
        span_type = message.get("span_type") or ("tool_result" if speaker == "tool" else "turn")
        if speaker == "document" or span_type == "document_chunk":
            spans.extend(
                chunk_document_message(
                    content,
                    scope,
                    _parse_ts(message.get("timestamp")) or timestamp,
                    speaker="document",
                    turn_id=message.get("turn_id") or f"doc_{index}",
                    source_uri=message.get("source_uri") or metadata.get("source_uri"),
                    metadata={**metadata, **message.get("metadata", {})},
                    chunk_size_tokens=int(message.get("chunk_size_tokens") or metadata.get("chunk_size_tokens") or config.chunk_size_tokens),
                    chunk_overlap_tokens=int(message.get("chunk_overlap_tokens") or metadata.get("chunk_overlap_tokens") or config.chunk_overlap_tokens),
                )
            )
            continue
        spans.append(
            EvidenceSpan(
                span_id=new_id("span"),
                scope=scope,
                turn_id=message.get("turn_id") or f"turn_{index}",
                speaker=speaker,
                span_type=span_type,
                content=content,
                content_hash=stable_hash(f"{speaker}:{content}"),
                timestamp=_parse_ts(message.get("timestamp")) or timestamp,
                source_uri=message.get("source_uri") or metadata.get("source_uri"),
                parent_span_id=message.get("parent_span_id"),
                entities=extract_entities(content),
                topics=[],
                metadata={**metadata, **message.get("metadata", {})},
            )
        )
    spans.extend(
        build_session_windows(
            spans,
            window_size=int(metadata.get("window_size", config.session_window_size)),
            min_window_spans=int(metadata.get("min_window_spans", config.min_window_spans)),
        )
    )
    return spans


def _to_messages(input_data: Any) -> list[dict[str, Any]]:
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if isinstance(input_data, dict):
        if "messages" in input_data:
            return list(input_data["messages"])
        return [input_data]
    if isinstance(input_data, list):
        out: list[dict[str, Any]] = []
        for item in input_data:
            if isinstance(item, str):
                out.append({"role": "user", "content": item})
            else:
                out.append(dict(item))
        return out
    raise TypeError("input must be a string, dict, or list of messages")


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
