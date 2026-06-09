from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from fusion_memory.core.config import DEFAULT_CONFIG
from fusion_memory.core.models import EvidenceSpan, Scope, new_id
from fusion_memory.core.text import compact_summary, extract_entities, stable_hash, tokenize


def chunk_document_message(
    content: str,
    scope: Scope,
    timestamp: datetime,
    *,
    speaker: str = "document",
    turn_id: str | None = None,
    source_uri: str | None = None,
    metadata: dict | None = None,
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> list[EvidenceSpan]:
    chunk_size_tokens = chunk_size_tokens or DEFAULT_CONFIG.chunk_size_tokens
    chunk_overlap_tokens = chunk_overlap_tokens or DEFAULT_CONFIG.chunk_overlap_tokens
    words = content.split()
    if len(words) <= chunk_size_tokens:
        return [
            EvidenceSpan(
                span_id=new_id("span"),
                scope=scope,
                turn_id=turn_id,
                speaker=speaker,
                span_type="document_chunk",
                content=content.strip(),
                content_hash=stable_hash(f"{speaker}:document_chunk:{content.strip()}"),
                timestamp=timestamp,
                source_uri=source_uri,
                entities=extract_entities(content),
                metadata={**(metadata or {}), "chunk_index": 0, "chunk_count": 1},
            )
        ]
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    chunks: list[EvidenceSpan] = []
    chunk_count = ((len(words) - chunk_overlap_tokens - 1) // step) + 1
    for index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size_tokens]
        if not chunk_words:
            continue
        text = " ".join(chunk_words)
        chunks.append(
            EvidenceSpan(
                span_id=new_id("span"),
                scope=scope,
                turn_id=f"{turn_id or 'doc'}_chunk_{index}",
                speaker=speaker,
                span_type="document_chunk",
                content=text,
                content_hash=stable_hash(f"{speaker}:document_chunk:{source_uri}:{index}:{text}"),
                timestamp=timestamp,
                source_uri=source_uri,
                entities=extract_entities(text),
                metadata={
                    **(metadata or {}),
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "token_start": start,
                    "token_end": start + len(chunk_words),
                },
            )
        )
        if start + chunk_size_tokens >= len(words):
            break
    return chunks


def build_session_windows(
    spans: Iterable[EvidenceSpan],
    *,
    window_size: int | None = None,
    min_window_spans: int | None = None,
) -> list[EvidenceSpan]:
    window_size = window_size or DEFAULT_CONFIG.session_window_size
    min_window_spans = min_window_spans or DEFAULT_CONFIG.min_window_spans
    raw_turns = [span for span in spans if span.span_type == "turn" and span.speaker in {"user", "assistant", "agent", "tool"}]
    if len(raw_turns) < min_window_spans:
        return []
    windows: list[EvidenceSpan] = []
    for index in range(0, len(raw_turns), window_size):
        group = raw_turns[index : index + window_size]
        if len(group) < min_window_spans:
            continue
        content = "\n".join(f"{span.speaker}: {span.content}" for span in group)
        scope = group[0].scope
        timestamp = group[-1].timestamp
        parent_ids = [span.span_id for span in group]
        windows.append(
            EvidenceSpan(
                span_id=new_id("span"),
                scope=scope,
                turn_id=f"window_{index // window_size}",
                speaker="system",
                span_type="window",
                content=content,
                content_hash=stable_hash(f"window:{content}"),
                timestamp=timestamp,
                parent_span_id=group[0].span_id,
                entities=sorted({entity for span in group for entity in span.entities}),
                topics=[],
                metadata={"parent_span_ids": parent_ids, "token_count": len(tokenize(content))},
            )
        )
    return windows


def build_session_summary_span(
    spans: Iterable[EvidenceSpan],
    scope: Scope,
    *,
    min_source_spans: int | None = None,
    max_source_spans: int | None = None,
    max_chars: int | None = None,
) -> EvidenceSpan | None:
    min_source_spans = min_source_spans or DEFAULT_CONFIG.session_summary_min_spans
    max_source_spans = max_source_spans or DEFAULT_CONFIG.session_summary_max_source_spans
    max_chars = max_chars or DEFAULT_CONFIG.session_summary_max_chars
    source_spans = [
        span
        for span in spans
        if span.span_type in {"turn", "tool_result", "document_chunk"} and span.speaker in {"user", "assistant", "agent", "tool", "document"}
    ]
    source_spans.sort(key=lambda span: (span.timestamp, span.turn_id or "", span.span_id))
    if len(source_spans) < min_source_spans:
        return None
    selected = source_spans[-max_source_spans:]
    parent_ids = [span.span_id for span in selected]
    lines = [
        f"Session summary for {scope.session_id or 'scope'}:",
        f"Covered {len(selected)} source spans from {selected[0].timestamp.isoformat()} to {selected[-1].timestamp.isoformat()}.",
    ]
    by_speaker: dict[str, list[str]] = {}
    for span in selected:
        by_speaker.setdefault(span.speaker, []).append(compact_summary(span.content, 180))
    for speaker in ["user", "assistant", "agent", "tool", "document"]:
        entries = by_speaker.get(speaker)
        if not entries:
            continue
        joined = " | ".join(entries)
        lines.append(f"{speaker}: {compact_summary(joined, max(160, max_chars // 4))}")
    entities = sorted({entity for span in selected for entity in span.entities})
    if entities:
        lines.append("Entities: " + ", ".join(entities[:24]))
    content = compact_summary("\n".join(lines), max_chars)
    source_hash = stable_hash("|".join(parent_ids))
    return EvidenceSpan(
        span_id=new_id("span"),
        scope=scope,
        turn_id=f"summary_{scope.session_id or source_hash[:12]}",
        speaker="system",
        span_type="summary",
        content=content,
        content_hash=stable_hash(f"summary:{scope.workspace_id}:{scope.user_id}:{scope.agent_id}:{scope.run_id}:{scope.session_id}:{source_hash}:{content}"),
        timestamp=selected[-1].timestamp,
        parent_span_id=selected[0].span_id,
        entities=entities,
        topics=[],
        metadata={
            "parent_span_ids": parent_ids,
            "source_span_count": len(parent_ids),
            "summary_version": "local-session-summary-v0",
            "token_count": len(tokenize(content)),
        },
    )
