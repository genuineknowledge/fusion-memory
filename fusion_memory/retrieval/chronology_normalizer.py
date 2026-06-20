from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from fusion_memory.core.chronology import ChronologyEventEdge, ChronologyEventNode, ChronologyPhase, ChronologyTopic
from fusion_memory.core.models import EvidenceSpan, MemoryEvent, Scope
from fusion_memory.core.text import compact_summary, stable_hash, tokenize
from fusion_memory.retrieval.taxonomy import TaxonomyEntry, taxonomy_entry_for_text
from fusion_memory.retrieval.topic_clustering import (
    TopicClusterDecision,
    cluster_topic_label,
    cluster_topic_telemetry,
)


@dataclass
class ChronologyWriteBatch:
    topics: list[ChronologyTopic]
    phases: list[ChronologyPhase]
    nodes: list[ChronologyEventNode]
    edges: list[ChronologyEventEdge]
    telemetry: dict[str, object]


ORDER_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("first", ("first", "initially", "started", "start", "首先", "一开始", "先")),
    ("then", ("after that", "then", "next", "later", "然后", "接着", "随后")),
    ("finally", ("finally", "最后")),
    ("before", ("before", "之前")),
    ("after", ("after", "之后")),
)

PHASE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("setup", ("set up", "setup", "initialize", "初始化", "schema", "配置")),
    ("implementation", ("implement", "build", "add", "实现", "开发", "完成")),
    ("debug", ("debug", "fix", "error", "修复", "报错")),
    ("validation", ("test", "verify", "coverage", "测试", "验证")),
    ("release", ("deploy", "release", "上线", "部署")),
)

STOPWORDS = {
    "a",
    "after",
    "an",
    "and",
    "first",
    "i",
    "implemented",
    "next",
    "later",
    "set",
    "the",
    "then",
    "up",
}

ASPECT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("core functionality", ("core functionality", "user authentication", "expense tracking", "data visualization")),
    ("triangle classification", ("classifying triangles", "triangle classification", "equilateral", "isosceles", "scalene")),
    ("triangle area methods", ("triangle areas", "area calculation", "median formulas", "altitude", "median formula")),
    ("transaction CRUD implementation", ("transaction crud", "crud implementation", "add transaction", "view transactions")),
    ("transaction error handling", ("error handling", "try-except", "try except", "integrityerror", "exception")),
    ("deployment configuration", ("deployment", "deploy", "netlify", "github pages", "render")),
    ("integration test coverage", ("integration test", "test coverage", "nock", "mock responses")),
    ("database schema", ("database schema", "schema", "models", "tables")),
    ("password hashing", ("password hashing", "password_hash", "check_password_hash", "generate_password_hash")),
    ("query optimization", ("query optimization", "optimize my database queries", "indexes")),
)

EPISODE_TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("budget tracker", ("budget tracker", "budget app", "transactions", "expense", "income")),
    ("triangle geometry", ("triangle", "triangles", "equilateral", "isosceles", "scalene", "median", "altitude", "cosine", "cosines")),
    ("weather app", ("weather app", "city autocomplete", "openweather", "invalid city", "fetchweatherdata")),
    ("career development", ("resume", "linkedin", "profile", "job", "interview", "relocation")),
    ("probability concepts", ("probability", "permutation", "combination", "conditional probability")),
)

PHASE_ORDER_HINTS = {
    "setup": 10,
    "implementation": 20,
    "debug": 30,
    "validation": 40,
    "release": 50,
    "unknown": None,
}

DETERMINISTIC_CREATED_AT_FALLBACK = datetime(1970, 1, 1, tzinfo=timezone.utc)


def build_chronology_write_batch(
    scope: Scope, spans: list[EvidenceSpan], events: list[MemoryEvent]
) -> ChronologyWriteBatch:
    span_by_id = {span.span_id: span for span in spans}
    created_at = _created_at(events, spans)
    topics_by_label: dict[str, ChronologyTopic] = {}
    phases_by_key: dict[tuple[str, str], ChronologyPhase] = {}
    nodes: list[ChronologyEventNode] = []
    last_topic_label: str | None = None
    session_topic_hint = _session_topic_hint(spans, events)
    cluster_decisions: list[TopicClusterDecision] = []

    for index, event in enumerate(events):
        span = _first_source_span(event, span_by_id)
        text = span.content if span is not None else event.description
        if _skip_chronology_node(event, span):
            continue
        language = _infer_language(text)
        topic_label, topic_is_strong, taxonomy_entry, cluster_decision = _infer_topic_label(
            text,
            session_topic_hint=session_topic_hint,
            previous_label=last_topic_label,
        )
        cluster_decisions.append(cluster_decision)
        if not topic_is_strong and last_topic_label is not None and _infer_order_marker(text) is not None:
            topic_label = last_topic_label
            taxonomy_entry = None
        last_topic_label = topic_label
        topic = _get_topic(scope, topics_by_label, topic_label, language, event, span, created_at, taxonomy_entry)
        phase_type = _infer_phase_type(text)
        phase = _get_phase(phases_by_key, topic, phase_type, event, span, created_at)
        actor = event.participants[0] if event.participants else "unknown"
        node = ChronologyEventNode(
            node_id=_id("chron_node", scope, event.event_id, span.span_id if span is not None else "", index),
            scope=scope,
            actor=actor,
            action=_infer_action(text),
            object=_infer_aspect_label(text, topic.canonical_label),
            topic_id=topic.topic_id,
            phase_id=phase.phase_id,
            timestamp=event.time_start or (span.timestamp if span is not None else None),
            source_span_id=span.span_id if span is not None else None,
            source_turn_id=span.turn_id if span is not None else None,
            text=compact_summary(text),
            language=language,
            confidence=event.confidence,
            explicit_order_marker=_infer_order_marker(text),
            created_at=created_at,
        )
        nodes.append(node)
    _append_user_span_nodes(
        scope,
        spans,
        topics_by_label,
        phases_by_key,
        nodes,
        created_at,
        session_topic_hint,
        cluster_decisions,
    )

    recognized_phase_ids = {
        phase.phase_id for phase in phases_by_key.values() if phase.phase_type != "unknown"
    }
    edges = _build_edges(nodes, created_at, recognized_phase_ids)
    telemetry = {
        "input_span_count": len(spans),
        "input_event_count": len(events),
        "topic_count": len(topics_by_label),
        "phase_count": len(phases_by_key),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "topic_cluster": cluster_topic_telemetry(cluster_decisions),
    }
    return ChronologyWriteBatch(
        topics=list(topics_by_label.values()),
        phases=list(phases_by_key.values()),
        nodes=nodes,
        edges=edges,
        telemetry=telemetry,
    )


def _created_at(events: list[MemoryEvent], spans: list[EvidenceSpan]) -> datetime:
    timestamps = [event.time_start for event in events if event.time_start is not None]
    timestamps.extend(span.timestamp for span in spans)
    return min(timestamps) if timestamps else DETERMINISTIC_CREATED_AT_FALLBACK


def _append_user_span_nodes(
    scope: Scope,
    spans: list[EvidenceSpan],
    topics_by_label: dict[str, ChronologyTopic],
    phases_by_key: dict[tuple[str, str], ChronologyPhase],
    nodes: list[ChronologyEventNode],
    created_at: datetime,
    session_topic_hint: str | None,
    cluster_decisions: list[TopicClusterDecision],
) -> None:
    existing_span_ids = {node.source_span_id for node in nodes if node.source_span_id}
    last_topic_label = _last_node_topic_label(nodes, topics_by_label)
    for index, span in enumerate(spans):
        if span.span_id in existing_span_ids or not _span_is_user_chronology_source(span):
            continue
        text = span.content
        language = _infer_language(text)
        topic_label, topic_is_strong, taxonomy_entry, cluster_decision = _infer_topic_label(
            text,
            session_topic_hint=session_topic_hint,
            previous_label=last_topic_label,
        )
        cluster_decisions.append(cluster_decision)
        if not topic_is_strong and last_topic_label is not None and _infer_order_marker(text) is not None:
            topic_label = last_topic_label
            taxonomy_entry = None
        topic = _get_topic_for_span(scope, topics_by_label, topic_label, language, span, created_at, taxonomy_entry)
        phase = _get_phase_for_span(phases_by_key, topic, _infer_phase_type(text), span, created_at)
        nodes.append(
            ChronologyEventNode(
                node_id=_id("chron_node", scope, "span", span.span_id, index),
                scope=scope,
                actor="user",
                action=_infer_action(text),
                object=_infer_aspect_label(text, topic.canonical_label),
                topic_id=topic.topic_id,
                phase_id=phase.phase_id,
                timestamp=span.timestamp,
                source_span_id=span.span_id,
                source_turn_id=span.turn_id,
                text=compact_summary(text),
                language=language,
                confidence=0.72,
                explicit_order_marker=_infer_order_marker(text),
                created_at=created_at,
            )
        )
        last_topic_label = topic_label


def _last_node_topic_label(nodes: list[ChronologyEventNode], topics_by_label: dict[str, ChronologyTopic]) -> str | None:
    if not nodes:
        return None
    topic_by_id = {topic.topic_id: topic for topic in topics_by_label.values()}
    topic = topic_by_id.get(str(nodes[-1].topic_id or ""))
    return topic.canonical_label if topic is not None else None


def _span_is_user_chronology_source(span: EvidenceSpan) -> bool:
    if span.speaker != "user" or span.span_type not in {"turn", "document_chunk", "tool_result"}:
        return False
    text = span.content.strip()
    if len(text) < 24:
        return False
    lowered = text.lower()
    if re.fullmatch(r"(?:thanks|thank you|ok|okay|sure)[.! ]*", lowered):
        return False
    return True


def _first_source_span(event: MemoryEvent, span_by_id: dict[str, EvidenceSpan]) -> EvidenceSpan | None:
    for span_id in event.source_span_ids:
        span = span_by_id.get(span_id)
        if span is not None:
            return span
    return None


def _infer_language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _infer_order_marker(text: str) -> str | None:
    lowered = text.lower()
    for marker, phrases in ORDER_MARKERS:
        if any(_contains_phrase(lowered, phrase) for phrase in phrases):
            return marker
    return None


def _infer_phase_type(text: str) -> str:
    lowered = text.lower()
    for phase_type, phrases in PHASE_RULES:
        if any(_contains_phrase(lowered, phrase) for phrase in phrases):
            return phase_type
    return "unknown"


def _contains_phrase(lowered_text: str, phrase: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", phrase):
        return phrase in lowered_text
    return re.search(rf"\b{re.escape(phrase)}\b", lowered_text) is not None


def _infer_topic_label(
    text: str,
    *,
    session_topic_hint: str | None = None,
    previous_label: str | None = None,
) -> tuple[str, bool, TaxonomyEntry | None, TopicClusterDecision]:
    decision = cluster_topic_label(text, session_hint=session_topic_hint, previous_label=previous_label)
    taxonomy_entry = taxonomy_entry_for_text(text)
    if taxonomy_entry is not None:
        return decision.label, _taxonomy_match_is_strong(taxonomy_entry), taxonomy_entry, decision
    topic_is_strong = "taxonomy" in decision.reasons or "session_hint" in decision.reasons
    return decision.label, topic_is_strong, None, decision


def _session_topic_hint(spans: list[EvidenceSpan], events: list[MemoryEvent]) -> str | None:
    text = " ".join(
        [
            *(span.content for span in spans if span.speaker == "user"),
            *(event.description for event in events if "assistant" not in {str(p).lower() for p in event.participants}),
        ]
    )
    lowered = text.lower()
    scored: list[tuple[int, str]] = []
    for label, phrases in EPISODE_TOPIC_RULES:
        score = sum(1 for phrase in phrases if _contains_phrase(lowered, phrase))
        if score:
            scored.append((score, label))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _text_matches_episode_topic(text: str, topic_label: str) -> bool:
    lowered = text.lower()
    for label, phrases in EPISODE_TOPIC_RULES:
        if label != topic_label:
            continue
        return any(_contains_phrase(lowered, phrase) for phrase in phrases)
    return False


def _get_topic(
    scope: Scope,
    topics_by_label: dict[str, ChronologyTopic],
    topic_label: str,
    language: str,
    event: MemoryEvent,
    span: EvidenceSpan | None,
    created_at: datetime,
    taxonomy_entry: TaxonomyEntry | None,
) -> ChronologyTopic:
    topic = topics_by_label.get(topic_label)
    if topic is not None:
        if span is not None and span.span_id not in topic.source_span_ids:
            topic.source_span_ids.append(span.span_id)
        if taxonomy_entry is not None:
            _merge_topic_taxonomy(topic, taxonomy_entry)
        return topic

    source_span_ids = [span.span_id] if span is not None else list(event.source_span_ids)
    aliases = list(dict.fromkeys(taxonomy_entry.aliases)) if taxonomy_entry is not None else []
    taxonomy_tags = list(dict.fromkeys(taxonomy_entry.tags)) if taxonomy_entry is not None else []
    topic_language = taxonomy_entry.language if taxonomy_entry is not None and taxonomy_entry.language != "unknown" else language
    topic = ChronologyTopic(
        topic_id=_id("chron_topic", scope, topic_label),
        scope=scope,
        canonical_label=topic_label,
        aliases=aliases,
        language=topic_language,
        taxonomy_tags=taxonomy_tags,
        source_span_ids=source_span_ids,
        confidence=event.confidence,
        created_at=created_at,
    )
    topics_by_label[topic_label] = topic
    return topic


def _get_topic_for_span(
    scope: Scope,
    topics_by_label: dict[str, ChronologyTopic],
    topic_label: str,
    language: str,
    span: EvidenceSpan,
    created_at: datetime,
    taxonomy_entry: TaxonomyEntry | None,
) -> ChronologyTopic:
    topic = topics_by_label.get(topic_label)
    if topic is not None:
        if span.span_id not in topic.source_span_ids:
            topic.source_span_ids.append(span.span_id)
        if taxonomy_entry is not None:
            _merge_topic_taxonomy(topic, taxonomy_entry)
        return topic

    aliases = list(dict.fromkeys(taxonomy_entry.aliases)) if taxonomy_entry is not None else []
    taxonomy_tags = list(dict.fromkeys(taxonomy_entry.tags)) if taxonomy_entry is not None else []
    topic_language = taxonomy_entry.language if taxonomy_entry is not None and taxonomy_entry.language != "unknown" else language
    topic = ChronologyTopic(
        topic_id=_id("chron_topic", scope, topic_label),
        scope=scope,
        canonical_label=topic_label,
        aliases=aliases,
        language=topic_language,
        taxonomy_tags=taxonomy_tags,
        source_span_ids=[span.span_id],
        confidence=0.72,
        created_at=created_at,
    )
    topics_by_label[topic_label] = topic
    return topic


def _merge_topic_taxonomy(topic: ChronologyTopic, taxonomy_entry: TaxonomyEntry) -> None:
    for alias in taxonomy_entry.aliases:
        if alias not in topic.aliases:
            topic.aliases.append(alias)
    for tag in taxonomy_entry.tags:
        if tag not in topic.taxonomy_tags:
            topic.taxonomy_tags.append(tag)
    if topic.language == "unknown" and taxonomy_entry.language != "unknown":
        topic.language = taxonomy_entry.language


def _taxonomy_match_is_strong(entry: TaxonomyEntry) -> bool:
    label = entry.label.strip()
    if re.search(r"[\u4e00-\u9fff]", label):
        return True
    return len(label.split()) >= 2


def _get_phase(
    phases_by_key: dict[tuple[str, str], ChronologyPhase],
    topic: ChronologyTopic,
    phase_type: str,
    event: MemoryEvent,
    span: EvidenceSpan | None,
    created_at: datetime,
) -> ChronologyPhase:
    key = (topic.topic_id, phase_type)
    phase = phases_by_key.get(key)
    if phase is not None:
        if span is not None and span.span_id not in phase.source_span_ids:
            phase.source_span_ids.append(span.span_id)
        return phase

    source_span_ids = [span.span_id] if span is not None else list(event.source_span_ids)
    phase = ChronologyPhase(
        phase_id=_id("chron_phase", topic.topic_id, phase_type),
        topic_id=topic.topic_id,
        phase_type=phase_type,
        order_hint=PHASE_ORDER_HINTS[phase_type],
        source_span_ids=source_span_ids,
        confidence=event.confidence,
        created_at=created_at,
    )
    phases_by_key[key] = phase
    return phase


def _get_phase_for_span(
    phases_by_key: dict[tuple[str, str], ChronologyPhase],
    topic: ChronologyTopic,
    phase_type: str,
    span: EvidenceSpan,
    created_at: datetime,
) -> ChronologyPhase:
    key = (topic.topic_id, phase_type)
    phase = phases_by_key.get(key)
    if phase is not None:
        if span.span_id not in phase.source_span_ids:
            phase.source_span_ids.append(span.span_id)
        return phase

    phase = ChronologyPhase(
        phase_id=_id("chron_phase", topic.topic_id, phase_type),
        topic_id=topic.topic_id,
        phase_type=phase_type,
        order_hint=PHASE_ORDER_HINTS[phase_type],
        source_span_ids=[span.span_id],
        confidence=0.72,
        created_at=created_at,
    )
    phases_by_key[key] = phase
    return phase


def _infer_action(text: str) -> str:
    lowered = text.lower()
    action_patterns = (
        "set up",
        "implemented",
        "implement",
        "started",
        "start",
        "initialize",
        "test",
        "verify",
        "完成",
        "开始",
        "测试",
    )
    for pattern in action_patterns:
        if _contains_phrase(lowered, pattern):
            return pattern
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    return tokens[0] if tokens else "unknown"


def _skip_chronology_node(event: MemoryEvent, span: EvidenceSpan | None) -> bool:
    speaker = str(getattr(span, "speaker", "") or "").lower()
    participants = {str(participant).lower() for participant in getattr(event, "participants", [])}
    if speaker in {"assistant", "agent"} or participants.intersection({"assistant", "agent"}):
        return True
    return False


def _infer_aspect_label(text: str, topic_label: str) -> str:
    lowered = text.lower()
    for label, phrases in ASPECT_RULES:
        if any(_contains_phrase(lowered, phrase) for phrase in phrases):
            return label
    candidates = _aspect_candidates(text)
    if candidates:
        return candidates[0]
    return topic_label


def _aspect_candidates(text: str) -> list[str]:
    cleaned = _strip_request_shell(text)
    lowered = cleaned.lower()
    phrase_patterns = (
        r"(?:about|on|for|with|including)\s+([a-z][a-z0-9+\- ]{3,80})",
        r"(?:implement(?:ing)?|build(?:ing)?|test(?:ing)?|fix(?:ing)?|deploy(?:ing)?|configure|customize)\s+([a-z][a-z0-9+\- ]{3,80})",
    )
    out: list[str] = []
    for pattern in phrase_patterns:
        for match in re.finditer(pattern, lowered):
            label = _clean_aspect_label(match.group(1))
            if label:
                out.append(label)
    tokens = [token for token in tokenize(cleaned) if token not in STOPWORDS and len(token) > 2]
    if len(tokens) >= 2:
        out.append(" ".join(tokens[:4]))
    return list(dict.fromkeys(out))


def _strip_request_shell(text: str) -> str:
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"\b(?:can you|could you|please|help me|i want to|i need to|i'm trying to|i am trying to)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s,.;:!?]+", " ", cleaned).strip()
    return cleaned


def _clean_aspect_label(value: str) -> str:
    value = re.split(r"\b(?:and|but|because|considering|so|while|when|which|that|with a|using)\b", value, maxsplit=1)[0]
    tokens = [token for token in tokenize(value) if token not in STOPWORDS and len(token) > 1]
    if not tokens:
        return ""
    return " ".join(tokens[:5])


def _build_edges(
    nodes: list[ChronologyEventNode], created_at: datetime, recognized_phase_ids: set[str]
) -> list[ChronologyEventEdge]:
    edges: list[ChronologyEventEdge] = []
    for from_node, to_node in zip(nodes, nodes[1:]):
        if from_node.topic_id != to_node.topic_id:
            continue
        evidence_type = _edge_evidence_type(from_node, to_node, recognized_phase_ids)
        if evidence_type is None:
            continue
        source_span_ids = [
            span_id for span_id in (from_node.source_span_id, to_node.source_span_id) if span_id is not None
        ]
        edges.append(
            ChronologyEventEdge(
                edge_id=_id("chron_edge", from_node.node_id, to_node.node_id, evidence_type),
                from_node_id=from_node.node_id,
                to_node_id=to_node.node_id,
                edge_type="before",
                evidence_type=evidence_type,
                source_span_ids=source_span_ids,
                confidence=min(from_node.confidence, to_node.confidence),
                created_at=created_at,
            )
        )
    return edges


def _edge_evidence_type(
    from_node: ChronologyEventNode, to_node: ChronologyEventNode, recognized_phase_ids: set[str]
) -> str | None:
    if from_node.explicit_order_marker or to_node.explicit_order_marker:
        return "explicit_marker"
    if (
        from_node.timestamp is not None
        and to_node.timestamp is not None
        and from_node.timestamp <= to_node.timestamp
        and (from_node.phase_id in recognized_phase_ids or to_node.phase_id in recognized_phase_ids)
    ):
        return "timestamp_phase"
    return None


def _id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{stable_hash('|'.join(str(part) for part in parts))[:16]}"
