from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


Speaker = Literal["user", "assistant", "agent", "tool", "system", "document"]
SpanType = Literal["turn", "window", "tool_result", "document_chunk", "summary"]
CandidateType = Literal["span", "fact", "event", "view", "profile"]
MemoryCandidateType = Literal["fact", "event", "relation", "current_view", "entity_profile"]
EncodingDecisionType = Literal["accept", "merge", "update_relation", "quarantine", "reject"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Scope:
    workspace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    app_id: str | None = None

    def validate_for_add(self) -> None:
        if not any([self.workspace_id, self.user_id, self.agent_id, self.run_id]):
            raise ValueError("add requires at least one of workspace_id, user_id, agent_id, or run_id")

    def validate_for_read(self) -> None:
        if not any([self.workspace_id, self.user_id, self.agent_id, self.run_id]):
            raise ValueError("read requires at least one of workspace_id, user_id, agent_id, or run_id")

    def as_filters(self, include_session: bool = True) -> dict[str, str | None]:
        data = {
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
        }
        if include_session:
            data["session_id"] = self.session_id
        return data


@dataclass
class EvidenceSpan:
    span_id: str
    scope: Scope
    turn_id: str | None
    speaker: Speaker
    span_type: SpanType
    content: str
    content_hash: str
    timestamp: datetime
    source_uri: str | None = None
    parent_span_id: str | None = None
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryFact:
    fact_id: str
    scope: Scope
    subject: str
    predicate: str
    object: str
    text: str
    category: str
    confidence: float
    salience: float
    source_span_ids: list[str]
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    polarity: str = "unknown"
    linked_fact_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class FactRelation:
    relation_id: str
    from_fact_id: str
    to_fact_id: str
    relation_type: str
    source_span_ids: list[str]
    confidence: float


@dataclass
class MemoryEvent:
    event_id: str
    scope: Scope
    event_type: str
    description: str
    participants: list[str]
    source_span_ids: list[str]
    fact_ids: list[str] = field(default_factory=list)
    time_start: datetime | None = None
    time_end: datetime | None = None
    time_granularity: str = "unknown"
    time_source: str = "unknown"
    confidence: float = 0.0


@dataclass
class EventEdge:
    edge_id: str
    from_event_id: str
    to_event_id: str
    edge_type: str
    source_span_ids: list[str]
    confidence: float


@dataclass
class CurrentView:
    view_id: str
    scope: Scope
    view_type: str
    subject: str
    text: str
    state_json: dict[str, Any]
    source_fact_ids: list[str]
    source_event_ids: list[str]
    source_span_ids: list[str]
    confidence: float
    updated_at: datetime = field(default_factory=utcnow)


@dataclass
class EntityProfile:
    profile_id: str
    scope: Scope
    entity_id: str
    entity_type: str
    profile_type: str
    text: str
    state_json: dict[str, Any]
    source_fact_ids: list[str]
    source_event_ids: list[str]
    source_span_ids: list[str]
    confidence: float
    support_count: int
    last_observed_at: datetime | None = None
    updated_at: datetime = field(default_factory=utcnow)


@dataclass
class EntityRecord:
    entity_id: str
    scope: Scope
    name: str
    entity_type: str
    aliases: list[str]
    source_span_ids: list[str]
    observed_count: int
    last_observed_at: datetime | None = None


@dataclass
class ExtractedCandidate:
    local_id: str
    candidate_type: MemoryCandidateType
    text: str
    structured: dict[str, Any]
    confidence: float
    source_span_ids: list[str]
    extractor_name: str
    prompt_version: str = "rules-v0"


@dataclass
class EncodingDecision:
    decision_id: str
    candidate_type: MemoryCandidateType
    candidate: ExtractedCandidate
    decision: EncodingDecisionType
    reason_codes: list[str]
    scores: dict[str, float]
    matched_existing_ids: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    id: str
    type: CandidateType
    text: str
    source: str
    scores: dict[str, float]
    source_span_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    query: str
    query_type: str
    entities: list[str]
    time_constraints: list[dict[str, Any]]
    speaker_focus: str = "any"
    needs_current_state: bool = False
    needs_source_evidence: bool = True
    must_include_sources: list[str] = field(default_factory=list)


@dataclass
class AddResult:
    span_ids: list[str]
    accepted_fact_ids: list[str]
    accepted_event_ids: list[str]
    updated_view_ids: list[str]
    updated_profile_ids: list[str]
    quarantined_candidate_ids: list[str]
    trace_id: str


@dataclass
class SearchResult:
    candidates: list[Candidate]
    trace_id: str
    coverage: dict[str, Any]


@dataclass
class EvidencePack:
    query: str
    answer_policy: str
    current_views: list[dict[str, Any]]
    entity_profiles: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    events: list[dict[str, Any]]
    source_spans: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    coverage: dict[str, Any]
    debug_trace: list[dict[str, Any]]
