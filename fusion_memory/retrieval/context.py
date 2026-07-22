from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from fusion_memory.core.models import Candidate, Scope


SearchMode = Literal["fast", "balanced"]


class ProviderKind(str, Enum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    CHRONOLOGY = "chronology"


class OrderingMode(str, Enum):
    RELEVANCE = "relevance"
    RECENCY = "recency"
    CHRONOLOGICAL = "chronological"


@dataclass(frozen=True)
class TimeRange:
    start: datetime | None = None
    end: datetime | None = None

    def contains(self, value: datetime) -> bool:
        return (self.start is None or value >= self.start) and (self.end is None or value <= self.end)


@dataclass(frozen=True)
class ProviderRequest:
    kind: ProviderKind
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("provider limit must be positive")


@dataclass(frozen=True)
class SearchRequest:
    query: str
    limit: int
    mode: SearchMode = "fast"
    time_range: TimeRange | None = None
    include_trace: bool = True
    enabled_providers: frozenset[ProviderKind] | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query is required")
        if self.limit < 1:
            raise ValueError("limit must be positive")
        if self.mode not in {"fast", "balanced"}:
            raise ValueError("mode must be fast or balanced")


@dataclass(frozen=True)
class RetrievalContext:
    scope: Scope
    user_id: str | None
    now: datetime
    trace_id: str
    deadline: datetime | None
    include_session: bool

    def __post_init__(self) -> None:
        if self.user_id is not None and self.scope.user_id != self.user_id:
            raise ValueError("retrieval context user_id must match scope.user_id")

    def check_deadline(self) -> None:
        if self.deadline is not None and datetime.now(timezone.utc) > self.deadline.astimezone(timezone.utc):
            raise TimeoutError("retrieval deadline exceeded")


@dataclass(frozen=True)
class ProductQueryPlan:
    intent: str
    provider_requests: tuple[ProviderRequest, ...]
    time_range: TimeRange | None
    entities: tuple[str, ...]
    speaker: str | None
    ordering: OrderingMode
    use_reranker: bool
    query_intent: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderFailure:
    provider: ProviderKind
    error_code: str
    retryable: bool


@dataclass(frozen=True)
class ProviderReport:
    provider: ProviderKind
    candidate_count: int
    elapsed_ms: float
    failure: ProviderFailure | None = None


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[Candidate, ...]
    coverage: dict[str, Any]
    trace: dict[str, Any]
    plan: ProductQueryPlan
