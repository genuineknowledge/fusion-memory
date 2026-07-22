from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fusion_memory.core.models import Candidate
from fusion_memory.retrieval.context import ProductQueryPlan, ProviderFailure, ProviderKind, RetrievalContext, SearchRequest
from fusion_memory.retrieval.ports import MemorySearchRepository


class ProviderUnavailable(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ProviderContext:
    runtime: RetrievalContext
    request: SearchRequest
    plan: ProductQueryPlan
    repository: MemorySearchRepository
    provider: ProviderKind
    limit: int


@dataclass(frozen=True)
class ProviderOutcome:
    provider: ProviderKind
    candidates: tuple[Candidate, ...]
    elapsed_ms: float
    failure: ProviderFailure | None = None


class CandidateProvider(Protocol):
    def recall(self, context: ProviderContext) -> ProviderOutcome: ...
