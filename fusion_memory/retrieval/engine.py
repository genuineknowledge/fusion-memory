from __future__ import annotations

from typing import Protocol

from fusion_memory.core.models import EvidencePack
from fusion_memory.retrieval.context import ProductQueryPlan, RetrievalContext, RetrievalResult, SearchRequest


class RetrievalUnavailable(RuntimeError):
    pass


class RetrievalEngine(Protocol):
    def search(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        plan: ProductQueryPlan | None = None,
    ) -> RetrievalResult: ...

    def build_evidence_pack(
        self,
        context: RetrievalContext,
        request: SearchRequest,
        result: RetrievalResult,
        token_budget: int,
    ) -> EvidencePack: ...
