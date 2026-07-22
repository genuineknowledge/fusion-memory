from __future__ import annotations

from typing import Protocol

from fusion_memory.core.chronology import (
    ChronologyEventEdge,
    ChronologyEventNode,
    ChronologyPhase,
    ChronologyTopic,
)
from fusion_memory.core.models import (
    CurrentView,
    EntityProfile,
    EntityRecord,
    EvidenceSpan,
    MemoryEvent,
    MemoryFact,
    Scope,
)


class MemorySearchRepository(Protocol):
    def search_spans(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        speaker: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[EvidenceSpan, dict[str, float]]]: ...

    def list_spans(self, scope: Scope, *, include_session: bool = False) -> list[EvidenceSpan]: ...

    def search_facts(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        category: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[tuple[MemoryFact, dict[str, float]]]: ...

    def list_facts(self, scope: Scope, category: str | None = None, *, include_session: bool = False) -> list[MemoryFact]: ...

    def search_events(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        *,
        include_session: bool = False,
    ) -> list[tuple[MemoryEvent, dict[str, float]]]: ...

    def list_events(self, scope: Scope, *, include_session: bool = False) -> list[MemoryEvent]: ...

    def list_current_views(
        self,
        scope: Scope,
        view_type: str | None = None,
        *,
        include_session: bool = False,
    ) -> list[CurrentView]: ...

    def search_entity_profiles(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        *,
        include_session: bool = False,
    ) -> list[tuple[EntityProfile, dict[str, float]]]: ...

    def search_entities(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        *,
        include_session: bool = False,
    ) -> list[tuple[EntityRecord, dict[str, float]]]: ...

    def get_span(
        self,
        span_id: str,
        scope: Scope | None = None,
        *,
        include_session: bool = False,
    ) -> EvidenceSpan | None: ...

    def get_fact(
        self,
        fact_id: str,
        scope: Scope | None = None,
        *,
        include_session: bool = False,
    ) -> MemoryFact | None: ...

    def list_chronology_topics(self, scope: Scope, include_session: bool = False) -> list[ChronologyTopic]: ...

    def list_chronology_phases(self, topic_ids: list[str]) -> list[ChronologyPhase]: ...

    def list_chronology_event_nodes(
        self,
        scope: Scope,
        include_session: bool = False,
        topic_ids: list[str] | None = None,
    ) -> list[ChronologyEventNode]: ...

    def list_chronology_event_edges(self, node_ids: list[str]) -> list[ChronologyEventEdge]: ...
