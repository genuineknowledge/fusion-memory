from __future__ import annotations

from collections import Counter

from fusion_memory.core.models import CurrentView, EntityProfile, MemoryFact, Scope
from fusion_memory.core.models import new_id
from fusion_memory.core.text import stable_hash, tokenize


def deterministic_id(prefix: str, *parts: str | None) -> str:
    return f"{prefix}_{stable_hash('|'.join(p or '' for p in parts))[:24]}"


class ViewBuilder:
    def build_current_views(self, scope: Scope, facts: list[MemoryFact], superseded_fact_ids: set[str]) -> list[CurrentView]:
        active = [fact for fact in facts if fact.fact_id not in superseded_fact_ids]
        views: list[CurrentView] = []
        for category, view_type in [
            ("preference", "current_preferences"),
            ("instruction", "standing_instructions"),
            ("project_state", "active_projects"),
            ("commitment", "open_commitments"),
            ("agent_action", "recent_agent_actions"),
        ]:
            category_facts = [fact for fact in active if fact.category == category]
            if not category_facts:
                continue
            latest = sorted(category_facts, key=lambda f: f.created_at)[-1]
            views.append(
                CurrentView(
                    view_id=deterministic_id(
                        "view",
                        scope.workspace_id,
                        scope.user_id,
                        scope.agent_id,
                        scope.run_id,
                        scope.session_id,
                        scope.app_id,
                        view_type,
                        latest.subject,
                    ),
                    scope=scope,
                    view_type=view_type,
                    subject=latest.subject,
                    text=latest.text,
                    state_json={"category": category, "object": latest.object},
                    source_fact_ids=[latest.fact_id],
                    source_event_ids=[],
                    source_span_ids=latest.source_span_ids,
                    confidence=latest.confidence,
                )
            )
        return views

    def build_entity_profiles(self, scope: Scope, facts: list[MemoryFact]) -> list[EntityProfile]:
        profile_facts = [
            fact
            for fact in facts
            if fact.category in {"preference", "instruction", "profile"}
            and fact.subject == "user"
            and fact.confidence >= 0.65
        ]
        if len(profile_facts) < 2:
            return []
        tokens = Counter()
        for fact in profile_facts:
            tokens.update(t for t in tokenize(fact.text) if len(t) > 3)
        style_tokens = [t for t, _ in tokens.most_common(8)]
        if not style_tokens:
            return []
        text = "User long-term profile: " + ", ".join(style_tokens)
        source_fact_ids = [fact.fact_id for fact in profile_facts]
        source_span_ids: list[str] = []
        for fact in profile_facts:
            source_span_ids.extend(fact.source_span_ids)
        return [
            EntityProfile(
                profile_id=deterministic_id(
                    "profile",
                    scope.workspace_id,
                    scope.user_id,
                    scope.agent_id,
                    scope.run_id,
                    scope.session_id,
                    scope.app_id,
                    "user",
                    "communication_style",
                ),
                scope=scope,
                entity_id=scope.user_id or "user",
                entity_type="user",
                profile_type="communication_style",
                text=text,
                state_json={"keywords": style_tokens},
                source_fact_ids=source_fact_ids,
                source_event_ids=[],
                source_span_ids=list(dict.fromkeys(source_span_ids)),
                confidence=min(0.90, 0.55 + 0.08 * len(profile_facts)),
                support_count=len(profile_facts),
                last_observed_at=max((fact.created_at for fact in profile_facts), default=None),
            )
        ]
