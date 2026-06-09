from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fusion_memory.core.auth import AllowAllAuthorizer, Authorizer
from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.embedding import Embedder
from fusion_memory.core.models import (
    AddResult,
    Candidate,
    CurrentView,
    EntityProfile,
    EventEdge,
    EvidenceSpan,
    EvidencePack,
    FactRelation,
    MemoryEvent,
    MemoryFact,
    Scope,
    SearchResult,
    new_id,
)
from fusion_memory.core.text import compact_summary, extract_entities, keyword_score, stable_hash
from fusion_memory.ingestion.encoding_gate import EncodingGate
from fusion_memory.ingestion.extractors import RuleBasedExtractor
from fusion_memory.ingestion.normalizer import normalize_input
from fusion_memory.ingestion.views import ViewBuilder
from fusion_memory.ingestion.window_builder import build_session_summary_span
from fusion_memory.retrieval.evidence_pack import EvidencePackBuilder
from fusion_memory.retrieval.mmr import mmr
from fusion_memory.retrieval.query_planner import QueryPlanner
from fusion_memory.retrieval.raw_evidence_quota import RawEvidenceQuota
from fusion_memory.retrieval.reranker import LexicalCrossEncoderReranker, Reranker, rerank_candidates
from fusion_memory.retrieval.rrf import reciprocal_rank_fusion
from fusion_memory.retrieval.scoring import score_candidate
from fusion_memory.retrieval.utility_model import LogisticUtilityScorer, UtilityTrainingReport
from fusion_memory.retrieval.utility_scorer import utility_example
from fusion_memory.storage.postgres_store import PostgresMemoryStore
from fusion_memory.storage.sqlite_store import SQLiteMemoryStore, dt_from_str


class MemoryService:
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        extractor: Any | None = None,
        reranker: Reranker | None = None,
        embedder: Embedder | None = None,
        config: MemoryConfig | None = None,
        authorizer: Authorizer | None = None,
        storage_backend: str = "sqlite",
        store: Any | None = None,
        store_connect: Any | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        if store is not None:
            self.store = store
        elif storage_backend == "sqlite":
            self.store = SQLiteMemoryStore(db_path, embedder=embedder)
        elif storage_backend == "postgres":
            self.store = PostgresMemoryStore(str(db_path), embedder=embedder, connect=store_connect)
        else:
            raise ValueError(f"unsupported storage_backend: {storage_backend}")
        self.authorizer = authorizer or AllowAllAuthorizer()
        self.extractor = extractor or RuleBasedExtractor()
        self.gate = EncodingGate(self.config)
        self.views = ViewBuilder()
        self.planner = QueryPlanner()
        self.quota = RawEvidenceQuota(self.store, self.config)
        self.pack_builder = EvidencePackBuilder(self.store, self.config)
        self.utility_scorer = LogisticUtilityScorer()
        self.reranker = reranker or LexicalCrossEncoderReranker()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "MemoryService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def add(self, input: Any, scope: Scope, session_time: datetime | None = None, metadata: dict[str, Any] | None = None) -> AddResult:
        scope.validate_for_add()
        self._authorize("memory.add", scope, {"metadata": metadata or {}})
        model_call_marks = self._model_call_marks()
        session_time = session_time or datetime.now(timezone.utc)
        trace_id = new_id("trace")
        trace: dict[str, Any] = {"operation": "add", "config": self.config.snapshot(), "steps": []}
        spans = normalize_input(input, scope, session_time, metadata, config=self.config)
        inserted_span_ids: list[str] = []
        for span in spans:
            duplicate = self.store.find_duplicate_span(span.content_hash, scope)
            if duplicate:
                inserted_span_ids.append(duplicate.span_id)
                trace["steps"].append({"step": "span_duplicate", "span_id": duplicate.span_id})
                continue
            self.store.insert_span(span)
            self._upsert_span_entities(span)
            inserted_span_ids.append(span.span_id)
        trace["steps"].append({"step": "l0_written", "span_ids": inserted_span_ids})

        existing_facts = self.store.list_facts(scope)
        extraction_spans = [span for span in spans if span.span_type not in {"window", "summary"}]
        candidates = self.extractor.extract(extraction_spans, existing_facts, session_time)
        decisions = self.gate.decide(candidates, existing_facts)
        accepted_fact_ids: list[str] = []
        accepted_event_ids: list[str] = []
        quarantined_candidate_ids: list[str] = []
        local_to_fact: dict[str, str] = {}
        local_to_event: dict[str, str] = {}

        for decision in decisions:
            self.store.insert_encoding_decision(scope, decision)
            candidate = decision.candidate
            if decision.decision == "quarantine":
                quarantined_candidate_ids.append(candidate.local_id)
                continue
            if decision.decision == "accept" and candidate.candidate_type == "fact":
                fact = self._candidate_to_fact(scope, candidate, session_time)
                self.store.insert_fact(fact)
                self._upsert_fact_entities(fact)
                accepted_fact_ids.append(fact.fact_id)
                local_to_fact[candidate.local_id] = fact.fact_id
            elif decision.decision == "accept" and candidate.candidate_type == "event":
                event = self._candidate_to_event(scope, candidate)
                self.store.insert_event(event)
                self._upsert_event_entities(event)
                accepted_event_ids.append(event.event_id)
                local_to_event[candidate.local_id] = event.event_id
            elif decision.decision == "update_relation" and candidate.candidate_type == "relation":
                relation = self._candidate_to_relation(candidate, local_to_fact)
                if relation:
                    self.store.insert_fact_relation(relation)

        self._create_session_event_edges(scope)
        self._create_explicit_event_edges(scope, accepted_event_ids)
        updated_views, updated_profiles = self._refresh_views_and_profiles(scope)
        summary_task = self._maybe_enqueue_session_summary_task(scope)
        trace["steps"].append(
            {
                "step": "encoding",
                "decisions": [
                    {
                        "candidate_id": decision.candidate.local_id,
                        "type": decision.candidate_type,
                        "extractor": decision.candidate.extractor_name,
                        "prompt_version": decision.candidate.prompt_version,
                        "decision": decision.decision,
                        "reasons": decision.reason_codes,
                    }
                    for decision in decisions
                ],
            }
        )
        trace["steps"].append(
            {
                "step": "derived_written",
                "facts": accepted_fact_ids,
                "events": accepted_event_ids,
                "views": [view.view_id for view in updated_views],
                "profiles": [profile.profile_id for profile in updated_profiles],
                "background_task_ids": [summary_task["task_id"]] if summary_task else [],
            }
        )
        model_calls = self._model_calls_since(model_call_marks)
        trace["model_calls"] = model_calls
        self.store.save_trace(trace_id, trace, scope)
        self.store.insert_audit_event(
            scope,
            "memory.add",
            object_type="trace",
            object_id=trace_id,
            trace_id=trace_id,
            payload={
                "span_count": len(inserted_span_ids),
                "accepted_fact_count": len(accepted_fact_ids),
                "accepted_event_count": len(accepted_event_ids),
                "quarantined_candidate_count": len(quarantined_candidate_ids),
                "background_task_id": summary_task["task_id"] if summary_task else None,
                "model_calls": _model_call_summary(model_calls),
            },
        )
        return AddResult(
            span_ids=inserted_span_ids,
            accepted_fact_ids=accepted_fact_ids,
            accepted_event_ids=accepted_event_ids,
            updated_view_ids=[view.view_id for view in updated_views],
            updated_profile_ids=[profile.profile_id for profile in updated_profiles],
            quarantined_candidate_ids=quarantined_candidate_ids,
            trace_id=trace_id,
        )

    def search(self, query: str, scope: Scope, options: dict[str, Any] | None = None) -> SearchResult:
        options = options or {}
        scope.validate_for_read()
        allow_cross_session = bool(options.get("allow_cross_session", False))
        include_session = bool(scope.session_id and not allow_cross_session)
        self._authorize(
            "memory.search",
            scope,
            {
                "query": query,
                "allow_cross_session": allow_cross_session,
                "include_session": include_session,
                "mode": options.get("mode", "fast"),
                "limit": options.get("limit", self.config.retrieval_output_n),
                "enabled_sources": options.get("enabled_sources"),
            },
        )
        model_call_marks = self._model_call_marks()
        trace_id = new_id("trace")
        plan = self.planner.plan(query)
        candidate_lists = self._candidate_lists(
            query,
            scope,
            plan,
            per_source_limit=options.get("per_source_limit", self.config.retrieval_top_k_per_source),
            enabled_sources=options.get("enabled_sources"),
            include_session=include_session,
        )
        fused = reciprocal_rank_fusion(candidate_lists, k=self.config.rrf_k)
        scored = [score_candidate(candidate, plan) for candidate in fused]
        quota_result = self.quota.enforce(plan, scope, scored, include_session=include_session)
        marked = self._mark_quota_selected(quota_result.candidates, quota_result.selected_span_ids)
        scored_again = [score_candidate(candidate, plan) for candidate in marked]
        scored_again.sort(key=lambda candidate: candidate.scores.get("utility_score", 0.0), reverse=True)
        mode = options.get("mode", "fast")
        limit = options.get("limit", self.config.retrieval_output_n)
        rerank_top_n = options.get("rerank_top_n") or (
            self.config.balanced_mode_rerank_top_n
            if mode == "balanced"
            else self.config.benchmark_mode_rerank_top_n
            if mode == "benchmark"
            else limit
        )
        preselected = mmr(scored_again, limit=rerank_top_n, lambda_=self.config.mmr_lambda)
        rerank_applied = mode in {"balanced", "benchmark"}
        if rerank_applied:
            reranked = rerank_candidates(query, preselected, self.reranker)
            selected = self._preserve_quota_after_rerank(reranked, quota_result.selected_span_ids, limit)
        else:
            selected = preselected[:limit]
        for candidate in selected:
            self.store.insert_utility_example(utility_example(trace_id, query, plan, candidate))
        shadow_ranking = self.utility_scorer.rank_shadow(selected, plan) if self.utility_scorer.trained else []
        coverage = {
            "query_type": plan.query_type,
            "source_span_quota_required": quota_result.required,
            "source_span_quota_selected": len(quota_result.selected_span_ids),
            "selected_span_ids": quota_result.selected_span_ids,
            "source_span_quota_met": not quota_result.coverage_insufficient,
            "coverage_insufficient": quota_result.coverage_insufficient,
            "raw_quota_backfilled": quota_result.backfilled,
        }
        trace = {
            "operation": "search",
            "query": query,
            "plan": plan.__dict__,
            "config": self.config.snapshot(),
            "candidate_counts": [len(items) for items in candidate_lists],
            "coverage": coverage,
            "mode": mode,
            "enabled_sources": options.get("enabled_sources"),
            "allow_cross_session": allow_cross_session,
            "include_session": include_session,
            "rerank": {
                "applied": rerank_applied,
                "model_version": getattr(self.reranker, "version", "custom"),
                "top_n": rerank_top_n,
            },
            "selected": [
                {
                    "id": candidate.id,
                    "type": candidate.type,
                    "source": candidate.source,
                    "scores": candidate.scores,
                    "source_span_ids": candidate.source_span_ids,
                }
                for candidate in selected
            ],
            "utility_shadow": {
                "enabled": self.utility_scorer.trained,
                "model_version": self.utility_scorer.version,
                "ranking": shadow_ranking,
            },
        }
        model_calls = self._model_calls_since(model_call_marks)
        trace["model_calls"] = model_calls
        self.store.save_trace(trace_id, trace, scope)
        self.store.insert_audit_event(
            scope,
            "memory.search",
            object_type="trace",
            object_id=trace_id,
            trace_id=trace_id,
            payload={
                "query": query,
                "query_type": plan.query_type,
                "mode": mode,
                "candidate_count": len(selected),
                "coverage_insufficient": quota_result.coverage_insufficient,
                "allow_cross_session": allow_cross_session,
                "model_calls": _model_call_summary(model_calls),
            },
        )
        return SearchResult(candidates=selected, trace_id=trace_id, coverage=coverage)

    def answer_context(self, query: str, scope: Scope, budget: dict[str, Any] | None = None) -> EvidencePack:
        budget = budget or {}
        scope.validate_for_read()
        self._authorize(
            "memory.answer_context",
            scope,
            {
                "query": query,
                "allow_cross_session": bool(budget.get("allow_cross_session", False)),
                "limit": budget.get("limit", self.config.retrieval_output_n),
                "mode": budget.get("mode", "fast"),
                "token_budget": budget.get("token_budget", self.config.answer_context_budget_tokens),
            },
        )
        result = self.search(
            query,
            scope,
            options={
                "limit": budget.get("limit", self.config.retrieval_output_n),
                "mode": budget.get("mode", "fast"),
                "rerank_top_n": budget.get("rerank_top_n"),
                "enabled_sources": budget.get("enabled_sources"),
                "allow_cross_session": budget.get("allow_cross_session", False),
            },
        )
        plan = self.planner.plan(query)
        trace = self.store.get_trace(result.trace_id, scope, include_session=bool(scope.session_id and not budget.get("allow_cross_session", False))) or {}
        return self.pack_builder.build(
            query,
            plan,
            result.candidates,
            result.coverage,
            trace.get("selected", []),
            token_budget=budget.get("token_budget", self.config.answer_context_budget_tokens),
        )

    def get(
        self,
        object_id: str,
        object_type: str | None = None,
        scope: Scope | None = None,
        allow_cross_session: bool = False,
    ) -> Any:
        include_session = False
        if scope:
            scope.validate_for_read()
            include_session = bool(scope.session_id and not allow_cross_session)
            self._authorize(
                "memory.get",
                scope,
                {"object_id": object_id, "object_type": object_type, "allow_cross_session": allow_cross_session, "include_session": include_session},
            )
        if object_type in {None, "span"}:
            span = self.store.get_span(object_id, scope, include_session=include_session)
            if span:
                return span
        if object_type in {None, "fact"}:
            fact = self.store.get_fact(object_id, scope, include_session=include_session)
            if fact:
                return fact
        if object_type in {None, "event"}:
            event = self.store.get_event(object_id, scope, include_session=include_session)
            if event:
                return event
        return None

    def history(
        self,
        scope: Scope,
        entity: str | None = None,
        fact_id: str | None = None,
        session_id: str | None = None,
        allow_cross_session: bool = False,
    ) -> dict[str, Any]:
        scope.validate_for_read()
        self._authorize(
            "memory.history",
            scope,
            {
                "entity": entity,
                "fact_id": fact_id,
                "session_id": session_id or scope.session_id,
                "allow_cross_session": allow_cross_session,
            },
        )
        effective_scope = Scope(
            workspace_id=scope.workspace_id,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            session_id=session_id or scope.session_id,
            app_id=scope.app_id,
        )
        include_session = bool(effective_scope.session_id and not allow_cross_session)
        facts = self.store.list_facts(effective_scope, include_session=include_session)
        if entity:
            facts = [fact for fact in facts if entity.lower() in (fact.text + " " + fact.object).lower()]
        relations = self.store.list_fact_relations(fact_id) if fact_id else self.store.list_fact_relations()
        if not fact_id:
            visible_fact_ids = {fact.fact_id for fact in facts}
            relations = [
                relation
                for relation in relations
                if relation.from_fact_id in visible_fact_ids or relation.to_fact_id in visible_fact_ids
            ]
        return {
            "facts": [fact.__dict__ for fact in facts],
            "relations": [relation.__dict__ for relation in relations],
            "events": [event.__dict__ for event in self.store.list_events(effective_scope, include_session=include_session)],
        }

    def debug_trace(self, trace_id: str, scope: Scope | None = None, allow_cross_session: bool = False) -> dict[str, Any] | None:
        include_session = False
        if scope:
            scope.validate_for_read()
            include_session = bool(scope.session_id and not allow_cross_session)
            self._authorize(
                "memory.debug_trace",
                scope,
                {"trace_id": trace_id, "allow_cross_session": allow_cross_session, "include_session": include_session},
            )
        return self.store.get_trace(trace_id, scope, include_session=include_session)

    def audit_events(self, scope: Scope, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        scope.validate_for_read()
        self._authorize("memory.audit", scope, {"event_type": event_type, "limit": limit})
        return self.store.list_audit_events(scope, event_type=event_type, limit=limit)

    def refresh_session_summary(
        self,
        scope: Scope,
        session_id: str | None = None,
        max_source_spans: int | None = None,
    ) -> EvidenceSpan | None:
        scope.validate_for_read()
        effective_scope = self._session_scope(scope, session_id)
        if not effective_scope.session_id:
            raise ValueError("refresh_session_summary requires session_id or scope.session_id")
        self._authorize(
            "memory.summary.refresh",
            effective_scope,
            {"session_id": effective_scope.session_id, "max_source_spans": max_source_spans or self.config.session_summary_max_source_spans},
        )
        source_spans = [
            span
            for span in self.store.list_spans(effective_scope, include_session=True)
            if span.span_type != "summary"
        ]
        summary = build_session_summary_span(
            source_spans,
            effective_scope,
            min_source_spans=self.config.session_summary_min_spans,
            max_source_spans=max_source_spans or self.config.session_summary_max_source_spans,
            max_chars=self.config.session_summary_max_chars,
        )
        if not summary:
            return None
        duplicate = self.store.find_duplicate_span(summary.content_hash, effective_scope)
        if duplicate and duplicate.span_type == "summary":
            return duplicate
        self.store.insert_span(summary)
        trace_id = new_id("trace")
        trace = {
            "operation": "refresh_session_summary",
            "session_id": effective_scope.session_id,
            "summary_span_id": summary.span_id,
            "source_span_ids": summary.metadata.get("parent_span_ids", []),
            "config": self.config.snapshot(),
        }
        self.store.save_trace(trace_id, trace, effective_scope)
        self.store.insert_audit_event(
            effective_scope,
            "memory.summary.refresh",
            object_type="span",
            object_id=summary.span_id,
            trace_id=trace_id,
            payload={
                "session_id": effective_scope.session_id,
                "source_span_count": summary.metadata.get("source_span_count", 0),
                "summary_version": summary.metadata.get("summary_version"),
            },
        )
        return summary

    def get_session_summaries(self, scope: Scope, session_id: str | None = None) -> list[EvidenceSpan]:
        scope.validate_for_read()
        effective_scope = self._session_scope(scope, session_id)
        if not effective_scope.session_id:
            raise ValueError("get_session_summaries requires session_id or scope.session_id")
        self._authorize("memory.summary.read", effective_scope, {"session_id": effective_scope.session_id})
        summaries = [
            span
            for span in self.store.list_spans(effective_scope, include_session=True)
            if span.span_type == "summary"
        ]
        summaries.sort(key=lambda span: span.timestamp, reverse=True)
        return summaries

    def list_background_tasks(
        self,
        scope: Scope,
        *,
        status: str | None = None,
        limit: int = 100,
        allow_cross_session: bool = False,
    ) -> list[dict[str, Any]]:
        scope.validate_for_read()
        include_session = bool(scope.session_id and not allow_cross_session)
        self._authorize(
            "memory.tasks.read",
            scope,
            {"status": status, "limit": limit, "allow_cross_session": allow_cross_session, "include_session": include_session},
        )
        return self.store.list_background_tasks(scope, status=status, limit=limit, include_session=include_session)

    def process_background_tasks(
        self,
        scope: Scope,
        *,
        limit: int = 10,
        allow_cross_session: bool = False,
    ) -> dict[str, Any]:
        scope.validate_for_read()
        include_session = bool(scope.session_id and not allow_cross_session)
        self._authorize(
            "memory.tasks.process",
            scope,
            {"limit": limit, "allow_cross_session": allow_cross_session, "include_session": include_session},
        )
        tasks = self.store.next_background_tasks(limit=limit, scope=scope, include_session=include_session)
        processed: list[dict[str, Any]] = []
        for task in tasks:
            self.store.update_background_task(task["task_id"], status="running")
            try:
                if task["task_type"] == "refresh_session_summary":
                    updated = self._process_refresh_session_summary_task(task)
                else:
                    updated = self.store.update_background_task(
                        task["task_id"],
                        status="skipped",
                        result={"reason": "unknown_task_type", "task_type": task["task_type"]},
                    )
                if updated:
                    processed.append(updated)
            except Exception as exc:
                failed = self.store.update_background_task(task["task_id"], status="failed", error=str(exc))
                if failed:
                    processed.append(failed)
        counts: dict[str, int] = {}
        for task in processed:
            counts[task["status"]] = counts.get(task["status"], 0) + 1
        self.store.insert_audit_event(
            scope,
            "memory.tasks.process",
            object_type="background_task",
            payload={
                "limit": limit,
                "processed_count": len(processed),
                "status_counts": counts,
                "task_ids": [task["task_id"] for task in processed],
            },
        )
        return {"processed_count": len(processed), "status_counts": counts, "tasks": processed}

    def encoding_report(self, scope: Scope, labels: dict[str, bool] | None = None) -> dict[str, Any]:
        scope.validate_for_read()
        self._authorize("memory.report.encoding", scope, {"has_labels": bool(labels)})
        decisions = self.store.list_encoding_decisions(scope)
        by_decision: dict[str, int] = {}
        accepted = [item for item in decisions if item["decision"] == "accept"]
        rejected = [item for item in decisions if item["decision"] == "reject"]
        for decision in decisions:
            by_decision[decision["decision"]] = by_decision.get(decision["decision"], 0) + 1
        report: dict[str, Any] = {
            "total": len(decisions),
            "by_decision": by_decision,
            "accept_source_coverage": _source_coverage(accepted),
            "reject_count": len(rejected),
        }
        if labels:
            report["accept_precision"] = _labeled_precision(accepted, labels, positive=True)
            report["reject_precision"] = _labeled_precision(rejected, labels, positive=False)
        return report

    def profile_report(self, scope: Scope, labels: dict[str, bool] | None = None) -> dict[str, Any]:
        scope.validate_for_read()
        self._authorize("memory.report.profiles", scope, {"has_labels": bool(labels)})
        profiles = self.store.list_entity_profiles(scope)
        report: dict[str, Any] = {
            "total": len(profiles),
            "source_coverage": _source_coverage([profile.__dict__ for profile in profiles]),
            "avg_support_count": sum(profile.support_count for profile in profiles) / len(profiles) if profiles else 0.0,
        }
        if labels:
            labeled = [
                {"decision_id": profile.profile_id, "candidate": {"local_id": profile.profile_id}, "label": labels.get(profile.profile_id)}
                for profile in profiles
            ]
            true_profiles = sum(1 for item in labeled if item["label"] is True)
            known = sum(1 for item in labeled if item["label"] is not None)
            report["profile_precision"] = true_profiles / known if known else None
        return report

    def get_current_views(self, scope: Scope, view_type: str | None = None, allow_cross_session: bool = False) -> list[CurrentView]:
        scope.validate_for_read()
        include_session = bool(scope.session_id and not allow_cross_session)
        self._authorize(
            "memory.views.read",
            scope,
            {"view_type": view_type, "allow_cross_session": allow_cross_session, "include_session": include_session},
        )
        return self.store.list_current_views(scope, view_type=view_type, include_session=include_session)

    def refresh_current_views(self, scope: Scope, affected_fact_ids: list[str] | None = None) -> list[CurrentView]:
        scope.validate_for_read()
        self._authorize("memory.views.refresh", scope, {"affected_fact_ids": affected_fact_ids or []})
        updated_views, _ = self._refresh_views_and_profiles(scope)
        if affected_fact_ids is None:
            return updated_views
        affected = set(affected_fact_ids)
        return [view for view in updated_views if affected.intersection(view.source_fact_ids)]

    def get_entity_profile(
        self,
        entity_id: str,
        scope: Scope,
        profile_type: str | None = None,
        allow_cross_session: bool = False,
    ) -> list[EntityProfile]:
        scope.validate_for_read()
        include_session = bool(scope.session_id and not allow_cross_session)
        self._authorize(
            "memory.profiles.read",
            scope,
            {
                "entity_id": entity_id,
                "profile_type": profile_type,
                "allow_cross_session": allow_cross_session,
                "include_session": include_session,
            },
        )
        profiles = self.store.list_entity_profiles(scope, entity_id=entity_id, include_session=include_session)
        if profile_type:
            profiles = [profile for profile in profiles if profile.profile_type == profile_type]
        return profiles

    def refresh_entity_profiles(self, scope: Scope, affected_entity_ids: list[str] | None = None) -> list[EntityProfile]:
        scope.validate_for_read()
        self._authorize("memory.profiles.refresh", scope, {"affected_entity_ids": affected_entity_ids or []})
        _, updated_profiles = self._refresh_views_and_profiles(scope)
        if affected_entity_ids is None:
            return updated_profiles
        affected = {entity_id.lower() for entity_id in affected_entity_ids}
        return [profile for profile in updated_profiles if profile.entity_id.lower() in affected]

    def timeline(
        self,
        entity: str | None,
        scope: Scope,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        allow_cross_session: bool = False,
    ) -> list[MemoryEvent]:
        scope.validate_for_read()
        self._authorize(
            "memory.timeline",
            scope,
            {"entity": entity, "start": str(start) if start else None, "end": str(end) if end else None, "allow_cross_session": allow_cross_session},
        )
        start_at = self._coerce_datetime(start)
        end_at = self._coerce_datetime(end)
        entity_text = (entity or "").lower()
        include_session = bool(scope.session_id and not allow_cross_session)
        events = self.store.list_events(scope, include_session=include_session)
        filtered: list[MemoryEvent] = []
        for event in events:
            if entity_text:
                haystack = " ".join([event.description, *event.participants]).lower()
                if entity_text not in haystack:
                    continue
            event_start = event.time_start or event.time_end
            event_end = event.time_end or event.time_start
            if start_at and (event_end is None or event_end < start_at):
                continue
            if end_at and (event_start is None or event_start > end_at):
                continue
            filtered.append(event)
        filtered.sort(key=lambda event: event.time_start or event.time_end or datetime.max.replace(tzinfo=timezone.utc))
        return filtered

    def compare_events(
        self,
        event_a: str | MemoryEvent | dict[str, Any],
        event_b: str | MemoryEvent | dict[str, Any],
        scope: Scope | None = None,
        allow_cross_session: bool = False,
    ) -> dict[str, Any]:
        include_session = False
        if scope:
            scope.validate_for_read()
            include_session = bool(scope.session_id and not allow_cross_session)
            self._authorize(
                "memory.events.compare",
                scope,
                {
                    "event_a": self._event_id(event_a),
                    "event_b": self._event_id(event_b),
                    "allow_cross_session": allow_cross_session,
                    "include_session": include_session,
                },
            )
        left = self._resolve_event(event_a, scope=scope, include_session=include_session)
        right = self._resolve_event(event_b, scope=scope, include_session=include_session)
        left_id = self._event_id(event_a)
        right_id = self._event_id(event_b)
        if not left or not right:
            return {
                "event_a": left_id,
                "event_b": right_id,
                "relation": "unknown",
                "basis": "missing_event",
                "confidence": 0.0,
            }

        direct = self._event_edge(left.event_id, right.event_id)
        if direct:
            return {
                "event_a": left.event_id,
                "event_b": right.event_id,
                "relation": direct["edge_type"],
                "basis": "event_edge",
                "confidence": direct["confidence"],
                "source_span_ids": direct["source_span_ids"],
            }
        reverse = self._event_edge(right.event_id, left.event_id)
        if reverse and reverse["edge_type"] == "before":
            return {
                "event_a": left.event_id,
                "event_b": right.event_id,
                "relation": "after",
                "basis": "event_edge",
                "confidence": reverse["confidence"],
                "source_span_ids": reverse["source_span_ids"],
            }
        if not left.time_start or not right.time_start:
            return {
                "event_a": left.event_id,
                "event_b": right.event_id,
                "relation": "unknown",
                "basis": "insufficient_time",
                "confidence": 0.0,
            }
        if left.time_start < right.time_start:
            relation = "before"
        elif left.time_start > right.time_start:
            relation = "after"
        else:
            relation = "same_time"
        return {
            "event_a": left.event_id,
            "event_b": right.event_id,
            "relation": relation,
            "basis": "time_start",
            "confidence": min(left.confidence, right.confidence),
            "source_span_ids": list(dict.fromkeys(left.source_span_ids + right.source_span_ids)),
        }

    def train_utility_scorer(self) -> UtilityTrainingReport:
        report = self.utility_scorer.fit(self.store.list_utility_examples())
        return report

    def save_utility_scorer(self, path: str | Path) -> None:
        self.utility_scorer.save(path)

    def load_utility_scorer(self, path: str | Path) -> None:
        self.utility_scorer = LogisticUtilityScorer.load(path)

    def _authorize(self, operation: str, scope: Scope, context: dict[str, Any] | None = None) -> None:
        self.authorizer.authorize(operation, scope, context or {})

    def _session_scope(self, scope: Scope, session_id: str | None = None) -> Scope:
        return Scope(
            workspace_id=scope.workspace_id,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            session_id=session_id or scope.session_id,
            app_id=scope.app_id,
        )

    def _maybe_enqueue_session_summary_task(self, scope: Scope) -> dict[str, Any] | None:
        if not self.config.auto_session_summary_tasks or not scope.session_id:
            return None
        source_spans, source_hash = self._session_summary_sources_and_hash(scope)
        if len(source_spans) < self.config.session_summary_min_spans:
            return None
        dedupe_key = "refresh_session_summary:" + stable_hash(
            "|".join(
                [
                    scope.workspace_id or "",
                    scope.user_id or "",
                    scope.agent_id or "",
                    scope.run_id or "",
                    scope.session_id or "",
                    scope.app_id or "",
                    source_hash,
                ]
            )
        )
        return self.store.enqueue_background_task(
            scope,
            "refresh_session_summary",
            payload={
                "session_id": scope.session_id,
                "source_span_ids": [span.span_id for span in source_spans],
                "source_hash": source_hash,
                "source_span_count": len(source_spans),
            },
            dedupe_key=dedupe_key,
        )

    def _process_refresh_session_summary_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        task_scope = Scope(**task["scope"])
        source_spans, current_hash = self._session_summary_sources_and_hash(task_scope)
        payload = task["payload"]
        if len(source_spans) < self.config.session_summary_min_spans:
            return self.store.update_background_task(
                task["task_id"],
                status="skipped",
                result={"reason": "insufficient_source_spans", "source_span_count": len(source_spans)},
            )
        if payload.get("source_hash") and payload["source_hash"] != current_hash:
            return self.store.update_background_task(
                task["task_id"],
                status="skipped",
                result={"reason": "stale_source_hash", "current_source_hash": current_hash},
            )
        summary = self.refresh_session_summary(task_scope)
        if not summary:
            return self.store.update_background_task(task["task_id"], status="skipped", result={"reason": "summary_not_created"})
        return self.store.update_background_task(
            task["task_id"],
            status="succeeded",
            result={
                "summary_span_id": summary.span_id,
                "source_span_count": summary.metadata.get("source_span_count", len(source_spans)),
            },
        )

    def _session_summary_sources_and_hash(self, scope: Scope) -> tuple[list[EvidenceSpan], str]:
        source_spans = [
            span
            for span in self.store.list_spans(scope, include_session=True)
            if span.span_type in {"turn", "tool_result", "document_chunk"} and span.speaker in {"user", "assistant", "agent", "tool", "document"}
        ]
        source_spans.sort(key=lambda span: (span.timestamp, span.turn_id or "", span.span_id))
        selected = source_spans[-self.config.session_summary_max_source_spans :]
        return selected, stable_hash("|".join(span.span_id for span in selected))

    def _model_call_sources(self) -> list[tuple[str, Any]]:
        sources: list[tuple[str, Any]] = [
            ("embedder", getattr(self.store, "embedder", None)),
            ("extractor", self.extractor),
            ("extractor_client", getattr(self.extractor, "client", None)),
            ("reranker", self.reranker),
        ]
        out: list[tuple[str, Any]] = []
        seen: set[int] = set()
        for component, source in sources:
            if source is None or id(source) in seen:
                continue
            seen.add(id(source))
            out.append((component, source))
        return out

    def _model_call_marks(self) -> dict[int, int]:
        marks: dict[int, int] = {}
        for _, source in self._model_call_sources():
            calls = getattr(source, "calls", None)
            if isinstance(calls, list):
                marks[id(source)] = len(calls)
        return marks

    def _model_calls_since(self, marks: dict[int, int]) -> list[dict[str, Any]]:
        calls_out: list[dict[str, Any]] = []
        for component, source in self._model_call_sources():
            calls = getattr(source, "calls", None)
            if not isinstance(calls, list):
                continue
            start = marks.get(id(source), 0)
            for call in calls[start:]:
                if isinstance(call, dict):
                    calls_out.append(_sanitize_model_call(component, source, call))
                else:
                    calls_out.append({"component": component, "model_version": getattr(source, "version", source.__class__.__name__)})
        return calls_out

    def _candidate_to_fact(self, scope: Scope, candidate, session_time: datetime) -> MemoryFact:
        structured = candidate.structured
        return MemoryFact(
            fact_id=new_id("fact"),
            scope=scope,
            subject=str(structured.get("subject", "user")),
            predicate=str(structured.get("predicate", "said")),
            object=str(structured.get("object", candidate.text)),
            text=candidate.text,
            category=str(structured.get("category", "general_fact")),
            confidence=float(structured.get("confidence", candidate.confidence)),
            salience=float(structured.get("salience", 0.5)),
            observed_at=session_time,
            valid_from=session_time,
            valid_to=None,
            source_span_ids=list(dict.fromkeys(candidate.source_span_ids)),
            metadata={"hash": stable_hash(candidate.text), "candidate_local_id": candidate.local_id},
        )

    def _candidate_to_event(self, scope: Scope, candidate) -> MemoryEvent:
        structured = candidate.structured
        return MemoryEvent(
            event_id=new_id("event"),
            scope=scope,
            event_type=str(structured.get("event_type", "user_action")),
            participants=list(structured.get("participants", [])),
            description=str(structured.get("description", candidate.text)),
            time_start=dt_from_str(structured.get("time_start")),
            time_end=dt_from_str(structured.get("time_end")),
            time_granularity=str(structured.get("time_granularity", "unknown")),
            time_source=str(structured.get("time_source", "unknown")),
            source_span_ids=list(dict.fromkeys(candidate.source_span_ids)),
            confidence=float(structured.get("confidence", candidate.confidence)),
        )

    def _candidate_to_relation(self, candidate, local_to_fact: dict[str, str]) -> FactRelation | None:
        structured = candidate.structured
        from_id = local_to_fact.get(str(structured.get("from_local_id")))
        to_id = structured.get("to_fact_id")
        if not from_id or not to_id:
            return None
        return FactRelation(
            relation_id=new_id("rel"),
            from_fact_id=from_id,
            to_fact_id=str(to_id),
            relation_type=str(structured.get("relation_type", "linked_to")),
            source_span_ids=list(dict.fromkeys(candidate.source_span_ids)),
            confidence=float(structured.get("confidence", candidate.confidence)),
        )

    def _create_session_event_edges(self, scope: Scope) -> None:
        events = [event for event in self.store.list_events(scope) if event.scope.session_id == scope.session_id]
        events = [event for event in events if event.time_start]
        events.sort(key=lambda event: event.time_start or datetime.max.replace(tzinfo=timezone.utc))
        for previous, current in zip(events, events[1:]):
            self._insert_event_edge_once(previous, current, confidence=0.70)

    def _create_explicit_event_edges(self, scope: Scope, new_event_ids: list[str]) -> None:
        if not new_event_ids:
            return
        events = [event for event in self.store.list_events(scope) if event.scope.session_id == scope.session_id]
        by_id = {event.event_id: event for event in events}
        for event_id in new_event_ids:
            event = by_id.get(event_id)
            if not event:
                continue
            for relation_text, direction in _explicit_order_mentions(event.description):
                target = self._best_event_text_match(relation_text, [candidate for candidate in events if candidate.event_id != event.event_id])
                if not target:
                    continue
                if direction == "after":
                    self._insert_event_edge_once(target, event, confidence=0.82)
                elif direction == "before":
                    self._insert_event_edge_once(event, target, confidence=0.82)

    def _best_event_text_match(self, text: str, events: list[MemoryEvent]) -> MemoryEvent | None:
        best: tuple[float, MemoryEvent | None] = (0.0, None)
        for event in events:
            score = keyword_score(text, event.description + " " + " ".join(event.participants))
            if score > best[0]:
                best = (score, event)
        return best[1] if best[0] > 0 else None

    def _insert_event_edge_once(self, previous: MemoryEvent, current: MemoryEvent, confidence: float) -> None:
        if self.store.has_event_edge(previous.event_id, current.event_id, edge_type="before"):
            return
        self.store.insert_event_edge(
            EventEdge(
                edge_id=new_id("edge"),
                from_event_id=previous.event_id,
                to_event_id=current.event_id,
                edge_type="before",
                source_span_ids=list(dict.fromkeys(previous.source_span_ids + current.source_span_ids)),
                confidence=confidence,
            )
        )

    def _refresh_views_and_profiles(self, scope: Scope) -> tuple[list[CurrentView], list[EntityProfile]]:
        facts = self.store.list_facts(scope)
        superseded = self.store.superseded_fact_ids()
        views = self.views.build_current_views(scope, facts, superseded)
        profiles = self.views.build_entity_profiles(scope, facts)
        for view in views:
            self.store.upsert_current_view(view)
        for profile in profiles:
            self.store.upsert_entity_profile(profile)
        return views, profiles

    def _candidate_lists(
        self,
        query: str,
        scope: Scope,
        plan,
        per_source_limit: int,
        enabled_sources: list[str] | set[str] | None = None,
        include_session: bool = False,
    ) -> list[list[Candidate]]:
        enabled = set(enabled_sources) if enabled_sources is not None else None
        candidate_lists: list[list[Candidate]] = []
        speaker = plan.speaker_focus if plan.speaker_focus != "any" else None
        if self._source_enabled("raw", enabled):
            raw_span_results = self.store.search_spans(query, scope, limit=per_source_limit, speaker=speaker, include_session=include_session)
            candidate_lists.append(
                [
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source="l0_raw_hybrid",
                        scores=scores,
                        source_span_ids=[span.span_id],
                        metadata={"speaker": span.speaker, "timestamp": span.timestamp.isoformat()},
                    )
                    for span, scores in raw_span_results
                ]
            )
        if self._source_enabled("facts", enabled):
            fact_results = self.store.search_facts(query, scope, limit=per_source_limit, include_session=include_session)
            candidate_lists.append(
                [
                    Candidate(
                        id=fact.fact_id,
                        type="fact",
                        text=fact.text,
                        source="l1_fact_hybrid",
                        scores={**scores, "view_or_profile_prior": 0.0},
                        source_span_ids=fact.source_span_ids,
                        metadata={"category": fact.category, "confidence": fact.confidence},
                    )
                    for fact, scores in fact_results
                ]
            )
        if self._source_enabled("events", enabled):
            event_results = self.store.search_events(query, scope, limit=per_source_limit, include_session=include_session)
            candidate_lists.append(
                [
                    Candidate(
                        id=event.event_id,
                        type="event",
                        text=event.description,
                        source="l2_event_graph",
                        scores={**scores, "graph_proximity": 0.55},
                        source_span_ids=event.source_span_ids,
                        metadata={"event_type": event.event_type, "time_start": event.time_start.isoformat() if event.time_start else None},
                    )
                    for event, scores in event_results
                ]
            )
        if self._source_enabled("views", enabled):
            views = self.store.list_current_views(scope, include_session=include_session)
            candidate_lists.append(
                [
                    Candidate(
                        id=view.view_id,
                        type="view",
                        text=view.text,
                        source="l3_current_view",
                        scores={
                            "bm25_score": keyword_score(query, view.text),
                            "view_or_profile_prior": 0.85,
                            "score": keyword_score(query, view.text) + 0.85,
                        },
                        source_span_ids=view.source_span_ids,
                        metadata={"view_type": view.view_type, "confidence": view.confidence},
                    )
                    for view in views
                ]
            )
        if self._source_enabled("profiles", enabled):
            profile_results = self.store.search_entity_profiles(query, scope, limit=per_source_limit, include_session=include_session)
            candidate_lists.append(
                [
                    Candidate(
                        id=profile.profile_id,
                        type="profile",
                        text=profile.text,
                        source="l3_entity_profile",
                        scores={
                            **scores,
                            "view_or_profile_prior": 0.55,
                            "score": scores.get("score", 0.0) + 0.55,
                        },
                        source_span_ids=profile.source_span_ids,
                        metadata={"profile_type": profile.profile_type, "support_count": profile.support_count},
                    )
                    for profile, scores in profile_results
                ]
            )
        if self._source_enabled("exact", enabled):
            exact = self._exact_candidates(query, scope, per_source_limit, include_session=include_session)
            candidate_lists.append(exact)
        if self._source_enabled("entities", enabled):
            entity = self._entity_candidates(query, scope, per_source_limit, include_session=include_session)
            candidate_lists.append(entity)
        return candidate_lists

    def _source_enabled(self, name: str, enabled: set[str] | None) -> bool:
        return enabled is None or name in enabled

    def _entity_candidates(self, query: str, scope: Scope, limit: int, *, include_session: bool = False) -> list[Candidate]:
        out: list[Candidate] = []
        seen_spans: set[str] = set()
        for entity, scores in self.store.search_entities(query, scope, limit=limit, include_session=include_session):
            for span_id in entity.source_span_ids:
                if span_id in seen_spans:
                    continue
                span = self.store.get_span(span_id)
                if not span:
                    continue
                seen_spans.add(span_id)
                out.append(
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source="entity_registry",
                        scores={**scores, "bm25_score": keyword_score(query, span.content), "score": scores["score"]},
                        source_span_ids=[span.span_id],
                        metadata={"entity": entity.name, "entity_id": entity.entity_id, "speaker": span.speaker},
                    )
                )
                if len(out) >= limit:
                    return out
        return out

    def _exact_candidates(self, query: str, scope: Scope, limit: int, *, include_session: bool = False) -> list[Candidate]:
        spans = self.store.list_spans(scope, include_session=include_session)
        terms = [term for term in query.replace("?", " ").split() if len(term) >= 4]
        out: list[Candidate] = []
        for span in spans:
            lower = span.content.lower()
            hits = sum(1 for term in terms if term.lower() in lower)
            if hits == 0:
                continue
            out.append(
                Candidate(
                    id=span.span_id,
                    type="span",
                    text=span.content,
                    source="exact_filter",
                    scores={"bm25_score": hits / max(1, len(terms)), "score": hits / max(1, len(terms))},
                    source_span_ids=[span.span_id],
                    metadata={"speaker": span.speaker, "summary": compact_summary(span.content)},
                )
            )
        out.sort(key=lambda candidate: candidate.scores["score"], reverse=True)
        return out[:limit]

    def _upsert_span_entities(self, span) -> None:
        for entity in span.entities:
            self.store.upsert_entity(
                span.scope,
                entity,
                entity_type="span_entity",
                source_span_ids=[span.span_id],
                observed_at=span.timestamp,
            )

    def _upsert_fact_entities(self, fact: MemoryFact) -> None:
        names = extract_entities(fact.text + " " + fact.object)
        if fact.subject and fact.subject not in {"user", "assistant", "agent", "tool"}:
            names.append(fact.subject)
        for entity in dict.fromkeys(names):
            self.store.upsert_entity(
                fact.scope,
                entity,
                entity_type="fact_entity",
                source_span_ids=fact.source_span_ids,
                observed_at=fact.observed_at or fact.created_at,
            )

    def _upsert_event_entities(self, event: MemoryEvent) -> None:
        names = list(event.participants) + extract_entities(event.description)
        for entity in dict.fromkeys(name for name in names if name):
            self.store.upsert_entity(
                event.scope,
                entity,
                entity_type="event_participant",
                source_span_ids=event.source_span_ids,
                observed_at=event.time_start,
            )

    def _mark_quota_selected(self, candidates: list[Candidate], span_ids: list[str]) -> list[Candidate]:
        selected = set(span_ids)
        out: list[Candidate] = []
        for candidate in candidates:
            metadata = dict(candidate.metadata)
            if candidate.type == "span" and candidate.id in selected:
                metadata["quota_selected"] = True
            out.append(
                Candidate(
                    id=candidate.id,
                    type=candidate.type,
                    text=candidate.text,
                    source=candidate.source,
                    scores=candidate.scores,
                    source_span_ids=candidate.source_span_ids,
                    metadata=metadata,
                )
            )
        return out

    def _preserve_quota_after_rerank(self, candidates: list[Candidate], quota_span_ids: list[str], limit: int) -> list[Candidate]:
        quota_set = set(quota_span_ids)
        required = [candidate for candidate in candidates if candidate.type == "span" and candidate.id in quota_set]
        selected: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in required + candidates:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= limit:
                break
        return selected

    def _coerce_datetime(self, value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        parsed = value if isinstance(value, datetime) else dt_from_str(value)
        if parsed and parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _event_id(self, value: str | MemoryEvent | dict[str, Any]) -> str | None:
        if isinstance(value, MemoryEvent):
            return value.event_id
        if isinstance(value, str):
            return value
        return value.get("event_id") or value.get("id")

    def _resolve_event(self, value: str | MemoryEvent | dict[str, Any], *, scope: Scope | None = None, include_session: bool = False) -> MemoryEvent | None:
        if isinstance(value, MemoryEvent):
            if scope:
                return self.store.get_event(value.event_id, scope, include_session=include_session)
            return value
        event_id = self._event_id(value)
        if not event_id:
            return None
        return self.store.get_event(event_id, scope, include_session=include_session)

    def _event_edge(self, from_event_id: str, to_event_id: str) -> dict[str, Any] | None:
        return self.store.get_event_edge(from_event_id, to_event_id)


def _source_coverage(items: list[Any]) -> float:
    if not items:
        return 0.0
    covered = 0
    for item in items:
        if isinstance(item, dict):
            source_span_ids = item.get("source_span_ids") or item.get("candidate", {}).get("source_span_ids") or []
        else:
            source_span_ids = getattr(item, "source_span_ids", [])
        covered += int(bool(source_span_ids))
    return covered / len(items)


def _sanitize_model_call(component: str, source: Any, call: dict[str, Any]) -> dict[str, Any]:
    model = call.get("model") or getattr(source, "model", None)
    model_version = getattr(source, "version", None) or model or source.__class__.__name__
    out: dict[str, Any] = {
        "component": component,
        "model_version": model_version,
    }
    if model:
        out["model"] = model
    prompt_version = call.get("prompt_version") or call.get("prompt")
    if isinstance(prompt_version, str):
        prompt_version = prompt_version.splitlines()[0]
        out["prompt_version"] = prompt_version
    latency_ms = call.get("latency_ms")
    if isinstance(latency_ms, int | float):
        out["latency_ms"] = latency_ms
    usage = call.get("usage")
    if isinstance(usage, dict):
        out["usage"] = usage
    cost = call.get("cost")
    if isinstance(cost, int | float):
        out["cost"] = cost
    for key in ("text_count", "doc_count"):
        if isinstance(call.get(key), int):
            out[key] = call[key]
    return out


def _model_call_summary(model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    usage_totals: dict[str, float] = {}
    for call in model_calls:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int | float):
                usage_totals[key] = usage_totals.get(key, 0.0) + float(value)
    return {
        "count": len(model_calls),
        "model_versions": sorted({str(call.get("model_version")) for call in model_calls if call.get("model_version")}),
        "total_latency_ms": sum(float(call.get("latency_ms", 0.0)) for call in model_calls if isinstance(call.get("latency_ms"), int | float)),
        "usage": usage_totals,
    }


def _labeled_precision(items: list[dict[str, Any]], labels: dict[str, bool], *, positive: bool) -> float | None:
    known = 0
    correct = 0
    for item in items:
        candidate = item.get("candidate", {})
        keys = [item.get("decision_id"), candidate.get("local_id"), candidate.get("text")]
        label = next((labels[key] for key in keys if key in labels), None)
        if label is None:
            continue
        known += 1
        correct += int(label is positive)
    return correct / known if known else None


ORDER_RE = re.compile(r"\b(after|before)\s+(?:the\s+)?(.+?)(?:,|\.|;|\bthen\b|\bi\s+|\bwe\s+|$)", re.I)


def _explicit_order_mentions(text: str) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    for match in ORDER_RE.finditer(text):
        direction = match.group(1).lower()
        target = match.group(2).strip()
        if target:
            mentions.append((target, direction))
    return mentions
