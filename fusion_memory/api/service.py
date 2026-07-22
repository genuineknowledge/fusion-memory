from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fusion_memory.api.service_telemetry import (
    _labeled_precision,
    _model_call_marks,
    _model_call_summary,
    _model_calls_since,
    _product_model_calls_since,
    _source_coverage,
)
from fusion_memory.core.auth import AllowAllAuthorizer, Authorizer
from fusion_memory.core.config import DEFAULT_CONFIG, MemoryConfig
from fusion_memory.core.embedding import Embedder
from fusion_memory.core.models import (
    AddResult,
    CurrentView,
    EntityProfile,
    EventEdge,
    EvidenceSpan,
    EvidencePack,
    MemoryEvent,
    Scope,
    SearchResult,
    new_id,
)
from fusion_memory.core.text import keyword_score, stable_hash
from fusion_memory.ingestion.candidate_records import (
    candidate_to_event,
    candidate_to_fact,
    candidate_to_relation,
)
from fusion_memory.ingestion.encoding_gate import EncodingGate
from fusion_memory.ingestion.entity_indexing import EntityIndexer
from fusion_memory.ingestion.extractors import RuleBasedExtractor
from fusion_memory.ingestion.normalizer import normalize_input
from fusion_memory.ingestion.order_markers import _explicit_order_mentions
from fusion_memory.ingestion.views import ViewBuilder
from fusion_memory.ingestion.window_builder import build_session_summary_span
from fusion_memory.retrieval.chronology_normalizer import build_chronology_write_batch
from fusion_memory.retrieval.context import RetrievalContext, RetrievalResult, SearchRequest
from fusion_memory.retrieval.engine import (
    RetrievalEngine,
    build_product_retrieval_engine,
    prepare_retrieval_engine_options,
    sanitize_retrieval_trace,
    summarize_product_model_calls,
)
from fusion_memory.retrieval.reranker import LexicalCrossEncoderReranker, Reranker
from fusion_memory.retrieval.rule_registry import collect_rule_hits
from fusion_memory.retrieval.utility_model import LogisticUtilityScorer, UtilityTrainingReport
from fusion_memory.storage.postgres_store import PostgresMemoryStore
from fusion_memory.storage.postgres_pool import PostgresConnectionPool
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
        postgres_pool: PostgresConnectionPool | None = None,
        postgres_acquire_timeout_seconds: float = 5.0,
        query_intent_refiner: Any | None = None,
        query_intent_refiner_min_confidence: float = 0.70,
        query_intent_refiner_mode: str = "auto",
        async_extractor: Any | None = None,
        retrieval_flags: Any | None = None,
        retrieval_engine: RetrievalEngine | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        if store is not None:
            self.store = store
        elif storage_backend == "sqlite":
            self.store = SQLiteMemoryStore(db_path, embedder=embedder)
        elif storage_backend == "postgres":
            self.store = PostgresMemoryStore(
                str(db_path),
                embedder=embedder,
                connect=store_connect,
                pool=postgres_pool,
                acquire_timeout_seconds=postgres_acquire_timeout_seconds,
            )
        else:
            raise ValueError(f"unsupported storage_backend: {storage_backend}")
        self.storage_backend = storage_backend
        self.authorizer = authorizer or AllowAllAuthorizer()
        self.extractor = extractor or RuleBasedExtractor()
        self.async_extractor = async_extractor
        self.retrieval_flags = retrieval_flags
        self.reranker = reranker or LexicalCrossEncoderReranker()
        self.retrieval_engine = (
            retrieval_engine
            if retrieval_engine is not None
            else build_product_retrieval_engine(
                self.store,
                self.config,
                self.reranker,
                query_intent_refiner=query_intent_refiner,
                query_intent_refiner_min_confidence=query_intent_refiner_min_confidence,
                query_intent_refiner_mode=query_intent_refiner_mode,
            )
        )
        self.gate = EncodingGate(self.config)
        self.views = ViewBuilder()
        self.entity_indexer = EntityIndexer()
        self.utility_scorer = LogisticUtilityScorer()

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
        with collect_rule_hits() as rule_hits:
            return self._add_with_rule_hits(input, scope, session_time, metadata, rule_hits)

    def _add_with_rule_hits(self, input: Any, scope: Scope, session_time: datetime | None, metadata: dict[str, Any] | None, rule_hits) -> AddResult:
        scope.validate_for_add()
        self._authorize("memory.add", scope, {"metadata": metadata or {}})
        model_call_marks = _model_call_marks(self)
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
            self.entity_indexer.upsert_span(self.store, span)
            inserted_span_ids.append(span.span_id)
        trace["steps"].append({"step": "l0_written", "span_ids": inserted_span_ids})

        existing_facts = self.store.list_facts(scope)
        extraction_spans = [span for span in spans if span.span_type not in {"window", "summary"}]
        candidates = self.extractor.extract(extraction_spans, existing_facts, session_time)
        extractor_telemetry = getattr(self.extractor, "last_telemetry", None)
        if isinstance(extractor_telemetry, dict) and extractor_telemetry:
            trace["steps"].append({"step": "extractor_telemetry", **extractor_telemetry})
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
                fact = candidate_to_fact(scope, candidate, session_time)
                self.store.insert_fact(fact)
                self.entity_indexer.upsert_fact(self.store, fact)
                accepted_fact_ids.append(fact.fact_id)
                local_to_fact[candidate.local_id] = fact.fact_id
            elif decision.decision == "accept" and candidate.candidate_type == "event":
                event = candidate_to_event(scope, candidate)
                self.store.insert_event(event)
                self.entity_indexer.upsert_event(self.store, event)
                accepted_event_ids.append(event.event_id)
                local_to_event[candidate.local_id] = event.event_id
            elif decision.decision == "update_relation" and candidate.candidate_type == "relation":
                relation = candidate_to_relation(candidate, local_to_fact)
                if relation:
                    self.store.insert_fact_relation(relation)

        self._create_session_event_edges(scope)
        self._create_explicit_event_edges(scope, accepted_event_ids)
        chronology_graph = self._write_chronology_graph(scope, extraction_spans, accepted_event_ids)
        updated_views, updated_profiles = self._refresh_views_and_profiles(scope)
        summary_task = self._maybe_enqueue_session_summary_task(scope)
        extraction_task = self._maybe_enqueue_llm_extraction_task(scope, extraction_spans, session_time)
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
                "chronology_graph": chronology_graph,
                "background_task_ids": [task["task_id"] for task in (summary_task, extraction_task) if task],
            }
        )
        model_calls = _model_calls_since(self, model_call_marks)
        trace["model_calls"] = model_calls
        trace["rule_hits"] = [hit.__dict__ for hit in rule_hits.drain()]
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
                "llm_extraction_task_id": extraction_task["task_id"] if extraction_task else None,
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
        prepared_options = prepare_retrieval_engine_options(options, self.config)
        _, _, result, trace_id = self._run_retrieval_engine(
            query,
            scope,
            prepared_options,
        )
        return SearchResult(
            candidates=list(result.candidates),
            trace_id=trace_id,
            coverage=dict(result.coverage),
        )

    def _run_retrieval_engine(
        self,
        query: str,
        scope: Scope,
        options: dict[str, Any],
    ) -> tuple[RetrievalContext, SearchRequest, RetrievalResult, str]:
        if self.retrieval_engine is None:
            raise RuntimeError("retrieval engine is not configured")
        scope.validate_for_read()
        allow_cross_session = options["allow_cross_session"]
        include_session = bool(scope.session_id and not allow_cross_session)
        mode = options["mode"]
        limit = options["limit"]
        self._authorize(
            "memory.search",
            scope,
            {
                "query": query,
                "allow_cross_session": allow_cross_session,
                "include_session": include_session,
                "mode": mode,
                "limit": limit,
                "enabled_sources": options.get("enabled_sources"),
            },
        )

        now = datetime.now(timezone.utc)
        trace_id = new_id("trace")
        context = RetrievalContext(
            scope=scope,
            user_id=scope.user_id,
            now=now,
            trace_id=trace_id,
            deadline=options.get("deadline"),
            include_session=include_session,
        )
        request = SearchRequest(
            query=query,
            limit=limit,
            mode=mode,
            time_range=options.get("time_range"),
            include_trace=bool(options.get("include_trace", True)),
            enabled_providers=options["enabled_providers"],
        )
        model_call_marks = _model_call_marks(self)
        result = self.retrieval_engine.search(context, request)
        model_calls = _product_model_calls_since(self, model_call_marks)
        trace = sanitize_retrieval_trace(result.trace)
        trace["operation"] = "search"
        trace["allow_cross_session"] = allow_cross_session
        trace["include_session"] = include_session
        trace["model_calls"] = model_calls
        self.store.save_trace(trace_id, trace, scope)
        self.store.insert_audit_event(
            scope,
            "memory.search",
            object_type="trace",
            object_id=trace_id,
            trace_id=trace_id,
            payload={
                "query_hash": stable_hash(query),
                "query_length": len(query),
                "intent": trace.get("intent", "unknown"),
                "mode": mode,
                "candidate_count": len(result.candidates),
                "allow_cross_session": allow_cross_session,
                "include_session": include_session,
                "model_calls": summarize_product_model_calls(model_calls),
            },
        )
        return context, request, result, trace_id

    def answer_context(self, query: str, scope: Scope, budget: dict[str, Any] | None = None) -> EvidencePack:
        product_budget = prepare_retrieval_engine_options(budget, self.config)
        token_budget = product_budget.get("token_budget")
        if token_budget is None:
            token_budget = self.config.answer_context_budget_tokens
        scope.validate_for_read()
        self._authorize(
            "memory.answer_context",
            scope,
            {
                "query": query,
                "allow_cross_session": product_budget["allow_cross_session"],
                "limit": product_budget["limit"],
                "mode": product_budget["mode"],
                "token_budget": token_budget,
            },
        )
        context, request, result, _ = self._run_retrieval_engine(
            query,
            scope,
            product_budget,
        )
        return self.retrieval_engine.build_evidence_pack(
            context,
            request,
            result,
            token_budget,
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
                elif task["task_type"] == "llm_extract":
                    updated = self._process_llm_extraction_task(task)
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

    def process_server_background_tasks(self, *, limit: int = 5, task_types: set[str] | None = None) -> dict[str, Any]:
        task_types = task_types or {"refresh_session_summary"}
        scan_limit = max(limit * 10, limit + 10)
        tasks = [
            task
            for task in self.store.next_background_tasks(limit=scan_limit, scope=None)
            if task.get("task_type") in task_types
        ][:limit]
        processed: list[dict[str, Any]] = []
        for task in tasks:
            task_scope = Scope(**task["scope"])
            self._authorize(
                "memory.tasks.process",
                task_scope,
                {"limit": limit, "allow_cross_session": False, "include_session": bool(task_scope.session_id), "server_background": True},
            )
            self.store.update_background_task(task["task_id"], status="running")
            try:
                if task["task_type"] == "refresh_session_summary":
                    updated = self._process_refresh_session_summary_task(task)
                else:
                    updated = self.store.update_background_task(
                        task["task_id"],
                        status="skipped",
                        result={"reason": "server_task_type_disabled", "task_type": task["task_type"]},
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

    def clear(self, scope: Scope, allow_cross_session: bool = False) -> dict[str, Any]:
        scope.validate_for_read()
        include_session = bool(scope.session_id and not allow_cross_session)
        self._authorize(
            "memory.clear",
            scope,
            {
                "allow_cross_session": allow_cross_session,
                "include_session": include_session,
            },
        )
        if not hasattr(self.store, "clear_scope"):
            raise RuntimeError("memory store does not support clear")
        result = self.store.clear_scope(scope, include_session=include_session)
        audit_id = self.store.insert_audit_event(
            scope,
            "memory.clear",
            object_type="scope",
            payload={
                "allow_cross_session": allow_cross_session,
                "include_session": include_session,
                "deleted": result.get("deleted", {}),
            },
        )
        return {
            "ok": True,
            "operation": "clear_scope",
            "allow_cross_session": allow_cross_session,
            "include_session": include_session,
            "audit_id": audit_id,
            **result,
        }

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

    def _maybe_enqueue_llm_extraction_task(
        self,
        scope: Scope,
        spans: list[EvidenceSpan],
        session_time: datetime,
    ) -> dict[str, Any] | None:
        if self.async_extractor is None or not spans:
            return None
        source_span_ids = [span.span_id for span in spans]
        dedupe_key = "llm_extract:" + stable_hash(
            "|".join(
                [
                    scope.workspace_id or "",
                    scope.user_id or "",
                    scope.agent_id or "",
                    scope.run_id or "",
                    scope.session_id or "",
                    scope.app_id or "",
                    *source_span_ids,
                ]
            )
        )
        return self.store.enqueue_background_task(
            scope,
            "llm_extract",
            payload={
                "source_span_ids": source_span_ids,
                "session_time": session_time.isoformat(),
                "mode": "quality_evaluation",
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

    def _process_llm_extraction_task(self, task: dict[str, Any]) -> dict[str, Any] | None:
        if self.async_extractor is None:
            return self.store.update_background_task(
                task["task_id"],
                status="skipped",
                result={"reason": "async_extractor_disabled"},
            )
        task_scope = Scope(**task["scope"])
        payload = task.get("payload") or {}
        source_span_ids = [span_id for span_id in payload.get("source_span_ids", []) if isinstance(span_id, str)]
        if not source_span_ids:
            return self.store.update_background_task(
                task["task_id"],
                status="skipped",
                result={"reason": "missing_source_spans"},
            )
        source_span_id_set = set(source_span_ids)
        spans = [
            span
            for span in self.store.list_spans(task_scope, include_session=True)
            if span.span_id in source_span_id_set
        ]
        spans.sort(key=lambda span: source_span_ids.index(span.span_id))
        if not spans:
            return self.store.update_background_task(
                task["task_id"],
                status="skipped",
                result={"reason": "source_spans_not_found"},
            )
        session_time = dt_from_str(str(payload.get("session_time"))) if payload.get("session_time") else max(span.timestamp for span in spans)
        existing_facts = self.store.list_facts(task_scope)
        candidates = self.async_extractor.extract(spans, existing_facts, session_time)
        decisions = self.gate.decide(candidates, existing_facts)
        decision_counts: dict[str, int] = {}
        extractor_counts: dict[str, int] = {}
        for decision in decisions:
            self.store.insert_encoding_decision(task_scope, decision)
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
            extractor = decision.candidate.extractor_name
            extractor_counts[extractor] = extractor_counts.get(extractor, 0) + 1
        telemetry = getattr(self.async_extractor, "last_telemetry", None)
        return self.store.update_background_task(
            task["task_id"],
            status="succeeded",
            result={
                "mode": "quality_evaluation",
                "source_span_count": len(spans),
                "candidate_count": len(candidates),
                "gate_decision_counts": decision_counts,
                "extractor_counts": extractor_counts,
                "accepted_candidate_count": decision_counts.get("accept", 0),
                "telemetry": telemetry if isinstance(telemetry, dict) else {},
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

    def _write_chronology_graph(
        self,
        scope: Scope,
        spans: list[EvidenceSpan],
        accepted_event_ids: list[str],
    ) -> dict[str, Any]:
        empty_counts = {"enabled": True, "node_count": 0, "edge_count": 0, "topic_count": 0, "phase_count": 0}
        if not spans:
            return empty_counts

        try:
            accepted_event_id_set = set(accepted_event_ids)
            events = [
                event
                for event in self.store.list_events(scope, include_session=True)
                if event.event_id in accepted_event_id_set
            ]
            batch = build_chronology_write_batch(scope, spans, events)
            for topic in batch.topics:
                self.store.upsert_chronology_topic(topic)
            for phase in batch.phases:
                self.store.upsert_chronology_phase(phase)
            for node in batch.nodes:
                self.store.upsert_chronology_event_node(node)
            inserted_edges = 0
            for edge in batch.edges:
                inserted_edges += int(self.store.insert_chronology_event_edge(edge))
            return {
                "enabled": True,
                "topic_count": len(batch.topics),
                "phase_count": len(batch.phases),
                "node_count": len(batch.nodes),
                "edge_count": inserted_edges,
                "telemetry": batch.telemetry,
            }
        except Exception as exc:
            return {**empty_counts, "error": exc.__class__.__name__}

    def _refresh_views_and_profiles(self, scope: Scope) -> tuple[list[CurrentView], list[EntityProfile]]:
        facts = self.store.list_facts(scope, include_session=bool(scope.session_id))
        superseded = self.store.superseded_fact_ids()
        views = self.views.build_current_views(scope, facts, superseded)
        profiles = self.views.build_entity_profiles(scope, facts)
        for view in views:
            self.store.upsert_current_view(view)
        for profile in profiles:
            self.store.upsert_entity_profile(profile)
        return views, profiles

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
