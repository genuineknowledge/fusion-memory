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
from fusion_memory.retrieval.chronology_normalizer import build_chronology_write_batch
from fusion_memory.retrieval.chronology_selector import select_persisted_graph_event_ordering_candidates
from fusion_memory.retrieval.candidate_lifecycle import CandidateLifecycleRecorder
from fusion_memory.retrieval.evidence_pack import EvidencePackBuilder
from fusion_memory.retrieval.event_graph_selection import (
    _event_milestone_group,
    _event_ordering_milestone_score,
    _select_event_ordering_representatives,
)
from fusion_memory.retrieval.candidate_provider import build_candidate_lists
from fusion_memory.retrieval.aggregation_keys import (
    aggregation_keys_for_query as _aggregation_keys,
    combinatorics_aggregation_keys,
    generic_aggregation_keys,
    generic_list_candidate_keys,
    is_combinatorics_aggregation_query as _is_combinatorics_aggregation_query,
    is_generic_count_or_list_query,
    is_stress_break_aggregation_query as _is_stress_break_aggregation_query,
    stress_break_aggregation_keys,
    vendor_tool_aggregation_keys,
)
from fusion_memory.retrieval.mmr import mmr
from fusion_memory.retrieval.pipeline import (
    CandidateFusionEngine,
    EvidencePackAssembler,
    QueryUnderstandingEngine,
    RecallOrchestrator,
    RetrievalExecutionContext,
    RetrievalTraceRecorder,
    build_pipeline_record,
    selected_temporal_relation_summary,
)
from fusion_memory.retrieval.query_planner import QueryPlanner
from fusion_memory.retrieval.raw_evidence_quota import RawEvidenceQuota
from fusion_memory.retrieval.preservation import annotate_runtime_preservation_candidates, must_preserve_reasons, preserve_required_candidates
from fusion_memory.retrieval.reranker import LexicalCrossEncoderReranker, Reranker, rerank_candidates
from fusion_memory.retrieval.rule_registry import RuleDefinition, collect_rule_hits, record_rule_hit, register_rule
from fusion_memory.retrieval.structured_annotations import select_event_ordering_timeline
from fusion_memory.retrieval.utility_model import LogisticUtilityScorer, UtilityTrainingReport
from fusion_memory.retrieval.utility_scorer import utility_example
from fusion_memory.storage.postgres_store import PostgresMemoryStore
from fusion_memory.storage.sqlite_store import SQLiteMemoryStore, dt_from_str
from fusion_memory.api.service_helpers import (
    _source_coverage,
    _broad_recall_candidate_allowed,
    _replaceable_low_synthesis_index,
    _dedupe_event_ordering_support_events,
    TOPIC_SCOPE_STOPWORDS,
    TOPIC_SCOPE_EQUIVALENTS,
    _topic_scope_group_limit,
    _topic_scope_groups,
    _topic_scope_score,
    _exact_overlap_score,
    _surface_claim_polarity,
    _topic_phrase_bonus,
    _topic_anchor_phrases_for_service,
    _clean_topic_anchor_for_service,
    _topic_anchor_score_for_service,
    _topic_scope_tokens,
    _expand_topic_tokens,
    _span_group_key,
    _date_signal,
    _temporal_target_roles_for_service,
    _temporal_roles_in_text,
    _temporal_focus_terms_for_service,
    _exact_query_terms,
    _aggregation_query_terms,
    _broad_raw_recall_queries,
    _intent_string_list,
    _intent_recall_signal,
    _scent_trail_queries,
    _ordered_topic_scope_tokens,
    _scent_trail_score,
    _quality_fallback_terms,
    _fallback_salience_score,
    _cjk_exact_match_phrases,
    _matched_query_conditions,
    _aggregation_signal,
    _adjacent_assistant_recommendation_spans,
    _aggregation_recommendation_request_signal,
    _recommendation_request_specificity,
    _assistant_recommendation_list_signal,
    _synthesis_evidence_signal,
    _is_cross_factor_synthesis_query,
    _synthesis_candidate_key,
    _aggregation_focus_priority,
    _is_generic_count_or_list_query,
    _generic_aggregation_keys,
    _clean_generic_aggregation_key,
    _quoted_title_candidates,
    _normalize_title_key,
    _is_non_title_quote,
    _has_value_intent,
    _has_current_intent,
    _compatible_value_mention,
    _value_signal,
    _current_state_signal,
    _code_identifier_signal,
    EVENT_ORDERING_STOPWORDS,
    _candidate_in_timeline_window,
    _natural_turn_key,
    _key_diverse_aggregation_candidates,
    _aggregation_scene_representatives,
    _aggregation_context_support_candidate,
    _aggregation_group_support_specificity,
    _high_value_aggregation_context_support,
    _aggregation_query_date_support,
    _aggregation_context_specificity,
    _aggregation_query_context_keys,
    _is_broad_exploration_aggregation_query,
    _service_date_scope_labels,
    _sanitize_model_call,
    _model_call_summary,
    _labeled_precision,
    ORDER_RE,
    _explicit_order_mentions,
)

try:
    from fusion_memory.retrieval.event_graph_selection import select_graph_first_event_ordering_candidates
except ImportError:
    select_graph_first_event_ordering_candidates = None


register_rule(
    RuleDefinition(
        rule_id="event_ordering.legacy_rescue",
        module=__name__,
        purpose="track legacy event-ordering fallback candidates selected instead of graph-first path",
        category="high_risk",
        pattern="graph_fallback|legacy_fallback",
        ability="event_ordering",
        protected=True,
        protected_reason="legacy_event_ordering_fallback",
    )
)


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
        query_intent_refiner: Any | None = None,
        query_intent_refiner_min_confidence: float = 0.70,
        query_intent_refiner_mode: str = "auto",
        async_extractor: Any | None = None,
        retrieval_flags: Any | None = None,
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
        self.storage_backend = storage_backend
        self.authorizer = authorizer or AllowAllAuthorizer()
        self.extractor = extractor or RuleBasedExtractor()
        self.async_extractor = async_extractor
        self.retrieval_flags = retrieval_flags
        self.gate = EncodingGate(self.config)
        self.views = ViewBuilder()
        self.planner = QueryPlanner(
            intent_refiner=query_intent_refiner,
            intent_refiner_min_confidence=query_intent_refiner_min_confidence,
            intent_refiner_mode=query_intent_refiner_mode,
        )
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
        with collect_rule_hits() as rule_hits:
            return self._add_with_rule_hits(input, scope, session_time, metadata, rule_hits)

    def _add_with_rule_hits(self, input: Any, scope: Scope, session_time: datetime | None, metadata: dict[str, Any] | None, rule_hits) -> AddResult:
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
        model_calls = self._model_calls_since(model_call_marks)
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
        with collect_rule_hits() as rule_hits:
            return self._search_with_rule_hits(query, scope, options, rule_hits)

    def _search_with_rule_hits(self, query: str, scope: Scope, options: dict[str, Any] | None, rule_hits) -> SearchResult:
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
        query_understanding = QueryUnderstandingEngine().run(query, scope, options, self.planner)
        plan = query_understanding.plan
        intent_telemetry = query_understanding.intent_telemetry
        lifecycle = CandidateLifecycleRecorder()
        retrieval_context = RetrievalExecutionContext(
            service=self,
            query=query,
            scope=scope,
            options=options,
            query_understanding=query_understanding,
            include_session=include_session,
            per_source_limit=options.get("per_source_limit", self.config.retrieval_top_k_per_source),
            enabled_sources=options.get("enabled_sources"),
            mode=options.get("mode", "fast"),
            limit=options.get("limit", self.config.retrieval_output_n),
            rerank_top_n=options.get("rerank_top_n") or 0,
            event_milestone_group=_event_milestone_group,
        )
        recall_result = self._recall_candidates(retrieval_context)
        candidate_lists = recall_result.candidate_lists
        for items in candidate_lists:
            lifecycle.extend(items, "recalled", "candidate_provider")
        fusion_result = CandidateFusionEngine().run(retrieval_context, recall_result)
        fused = fusion_result.fused
        scored = fusion_result.scored
        quota_result = fusion_result.quota_result
        marked = fusion_result.marked
        scored_again = fusion_result.scored_again
        mode = fusion_result.mode
        limit = fusion_result.limit
        rerank_top_n = fusion_result.rerank_top_n
        lifecycle.extend(scored_again, "scored", "utility_scoring")

        def candidate_key(candidate: Candidate) -> tuple[str, str]:
            return (str(candidate.type), str(candidate.id))

        def record_rescued_delta(before: list[Candidate], after: list[Candidate], reason_code: str) -> list[Candidate]:
            before_keys = {candidate_key(candidate) for candidate in before}
            for candidate in after:
                if candidate_key(candidate) not in before_keys:
                    lifecycle.record(candidate, "rescued", reason_code, contributed=True)
            return after

        def record_filtered_delta(before: list[Candidate], after: list[Candidate], reason_code: str) -> list[Candidate]:
            after_keys = {candidate_key(candidate) for candidate in after}
            for candidate in before:
                if candidate_key(candidate) not in after_keys:
                    lifecycle.record(candidate, "filtered", reason_code, contributed=False)
            return after

        preselected = mmr(scored_again, limit=rerank_top_n, lambda_=self.config.mmr_lambda)
        before = preselected
        preselected = record_rescued_delta(before, self._preserve_high_signal_exact(scored_again, before, rerank_top_n), "preserve_high_signal_exact")
        before = preselected
        preselected = record_rescued_delta(before, self._preserve_scent_trail(scored_again, before, rerank_top_n), "preserve_scent_trail")
        before = preselected
        preselected = record_rescued_delta(before, self._preserve_broad_raw_recall(query, plan, scored_again, before, rerank_top_n), "preserve_broad_raw_recall")
        if plan.query_type == "temporal_lookup":
            before = preselected
            preselected = record_rescued_delta(before, self._preserve_temporal_coverage(scored_again, before, rerank_top_n), "preserve_temporal_coverage")
        if plan.query_type == "multi_session_reasoning":
            before = preselected
            preselected = record_rescued_delta(before, self._preserve_aggregation_coverage(query, scored_again, before, rerank_top_n), "preserve_aggregation_coverage")
            before = preselected
            preselected = record_rescued_delta(before, self._preserve_user_synthesis_anchors(scored_again, before, rerank_top_n), "preserve_user_synthesis_anchors")
        if plan.query_type == "contradiction_resolution":
            before = preselected
            preselected = record_rescued_delta(before, self._preserve_contradiction_claim_coverage(scored_again, before, rerank_top_n), "preserve_contradiction_claim_coverage")
        before = preselected
        preselected = record_rescued_delta(before, self._preserve_high_ranked_summaries(scored_again, before, rerank_top_n), "preserve_high_ranked_summaries")
        if plan.query_type == "event_ordering":
            before = preselected
            preselected = record_rescued_delta(before, self._preserve_event_ordering_events(query, scored_again, before, rerank_top_n), "preserve_event_ordering_events")
            before = preselected
            preselected = record_rescued_delta(before, self._preserve_event_ordering_raw_facets(query, scored_again, before, rerank_top_n), "preserve_event_ordering_raw_facets")
        rerank_applied = mode in {"balanced", "benchmark"}
        if rerank_applied:
            reranked = rerank_candidates(query, preselected, self.reranker)
            selected = record_rescued_delta(reranked, self._preserve_quota_after_rerank(reranked, quota_result.selected_span_ids, limit), "preserve_quota_after_rerank")
            before = selected
            selected = record_rescued_delta(before, self._preserve_high_signal_exact(scored_again, before, limit), "preserve_high_signal_exact")
            before = selected
            selected = record_rescued_delta(before, self._preserve_scent_trail(scored_again, before, limit), "preserve_scent_trail")
            before = selected
            selected = record_rescued_delta(before, self._preserve_broad_raw_recall(query, plan, scored_again, before, limit), "preserve_broad_raw_recall")
            if plan.query_type == "temporal_lookup":
                before = selected
                selected = record_rescued_delta(before, self._preserve_temporal_coverage(scored_again, before, limit), "preserve_temporal_coverage")
            if plan.query_type == "multi_session_reasoning":
                before = selected
                selected = record_rescued_delta(before, self._preserve_aggregation_coverage(query, scored_again, before, limit), "preserve_aggregation_coverage")
                before = selected
                selected = record_rescued_delta(before, self._preserve_user_synthesis_anchors(scored_again, before, limit), "preserve_user_synthesis_anchors")
            if plan.query_type == "contradiction_resolution":
                before = selected
                selected = record_rescued_delta(before, self._preserve_contradiction_claim_coverage(scored_again, before, limit), "preserve_contradiction_claim_coverage")
            before = selected
            selected = record_rescued_delta(before, self._preserve_high_ranked_summaries(scored_again, before, limit), "preserve_high_ranked_summaries")
            if plan.query_type == "event_ordering":
                before = selected
                selected = record_rescued_delta(before, self._preserve_event_ordering_events(query, scored_again, before, limit), "preserve_event_ordering_events")
                before = selected
                selected = record_rescued_delta(before, self._preserve_event_ordering_raw_facets(query, scored_again, before, limit), "preserve_event_ordering_raw_facets")
        else:
            selected = preselected[:limit]
        before = selected
        selected = record_filtered_delta(before, self._apply_topic_scope_filter(query, plan, scored_again, before, limit), "topic_scope_filter")
        before = selected
        selected = record_rescued_delta(before, self._apply_quality_fallback(query, plan, scope, scored_again, before, limit, include_session=include_session), "quality_fallback")
        before = selected
        selected = record_rescued_delta(before, self._preserve_broad_raw_recall(query, plan, scored_again, before, limit), "preserve_broad_raw_recall")
        if plan.query_type == "temporal_lookup":
            before = selected
            selected = record_rescued_delta(before, self._preserve_temporal_coverage(scored_again, before, limit), "preserve_temporal_coverage")
        if plan.query_type == "multi_session_reasoning":
            before = selected
            selected = record_rescued_delta(before, self._preserve_aggregation_coverage(query, scored_again, before, limit), "preserve_aggregation_coverage")
            before = selected
            selected = record_rescued_delta(before, self._preserve_user_synthesis_anchors(scored_again, before, limit), "preserve_user_synthesis_anchors")
        if plan.query_type == "contradiction_resolution":
            before = selected
            selected = record_rescued_delta(before, self._preserve_contradiction_claim_coverage(scored_again, before, limit), "preserve_contradiction_claim_coverage")
        if plan.query_type == "event_ordering":
            before = selected
            selected = record_rescued_delta(before, self._preserve_event_ordering_events(query, scored_again, before, limit), "preserve_event_ordering_events")
            before = selected
            selected = record_rescued_delta(before, self._preserve_event_ordering_raw_facets(query, scored_again, before, limit), "preserve_event_ordering_raw_facets")
        before = selected
        selected = record_filtered_delta(before, self._apply_topic_scope_filter(query, plan, scored_again, before, limit), "topic_scope_filter")
        before = selected
        selected = record_rescued_delta(before, self._preserve_scent_trail(scored_again, before, limit), "preserve_scent_trail")
        before = selected
        selected = record_rescued_delta(before, self._preserve_broad_raw_recall(query, plan, scored_again, before, limit), "preserve_broad_raw_recall")
        before = selected
        selected = record_filtered_delta(before, self._filter_stale_current_value_candidates(query, plan, before), "stale_current_value_filter")
        annotated_candidates = annotate_runtime_preservation_candidates(scored_again)
        before = selected
        selected, dropped_high_signal = preserve_required_candidates(annotated_candidates, before, limit)
        selected = record_rescued_delta(before, selected, "runtime_required_preservation")
        if plan.query_type == "event_ordering":
            before = selected
            selected, scope_dropped_high_signal = self._apply_event_ordering_post_preservation_topic_scope_filter(
                query,
                plan,
                scored_again,
                before,
                limit,
            )
            selected = record_filtered_delta(before, selected, "event_ordering_topic_scope_filter")
            dropped_high_signal.extend(scope_dropped_high_signal)
        for candidate in selected:
            lifecycle.record(candidate, "selected", "final_selection", contributed=True)
        for dropped in dropped_high_signal:
            candidate = dropped.get("candidate") if isinstance(dropped, dict) else None
            if isinstance(candidate, Candidate):
                lifecycle.record(candidate, "filtered", "runtime_required_preservation", contributed=False)
        for candidate in selected:
            self.store.insert_utility_example(utility_example(trace_id, query, plan, candidate))
        shadow_ranking = self.utility_scorer.rank_shadow(selected, plan) if self.utility_scorer.trained else []
        coverage = {
            "query_type": plan.query_type,
            "source_span_quota_required": quota_result.required,
            "source_span_quota_selected": len(quota_result.selected_span_ids),
            "selected_span_ids": quota_result.selected_span_ids,
            "selected_candidate_sources": list(dict.fromkeys(candidate.source for candidate in selected)),
            "quality_fallback_selected": any(candidate.source == "quality_fallback" for candidate in selected),
            "source_span_quota_met": not quota_result.coverage_insufficient,
            "coverage_insufficient": quota_result.coverage_insufficient,
            "raw_quota_backfilled": quota_result.backfilled,
            "dropped_high_signal_candidates": dropped_high_signal,
        }
        if plan.query_type == "event_ordering":
            coverage.update(self._event_ordering_shadow_coverage(candidate_lists, selected))
            if getattr(self.retrieval_flags, "dual_event_ordering_shadow", False):
                coverage["event_ordering_dual_shadow"] = self._event_ordering_dual_shadow_coverage(
                    query,
                    scope,
                    plan,
                    limit,
                    include_session,
                )
        if intent_telemetry:
            coverage["query_intent_telemetry"] = intent_telemetry
        coverage["candidate_lifecycle"] = lifecycle.summary()
        lifecycle_trace = lifecycle.to_trace()
        pipeline_record = build_pipeline_record(
            plan.query_type,
            str(mode),
            language=query_understanding.language,
            intent=query_understanding.intent,
            features=list(query_understanding.features),
            recalled=recall_result.recalled_candidates,
            selected=selected,
            dropped_count=len(dropped_high_signal),
            source_span_count=len(quota_result.selected_span_ids),
            coverage_insufficient=quota_result.coverage_insufficient,
            temporal_relation_summary=selected_temporal_relation_summary(selected),
            provider_summary=recall_result.provider_summary,
        )
        pipeline_trace = RetrievalTraceRecorder(pipeline_record).flush()
        coverage["pipeline_trace"] = pipeline_trace
        trace = {
            "operation": "search",
            "query": query,
            "plan": plan.__dict__,
            "config": self.config.snapshot(),
            "candidate_counts": [len(items) for items in candidate_lists],
            "coverage": coverage,
            "pipeline_trace": pipeline_trace,
            "retrieval_trace": pipeline_trace,
            "candidate_lifecycle_trace": lifecycle_trace,
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
        for candidate in selected:
            if plan.query_type == "contradiction_resolution" and "contradiction_claim_" in candidate.source:
                claim_source = f"contradiction_claim_{candidate.metadata.get('claim_polarity') or 'uncertain'}"
                trace["selected"].append(
                    {
                        "id": candidate.id,
                        "type": candidate.type,
                        "source": claim_source,
                        "scores": candidate.scores,
                        "source_span_ids": candidate.source_span_ids,
                        "claim_polarity": candidate.metadata.get("claim_polarity"),
                    }
                )
        model_calls = self._model_calls_since(model_call_marks)
        trace["model_calls"] = model_calls
        trace["rule_hits"] = [hit.__dict__ for hit in rule_hits.drain()]
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
        with collect_rule_hits() as rule_hits:
            return self._answer_context_with_rule_hits(query, scope, budget, rule_hits)

    def _answer_context_with_rule_hits(self, query: str, scope: Scope, budget: dict[str, Any] | None, rule_hits) -> EvidencePack:
        budget = budget or {}
        mode = budget.get("mode", "fast")
        limit = budget.get("limit")
        rerank_top_n = budget.get("rerank_top_n")
        token_budget = budget.get("token_budget")
        plan = self.planner.plan(query, query_type_hint=budget.get("query_type_hint"))
        if mode == "benchmark":
            if plan.query_type == "event_ordering":
                limit = limit or max(self.config.retrieval_output_n, 24)
                rerank_top_n = rerank_top_n or max(self.config.benchmark_mode_rerank_top_n, min(72, limit * 2))
            else:
                limit = limit or max(self.config.retrieval_output_n, 50)
                rerank_top_n = rerank_top_n or max(self.config.benchmark_mode_rerank_top_n, min(160, limit * 3))
            token_budget = token_budget or max(self.config.answer_context_budget_tokens, 24000)
        else:
            limit = limit or self.config.retrieval_output_n
            token_budget = token_budget or self.config.answer_context_budget_tokens
        intent_telemetry = self.planner.last_intent_telemetry
        scope.validate_for_read()
        self._authorize(
            "memory.answer_context",
            scope,
            {
                "query": query,
                "allow_cross_session": bool(budget.get("allow_cross_session", False)),
                "limit": limit,
                "mode": mode,
                "token_budget": token_budget,
            },
        )
        result = self.search(
            query,
            scope,
            options={
                "limit": limit,
                "mode": mode,
                "rerank_top_n": rerank_top_n,
                "enabled_sources": budget.get("enabled_sources"),
                "allow_cross_session": budget.get("allow_cross_session", False),
                "query_type_hint": budget.get("query_type_hint"),
                "_plan": plan,
                "_intent_telemetry": intent_telemetry,
            },
        )
        trace = self.store.get_trace(result.trace_id, scope, include_session=bool(scope.session_id and not budget.get("allow_cross_session", False))) or {}
        existing_rule_hits = list(trace.get("rule_hits") or [])
        pack = self.pack_builder.build(
            query,
            plan,
            result.candidates,
            result.coverage,
            trace.get("selected", []),
            token_budget=token_budget,
        )
        if "pipeline_trace" in pack.coverage:
            pack.coverage["pipeline_trace"] = EvidencePackAssembler().update_pipeline_output(
                pack.coverage["pipeline_trace"],
                source_span_count=len(pack.source_spans),
                coverage_insufficient=bool(pack.coverage.get("coverage_insufficient", False)),
            )
        lifecycle = dict(pack.coverage.get("candidate_lifecycle") or {})
        stage_counts = dict(lifecycle.get("stage_counts") or {})
        stage_counts["packed"] = len(pack.source_spans)
        lifecycle["stage_counts"] = stage_counts
        lifecycle["packed_source_span_count"] = len(pack.source_spans)
        pack.coverage["candidate_lifecycle"] = lifecycle
        pack_rule_hits = [hit.__dict__ for hit in rule_hits.drain()]
        if pack_rule_hits:
            seen_hit_keys = {
                (
                    str(hit.get("rule_id")),
                    str(hit.get("stage")),
                    str(hit.get("text_hash")),
                    str(hit.get("contributed_candidate_id")),
                )
                for hit in existing_rule_hits
                if isinstance(hit, dict)
            }
            for hit in pack_rule_hits:
                key = (
                    str(hit.get("rule_id")),
                    str(hit.get("stage")),
                    str(hit.get("text_hash")),
                    str(hit.get("contributed_candidate_id")),
                )
                if key in seen_hit_keys:
                    continue
                existing_rule_hits.append(hit)
                seen_hit_keys.add(key)
        if existing_rule_hits:
            pack.coverage["rule_hits"] = existing_rule_hits
        return pack

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

    def _model_call_sources(self) -> list[tuple[str, Any]]:
        sources: list[tuple[str, Any]] = [
            ("embedder", getattr(self.store, "embedder", None)),
            ("extractor", self.extractor),
            ("extractor_client", getattr(self.extractor, "client", None)),
            ("async_extractor", self.async_extractor),
            ("async_extractor_client", getattr(self.async_extractor, "client", None)),
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
            polarity=str(structured.get("polarity", "unknown")),
            source_span_ids=list(dict.fromkeys(candidate.source_span_ids)),
            metadata={
                "hash": stable_hash(candidate.text),
                "candidate_local_id": candidate.local_id,
                **({"value_mentions": structured["value_mentions"]} if structured.get("value_mentions") else {}),
                **({"topic_terms": structured["topic_terms"]} if structured.get("topic_terms") else {}),
            },
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
        return build_candidate_lists(
            self,
            query,
            scope,
            plan,
            per_source_limit,
            enabled_sources=enabled_sources,
            include_session=include_session,
            event_milestone_group=_event_milestone_group,
        )

    def _recall_candidates(self, context: RetrievalExecutionContext) -> RecallResult:
        return RecallOrchestrator().run(context)

    def _event_ordering_graph_selector_candidates(
        self,
        query: str,
        scope: Scope,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        persisted_candidates, persisted_telemetry = select_persisted_graph_event_ordering_candidates(
            query,
            scope,
            self.store,
            limit=limit,
            include_session=include_session,
        )
        if persisted_candidates:
            for candidate in persisted_candidates:
                candidate.metadata["graph_selector_telemetry"] = persisted_telemetry
            return persisted_candidates

        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"}
            and span.speaker in {"user", "assistant", "agent", "document"}
        ]
        events = self.store.list_events(scope, include_session=include_session)
        if select_graph_first_event_ordering_candidates is None:
            fallback_candidates = select_event_ordering_timeline(query, spans, events, limit=limit)
        else:
            fallback_candidates = select_graph_first_event_ordering_candidates(query, spans, events, limit=limit)
        for candidate in fallback_candidates:
            candidate.metadata["persisted_graph_telemetry"] = persisted_telemetry
        return fallback_candidates

    def _event_ordering_shadow_coverage(
        self,
        candidate_lists: list[list[Candidate]],
        selected: list[Candidate],
    ) -> dict[str, Any]:
        all_candidates = [candidate for items in candidate_lists for candidate in items]
        graph_candidates = [
            candidate
            for candidate in all_candidates
            if _event_ordering_graph_candidate(candidate)
        ]
        legacy_candidates = [
            candidate
            for candidate in all_candidates
            if _event_ordering_legacy_candidate(candidate)
        ]
        selected_graph = [
            candidate
            for candidate in selected
            if _event_ordering_graph_candidate(candidate)
        ]
        selected_legacy = [
            candidate
            for candidate in selected
            if _event_ordering_legacy_candidate(candidate)
        ]
        selected_keys = {_event_ordering_candidate_path_key(candidate) for candidate in selected}
        dropped_graph = [
            candidate
            for candidate in graph_candidates
            if _event_ordering_candidate_path_key(candidate) not in selected_keys
        ]
        selected_driver = "none"
        selected_span_source = "none"
        if selected_graph:
            selected_driver = "graph"
            selected_span_source = "graph"
        elif selected_legacy:
            selected_driver = "legacy_fallback"
            selected_span_source = "legacy"
            _record_event_ordering_legacy_rescue_hits(selected_legacy)
        graph_payload = {
            "graph_candidate_count": len(graph_candidates),
            "legacy_candidate_count": len(legacy_candidates),
            "selected_count": len(selected_graph),
            "selected_sources": list(dict.fromkeys(candidate.source for candidate in selected_graph)),
            "selected_driver": selected_driver,
            "selected_span_source": selected_span_source,
            "graph_candidates_dropped_by_filters": bool(graph_candidates and dropped_graph),
            "dropped_count": len(dropped_graph),
        }
        if graph_candidates:
            graph_payload["available_sources"] = list(dict.fromkeys(candidate.source for candidate in graph_candidates))
            graph_payload.update(_event_ordering_graph_selector_coverage_telemetry(graph_candidates))
        shadow_payload = {
            "graph_candidate_count": len(graph_candidates),
            "legacy_candidate_count": len(legacy_candidates),
            "selected_graph_count": len(selected_graph),
            "selected_legacy_count": len(selected_legacy),
            "selected_driver": selected_driver,
            "selected_span_source": selected_span_source,
            "graph_candidates_dropped_by_filters": bool(graph_candidates and dropped_graph),
            "dropped_graph_count": len(dropped_graph),
        }
        out: dict[str, Any] = {"event_ordering_shadow": shadow_payload}
        if graph_candidates:
            out["event_ordering_graph"] = graph_payload
        return out

    def _event_ordering_dual_shadow_coverage(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        include_session: bool,
    ) -> dict[str, Any]:
        graph_candidates = self._event_ordering_graph_selector_candidates(
            query,
            scope,
            limit=limit,
            include_session=include_session,
        )
        legacy_candidates, legacy_sources = self._event_ordering_legacy_recall_for_shadow(
            query,
            scope,
            plan,
            limit,
            include_session,
        )
        return {
            "selected_driver": "dual_shadow",
            "graph_candidate_count": len(graph_candidates),
            "legacy_candidate_count": len(legacy_candidates),
            "candidate_count": min(
                limit,
                len(
                    {
                        _event_ordering_dual_shadow_candidate_key(candidate)
                        for candidate in graph_candidates
                    }
                    | {
                        _event_ordering_dual_shadow_candidate_key(candidate)
                        for candidate in legacy_candidates
                    }
                ),
            ),
            "sources": list(dict.fromkeys([candidate.source for candidate in graph_candidates] + legacy_sources)),
            "production_selector": getattr(self.retrieval_flags, "production_selector", "legacy"),
        }

    def _event_ordering_legacy_recall_for_shadow(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        include_session: bool,
    ) -> tuple[list[Candidate], list[str]]:
        candidates: list[Candidate] = []
        candidates.extend(
            self._event_ordering_episode_recall_candidates(
                query,
                scope,
                plan,
                limit=max(limit * 4, limit + 24),
                include_session=include_session,
            )
        )
        candidates.extend(
            self._event_ordering_timeline_candidates(
                query,
                plan,
                scope,
                limit=max(limit * 3, limit + 12),
                include_session=include_session,
            )
        )
        ordered = _dedupe_event_ordering_candidates_for_shadow(candidates)
        return ordered[:limit], [candidate.source for candidate in ordered[:limit]]

    def _event_ordering_coverage_candidates(
        self,
        query: str,
        scope: Scope,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"}
            and span.speaker in {"user", "assistant", "agent", "document"}
        ]
        if not spans:
            return []
        events = self.store.list_events(scope, include_session=include_session)
        return select_event_ordering_timeline(query, spans, events, limit=limit)

    def _event_ordering_timeline_candidates(
        self,
        query: str,
        plan: Any,
        scope: Scope,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"} and span.speaker in {"user", "assistant", "agent", "document"}
        ]
        topic_groups = _topic_scope_groups(query, plan, spans, max_groups=1)
        if topic_groups:
            spans = [span for span in spans if _span_group_key(span) in topic_groups]
        scored: list[Candidate] = []
        for span in spans:
            milestone_score = _event_ordering_milestone_score(span.content)
            query_overlap = keyword_score(query, span.content)
            topic_score = _topic_scope_score(query, span.content, plan)
            speaker_prior = 1.0 if span.speaker == "user" else 0.35 if span.speaker in {"assistant", "agent"} else 0.55
            if milestone_score <= 0 and query_overlap <= 0 and topic_score <= 0:
                continue
            score = (0.48 * milestone_score) + (0.18 * query_overlap) + (0.24 * topic_score) + (0.10 * speaker_prior)
            scored.append(
                Candidate(
                    id=span.span_id,
                    type="span",
                    text=span.content,
                    source="event_ordering_timeline",
                    scores={
                        "semantic_score": query_overlap,
                        "bm25_score": query_overlap,
                        "temporal_fit": 0.60 if milestone_score > 0 else 0.25,
                        "milestone_score": milestone_score,
                        "topic_scope_score": topic_score,
                        "topic_group": _span_group_key(span),
                        "speaker_prior": speaker_prior,
                        "score": score,
                    },
                    source_span_ids=[span.span_id],
                    metadata={
                        "speaker": span.speaker,
                        "timestamp": span.timestamp.isoformat(),
                        "turn_id": span.turn_id,
                        "source_uri": span.source_uri,
                        "milestone_score": milestone_score,
                    },
                )
            )
        scored.sort(
            key=lambda candidate: (
                candidate.scores.get("score", 0.0),
                candidate.scores.get("speaker_prior", 0.0),
                candidate.metadata.get("timestamp") or "",
            ),
            reverse=True,
        )
        return scored[:limit]

    def _event_ordering_episode_recall_candidates(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        requested = _event_ordering_requested_count_for_service(query) or 0
        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"}
            and span.speaker in {"user", "document"}
            and span.content
        ]
        if not spans:
            return []
        spans.sort(key=lambda span: (_natural_turn_key(span.source_uri), _natural_turn_key(span.turn_id), span.timestamp.isoformat(), span.span_id))
        facet_terms = _event_ordering_query_facets_for_service(query)
        anchor_terms = _event_ordering_anchor_terms_for_service(query)
        broad_scope = bool(re.search(r"\b(?:throughout|across|during|over|chats?|conversations?)\b", query, flags=re.I))
        scored: list[tuple[float, Candidate]] = []
        for span in spans:
            if _event_ordering_low_value_raw_facet_text(span.content):
                continue
            text_terms = _topic_scope_tokens(span.content)
            expanded_terms = set(text_terms)
            for term in list(text_terms):
                expanded_terms.update(_event_ordering_term_variants_for_service(term))
            facet_hits = facet_terms & expanded_terms
            anchor_hits = anchor_terms & expanded_terms
            topic_score = _topic_scope_score(query, span.content, plan)
            exact = _exact_overlap_score(query, span.content)
            episode_signal = _event_ordering_raw_episode_signal(span.content)
            detail_signal = _event_ordering_referenceable_detail_signal(span.content)
            support_option_signal = _event_ordering_support_option_signal(query, span.content)
            if not facet_hits and not anchor_hits and topic_score < 0.12 and exact < 0.18 and support_option_signal < 0.30:
                continue
            if not broad_scope and not anchor_hits and topic_score < 0.22 and support_option_signal < 0.30:
                continue
            facet_score = min(1.0, len(facet_hits) / max(1, len(facet_terms)))
            anchor_score = min(1.0, len(anchor_hits) / max(1, len(anchor_terms))) if anchor_terms else 0.0
            score = (0.30 * facet_score) + (0.22 * anchor_score)
            score += (
                0.18 * topic_score
                + 0.12 * exact
                + 0.12 * episode_signal
                + 0.06 * detail_signal
                + 0.12 * support_option_signal
            )
            if len(facet_hits) >= 2 and episode_signal >= 0.35:
                score += 0.10
            if facet_hits and support_option_signal >= 0.45:
                score += 0.08
            if score < 0.16:
                continue
            scored.append(
                (
                    score,
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source="event_ordering_episode_recall",
                        scores={
                            "semantic_score": max(topic_score, exact),
                            "bm25_score": max(exact, keyword_score(query, span.content)),
                            "exact_signal": min(1.0, exact + 0.10 * len(facet_hits)),
                            "topic_scope_score": topic_score,
                            "event_episode_signal": episode_signal,
                            "event_detail_signal": detail_signal,
                            "event_support_option_signal": support_option_signal,
                            "event_facet_coverage": min(1.0, len(facet_hits) / max(1, len(facet_terms))),
                            "score": min(1.0, score),
                        },
                        source_span_ids=[span.span_id],
                        metadata={
                            "speaker": span.speaker,
                            "span_type": span.span_type,
                            "timestamp": span.timestamp.isoformat(),
                            "source_uri": span.source_uri,
                            "turn_id": span.turn_id,
                            "topic_group": _span_group_key(span),
                            "event_ordering_episode_recall": True,
                            "event_ordering_facet_hits": sorted(facet_hits)[:12],
                            "event_ordering_anchor_hits": sorted(anchor_hits)[:8],
                        },
                    ),
                )
            )
        if not scored:
            return []
        return _event_ordering_select_episode_recall_candidates(scored, limit, requested)

    def _event_ordering_event_candidates(
        self,
        query: str,
        scope: Scope,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[tuple[MemoryEvent, dict[str, float]]]:
        events = [
            event
            for event in self.store.list_events(scope, include_session=include_session)
            if event.event_type == "milestone" or _event_milestone_group(event)
        ]
        topic_groups = _topic_scope_groups(query, self.planner.plan(query), self.store.list_spans(scope, include_session=include_session), max_groups=1)
        if topic_groups:
            events = [
                event
                for event in events
                if any(
                    _span_group_key(span)
                    in topic_groups
                    for span in (self.store.get_span(span_id) for span_id in event.source_span_ids)
                    if span
                )
            ]
        events.sort(key=self._event_ordering_sort_key)
        selected_events = _select_event_ordering_representatives(query, events, limit, self._event_ordering_sort_key)
        if selected_events:
            return [
                (
                    event,
                    {
                        "bm25_score": max(0.20, keyword_score(query, event.description)),
                        "temporal_fit": 0.95 if event.time_start else 0.55,
                        "graph_proximity": 0.95,
                        "milestone_score": 1.0,
                        "score": 1.0,
                    },
                )
                for event in selected_events
            ]
        selected: list[tuple[MemoryEvent, dict[str, float]]] = []
        seen_groups: set[str] = set()
        for event in events:
            group = _event_milestone_group(event) or "milestone"
            if group in seen_groups and len(seen_groups) < 5:
                continue
            seen_groups.add(group)
            selected.append(
                (
                    event,
                    {
                        "bm25_score": 0.20,
                        "temporal_fit": 0.90 if event.time_start else 0.55,
                        "graph_proximity": 0.90,
                        "milestone_score": 1.0,
                        "score": 1.0,
                    },
                )
            )
            if len(selected) >= limit:
                break
        return selected

    def _contradiction_claim_candidates(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"} and span.speaker in {"user", "assistant", "agent", "document"}
        ]
        groups = _topic_scope_groups(query, plan, spans, max_groups=2)
        if groups:
            spans = [span for span in spans if _span_group_key(span) in groups]
        buckets: dict[str, list[Candidate]] = {"positive": [], "negative": [], "uncertain": []}
        for span in spans:
            topic_score = _topic_scope_score(query, span.content, plan)
            exact = _exact_overlap_score(query, span.content)
            polarity = _surface_claim_polarity(query, span.content)
            if topic_score <= 0 and exact <= 0:
                continue
            polarity_bonus = 0.28 if polarity in {"positive", "negative"} else 0.08
            speaker_prior = 0.40 if span.speaker == "user" else 0.24
            score = (0.48 * topic_score) + (0.24 * exact) + polarity_bonus + (0.06 * speaker_prior)
            buckets[polarity].append(
                Candidate(
                    id=span.span_id,
                    type="span",
                    text=span.content,
                    source=f"contradiction_claim_{polarity}",
                    scores={
                        "semantic_score": max(topic_score, exact),
                        "bm25_score": exact,
                        "exact_signal": min(1.0, exact + polarity_bonus),
                        "topic_scope_score": topic_score,
                        "claim_polarity_score": polarity_bonus,
                        "score": score,
                    },
                    source_span_ids=[span.span_id],
                    metadata={
                        "speaker": span.speaker,
                        "span_type": span.span_type,
                        "timestamp": span.timestamp.isoformat(),
                        "source_uri": span.source_uri,
                        "turn_id": span.turn_id,
                        "topic_group": _span_group_key(span),
                        "claim_polarity": polarity,
                    },
                )
            )
        for items in buckets.values():
            items.sort(
                key=lambda candidate: (
                    candidate.scores.get("score", 0.0),
                    _natural_turn_key(candidate.metadata.get("source_uri")),
                    _natural_turn_key(candidate.metadata.get("turn_id")),
                ),
                reverse=True,
            )
        out: list[Candidate] = []
        per_bucket = max(2, limit // 3)
        for polarity in ("positive", "negative", "uncertain"):
            out.extend(buckets[polarity][:per_bucket])
        out.sort(key=lambda candidate: candidate.scores.get("score", 0.0), reverse=True)
        return out[:limit]

    def _temporal_coverage_candidates(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"} and span.speaker in {"user", "assistant", "agent", "document"}
        ]
        groups = _topic_scope_groups(query, plan, spans, max_groups=_topic_scope_group_limit(plan.query_type))
        target_roles = _temporal_target_roles_for_service(query)
        focus_terms = _temporal_focus_terms_for_service(query)
        wants_duration = bool(re.search(r"\b(?:how many days|how long|did it take|duration)\b", query.lower()))
        scored: list[Candidate] = []
        for span in spans:
            if groups and _span_group_key(span) not in groups:
                continue
            lower = span.content.lower()
            roles = _temporal_roles_in_text(query, span.content)
            role_hits = roles & target_roles
            focus_hits = focus_terms & _topic_scope_tokens(span.content)
            date_signal = _date_signal(span.content)
            duration_signal = 1.0 if re.search(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?)\b", lower) else 0.0
            if not role_hits and not (wants_duration and duration_signal and focus_hits):
                continue
            if target_roles and not focus_hits and span.speaker not in {"user", "document"}:
                continue
            focus_score = len(focus_hits) / max(1, len(focus_terms)) if focus_terms else 0.0
            role_score = len(role_hits) / max(1, len(target_roles)) if target_roles else 0.0
            speaker_prior = 0.24
            if span.speaker == "user":
                speaker_prior = 0.60
            elif span.speaker == "document":
                speaker_prior = 0.48
            score = (
                0.36 * role_score
                + 0.30 * focus_score
                + 0.16 * date_signal
                + 0.10 * duration_signal
                + 0.08 * speaker_prior
            )
            if len(focus_hits) >= 2 and role_hits:
                score += 0.18
            if wants_duration and duration_signal and len(focus_hits) >= 2:
                score += 0.16
            if score <= 0.16:
                continue
            scored.append(
                Candidate(
                    id=span.span_id,
                    type="span",
                    text=span.content,
                    source="temporal_coverage_raw",
                    scores={
                        "semantic_score": max(focus_score, role_score),
                        "bm25_score": focus_score,
                        "exact_signal": min(1.0, focus_score + role_score),
                        "topic_scope_score": _topic_scope_score(query, span.content, plan),
                        "temporal_fit": min(1.0, role_score + date_signal + duration_signal),
                        "temporal_focus_score": focus_score,
                        "temporal_role_score": role_score,
                        "score": min(1.0, score),
                    },
                    source_span_ids=[span.span_id],
                    metadata={
                        "speaker": span.speaker,
                        "span_type": span.span_type,
                        "timestamp": span.timestamp.isoformat(),
                        "source_uri": span.source_uri,
                        "turn_id": span.turn_id,
                        "topic_group": _span_group_key(span),
                        "temporal_coverage": True,
                        "temporal_roles": sorted(roles),
                        "temporal_focus_terms": sorted(focus_hits),
                    },
                )
            )
        scored.sort(
            key=lambda candidate: (
                candidate.scores.get("score", 0.0),
                1.0 if candidate.metadata.get("speaker") == "user" else 0.0,
                candidate.scores.get("temporal_focus_score", 0.0),
                _natural_turn_key(candidate.metadata.get("source_uri")),
                _natural_turn_key(candidate.metadata.get("turn_id")),
            ),
            reverse=True,
        )
        return scored[:limit]

    def _event_ordering_sort_key(self, event: MemoryEvent) -> tuple[int, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...], str, str]:
        span = self.store.get_span(event.source_span_ids[0]) if event.source_span_ids else None
        if span:
            return (
                0,
                _natural_turn_key(span.source_uri),
                _natural_turn_key(span.turn_id),
                span.timestamp.isoformat(),
                event.event_id,
            )
        return (
            1,
            (),
            (),
            event.time_start.isoformat() if event.time_start else "",
            event.event_id,
        )

    def _event_ordering_observed_at(self, event: MemoryEvent) -> datetime | None:
        span = self.store.get_span(event.source_span_ids[0]) if event.source_span_ids else None
        return span.timestamp if span else event.time_start

    def _topic_scoped_raw_candidates(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        if plan.query_type == "abstention":
            return []
        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"} and span.speaker in {"user", "assistant", "agent", "document"}
        ]
        groups = _topic_scope_groups(query, plan, spans, max_groups=_topic_scope_group_limit(plan.query_type))
        if not groups:
            return []
        target_roles = set(_temporal_target_roles_for_service(query)) if plan.query_type == "temporal_lookup" else set()
        scored: list[tuple[Candidate, tuple[float, float, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]]] = []
        for span in spans:
            group = _span_group_key(span)
            if group not in groups:
                continue
            topic_score = _topic_scope_score(query, span.content, plan)
            keyword = keyword_score(query, span.content)
            date_signal = _date_signal(span.content)
            exact_terms = _exact_query_terms(query)
            exact_hit_ratio = sum(1 for term in exact_terms if term.lower() in span.content.lower()) / max(1, len(exact_terms))
            role_signal = 0.0
            if target_roles:
                roles = _temporal_roles_in_text(query, span.content)
                if roles & target_roles:
                    role_signal = 0.45 + 0.15 * min(len(roles & target_roles), 2)
            speaker_prior = 0.20
            if span.speaker == "user":
                speaker_prior = 0.45
            elif span.speaker in {"assistant", "agent"}:
                speaker_prior = 0.28
            if plan.query_type == "event_ordering" and span.speaker == "user":
                speaker_prior = 0.65
            synthesis_signal = 0.0
            if plan.query_type == "multi_session_reasoning":
                synthesis_signal = _synthesis_evidence_signal(query, span.content)
            score = (
                0.46 * topic_score
                + 0.18 * keyword
                + 0.14 * exact_hit_ratio
                + 0.12 * role_signal
                + 0.06 * date_signal
                + 0.04 * speaker_prior
            )
            if plan.query_type == "multi_session_reasoning":
                score += 0.16 * synthesis_signal
            if plan.query_type == "summarization":
                score = max(score, topic_score * 0.70 + speaker_prior * 0.10)
            if score <= 0.05:
                continue
            candidate = Candidate(
                id=span.span_id,
                type="span",
                text=span.content,
                source="topic_scope_raw",
                scores={
                    "semantic_score": max(topic_score, keyword),
                    "bm25_score": max(keyword, exact_hit_ratio),
                    "exact_signal": min(1.0, exact_hit_ratio + role_signal),
                    "topic_scope_score": topic_score,
                    "topic_group_prior": 0.85,
                    "temporal_fit": max(date_signal, role_signal),
                    "speaker_prior": speaker_prior,
                    "synthesis_signal": synthesis_signal,
                    "score": score,
                },
                source_span_ids=[span.span_id],
                metadata={
                    "speaker": span.speaker,
                    "span_type": span.span_type,
                    "timestamp": span.timestamp.isoformat(),
                    "source_uri": span.source_uri,
                    "turn_id": span.turn_id,
                    "topic_group": group,
                    "topic_scope_score": topic_score,
                    "synthesis_signal": synthesis_signal,
                },
            )
            scored.append((candidate, (score, topic_score, _natural_turn_key(span.source_uri), _natural_turn_key(span.turn_id))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [candidate for candidate, _ in scored[:limit]]

    def _raw_scent_trail_candidates(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        seed_candidates: list[Candidate],
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        if limit <= 0 or plan.query_type == "abstention":
            return []
        if plan.query_type not in {
            "factual_exact",
            "instruction",
            "summarization",
            "multi_session_reasoning",
            "knowledge_update",
            "temporal_lookup",
            "contradiction_resolution",
        }:
            return []
        seed_texts = [candidate.text for candidate in seed_candidates if candidate.type == "span" and candidate.text]
        trail_queries = _scent_trail_queries(query, seed_texts)
        if not trail_queries:
            return []
        out: list[Candidate] = []
        seen: set[str] = set()
        per_query_limit = max(4, min(10, limit))
        for trail_query in trail_queries:
            for span, scores in self.store.search_spans(
                trail_query,
                scope,
                limit=per_query_limit,
                include_session=include_session,
            ):
                if span.span_id in seen:
                    continue
                if span.span_type not in {"turn", "tool_result", "document_chunk"}:
                    continue
                topic_score = _topic_scope_score(query, span.content, plan)
                exact = _exact_overlap_score(query, span.content)
                trail_score = _scent_trail_score(trail_query, span.content)
                if max(topic_score, exact, trail_score) <= 0.08:
                    continue
                score = (0.42 * trail_score) + (0.28 * topic_score) + (0.18 * exact) + (0.12 * float(scores.get("score", 0.0)))
                out.append(
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source="raw_scent_trail",
                        scores={
                            **scores,
                            "semantic_score": max(float(scores.get("semantic_score", 0.0)), topic_score),
                            "bm25_score": max(float(scores.get("bm25_score", 0.0)), exact, trail_score),
                            "topic_scope_score": topic_score,
                            "trail_score": trail_score,
                            "score": score,
                        },
                        source_span_ids=[span.span_id],
                        metadata={
                            "speaker": span.speaker,
                            "span_type": span.span_type,
                            "timestamp": span.timestamp.isoformat(),
                            "source_uri": span.source_uri,
                            "turn_id": span.turn_id,
                            "topic_group": _span_group_key(span),
                            "trail_query": trail_query,
                        },
                    )
                )
                seen.add(span.span_id)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        out.sort(
            key=lambda candidate: (
                candidate.scores.get("score", 0.0),
                candidate.metadata.get("speaker") == "user",
                _natural_turn_key(candidate.metadata.get("source_uri")),
                _natural_turn_key(candidate.metadata.get("turn_id")),
            ),
            reverse=True,
        )
        return out[:limit]

    def _broad_raw_recall_candidates(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        if limit <= 0 or plan.query_type == "abstention":
            return []
        recall_queries = _broad_raw_recall_queries(query, plan)
        if not recall_queries:
            return []
        speaker = plan.speaker_focus if plan.speaker_focus != "any" else None
        per_query_limit = max(6, min(18, limit // max(1, len(recall_queries)) + 4))
        best: dict[str, Candidate] = {}
        for recall_index, recall_query in enumerate(recall_queries):
            for span, scores in self.store.search_spans(
                recall_query,
                scope,
                limit=per_query_limit,
                speaker=speaker,
                include_session=include_session,
            ):
                if span.span_type not in {"turn", "tool_result", "document_chunk"}:
                    continue
                topic_score = _topic_scope_score(query, span.content, plan)
                exact = _exact_overlap_score(query, span.content)
                recall_overlap = _exact_overlap_score(recall_query, span.content)
                salience = _fallback_salience_score(span.content)
                intent_signal = _intent_recall_signal(query, plan, span.content)
                if max(topic_score, exact, recall_overlap, intent_signal) <= 0.035:
                    continue
                sparse = float(scores.get("bm25_score", 0.0) or 0.0)
                dense = float(scores.get("semantic_score", 0.0) or 0.0)
                source_score = float(scores.get("score", 0.0) or 0.0)
                speaker_prior = 0.36 if span.speaker == "user" else 0.24 if span.speaker in {"assistant", "agent"} else 0.30
                if plan.query_type == "event_ordering" and span.speaker == "user":
                    speaker_prior = 0.62
                score = (
                    0.24 * max(topic_score, exact)
                    + 0.22 * recall_overlap
                    + 0.18 * intent_signal
                    + 0.14 * min(1.0, source_score)
                    + 0.10 * min(1.0, dense)
                    + 0.07 * min(1.0, sparse)
                    + 0.05 * salience
                    + 0.03 * speaker_prior
                )
                if plan.query_type in {"knowledge_update", "temporal_lookup", "event_ordering"}:
                    score += 0.05 * _date_signal(span.content)
                candidate = Candidate(
                    id=span.span_id,
                    type="span",
                    text=span.content,
                    source="broad_raw_recall",
                    scores={
                        **scores,
                        "semantic_score": max(dense, topic_score, intent_signal),
                        "bm25_score": max(sparse, exact, recall_overlap),
                        "exact_signal": max(exact, recall_overlap),
                        "topic_scope_score": topic_score,
                        "intent_recall_signal": intent_signal,
                        "recall_query_overlap": recall_overlap,
                        "salience_score": salience,
                        "speaker_prior": speaker_prior,
                        "score": min(1.0, score),
                    },
                    source_span_ids=[span.span_id],
                    metadata={
                        "speaker": span.speaker,
                        "span_type": span.span_type,
                        "timestamp": span.timestamp.isoformat(),
                        "source_uri": span.source_uri,
                        "turn_id": span.turn_id,
                        "topic_group": _span_group_key(span),
                        "recall_query": recall_query,
                        "recall_query_index": recall_index,
                        "broad_raw_recall": True,
                    },
                )
                matched_conditions = _matched_query_conditions(query, span.content)
                if matched_conditions:
                    candidate = Candidate(
                        id=candidate.id,
                        type=candidate.type,
                        text=candidate.text,
                        source=candidate.source,
                        scores=candidate.scores,
                        source_span_ids=candidate.source_span_ids,
                        metadata={**candidate.metadata, "matched_conditions": matched_conditions},
                    )
                previous = best.get(span.span_id)
                if previous is None or candidate.scores.get("score", 0.0) > previous.scores.get("score", 0.0):
                    best[span.span_id] = candidate
        out = list(best.values())
        out.sort(
            key=lambda candidate: (
                candidate.scores.get("score", 0.0),
                candidate.scores.get("intent_recall_signal", 0.0),
                candidate.scores.get("topic_scope_score", 0.0),
                candidate.metadata.get("speaker") == "user",
                _natural_turn_key(candidate.metadata.get("source_uri")),
                _natural_turn_key(candidate.metadata.get("turn_id")),
            ),
            reverse=True,
        )
        return out[:limit]

    def _aggregation_coverage_candidates(
        self,
        query: str,
        scope: Scope,
        plan: Any,
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        spans = [
            span
            for span in self.store.list_spans(scope, include_session=include_session)
            if span.span_type in {"turn", "tool_result", "document_chunk"} and span.speaker in {"user", "assistant", "agent", "document"}
        ]
        groups = _topic_scope_groups(query, plan, spans, max_groups=_topic_scope_group_limit(plan.query_type))
        if not groups:
            return []
        query_terms = _aggregation_query_terms(query)
        scored: list[tuple[float, Candidate, tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]] = []
        ordered_spans = sorted(spans, key=lambda span: (_natural_turn_key(span.source_uri), _natural_turn_key(span.turn_id), span.timestamp.isoformat(), span.span_id))
        adjacent_assistant_by_user_id = _adjacent_assistant_recommendation_spans(query, ordered_spans)
        for span in spans:
            group = _span_group_key(span)
            if group not in groups:
                continue
            signal = _aggregation_signal(query, span.content, query_terms)
            if signal <= 0:
                continue
            focus = _aggregation_focus_priority(query, span.content)
            topic_score = _topic_scope_score(query, span.content, plan)
            exact = _exact_overlap_score(query, span.content)
            speaker_prior = 0.40 if span.speaker == "user" else 0.16
            score = 0.46 * signal + 0.20 * topic_score + 0.14 * exact + 0.10 * speaker_prior + 0.10 * focus
            candidate = Candidate(
                id=span.span_id,
                type="span",
                text=span.content,
                source="aggregation_coverage_raw",
                scores={
                    "semantic_score": max(topic_score, signal),
                    "bm25_score": max(keyword_score(query, span.content), exact),
                    "exact_signal": min(1.0, exact + signal),
            "aggregation_signal": signal,
            "aggregation_focus": focus,
            "synthesis_signal": _synthesis_evidence_signal(query, span.content),
            "topic_scope_score": topic_score,
                    "topic_group_prior": 0.90,
                    "speaker_prior": speaker_prior,
                    "score": score,
                },
                source_span_ids=[span.span_id],
                metadata={
                    "speaker": span.speaker,
                    "span_type": span.span_type,
                    "timestamp": span.timestamp.isoformat(),
                    "source_uri": span.source_uri,
                    "turn_id": span.turn_id,
                    "topic_group": group,
                    "aggregation_coverage": True,
                    "aggregation_signal": signal,
                    "aggregation_focus": focus,
                    "synthesis_signal": _synthesis_evidence_signal(query, span.content),
                    "aggregation_keys": [
                        *_aggregation_keys(query, span.content, speaker=span.speaker),
                        *_aggregation_query_context_keys(query, span.content),
                    ],
                },
            )
            scored.append((score, candidate, _natural_turn_key(span.source_uri), _natural_turn_key(span.turn_id)))
            for support_span in adjacent_assistant_by_user_id.get(span.span_id, []):
                if _span_group_key(support_span) not in groups:
                    continue
                support_signal = _assistant_recommendation_list_signal(query, support_span.content)
                if support_signal <= 0:
                    continue
                request_specificity = _recommendation_request_specificity(span.content)
                support_topic_score = max(topic_score, _topic_scope_score(query, support_span.content, plan))
                support_exact = _exact_overlap_score(query, support_span.content)
                support_score = max(score * 0.92, 0.38 + 0.24 * support_signal + 0.18 * support_topic_score + 0.10 * support_exact + 0.18 * request_specificity)
                support_candidate = Candidate(
                    id=support_span.span_id,
                    type="span",
                    text=support_span.content,
                    source="aggregation_context_support",
                    scores={
                        "semantic_score": max(support_topic_score, support_signal),
                        "bm25_score": max(keyword_score(query, support_span.content), support_exact),
                        "exact_signal": min(1.0, support_exact + support_signal),
                        "aggregation_signal": support_signal,
                        "aggregation_focus": max(focus, 0.55),
                        "request_specificity": request_specificity,
                        "synthesis_signal": _synthesis_evidence_signal(query, support_span.content),
                        "topic_scope_score": support_topic_score,
                        "topic_group_prior": 0.90,
                        "speaker_prior": 0.18,
                        "score": min(1.0, support_score),
                    },
                    source_span_ids=[support_span.span_id],
                    metadata={
                        "speaker": support_span.speaker,
                        "span_type": support_span.span_type,
                        "timestamp": support_span.timestamp.isoformat(),
                        "source_uri": support_span.source_uri,
                        "turn_id": support_span.turn_id,
                        "topic_group": _span_group_key(support_span),
                        "aggregation_coverage": True,
                        "aggregation_context_support": True,
                        "supports_request_span_id": span.span_id,
                        "aggregation_signal": support_signal,
                        "aggregation_focus": max(focus, 0.55),
                        "synthesis_signal": _synthesis_evidence_signal(query, support_span.content),
                        "aggregation_keys": [
                            f"group_support:{span.span_id}",
                            *_aggregation_query_context_keys(query, support_span.content),
                        ],
                    },
                )
                scored.append(
                    (
                        support_candidate.scores["score"],
                        support_candidate,
                        _natural_turn_key(support_span.source_uri),
                        _natural_turn_key(support_span.turn_id),
                    )
                )
        scored.sort(key=lambda item: (item[0], item[2], item[3]), reverse=True)
        return _key_diverse_aggregation_candidates(scored, limit)

    def _retrieval_query(self, query: str, plan: Any, source: str) -> str:
        hints = [hint for hint in getattr(plan, "retrieval_hints", []) if hint]
        if not hints:
            return query
        if plan.query_type == "event_ordering" and source in {"events", "facts", "entities", "exact"}:
            return " ".join(hints)
        if plan.query_type == "temporal_lookup" and source in {"events", "facts"}:
            return " ".join(hints)
        if plan.query_type == "factual_exact" and source == "profiles":
            return " ".join(hints)
        return query

    def _plan_uses_source(self, plan: Any, source: str) -> bool:
        if source == "raw":
            return True
        if source == "events":
            return plan.query_type in {"event_ordering", "temporal_lookup", "contradiction_resolution", "knowledge_update", "assistant_reference"}
        if source == "views":
            return plan.needs_current_state or plan.query_type in {"preference", "instruction", "knowledge_update"}
        if source == "profiles":
            return plan.query_type in {"preference", "instruction"}
        if source == "facts":
            if plan.query_type == "event_ordering":
                return False
            return plan.query_type != "abstention"
        return True

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

    def _exact_candidates(self, query: str, scope: Scope, limit: int, *, plan: Any | None = None, include_session: bool = False) -> list[Candidate]:
        spans = self.store.list_spans(scope, include_session=include_session)
        terms = _exact_query_terms(query)
        query_lower = query.lower()
        query_has_value_intent = _has_value_intent(query_lower)
        query_has_current_intent = _has_current_intent(query_lower) or getattr(plan, "query_type", None) in {"knowledge_update", "temporal_lookup"}
        query_is_aggregation = getattr(plan, "query_type", None) == "multi_session_reasoning"
        out: list[Candidate] = []
        for span in spans:
            lower = span.content.lower()
            hits = sum(1 for term in terms if term.lower() in lower)
            compatible_value = _compatible_value_mention(query_lower, lower) if query_has_value_intent else True
            value_signal = _value_signal(query_lower, lower) if query_has_value_intent and compatible_value else 0.0
            current_signal = _current_state_signal(lower) if query_has_current_intent else 0.0
            code_signal = _code_identifier_signal(query_lower, lower)
            hit_ratio = hits / max(1, len(terms))
            exact_signal = hit_ratio + value_signal + current_signal + code_signal
            if query_has_value_intent and not compatible_value:
                exact_signal = min(exact_signal, 0.55)
            value_exact_signal = min(1.0, hit_ratio + value_signal + current_signal + code_signal) if value_signal > 0 else 0.0
            if exact_signal <= 0:
                continue
            score = min(1.0, exact_signal)
            aggregation_keys = _aggregation_keys(query, span.content, speaker=span.speaker) if query_is_aggregation else []
            metadata = {
                "speaker": span.speaker,
                "span_type": span.span_type,
                "summary": compact_summary(span.content),
                "exact_signal": score,
                "value_exact_signal": value_exact_signal,
                "value_query": query_has_value_intent,
                "exact_hit_ratio": hit_ratio,
                "value_signal": value_signal,
                "current_signal": current_signal,
                "source_uri": span.source_uri,
                "turn_id": span.turn_id,
                **({"aggregation_keys": aggregation_keys} if aggregation_keys else {}),
            }
            cjk_matches = _cjk_exact_match_phrases(query, span.content)
            if cjk_matches:
                reasons = list(metadata.get("must_preserve_reason") or [])
                if "language_exact_match" not in reasons:
                    reasons.append("language_exact_match")
                metadata["must_preserve_reason"] = reasons
                metadata["language_match"] = "exact"
                metadata["language_exact_phrases"] = cjk_matches
            out.append(
                Candidate(
                    id=span.span_id,
                    type="span",
                    text=span.content,
                    source="exact_filter",
                    scores={
                        "bm25_score": score,
                        "semantic_score": score,
                        "exact_signal": score,
                        "value_exact_signal": value_exact_signal,
                        "score": score,
                        "exact_hit_ratio": hit_ratio,
                        "value_signal": value_signal,
                        "current_signal": current_signal,
                    },
                    source_span_ids=[span.span_id],
                    metadata=metadata,
                )
            )
        out.sort(
            key=lambda candidate: (
                candidate.scores.get("value_exact_signal", 0.0),
                candidate.scores["exact_signal"],
                _natural_turn_key(candidate.metadata.get("source_uri")),
                _natural_turn_key(candidate.metadata.get("turn_id")),
            ),
            reverse=True,
        )
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

    def _preserve_high_signal_exact(self, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        required = [
            candidate
            for candidate in candidates
            if candidate.type == "span"
            and (
                (candidate.metadata.get("value_query") and candidate.scores.get("value_exact_signal", 0.0) >= 0.75)
                or (
                    not candidate.metadata.get("value_query")
                    and
                    candidate.scores.get("exact_signal", 0.0) >= 0.90
                )
            )
        ]
        required.sort(
            key=lambda candidate: (
                candidate.scores.get("value_exact_signal", 0.0),
                candidate.metadata.get("current_signal", 0.0),
                candidate.metadata.get("exact_hit_ratio", 0.0),
                candidate.metadata.get("value_signal", 0.0),
                candidate.scores.get("utility_score", 0.0),
            ),
            reverse=True,
        )
        required = required[: max(1, min(3, limit // 3))]
        if not required:
            return selected
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in required + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _preserve_temporal_coverage(self, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        required = [
            candidate
            for candidate in candidates
            if candidate.type == "span"
            and candidate.metadata.get("temporal_coverage")
            and (
                candidate.metadata.get("speaker") in {"user", "document"}
                or candidate.scores.get("temporal_focus_score", 0.0) >= 0.45
            )
            and candidate.scores.get("temporal_focus_score", 0.0) >= 0.20
        ]
        required.sort(
            key=lambda candidate: (
                1.0 if candidate.metadata.get("speaker") == "user" else 0.0,
                candidate.scores.get("temporal_focus_score", 0.0),
                candidate.scores.get("temporal_role_score", 0.0),
                candidate.scores.get("temporal_fit", 0.0),
                candidate.scores.get("utility_score", 0.0),
            ),
            reverse=True,
        )
        diverse: list[Candidate] = []
        seen_focus: set[str] = set()
        for candidate in required:
            focus_terms = tuple(candidate.metadata.get("temporal_focus_terms") or [])
            focus_key = "_".join(focus_terms[:3]) or candidate.id
            if focus_key in seen_focus:
                continue
            seen_focus.add(focus_key)
            diverse.append(candidate)
            if len(diverse) >= max(2, min(6, limit // 3)):
                break
        if not diverse:
            return selected
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in diverse + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _preserve_contradiction_claim_coverage(self, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        if limit <= 0:
            return selected
        required: list[Candidate] = []
        for polarity in ("positive", "negative"):
            bucket = [
                candidate
                for candidate in candidates
                if candidate.type == "span"
                and candidate.metadata.get("claim_polarity") == polarity
                and str(candidate.source).startswith("contradiction_claim_")
                and (
                    candidate.metadata.get("speaker") == "user"
                    or candidate.scores.get("topic_scope_score", 0.0) >= 0.55
                    or candidate.scores.get("exact_signal", 0.0) >= 0.75
                )
            ]
            bucket.sort(
                key=lambda candidate: (
                    1.0 if candidate.metadata.get("speaker") == "user" else 0.0,
                    candidate.scores.get("topic_scope_score", 0.0),
                    candidate.scores.get("exact_signal", 0.0),
                    candidate.scores.get("claim_polarity_score", 0.0),
                    candidate.scores.get("utility_score", candidate.scores.get("score", 0.0)),
                ),
                reverse=True,
            )
            required.extend(bucket[:2])
        if not required:
            return selected
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in required + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _filter_stale_current_value_candidates(self, query: str, plan: Any, selected: list[Candidate]) -> list[Candidate]:
        if not selected:
            return selected
        query_lower = query.lower()
        if re.search(r"\b(?:average|mean|median|trend|history|historical|over time)\b", query_lower) or re.search(r"平均|趋势|历史|变化", query_lower):
            return selected
        intent = getattr(plan, "intent", {}) if isinstance(getattr(plan, "intent", {}), dict) else {}
        if not (
            getattr(plan, "needs_current_state", False)
            or getattr(plan, "query_type", "") == "knowledge_update"
            or bool(intent.get("needs_current_state"))
            or "currently" in query_lower
            or "current" in query_lower
            or "现在" in query_lower
            or "当前" in query_lower
        ):
            return selected
        scored_candidates: list[tuple[Candidate, float]] = []
        for candidate in selected:
            current_score = float(candidate.metadata.get("current_signal", 0.0) or 0.0)
            if candidate.text:
                current_score = max(current_score, _current_state_signal(candidate.text.lower()))
            if candidate.type == "view" and candidate.metadata.get("current_value"):
                current_score = max(current_score, 1.0)
            scored_candidates.append((candidate, current_score))
        best_current = max((score for _candidate, score in scored_candidates), default=0.0)
        if best_current <= 0.0:
            return selected
        stale = [
            candidate
            for candidate, current_score in scored_candidates
            if _stale_current_value_candidate_text(candidate.text) or current_score + 0.05 < best_current
        ]
        if not stale:
            return selected
        for candidate in stale:
            record_rule_hit(
                "current_value.stale_history_marker",
                query=query,
                text=candidate.text,
                stage="search_filter",
                contributed_candidate_id=candidate.id,
                metadata={"decision": "drop_stale_history", "source": candidate.source},
            )
        kept = [candidate for candidate in selected if candidate not in stale]
        return kept if kept else selected

    def _preserve_scent_trail(self, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        required = [
            candidate
            for candidate in candidates
            if candidate.type == "span"
            and _candidate_has_source(candidate, "raw_scent_trail")
            and candidate.scores.get("trail_score", 0.0) >= 0.20
        ]
        required.sort(
            key=lambda candidate: (
                candidate.scores.get("trail_score", 0.0),
                candidate.scores.get("topic_scope_score", 0.0),
                candidate.scores.get("utility_score", 0.0),
            ),
            reverse=True,
        )
        required = required[: max(1, min(3, limit // 4))]
        if not required:
            return selected
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in required + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _preserve_event_ordering_raw_facets(self, query: str, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        requested = _event_ordering_requested_count_for_service(query)
        if requested is None or requested <= 0:
            return selected
        facet_terms = _event_ordering_query_facets_for_service(query)
        if not facet_terms:
            return selected
        broad_episode_recall = _event_ordering_support_option_query(query)
        pool = [
            candidate
            for candidate in candidates
            if candidate.type == "span"
            and candidate.metadata.get("speaker") in {"user", "document"}
            and candidate.text
        ]
        scored: list[tuple[float, Candidate]] = []
        for candidate in pool:
            text_terms = _topic_scope_tokens(candidate.text)
            expanded_text_terms = set(text_terms)
            for term in list(text_terms):
                expanded_text_terms.update(_event_ordering_term_variants_for_service(term))
            facet_hits = facet_terms & expanded_text_terms
            is_episode_recall = "event_ordering_episode_recall" in candidate.source
            strong_episode_recall = is_episode_recall and (
                candidate.scores.get("event_episode_signal", 0.0) >= 0.35
                or candidate.scores.get("event_detail_signal", 0.0) >= 0.36
                or candidate.scores.get("event_support_option_signal", 0.0) >= 0.30
                or candidate.scores.get("event_facet_coverage", 0.0) >= 0.16
            )
            if not facet_hits and not strong_episode_recall:
                continue
            if _event_ordering_low_value_raw_facet_candidate(candidate):
                continue
            score = (
                0.28 * min(1.0, len(facet_hits) / max(1, len(facet_terms)))
                + 0.20 * float(candidate.scores.get("topic_scope_score", 0.0) or 0.0)
                + 0.20 * float(candidate.scores.get("exact_signal", 0.0) or 0.0)
                + 0.16 * float(candidate.scores.get("intent_recall_signal", 0.0) or 0.0)
                + 0.10 * _event_ordering_raw_episode_signal(candidate.text)
                + 0.08 * float(candidate.scores.get("event_support_option_signal", 0.0) or 0.0)
                + 0.06 * float(candidate.scores.get("utility_score", candidate.scores.get("score", 0.0)) or 0.0)
            )
            if strong_episode_recall:
                score = max(score, 0.18 + 0.12 * candidate.scores.get("event_detail_signal", 0.0))
            if candidate.metadata.get("broad_raw_recall"):
                score -= 0.05
            if score >= 0.16:
                scored.append((score, candidate))
        if not scored:
            return selected
        episode_required = _event_ordering_select_episode_recall_candidates(
            [
                (score, candidate)
                for score, candidate in scored
                if "event_ordering_episode_recall" in candidate.source
                and _event_ordering_raw_facet_episode_required(candidate, broad_episode_recall=broad_episode_recall)
            ],
            max(1, min(requested + 2 if broad_episode_recall else 2, limit // 2)),
            requested,
        )
        scored.sort(
            key=lambda item: (
                item[0],
                item[1].metadata.get("broad_raw_recall") is not True,
                _natural_turn_key(item[1].metadata.get("source_uri")),
                _natural_turn_key(item[1].metadata.get("turn_id")),
            ),
            reverse=True,
        )
        required: list[Candidate] = []
        seen_ids: set[tuple[str, str]] = set()
        seen_turns: set[tuple[tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...]]] = set()
        for candidate in selected:
            if not _event_ordering_episode_recall_candidate(candidate):
                continue
            if not broad_episode_recall and candidate.scores.get("event_support_option_signal", 0.0) < 0.30:
                continue
            key = (candidate.type, candidate.id)
            turn_key = (
                _natural_turn_key(candidate.metadata.get("source_uri")),
                _natural_turn_key(candidate.metadata.get("turn_id")),
            )
            required.append(candidate)
            seen_ids.add(key)
            seen_turns.add(turn_key)
        for candidate in episode_required:
            key = (candidate.type, candidate.id)
            turn_key = (
                _natural_turn_key(candidate.metadata.get("source_uri")),
                _natural_turn_key(candidate.metadata.get("turn_id")),
            )
            if key in seen_ids or turn_key in seen_turns:
                continue
            required.append(candidate)
            seen_ids.add(key)
            seen_turns.add(turn_key)
        for _score, candidate in scored:
            key = (candidate.type, candidate.id)
            turn_key = (
                _natural_turn_key(candidate.metadata.get("source_uri")),
                _natural_turn_key(candidate.metadata.get("turn_id")),
            )
            if key in seen_ids or turn_key in seen_turns:
                continue
            required.append(candidate)
            seen_ids.add(key)
            seen_turns.add(turn_key)
            if len(required) >= max(2, min(requested + 2, limit // 2)):
                break
        if not required:
            return selected
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in required + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _preserve_broad_raw_recall(self, query: str, plan: Any, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        required = [
            candidate
            for candidate in candidates
            if candidate.type == "span"
            and candidate.metadata.get("broad_raw_recall")
            and _broad_recall_candidate_allowed(query, plan, candidate)
            and (
                candidate.scores.get("intent_recall_signal", 0.0) >= 0.16
                or candidate.scores.get("topic_scope_score", 0.0) >= 0.22
                or candidate.scores.get("exact_signal", 0.0) >= 0.45
            )
        ]
        if not required:
            return selected
        required.sort(
            key=lambda candidate: (
                candidate.scores.get("intent_recall_signal", 0.0),
                candidate.scores.get("topic_scope_score", 0.0),
                candidate.scores.get("exact_signal", 0.0),
                candidate.scores.get("utility_score", candidate.scores.get("score", 0.0)),
                candidate.metadata.get("speaker") == "user",
            ),
            reverse=True,
        )
        diverse: list[Candidate] = []
        seen_queries: set[str] = set()
        for candidate in required:
            recall_query = str(candidate.metadata.get("recall_query") or candidate.id)
            if recall_query in seen_queries:
                continue
            seen_queries.add(recall_query)
            diverse.append(candidate)
            if len(diverse) >= max(2, min(5, limit // 5)):
                break
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in diverse + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _preserve_aggregation_coverage(self, query: str, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        coverage = [
            candidate
            for candidate in candidates
            if candidate.type == "span"
            and (candidate.metadata.get("aggregation_coverage") or _aggregation_context_support_candidate(candidate))
        ]
        group_support = [
            candidate
            for candidate in coverage
            if _aggregation_group_support_specificity(candidate) > 0.0
        ]
        group_support.sort(
            key=lambda candidate: (
                _aggregation_group_support_specificity(candidate),
                candidate.scores.get("score", 0.0),
                candidate.scores.get("topic_scope_score", 0.0),
                candidate.scores.get("aggregation_signal", 0.0),
                candidate.metadata.get("timestamp") or "",
            ),
            reverse=True,
        )
        coverage.sort(
            key=lambda candidate: (
                _aggregation_query_date_support(query, candidate),
                _aggregation_context_specificity(candidate),
                _aggregation_context_support_candidate(candidate),
                candidate.metadata.get("speaker") == "user",
                candidate.scores.get("synthesis_signal", 0.0),
                candidate.scores.get("request_specificity", candidate.metadata.get("request_specificity", 0.0)),
                candidate.scores.get("aggregation_focus", 0.0),
                candidate.scores.get("aggregation_signal", 0.0),
                candidate.scores.get("topic_scope_score", 0.0),
                candidate.metadata.get("timestamp") or "",
            ),
            reverse=True,
        )
        required: list[Candidate] = []
        seen_ids: set[tuple[str, str]] = set()
        group_support_budget = max(1, min(3, limit // 8 if limit >= 8 else 1))
        for candidate in group_support[:group_support_budget]:
            key = (candidate.type, candidate.id)
            if key in seen_ids:
                continue
            required.append(candidate)
            seen_ids.add(key)
        scene_limit = max(1, min(8, limit // 4 if limit >= 4 else 1))
        if _is_broad_exploration_aggregation_query(query.lower()):
            scene_limit = max(scene_limit, min(8, limit))
        scene_representatives = _aggregation_scene_representatives(coverage, limit=scene_limit)
        for candidate in scene_representatives:
            key = (candidate.type, candidate.id)
            if key in seen_ids:
                continue
            required.append(candidate)
            seen_ids.add(key)
        seen_keys: set[str] = set()
        for candidate in required:
            for key in candidate.metadata.get("aggregation_keys") or []:
                if key:
                    seen_keys.add(str(key))
        for candidate in coverage:
            identity = (candidate.type, candidate.id)
            if identity in seen_ids:
                continue
            keys = [str(key) for key in candidate.metadata.get("aggregation_keys") or [] if key]
            keys.extend(_aggregation_query_context_keys(query, candidate.text))
            synthesis_key = _synthesis_candidate_key(candidate)
            if synthesis_key:
                keys.append(synthesis_key)
            if not keys:
                keys = [candidate.id]
            if all(key in seen_keys for key in keys) and not _aggregation_context_support_candidate(candidate):
                continue
            required.append(candidate)
            seen_ids.add(identity)
            seen_keys.update(keys)
            if len(required) >= max(4, min(12, limit)):
                break
        if len(required) < max(2, min(4, limit // 3)):
            for candidate in coverage:
                identity = (candidate.type, candidate.id)
                if identity in seen_ids:
                    continue
                if candidate in required:
                    continue
                required.append(candidate)
                seen_ids.add(identity)
                if len(required) >= max(4, min(12, limit)):
                    break
        required = required[: max(4, min(12, limit))]
        if not required:
            return selected
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in required + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _preserve_user_synthesis_anchors(self, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        required = [
            candidate
            for candidate in candidates
            if candidate.type == "span"
            and candidate.metadata.get("speaker") == "user"
            and (
                float(candidate.scores.get("synthesis_signal", 0.0) or 0.0) > 0
                or bool(candidate.metadata.get("aggregation_keys"))
            )
        ]
        required.sort(
            key=lambda candidate: (
                float(candidate.scores.get("synthesis_signal", 0.0) or 0.0),
                float(candidate.scores.get("topic_scope_score", 0.0) or 0.0),
                float(candidate.scores.get("utility_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        diversified: list[Candidate] = []
        seen_synthesis_keys: set[str] = set()
        target = max(2, min(8, limit))
        for candidate in required:
            synthesis_key = _synthesis_candidate_key(candidate) or candidate.id
            if synthesis_key in seen_synthesis_keys:
                continue
            diversified.append(candidate)
            seen_synthesis_keys.add(synthesis_key)
            if len(diversified) >= target:
                break
        if len(diversified) < target:
            for candidate in required:
                if candidate in diversified:
                    continue
                diversified.append(candidate)
                if len(diversified) >= target:
                    break
        required = diversified[:target]
        if not required:
            return selected
        out = list(selected[:limit])
        seen = {(candidate.type, candidate.id) for candidate in out}
        for candidate in required:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            if len(out) >= limit:
                replace_index = _replaceable_low_synthesis_index(out)
                if replace_index is None:
                    break
                seen.discard((out[replace_index].type, out[replace_index].id))
                out.pop(replace_index)
            out.append(candidate)
            seen.add(key)
        return out

    def _preserve_event_ordering_events(self, query: str, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        coverage_anchors = [
            candidate
            for candidate in candidates
            if "event_ordering_coverage" in candidate.source
            and candidate.metadata.get("timeline_role") in {"user_aspect_anchor", "user_introduced_topic"}
            and candidate.type != "event"
        ]
        coverage_anchors = sorted(
            enumerate(coverage_anchors),
            key=lambda item: (
                0 if item[1].metadata.get("coverage_origin") == "query_required_facet" else 1,
                item[0],
            ),
        )
        coverage_anchors = [candidate for _index, candidate in coverage_anchors]
        coverage_support = [
            candidate
            for candidate in candidates
            if "event_ordering_coverage" in candidate.source
            and candidate.metadata.get("timeline_role") == "supporting_context"
            and candidate.type != "event"
        ]
        event_required = [
            candidate
            for candidate in candidates
            if candidate.type == "event" and "event_timeline_graph" in candidate.source
        ]
        event_required.sort(
            key=lambda candidate: (
                0 if candidate.metadata.get("milestone_group") else 1,
                candidate.metadata.get("time_start") or "",
                candidate.id,
            )
        )
        selector_required = [
            candidate
            for candidate in candidates
            if (
                "event_ordering_graph_selector" in candidate.source
                or "event_ordering_coverage" in candidate.source
            )
            and candidate.type != "event"
            and candidate not in coverage_anchors
            and candidate not in coverage_support
        ]
        event_required = _dedupe_event_ordering_support_events(event_required)
        requested = _event_ordering_requested_count_for_service(query) or 0
        broad_episode_recall = _event_ordering_support_option_query(query)
        episode_scored = [
            (
                _event_ordering_episode_preserve_score(candidate),
                candidate,
            )
            for candidate in candidates
            if _event_ordering_episode_recall_candidate(candidate)
            and (broad_episode_recall or candidate.scores.get("event_facet_coverage", 0.0) >= 0.20)
        ]
        if episode_scored and limit > 0:
            episode_cap = requested + 1 if broad_episode_recall and requested else 6 if broad_episode_recall else 2
            episode_budget = max(
                1,
                min(
                    len(episode_scored),
                    episode_cap,
                    max(2, limit // 2),
                    limit,
                ),
            )
            episode_required = _event_ordering_select_episode_recall_candidates(
                episode_scored,
                limit=episode_budget,
                requested=requested,
            )
        else:
            episode_budget = 0
            episode_required = []
        remaining_after_episode = max(0, limit - episode_budget)
        anchor_cap = max(1, remaining_after_episode)
        if coverage_anchors and requested and not episode_budget:
            anchor_cap = max(anchor_cap, min(len(coverage_anchors), requested, limit))
        if episode_budget and remaining_after_episode >= 3 and not (coverage_anchors and requested):
            anchor_cap = max(1, min(anchor_cap, remaining_after_episode - 1))
        anchor_budget = max(1, min(len(coverage_anchors), anchor_cap)) if coverage_anchors and limit > 0 else 0
        support_budget = max(0, min(len(coverage_support), min(2, limit - episode_budget - anchor_budget)))
        event_budget = max(0, min(len(event_required), limit - episode_budget - anchor_budget - support_budget))
        selector_budget = max(0, limit - episode_budget - anchor_budget - support_budget - event_budget)
        required = (
            coverage_anchors[:anchor_budget]
            + episode_required
            + coverage_support[:support_budget]
            + event_required[:event_budget]
            + selector_required[:selector_budget]
        )
        if not required:
            return selected
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        anchor_groups = {
            str(candidate.metadata.get("topic_group"))
            for candidate in coverage_anchors
            if candidate.metadata.get("topic_group")
        }
        anchor_positions = [
            position
            for candidate in coverage_anchors
            for position in [self._candidate_timeline_position(candidate)]
            if position is not None
        ]
        anchor_min = min(anchor_positions) if anchor_positions else None
        anchor_max = max(anchor_positions) if anchor_positions else None
        fallback_pool = [
            candidate
            for candidate in selected + candidates
            if not anchor_groups or self._candidate_in_topic_groups(candidate, anchor_groups)
            if _candidate_in_timeline_window(
                self._candidate_timeline_position(candidate),
                anchor_min,
                anchor_max,
            )
        ]
        pool = required + fallback_pool
        for candidate in pool:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        return out

    def _candidate_timeline_position(self, candidate: Candidate) -> tuple[tuple[tuple[int, int | str], ...], tuple[tuple[int, int | str], ...], str] | None:
        source_uri = candidate.metadata.get("source_uri")
        turn_id = candidate.metadata.get("turn_id")
        timestamp = candidate.metadata.get("timestamp") or candidate.metadata.get("time_start") or ""
        if source_uri or turn_id:
            return (_natural_turn_key(source_uri), _natural_turn_key(turn_id), str(timestamp))
        for span_id in candidate.source_span_ids:
            span = self.store.get_span(span_id)
            if span:
                return (_natural_turn_key(span.source_uri), _natural_turn_key(span.turn_id), span.timestamp.isoformat())
        return None

    def _apply_topic_scope_filter(self, query: str, plan: Any, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        if plan.query_type == "abstention":
            return selected
        if plan.query_type == "multi_session_reasoning" and _is_broad_exploration_aggregation_query(query.lower()):
            return selected
        topic_groups = {
            str(candidate.metadata.get("topic_group"))
            for candidate in candidates
            if candidate.metadata.get("topic_group")
            and (
                "topic_scope" in candidate.source
                or (plan.query_type == "event_ordering" and "event_ordering_coverage" in candidate.source)
            )
        }
        if not topic_groups:
            return selected
        in_scope = [
            candidate
            for candidate in selected
            if self._candidate_in_topic_groups(candidate, topic_groups)
            or (plan.query_type == "event_ordering" and _event_ordering_topic_scope_preserved_candidate(candidate))
        ]
        if plan.query_type == "event_ordering":
            in_scope = [
                candidate
                for candidate in in_scope
                if self._event_ordering_candidate_allowed_in_topic_scope(query, candidate, topic_groups)
            ]
        if len(in_scope) == len(selected):
            return selected
        replacements = [
            candidate
            for candidate in candidates
            if self._candidate_in_topic_groups(candidate, topic_groups)
            or (plan.query_type == "event_ordering" and _event_ordering_topic_scope_preserved_candidate(candidate))
        ]
        if plan.query_type == "event_ordering":
            replacements = [
                candidate
                for candidate in replacements
                if self._event_ordering_candidate_allowed_in_topic_scope(query, candidate, topic_groups)
            ]
        out: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in in_scope + replacements:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            out.append(candidate)
            seen.add(key)
            if len(out) >= limit:
                break
        if len(out) < max(1, min(limit, len(selected)) // 2):
            return selected
        return out

    def _event_ordering_candidate_allowed_in_topic_scope(self, query: str, candidate: Candidate, topic_groups: set[str]) -> bool:
        if _event_ordering_graph_candidate(candidate):
            return self._candidate_in_topic_groups(candidate, topic_groups) or _event_ordering_graph_candidate_allowed_by_query(query, candidate)
        if _event_ordering_topic_scope_preserved_candidate(candidate):
            return self._candidate_in_topic_groups(candidate, topic_groups) or _event_ordering_episode_candidate_allowed_by_query(query, candidate)
        return True

    def _apply_event_ordering_post_preservation_topic_scope_filter(
        self,
        query: str,
        plan: Any,
        candidates: list[Candidate],
        selected: list[Candidate],
        limit: int,
    ) -> tuple[list[Candidate], list[dict[str, object]]]:
        filtered = self._apply_topic_scope_filter(query, plan, candidates, selected, limit)
        if filtered is selected:
            return selected, []
        filtered_keys = {(candidate.type, candidate.id) for candidate in filtered}
        dropped: list[dict[str, object]] = []
        for candidate in selected:
            if (candidate.type, candidate.id) in filtered_keys:
                continue
            reasons = must_preserve_reasons(candidate)
            if not reasons:
                continue
            dropped.append(
                {
                    "candidate_id": candidate.id,
                    "reason": "topic_scope_filter",
                    "must_preserve_reasons": reasons,
                    "evidence_role": candidate.metadata.get("evidence_role", "answer"),
                    "source": candidate.source,
                    "occupying_candidate_ids": [selected_candidate.id for selected_candidate in filtered],
                    "occupying_candidate_sources": [selected_candidate.source for selected_candidate in filtered],
                    "replaced_by": [selected_candidate.id for selected_candidate in filtered],
                }
            )
        return filtered, dropped

    def _apply_quality_fallback(
        self,
        query: str,
        plan: Any,
        scope: Scope,
        candidates: list[Candidate],
        selected: list[Candidate],
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        if not selected or len(selected) < 3:
            return self._quality_fallback_candidates(query, plan, scope, candidates, selected, limit, include_session=include_session)
        top_scores = [candidate.scores.get("utility_score", candidate.scores.get("score", 0.0)) for candidate in selected[:5]]
        if not top_scores:
            return selected
        avg_score = sum(float(score or 0.0) for score in top_scores) / len(top_scores)
        unique_texts = {candidate.text[:120] for candidate in selected[:5] if candidate.text}
        if avg_score > 0.12 and len(unique_texts) >= 3:
            return selected
        return self._quality_fallback_candidates(query, plan, scope, candidates, selected, limit, include_session=include_session)

    def _quality_fallback_candidates(
        self,
        query: str,
        plan: Any,
        scope: Scope,
        candidates: list[Candidate],
        selected: list[Candidate],
        limit: int,
        *,
        include_session: bool = False,
    ) -> list[Candidate]:
        if plan.query_type == "abstention":
            return selected
        fallback_terms = _quality_fallback_terms(query)
        if not fallback_terms:
            return selected
        if plan.query_type in {"event_ordering", "knowledge_update"}:
            fallback_limit = max(limit, 6)
        elif plan.query_type == "multi_session_reasoning":
            fallback_limit = max(limit, 8)
        else:
            fallback_limit = max(limit, 5)
        existing_ids = {candidate.id for candidate in selected}
        fallback_results: list[Candidate] = []
        seen_terms: set[str] = set()
        for term in fallback_terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            for span, scores in self.store.search_spans(
                term,
                scope,
                limit=fallback_limit,
                include_session=include_session,
            ):
                if span.span_id in existing_ids:
                    continue
                if span.span_type not in {"turn", "tool_result", "document_chunk"}:
                    continue
                topic_score = _topic_scope_score(query, span.content, plan)
                exact = _exact_overlap_score(query, span.content)
                if max(topic_score, exact) <= 0.04:
                    continue
                salience = _fallback_salience_score(span.content)
                score = (0.34 * float(scores.get("score", 0.0))) + (0.28 * topic_score) + (0.18 * exact) + (0.20 * salience)
                fallback_results.append(
                    Candidate(
                        id=span.span_id,
                        type="span",
                        text=span.content,
                        source="quality_fallback",
                        scores={
                            **scores,
                            "semantic_score": max(float(scores.get("semantic_score", 0.0)), topic_score),
                            "bm25_score": max(float(scores.get("bm25_score", 0.0)), exact),
                            "topic_scope_score": topic_score,
                            "salience_score": salience,
                            "score": score,
                        },
                        source_span_ids=[span.span_id],
                        metadata={
                            "speaker": span.speaker,
                            "span_type": span.span_type,
                            "timestamp": span.timestamp.isoformat(),
                            "source_uri": span.source_uri,
                            "turn_id": span.turn_id,
                            "topic_group": _span_group_key(span),
                            "fallback_term": term,
                            "quality_fallback": True,
                        },
                    )
                )
                existing_ids.add(span.span_id)
                if len(fallback_results) >= fallback_limit:
                    break
            if len(fallback_results) >= fallback_limit:
                break
        if not fallback_results:
            return selected
        fallback_results.sort(
            key=lambda candidate: (
                candidate.scores.get("score", 0.0),
                candidate.scores.get("salience_score", 0.0),
                candidate.scores.get("topic_scope_score", 0.0),
                candidate.metadata.get("timestamp") or "",
            ),
            reverse=True,
        )
        merged: list[Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in fallback_results + selected:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            merged.append(candidate)
            seen.add(key)
            if len(merged) >= limit:
                break
        return merged

    def _candidate_in_topic_groups(self, candidate: Candidate, topic_groups: set[str]) -> bool:
        if candidate.metadata.get("topic_group") in topic_groups:
            return True
        for span_id in candidate.source_span_ids:
            span = self.store.get_span(span_id)
            if span and _span_group_key(span) in topic_groups:
                return True
        return False

    def _preserve_high_ranked_summaries(self, candidates: list[Candidate], selected: list[Candidate], limit: int) -> list[Candidate]:
        required = [
            candidate
            for candidate in candidates[:limit]
            if candidate.type == "span" and candidate.metadata.get("span_type") == "summary"
        ]
        if not required:
            return selected
        out = list(selected[:limit])
        seen = {(candidate.type, candidate.id) for candidate in out}
        required_keys = {(candidate.type, candidate.id) for candidate in required}
        for candidate in required:
            key = (candidate.type, candidate.id)
            if key in seen:
                continue
            if len(out) >= limit:
                replaced = False
                for index in range(len(out) - 1, -1, -1):
                    if (out[index].type, out[index].id) not in required_keys:
                        seen.discard((out[index].type, out[index].id))
                        out.pop(index)
                        replaced = True
                        break
                if not replaced:
                    break
            out.append(candidate)
            seen.add(key)
        return out

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


def _event_ordering_requested_count_for_service(query: str) -> int | None:
    lower = query.lower()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"\b(?:only|exactly|mention|list|name)\s+(?:and\s+)?(?:only\s+)?"
        r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        lower,
    )
    if not match:
        match = re.search(
            r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(?:items?|aspects?|steps?|phases?)\b",
            lower,
        )
    if not match:
        return None
    raw = match.group(1)
    try:
        value = int(raw)
    except ValueError:
        value = word_numbers.get(raw)
    return max(1, min(value, 12)) if value is not None else None


def _event_ordering_query_facets_for_service(query: str) -> set[str]:
    terms = _topic_scope_tokens(query)
    terms -= TOPIC_SCOPE_STOPWORDS
    terms -= {
        "different",
        "related",
        "throughout",
        "conversation",
        "conversations",
        "chats",
        "items",
        "aspects",
        "order",
        "only",
        "mention",
        "walk",
        "list",
        "ways",
        "concepts",
        "details",
        "plans",
        "personal",
        "professional",
        "work",
    }
    facets = {term for term in terms if len(term) >= 3}
    expanded = set(facets)
    for term in list(facets):
        expanded.update(_event_ordering_term_variants_for_service(term))
    return expanded


def _event_ordering_anchor_terms_for_service(query: str) -> set[str]:
    lower = query.lower()
    anchors: list[str] = []
    patterns = [
        r"\b(?:aspects of|features of|concerns about|topics related to|ideas related to|plans involving|interactions with)\s+(?:implementing|developing|building|creating|setting up|working on|handling|managing|using|balancing)?\s*(?:my|the|this|that)?\s*([a-z0-9][a-z0-9 +#./'&-]{4,90}?)(?:\s+throughout|\s+across|\s+during|\s+in order|\?|$)",
        r"\b(?:my|the|this|that)\s+([a-z0-9][a-z0-9 +#./'&-]*(?:feature|app|application|website|tracker|dashboard|project|system|tool|api|code|statement|workload|marathon|process|plans?))\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            anchors.append(match.group(1))
    if not anchors:
        return set()
    terms = _topic_scope_tokens(anchors[0])
    terms -= TOPIC_SCOPE_STOPWORDS
    terms -= {"different", "aspects", "features", "concerns", "topics", "ideas", "plans", "personal", "professional", "work"}
    expanded = set(terms)
    for term in list(terms):
        expanded.update(_event_ordering_term_variants_for_service(term))
    return {term for term in expanded if len(term) >= 3}


def _event_ordering_raw_episode_signal(text: str) -> float:
    lower = text.lower()
    score = 0.0
    if re.search(
        r"\b(?:i|we)\s+(?:am|was|were|have|had|'m|'re|'ve)\b|\b(?:i|we)\s+(?:started|used|implemented|configured|fixed|planned|decided|chose|selected|met|shared|asked|tried|tracked|tracking)\b",
        lower,
    ):
        score += 0.35
    if re.search(r"\b(?:started|using|used|implemented|configured|fixed|debug|debugged|planned|planning|decided|chose|selected|compared|reviewed|updated|improved|tracking|met|shared|recommended|suggested)\b", lower):
        score += 0.25
    if re.search(r"\b(?:issue|problem|bug|error|concern|concerns|deadline|meeting|feedback|advice|result|progress|budget|goal|decision)\b", lower):
        score += 0.18
    if re.search(r"\b[A-Z][A-Za-z0-9.+#&-]{2,}\b|\$[\d,]+|\d+(?:\.\d+)?%?\b", text):
        score += 0.12
    if re.search(r"\b(?:thanks|thank you|sounds good|got it|sure,?|ok(?:ay)?)\b", lower):
        score -= 0.20
    return max(0.0, min(1.0, score))


def _event_ordering_referenceable_detail_signal(text: str) -> float:
    score = 0.0
    if re.search(r"\b[A-Z][A-Za-z0-9.+#&-]{2,}\b", text):
        score += 0.20
    if re.search(r"\$[\d,]+|v?\d+(?:\.\d+){1,3}|\d+(?:\.\d+)?%", text, flags=re.I):
        score += 0.20
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:minutes?|hours?|days?|weeks?|months?|items?|people|attendees?|interviews?|candidates?)\b", text, flags=re.I):
        score += 0.18
    if re.search(r"[`'\"][^`'\"]{3,80}[`'\"]|[a-z][a-z0-9]+(?:[-_][a-z0-9]+)+", text):
        score += 0.18
    if re.search(r"\b(?:recommended|suggested|advised|told|feedback|review meeting|hired|started|finalized|upgraded|adjusted|scheduled|coordinated)\b", text, flags=re.I):
        score += 0.18
    return min(1.0, score)


def _event_ordering_support_option_signal(query: str, text: str) -> float:
    query_lower = query.lower()
    if not _event_ordering_support_option_query(query):
        return 0.0
    lower = text.lower()
    score = 0.0
    if re.search(r"\b(?:hire[ds]?|hiring|brought on|bring(?:ing)? on|contract(?:ed)?|outsourc(?:e|ed|ing)|delegate[ds]?|delegating|delegation)\b", lower):
        score += 0.36
    if re.search(r"\b(?:assistant|agency|mentor|coach|consultant|advisor|adviser|specialist|contractor|service|team|colleague|partner)\b", lower):
        score += 0.26
    if re.search(r"\b(?:tool|tools|board|boards|calendar|reminder|workflow|automation|software|app|platform|system|template|checklist)\b", lower):
        score += 0.24
    if re.search(r"\b(?:advice|recommended|suggested|strategy|strategies|approach|plan|process|method|option|support)\b", lower):
        score += 0.18
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|/week|per week)\b|\$[\d,]+(?:\.\d+)?(?:/hour|/month| per hour| monthly)?\b", lower):
        score += 0.14
    return min(1.0, score)


def _event_ordering_support_option_query(query: str) -> bool:
    return bool(re.search(r"\b(?:strateg(?:y|ies)|support|options?|resources?|tools?|help|ways?)\b", query.lower()))


def _event_ordering_term_variants_for_service(term: str) -> set[str]:
    variants = {term}
    if len(term) > 5 and term.endswith("ing"):
        stem = term[:-3]
        variants.add(stem)
        variants.add(stem + "e")
    if len(term) > 4 and term.endswith("ed"):
        stem = term[:-2]
        variants.add(stem)
        variants.add(stem + "e")
    if len(term) > 4 and term.endswith("e"):
        variants.add(term[:-1])
    if term.endswith("iz") or term.endswith("is"):
        variants.add(term + "e")
    if term.endswith("izing"):
        variants.add(term[:-5])
        variants.add(term[:-3])
    if term.endswith("ization"):
        variants.add(term[:-7])
        variants.add(term[:-5])
    return {variant for variant in variants if len(variant) >= 3}


def _event_ordering_select_episode_recall_candidates(
    scored: list[tuple[float, Candidate]],
    limit: int,
    requested: int,
) -> list[Candidate]:
    if not scored:
        return []
    ordered = sorted(
        scored,
        key=lambda item: (
            _natural_turn_key(item[1].metadata.get("source_uri")),
            _natural_turn_key(item[1].metadata.get("turn_id")),
            item[1].metadata.get("timestamp") or "",
            item[1].id,
        ),
    )
    if limit <= 0:
        return []
    target = max(requested * 3, requested + 8) if requested else limit
    desired = min(limit, max(1, target))
    selected: list[tuple[float, Candidate]] = []
    seen_ids: set[str] = set()
    total = len(ordered)
    buckets = max(1, min(desired, total))
    for bucket in range(buckets):
        start = round(bucket * total / buckets)
        end = round((bucket + 1) * total / buckets)
        if end <= start:
            end = min(total, start + 1)
        window = [item for item in ordered[start:end] if item[1].id not in seen_ids]
        if not window:
            continue
        choice = max(
            window,
            key=lambda item: (
                item[0],
                item[1].scores.get("event_facet_coverage", 0.0),
                item[1].scores.get("event_support_option_signal", 0.0),
                item[1].scores.get("event_episode_signal", 0.0),
                item[1].scores.get("event_detail_signal", 0.0),
            ),
        )
        selected.append(choice)
        seen_ids.add(choice[1].id)
        if len(selected) >= desired:
            break
    if len(selected) < desired:
        for item in sorted(
            scored,
            key=lambda value: (
                value[0],
                value[1].scores.get("event_facet_coverage", 0.0),
                value[1].scores.get("event_support_option_signal", 0.0),
                value[1].scores.get("event_episode_signal", 0.0),
                value[1].scores.get("event_detail_signal", 0.0),
            ),
            reverse=True,
        ):
            if item[1].id in seen_ids:
                continue
            selected.append(item)
            seen_ids.add(item[1].id)
            if len(selected) >= desired:
                break
    selected_candidates = [candidate for _score, candidate in selected]
    selected_candidates.sort(
        key=lambda candidate: (
            _natural_turn_key(candidate.metadata.get("source_uri")),
            _natural_turn_key(candidate.metadata.get("turn_id")),
            candidate.metadata.get("timestamp") or "",
            candidate.id,
        )
    )
    return selected_candidates[:limit]


def _event_ordering_episode_recall_candidate(candidate: Candidate) -> bool:
    if "event_ordering_episode_recall" not in candidate.source:
        return False
    if candidate.type != "span":
        return False
    return (
        float(candidate.scores.get("event_episode_signal", 0.0) or 0.0) >= 0.35
        or float(candidate.scores.get("event_detail_signal", 0.0) or 0.0) >= 0.30
        or float(candidate.scores.get("event_support_option_signal", 0.0) or 0.0) >= 0.30
        or float(candidate.scores.get("event_facet_coverage", 0.0) or 0.0) >= 0.12
        or bool(candidate.metadata.get("event_ordering_facet_hits"))
        or bool(candidate.metadata.get("event_ordering_anchor_hits"))
    )


def _event_ordering_episode_preserve_score(candidate: Candidate) -> float:
    return min(
        1.0,
        float(candidate.scores.get("score", 0.0) or 0.0)
        + 0.18 * float(candidate.scores.get("event_facet_coverage", 0.0) or 0.0)
        + 0.12 * float(candidate.scores.get("event_support_option_signal", 0.0) or 0.0)
        + 0.12 * float(candidate.scores.get("event_episode_signal", 0.0) or 0.0)
        + 0.10 * float(candidate.scores.get("event_detail_signal", 0.0) or 0.0),
    )


def _event_ordering_topic_scope_preserved_candidate(candidate: Candidate) -> bool:
    if not _event_ordering_episode_recall_candidate(candidate):
        return False
    if candidate.metadata.get("speaker") not in {"user", "document"}:
        return False
    if _event_ordering_low_value_raw_facet_candidate(candidate):
        return False
    return True


def _event_ordering_raw_facet_episode_required(candidate: Candidate, *, broad_episode_recall: bool) -> bool:
    if broad_episode_recall:
        return (
            float(candidate.scores.get("event_support_option_signal", 0.0) or 0.0) >= 0.30
            or float(candidate.scores.get("event_episode_signal", 0.0) or 0.0) >= 0.35
            or float(candidate.scores.get("event_detail_signal", 0.0) or 0.0) >= 0.36
            or float(candidate.scores.get("event_facet_coverage", 0.0) or 0.0) >= 0.20
        )
    return (
        float(candidate.scores.get("event_facet_coverage", 0.0) or 0.0) >= 0.20
        and float(candidate.scores.get("event_episode_signal", 0.0) or 0.0) >= 0.35
    )


def _event_ordering_low_value_raw_facet_candidate(candidate: Candidate) -> bool:
    lower = (candidate.text or "").lower()
    if candidate.metadata.get("speaker") not in {"user", "document"}:
        return True
    if re.fullmatch(r"\s*(?:thanks|thank you|sounds good|ok(?:ay)?|sure)[.!]?\s*", lower):
        return True
    if re.search(r"\balways provide\b|\bwhen i ask\b", lower) and len(_topic_scope_tokens(lower)) <= 6:
        return True
    return False


def _event_ordering_low_value_raw_facet_text(text: str) -> bool:
    lower = text.lower()
    if re.fullmatch(r"\s*(?:thanks|thank you|sounds good|ok(?:ay)?|sure)[.!]?\s*", lower):
        return True
    if re.search(r"\balways provide\b|\bwhen i ask\b", lower) and len(_topic_scope_tokens(lower)) <= 6:
        return True
    if re.fullmatch(r"\s*(?:yes|yeah|no|maybe|okay|ok)[,.! ]{0,8}\s*", lower):
        return True
    return False


def _event_ordering_legacy_candidate(candidate: Candidate) -> bool:
    source = str(candidate.source or "")
    legacy_markers = (
        "event_ordering_coverage",
        "event_ordering_timeline",
        "event_ordering_episode_recall",
        "event_ordering_graph_selector_event",
        "event_timeline_graph",
    )
    is_legacy = bool(candidate.metadata.get("graph_fallback")) or (
        any(marker in source for marker in legacy_markers) and not source.startswith("event_ordering_graph")
    )
    return is_legacy


def _record_event_ordering_legacy_rescue_hits(candidates: list[Candidate]) -> None:
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (str(candidate.id), str(candidate.source or ""))
        if key in seen:
            continue
        seen.add(key)
        record_rule_hit(
            "event_ordering.legacy_rescue",
            query="",
            text=candidate.text,
            stage="event_ordering_selection",
            contributed_candidate_id=candidate.id,
            metadata={"decision": "legacy_fallback", "source": candidate.source},
        )


def _event_ordering_graph_candidate(candidate: Candidate) -> bool:
    source = str(candidate.source or "")
    return source in {"event_ordering_persisted_graph"} or (
        source.startswith("event_ordering_graph") and not bool(candidate.metadata.get("graph_fallback"))
    )


def _event_ordering_graph_selector_coverage_telemetry(candidates: list[Candidate]) -> dict[str, Any]:
    for candidate in candidates:
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        telemetry = metadata.get("graph_selector_telemetry") or metadata.get("persisted_graph_telemetry")
        if not isinstance(telemetry, dict):
            continue
        out: dict[str, Any] = {}
        expanded_topic_ids = telemetry.get("cluster_expanded_topic_ids")
        if isinstance(expanded_topic_ids, list):
            out["cluster_expanded_topic_ids"] = [str(topic_id) for topic_id in expanded_topic_ids if str(topic_id).strip()]
        selected_topic_count = telemetry.get("selected_topic_count")
        if isinstance(selected_topic_count, int):
            out["selected_topic_count"] = selected_topic_count
        graph_ordered_legacy_recall_count = telemetry.get("graph_ordered_legacy_recall_count")
        out["graph_ordered_legacy_recall_count"] = (
            graph_ordered_legacy_recall_count
            if isinstance(graph_ordered_legacy_recall_count, int)
            else None
        )
        return out
    return {}


def _event_ordering_graph_candidate_allowed_by_query(query: str, candidate: Candidate) -> bool:
    if not _event_ordering_graph_candidate(candidate):
        return True
    candidate_tokens = _topic_scope_tokens(candidate.text)
    if not candidate_tokens:
        return True
    query_tokens = _topic_scope_tokens(query)
    if not query_tokens:
        return True
    expanded_candidate_tokens = set(candidate_tokens)
    for token in list(candidate_tokens):
        expanded_candidate_tokens.update(_event_ordering_term_variants_for_service(token))
    expanded_query_tokens = set(query_tokens)
    for token in list(query_tokens):
        expanded_query_tokens.update(_event_ordering_term_variants_for_service(token))
    return bool(expanded_candidate_tokens & expanded_query_tokens)


def _event_ordering_episode_candidate_allowed_by_query(query: str, candidate: Candidate) -> bool:
    if candidate.metadata.get("event_ordering_facet_hits") or candidate.metadata.get("event_ordering_anchor_hits"):
        return True
    candidate_tokens = _topic_scope_tokens(candidate.text)
    query_tokens = _topic_scope_tokens(query)
    if not candidate_tokens or not query_tokens:
        return True
    expanded_candidate_tokens = set(candidate_tokens)
    for token in list(candidate_tokens):
        expanded_candidate_tokens.update(_event_ordering_term_variants_for_service(token))
    expanded_query_tokens = set(query_tokens)
    for token in list(query_tokens):
        expanded_query_tokens.update(_event_ordering_term_variants_for_service(token))
    return bool(expanded_candidate_tokens & expanded_query_tokens)


def _event_ordering_candidate_path_key(candidate: Candidate) -> tuple[str, str, bool]:
    return (
        str(candidate.id),
        str(candidate.source or ""),
        bool(candidate.metadata.get("graph_fallback")),
    )


def _event_ordering_dual_shadow_candidate_key(candidate: Candidate) -> tuple[tuple[str, ...], str, str, str]:
    span_ids = tuple(dict.fromkeys(str(span_id) for span_id in candidate.source_span_ids if span_id))
    if not span_ids:
        normalized_text = " ".join(re.findall(r"[a-zA-Z0-9]+", candidate.text.lower()))
        source = str(candidate.source or "")
        candidate_id = str(candidate.id or "")
        return ((), normalized_text, source, candidate_id)
    return (span_ids, "", "", "")


def _dedupe_event_ordering_candidates_for_shadow(candidates: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for candidate in candidates:
        key = _event_ordering_dual_shadow_candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _stale_current_value_candidate_text(text: str) -> bool:
    lower = (text or "").lower()
    if re.search(r"\b(?:no longer|not anymore|instead|keep it only as historical|historical context)\b", lower):
        return False
    if re.search(r"\b(?:but|while|although)\b.{0,80}\b(?:current|currently|default|latest|now)\b", lower):
        return False
    if re.search(r"(?:不是|不再是|替代|改成|更新为).{0,40}(?:以前|之前|原来|曾经|早期)", lower):
        return False
    if re.search(r"\b(?:switched|changed|updated|moved)\s+(?:from|to)\b", lower):
        return False
    return bool(
        re.search(r"\b(?:initially|previously|formerly|originally|used to|before the switch|at first)\b", lower)
        or re.search(r"最初|以前|之前|原来|曾经", lower)
    )


def _candidate_has_source(candidate: Candidate, source: str) -> bool:
    return source in {part.strip() for part in str(candidate.source or "").split("+") if part.strip()}
