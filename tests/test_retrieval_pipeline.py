from __future__ import annotations

import unittest
from types import MethodType

from fusion_memory.api.service import MemoryService
from fusion_memory.core.models import Candidate
from fusion_memory.core.models import EvidencePack
from fusion_memory.core.models import QueryPlan
from fusion_memory.core.models import Scope
from fusion_memory.core.models import SearchResult
from fusion_memory.retrieval.pipeline import CandidateFusionEngine
from fusion_memory.retrieval.pipeline import QueryUnderstandingEngine
from fusion_memory.retrieval.pipeline import QueryUnderstandingResult
from fusion_memory.retrieval.pipeline import RecallOrchestrator
from fusion_memory.retrieval.pipeline import RecallResult
from fusion_memory.retrieval.pipeline import RetrievalExecutionContext
from fusion_memory.retrieval.pipeline import RetrievalTraceRecorder
from fusion_memory.retrieval.pipeline import build_pipeline_record
from fusion_memory.retrieval.pipeline import selected_temporal_relation_summary
from fusion_memory.retrieval.raw_evidence_quota import QuotaResult


class RetrievalPipelineTests(unittest.TestCase):
    def test_candidate_fusion_engine_matches_existing_scoring_shape(self) -> None:
        class FakeConfig:
            rrf_k = 1
            retrieval_output_n = 2
            balanced_mode_rerank_top_n = 5
            benchmark_mode_rerank_top_n = 7

        class FakeQuota:
            def __init__(self) -> None:
                self.calls = []

            def enforce(self, plan, scope, candidates, *, include_session=False):
                self.calls.append((plan, scope, [candidate.id for candidate in candidates], include_session))
                return QuotaResult(
                    candidates=list(candidates),
                    selected_span_ids=["span-2"],
                    required=1,
                    coverage_insufficient=False,
                    backfilled=0,
                )

        class FakeService:
            def __init__(self) -> None:
                self.config = FakeConfig()
                self.quota = FakeQuota()

        def event_milestone_group(value) -> str | None:
            return None

        plan = QueryPlan(query="private token zinc-sparrow-17", query_type="fact_lookup", entities=[], time_constraints=[])
        query_understanding = QueryUnderstandingResult(
            plan=plan,
            language="en",
            intent="fact_lookup",
            features=(),
            intent_telemetry=None,
            precomputed=True,
        )
        scope = Scope(workspace_id="ws-fusion", user_id="u", agent_id="a", session_id="s")
        service = FakeService()
        context = RetrievalExecutionContext(
            service=service,
            query="private token zinc-sparrow-17",
            scope=scope,
            options={"mode": "balanced", "limit": 2},
            query_understanding=query_understanding,
            include_session=True,
            per_source_limit=3,
            enabled_sources=None,
            mode="balanced",
            limit=2,
            rerank_top_n=5,
            event_milestone_group=event_milestone_group,
        )
        span_1 = Candidate("span-1", "span", "private candidate alpha", "semantic", {"semantic_score": 0.7}, ["span-1"], {})
        span_2_semantic = Candidate("span-2", "span", "private candidate beta", "semantic", {"semantic_score": 0.2}, ["span-2"], {})
        span_2_exact = Candidate("span-2", "span", "private candidate beta", "exact", {"exact_signal": 0.9}, ["span-2"], {})
        recall_result = RecallResult(candidate_lists=[[span_1, span_2_semantic], [span_2_exact]], recalled_candidates=[span_1, span_2_semantic, span_2_exact])

        result = CandidateFusionEngine().run(context, recall_result)

        self.assertEqual([candidate.id for candidate in result.fused], ["span-2", "span-1"])
        self.assertEqual(service.quota.calls, [(plan, scope, ["span-2", "span-1"], True)])
        self.assertEqual([candidate.id for candidate in result.scored], ["span-2", "span-1"])
        self.assertEqual([candidate.id for candidate in result.marked], ["span-2", "span-1"])
        self.assertTrue(result.marked[0].metadata["quota_selected"])
        self.assertEqual([candidate.id for candidate in result.scored_again], ["span-2", "span-1"])
        self.assertGreaterEqual(result.scored_again[0].scores["utility_score"], result.scored_again[1].scores["utility_score"])
        self.assertIs(result.quota_result.candidates[0], result.scored[0])
        self.assertEqual(result.mode, "balanced")
        self.assertEqual(result.limit, 2)
        self.assertEqual(result.rerank_top_n, 5)
        self.assertNotIn("zinc-sparrow-17", repr(result.safe_record()))

    def test_recall_orchestrator_returns_sanitized_result(self) -> None:
        test_case = self

        class FakeRegistry:
            def __init__(self) -> None:
                self.contexts = []

            def recall(self, context):
                self.contexts.append(context)
                return [
                    [
                        Candidate("c1", "span", "private token zinc-sparrow-17", "l0_raw", {}, ["s1"], {}),
                        Candidate("c2", "fact", "another private token", "l1_fact", {}, ["s2"], {}),
                    ],
                    [Candidate("c3", "span", "raw session secret", "l0_raw", {}, ["s3"], {})],
                ]

            def summary(self, context):
                test_case.assertIs(context, self.contexts[0])
                return [
                    {
                        "provider_id": "fake_raw",
                        "source_family": "raw",
                        "output_count": 3,
                        "output_source_counts": {"l0_raw": 2, "l1_fact": 1},
                        "sample_text": "private token zinc-sparrow-17",
                    }
                ]

        def event_milestone_group(value) -> str | None:
            return None

        plan = QueryPlan(query="raw private query", query_type="fact_lookup", entities=[], time_constraints=[])
        query_understanding = QueryUnderstandingResult(
            plan=plan,
            language="en",
            intent="fact_lookup",
            features=("current_value",),
            intent_telemetry=None,
            precomputed=True,
        )
        context = RetrievalExecutionContext(
            service=object(),
            query="raw private query with zinc-sparrow-17",
            scope=Scope(workspace_id="ws-recall", user_id="u", agent_id="a", session_id="s"),
            options={"enabled_sources": ["raw"], "mode": "balanced"},
            query_understanding=query_understanding,
            include_session=True,
            per_source_limit=5,
            enabled_sources=["raw"],
            mode="balanced",
            limit=4,
            rerank_top_n=3,
            event_milestone_group=event_milestone_group,
        )
        registry = FakeRegistry()

        result = RecallOrchestrator(registry=registry).run(context)

        self.assertEqual(len(result.candidate_lists), 2)
        self.assertEqual([candidate.id for candidate in result.recalled_candidates], ["c1", "c2", "c3"])
        self.assertEqual(result.safe_record()["source_counts"], {"l0_raw": 2, "l1_fact": 1})
        self.assertEqual(
            result.safe_record()["provider_summary"],
            [
                {
                    "provider_id": "fake_raw",
                    "source_family": "raw",
                    "output_count": 3,
                    "output_source_counts": {"l0_raw": 2, "l1_fact": 1},
                }
            ],
        )
        self.assertEqual(registry.contexts[0].enabled_sources, {"raw"})
        self.assertTrue(registry.contexts[0].include_session)
        self.assertNotIn("zinc-sparrow-17", repr(result.safe_record()))
        self.assertNotIn("raw private query", repr(result.safe_record()))

    def test_query_understanding_engine_sanitizes_raw_query(self) -> None:
        class FakePlanner:
            def __init__(self) -> None:
                self.last_intent_telemetry = {"route": "heuristic", "raw_query": "planner-internal"}
                self.calls = []

            def plan(self, query: str, *, query_type_hint: str | None = None) -> QueryPlan:
                self.calls.append((query, query_type_hint))
                return QueryPlan(
                    query=query,
                    query_type=query_type_hint or "fact_lookup",
                    entities=[],
                    time_constraints=[],
                    intent={"answer_shape": "fact"},
                )

        raw_query = "Which private token zinc-sparrow-17 did I ask you to remember?"
        planner = FakePlanner()
        scope = Scope(workspace_id="ws-pipeline", user_id="u", agent_id="a")

        result = QueryUnderstandingEngine().run(
            raw_query,
            scope,
            {"query_type_hint": "event_ordering"},
            planner,
        )

        self.assertEqual(planner.calls, [(raw_query, "event_ordering")])
        self.assertIs(result.plan.query, raw_query)
        self.assertEqual(result.language, "en")
        self.assertEqual(result.intent, "event_ordering")
        self.assertEqual(result.features, ("temporal",))
        self.assertEqual(result.intent_telemetry, planner.last_intent_telemetry)
        self.assertFalse(result.precomputed)
        self.assertEqual(
            result.safe_record(),
            {"language": "en", "intent": "event_ordering", "features": ["temporal"]},
        )
        self.assertNotIn(raw_query, repr(result.safe_record()))

        precomputed_plan = QueryPlan(
            query=raw_query,
            query_type="knowledge_update",
            entities=[],
            time_constraints=[],
        )
        precomputed = QueryUnderstandingEngine().run(
            raw_query,
            scope,
            {"_plan": precomputed_plan, "_intent_telemetry": {"route": "precomputed"}},
            planner,
        )

        self.assertIs(precomputed.plan, precomputed_plan)
        self.assertEqual(precomputed.intent_telemetry, {"route": "precomputed"})
        self.assertTrue(precomputed.precomputed)

    def test_build_pipeline_record_counts_sources_without_raw_text(self) -> None:
        recalled = [
            Candidate("c1", "span", "raw secret text", "l0_raw", {"utility_score": 0.8}, ["s1"], {}),
            Candidate("c2", "fact", "another secret", "l3_current_view", {"utility_score": 0.7}, ["s2"], {}),
        ]
        selected = [recalled[1]]

        record = build_pipeline_record(
            "current_value",
            "benchmark",
            language="en",
            intent="current_value",
            features=["current_value"],
            recalled=recalled,
            selected=selected,
            dropped_count=1,
            source_span_count=1,
            coverage_insufficient=False,
        )
        payload = record.to_dict()

        self.assertEqual(payload["pipeline_layers"]["CandidateRecall"]["source_counts"]["l0_raw"], 1)
        self.assertEqual(payload["pipeline_layers"]["CandidateFusion"]["selected_sources"], ["l3_current_view"])
        self.assertNotIn("raw secret text", repr(payload))

    def test_build_pipeline_record_can_include_temporal_relation_summary(self) -> None:
        record = build_pipeline_record(
            "temporal_lookup",
            "default",
            language="en",
            intent="temporal_lookup",
            features=["temporal_terms"],
            recalled=[],
            selected=[],
            dropped_count=0,
            source_span_count=0,
            coverage_insufficient=False,
            temporal_relation_summary={
                "relation_count": 2,
                "relation_types": ["deadline"],
                "source_span_count": 1,
            },
        )

        data = record.to_dict()

        self.assertEqual(data["pipeline_layers"]["TemporalRelations"]["relation_count"], 2)

    def test_retrieval_trace_recorder_flushes_sanitized_pipeline_layers(self) -> None:
        raw_query_text = "Which private token zinc-sparrow-17 happened before the deadline?"
        recalled = [
            Candidate("c1", "event", "raw secret zinc-sparrow-17 first", "l0_raw", {"utility_score": 0.8}, ["s1"], {}),
            Candidate("c2", "event", "raw secret zinc-sparrow-17 second", "graph", {"utility_score": 0.7}, ["s2"], {}),
        ]
        selected = [recalled[0]]
        kwargs = {
            "language": "en",
            "intent": "event_ordering",
            "features": ["temporal"],
            "recalled": recalled,
            "selected": selected,
            "dropped_count": 1,
            "source_span_count": 1,
            "coverage_insufficient": False,
            "temporal_relation_summary": {
                "relation_count": 1,
                "relation_types": ["before"],
                "role_labels": ["earlier_event"],
                "reason_codes": ["explicit_order_marker"],
                "source_span_count": 1,
                "source_span_ids": ["s1"],
            },
        }
        expected = build_pipeline_record("event_ordering", "benchmark", **kwargs).to_dict()

        payload = RetrievalTraceRecorder(build_pipeline_record("event_ordering", "benchmark", **kwargs)).flush()

        self.assertEqual(set(payload), set(expected))
        self.assertEqual(payload, expected)
        self.assertIn("TemporalRelations", payload["pipeline_layers"])
        self.assertNotIn(raw_query_text, repr(payload))
        self.assertNotIn("zinc-sparrow-17", repr(payload))
        self.assertNotIn("raw secret", repr(payload))

    def test_selected_temporal_relation_summary_merges_summary_and_safe_records(self) -> None:
        candidates = [
            Candidate(
                "c1",
                "event",
                "before the deadline",
                "source_a",
                {"utility_score": 0.6},
                ["span_1"],
                {
                    "temporal_relations": [
                        {
                            "relation_type": "before",
                            "reason_code": "explicit_order_marker",
                            "role_labels": ["earlier_event"],
                            "source_span_ids": ["span_1"],
                        }
                    ]
                },
            ),
            Candidate(
                "c2",
                "fact",
                "deadline summary",
                "source_b",
                {"utility_score": 0.7},
                ["span_2"],
                {
                    "temporal_relation_summary": {
                        "relation_count": 2,
                        "relation_types": ["deadline"],
                        "role_labels": ["deadline"],
                        "reason_codes": ["deadline_marker"],
                        "source_span_count": 1,
                        "source_span_ids": ["span_2"],
                    }
                },
            ),
        ]

        summary = selected_temporal_relation_summary(candidates)

        self.assertEqual(summary["relation_count"], 3)
        self.assertEqual(summary["relation_types"], ["before", "deadline"])
        self.assertEqual(summary["role_labels"], ["deadline", "earlier_event"])
        self.assertEqual(summary["reason_codes"], ["deadline_marker", "explicit_order_marker"])
        self.assertEqual(summary["source_span_count"], 2)
        self.assertEqual(summary["source_span_ids"], ["span_1", "span_2"])

    def test_selected_temporal_relation_summary_prefers_relation_records_over_summary_for_same_candidate(self) -> None:
        candidates = [
            Candidate(
                "c1",
                "event",
                "before the deadline",
                "source_a",
                {"utility_score": 0.6},
                ["span_1"],
                {
                    "temporal_relations": [
                        {
                            "relation_type": "before",
                            "reason_code": "explicit_order_marker",
                            "role_labels": ["earlier_event"],
                            "source_span_ids": ["span_1"],
                        }
                    ],
                    "temporal_relation_summary": {
                        "relation_count": 2,
                        "relation_types": ["before"],
                        "role_labels": ["earlier_event"],
                        "reason_codes": ["explicit_order_marker"],
                        "source_span_count": 1,
                        "source_span_ids": ["span_1"],
                    },
                },
            ),
        ]

        summary = selected_temporal_relation_summary(candidates)

        self.assertEqual(summary["relation_count"], 1)
        self.assertEqual(summary["relation_types"], ["before"])
        self.assertEqual(summary["role_labels"], ["earlier_event"])
        self.assertEqual(summary["reason_codes"], ["explicit_order_marker"])
        self.assertEqual(summary["source_span_count"], 1)
        self.assertEqual(summary["source_span_ids"], ["span_1"])

    def test_search_attaches_pipeline_trace_without_raw_text(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-pipeline", user_id="u", agent_id="a")
        raw_memory_text = "Remember private token zinc-sparrow-17 for the pipeline trace test."
        raw_query_text = "Which private token did I ask you to remember for pipeline trace?"
        try:
            memory.add({"role": "user", "content": raw_memory_text}, scope)

            result = memory.search(raw_query_text, scope)
            trace = memory.debug_trace(result.trace_id, scope)

            self.assertIn("pipeline_trace", result.coverage)
            self.assertEqual(trace["pipeline_trace"], result.coverage["pipeline_trace"])
            self.assertIn("retrieval_trace", trace)
            self.assertNotIn(raw_query_text, repr(result.coverage["pipeline_trace"]))
            self.assertNotIn(raw_memory_text, repr(result.coverage["pipeline_trace"]))
        finally:
            memory.close()

    def test_answer_context_exposes_pipeline_trace_in_coverage(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-pipeline-pack", user_id="u", agent_id="a")
        raw_memory_text = "Remember private token cobalt-delta-29 for the answer pack trace test."
        raw_query_text = "Which private token did I ask you to remember for answer pack trace?"
        try:
            memory.add({"role": "user", "content": raw_memory_text}, scope)

            pack = memory.answer_context(raw_query_text, scope)

            self.assertIn("pipeline_trace", pack.coverage)
            self.assertIsInstance(pack.debug_trace, list)
            self.assertNotIn(raw_query_text, repr(pack.coverage["pipeline_trace"]))
            self.assertNotIn(raw_memory_text, repr(pack.coverage["pipeline_trace"]))
        finally:
            memory.close()

    def test_answer_context_pipeline_trace_uses_actual_pack_source_span_count(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-pipeline-pack-count", user_id="u", agent_id="a")
        stale_pipeline_trace = {
            "query_type": "fact_lookup",
            "mode": "fast",
            "pipeline_layers": {
                "QueryUnderstanding": {"language": "en", "intent": "fact_lookup", "features": []},
                "CandidateRecall": {"source_counts": {"l0_raw": 5}},
                "CandidateFusion": {"selected_sources": ["l0_raw"], "dropped_count": 0},
                "EvidencePackBuilder": {"source_span_count": 5, "coverage_insufficient": False},
            },
            "query_understanding": {"language": "en", "intent": "fact_lookup", "features": []},
            "candidate_recall": {"source_counts": {"l0_raw": 5}},
            "candidate_fusion": {"selected_sources": ["l0_raw"], "dropped_count": 0},
            "evidence_output": {"source_span_count": 5, "coverage_insufficient": False},
        }

        def fake_search(self: MemoryService, query: str, scope: Scope, options: dict | None = None) -> SearchResult:
            return SearchResult(candidates=[], trace_id="trace-pipeline", coverage={"pipeline_trace": stale_pipeline_trace})

        def fake_get_trace(self, trace_id: str, scope: Scope, include_session: bool = False) -> dict:
            return {"selected": [], "rule_hits": []}

        def fake_build(self, query, plan, candidates, coverage, trace, token_budget=None) -> EvidencePack:
            return EvidencePack(
                query=query,
                answer_policy="test",
                current_views=[],
                entity_profiles=[],
                facts=[],
                events=[],
                source_spans=[{"id": "span_1"}],
                conflicts=[],
                coverage=dict(coverage),
                debug_trace=[],
            )

        memory.search = MethodType(fake_search, memory)
        memory.store.get_trace = MethodType(fake_get_trace, memory.store)
        memory.pack_builder.build = MethodType(fake_build, memory.pack_builder)

        try:
            pack = memory.answer_context("What did I ask you to remember?", scope)
        finally:
            memory.close()

        evidence_layer = pack.coverage["pipeline_trace"]["pipeline_layers"]["EvidencePackBuilder"]
        self.assertEqual(evidence_layer["source_span_count"], len(pack.source_spans))

    def test_search_coverage_includes_candidate_lifecycle_without_raw_text(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-lifecycle", user_id="u", agent_id="a")
        raw_memory_text = "Remember private token violet-river-42 for lifecycle tracing."
        raw_query_text = "Which private token did I ask you to remember for lifecycle tracing?"
        try:
            memory.add({"role": "user", "content": raw_memory_text}, scope)
            result = memory.search(raw_query_text, scope)
            trace = memory.debug_trace(result.trace_id, scope)
        finally:
            memory.close()

        lifecycle = result.coverage["candidate_lifecycle"]
        self.assertGreater(lifecycle["stage_counts"].get("recalled", 0), 0)
        self.assertGreater(lifecycle["stage_counts"].get("selected", 0), 0)
        self.assertIn("candidate_lifecycle_trace", trace)
        self.assertNotIn(raw_memory_text, repr(lifecycle))
        self.assertNotIn(raw_query_text, repr(lifecycle))
        self.assertNotIn(raw_memory_text, repr(trace["candidate_lifecycle_trace"]))
        self.assertNotIn(raw_query_text, repr(trace["candidate_lifecycle_trace"]))

    def test_search_lifecycle_records_preservation_rescue_delta_without_raw_text(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-lifecycle-rescue", user_id="u", agent_id="a")
        base = Candidate("base", "span", "base raw token zinc-sparrow-17", "l0_raw", {"utility_score": 0.9}, ["base"], {})
        rescue = Candidate(
            "rescue",
            "span",
            "rescued raw token zinc-sparrow-17",
            "raw_scent_trail",
            {"utility_score": 0.1, "trail_score": 0.9},
            ["rescue"],
            {},
        )

        def fake_recall_candidates(context):
            return RecallResult(candidate_lists=[[base, rescue]], recalled_candidates=[base, rescue])

        def fake_enforce(plan, search_scope, candidates, *, include_session=False):
            return QuotaResult(candidates=list(candidates), selected_span_ids=["base"], required=1, coverage_insufficient=False, backfilled=0)

        def fake_mmr_selected(candidates, selected, limit):
            return [base]

        def fake_preserve_scent(candidates, selected, limit):
            return [rescue, base][:limit]

        memory._recall_candidates = fake_recall_candidates
        memory.quota.enforce = fake_enforce
        memory.planner.plan = lambda query, query_type_hint=None: QueryPlan(query=query, query_type="fact_lookup", entities=[], time_constraints=[])
        memory._preserve_high_signal_exact = fake_mmr_selected
        memory._preserve_scent_trail = fake_preserve_scent
        memory.store.insert_utility_example = lambda example: None

        try:
            result = memory.search("Which raw token?", scope, {"limit": 2, "mode": "fast"})
            trace = memory.debug_trace(result.trace_id, scope)
        finally:
            memory.close()

        lifecycle = result.coverage["candidate_lifecycle"]
        self.assertGreater(lifecycle["stage_counts"].get("rescued", 0), 0)
        self.assertGreater(lifecycle["reason_counts"].get("preserve_scent_trail", 0), 0)
        self.assertNotIn("zinc-sparrow-17", repr(lifecycle))
        self.assertNotIn("zinc-sparrow-17", repr(trace["candidate_lifecycle_trace"]))

    def test_answer_context_lifecycle_reports_packed_source_spans(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-lifecycle-pack", user_id="u", agent_id="a")
        try:
            memory.add({"role": "user", "content": "Remember private token silver-forest-77 for packed lifecycle."}, scope)
            pack = memory.answer_context("Which private token did I ask you to remember for packed lifecycle?", scope)
        finally:
            memory.close()

        lifecycle = pack.coverage["candidate_lifecycle"]
        self.assertEqual(lifecycle["stage_counts"].get("packed", 0), len(pack.source_spans))
        self.assertEqual(lifecycle["packed_source_span_count"], len(pack.source_spans))
        self.assertNotIn("silver-forest-77", repr(lifecycle))
