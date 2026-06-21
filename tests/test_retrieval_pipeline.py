from __future__ import annotations

import unittest
from types import MethodType

from fusion_memory.api.service import MemoryService
from fusion_memory.core.models import Candidate
from fusion_memory.core.models import EvidencePack
from fusion_memory.core.models import QueryPlan
from fusion_memory.core.models import Scope
from fusion_memory.core.models import SearchResult
from fusion_memory.retrieval.pipeline import build_pipeline_record
from fusion_memory.retrieval.pipeline import selected_temporal_relation_summary
from fusion_memory.retrieval.raw_evidence_quota import QuotaResult


class RetrievalPipelineTests(unittest.TestCase):
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

        def fake_candidate_lists(*args, **kwargs):
            return [[base, rescue]]

        def fake_enforce(plan, search_scope, candidates, *, include_session=False):
            return QuotaResult(candidates=list(candidates), selected_span_ids=["base"], required=1, coverage_insufficient=False, backfilled=0)

        def fake_mmr_selected(candidates, selected, limit):
            return [base]

        def fake_preserve_scent(candidates, selected, limit):
            return [rescue, base][:limit]

        memory._candidate_lists = fake_candidate_lists
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
