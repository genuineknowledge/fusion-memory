from __future__ import annotations

import unittest
from types import MethodType

from fusion_memory.api.service import MemoryService
from fusion_memory.core.models import Candidate
from fusion_memory.core.models import EvidencePack
from fusion_memory.core.models import Scope
from fusion_memory.core.models import SearchResult
from fusion_memory.retrieval.pipeline import build_pipeline_record


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
