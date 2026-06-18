from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timezone
from types import MethodType

from fusion_memory import MemoryService, Scope
from fusion_memory.core.models import EvidencePack, QueryPlan, SearchResult
from fusion_memory.retrieval.rule_registry import (
    RuleDefinition,
    drain_rule_hits,
    record_rule_hit,
    register_rule,
    registered_rules,
)


class RuleRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        drain_rule_hits()

    def test_record_rule_hit_uses_sha1_prefix_without_raw_text(self) -> None:
        rule = register_rule(
            RuleDefinition(
                rule_id="current_value.stale_history_marker",
                module="fusion_memory.retrieval.evidence_pack",
                purpose="avoid stale current-value evidence",
                category="generic",
                pattern="initially|previously",
            )
        )

        hit = record_rule_hit(
            rule.rule_id,
            query="What is current?",
            text="I initially used SQLite.",
            stage="evidence_pack_filter",
            contributed_candidate_id="span_1",
        )

        self.assertEqual(hit.rule_id, rule.rule_id)
        self.assertEqual(
            hit.text_hash,
            hashlib.sha1("I initially used SQLite.".encode("utf-8")).hexdigest()[:12],
        )
        self.assertNotIn("SQLite", hit.text_hash)
        self.assertEqual(hit.contributed_candidate_id, "span_1")

    def test_drain_rule_hits_returns_and_clears_queue(self) -> None:
        record_rule_hit(
            rule_id="rule.one",
            query="What is current?",
            text="First hit",
            stage="evidence_pack_filter",
        )
        record_rule_hit(
            rule_id="rule.two",
            query="What changed?",
            text="Second hit",
            stage="answer_requirements",
        )

        drained_hits = drain_rule_hits()

        self.assertEqual([hit.rule_id for hit in drained_hits], ["rule.one", "rule.two"])
        self.assertEqual(drain_rule_hits(), [])

    def test_registered_rules_includes_new_definition(self) -> None:
        rule = RuleDefinition(
            rule_id="current_value.current_marker",
            module="fusion_memory.retrieval.current_value",
            purpose="prefer current state evidence",
            category="generic",
            pattern="currently|now",
        )

        registered = register_rule(rule)

        self.assertIn(registered, registered_rules())

    def test_record_rule_hit_copies_metadata_without_storing_raw_text(self) -> None:
        metadata = {
            "confidence": 0.75,
            "details": {"source": "candidate_1"},
            "raw_text": "I initially used SQLite.",
            "decision": "suppress",
            "span_message": "I initially used SQLite.",
            "label": "history-marker",
        }

        hit = record_rule_hit(
            rule_id="current_value.stale_history_marker",
            query="What is current?",
            text="I initially used SQLite.",
            stage="evidence_pack_filter",
            metadata=metadata,
        )

        metadata["confidence"] = 0.10
        metadata["details"] = {"source": "candidate_2"}
        metadata["raw_text"] = "mutated"
        metadata["span_message"] = "mutated"

        self.assertEqual(hit.metadata["confidence"], 0.75)
        self.assertEqual(hit.metadata["details"], {"source": "candidate_1"})
        self.assertEqual(hit.metadata["decision"], "suppress")
        self.assertEqual(hit.metadata["label"], "history-marker")
        self.assertNotEqual(hit.metadata["raw_text"], "I initially used SQLite.")
        self.assertNotEqual(hit.metadata["span_message"], "I initially used SQLite.")
        self.assertRegex(str(hit.metadata["raw_text"]), r"^[0-9a-f]{12}$")
        self.assertRegex(str(hit.metadata["span_message"]), r"^[0-9a-f]{12}$")
        self.assertNotIn("I initially used SQLite.", hit.text_hash)


class RuleInstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        drain_rule_hits()

    def test_current_value_stale_filter_records_rule_hit(self) -> None:
        from fusion_memory.retrieval.evidence_pack import _is_stale_historical_current_value_span

        drain_rule_hits()

        self.assertTrue(_is_stale_historical_current_value_span("I initially used SQLite."))
        hits = drain_rule_hits()

        self.assertTrue(any(hit.rule_id == "current_value.stale_history_marker" for hit in hits))

    def test_cjk_exact_match_rule_hit_sanitizes_phrase_metadata(self) -> None:
        from fusion_memory.api.service_helpers import _cjk_exact_match_phrases

        matches = _cjk_exact_match_phrases("我的默认数据库是什么？", "请记住：我的默认数据库是 PostgreSQL。")

        self.assertIn("默认数据", matches)
        hits = drain_rule_hits()
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].rule_id, "exact_match.cjk_phrase")
        self.assertEqual(hits[0].metadata["decision"], "preserve_language_exact_match")
        self.assertEqual(hits[0].metadata["match_count"], len(matches))
        self.assertRegex(str(hits[0].metadata["phrases"]), r"^[0-9a-f]{12}$")
        self.assertNotIn("默认数据库", str(hits[0].metadata["phrases"]))

    def test_multi_condition_rule_hit_sanitizes_condition_metadata(self) -> None:
        from fusion_memory.api.service_helpers import _matched_query_conditions

        matches = _matched_query_conditions(
            "What OpenClaw adapter requirements mention install and beginner friendly errors?",
            "For the OpenClaw adapter, install must be one command and errors must be beginner friendly.",
        )

        self.assertIn("openclaw", matches)
        self.assertIn("install", matches)
        hits = drain_rule_hits()
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].rule_id, "multi_condition.query_token_match")
        self.assertEqual(hits[0].metadata["decision"], "attach_matched_conditions")
        self.assertEqual(hits[0].metadata["match_count"], len(matches))
        self.assertRegex(str(hits[0].metadata["conditions"]), r"^[0-9a-f]{12}$")
        self.assertNotIn("install", str(hits[0].metadata["conditions"]))

    def test_taxonomy_rule_hit_sanitizes_taxonomy_metadata(self) -> None:
        from fusion_memory.core.models import MemoryEvent, Scope
        from fusion_memory.retrieval.event_graph_selection import _event_ordering_event_relevance

        score = _event_ordering_event_relevance(
            "walk me through the deployment work",
            MemoryEvent(
                event_id="evt-1",
                scope=Scope(workspace_id="ws"),
                event_type="milestone",
                description="Configured Render deployment and Gunicorn server settings.",
                participants=[],
                source_span_ids=[],
            ),
        )

        self.assertGreater(score, 0.0)
        hits = drain_rule_hits()
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].rule_id, "event_ordering.taxonomy_alias_hit")
        self.assertEqual(hits[0].metadata["decision"], "boost_salient_hits")
        self.assertRegex(str(hits[0].metadata["taxonomy_hits"]), r"^[0-9a-f]{12}$")
        self.assertNotIn("deployment", str(hits[0].metadata["taxonomy_hits"]))
        self.assertNotIn("render", str(hits[0].metadata["taxonomy_hits"]))

    def test_search_trace_attaches_rule_hits(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="zh-trace", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "请记住：我的默认数据库是 PostgreSQL，嵌入模型是 qwen0.6B。",
            scope,
            datetime(2026, 6, 18, tzinfo=timezone.utc),
            {"source_uri": "zh1"},
        )

        result = memory.search("我的默认数据库是什么？", scope, {"mode": "fast", "limit": 5})
        trace = memory.debug_trace(result.trace_id, scope)

        self.assertIsNotNone(trace)
        rule_hits = trace.get("rule_hits") if trace else None
        self.assertIsInstance(rule_hits, list)
        self.assertTrue(any(hit.get("rule_id") == "multi_condition.query_token_match" for hit in rule_hits or []))

    def test_answer_context_preserves_search_rule_hits_and_deduplicates_pack_hits(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws", user_id="u", agent_id="a", session_id="s")
        plan = QueryPlan(query="What is current?", query_type="fact_lookup", entities=[], time_constraints=[])

        existing = record_rule_hit(
            "current_value.stale_history_marker",
            query="What is current?",
            text="I initially used SQLite.",
            stage="evidence_pack_filter",
            contributed_candidate_id="span-1",
            metadata={"decision": "drop_stale_history"},
        )
        drain_rule_hits()

        def fake_search(self: MemoryService, query: str, scope: Scope, options: dict | None = None) -> SearchResult:
            return SearchResult(candidates=[], trace_id="trace-search", coverage={})

        def fake_get_trace(self, trace_id: str, scope: Scope, include_session: bool = False) -> dict[str, object]:
            return {"rule_hits": [existing.__dict__], "selected": []}

        def fake_build(
            self,
            query: str,
            plan: QueryPlan,
            candidates: list,
            coverage: dict,
            trace: list,
            token_budget: int | None = None,
        ) -> EvidencePack:
            record_rule_hit(
                "current_value.stale_history_marker",
                query="What is current?",
                text="I initially used SQLite.",
                stage="evidence_pack_filter",
                contributed_candidate_id="span-1",
                metadata={"decision": "drop_stale_history"},
            )
            record_rule_hit(
                "event_ordering.legacy_rescue",
                query="What is current?",
                text="Legacy rescue path.",
                stage="event_ordering_pack",
                contributed_candidate_id="span-2",
                metadata={"decision": "fallback"},
            )
            return EvidencePack(
                query=query,
                answer_policy="test",
                current_views=[],
                entity_profiles=[],
                facts=[],
                events=[],
                source_spans=[],
                conflicts=[],
                coverage={},
                debug_trace=[],
            )

        memory.search = MethodType(fake_search, memory)
        memory.store.get_trace = MethodType(fake_get_trace, memory.store)
        memory.pack_builder.build = MethodType(fake_build, memory.pack_builder)

        pack = memory.answer_context("What is current?", scope, {"mode": "fast"})

        rule_hits = pack.coverage.get("rule_hits")
        self.assertIsInstance(rule_hits, list)
        self.assertEqual(len(rule_hits or []), 2)
        self.assertEqual(
            [hit.get("rule_id") for hit in rule_hits or []],
            ["current_value.stale_history_marker", "event_ordering.legacy_rescue"],
        )
