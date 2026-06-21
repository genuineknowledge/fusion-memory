from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from fusion_memory import MemoryService, Scope
from fusion_memory.api.service import (
    _aggregation_query_context_keys,
    _event_ordering_select_episode_recall_candidates,
    _event_ordering_support_option_signal,
    _event_ordering_milestone_score,
    _key_diverse_aggregation_candidates,
)
from fusion_memory.core.llm import StaticLLMClient
from fusion_memory.core.models import Candidate, QueryPlan
from fusion_memory.ingestion.extractors import classify_milestone, classify_milestones, extract_generic_event_facets, extract_milestone_mentions
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor
from fusion_memory.retrieval.evidence_pack import _exact_answer_candidates, _value_context_is_target_goal, _value_mentions
from fusion_memory.retrieval.mmr import mmr
from fusion_memory.retrieval.rrf import reciprocal_rank_fusion
from fusion_memory.retrieval.temporal_pack import temporal_candidate_table, temporal_mentions
from fusion_memory.retrieval.value_history_pack import build_value_history_table


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class FusionMemoryTests(unittest.TestCase):
    def test_value_history_rows_include_safe_temporal_relations_without_affecting_sort(self) -> None:
        spans = [
            {"id": "old", "speaker": "user", "content": "Previously my snack budget was $20.", "timeline_index": 1, "recency_rank": 2},
            {"id": "new", "speaker": "user", "content": "I updated my snack budget to $35 now.", "timeline_index": 2, "recency_rank": 1},
        ]

        rows = build_value_history_table("what is my current snack budget?", spans, [])

        self.assertEqual(rows[0]["source_span_id"], "new")
        self.assertTrue(rows[0]["temporal_relations"])
        self.assertIn("changed_to", {item["relation_type"] for item in rows[0]["temporal_relations"]})
        self.assertNotIn("content", rows[0]["temporal_relations"][0])

    def test_temporal_candidate_table_includes_safe_relation_summary(self) -> None:
        mention_rows = [
            {
                "id": "span-date",
                "speaker": "user",
                "timeline_index": 1,
                "temporal_mentions": temporal_mentions(
                    "when is the deployment deadline?",
                    "The deployment deadline is July 1, 2026.",
                ),
            }
        ]

        candidates = temporal_candidate_table("when is the deployment deadline?", mention_rows)

        self.assertTrue(candidates)
        self.assertTrue(candidates[0]["temporal_relations"])
        self.assertIn("deadline", {item["relation_type"] for item in candidates[0]["temporal_relations"]})

    def test_event_ordering_dual_shadow_is_disabled_by_default(self) -> None:
        service = MemoryService()
        try:
            self.assertFalse(getattr(service.retrieval_flags, "dual_event_ordering_shadow", False))
            self.assertEqual(getattr(service.retrieval_flags, "production_selector", "legacy"), "legacy")
        finally:
            service.close()

    def test_exact_answer_candidates_rank_user_distance_location_fact(self) -> None:
        plan = QueryPlan(
            query="How far away did I say my parents live from me, and in which town?",
            query_type="factual_exact",
            entities=[],
            time_constraints=[],
        )
        spans = [
            SimpleNamespace(
                span_id="generic",
                span_type="turn",
                speaker="user",
                content="My parents Amy and Kyle are retired and like gardening.",
                source_uri="s",
                turn_id="1",
                timestamp=ts("2026-06-01T10:00:00+00:00"),
            ),
            SimpleNamespace(
                span_id="target",
                span_type="turn",
                speaker="user",
                content="My parents Amy and Kyle live 15 miles away in West Janethaven.",
                source_uri="s",
                turn_id="2",
                timestamp=ts("2026-06-01T10:01:00+00:00"),
            ),
        ]

        candidates = _exact_answer_candidates(plan.query, plan, spans, set())

        self.assertEqual(candidates[0]["source_span_id"], "target")
        self.assertIn("15 miles away", candidates[0]["content"])
        self.assertIn("West Janethaven", candidates[0]["content"])

    def test_exact_answer_candidates_rank_multi_session_count_fraction_user_fact(self) -> None:
        plan = QueryPlan(
            query="How many scenes had I filmed in total by July 5 and how many were left to film after that?",
            query_type="multi_session_reasoning",
            entities=[],
            time_constraints=[],
        )
        spans = [
            SimpleNamespace(
                span_id="assistant-plan",
                span_type="turn",
                speaker="assistant",
                content="Break down the pilot episode into tasks and set filming deadlines before July 10.",
                source_uri="s",
                turn_id="1",
                timestamp=ts("2026-06-01T10:00:00+00:00"),
            ),
            SimpleNamespace(
                span_id="target",
                span_type="turn",
                speaker="user",
                content="By July 5, my pilot was 75% complete, with 12 of 16 scenes filmed.",
                source_uri="s",
                turn_id="2",
                timestamp=ts("2026-06-01T10:01:00+00:00"),
            ),
        ]

        candidates = _exact_answer_candidates(plan.query, plan, spans, set())

        self.assertEqual(candidates[0]["source_span_id"], "target")
        self.assertIn("12 of 16 scenes", candidates[0]["content"])
        self.assertTrue(any(value["text"] == "12 of 16 scenes" for value in candidates[0]["value_mentions"]))

    def test_exact_answer_candidates_bind_person_location_events_to_user_plan(self) -> None:
        plan = QueryPlan(
            query="What two special events am I planning with David, and where will they take place?",
            query_type="multi_session_reasoning",
            entities=[],
            time_constraints=[],
        )
        spans = [
            SimpleNamespace(
                span_id="other-person",
                span_type="turn",
                speaker="user",
                content="Erica's home office has mock interviews scheduled from 6:30 to 7:30 PM.",
                source_uri="s",
                turn_id="1",
                timestamp=ts("2026-06-01T10:00:00+00:00"),
            ),
            SimpleNamespace(
                span_id="david-planned",
                span_type="turn",
                speaker="user",
                content="David planned a surprise picnic at Montserrat Botanical Gardens for my promotion.",
                source_uri="s",
                turn_id="2",
                timestamp=ts("2026-06-01T10:01:00+00:00"),
            ),
            SimpleNamespace(
                span_id="anniversary",
                span_type="turn",
                speaker="user",
                content="I'm nervous about my upcoming anniversary dinner with David at The Coral Reef, East Janethaven.",
                source_uri="s",
                turn_id="3",
                timestamp=ts("2026-06-01T10:02:00+00:00"),
            ),
            SimpleNamespace(
                span_id="getaway",
                span_type="turn",
                speaker="user",
                content="I'm planning a weekend getaway to Blue Bay Resort with David.",
                source_uri="s",
                turn_id="4",
                timestamp=ts("2026-06-01T10:03:00+00:00"),
            ),
        ]

        candidates = _exact_answer_candidates(plan.query, plan, spans, set())
        top_ids = [candidate["source_span_id"] for candidate in candidates[:2]]

        self.assertEqual(top_ids, ["getaway", "anniversary"])
        self.assertNotIn("other-person", top_ids)

    def test_exact_answer_candidates_rank_assistant_work_transition_recommendation(self) -> None:
        plan = QueryPlan(
            query="What steps did you recommend I take to prepare for changing work environment?",
            query_type="assistant_reference",
            entities=[],
            time_constraints=[],
        )
        spans = [
            SimpleNamespace(
                span_id="generic",
                span_type="turn",
                speaker="assistant",
                content="You could reflect on your decision process and list pros and cons before acting.",
                source_uri="s",
                turn_id="1",
                timestamp=ts("2026-06-01T10:00:00+00:00"),
            ),
            SimpleNamespace(
                span_id="target",
                span_type="turn",
                speaker="assistant",
                content=(
                    "Preparing for the Transition: do due diligence. Research the company mission, "
                    "values, and financial health; talk to current employees; clarify workload, "
                    "hours, and performance metrics."
                ),
                source_uri="s",
                turn_id="2",
                timestamp=ts("2026-06-01T10:01:00+00:00"),
            ),
        ]

        candidates = _exact_answer_candidates(plan.query, plan, spans, set())

        self.assertEqual(candidates[0]["source_span_id"], "target")
        self.assertIn("financial health", candidates[0]["content"])
        self.assertIn("current employees", candidates[0]["content"])

    def test_exact_answer_candidates_rank_assistant_writing_timeline_plan(self) -> None:
        plan = QueryPlan(
            query="How did you recommend I structure the process of writing and submitting my scholarship essay?",
            query_type="assistant_reference",
            entities=[],
            time_constraints=[],
        )
        spans = [
            SimpleNamespace(
                span_id="generic",
                span_type="turn",
                speaker="assistant",
                content="It is good to submit grant applications early and stay organized.",
                source_uri="s",
                turn_id="1",
                timestamp=ts("2026-06-01T10:00:00+00:00"),
            ),
            SimpleNamespace(
                span_id="target",
                span_type="turn",
                speaker="assistant",
                content=(
                    "Timeline and Plan: March 15 initial draft, March 25 review, April 5 second draft "
                    "and final edits, April 15 final review, April 20 scholarship submission, with later "
                    "deadlines in mid-May and early June."
                ),
                source_uri="s",
                turn_id="2",
                timestamp=ts("2026-06-01T10:01:00+00:00"),
            ),
        ]

        candidates = _exact_answer_candidates(plan.query, plan, spans, set())

        self.assertEqual(candidates[0]["source_span_id"], "target")
        self.assertIn("initial draft", candidates[0]["content"])
        self.assertIn("scholarship submission", candidates[0]["content"])

    def test_exact_answer_candidates_include_adjacent_assistant_support_for_recommendation_request(self) -> None:
        plan = QueryPlan(
            query="How many unique movies did I plan for April 8?",
            query_type="multi_session_reasoning",
            entities=[],
            time_constraints=[],
        )
        spans = [
            SimpleNamespace(
                span_id="request",
                span_type="turn",
                speaker="user",
                content='What movies would you recommend for April 8, considering "River Quest" and "Garden Bears"?',
                source_uri="session:msg1",
                turn_id="session:msg1",
                timestamp=ts("2026-06-01T10:00:00+00:00"),
            ),
            SimpleNamespace(
                span_id="answer",
                span_type="turn",
                speaker="assistant",
                content='Here are good recommendations:\n1. **"Sky Boats"**\n2. **"Moon Kitchen"**\n3. **"City Parade"**',
                source_uri="session:msg2",
                turn_id="session:msg2",
                timestamp=ts("2026-06-01T10:01:00+00:00"),
            ),
        ]

        candidates = _exact_answer_candidates(plan.query, plan, spans, set())
        by_id = {candidate["source_span_id"]: candidate for candidate in candidates}

        self.assertIn("request", by_id)
        self.assertIn("answer", by_id)
        self.assertEqual(by_id["answer"]["candidate_source"], "adjacent_exact_answer_support")
        self.assertEqual(by_id["request"]["history_index"], 1)
        self.assertEqual(by_id["answer"]["history_index"], 2)

    def test_value_mentions_extracts_product_metrics_and_word_counts(self) -> None:
        mentions = _value_mentions(
            "The new quota is 1,200 calls per day, coverage reached 78%, "
            "and I am scheduled for three days a week remotely. My reading list has 7 series and 4,200 pages. "
            "The weekly target is 1,350 words."
        )
        values = {(item["type"], item["text"].lower()) for item in mentions}

        self.assertIn(("count", "1,200 calls per day"), values)
        self.assertIn(("percentage", "78%"), values)
        self.assertIn(("count", "three days a week"), values)
        self.assertIn(("count", "7 series"), values)
        self.assertIn(("count", "4,200 pages"), values)
        self.assertIn(("count", "1,350 words"), values)

    def test_value_mentions_extracts_chinese_month_day_dates(self) -> None:
        mentions = _value_mentions("现在星桥的当前发布目标是 6 月 30 日完成 alpha，不是之前说的 6 月 20 日。")
        values = {(item["type"], item["text"]) for item in mentions}

        self.assertIn(("date", "6 月 30 日"), values)
        self.assertIn(("date", "6 月 20 日"), values)

    def test_value_context_distinguishes_goal_from_current_value(self) -> None:
        self.assertTrue(_value_context_is_target_goal("I am trying to achieve 100% coverage and currently reached 65%.", "100%"))
        self.assertFalse(_value_context_is_target_goal("I am trying to achieve 100% coverage and currently reached 65%.", "65%"))

    def test_key_diverse_aggregation_keeps_contextual_duplicate_support(self) -> None:
        user_choice = Candidate(
            id="choice",
            type="span",
            text='I think "Moana" and "Zootopia" sound perfect.',
            source="aggregation_coverage_raw",
            scores={},
            source_span_ids=["choice"],
            metadata={"speaker": "user", "aggregation_keys": ["title:moana", "title:zootopia"]},
        )
        dated_schedule = Candidate(
            id="schedule",
            type="span",
            text='Schedule for April 8, 2024: 10:00 AM "Moana", 11:45 AM "Zootopia".',
            source="aggregation_coverage_raw",
            scores={},
            source_span_ids=["schedule"],
            metadata={"speaker": "assistant", "aggregation_keys": ["title:moana", "title:zootopia"]},
        )

        selected = _key_diverse_aggregation_candidates(
            [
                (0.45, user_choice, ((1, "session"),), ((0, 112),)),
                (0.90, dated_schedule, ((1, "session"),), ((0, 113),)),
            ],
            limit=4,
        )

        self.assertEqual([candidate.id for candidate in selected], ["choice", "schedule"])

    def test_aggregation_context_keys_preserve_exploration_features_without_query_scene_terms(self) -> None:
        query = "How many different book series or genres have I mentioned wanting to explore across my conversations?"

        self.assertIn(
            "query_context:feature:event",
            _aggregation_query_context_keys(
                query,
                "I'm trying to decide on a must-read fiction series and have been invited to co-host a live chat on sci-fi series.",
            ),
        )
        self.assertIn(
            "query_context:feature:budget",
            _aggregation_query_context_keys(
                query,
                "With a $120 budget for print editions from Montserrat Books, which fiction series should I buy?",
            ),
        )
        self.assertIn(
            "query_context:feature:social",
            _aggregation_query_context_keys(
                query,
                "Douglas and I want a series that we can read together and discuss.",
            ),
        )

    def test_key_diverse_aggregation_promotes_distinct_context_scenes(self) -> None:
        store_budget = Candidate(
            id="store",
            type="span",
            text="With a $120 budget from Montserrat Books, I need fiction series options.",
            source="aggregation_coverage_raw",
            scores={"score": 0.95, "aggregation_focus": 0.8, "aggregation_signal": 0.8},
            source_span_ids=["store"],
            metadata={"speaker": "user", "aggregation_keys": ["title:fiction", "query_context:feature:budget"]},
        )
        store_budget_duplicate = Candidate(
            id="store-duplicate",
            type="span",
            text="More Montserrat Books budget options for fiction series.",
            source="aggregation_coverage_raw",
            scores={"score": 0.90, "aggregation_focus": 0.7, "aggregation_signal": 0.7},
            source_span_ids=["store-duplicate"],
            metadata={"speaker": "assistant", "aggregation_keys": ["title:fiction", "query_context:feature:budget"]},
        )
        live_event = Candidate(
            id="live-event",
            type="span",
            text="I chose The Dune Series for the live chat on sci-fi series with Wyatt.",
            source="aggregation_coverage_raw",
            scores={"score": 0.30, "aggregation_focus": 0.45, "aggregation_signal": 0.33},
            source_span_ids=["live-event"],
            metadata={"speaker": "user", "aggregation_keys": ["title:dune", "query_context:feature:event"]},
        )

        selected = _key_diverse_aggregation_candidates(
            [
                (0.95, store_budget, ((1, "s"),), ((0, 1),)),
                (0.90, store_budget_duplicate, ((1, "s"),), ((0, 2),)),
                (0.30, live_event, ((1, "s"),), ((0, 3),)),
            ],
            limit=2,
        )

        self.assertEqual([candidate.id for candidate in selected], ["store", "live-event"])

    def test_topic_scope_filter_does_not_remove_broad_exploration_context_scenes(self) -> None:
        service = MemoryService()
        query = "How many different book series or genres have I mentioned wanting to explore across my conversations?"
        plan = QueryPlan(query=query, query_type="multi_session_reasoning", entities=[], time_constraints=[])
        selected = [
            Candidate(
                id="store",
                type="span",
                text="With a $120 budget from Montserrat Books, I need fiction series options.",
                source="aggregation_coverage_raw",
                scores={},
                source_span_ids=["store"],
                metadata={"topic_group": "books", "aggregation_keys": ["query_context:feature:budget"]},
            ),
            Candidate(
                id="live-event",
                type="span",
                text="I chose The Dune Series for the live chat on sci-fi series with Wyatt.",
                source="aggregation_coverage_raw",
                scores={},
                source_span_ids=["live-event"],
                metadata={"topic_group": "events", "aggregation_keys": ["query_context:feature:event"]},
            ),
        ]
        candidates = [
            Candidate(
                id="topic-anchor",
                type="span",
                text="Book series topic anchor",
                source="topic_scope_raw",
                scores={},
                source_span_ids=["topic-anchor"],
                metadata={"topic_group": "books"},
            ),
            *selected,
        ]

        filtered = service._apply_topic_scope_filter(query, plan, candidates, selected, limit=2)

        self.assertEqual([candidate.id for candidate in filtered], ["store", "live-event"])

    def test_event_ordering_preserve_reserves_episode_recall_slots(self) -> None:
        service = MemoryService()
        query = "Can you list the order in which I brought up different support options and strategies for the dashboard project across conversations? Mention ONLY and ONLY three items."
        anchors = [
            Candidate(
                id=f"anchor-{index}",
                type="span",
                text=f"Dashboard project anchor {index}",
                source="event_ordering_coverage",
                scores={"score": 0.9},
                source_span_ids=[f"anchor-{index}"],
                metadata={
                    "speaker": "user",
                    "timeline_role": "user_aspect_anchor",
                    "coverage_origin": "query_required_facet",
                    "topic_group": "dashboard",
                    "source_uri": f"session:{index}",
                    "turn_id": f"msg{index}",
                },
            )
            for index in range(6)
        ]
        episodes = [
            Candidate(
                id="episode-a",
                type="span",
                text="I added the chart export option with CSV support.",
                source="event_ordering_episode_recall+l0_raw_hybrid",
                scores={"score": 0.62, "event_episode_signal": 0.42, "event_detail_signal": 0.32, "event_facet_coverage": 0.20},
                source_span_ids=["episode-a"],
                metadata={
                    "speaker": "user",
                    "topic_group": "reports",
                    "event_ordering_facet_hits": ["dashboard"],
                    "source_uri": "session:7",
                    "turn_id": "msg7",
                },
            ),
            Candidate(
                id="episode-b",
                type="span",
                text="I configured role-based dashboard permissions for admins.",
                source="event_ordering_episode_recall",
                scores={"score": 0.59, "event_episode_signal": 0.40, "event_detail_signal": 0.38, "event_facet_coverage": 0.18},
                source_span_ids=["episode-b"],
                metadata={
                    "speaker": "user",
                    "topic_group": "security",
                    "event_ordering_facet_hits": ["dashboard"],
                    "source_uri": "session:8",
                    "turn_id": "msg8",
                },
            ),
        ]

        preserved = service._preserve_event_ordering_events(query, anchors + episodes, anchors[:4], limit=4)

        self.assertIn("episode-a", [candidate.id for candidate in preserved])
        self.assertIn("episode-b", [candidate.id for candidate in preserved])
        self.assertLess([candidate.id for candidate in preserved].count("anchor-0") + [candidate.id for candidate in preserved].count("anchor-1") + [candidate.id for candidate in preserved].count("anchor-2") + [candidate.id for candidate in preserved].count("anchor-3"), 4)

    def test_event_ordering_topic_scope_filter_keeps_typed_episode_recall(self) -> None:
        service = MemoryService()
        query = "Can you list the order in which I brought up different aspects of the dashboard project across conversations?"
        plan = QueryPlan(query=query, query_type="event_ordering", entities=[], time_constraints=[])
        selected = [
            Candidate(
                id="anchor",
                type="span",
                text="Dashboard project topic anchor",
                source="event_ordering_coverage",
                scores={},
                source_span_ids=["anchor"],
                metadata={"topic_group": "dashboard", "timeline_role": "user_aspect_anchor", "speaker": "user"},
            ),
            Candidate(
                id="episode",
                type="span",
                text="I added CSV export and permission details for the dashboard.",
                source="event_ordering_episode_recall+l0_raw_hybrid",
                scores={"event_episode_signal": 0.40, "event_detail_signal": 0.32, "event_facet_coverage": 0.18},
                source_span_ids=["episode"],
                metadata={
                    "topic_group": "reporting",
                    "speaker": "user",
                    "event_ordering_facet_hits": ["dashboard"],
                },
            ),
        ]
        candidates = [
            selected[0],
            selected[1],
            Candidate(
                id="replacement",
                type="span",
                text="Another dashboard anchor.",
                source="event_ordering_coverage",
                scores={},
                source_span_ids=["replacement"],
                metadata={"topic_group": "dashboard", "timeline_role": "user_aspect_anchor", "speaker": "user"},
            ),
        ]

        filtered = service._apply_topic_scope_filter(query, plan, candidates, selected, limit=2)

        self.assertEqual([candidate.id for candidate in filtered], ["anchor", "episode"])

    def test_event_ordering_post_preservation_topic_filter_reports_dropped_graph_anchor(self) -> None:
        service = MemoryService()
        query = "Can you list the order in which I brought up different sneaker shopping experiences?"
        plan = QueryPlan(query=query, query_type="event_ordering", entities=[], time_constraints=[])
        selected = [
            Candidate(
                id="anchor",
                type="span",
                text="I compared sneaker styles for the festival.",
                source="event_ordering_coverage",
                scores={},
                source_span_ids=["anchor"],
                metadata={"topic_group": "sneakers", "timeline_role": "user_aspect_anchor", "speaker": "user"},
            ),
            Candidate(
                id="graph-off-topic",
                type="event",
                text="track with my savings goals",
                source="event_ordering_persisted_graph",
                scores={},
                source_span_ids=["savings"],
                metadata={"must_preserve_reason": ["graph_chronology_anchor"], "evidence_role": "answer"},
            ),
        ]

        filtered, dropped = service._apply_event_ordering_post_preservation_topic_scope_filter(
            query,
            plan,
            selected,
            selected,
            limit=2,
        )

        self.assertEqual([candidate.id for candidate in filtered], ["anchor"])
        self.assertEqual(dropped[0]["candidate_id"], "graph-off-topic")
        self.assertEqual(dropped[0]["reason"], "topic_scope_filter")
        self.assertEqual(dropped[0]["must_preserve_reasons"], ["graph_chronology_anchor"])

    def test_event_ordering_preserve_episode_recall_uses_time_bucket_coverage(self) -> None:
        service = MemoryService()
        query = "Can you list the order in which I brought up different strategies and support options for managing my workload throughout our conversations in order? Mention ONLY and ONLY five items."
        anchors = [
            Candidate(
                id=f"anchor-{index}",
                type="span",
                text=f"Workload strategy anchor {index}",
                source="event_ordering_coverage",
                scores={"score": 0.9},
                source_span_ids=[f"anchor-{index}"],
                metadata={
                    "speaker": "user",
                    "timeline_role": "user_aspect_anchor",
                    "coverage_origin": "query_required_facet",
                    "topic_group": "workload",
                    "source_uri": f"batch1:msg{index}",
                    "turn_id": f"msg{index}",
                },
            )
            for index in range(5)
        ]
        episodes = [
            Candidate(
                id=f"episode-{index}",
                type="span",
                text=f"Workload support strategy episode {index}",
                source="event_ordering_episode_recall+l0_raw_hybrid",
                scores={
                    "score": 0.30 + index * 0.01,
                    "event_episode_signal": 0.40,
                    "event_detail_signal": 0.40,
                    "event_facet_coverage": 0.18,
                },
                source_span_ids=[f"episode-{index}"],
                metadata={
                    "speaker": "user",
                    "topic_group": "workload",
                    "event_ordering_facet_hits": ["manage"],
                    "source_uri": f"batch{index + 1}:msg{index * 10}",
                    "turn_id": f"msg{index * 10}",
                },
            )
            for index in range(6)
        ]

        preserved = service._preserve_event_ordering_events(query, anchors + episodes, anchors[:4], limit=8)
        episode_ids = [candidate.id for candidate in preserved if candidate.id.startswith("episode-")]

        self.assertEqual(len(episode_ids), 4)
        self.assertIn("episode-5", episode_ids)

    def test_event_ordering_support_option_signal_promotes_resource_episode(self) -> None:
        query = "Can you list the order in which I brought up different strategies and support options for managing my workload throughout our conversations in order? Mention ONLY and ONLY five items."
        generic_strategy = Candidate(
            id="generic-strategy",
            type="span",
            text="I felt good about my collaboration strategies after the meeting.",
            source="event_ordering_episode_recall",
            scores={"score": 0.42, "event_episode_signal": 0.35, "event_detail_signal": 0.20, "event_facet_coverage": 0.27},
            source_span_ids=["generic-strategy"],
            metadata={"speaker": "user", "source_uri": "batch4:msg218", "turn_id": "msg218"},
        )
        support_option = Candidate(
            id="support-option",
            type="span",
            text="I hired a part-time assistant for 20 hours/week at $25/hour after a mentor recommended hiring one to help manage my schedule.",
            source="event_ordering_episode_recall+l0_raw_hybrid",
            scores={
                "score": 0.34,
                "event_episode_signal": 0.35,
                "event_detail_signal": 0.80,
                "event_support_option_signal": _event_ordering_support_option_signal(
                    query,
                    "I hired a part-time assistant for 20 hours/week at $25/hour after a mentor recommended hiring one to help manage my schedule.",
                ),
                "event_facet_coverage": 0.18,
            },
            source_span_ids=["support-option"],
            metadata={"speaker": "user", "source_uri": "batch4:msg202", "turn_id": "msg202"},
        )

        selected = _event_ordering_select_episode_recall_candidates(
            [(generic_strategy.scores["score"], generic_strategy), (support_option.scores["score"] + 0.30, support_option)],
            limit=1,
            requested=5,
        )

        self.assertGreater(_event_ordering_support_option_signal(query, support_option.text), 0.60)
        self.assertEqual([candidate.id for candidate in selected], ["support-option"])

    def test_event_ordering_raw_facet_preserve_keeps_existing_strong_episode(self) -> None:
        service = MemoryService()
        query = "Can you list the order in which I brought up different strategies and support options for managing my workload throughout our conversations in order? Mention ONLY and ONLY five items."
        strong_episode = Candidate(
            id="support-option",
            type="span",
            text="I hired a part-time assistant for 20 hours/week at $25/hour after a mentor recommended hiring one to help manage my schedule.",
            source="event_ordering_episode_recall+l0_raw_hybrid",
            scores={
                "score": 0.34,
                "event_episode_signal": 0.35,
                "event_detail_signal": 0.80,
                "event_support_option_signal": 0.94,
                "event_facet_coverage": 0.18,
            },
            source_span_ids=["support-option"],
            metadata={"speaker": "user", "source_uri": "batch4:msg202", "turn_id": "msg202", "event_ordering_facet_hits": ["manage"]},
        )
        raw_facets = [
            Candidate(
                id=f"raw-{index}",
                type="span",
                text=f"I discussed workload strategies and manage options {index}.",
                source="event_ordering_episode_recall",
                scores={"score": 0.45, "event_episode_signal": 0.40, "event_detail_signal": 0.30, "event_facet_coverage": 0.27},
                source_span_ids=[f"raw-{index}"],
                metadata={"speaker": "user", "source_uri": f"batch{index}:msg{index}", "turn_id": f"msg{index}", "event_ordering_facet_hits": ["manage", "strategies"]},
            )
            for index in range(8)
        ]

        preserved = service._preserve_event_ordering_raw_facets(query, [strong_episode, *raw_facets], [strong_episode], limit=5)

        self.assertIn("support-option", [candidate.id for candidate in preserved])

    def test_scope_required_for_add(self) -> None:
        memory = MemoryService()
        with self.assertRaises(ValueError):
            memory.add("Remember that I prefer Qdrant.", Scope())

    def test_scope_isolation(self) -> None:
        memory = MemoryService()
        scope_a = Scope(workspace_id="w", user_id="u1", agent_id="a")
        scope_b = Scope(workspace_id="w", user_id="u2", agent_id="a")
        memory.add("I prefer Qdrant for Atlas retrieval.", scope_a, ts("2026-06-01T10:00:00+00:00"))
        result = memory.search("Qdrant Atlas", scope_b)
        self.assertEqual(result.candidates, [])

    def test_search_trace_contains_retrieval_pipeline_sections(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws-trace", user_id="u", agent_id="a")
        try:
            memory.add({"role": "user", "content": "I now prefer PostgreSQL for the memory database."}, scope)
            result = memory.search("What database do I currently prefer?", scope)
            trace = memory.store.get_trace(result.trace_id, scope)

            retrieval_trace = trace["retrieval_trace"]
            self.assertIn("query_understanding", retrieval_trace)
            self.assertIn("candidate_recall", retrieval_trace)
            self.assertIn("candidate_fusion", retrieval_trace)
            self.assertIn("evidence_output", retrieval_trace)
            self.assertIn("pipeline_layers", retrieval_trace)
            self.assertIn("QueryUnderstanding", retrieval_trace["pipeline_layers"])
            self.assertIn("CandidateRecall", retrieval_trace["pipeline_layers"])
        finally:
            memory.close()

    def test_preference_update_writes_source_fact_relation_and_current_view(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        first = memory.add("For Atlas, I prefer Chroma because it is simple.", scope, ts("2026-06-01T10:00:00+00:00"))
        second = memory.add(
            "We switched Atlas to Qdrant. Remember that Qdrant is now preferred.",
            scope,
            ts("2026-06-08T10:00:00+00:00"),
        )

        self.assertTrue(first.accepted_fact_ids)
        self.assertTrue(second.accepted_fact_ids)
        relations = memory.store.list_fact_relations(relation_type="supersedes")
        self.assertTrue(relations)
        latest_fact = memory.store.get_fact(second.accepted_fact_ids[0])
        self.assertIsNotNone(latest_fact)
        self.assertTrue(latest_fact.source_span_ids)

        pack = memory.answer_context("What do I currently prefer for Atlas?", scope)
        self.assertTrue(pack.current_views)
        self.assertTrue("Qdrant" in str(pack.current_views) or "Qdrant" in str(pack.facts))
        self.assertTrue(pack.source_spans)

    def test_speaker_attribution_rejects_assistant_suggestion_as_user_preference(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        result = memory.add(
            [
                {"role": "assistant", "content": "You may want to use PostgreSQL."},
                {"role": "user", "content": "Good idea, but don't remember that as my preference yet."},
            ],
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        facts = memory.store.list_facts(scope)
        user_preference_facts = [fact for fact in facts if fact.category == "preference"]
        self.assertEqual(user_preference_facts, [])
        trace = memory.debug_trace(result.trace_id)
        self.assertIsNotNone(trace)
        self.assertTrue("explicit_negative_memory_instruction" in str(trace) or "speaker_attribution" in str(trace))

    def test_temporal_ordering_builds_events_and_raw_evidence(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I tested BM25 yesterday.", scope, ts("2026-06-03T12:00:00+00:00"))
        memory.add("After the BM25 test, I added dense retrieval.", scope, ts("2026-06-05T12:00:00+00:00"))

        events = memory.store.list_events(scope)
        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(any(event.time_start and event.time_start.date().isoformat() == "2026-06-02" for event in events))

        pack = memory.answer_context("Which happened before dense retrieval?", scope)
        self.assertGreaterEqual(pack.coverage["source_span_quota_required"], 4)
        self.assertTrue(pack.source_spans)
        self.assertTrue(any("BM25" in span["content"] for span in pack.source_spans))

    def test_mmr_preserves_relevance_while_penalizing_duplicate_sources(self) -> None:
        candidates = [
            Candidate(id="a", type="span", text="budget tracker auth setup", source="test", scores={"utility_score": 1.0}, source_span_ids=["s1"]),
            Candidate(id="b", type="span", text="budget tracker auth setup duplicate", source="test", scores={"utility_score": 0.95}, source_span_ids=["s1"]),
            Candidate(id="c", type="span", text="deployment render gunicorn", source="test", scores={"utility_score": 0.80}, source_span_ids=["s2"]),
        ]

        selected = mmr(candidates, limit=2, lambda_=0.72)

        self.assertEqual(selected[0].id, "a")
        self.assertEqual(selected[1].id, "c")

    def test_event_ordering_pack_includes_timeline_indices_for_milestones(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I brought up Core functionality for the budget tracker.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I mentioned Transaction error handling for the budget tracker.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I discussed Security and deployment for the budget tracker.", scope, ts("2026-06-03T10:00:00+00:00"))

        events = memory.store.list_events(scope)
        self.assertGreaterEqual(len(events), 3)
        self.assertTrue(any(event.event_type == "milestone" for event in events))

        pack = memory.answer_context(
            "Can you list the order in which I brought up different aspects of developing my personal budget tracker throughout our conversations, in order?",
            scope,
            budget={"limit": 8},
        )

        self.assertEqual(pack.coverage["query_type"], "event_ordering")
        self.assertEqual(pack.coverage["timeline_span_count"], len(pack.source_spans))
        self.assertTrue(pack.source_spans)
        self.assertEqual([span["timeline_index"] for span in pack.source_spans], list(range(1, len(pack.source_spans) + 1)))
        timestamps = [span["timestamp"] for span in pack.source_spans]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_event_ordering_timeline_recall_boosts_user_milestones(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("Here is a general assistant recap about the budget tracker.", scope, ts("2026-06-01T09:00:00+00:00"))
        memory.add("I'm setting up the initial project schema and local server for the budget tracker.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I implemented transaction CRUD response handling and validation errors.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I configured Render deployment with Gunicorn workers and port 10000.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("I expanded integration tests for endpoint coverage.", scope, ts("2026-06-04T10:00:00+00:00"))
        memory.add("I reviewed security auth changes before deployment.", scope, ts("2026-06-05T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you walk me through the order in which I brought up different aspects of my app development and deployment across our conversations?",
            scope,
            budget={"limit": 8},
        )

        contents = " ".join(span["content"] for span in pack.source_spans).lower()
        self.assertIn("initial project schema", contents)
        self.assertIn("transaction crud", contents)
        self.assertIn("render deployment", contents)
        self.assertIn("integration tests", contents)
        self.assertIn("security auth", contents)
        self.assertTrue(any("event_ordering_coverage" in item["source"] for item in pack.debug_trace))

    def test_event_ordering_coverage_segments_user_aspects_before_rerank(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "For my personal budget tracker, I brought up:",
                        "1. Core functionality: login, income tracking, expenses, and analytics.",
                        "2. Database schema: users, transactions, categories, and recurring payments.",
                    ]
                ),
                "turn_id": "beam:test:1:batch1:msg1",
                "timestamp": "2026-06-01T10:00:00+00:00",
            },
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg1"},
        )
        memory.add(
            {
                "role": "assistant",
                "content": "That plan gives the app a solid foundation before implementation.",
                "turn_id": "beam:test:1:batch1:msg2",
                "timestamp": "2026-06-01T10:05:00+00:00",
            },
            scope,
            ts("2026-06-01T10:05:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg2"},
        )
        memory.add(
            {
                "role": "user",
                "content": "Next I brought up transaction CRUD response handling and validation errors.",
                "turn_id": "beam:test:1:batch1:msg3",
                "timestamp": "2026-06-02T10:00:00+00:00",
            },
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg3"},
        )
        memory.add(
            {
                "role": "user",
                "content": "Finally I brought up deployment on Render with Gunicorn workers and port configuration.",
                "turn_id": "beam:test:1:batch1:msg4",
                "timestamp": "2026-06-03T10:00:00+00:00",
            },
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg4"},
        )
        memory.add(
            {
                "role": "user",
                "content": "For stress management, I brought up no-work Sundays and emergency savings.",
                "turn_id": "beam:test:16:batch1:msg1",
                "timestamp": "2026-06-01T09:00:00+00:00",
            },
            scope,
            ts("2026-06-01T09:00:00+00:00"),
            {"source_uri": "beam:test:16:batch1:msg1"},
        )

        pack = memory.answer_context(
            "Can you list the order in which I brought up different aspects of developing my personal budget tracker throughout our conversations? Mention ONLY and ONLY four items.",
            scope,
            budget={"limit": 10},
        )

        anchors = [span for span in pack.source_spans if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor"]
        self.assertGreaterEqual(len(anchors), 3)
        anchor_text = " ".join(span["content"] for span in anchors[:4]).lower()
        self.assertTrue("core functionality" in anchor_text or "initial project setup" in anchor_text or "database schema" in anchor_text)
        self.assertIn("transaction crud", anchor_text)
        self.assertIn("deployment", anchor_text)
        self.assertNotIn("stress management", anchor_text)
        self.assertEqual([span["timeline_index"] for span in anchors[:3]], [1, 2, 3])
        self.assertTrue(all(span["speaker"] == "user" for span in anchors[:4]))
        self.assertTrue(any(span.get("timeline_role") == "supporting_context" and span["speaker"] == "assistant" for span in pack.source_spans))

    def test_event_ordering_pack_preserves_event_timeline_graph_candidates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I'm trying to set up the database schema and local server for the initial project.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I'm trying to implement transaction CRUD with response handling and validation errors.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I'm configuring deployment on Render with Gunicorn workers and port 10000.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("I expanded integration tests for endpoint coverage and the test suite.", scope, ts("2026-06-04T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you walk me through the order in which I brought up different aspects of my app development and deployment across our conversations?",
            scope,
            budget={"limit": 12},
        )

        groups = [event.get("milestone_group") for event in pack.events if event.get("milestone_group")]
        self.assertIn("initial_project_setup", groups)
        self.assertTrue(any(group in groups for group in {"transaction_crud_implementation", "transaction_error_handling"}))
        self.assertIn("deployment_configuration", groups)
        self.assertIn("integration_test_coverage", groups)
        self.assertEqual([event["timeline_index"] for event in pack.events], list(range(1, len(pack.events) + 1)))
        self.assertTrue(any(item["type"] == "event" and "event_timeline_graph" in item["source"] for item in pack.debug_trace))
        self.assertNotIn("event_ordering_graph", pack.coverage)
        self.assertFalse(
            any(
                item["source"].startswith("event_ordering_graph")
                for item in pack.debug_trace
            )
        )

    def test_event_ordering_pack_tracks_graph_shadow_metrics_when_legacy_fallback_wins(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I set up the initial project schema and local server for the app.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I configured deployment on Render with Gunicorn workers and port 10000.", scope, ts("2026-06-02T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you walk me through the order in which I brought up different aspects of my app development and deployment across our conversations?",
            scope,
            budget={"limit": 1},
        )

        self.assertNotIn("event_ordering_graph", pack.coverage)
        shadow_coverage = pack.coverage.get("event_ordering_shadow")
        self.assertIsInstance(shadow_coverage, dict)
        self.assertEqual(shadow_coverage.get("selected_span_source"), "legacy")

    def test_event_ordering_shadow_metrics_track_dropped_graph_by_path_not_id(self) -> None:
        memory = MemoryService()
        graph_candidate = Candidate(
            id="shared-event",
            type="event",
            text="Graph event",
            source="event_ordering_graph_selector",
            scores={"score": 1.0},
            source_span_ids=["s1"],
            metadata={},
        )
        legacy_candidate = Candidate(
            id="shared-event",
            type="event",
            text="Legacy event",
            source="event_timeline_graph",
            scores={"score": 1.0},
            source_span_ids=["s1"],
            metadata={},
        )

        coverage = memory._event_ordering_shadow_coverage([[graph_candidate, legacy_candidate]], [legacy_candidate])

        graph_coverage = coverage["event_ordering_graph"]
        self.assertEqual(graph_coverage["graph_candidate_count"], 1)
        self.assertEqual(graph_coverage["legacy_candidate_count"], 1)
        self.assertEqual(graph_coverage["selected_span_source"], "legacy")
        self.assertTrue(graph_coverage["graph_candidates_dropped_by_filters"])
        self.assertEqual(graph_coverage["dropped_count"], 1)

    def test_event_ordering_graph_coverage_propagates_safe_selector_telemetry(self) -> None:
        memory = MemoryService()
        graph_candidate = Candidate(
            id="graph-event",
            type="event",
            text="Graph event",
            source="event_ordering_persisted_graph",
            scores={"score": 1.0},
            source_span_ids=["s1"],
            metadata={
                "graph_selector_telemetry": {
                    "cluster_expanded_topic_ids": ["topic-b"],
                    "selected_topic_count": 4,
                    "topic_ids": ["topic-a", "topic-b"],
                    "cluster_labels": ["triangle geometry"],
                    "raw_text": "do not expose",
                }
            },
        )

        coverage = memory._event_ordering_shadow_coverage([[graph_candidate]], [graph_candidate])

        graph_coverage = coverage["event_ordering_graph"]
        self.assertEqual(graph_coverage["cluster_expanded_topic_ids"], ["topic-b"])
        self.assertEqual(graph_coverage["selected_topic_count"], 4)
        self.assertIsNone(graph_coverage["graph_ordered_legacy_recall_count"])
        self.assertNotIn("topic_ids", graph_coverage)
        self.assertNotIn("cluster_labels", graph_coverage)
        self.assertNotIn("raw_text", graph_coverage)

    def test_event_ordering_dual_shadow_reports_without_replacing_selected_candidates(self) -> None:
        class Flags:
            dual_event_ordering_shadow = True
            production_selector = "legacy"

        service = MemoryService(retrieval_flags=Flags())
        scope = Scope(workspace_id="ws-dual-shadow", user_id="u", agent_id="a")
        try:
            service.add({"role": "user", "content": "First I set up schema. Then I implemented transaction CRUD."}, scope)
            result = service.search(
                "What order did I discuss the budget tracker work?",
                scope,
                {"query_type_hint": "event_ordering", "limit": 5},
            )

            self.assertIn("event_ordering_dual_shadow", result.coverage)
            shadow = result.coverage["event_ordering_dual_shadow"]
            self.assertEqual(shadow["selected_driver"], "dual_shadow")
            self.assertIn("candidate_count", shadow)
            self.assertIn("sources", shadow)
            self.assertGreaterEqual(len(result.candidates), 1)
        finally:
            service.close()

    def test_dual_shadow_does_not_replace_event_ordering_selected_candidates(self) -> None:
        class Flags:
            dual_event_ordering_shadow = True
            production_selector = "legacy"

        service = MemoryService(retrieval_flags=Flags())
        scope = Scope(workspace_id="shadow-default", user_id="u", agent_id="a")
        try:
            service.add(
                {
                    "role": "user",
                    "content": "First I created the schema. Then I added CRUD. Finally I tested errors.",
                },
                scope,
            )
            result = service.search(
                "What order did I describe the work?",
                scope,
                {"query_type_hint": "event_ordering", "limit": 5},
            )
        finally:
            service.close()

        self.assertIn("event_ordering_dual_shadow", result.coverage)
        self.assertNotEqual(result.coverage["event_ordering_dual_shadow"].get("selected_driver"), "production")
        self.assertTrue(result.candidates)

    def test_event_ordering_default_search_does_not_select_graph_candidates(self) -> None:
        service = MemoryService()
        scope = Scope(workspace_id="ws-legacy-default", user_id="u", agent_id="a")
        graph_candidate = Candidate(
            id="graph-default-candidate",
            type="event",
            text="Graph candidate should stay out of production selection.",
            source="event_ordering_graph_selector",
            scores={"score": 10.0, "utility_score": 10.0},
            source_span_ids=["span-graph"],
            metadata={},
        )
        try:
            service.add({"role": "user", "content": "First I set up schema. Then I implemented transaction CRUD."}, scope)
            service._event_ordering_graph_selector_candidates = lambda query, scope, limit, include_session=False: [graph_candidate]

            result = service.search(
                "What order did I describe the work?",
                scope,
                {"query_type_hint": "event_ordering", "limit": 5},
            )
        finally:
            service.close()

        self.assertTrue(result.candidates)
        self.assertFalse(any("event_ordering_graph" in candidate.source for candidate in result.candidates))

    def test_event_ordering_search_pipeline_trace_includes_temporal_relations_layer_for_graph_candidate(self) -> None:
        service = MemoryService()
        scope = Scope(workspace_id="ws-graph-pipeline-trace", user_id="u", agent_id="a")
        graph_candidate = Candidate(
            id="graph-trace-candidate",
            type="event",
            text="First I created the schema. Then I implemented transaction CRUD.",
            source="event_ordering_graph_selector",
            scores={"score": 10.0, "utility_score": 10.0},
            source_span_ids=["span-graph-trace"],
            metadata={
                "temporal_relations": [
                    {
                        "relation_type": "before",
                        "confidence": 0.72,
                        "reason_code": "explicit_order_marker",
                        "role_labels": ["earlier_event"],
                        "source_span_ids": ["span-graph-trace"],
                    }
                ]
            },
        )
        try:
            service.add({"role": "user", "content": "First I created the schema. Then I implemented transaction CRUD."}, scope)
            service._event_ordering_graph_selector_candidates = lambda query, scope, limit, include_session=False: [graph_candidate]
            service._candidate_lists = lambda *args, **kwargs: [[graph_candidate]]

            result = service.search(
                "What order did I describe the work?",
                scope,
                {"query_type_hint": "event_ordering", "limit": 5},
            )
        finally:
            service.close()

        temporal_layer = result.coverage["pipeline_trace"]["pipeline_layers"]["TemporalRelations"]
        self.assertEqual(temporal_layer["relation_count"], 1)
        self.assertEqual(temporal_layer["relation_types"], ["before"])
        self.assertEqual(temporal_layer["role_labels"], ["earlier_event"])
        self.assertEqual(temporal_layer["reason_codes"], ["explicit_order_marker"])
        self.assertEqual(temporal_layer["source_span_count"], 1)
        self.assertEqual(temporal_layer["source_span_ids"], ["span-graph-trace"])
        self.assertNotIn("text", temporal_layer)
        self.assertNotIn("confidence", temporal_layer)

    def test_event_ordering_dual_shadow_reports_fallback_graph_candidates(self) -> None:
        class Flags:
            dual_event_ordering_shadow = True
            production_selector = "legacy"

        service = MemoryService(retrieval_flags=Flags())
        scope = Scope(workspace_id="ws-dual-shadow-graph", user_id="u", agent_id="a")
        fallback_graph_candidate = Candidate(
            id="graph-fallback-span",
            type="event",
            text="Fallback graph candidate",
            source="event_ordering_graph_selector_event",
            scores={"score": 1.0},
            source_span_ids=["span-1"],
            metadata={"graph_fallback": True},
        )
        legacy_candidate = Candidate(
            id="legacy-span",
            type="event",
            text="Legacy candidate",
            source="event_timeline_graph",
            scores={"score": 1.0},
            source_span_ids=["span-2"],
            metadata={},
        )
        try:
            service.add({"role": "user", "content": "First I set up schema. Then I implemented transaction CRUD."}, scope)
            service._event_ordering_graph_selector_candidates = lambda query, scope, limit, include_session=False: [fallback_graph_candidate]
            service._event_ordering_legacy_recall_for_shadow = lambda query, scope, plan, limit, include_session: ([legacy_candidate], [legacy_candidate.source])

            result = service.search(
                "What order did I discuss the budget tracker work?",
                scope,
                {"query_type_hint": "event_ordering", "limit": 5},
            )

            shadow = result.coverage["event_ordering_dual_shadow"]
            self.assertEqual(shadow["graph_candidate_count"], 1)
            self.assertIn("event_ordering_graph_selector_event", shadow["sources"])
            self.assertIn("event_timeline_graph", shadow["sources"])
        finally:
            service.close()

    def test_event_ordering_dual_shadow_candidate_count_dedupes_overlapping_graph_and_legacy_candidates(self) -> None:
        class Flags:
            dual_event_ordering_shadow = True
            production_selector = "legacy"

        service = MemoryService(retrieval_flags=Flags())
        scope = Scope(workspace_id="ws-dual-shadow-dedupe", user_id="u", agent_id="a")
        shared_graph_candidate = Candidate(
            id="shared-span",
            type="event",
            text="I set up the schema and then implemented transaction CRUD.",
            source="event_ordering_persisted_graph",
            scores={"score": 1.0},
            source_span_ids=["span-shared"],
            metadata={},
        )
        shared_legacy_candidate = Candidate(
            id="legacy-shared-span",
            type="event",
            text="I set up the schema and then implemented transaction CRUD.",
            source="event_timeline_graph",
            scores={"score": 1.0},
            source_span_ids=["span-shared"],
            metadata={},
        )
        try:
            service.add({"role": "user", "content": "First I set up schema. Then I implemented transaction CRUD."}, scope)
            service._event_ordering_graph_selector_candidates = lambda query, scope, limit, include_session=False: [shared_graph_candidate]
            service._event_ordering_legacy_recall_for_shadow = lambda query, scope, plan, limit, include_session: ([shared_legacy_candidate], [shared_legacy_candidate.source])

            result = service.search(
                "What order did I discuss the budget tracker work?",
                scope,
                {"query_type_hint": "event_ordering", "limit": 5},
            )

            shadow = result.coverage["event_ordering_dual_shadow"]
            self.assertEqual(shadow["graph_candidate_count"], 1)
            self.assertEqual(shadow["legacy_candidate_count"], 1)
            self.assertEqual(shadow["candidate_count"], 1)
        finally:
            service.close()

    def test_event_ordering_shadow_replay_keeps_graph_and_legacy_paths_comparable(self) -> None:
        cases = [
            (
                [
                    ("I first set up the schema and local server for the project.", "2026-06-01T10:00:00+00:00"),
                    ("Then I implemented transaction CRUD and validation error handling.", "2026-06-02T10:00:00+00:00"),
                    ("After that I configured Render deployment with Gunicorn workers.", "2026-06-03T10:00:00+00:00"),
                    ("Finally I expanded integration tests for endpoint coverage.", "2026-06-04T10:00:00+00:00"),
                ],
                ["schema", "transaction CRUD", "Render deployment"],
            ),
            (
                [
                    ("For the weather app, I started with friendly 404 and invalid city errors.", "2026-07-01T10:00:00+00:00"),
                    ("Next I added a try-catch wrapper around the OpenWeather API call.", "2026-07-02T10:00:00+00:00"),
                    ("Later I debugged an Unhandled Promise Rejection in fetchWeatherData().", "2026-07-03T10:00:00+00:00"),
                    ("Then I refined the UX copy and promise chaining for the error flow.", "2026-07-04T10:00:00+00:00"),
                ],
                ["friendly 404", "try-catch", "Unhandled Promise Rejection"],
            ),
        ]
        for index, (turns, expected_terms) in enumerate(cases, start=1):
            memory = MemoryService()
            scope = Scope(workspace_id=f"beam-shadow-{index}", user_id="u", agent_id="a", session_id="s")
            for content, timestamp in turns:
                memory.add(content, scope, ts(timestamp))

            pack = memory.answer_context(
                "Can you walk me through the order in which I brought up these implementation topics across our conversations?",
                scope,
                budget={"limit": 8, "mode": "benchmark"},
            )

            shadow_coverage = pack.coverage.get("event_ordering_shadow")
            self.assertIsInstance(shadow_coverage, dict)
            self.assertNotIn("event_ordering_graph", pack.coverage)
            self.assertEqual(shadow_coverage.get("selected_span_source"), "legacy")
            evidence = "\n".join(
                [event.get("description", "") for event in pack.events]
                + [span.get("content", "") for span in pack.source_spans]
            )
            for term in expected_terms:
                self.assertIn(term, evidence)

        memory = MemoryService()
        scope = Scope(workspace_id="beam-shadow-current", user_id="u", agent_id="a", session_id="s")
        memory.add("For Project Atlas, I initially preferred Qdrant for retrieval experiments.", scope, ts("2026-01-01T10:00:00+00:00"))
        memory.add("I switched Project Atlas retrieval from Qdrant to Postgres pgvector for production.", scope, ts("2026-01-08T10:00:00+00:00"))
        current_pack = memory.answer_context(
            "What retrieval backend does Project Atlas currently use?",
            scope,
            budget={"allow_cross_session": True, "limit": 4},
        )
        current_evidence = "\n".join(span["content"] for span in current_pack.source_spans)
        self.assertIn("Postgres pgvector", current_evidence)
        self.assertNotEqual(current_pack.coverage.get("query_type"), "event_ordering")

    def test_event_ordering_short_timeline_covers_broad_project_phases(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I planned core functionality for login, transaction tracking, and analytics.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I designed a narrow database schema detail for the initial project.", scope, ts("2026-06-01T11:00:00+00:00"))
        memory.add("I implemented transaction CRUD and response error handling.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I wrote authentication tests for password hashing.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("I finalized deployment with Render, Gunicorn, and security hardening.", scope, ts("2026-06-04T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you list the order in which I brought up different aspects of developing my app, in order? Mention ONLY and ONLY three items.",
            scope,
            budget={"limit": 8},
        )

        groups = [event.get("milestone_group") for event in pack.events if event.get("milestone_group")]
        self.assertIn("core_functionality", groups)
        self.assertTrue(any(group in groups for group in ["transaction_error_handling", "transaction_crud_implementation"]))
        self.assertTrue(any(group in groups for group in ["security_and_deployment", "deployment_configuration"]))
        self.assertLess(groups.index("core_functionality"), groups.index(next(group for group in groups if group.startswith("transaction"))))

    def test_event_ordering_long_timeline_keeps_early_setup_anchor(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I started by setting up the Flask project, local server, and initial database schema.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I later integrated Flask-Login for core session management and user logins.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I fixed transaction POST validation errors and missing amount handling.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("I configured Render deployment with Gunicorn workers and port settings.", scope, ts("2026-06-04T10:00:00+00:00"))
        memory.add("I expanded integration tests for auth and security coverage.", scope, ts("2026-06-05T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you walk me through the order in which I brought up different aspects of my app development and deployment across our conversations? Mention ONLY and ONLY five items.",
            scope,
            budget={"limit": 12},
        )

        anchors = [
            span
            for span in pack.source_spans
            if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor"
        ]
        labels = [span.get("timeline_label", "") for span in anchors[:5]]
        contents = " ".join(span["content"].lower() for span in anchors[:5])
        self.assertGreaterEqual(len(anchors), 5)
        self.assertTrue("initial project setup" in labels[0].lower() or "schema" in contents[:300] or "local server" in contents[:300])
        self.assertIn("transaction", contents)
        self.assertIn("deployment", contents)
        self.assertTrue("integration tests" in contents or "coverage" in contents)

    def test_exact_signal_preserved_through_rrf_merge(self) -> None:
        raw = Candidate(
            id="s1",
            type="span",
            text="Dashboard API response time is 250ms.",
            source="l0_raw_hybrid",
            scores={"bm25_score": 0.2, "score": 0.2},
            source_span_ids=["s1"],
            metadata={"speaker": "user"},
        )
        exact = Candidate(
            id="s1",
            type="span",
            text="Dashboard API response time is 250ms.",
            source="exact_filter",
            scores={"bm25_score": 1.0, "exact_signal": 1.0, "score": 1.0},
            source_span_ids=["s1"],
            metadata={"exact_signal": 1.0},
        )

        fused = reciprocal_rank_fusion([[raw], [exact]])

        self.assertEqual(len(fused), 1)
        self.assertIn("exact_filter", fused[0].source)
        self.assertEqual(fused[0].scores["exact_signal"], 1.0)
        self.assertEqual(fused[0].metadata["exact_signal"], 1.0)

    def test_current_value_exact_signal_surfaces_latest_numeric_evidence(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("The dashboard API response time was initially 800ms.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("The dashboard API response time was reduced to 300ms after query optimization.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("The dashboard API response time has recently improved to 250ms after caching tweaks.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("I changed the dashboard color palette and sidebar layout.", scope, ts("2026-06-04T10:00:00+00:00"))

        pack = memory.answer_context("What is the average response time of the dashboard API?", scope, budget={"limit": 4})

        contents = [span["content"] for span in pack.source_spans]
        self.assertTrue(any("250ms" in content for content in contents[:2]))
        self.assertTrue(any("800ms" in content for content in contents))

    def test_temporal_lookup_pack_labels_date_roles(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "Phase 3 transaction management features are completed by January 15, 2024.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        memory.add(
            "The final deployment deadline is March 15, 2024.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many weeks do I have between finishing the transaction management features and the final deployment deadline?",
            scope,
            budget={"limit": 4},
        )

        self.assertEqual(pack.coverage["query_type"], "temporal_lookup")
        mentions = [mention for span in pack.source_spans for mention in span.get("temporal_mentions", [])]
        roles = {mention["role"] for mention in mentions}
        self.assertIn("feature_finish_date", roles)
        self.assertIn("deployment_deadline", roles)
        self.assertIn("2024-01-15", {mention["normalized_date"] for mention in mentions})
        self.assertIn("2024-03-15", {mention["normalized_date"] for mention in mentions})
        self.assertIn("feature_finish_date", pack.coverage["temporal_target_roles"])
        self.assertIn("deployment_deadline", pack.coverage["temporal_target_roles"])
        self.assertNotIn("temporal_role_candidates", pack.coverage)

    def test_temporal_lookup_uses_topic_scope_before_date_roles(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "The screenplay launch preparation has a final deployment deadline of March 20, 2024.",
            scope,
            ts("2026-06-01T09:00:00+00:00"),
            {"source_uri": "beam:test:20:batch1:msg1"},
        )
        memory.add(
            "The transaction management features are completed by January 15, 2024.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg1"},
        )
        memory.add(
            "The final deployment deadline for the budget tracker is March 15, 2024.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg2"},
        )

        pack = memory.answer_context(
            "How many weeks do I have between finishing the transaction management features and the final deployment deadline?",
            scope,
            budget={"limit": 6},
        )

        content = " ".join(span["content"] for span in pack.source_spans)
        self.assertIn("transaction management features", content)
        self.assertIn("March 15, 2024", content)
        self.assertNotIn("screenplay launch preparation", content)

    def test_temporal_lookup_infers_year_for_month_day_ranges(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "Phase 3 runs Dec 16, 2023 - Jan 15, with transaction management features complete at the end.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "When did the transaction management features finish?",
            scope,
            budget={"limit": 4},
        )

        mentions = [mention for span in pack.source_spans for mention in span.get("temporal_mentions", [])]
        self.assertIn("2024-01-15", {mention["normalized_date"] for mention in mentions})
        self.assertNotIn("2026-01-15", {mention["normalized_date"] for mention in mentions})
        self.assertIn("feature_finish_date", {mention["role"] for mention in mentions})

    def test_temporal_lookup_summary_preserves_late_date_contexts(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "Project overview. " * 40
            + "Milestones: Nov 16 - Dec 15, 2023 for authentication. "
            + "Dec 16, 2023 - Jan 15, 2024: Develop transaction management features. "
            + "Feb 16 - Mar 15, 2024: Final adjustments, testing, and deployment.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many weeks do I have between finishing the transaction management features and the final deployment deadline?",
            scope,
            budget={"limit": 4},
        )

        content = " ".join(span["content"] for span in pack.source_spans)
        mentions = [mention for span in pack.source_spans for mention in span.get("temporal_mentions", [])]
        by_date = {mention["normalized_date"]: mention for mention in mentions}
        self.assertIn("Jan 15, 2024", content)
        self.assertEqual(by_date["2024-01-15"]["role"], "feature_finish_date")
        self.assertEqual(by_date["2024-03-15"]["role"], "deployment_deadline")

    def test_temporal_lookup_distinguishes_mvp_and_deployment_deadlines(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "The MVP scope deadline is April 15, 2024.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        memory.add(
            "The final deployment deadline is March 15, 2024.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How much time is there until the final deployment deadline?",
            scope,
            budget={"limit": 4},
        )

        by_date = {
            mention["normalized_date"]: mention["role"]
            for span in pack.source_spans
            for mention in span.get("temporal_mentions", [])
        }
        self.assertEqual(by_date["2024-04-15"], "mvp_deadline")
        self.assertEqual(by_date["2024-03-15"], "deployment_deadline")

    def test_temporal_lookup_uses_specific_feature_anchors(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "Sprint 1 ends March 29, 2024. Sprint 2 targets analytics features by April 19, 2024.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        memory.add(
            "Dec 16, 2023 - Jan 15, 2024: Develop transaction management features.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many days were there between the end of my first sprint and the deadline for completing the analytics features in sprint 2?",
            scope,
            budget={"limit": 4},
        )

        self.assertIn("sprint_end_date", pack.coverage["temporal_target_roles"])
        mentions = [mention for span in pack.source_spans for mention in span.get("temporal_mentions", [])]
        by_date = {mention["normalized_date"]: mention["role"] for mention in mentions}
        self.assertEqual(by_date["2024-03-29"], "sprint_end_date")
        self.assertEqual(by_date["2024-04-19"], "feature_finish_date")
        self.assertNotEqual(by_date["2024-01-15"], "feature_finish_date")

    def test_temporal_lookup_labels_decision_and_reschedule_dates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "I decided to reject the raise on March 12, 2026 after reviewing the offer.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        memory.add(
            "I rescheduled my final meeting to March 30, 2026 so I could have more time.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many days passed between when I decided to reject the raise and when I rescheduled my final meeting to give myself more time?",
            scope,
            budget={"limit": 4},
        )

        self.assertIn("decision_date", pack.coverage["temporal_target_roles"])
        self.assertIn("reschedule_date", pack.coverage["temporal_target_roles"])
        candidates = pack.coverage["temporal_candidates"]
        roles_by_date = {candidate["normalized_date"]: candidate["role"] for candidate in candidates}
        self.assertEqual(roles_by_date["2026-03-12"], "decision_date")
        self.assertEqual(roles_by_date["2026-03-30"], "reschedule_date")
        self.assertTrue(candidates[0]["target_role_match"])

    def test_temporal_candidates_prioritize_user_target_roles_over_assistant_examples(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "Example plan: March 1, 2024 gather options; March 11, 2024 interview advisors.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        memory.add(
            "I decided to reject the raise on March 12, 2026.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
        )
        memory.add(
            "I rescheduled my final meeting to March 30, 2026.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many days passed between when I decided to reject the raise and when I rescheduled my final meeting?",
            scope,
            budget={"limit": 6},
        )

        candidates = pack.coverage["temporal_candidates"]
        top_dates = [candidate["normalized_date"] for candidate in candidates[:2]]
        self.assertEqual(top_dates, ["2026-03-12", "2026-03-30"])

    def test_temporal_pack_exposes_explicit_date_range_pairs(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "I completed the clarity editing challenge from May 10 to May 25, 2026.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many days were there between when I completed the clarity editing challenge and another milestone?",
            scope,
            budget={"limit": 4},
        )

        pairs = pack.coverage["temporal_range_pairs"]
        self.assertEqual(pairs[0]["start_date"], "2026-05-10")
        self.assertEqual(pairs[0]["end_date"], "2026-05-25")
        self.assertEqual(pairs[0]["start_role"], "start_date")
        self.assertIn(pairs[0]["end_role"], {"completion_date", "feature_finish_date"})

    def test_temporal_coverage_recovers_decision_object_span(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "I decided on a film festival path on October 15, 2026.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        memory.add(
            "I'm torn about rejecting that $10,000 raise on March 12, 2026.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
        )
        memory.add(
            "I rescheduled my final meeting to March 30, 2026 to give myself more time.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many days passed between when I decided to reject the raise and when I rescheduled my final meeting to give myself more time?",
            scope,
            budget={"limit": 4},
        )

        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("rejecting that $10,000 raise on march 12", content)
        self.assertIn("rescheduled my final meeting to march 30", content)
        roles_by_date = {candidate["normalized_date"]: candidate["role"] for candidate in pack.coverage["temporal_candidates"]}
        self.assertEqual(roles_by_date["2026-03-12"], "decision_date")
        self.assertEqual(roles_by_date["2026-03-30"], "reschedule_date")

    def test_temporal_coverage_recovers_duration_completion_span(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            'I downloaded "The Poppy War" trilogy on December 7, 2026.',
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        memory.add(
            'I finished "The Poppy War" trilogy with 1,150 pages in 12 days.',
            scope,
            ts("2026-06-02T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "How many days did it take me to finish reading the trilogy after I downloaded it?",
            scope,
            budget={"limit": 4},
        )

        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("downloaded", content)
        self.assertIn("12 days", content)

    def test_event_ordering_topic_scope_prevents_generic_milestone_bleed(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "I'm implementing transaction CRUD response handling for my budget tracker.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg1"},
        )
        memory.add(
            "I'm configuring Render deployment with Gunicorn workers for the budget tracker.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:1:batch1:msg2"},
        )
        memory.add(
            "I started managing stress by setting no-work Sundays and reducing burnout.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:16:batch1:msg1"},
        )
        memory.add(
            "I handled financial concerns by tracking rent, groceries, and emergency savings.",
            scope,
            ts("2026-06-04T10:00:00+00:00"),
            {"source_uri": "beam:test:16:batch1:msg2"},
        )

        pack = memory.answer_context(
            "Can you walk me through the order in which I brought up different ways I’ve been managing stress and financial concerns throughout our chats, in order?",
            scope,
            budget={"limit": 8},
        )

        content = " ".join(span["content"] for span in pack.source_spans).lower()
        self.assertIn("managing stress", content)
        self.assertIn("financial concerns", content)
        self.assertNotIn("transaction crud", content)
        self.assertNotIn("gunicorn", content)

    def test_summarization_expands_same_topic_group_timeline(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "My portfolio website started with HTML5 sections for About, Skills, Projects, and Contact.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:3:batch1:msg1"},
        )
        memory.add(
            "I added Bootstrap 5.3.0 cards and modal popups to the portfolio website project gallery.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:3:batch1:msg2"},
        )
        memory.add(
            "I debugged CSS box model and layout issues with Chrome DevTools for the portfolio website.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:3:batch1:msg3"},
        )
        memory.add(
            "I updated my Behance digital portfolio for a salary negotiation.",
            scope,
            ts("2026-06-04T10:00:00+00:00"),
            {"source_uri": "beam:test:30:batch1:msg1"},
        )

        pack = memory.answer_context(
            "Can you give me a comprehensive summary of how my portfolio website project has developed, including the key features and challenges I've worked through so far?",
            scope,
            budget={"limit": 6},
        )

        content = " ".join(span["content"] for span in pack.source_spans)
        self.assertIn("HTML5 sections", content)
        self.assertIn("Bootstrap 5.3.0", content)
        self.assertIn("Chrome DevTools", content)
        self.assertNotIn("salary negotiation", content)
        self.assertIn("summary_clusters", pack.coverage)

    def test_contradiction_pack_groups_surface_claim_polarity(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I have never used Bootstrap components in my portfolio website.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:3:batch1:msg1"})
        memory.add("I used Bootstrap cards and modal components in the portfolio website gallery.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:3:batch1:msg2"})

        pack = memory.answer_context("Have I used Bootstrap components in my project before?", scope, budget={"limit": 6})

        self.assertEqual(pack.coverage["query_type"], "contradiction_resolution")
        polarities = {span.get("claim_polarity") for span in pack.source_spans}
        self.assertIn("negative", polarities)
        self.assertIn("positive", polarities)
        self.assertTrue(pack.conflicts)
        self.assertGreaterEqual(pack.coverage["claim_polarity_counts"]["positive"], 1)
        self.assertGreaterEqual(pack.coverage["claim_polarity_counts"]["negative"], 1)
        self.assertTrue(any(str(item["source"]).startswith("contradiction_claim_") for item in pack.debug_trace))

    def test_contradiction_pack_preserves_both_polarities_through_noise(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        for index in range(20):
            memory.add(
                f"Budget planning note {index}: track income, expenses, and monthly savings carefully.",
                scope,
                ts(f"2026-06-01T10:{index:02d}:00+00:00"),
                {"source_uri": f"beam:test:16:noise:{index}"},
            )
        memory.add(
            "I have never used Excel for tracking expenses before.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:16:batch1:msg1"},
        )
        memory.add(
            "I've been using Excel to track my daily expenses since March 1.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:16:batch1:msg2"},
        )

        pack = memory.answer_context(
            "Have I been using Excel to track my daily expenses?",
            scope,
            budget={"limit": 6, "query_type_hint": "contradiction_resolution"},
        )

        polarities = {span.get("claim_polarity") for span in pack.source_spans}
        self.assertIn("negative", polarities)
        self.assertIn("positive", polarities)
        self.assertGreaterEqual(pack.coverage["claim_polarity_counts"]["positive"], 1)
        self.assertGreaterEqual(pack.coverage["claim_polarity_counts"]["negative"], 1)

    def test_knowledge_update_pack_marks_history_and_value_mentions(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("The dashboard API response time was initially 800ms.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:1:batch1:msg1"})
        memory.add("The dashboard API response time improved to 300ms after optimization.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:1:batch1:msg2"})
        memory.add("The dashboard API response time is now 250ms after caching tweaks.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:1:batch1:msg3"})
        memory.add("The mobile animation duration is now 300ms.", scope, ts("2026-06-04T10:00:00+00:00"), {"source_uri": "beam:test:other:batch1:msg1"})

        pack = memory.answer_context("What is the average response time of the dashboard API?", scope, budget={"limit": 6})

        self.assertEqual(pack.coverage["query_type"], "knowledge_update")
        self.assertEqual(sorted(span["history_index"] for span in pack.source_spans), list(range(1, len(pack.source_spans) + 1)))
        self.assertTrue(any(span.get("recency_rank") == 1 and "250ms" in span["content"] for span in pack.source_spans))
        self.assertTrue(any(span.get("value_mentions") for span in pack.source_spans))
        self.assertTrue(any(row.get("subject_key") and "response" in row["subject_key"] for row in pack.coverage["value_history"]))

    def test_current_chinese_target_date_recall_includes_latest_update(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w-zh", user_id="u", agent_id="a", session_id="s2")
        memory.add(
            "现在星桥的当前发布目标是 6 月 30 日完成 alpha，不是之前说的 6 月 20 日。",
            scope,
            ts("2026-06-09T10:00:00+00:00"),
            {"source_uri": "sim:zh:date", "speaker": "user"},
        )
        memory.add(
            "预算更新：星桥每月预算现在是 600 元，不再是早期估算的 300 元。",
            scope,
            ts("2026-06-08T10:00:00+00:00"),
            {"source_uri": "sim:zh:budget", "speaker": "user"},
        )

        pack = memory.answer_context("星桥当前发布目标日期是什么？", scope, budget={"limit": 6})
        content = "\n".join(span["content"] for span in pack.source_spans)

        self.assertIn("6 月 30 日", content)
        self.assertTrue(any(row.get("value") == "6 月 30 日" for row in pack.coverage.get("value_history", [])))

    def test_multi_session_pack_marks_history_order_for_aggregation(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I want to add a category column to the transactions table.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:1:batch1:msg1"})
        memory.add("I also want to add a notes column to the transactions table.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:1:batch2:msg1"})

        pack = memory.answer_context("How many new columns did I want to add to the transactions table across my requests?", scope, budget={"limit": 6})

        self.assertEqual(pack.coverage["query_type"], "multi_session_reasoning")
        content = " ".join(span["content"] for span in pack.source_spans)
        self.assertIn("category column", content)
        self.assertIn("notes column", content)
        self.assertEqual([span["history_index"] for span in pack.source_spans], list(range(1, len(pack.source_spans) + 1)))

    def test_multi_session_aggregation_preserves_estate_asset_candidates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "My asset inventory includes $25,000 in savings, $15,000 in film equipment, and a 2018 Toyota RAV4 valued at $18,000.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:19:batch1:msg8"},
        )
        memory.add(
            "I started listing my assets on March 1, and my $350,000 home on 45 Coral Bay Rd is a big part of that.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:19:batch1:msg9"},
        )
        memory.add(
            "I installed a fireproof safe at 45 Coral Bay Rd for storing my original will and important documents.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:19:batch3:msg8"},
        )
        memory.add(
            "A generic estate checklist can mention bank accounts, retirement accounts, and vacation homes.",
            scope,
            ts("2026-06-04T10:00:00+00:00"),
            {"source_uri": "beam:test:19:batch4:msg1"},
        )

        pack = memory.answer_context(
            "How many specific assets or items have I mentioned across my conversations that are part of my estate planning?",
            scope,
            budget={"limit": 6},
        )

        self.assertEqual(pack.coverage["query_type"], "multi_session_reasoning")
        self.assertTrue(any("aggregation_coverage" in item["source"] for item in pack.debug_trace))
        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("2018 toyota rav4", content)
        self.assertIn("film equipment", content)
        self.assertIn("45 coral bay", content)
        self.assertIn("fireproof safe", content)
        self.assertIn("original will", content)

    def test_multi_session_count_query_with_did_i_mention_is_not_yes_no_contradiction(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "I mentioned arranging 3 colored balls in a row with 3! equals 6 ways.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:5:batch1:msg1"},
        )
        memory.add(
            "I also mentioned choosing 2 balls out of 3 with 3C2 equals 3 ways.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:5:batch2:msg1"},
        )
        memory.add(
            "Then I asked about choosing 2 cards from a 52-card deck.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:5:batch3:msg1"},
        )

        pack = memory.answer_context(
            "How many total ways did I mention for arranging or choosing balls and cards across my questions?",
            scope,
            budget={"limit": 8},
        )

        self.assertEqual(pack.coverage["query_type"], "multi_session_reasoning")
        self.assertTrue(any("aggregation_coverage" in item["source"] for item in pack.debug_trace))
        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("arranging 3 colored balls", content)
        self.assertIn("choosing 2 balls", content)
        self.assertIn("choosing 2 cards", content)

    def test_multi_session_combinatorics_aggregation_ignores_other_topic_groups(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "I asked about arranging 3 different colored balls in a row, where 3! equals 6 ways.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:5:batch1:msg27"},
        )
        memory.add(
            "I asked if choosing 2 balls from 3 means 3C2 equals 3 ways.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:5:batch1:msg28"},
        )
        memory.add(
            "I asked how many ways there are to choose cards from a 52-card deck.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:5:batch2:msg79"},
        )
        memory.add(
            "In my sneaker notes, I compared many total ways to choose black or white sneakers with cards and discounts.",
            scope,
            ts("2026-06-04T10:00:00+00:00"),
            {"source_uri": "beam:test:15:batch1:msg37"},
        )

        pack = memory.answer_context(
            "How many total ways did I mention for arranging or choosing balls and cards across my questions?",
            scope,
            budget={"limit": 8},
        )

        self.assertEqual(pack.coverage["query_type"], "multi_session_reasoning")
        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("3c2 equals 3 ways", content)
        self.assertIn("52-card deck", content)
        self.assertNotIn("sneaker", content)

    def test_multi_session_aggregation_preserves_column_candidates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("My transactions table currently has id, user_id, type, amount, and date.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:1:batch1:msg1"})
        memory.add("I want to add a category column to the transactions table for reporting.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:1:batch2:msg1"})
        memory.add("In another request, I also wanted a notes column on transactions for free-form details.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:1:batch3:msg1"})
        memory.add("I discussed unrelated auth login copy.", scope, ts("2026-06-04T10:00:00+00:00"), {"source_uri": "beam:test:1:batch4:msg1"})

        pack = memory.answer_context(
            "How many new columns did I want to add to the transactions table across my requests?",
            scope,
            budget={"limit": 6},
        )

        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("category column", content)
        self.assertIn("notes column", content)
        self.assertTrue(any("aggregation_coverage" in item["source"] for item in pack.debug_trace))

    def test_multi_session_aggregation_preserves_many_distinct_candidates(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        names = ["drama", "comedy", "thriller", "animation", "documentary", "sci-fi", "fantasy", "mystery"]
        for index, name in enumerate(names, start=1):
            memory.add(
                f"In movie marathon note {index}, I planned a {name} movie.",
                scope,
                ts(f"2026-06-{index:02d}T10:00:00+00:00"),
                {"source_uri": f"beam:test:movies:batch{index}:msg1"},
            )

        pack = memory.answer_context(
            "How many unique movies have I planned to watch across all my family movie marathons?",
            scope,
            budget={"limit": 12},
        )

        content = " ".join(span["content"].lower() for span in pack.source_spans)
        for name in names:
            self.assertIn(name, content)

    def test_multi_session_generic_aggregation_keys_are_added_to_user_spans(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "I focused on adapting my resume to international standards.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:generic:batch1:msg1"},
        )
        memory.add(
            "I also wanted to improve my portfolio project selection.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:generic:batch2:msg1"},
        )
        memory.add(
            "You could also improve general interview practice.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:generic:batch3:msg1"},
        )

        pack = memory.answer_context(
            "How many different planning areas did I mention across my sessions?",
            scope,
            budget={"limit": 6},
        )

        keyed_spans = [span for span in pack.source_spans if span.get("aggregation_keys")]
        keys = {key for span in keyed_spans for key in span["aggregation_keys"]}
        self.assertIn("area:resume_international_standards", keys)
        self.assertIn("area:portfolio_project_selection", keys)
        self.assertFalse(any("interview_practice" in key for key in keys))

    def test_multi_session_aggregation_prefers_user_spans_for_shared_keys(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "I focused on adapting my resume to international standards.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:generic:batch1:msg1"},
        )
        memory.add(
            "I also wanted to improve my portfolio project selection.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:generic:batch2:msg1"},
        )
        memory.add(
            "You could also improve general interview practice.",
            scope,
            ts("2026-06-03T10:00:00+00:00"),
            {"source_uri": "beam:test:generic:batch3:msg1"},
        )

        pack = memory.answer_context(
            "How many different planning areas did I mention across my sessions?",
            scope,
            budget={"limit": 6},
        )

        keyed = [span for span in pack.source_spans if span.get("aggregation_keys")]
        self.assertTrue(keyed)
        self.assertTrue(all(span.get("speaker") == "user" for span in keyed))
        self.assertNotIn("interview practice", " ".join(span["content"].lower() for span in keyed))

    def test_multi_session_generic_user_spans_are_preserved_by_aggregation_signal(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I focused on adapting my resume to international standards.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:generic:batch1:msg1"})
        memory.add("I also wanted to improve my portfolio project selection.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:generic:batch2:msg1"})
        memory.add("You could also improve general interview practice.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:generic:batch3:msg1"})

        pack = memory.answer_context(
            "How many different planning areas did I mention across my sessions?",
            scope,
            budget={"limit": 6},
        )

        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("resume to international standards", content)
        self.assertIn("portfolio project selection", content)
        keyed_content = " ".join(span["content"].lower() for span in pack.source_spans if span.get("aggregation_keys"))
        self.assertNotIn("interview practice", keyed_content)

    def test_multi_session_expansion_spans_receive_generic_aggregation_keys(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        for index, text in enumerate(
            [
                "I want city autocomplete for my app.",
                "I need error handling for invalid city names.",
                "I am trying to support responsive design on mobile.",
                "I implemented saved search filters for returning users.",
            ],
            start=1,
        ):
            memory.add(
                text,
                scope,
                ts(f"2026-06-{index:02d}T10:00:00+00:00"),
                {"source_uri": f"beam:test:features:batch{index}:msg1"},
            )

        pack = memory.answer_context(
            "How many different features or concerns did I mention wanting to handle across my app conversations?",
            scope,
            budget={"limit": 3, "token_budget": 4000},
        )

        keys = {key for span in pack.source_spans for key in span.get("aggregation_keys", [])}
        self.assertGreaterEqual(len(keys), 2)
        self.assertIn("feature:city_autocomplete", keys)
        self.assertIn("feature:error_handling", keys)

    def test_multi_session_cross_factor_effect_pack_preserves_all_factors(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("Our grocery budget is increasing to $500 per month starting September 1.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:budget:batch1:msg1"})
        memory.add("I am taking on a freelance contract that adds $900 per month for three months.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:budget:batch2:msg1"})
        memory.add("I need to support Ashlee's medical bills, which are $350 per month.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:budget:batch3:msg1"})
        memory.add("My savings goal is still $600 per month for the emergency fund.", scope, ts("2026-06-04T10:00:00+00:00"), {"source_uri": "beam:test:budget:batch4:msg1"})
        memory.add("A recipe note mentioned groceries but not my budget plan.", scope, ts("2026-06-05T10:00:00+00:00"), {"source_uri": "beam:test:cooking:batch1:msg1"})

        pack = memory.answer_context(
            "How will increasing our grocery budget while taking on the freelance contract affect my ability to support Ashlee's medical bills and still meet my savings goals?",
            scope,
            budget={"limit": 4},
        )

        self.assertEqual(pack.coverage["query_type"], "multi_session_reasoning")
        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("grocery budget", content)
        self.assertIn("freelance contract", content)
        self.assertIn("medical bills", content)
        self.assertIn("savings goal", content)

    def test_multi_session_stress_break_aggregation_does_not_crash(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            "I took a one-hour yoga break because I was stressed and needed to reset.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:breaks:batch1:msg1"},
        )
        memory.add(
            "I took two full days off to prevent burnout before continuing the project.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:breaks:batch2:msg1"},
        )

        pack = memory.answer_context(
            "How many total days did I take off or breaks to manage stress and prevent burnout across my sessions?",
            scope,
            budget={"limit": 6},
        )

        keys = {key for span in pack.source_spans for key in span.get("aggregation_keys", [])}
        self.assertIn("break:one_hour_stress_day", keys)
        self.assertIn("break:full_days_off", keys)

    def test_multi_session_pack_extracts_generic_size_values(self) -> None:
        from fusion_memory.eval.model_adapters import _pack_for_model

        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I usually wear size 11 running shoes.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:sizes:batch1:msg1"})
        memory.add("For narrow sneakers I sometimes need size 11.5.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:sizes:batch2:msg1"})

        pack = memory.answer_context(
            "How many different shoe sizes have I mentioned across my messages?",
            scope,
            budget={"limit": 6},
        )

        keys = {key for span in pack.source_spans for key in span.get("aggregation_keys", [])}
        self.assertIn("value:size_11", keys)
        self.assertIn("value:size_11_5", keys)
        labels = {item["label"] for item in _pack_for_model(pack).get("aggregation_items", [])}
        self.assertIn("size 11.5", labels)

    def test_multi_session_pack_exposes_quoted_titles_as_aggregation_items(self) -> None:
        from fusion_memory.eval.model_adapters import _pack_for_model

        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add('I planned "Soul" and "Paddington 2" for the first movie night.', scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:movies:batch1:msg1"})
        memory.add('I added "Coco" for the second movie night.', scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:movies:batch2:msg1"})
        memory.add("I want to make the movie night special with themed snacks.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:movies:batch3:msg1"})

        pack = memory.answer_context(
            "How many unique movies have I planned across all movie nights?",
            scope,
            budget={"limit": 6},
        )
        model_pack = _pack_for_model(pack)
        item_keys = {item["key"] for item in model_pack.get("aggregation_items", [])}

        self.assertIn("title:soul", item_keys)
        self.assertIn("title:paddington_2", item_keys)
        self.assertIn("title:coco", item_keys)
        self.assertTrue(all(key.startswith("title:") for key in item_keys))
        self.assertFalse(any("snack" in key or "special" in key for key in item_keys))

    def test_multi_session_preserves_adjacent_assistant_recommendation_group(self) -> None:
        from fusion_memory.eval.model_adapters import _pack_for_model

        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add(
            {
                "role": "user",
                "content": "Can you suggest some must-read fiction series that fit my winter budget?",
                "turn_id": "beam:test:series:batch1:msg1",
                "timestamp": "2026-06-01T10:00:00+00:00",
            },
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"source_uri": "beam:test:series:batch1:msg1"},
        )
        memory.add(
            {
                "role": "assistant",
                "content": 'Here are a few suggestions:\n\n### "The Kingkiller Chronicle"\nDetails...\n\n### "The Mistborn Trilogy"\nDetails...\n\n### "The Lies of Locke Lamora"\nDetails...',
                "turn_id": "beam:test:series:batch1:msg2",
                "timestamp": "2026-06-01T10:01:00+00:00",
            },
            scope,
            ts("2026-06-01T10:01:00+00:00"),
            {"source_uri": "beam:test:series:batch1:msg2"},
        )
        memory.add(
            {
                "role": "user",
                "content": 'I later mentioned that "The Vorkosigan Saga" sounded interesting, and I want to explore it as a science fiction series.',
                "turn_id": "beam:test:series:batch2:msg1",
                "timestamp": "2026-06-02T10:00:00+00:00",
            },
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:series:batch2:msg1"},
        )

        pack = memory.answer_context(
            "How many different book series or genres have I mentioned wanting to explore across my conversations?",
            scope,
            budget={"mode": "benchmark", "limit": 8},
        )
        model_pack = _pack_for_model(pack)
        content = "\n".join(span["content"] for span in pack.source_spans)
        included = {item["key"]: item for item in model_pack.get("aggregation_items", []) if item["included"]}

        self.assertIn("The Kingkiller Chronicle", content)
        self.assertTrue(any(key.startswith("group_count:series:") and item["value"] == 3 for key, item in included.items()))
        self.assertIn("title:the_vorkosigan_saga", included)

    def test_multi_session_prefers_specific_adjacent_recommendation_group(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        turns = [
            ("user", "Can you suggest a fiction series for cozy evenings?", "msg1"),
            ("assistant", 'Here are options:\n\n### "Series A"\n...\n\n### "Series B"\n...\n\n### "Series C"\n...', "msg2"),
            ("user", "Can you suggest a few fantasy books for a club chat?", "msg3"),
            ("assistant", 'Here are options:\n\n### "Series D"\n...\n\n### "Series E"\n...\n\n### "Series F"\n...', "msg4"),
            (
                "user",
                "I have a $120 budget and want to buy print editions from North Street Books; can you suggest three must-read fiction series that fit?",
                "msg5",
            ),
            (
                "assistant",
                'With that budget and print-edition constraint, I would suggest:\n\n### "Specific One"\n...\n\n### "Specific Two"\n...\n\n### "Specific Three"\n...',
                "msg6",
            ),
            ("user", 'I later mentioned "Specific Four" as a science fiction series to explore.', "msg7"),
        ]
        for index, (role, content, turn_id) in enumerate(turns):
            memory.add(
                {
                    "role": role,
                    "content": content,
                    "turn_id": f"beam:test:specific:{turn_id}",
                    "timestamp": f"2026-06-01T10:{index:02d}:00+00:00",
                },
                scope,
                ts(f"2026-06-01T10:{index:02d}:00+00:00"),
                {"source_uri": f"beam:test:specific:{turn_id}"},
            )

        pack = memory.answer_context(
            "How many different book series or genres have I mentioned wanting to explore across my conversations?",
            scope,
            budget={"mode": "benchmark", "limit": 6},
        )
        from fusion_memory.eval.model_adapters import _pack_for_model

        content = "\n".join(span["content"] for span in pack.source_spans)
        model_pack = _pack_for_model(pack)
        included_group_counts = [
            item for item in model_pack.get("aggregation_items", [])
            if item.get("included") and str(item.get("key", "")).startswith("group_count:")
        ]
        self.assertIn("Specific One", content)
        self.assertIn("Specific Four", content)
        self.assertEqual(1, len(included_group_counts))
        self.assertIn("Specific One", included_group_counts[0]["context"])

    def test_scent_trail_recovers_followup_version_details(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("The project uses Flask for the backend.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:trail:batch1:msg1"})
        memory.add("I also want to keep Flask-Login and Flask-WTF in the stack.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:trail:batch1:msg2"})
        memory.add("The project also uses Flask==2.3.1, Flask-Login==0.6.2, and Flask-WTF==1.0.1.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:trail:batch1:msg3"})

        result = memory.search("Which libraries are used in this project? Include version details.", scope, options={"limit": 5})
        pack = memory.answer_context("Which libraries are used in this project? Include version details.", scope, budget={"limit": 2})

        content = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("flask", content)
        self.assertTrue(any("raw_scent_trail" in candidate.source and "0.6.2" in candidate.text for candidate in result.candidates))

    def test_quality_fallback_recovers_high_signal_span_from_weak_selection(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("Sounds good.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:fallback:batch1:msg1"})
        memory.add(
            "The frontend launch checklist included fixing the navbar error, adding retry backoff, and testing image loading on June 12.",
            scope,
            ts("2026-06-02T10:00:00+00:00"),
            {"source_uri": "beam:test:fallback:batch1:msg2"},
        )
        memory.add("Thanks, that helps.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:fallback:batch1:msg3"})
        plan = memory.planner.plan("Summarize the frontend launch checklist issues and fixes.")
        weak_selected = [
            Candidate(
                id="weak",
                type="span",
                text="Sounds good.",
                source="test",
                scores={"score": 0.01, "utility_score": 0.01},
                source_span_ids=["weak"],
            )
        ]

        recovered = memory._apply_quality_fallback(
            "Summarize the frontend launch checklist issues and fixes.",
            plan,
            scope,
            [],
            weak_selected,
            3,
        )

        contents = " ".join(candidate.text.lower() for candidate in recovered)
        self.assertIn("navbar error", contents)
        self.assertTrue(any(candidate.source == "quality_fallback" for candidate in recovered))

    def test_instruction_pack_exposes_format_requirements(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("The project uses Flask==2.3.1 and Flask-Login==0.6.2.", scope, ts("2026-06-01T10:00:00+00:00"))

        pack = memory.answer_context("Which libraries are used in this project? Include version details.", scope, budget={"limit": 4})

        self.assertIn("include_exact_versions_if_supported", pack.coverage["format_requirements"])

    def test_rule_extractor_preserves_fact_polarity_values_and_event_topic_terms(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        result = memory.add(
            "Remember that I have never used Bootstrap components, but I tested the portfolio website on March 15, 2024 with Flask==2.3.1.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )

        facts = [memory.store.get_fact(fact_id) for fact_id in result.accepted_fact_ids]
        facts = [fact for fact in facts if fact]
        self.assertTrue(any(fact.polarity == "negative" for fact in facts))
        self.assertTrue(any(fact.metadata.get("value_mentions") for fact in facts))
        events = memory.store.list_events(scope)
        self.assertTrue(any("portfolio" in event.participants or "bootstrap" in event.participants for event in events))

    def test_rule_extractor_emits_cross_domain_event_facets(self) -> None:
        facets = extract_generic_event_facets(
            "I'm worried about emergency savings, decided to create a monthly budget, "
            "and I need to track every expense over $20. I compared checking vs. savings and listed three options."
        )

        facet_types = [facet for facet, _label, _snippet in facets]
        self.assertIn("concern", facet_types)
        self.assertIn("decision", facet_types)
        self.assertIn("constraint", facet_types)
        self.assertIn("request_for_comparison", facet_types)
        self.assertIn("count_list_mention", facet_types)

        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "I'm worried about emergency savings, decided to create a monthly budget, and I need to track every expense over $20.",
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )

        event_types = {event.event_type for event in memory.store.list_events(scope)}
        self.assertIn("concern", event_types)
        self.assertIn("decision", event_types)
        self.assertIn("constraint", event_types)

    def test_event_ordering_uses_generic_event_facets_for_cross_domain_timeline(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I'm worried about emergency savings after the phishing attempt.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I decided to create a monthly budget for rent, groceries, and transport.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I need to track every expense over $20 in Excel.", scope, ts("2026-06-03T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you list the order in which I brought up different money-management concerns and decisions?",
            scope,
            budget={"limit": 8},
        )

        anchors = [
            span
            for span in pack.source_spans
            if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor"
        ]
        self.assertGreaterEqual(len(anchors), 3)
        contents = " ".join(span["content"].lower() for span in anchors)
        self.assertIn("emergency savings", contents)
        self.assertIn("monthly budget", contents)
        self.assertIn("expense over $20", contents)

    def test_event_ordering_prefers_single_anchor_for_non_list_turns(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            "I'm planning to wear my Nike Dunk Low to the festival, but with a moderate water resistance rating, should I take any extra precautions or consider Bradley's suggestion to carry a sneaker protector spray, given the weather forecast?",
            scope,
            ts("2026-06-13T10:47:34.798910+08:00"),
        )
        pack = memory.answer_context(
            "Can you list the order in which I brought up different sneaker shopping experiences and related details throughout our conversations in order? Mention ONLY and ONLY four items.",
            scope,
            budget={"limit": 8},
        )

        anchors = [
            span
            for span in pack.source_spans
            if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor"
        ]
        self.assertEqual(len(anchors), 1)
        self.assertIn("Nike Dunk Low", anchors[0]["content"])

    def test_event_ordering_event_graph_candidates_stay_topic_scoped(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I'm trying to increase my social media followers by 15% by showcasing my sneaker style at the festival.", scope, ts("2026-06-13T10:25:59.059829+08:00"), {"source_uri": "beam:test:15:batch1:msg1"})
        memory.add("I'm trying to set up the automatic transfers. It makes sense to automate it to keep me on track with my savings goals.", scope, ts("2026-06-13T10:52:47.975134+08:00"), {"source_uri": "beam:test:16:batch1:msg1"})

        pack = memory.answer_context(
            "Can you list the order in which I brought up different sneaker shopping experiences and related details throughout our conversations in order? Mention ONLY and ONLY four items.",
            scope,
            budget={"limit": 8},
        )

        contents = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("sneaker", contents)
        self.assertNotIn("automatic transfers", contents)
        self.assertNotIn("savings goals", contents)

    def test_event_ordering_user_chronology_beats_event_phase_priority(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I first asked about choosing running shoes for daily wear.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Then I compared Nike React versus Adidas Ultraboost for comfort.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("Later I worried my parents would prefer classic leather shoes.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("Finally I decided on Adidas Ultraboost for the trip.", scope, ts("2026-06-04T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you list the order in which I brought up different sneaker shopping experiences and related details? Mention ONLY and ONLY four items.",
            scope,
            budget={"limit": 8},
        )

        anchors = [
            span
            for span in pack.source_spans
            if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor"
        ]
        self.assertGreaterEqual(len(anchors), 4)
        anchor_texts = [span["content"].lower() for span in anchors[:4]]
        self.assertIn("running shoes", anchor_texts[0])
        self.assertIn("nike react", anchor_texts[1])
        self.assertIn("parents", anchor_texts[2])
        self.assertIn("adidas ultraboost", anchor_texts[3])

    def test_event_ordering_scopes_timeline_to_query_topic_inside_mixed_chat(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I'm nervous about improving my writing skills and using Grammarly.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:1:batch1:msg1"})
        memory.add("Practice dialogue flow with a screenplay exercise.", scope, ts("2026-06-01T10:01:00+00:00"), {"source_uri": "beam:test:1:batch1:msg2", "speaker": "assistant"})
        memory.add("I'm building a personal budget tracker in Flask and SQLite.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:1:batch2:msg1"})
        memory.add("Set up the Flask app, database schema, and local server first.", scope, ts("2026-06-02T10:01:00+00:00"), {"source_uri": "beam:test:1:batch2:msg2", "speaker": "assistant"})
        memory.add("Then I wanted transaction CRUD with validation errors.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:1:batch3:msg1"})
        memory.add("Later I worked on deployment to Render with Gunicorn.", scope, ts("2026-06-04T10:00:00+00:00"), {"source_uri": "beam:test:1:batch4:msg1"})

        pack = memory.answer_context(
            "Can you list the order in which I brought up different aspects of developing my personal budget tracker? Mention ONLY and ONLY three items.",
            scope,
            budget={"limit": 8},
        )

        packed_text = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertNotIn("writing skills", packed_text)
        anchors = [
            span
            for span in pack.source_spans
            if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor"
        ]
        self.assertGreaterEqual(len(anchors), 3)
        anchor_text = " ".join(span["content"].lower() for span in anchors)
        self.assertTrue("flask" in anchor_text or "database schema" in anchor_text or "local server" in anchor_text)
        self.assertTrue("transaction crud" in anchor_text or "transaction error" in anchor_text)
        self.assertIn("deployment", anchor_text)

    def test_event_ordering_pack_expands_topic_user_chronology_beyond_selected_anchors(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I started the garden redesign by comparing soil drainage and sunlight.", scope, ts("2026-06-01T10:00:00+00:00"), {"source_uri": "beam:test:30:batch1:msg1"})
        memory.add(
            {
                "role": "assistant",
                "content": "Use raised beds if the drainage stays poor.",
                "turn_id": "beam:test:30:batch1:msg2",
                "timestamp": "2026-06-01T10:01:00+00:00",
            },
            scope,
            ts("2026-06-01T10:01:00+00:00"),
            {"source_uri": "beam:test:30:batch1:msg2"},
        )
        memory.add("Then I chose drought-tolerant plants for the side yard.", scope, ts("2026-06-02T10:00:00+00:00"), {"source_uri": "beam:test:30:batch2:msg1"})
        memory.add("Later I planned drip irrigation zones around the beds.", scope, ts("2026-06-03T10:00:00+00:00"), {"source_uri": "beam:test:30:batch3:msg1"})
        memory.add("Finally I scheduled a weekend mulch delivery and cleanup.", scope, ts("2026-06-04T10:00:00+00:00"), {"source_uri": "beam:test:30:batch4:msg1"})
        memory.add("I also compared headphone noise cancellation for travel.", scope, ts("2026-06-02T09:00:00+00:00"), {"source_uri": "beam:test:31:batch1:msg1"})

        pack = memory.answer_context(
            "Can you list the order in which I brought up different aspects of the garden redesign throughout our conversations?",
            scope,
            budget={"limit": 6, "token_budget": 4000},
        )

        packed_text = " ".join(span["content"].lower() for span in pack.source_spans)
        self.assertIn("soil drainage", packed_text)
        self.assertIn("drought-tolerant plants", packed_text)
        self.assertIn("drip irrigation", packed_text)
        self.assertIn("mulch delivery", packed_text)
        self.assertNotIn("headphone", packed_text)
        user_spans = [span for span in pack.source_spans if span["speaker"] == "user"]
        self.assertGreaterEqual(len(user_spans), 4)
        self.assertEqual([span["timestamp"] for span in user_spans], sorted(span["timestamp"] for span in user_spans))

    def test_event_ordering_coverage_survives_topic_scope_filter(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I'm trying to implement city autocomplete using OpenWeather's Geocoding API v1.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I want a 5-item dropdown and 300ms debounce.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I'm handling invalid city name messages and API failures.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("Later I worked on deployment to GitHub Pages and custom domain support.", scope, ts("2026-06-04T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you list the order in which I brought up different aspects of implementing the city autocomplete feature throughout our conversations? Mention ONLY and ONLY four items.",
            scope,
            budget={"limit": 8},
        )

        anchors = [
            span
            for span in pack.source_spans
            if span.get("selector") == "event_ordering_coverage" and span.get("timeline_role") == "user_aspect_anchor"
        ]
        self.assertGreaterEqual(len(anchors), 4)
        contents = " ".join(span["content"].lower() for span in anchors)
        self.assertIn("autocomplete", contents)
        self.assertIn("dropdown", contents)
        self.assertIn("invalid city", contents)
        self.assertIn("deployment", contents)

    def test_event_ordering_milestone_score_prefers_specific_milestones(self) -> None:
        generic = _event_ordering_milestone_score("I decided on blueprints for auth, transactions, and analytics.")
        specific = _event_ordering_milestone_score("I implemented transaction CRUD response handling and validation errors.")

        self.assertGreater(specific, generic)
        self.assertGreater(_event_ordering_milestone_score("I configured Render deployment with Gunicorn workers and port 10000."), 0.0)
        self.assertGreater(_event_ordering_milestone_score("I expanded integration tests for endpoint coverage."), 0.0)
        self.assertEqual(_event_ordering_milestone_score("I decided deployment can wait until later."), 0.0)

    def test_event_ordering_milestone_classifier_avoids_generic_deployment_noise(self) -> None:
        self.assertIsNone(classify_milestone("I decided Chart.js and dashboard API dependencies can be reviewed before deployment."))
        self.assertNotIn(
            "deployment_configuration",
            classify_milestones("I'm getting TemplateNotFound when I call render_template in my Flask route."),
        )
        self.assertNotIn(
            "deployment_configuration",
            classify_milestones("I'm rendering Chart.js expense pie charts in the dashboard."),
        )
        self.assertEqual(
            classify_milestone("I configured Render deployment with Gunicorn workers and port 10000."),
            "deployment_configuration",
        )
        self.assertEqual(
            classify_milestone("I expanded the deployment test suite with additional security-related tests."),
            "deployment_and_test_improvements",
        )
        self.assertEqual(
            classify_milestone("I want to improve the transaction creation response and error handling."),
            "transaction_error_handling",
        )
        groups = classify_milestones(
            "I'm having issues with deployment on Render.com and Gunicorn port 10000. "
            "I've also completed integration tests covering auth, transaction CRUD, and analytics endpoints with a 95% pass rate."
        )
        self.assertIn("deployment_configuration", groups)
        self.assertIn("integration_test_coverage", groups)

        mentions = extract_milestone_mentions(
            "I'm trying to design a database schema for my personal budget tracker. "
            "The transactions table should include type, amount, date, category, and notes."
        )
        self.assertIn("initial_project_setup", [group for group, _ in mentions])
        self.assertNotIn("transaction_error_handling", [group for group, _ in mentions])

        split_mentions = extract_milestone_mentions(
            "I'm having issues with deployment on Render.com and Gunicorn port 10000. "
            "I've also completed integration tests covering auth, transaction CRUD, and analytics endpoints with a 95% pass rate."
        )
        split_groups = [group for group, _ in split_mentions]
        self.assertIn("deployment_configuration", split_groups)
        self.assertIn("integration_test_coverage", split_groups)
        self.assertNotIn(
            "transaction_crud_implementation",
            classify_milestones("I've completed integration tests covering user auth, transaction CRUD, and analytics endpoints."),
        )
        core_groups = classify_milestones(
            "Can you help me implement the core functionality of my budget tracker, including user authentication, expense tracking, and data visualization?"
        )
        self.assertIn("core_functionality", core_groups)
        self.assertNotIn("security_auth", core_groups)
        setup_groups = classify_milestones(
            "I'm trying to set up my Flask app, but I keep getting TemplateNotFound and OperationalError: no such table."
        )
        self.assertIn("setup_debugging", setup_groups)
        self.assertNotIn("transaction_error_handling", setup_groups)
        long_mentions = extract_milestone_mentions(
            "I'm trying to optimize the dashboard API response time after caching tweaks, "
            "and I want the code to stay compatible with my existing database schema, "
            "and later I need examples for writing unit tests and integration tests for the application with pytest."
        )
        integration_snippets = [snippet for group, snippet in long_mentions if group == "integration_test_coverage"]
        self.assertTrue(integration_snippets)
        self.assertTrue(all("integration tests" in snippet.lower() or "unit tests" in snippet.lower() for snippet in integration_snippets))

    def test_event_ordering_pack_prefers_milestone_events_when_available(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I'm trying to set up the database schema and local server for the app.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I'm implementing transaction CRUD with response handling and validation errors.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I'm configuring Render deployment with Gunicorn workers and port 10000.", scope, ts("2026-06-03T10:00:00+00:00"))
        memory.add("I'm expanding integration tests for endpoint coverage.", scope, ts("2026-06-04T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you walk me through the order in which I brought up different aspects of my app development and deployment across our conversations?",
            scope,
            budget={"limit": 10},
        )

        groups = [event.get("milestone_group") for event in pack.events]
        self.assertIn("initial_project_setup", groups)
        self.assertIn("transaction_crud_implementation", groups)
        self.assertIn("deployment_configuration", groups)
        self.assertIn("integration_test_coverage", groups)
        self.assertEqual([event["timeline_index"] for event in pack.events], list(range(1, len(pack.events) + 1)))

    def test_benchmark_answer_context_expands_retrieval_budget(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I brought up Core functionality for the budget tracker.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("I mentioned Transaction error handling for the budget tracker.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("I discussed Security and deployment for the budget tracker.", scope, ts("2026-06-03T10:00:00+00:00"))

        pack = memory.answer_context(
            "Can you list the order in which I brought up different aspects of developing my personal budget tracker throughout our conversations, in order?",
            scope,
            budget={"mode": "benchmark"},
        )

        self.assertGreaterEqual(pack.coverage["token_budget"], 24000)
        self.assertGreaterEqual(pack.coverage["source_span_quota_required"], 4)
        self.assertGreaterEqual(len(pack.source_spans), 3)

    def test_abstention_sets_policy_when_evidence_insufficient(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("Remember that my database is PostgreSQL.", scope, ts("2026-06-01T10:00:00+00:00"))
        pack = memory.answer_context("What is my Kubernetes cluster name?", scope)
        self.assertEqual(pack.answer_policy, "abstain_if_not_supported")
        self.assertIs(pack.coverage["coverage_insufficient"], True)

    def test_entity_profile_requires_repeated_support(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("Always give me concise technical answers.", scope, ts("2026-06-01T10:00:00+00:00"))
        self.assertEqual(memory.store.list_entity_profiles(scope), [])
        memory.add("Please keep responses concise but include implementation tradeoffs.", scope, ts("2026-06-02T10:00:00+00:00"))
        profiles = memory.store.list_entity_profiles(scope)
        self.assertTrue(profiles)
        self.assertTrue(profiles[0].source_span_ids)
        self.assertGreaterEqual(profiles[0].support_count, 2)

    def test_document_input_is_chunked_with_overlap(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        content = " ".join(f"Token{i}" for i in range(35))
        result = memory.add(
            {"role": "document", "content": content, "source_uri": "doc://atlas", "chunk_size_tokens": 10, "chunk_overlap_tokens": 2},
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        spans = [memory.store.get_span(span_id) for span_id in result.span_ids]
        chunks = [span for span in spans if span and span.span_type == "document_chunk"]
        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(all(chunk.source_uri == "doc://atlas" for chunk in chunks))
        self.assertIn("Token8", chunks[0].content)
        self.assertIn("Token8", chunks[1].content)

    def test_session_window_is_written_but_not_extracted_as_fact(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        result = memory.add(
            [
                {"role": "user", "content": "I prefer Qdrant for Atlas."},
                {"role": "assistant", "content": "I will remember that."},
                {"role": "user", "content": "Always answer with concise tradeoffs."},
            ],
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"min_window_spans": 3, "window_size": 3},
        )
        spans = [memory.store.get_span(span_id) for span_id in result.span_ids]
        windows = [span for span in spans if span and span.span_type == "window"]
        self.assertEqual(len(windows), 1)
        facts = memory.store.list_facts(scope)
        self.assertFalse(any(fact.metadata.get("candidate_local_id") and "assistant: I will remember" in fact.text for fact in facts))

    def test_session_summary_is_refreshable_idempotent_and_retrievable(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            [
                {"role": "user", "content": "Atlas uses Qdrant for retrieval."},
                {"role": "assistant", "content": "I noted the Atlas backend."},
                {"role": "user", "content": "Reports use PostgreSQL."},
                {"role": "assistant", "content": "I will keep reports on PostgreSQL."},
                {"role": "user", "content": "Reranking should use a cross encoder."},
                {"role": "assistant", "content": "I will include reranking in the retrieval plan."},
            ],
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"min_window_spans": 3, "window_size": 3},
        )
        fact_count = len(memory.store.list_facts(scope))

        summary = memory.refresh_session_summary(scope)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.span_type, "summary")
        self.assertIn("Atlas", summary.content)
        self.assertIn("Qdrant", summary.content)
        self.assertEqual(summary.metadata["source_span_count"], 6)
        self.assertEqual(len(memory.store.list_facts(scope)), fact_count)

        duplicate = memory.refresh_session_summary(scope)
        self.assertEqual(duplicate.span_id, summary.span_id)
        summaries = memory.get_session_summaries(scope)
        self.assertEqual([item.span_id for item in summaries], [summary.span_id])

        result = memory.search("Which retrieval backend did Atlas use?", scope, options={"enabled_sources": ["raw"], "limit": 6})
        self.assertTrue(any(candidate.id == summary.span_id for candidate in result.candidates))

    def test_session_summary_background_task_is_enqueued_processed_and_idempotent(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")

        memory.add(
            [
                {"role": "user", "content": "Atlas uses Qdrant for retrieval."},
                {"role": "assistant", "content": "I noted the Atlas backend."},
                {"role": "user", "content": "Reports use PostgreSQL."},
                {"role": "assistant", "content": "I will keep reports on PostgreSQL."},
                {"role": "user", "content": "Reranking should use a cross encoder."},
                {"role": "assistant", "content": "I will include reranking in the retrieval plan."},
            ],
            scope,
            ts("2026-06-01T10:00:00+00:00"),
            {"min_window_spans": 3, "window_size": 3},
        )

        pending = memory.list_background_tasks(scope, status="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["task_type"], "refresh_session_summary")
        self.assertEqual(pending[0]["payload"]["source_span_count"], 6)
        self.assertEqual(pending[0]["attempts"], 0)

        processed = memory.process_background_tasks(scope, limit=5)
        self.assertEqual(processed["processed_count"], 1)
        self.assertEqual(processed["status_counts"], {"succeeded": 1})
        self.assertEqual(processed["tasks"][0]["attempts"], 1)

        summaries = memory.get_session_summaries(scope)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(processed["tasks"][0]["payload"]["result"]["summary_span_id"], summaries[0].span_id)

        duplicate_run = memory.process_background_tasks(scope, limit=5)
        self.assertEqual(duplicate_run["processed_count"], 0)
        self.assertEqual(len(memory.get_session_summaries(scope)), 1)

    def test_session_summary_background_task_is_not_enqueued_below_threshold(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")

        memory.add(
            [
                {"role": "user", "content": "Atlas uses Qdrant for retrieval."},
                {"role": "assistant", "content": "I noted the Atlas backend."},
            ],
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )

        self.assertEqual(memory.list_background_tasks(scope), [])

    def test_async_llm_extractor_does_not_block_add_and_runs_in_background(self) -> None:
        class AttributedClient(StaticLLMClient):
            def __init__(self) -> None:
                super().__init__({})

            def structured(self, prompt: str, schema: dict[str, object], input: dict[str, object]) -> dict[str, object]:
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                span_id = input["spans"][0]["span_id"]  # type: ignore[index]
                return {
                    "facts": [
                        {
                            "text": "Atlas production retrieval uses Postgres pgvector.",
                            "subject": "Atlas production retrieval",
                            "predicate": "uses",
                            "object": "Postgres pgvector",
                            "category": "project_state",
                            "confidence": 0.91,
                            "salience": 0.8,
                            "source_span_ids": [span_id],
                        }
                    ],
                    "events": [],
                    "relations": [],
                }

        client = AttributedClient()
        extractor = StructuredLLMExtractor(client)
        memory = MemoryService(async_extractor=extractor)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")

        result = memory.add(
            {"role": "user", "content": "For Atlas production retrieval, use Postgres pgvector."},
            scope,
            ts("2026-01-01T10:00:00+00:00"),
        )

        self.assertTrue(result.span_ids)
        self.assertEqual(client.calls, [])
        pending = memory.list_background_tasks(scope, status="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["task_type"], "llm_extract")
        self.assertEqual(pending[0]["payload"]["source_span_ids"], result.span_ids)

        processed = memory.process_background_tasks(scope, limit=5)

        self.assertEqual(processed["status_counts"], {"succeeded": 1})
        self.assertEqual(len(client.calls), 1)
        task_result = processed["tasks"][0]["payload"]["result"]
        self.assertEqual(task_result["candidate_count"], 1)
        self.assertEqual(task_result["gate_decision_counts"].get("accept"), 1)
        self.assertEqual(len(memory.store.list_facts(scope)), 0)

    def test_entities_are_persisted_and_used_as_retrieval_source(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        memory.add("I prefer Qdrant for Atlas retrieval.", scope, ts("2026-06-01T10:00:00+00:00"))
        entities = memory.store.list_entities(scope)
        names = {entity.name for entity in entities}
        self.assertIn("Qdrant", names)
        self.assertIn("Atlas", names)
        result = memory.search("Atlas", scope)
        self.assertTrue(any("entity_registry" in candidate.source for candidate in result.candidates))

    def test_broad_raw_recall_uses_query_intent_to_rescue_current_state_evidence(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            [
                {"role": "user", "content": "I used to track the project status in a spreadsheet."},
                {"role": "assistant", "content": "That was the previous tracking setup."},
                {"role": "user", "content": "Now the project status tracker is updated in Notion, with current owners and due dates."},
                {"role": "user", "content": "For dinner I bought oranges and rice."},
            ],
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        plan = memory.planner.plan("What is the current project status tracker?")

        candidates = memory._broad_raw_recall_candidates(
            "What is the current project status tracker?",
            scope,
            plan,
            limit=8,
            include_session=True,
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].source, "broad_raw_recall")
        self.assertTrue(candidates[0].metadata["broad_raw_recall"])
        self.assertIn("Notion", candidates[0].text)
        self.assertGreaterEqual(candidates[0].scores["intent_recall_signal"], 0.16)
        self.assertFalse(any("oranges" in candidate.text for candidate in candidates[:3]))

    def test_current_value_query_prioritizes_latest_correction_over_historical_value(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            {"role": "user", "content": "For Project Atlas, I initially prefer Qdrant for retrieval experiments."},
            scope,
            ts("2026-01-01T10:00:00+00:00"),
        )
        memory.add(
            {"role": "user", "content": "I switched Project Atlas retrieval from Qdrant to Postgres pgvector for production."},
            scope,
            ts("2026-01-08T10:00:00+00:00"),
        )
        memory.add(
            {"role": "user", "content": "I no longer want Qdrant for Atlas production; keep it only as historical context."},
            scope,
            ts("2026-03-01T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "What retrieval backend does Project Atlas currently use?",
            scope,
            budget={"allow_cross_session": True, "limit": 4},
        )

        evidence = "\n".join(span["content"] for span in pack.source_spans)
        self.assertIn("Postgres pgvector", evidence)
        self.assertNotIn("initially prefer Qdrant", evidence)

    def test_chinese_error_query_recalls_traceback_guidance(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            {"role": "user", "content": "中文备注：新手错误提示必须说明下一步，不要暴露 traceback。"},
            scope,
            ts("2026-01-01T10:00:00+00:00"),
        )
        memory.add(
            {"role": "user", "content": "Security note: API keys must be referenced by environment variable or file path only."},
            scope,
            ts("2026-01-02T10:00:00+00:00"),
        )
        memory.add(
            {"role": "user", "content": "写入测试失败的原因是数据库没启动。给用户的提示应该是“数据库还没启动，请点击启动或重试”，不要说 psycopg 连接异常。"},
            scope,
            ts("2026-01-03T10:00:00+00:00"),
        )
        memory.add(
            {"role": "user", "content": "如果端口被占用，产品提示要说“端口被占用，请关闭旧服务或换一个端口”，不能暴露 socket bind failed。"},
            scope,
            ts("2026-01-04T10:00:00+00:00"),
        )

        pack = memory.answer_context(
            "新手错误提示不能暴露什么？",
            scope,
            budget={"allow_cross_session": True, "limit": 4},
        )

        evidence = "\n".join(span["content"] for span in pack.source_spans)
        self.assertIn("traceback", evidence)

        pack = memory.answer_context(
            "如果数据库没启动或端口被占用，应该怎样提示小白用户？",
            scope,
            budget={"allow_cross_session": True, "limit": 6, "mode": "benchmark"},
        )

        evidence = "\n".join(span["content"] for span in pack.source_spans)
        self.assertIn("数据库还没启动", evidence)
        self.assertIn("端口被占用", evidence)

    def test_event_ordering_compaction_preserves_broad_raw_provenance(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            [
                {"role": "user", "content": "I am concerned about using AI for hiring in my company."},
                {"role": "assistant", "content": "That raises fairness and transparency issues."},
                {"role": "user", "content": "I also want to make sure the pilot program improves screening quality."},
            ],
            scope,
            ts("2026-06-01T10:00:00+00:00"),
        )
        plan = memory.planner.plan("What order did I bring up the hiring concerns?")
        pack = memory.answer_context(
            "What order did I bring up the hiring concerns?",
            scope,
            budget={"mode": "benchmark", "limit": 24, "rerank_top_n": 48},
        )
        model_pack = __import__("fusion_memory.eval.model_adapters", fromlist=["_pack_for_model"])._pack_for_model(pack)

        self.assertTrue(pack.coverage["selected_candidate_sources"])
        self.assertIn("broad_raw_recall", str(pack.coverage["selected_candidate_sources"]))
        if model_pack.get("sequence_items"):
            item = model_pack["sequence_items"][0]
            self.assertIn("source_span_id", item)
        self.assertTrue(any(span.get("candidate_source") for span in model_pack.get("timeline", [])))


if __name__ == "__main__":
    unittest.main()
