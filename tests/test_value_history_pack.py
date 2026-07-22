from __future__ import annotations

import unittest

from fusion_memory.retrieval.slot_state_transition import value_state_summary
from fusion_memory.retrieval.value_history_pack import (
    build_value_history_table,
    exact_candidate_value_rows,
    value_history_summary,
    value_history_target_type_priority,
    value_mentions,
)


class ValueHistoryPackTests(unittest.TestCase):
    def test_value_history_section_prefers_query_unit_current_value(self) -> None:
        summary = value_history_summary(
            "What is the current response time in milliseconds?",
            [
                {
                    "value_type": "duration",
                    "value": "4 hours",
                    "context": "The planning window is currently 4 hours.",
                    "current": True,
                    "speaker": "assistant",
                    "query_overlap": 1,
                },
                {
                    "value_type": "latency",
                    "value": "250ms",
                    "context": "The dashboard API response time is currently 250ms.",
                    "current": True,
                    "speaker": "user",
                    "query_overlap": 4,
                },
            ],
        )

        self.assertEqual(summary["resolved_current_value"], "250ms")

    def test_exact_candidate_value_rows_keep_target_goal_out_of_current(self) -> None:
        rows = exact_candidate_value_rows(
            "How many books am I aiming to read?",
            [
                {
                    "source_span_id": "s1",
                    "speaker": "user",
                    "content": "I am aiming for 12 books by March 1.",
                    "value_mentions": [
                        {
                            "type": "count",
                            "text": "12 books",
                            "context": "I am aiming for 12 books by March 1.",
                            "update_marker_strength": 1.0,
                        }
                    ],
                }
            ],
        )

        self.assertEqual(rows[0]["value"], "12 books")
        self.assertFalse(rows[0]["current"])

    def test_value_history_summary_prefers_current_percentage_over_baseline_in_same_context(self) -> None:
        summary = value_history_summary(
            "What accuracy rate does the AI screening tool achieve in its evaluations?",
            [
                {
                    "value_type": "percentage",
                    "value": "75%",
                    "context": "The AI screening accuracy is now rated 87%, a big improvement from the 75% manual match rate last year.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "previous",
                    "current": False,
                },
                {
                    "value_type": "percentage",
                    "value": "87%",
                    "context": "The AI screening accuracy is now rated 87%, a big improvement from the 75% manual match rate last year.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                },
            ],
        )

        self.assertEqual(summary["resolved_current_value"], "87%")

    def test_value_mentions_extracts_duration_ranges(self) -> None:
        values = value_mentions("The probate process was shortened to 5-7 months after reforms.")

        self.assertTrue(any(item["type"] == "duration" and item["text"] == "5-7 months" for item in values))
        self.assertFalse(any(item["type"] == "duration" and item["text"] == "7 months" for item in values))

    def test_value_history_target_type_detects_how_long_duration(self) -> None:
        target_types = value_history_target_type_priority("How long does the probate process usually take?")

        self.assertEqual(target_types[0], "duration")

    def test_value_history_target_type_detects_how_many_days_duration(self) -> None:
        target_types = value_history_target_type_priority("How many days are scheduled for the sessions?")

        self.assertEqual(target_types[0], "duration")

    def test_value_history_table_pairs_dates_to_nearest_labeled_slot(self) -> None:
        rows = build_value_history_table(
            "By what date am I aiming to complete all my onboarding modules?",
            [
                {
                    "id": "s1",
                    "speaker": "user",
                    "content": (
                        "Sure, let's confirm a few dates: **Team-Building Event**: Yep, it's on April 10. "
                        "**Onboarding Modules Deadline**: Correct, I need to finish all the onboarding modules by April 25."
                    ),
                    "history_index": 4,
                    "recency_rank": 1,
                }
            ],
            [],
        )

        by_value = {row["value"]: row for row in rows}
        self.assertIn("April 25", by_value)
        self.assertIn("Onboarding Modules Deadline", by_value["April 25"]["context"])
        self.assertGreater(by_value["April 25"]["query_overlap"], by_value["April 10"]["query_overlap"])
        summary = value_history_summary("By what date am I aiming to complete all my onboarding modules?", rows)
        self.assertEqual(summary["resolved_current_value"], "April 25")
        self.assertEqual(summary["preferred_current_candidate"]["value_role"], "target")
        self.assertFalse(by_value["April 25"]["current"])

    def test_slot_state_transition_prefers_updated_quota_over_adjacent_count(self) -> None:
        summary = value_state_summary(
            "What is the daily call quota for the API key used in my application?",
            [
                {
                    "value_type": "count",
                    "value": "5 items",
                    "context": "I decided to limit autocomplete results to 5 items to reduce API calls.",
                    "speaker": "user",
                    "query_overlap": 2,
                    "slot_overlap": 2,
                    "value_role": "mentioned",
                    "current": True,
                    "recency_rank": 1,
                },
                {
                    "value_type": "count",
                    "value": "1,200 calls per day",
                    "context": "I'm trying to update my API key settings to reflect the new daily quota of 1,200 calls per day.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 5,
                    "value_role": "target",
                    "current": False,
                    "update_marker_strength": 1.5,
                    "history_index": 22,
                    "recency_rank": 8,
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "1,200 calls per day")
        self.assertIn("state_transition", summary["preferred_state"]["state_reasons"])

    def test_slot_state_transition_prefers_updated_same_slot_value(self) -> None:
        summary = value_state_summary(
            "What is the test coverage percentage for my API integration module?",
            [
                {
                    "value_type": "percentage",
                    "value": "65%",
                    "context": "I'm trying to achieve 100% test coverage on my API integration module, and I've currently reached 65% after my initial test suite run.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "update_marker_strength": 0.2,
                    "history_index": 16,
                    "recency_rank": 20,
                },
                {
                    "value_type": "percentage",
                    "value": "78%",
                    "context": "I'm trying to increase the unit test coverage for my API integration, which has recently improved to 78%.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "update_marker_strength": 2.0,
                    "history_index": 23,
                    "recency_rank": 1,
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "78%")

    def test_slot_state_transition_prefers_later_same_slot_count_replacement(self) -> None:
        summary = value_state_summary(
            "How many sources are in my Zotero library?",
            [
                {
                    "value_type": "count",
                    "value": "45 sources",
                    "context": "My Zotero library has 45 sources, tagged by theme and date.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 16,
                    "recency_rank": 35,
                    "subject_key": "subject:library_source_sources_zotero",
                },
                {
                    "value_type": "count",
                    "value": "52 sources",
                    "context": "I've added 52 sources to my Zotero library and need to organize them for my essay.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "mentioned",
                    "current": False,
                    "history_index": 20,
                    "recency_rank": 31,
                    "subject_key": "subject:library_source_sources_zotero",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "52 sources")
        self.assertEqual(summary["preferred_state"]["state_role"], "updated")
        self.assertIn("same_slot_replacement", summary["preferred_state"]["state_reasons"])

    def test_slot_state_transition_prefers_later_same_slot_deadline_replacement(self) -> None:
        summary = value_state_summary(
            "What is the deadline for completing the first sprint focused on the basic layout and navigation?",
            [
                {
                    "value_type": "date",
                    "value": "April 1, 2024",
                    "context": "The first sprint deadline is April 1, 2024 for the basic layout and navigation.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 6,
                    "value_role": "target",
                    "current": False,
                    "history_index": 8,
                    "recency_rank": 42,
                    "subject_key": "subject:basic_deadline_first_layout",
                },
                {
                    "value_type": "date",
                    "value": "April 5, 2024",
                    "context": "I need to update the project timeline to reflect the new sprint deadline of April 5, 2024.",
                    "speaker": "user",
                    "query_overlap": 3,
                    "slot_overlap": 3,
                    "value_role": "current",
                    "current": True,
                    "history_index": 25,
                    "recency_rank": 25,
                    "subject_key": "subject:deadline_sprint",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "April 5, 2024")
        self.assertIn("same_slot_replacement", summary["preferred_state"]["state_reasons"])

    def test_slot_state_transition_ignores_later_different_phase_effective_date(self) -> None:
        summary = value_state_summary(
            "What is the deadline for completing the first sprint focused on the basic layout and navigation?",
            [
                {
                    "value_type": "date",
                    "value": "April 1, 2024",
                    "context": "The first sprint deadline is April 1, 2024 for the basic layout and navigation.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 6,
                    "value_role": "target",
                    "current": False,
                    "history_index": 8,
                    "recency_rank": 42,
                    "subject_key": "subject:basic_deadline_first_layout",
                },
                {
                    "value_type": "date",
                    "value": "April 5, 2024",
                    "context": "I need to update the project timeline to reflect the new sprint deadline of April 5, 2024.",
                    "speaker": "user",
                    "query_overlap": 3,
                    "slot_overlap": 3,
                    "value_role": "current",
                    "current": True,
                    "history_index": 25,
                    "recency_rank": 25,
                    "subject_key": "subject:deadline_sprint",
                },
                {
                    "value_type": "date",
                    "value": "April 20, 2024",
                    "context": "Sprint 2 has a deadline of April 20, 2024 and focuses on SEO basics.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "target",
                    "current": False,
                    "history_index": 29,
                    "recency_rank": 21,
                    "subject_key": "subject:basic_deadline_focus_sprint",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "April 5, 2024")
        self.assertIn("same_slot_replacement", summary["preferred_state"]["state_reasons"])

    def test_slot_state_transition_uses_effective_date_over_later_recap(self) -> None:
        summary = value_state_summary(
            "What is the monthly grocery budget Alexis and I have agreed on?",
            [
                {
                    "value_type": "money",
                    "value": "$550",
                    "context": "The grocery budget was increased to $550 monthly starting September 15.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "update_marker_strength": 2.0,
                    "history_index": 21,
                    "recency_rank": 30,
                    "subject_key": "subject:budget_grocery_monthly",
                },
                {
                    "value_type": "money",
                    "value": "$500",
                    "context": "Considering we've already increased our grocery budget to $500 monthly starting Sept 1, how should we adjust the joint budget?",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 25,
                    "recency_rank": 26,
                    "subject_key": "subject:budget_grocery_monthly",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "$550")

    def test_slot_state_transition_resolves_pronominal_attribute_adjustment(self) -> None:
        summary = value_state_summary(
            "What is my weekly word count target for my writing goals?",
            [
                {
                    "value_type": "count",
                    "value": "1,200 words",
                    "context": "I'm targeting 1,200 words per week for my writing goals.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 4,
                    "value_role": "target",
                    "current": False,
                    "history_index": 4,
                    "recency_rank": 47,
                    "subject_key": "subject:count_goal_goals_target",
                },
                {
                    "value_type": "count",
                    "value": "1,350 words",
                    "context": "I'm trying to increase my weekly word count, and I just found out it was adjusted to 1,350 words.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 2,
                    "value_role": "current",
                    "current": True,
                    "update_marker_strength": 2.0,
                    "history_index": 16,
                    "recency_rank": 35,
                    "subject_key": "subject:count_weekly_word",
                },
                {
                    "value_type": "count",
                    "value": "1,800 words",
                    "context": "I'm trying to meet my writing goals by October 1, 2024, with a weekly target of 1,800 words.",
                    "speaker": "user",
                    "query_overlap": 7,
                    "slot_overlap": 5,
                    "value_role": "target",
                    "current": False,
                    "update_marker_strength": 1.0,
                    "history_index": 36,
                    "recency_rank": 15,
                    "subject_key": "subject:goal_goals_target_weekly",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "1,350 words")
        self.assertIn("strong_transition", summary["preferred_state"]["state_reasons"])
        self.assertIn("word", summary["preferred_state"]["slot_terms"])

    def test_slot_state_transition_does_not_replace_different_meeting(self) -> None:
        summary = value_state_summary(
            "When is my final decision meeting scheduled to take place?",
            [
                {
                    "value_type": "date",
                    "value": "March 30",
                    "context": "I'm worried about making the right decision on March 30, so I rescheduled my final meeting to have more time.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 6,
                    "recency_rank": 45,
                    "subject_key": "subject:decision_final_meet_meeting",
                },
                {
                    "value_type": "date",
                    "value": "June 3",
                    "context": "I feel bad about missing the meeting with Matthew, and now it's rescheduled for June 3 at 11 AM.",
                    "speaker": "user",
                    "query_overlap": 2,
                    "slot_overlap": 2,
                    "value_role": "current",
                    "current": True,
                    "update_marker_strength": 2.0,
                    "history_index": 24,
                    "recency_rank": 27,
                    "subject_key": "subject:meet_meeting",
                },
            ],
        )

        self.assertNotEqual(summary.get("resolved_value"), "June 3")
        self.assertFalse(
            any(candidate.get("value") == "June 3" and "same_slot_replacement" in candidate.get("state_reasons", []) for candidate in summary["state_candidates"])
        )

    def test_slot_state_transition_does_not_replace_different_slot_value(self) -> None:
        summary = value_state_summary(
            "What is the deadline for completing the first sprint focused on the basic layout and navigation?",
            [
                {
                    "value_type": "date",
                    "value": "April 1, 2024",
                    "context": "The first sprint deadline is April 1, 2024 for the basic layout and navigation.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 6,
                    "value_role": "target",
                    "current": False,
                    "history_index": 8,
                    "recency_rank": 42,
                    "subject_key": "subject:basic_deadline_first_layout",
                },
                {
                    "value_type": "date",
                    "value": "April 20, 2024",
                    "context": "Sprint 2 has a deadline of April 20, 2024 and focuses on SEO basics.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "target",
                    "current": False,
                    "history_index": 29,
                    "recency_rank": 21,
                    "subject_key": "subject:basic_deadline_focus_sprint",
                },
            ],
        )

        self.assertTrue(summary["candidate_only"])
        self.assertNotIn("resolved_value", summary)

    def test_slot_state_transition_does_not_resolve_adjacent_current_value_without_slot_anchor(self) -> None:
        summary = value_state_summary(
            "What is the test coverage percentage for my API integration module?",
            [
                {
                    "value_type": "percentage",
                    "value": "90%",
                    "context": "The API integration module rollout is now 90% complete after the deployment checklist.",
                    "speaker": "user",
                    "query_overlap": 3,
                    "slot_overlap": 3,
                    "value_role": "current",
                    "current": True,
                    "update_marker_strength": 1.2,
                    "history_index": 25,
                    "recency_rank": 1,
                },
            ],
        )

        self.assertTrue(summary["candidate_only"])
        self.assertNotIn("resolved_value", summary)

    def test_slot_state_transition_does_not_single_resolve_multi_facet_query(self) -> None:
        summary = value_state_summary(
            "How many classification problems have I completed, and what accuracy rate have I maintained?",
            [
                {
                    "value_type": "count",
                    "value": "12 problems",
                    "context": "I completed 12 area calculation problems and improved to 90% accuracy.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 20,
                },
                {
                    "value_type": "count",
                    "value": "15 problems",
                    "context": "I completed 15 classification problems with 80% accuracy.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 18,
                },
            ],
        )

        self.assertTrue(summary["candidate_only"])
        self.assertNotIn("resolved_value", summary)

    def test_slot_state_transition_prefers_later_same_slot_secured_count(self) -> None:
        summary = value_state_summary(
            "How many interviews have I secured for executive producer roles during the recent period?",
            [
                {
                    "value_type": "count",
                    "value": "3 interviews",
                    "context": "I only secured 3 interviews for executive producer roles between April 25 and May 1.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 63,
                    "recency_rank": 6,
                    "subject_key": "subject:executive_interviews_producer_roles",
                },
                {
                    "value_type": "count",
                    "value": "5 interviews",
                    "context": "I secured 5 interviews for executive producer roles between April 25 and May 1 and want to leverage that fact.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 93,
                    "recency_rank": 1,
                    "subject_key": "subject:executive_interviews_producer_roles",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "5 interviews")
        self.assertIn("latest_same_slot", summary["preferred_state"]["state_reasons"])

    def test_slot_state_transition_treats_free_time_as_plan_update(self) -> None:
        summary = value_state_summary(
            "What time should I plan to visit Foot Locker next Saturday?",
            [
                {
                    "value_type": "time",
                    "value": "3 PM",
                    "context": "I'm planning to visit Foot Locker next Saturday at 3 PM.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 6,
                    "value_role": "target",
                    "current": False,
                    "history_index": 35,
                    "recency_rank": 8,
                    "subject_key": "subject:foot_locker_saturday_visit",
                },
                {
                    "value_type": "time",
                    "value": "4 PM",
                    "context": "I'm going to reschedule other appointments to make sure I'm free at 4 PM next Saturday for my Foot Locker visit.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 6,
                    "value_role": "mentioned",
                    "current": False,
                    "history_index": 57,
                    "recency_rank": 1,
                    "subject_key": "subject:foot_locker_saturday_visit",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "4 PM")

    def test_slot_state_transition_does_not_treat_budget_cap_as_superseding_adjusted_budget(self) -> None:
        summary = value_state_summary(
            "What is my total budget for holiday gifts this year?",
            [
                {
                    "value_type": "money",
                    "value": "$450",
                    "context": "I've adjusted our holiday gift budget to $450 and need to allocate it among family members.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 267,
                    "recency_rank": 5,
                    "subject_key": "subject:holiday_gifts_budget",
                },
                {
                    "value_type": "money",
                    "value": "$400",
                    "context": "I'm anxious about holiday spending, especially since I capped my gifts budget at $400 total.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 315,
                    "recency_rank": 1,
                    "subject_key": "subject:holiday_gifts_budget",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "$450")

    def test_slot_state_transition_does_not_replace_goal_count_with_progress_count(self) -> None:
        summary = value_state_summary(
            "How many books am I aiming to read in my winter reading challenge?",
            [
                {
                    "value_type": "count",
                    "value": "12 books",
                    "context": "I've extended my reading challenge goal to 12 books by March 1.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 244,
                    "recency_rank": 8,
                    "subject_key": "subject:winter_reading_challenge_goal",
                },
                {
                    "value_type": "count",
                    "value": "three books",
                    "context": "I'm deciding on my next read after finishing the first three books of The Expanse series.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 280,
                    "recency_rank": 1,
                    "subject_key": "subject:winter_reading_challenge_books",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "12 books")
        self.assertEqual(summary["resolved_label"], "12 books by March 1")
        self.assertIn({"type": "deadline", "value": "March 1"}, summary["preferred_state"]["qualifiers"])

    def test_slot_state_transition_prefers_later_adjusted_same_slot_budget(self) -> None:
        summary = value_state_summary(
            "What is my snack budget for ordering themed treats for the movie marathon?",
            [
                {
                    "value_type": "money",
                    "value": "$65",
                    "context": "I've increased my snack budget to $65 so I can order themed cupcakes from The Sweet Spot bakery.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 151,
                    "recency_rank": 118,
                    "subject_key": "subject:snack_budget_movie_marathon",
                },
                {
                    "value_type": "money",
                    "value": "$75",
                    "context": "I'm planning a movie marathon for my family and I've adjusted the snack budget to $75.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 175,
                    "recency_rank": 94,
                    "subject_key": "subject:snack_budget_movie_marathon",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "$75")

    def test_slot_state_transition_resolves_multi_money_fee_pair_before_single_update(self) -> None:
        summary = value_state_summary(
            "What budget have I set for the initial patent filing fees and attorney fees?",
            [
                {
                    "value_type": "money",
                    "value": "$4,000",
                    "context": "I've allocated $4,000 for initial patent filing fees and $5,500 for attorney fees by July 2024.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 46,
                    "recency_rank": 30,
                    "subject_key": "subject:initial_patent_filing_fees_attorney_fees",
                },
                {
                    "value_type": "money",
                    "value": "$5,500",
                    "context": "I've allocated $4,000 for initial patent filing fees and $5,500 for attorney fees by July 2024.",
                    "speaker": "user",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 46,
                    "recency_rank": 30,
                    "subject_key": "subject:initial_patent_filing_fees_attorney_fees",
                },
                {
                    "value_type": "money",
                    "value": "$3,500",
                    "context": "You have budgeted $3,500 for initial patent filing fees and $5,000 for attorney fees by July 2024.",
                    "speaker": "assistant",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 237,
                    "recency_rank": 5,
                    "subject_key": "subject:initial_patent_filing_fees_attorney_fees",
                },
                {
                    "value_type": "money",
                    "value": "$5,000",
                    "context": "You have budgeted $3,500 for initial patent filing fees and $5,000 for attorney fees by July 2024.",
                    "speaker": "assistant",
                    "query_overlap": 6,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 237,
                    "recency_rank": 5,
                    "subject_key": "subject:initial_patent_filing_fees_attorney_fees",
                },
            ],
        )

        self.assertEqual(summary["resolved_values"], ["$4,000", "$5,500"])

    def test_slot_state_transition_keeps_structured_assistant_event_update_as_candidate(self) -> None:
        summary = value_state_summary(
            "When is my Zoom call with the creative director scheduled?",
            [
                {
                    "value_type": "date",
                    "value": "April 21",
                    "context": "I've accepted Leslie's introduction offer and have a Zoom call with the creative director on April 21 at 3 PM.",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 87,
                    "recency_rank": 182,
                    "subject_key": "subject:zoom_call_creative_director",
                },
                {
                    "value_type": "date",
                    "value": "April 22",
                    "context": "**Event**: Zoom call with the creative director on April 22 at 11 AM",
                    "speaker": "assistant",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 157,
                    "recency_rank": 112,
                    "subject_key": "subject:zoom_call_creative_director",
                },
            ],
        )

        self.assertTrue(any(candidate.get("value") == "April 22" for candidate in summary["state_candidates"]))

    def test_slot_state_transition_rejects_adjacent_percentage_slot(self) -> None:
        summary = value_state_summary(
            "What accuracy rate does the AI screening tool achieve in its evaluations?",
            [
                {
                    "value_type": "percentage",
                    "value": "45%",
                    "context": "I've reduced screening time by 45% with HireVue, but I need to ensure data quality.",
                    "speaker": "user",
                    "query_overlap": 2,
                    "slot_overlap": 2,
                    "value_role": "current",
                    "current": True,
                    "history_index": 287,
                    "recency_rank": 101,
                    "subject_key": "subject:ai_screening_hirevue",
                },
                {
                    "value_type": "percentage",
                    "value": "90%",
                    "context": "What's the best way to ensure the AI screening tool's 90% accuracy doesn't introduce bias into my hiring process?",
                    "speaker": "user",
                    "query_overlap": 4,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 47,
                    "recency_rank": 341,
                    "subject_key": "subject:ai_screening_tool_accuracy",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "90%")

    def test_slot_state_transition_prefers_shortened_duration_update(self) -> None:
        summary = value_state_summary(
            "How long does the probate process usually take in Montserrat?",
            [
                {
                    "value_type": "duration",
                    "value": "6-9 months",
                    "context": "The probate process in Montserrat typically takes 6-9 months.",
                    "speaker": "assistant",
                    "query_overlap": 5,
                    "slot_overlap": 4,
                    "value_role": "current",
                    "current": True,
                    "history_index": 20,
                    "recency_rank": 20,
                    "subject_key": "subject:probate_process_montserrat",
                },
                {
                    "value_type": "duration",
                    "value": "5-7 months",
                    "context": "The probate process was shortened to 5-7 months after recent legal reforms.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 4,
                    "value_role": "mentioned",
                    "current": False,
                    "history_index": 40,
                    "recency_rank": 1,
                    "subject_key": "subject:probate_process_montserrat",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "5-7 months")

    def test_slot_state_transition_derives_days_from_extended_date_range(self) -> None:
        summary = value_state_summary(
            "How many days are scheduled for the sound mixing sessions with Jeremy?",
            [
                {
                    "value_type": "date",
                    "value": "July 15",
                    "context": "The sound mixing schedule was finalized for July 12-15 at Montserrat Studios.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 21,
                    "recency_rank": 30,
                    "subject_key": "subject:jeremy_sound_mixing_sessions",
                },
                {
                    "value_type": "date",
                    "value": "July 16",
                    "context": "The sound mixing sessions with Jeremy got extended to July 16 due to additional edits requested on July 10.",
                    "speaker": "user",
                    "query_overlap": 5,
                    "slot_overlap": 5,
                    "value_role": "current",
                    "current": True,
                    "history_index": 26,
                    "recency_rank": 1,
                    "subject_key": "subject:jeremy_sound_mixing_sessions",
                },
            ],
        )

        self.assertEqual(summary["resolved_value"], "5 days")


if __name__ == "__main__":
    unittest.main()
