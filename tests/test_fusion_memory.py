from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class FusionMemoryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
