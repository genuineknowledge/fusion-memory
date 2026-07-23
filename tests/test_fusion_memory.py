from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fusion_memory import MemoryService, Scope
from fusion_memory.core.llm import StaticLLMClient
from fusion_memory.ingestion.extractors import extract_generic_event_facets
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class FusionMemoryTests(unittest.TestCase):
    def test_explicit_event_order_mentions_preserve_write_side_marker_parsing(self) -> None:
        from fusion_memory.ingestion.order_markers import _explicit_order_mentions

        self.assertEqual(
            _explicit_order_mentions(
                "After the BM25 test, I added dense retrieval. Before deployment, we reviewed alerts."
            ),
            [("BM25 test", "after"), ("deployment", "before")],
        )

    def test_chinese_rule_extractor_accepts_user_preference(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")

        result = memory.add("我现在更喜欢用 PostgreSQL 做报表。", scope, ts("2026-06-30T10:00:00+00:00"))

        self.assertTrue(result.accepted_fact_ids)
        facts = memory.store.list_facts(scope)
        self.assertTrue(any(fact.category == "preference" and fact.predicate == "prefers" and "PostgreSQL" in fact.object for fact in facts))

    def test_chinese_preference_query_populates_public_answer_context(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        try:
            memory.add(
                "Father 喜欢喝冰美式咖啡。",
                scope,
                ts("2026-07-06T10:00:00+00:00"),
            )

            pack = memory.answer_context(
                "Father 喝什么饮料？",
                scope,
                budget={"limit": 6, "allow_cross_session": True},
            )

            self.assertTrue(pack.current_views)
            self.assertTrue(pack.facts)
            self.assertTrue(any("Father" in view["text"] for view in pack.current_views))
            self.assertTrue(any("冰美式咖啡" in view["text"] for view in pack.current_views))
            self.assertTrue(any("Father" in fact["text"] for fact in pack.facts))
            self.assertTrue(any("冰美式咖啡" in fact["text"] for fact in pack.facts))
            self.assertTrue(pack.source_spans)
        finally:
            memory.close()

    def test_answer_context_honors_an_explicit_zero_token_budget(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        try:
            memory.add(
                "Father 喜欢喝冰美式咖啡。",
                scope,
                ts("2026-07-06T10:00:00+00:00"),
            )

            pack = memory.answer_context(
                "Father 喝什么饮料？",
                scope,
                budget={"limit": 6, "token_budget": 0},
            )

            self.assertEqual(pack.coverage["token_budget"], 0)
            self.assertEqual(pack.source_spans, [])
            self.assertEqual(pack.current_views, [])
            self.assertEqual(pack.facts, [])
            self.assertEqual(pack.answer_policy, "abstain_if_not_supported")
        finally:
            memory.close()

    def test_chinese_rule_extractor_preserves_named_preference_subject(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")

        result = memory.add("Father 喜欢喝冰美式咖啡。", scope, ts("2026-07-06T10:00:00+00:00"))

        self.assertTrue(result.accepted_fact_ids)
        facts = memory.store.list_facts(scope)
        self.assertTrue(
            any(
                fact.category == "preference"
                and fact.subject == "Father"
                and "冰美式咖啡" in fact.object
                for fact in facts
            )
        )

    def test_current_views_remain_session_scoped_for_same_subject(self) -> None:
        memory = MemoryService()
        session_one = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s1")
        session_two = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s2")
        try:
            memory.add("I prefer Qdrant for Atlas retrieval.", session_one, ts("2026-07-06T10:00:00+00:00"))
            memory.add("I prefer Pinecone for Atlas retrieval.", session_two, ts("2026-07-06T11:00:00+00:00"))

            views = memory.get_current_views(session_one)

            self.assertTrue(any("Qdrant" in view.text for view in views))
            self.assertFalse(any("Pinecone" in view.text for view in views))
        finally:
            memory.close()

    def test_chinese_rule_extractor_accepts_switch_decision_and_event(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")

        result = memory.add("我决定把 Atlas 检索从 Qdrant 切换到 pgvector。", scope, ts("2026-06-30T10:00:00+00:00"))

        self.assertTrue(result.accepted_fact_ids)
        self.assertTrue(result.accepted_event_ids)
        facts = memory.store.list_facts(scope)
        events = memory.store.list_events(scope)
        self.assertTrue(any(fact.predicate == "switched_to" and "pgvector" in fact.object for fact in facts))
        self.assertTrue(any(event.event_type in {"decision", "preference_change", "state_change"} for event in events))

    def test_chinese_rule_extractor_accepts_instruction_and_activity_event(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")

        instruction = memory.add("以后默认用中文回复我。", scope, ts("2026-06-30T10:00:00+00:00"))
        activity = memory.add("昨天我完成了 Render 部署并修复了端口配置问题。", scope, ts("2026-06-30T11:00:00+00:00"))

        self.assertTrue(instruction.accepted_fact_ids)
        self.assertTrue(activity.accepted_event_ids)
        facts = memory.store.list_facts(scope)
        events = memory.store.list_events(scope)
        self.assertTrue(any(fact.category == "instruction" and "中文" in fact.object for fact in facts))
        self.assertTrue(any("Render" in event.description and "部署" in event.description for event in events))

    def test_chinese_resolved_problem_activity_does_not_create_concern_noise(self) -> None:
        facets = [facet for facet, _label, _snippet in extract_generic_event_facets("昨天我完成了 Render 部署并修复了端口配置问题。")]

        self.assertIn("activity", facets)
        self.assertNotIn("concern", facets)

    def test_event_ordering_runtime_flags_do_not_expose_dual_shadow_switch(self) -> None:
        service = MemoryService()
        try:
            self.assertFalse(hasattr(service.retrieval_flags, "dual_event_ordering_shadow"))
            self.assertEqual(getattr(service.retrieval_flags, "production_selector", "legacy"), "legacy")
        finally:
            service.close()

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

        views = memory.get_current_views(scope)
        self.assertTrue(views)
        self.assertTrue(any("Qdrant" in view.text for view in views))

        pack = memory.answer_context("What do I currently prefer for Atlas?", scope)
        self.assertTrue(pack.current_views)
        self.assertTrue(pack.facts)
        self.assertTrue(any("Qdrant" in view["text"] for view in pack.current_views))
        self.assertTrue(any("Qdrant" in fact["text"] for fact in pack.facts))
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
        self.assertTrue(
            any(
                memory.store.has_event_edge(left.event_id, right.event_id)
                for left in events
                for right in events
                if left.event_id != right.event_id
            )
        )

        pack = memory.answer_context("Which happened before dense retrieval?", scope)
        self.assertEqual(pack.coverage["intent"], "chronology")
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

    def test_server_background_worker_skips_disabled_task_types_without_starving_summaries(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.store.enqueue_background_task(scope, "llm_extract", payload={"source_span_ids": ["span_missing"]}, dedupe_key="llm:first")

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

        processed = memory.process_server_background_tasks(limit=1)

        self.assertEqual(processed["processed_count"], 1)
        self.assertEqual(processed["status_counts"], {"succeeded": 1})
        self.assertEqual(processed["tasks"][0]["task_type"], "refresh_session_summary")
        self.assertEqual(len(memory.list_background_tasks(scope, status="pending", allow_cross_session=True)), 1)

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
        self.assertTrue(any("product_entity" in candidate.source for candidate in result.candidates))

    def test_current_value_query_prioritizes_latest_correction_over_historical_value(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add(
            {"role": "user", "content": "For Project Atlas, I initially prefer Qdrant for retrieval experiments."},
            scope,
            ts("2026-01-01T10:00:00+00:00"),
        )
        current_add = memory.add(
            {
                "role": "user",
                "content": (
                    "I switched Project Atlas retrieval from Qdrant to "
                    "Postgres pgvector for production."
                ),
            },
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

        contents = [span["content"] for span in pack.source_spans]
        current_index = next(
            index for index, content in enumerate(contents) if "Postgres pgvector" in content
        )
        history_index = next(
            index
            for index, content in enumerate(contents)
            if "initially prefer Qdrant" in content
        )
        self.assertLess(current_index, history_index)

        current_view = next(
            view for view in pack.current_views if "Postgres pgvector" in view["text"]
        )
        self.assertTrue(set(current_add.span_ids) <= set(current_view["source_span_ids"]))

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
            budget={"allow_cross_session": True, "limit": 6, "mode": "balanced"},
        )

        evidence = "\n".join(span["content"] for span in pack.source_spans)
        self.assertIn("数据库还没启动", evidence)
        self.assertIn("端口被占用", evidence)

    def test_search_preserves_source_provenance_and_sanitized_trace(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        try:
            added = memory.add(
                "Atlas stores the private cobalt-key in Qdrant.",
                scope,
                ts("2026-06-01T10:00:00+00:00"),
            )

            result = memory.search("Atlas cobalt-key", scope, options={"mode": "balanced"})

            self.assertTrue(result.candidates)
            self.assertTrue(
                any(set(added.span_ids) & set(candidate.source_span_ids) for candidate in result.candidates)
            )
            trace = memory.debug_trace(result.trace_id)
            self.assertEqual(trace["stages"], ["plan", "recall", "fusion", "selection"])
            self.assertNotIn("cobalt-key", repr(trace))
        finally:
            memory.close()

    def test_query_intent_runtime_uses_canonical_current_state_intent(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        try:
            memory.add(
                "Atlas currently uses Postgres pgvector for retrieval.",
                scope,
                ts("2026-06-01T10:00:00+00:00"),
            )

            result = memory.search(
                "What retrieval backend does Atlas currently use?",
                scope,
                options={"mode": "balanced"},
            )

            trace = memory.debug_trace(result.trace_id)
            self.assertEqual(trace["intent"], "current_state")
            self.assertTrue(any("Postgres pgvector" in candidate.text for candidate in result.candidates))
        finally:
            memory.close()


if __name__ == "__main__":
    unittest.main()
