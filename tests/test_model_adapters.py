from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tempfile
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fusion_memory import MemoryService, Scope
from fusion_memory.cli import _build_eval_models, _eval_model_config_from_file
from fusion_memory.core.config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from fusion_memory.core.embedding import DeterministicEmbedder, HTTPEmbeddingClient, Qwen3EmbeddingClient
from fusion_memory.core.llm import OpenAICompatibleLLMClient, _extract_structured_response, sanitize_error_text
from fusion_memory.core.runtime_config import memory_service_from_env
from fusion_memory.core.models import Candidate, EvidencePack
from fusion_memory.eval.adapter import BenchmarkAdapter, EvalDocument, EvalQuery
from fusion_memory.eval.model_adapters import OpenAICompatibleAnswerModel, OpenAICompatibleJudgeModel
from fusion_memory.eval.model_adapters import _event_ordering_compact_aspect_label
from fusion_memory.eval.model_adapters import _event_ordering_sequence_output_sort_key
from fusion_memory.eval.model_adapters import _event_ordering_sequence_label
from fusion_memory.eval.model_adapters import _deterministic_temporal_answer
from fusion_memory.eval.model_adapters import _pack_for_model
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor
from fusion_memory.retrieval.evidence_pack import _event_ordering_chronology_rescue_score
from fusion_memory.retrieval.exact_answer_operators import exact_answer_operator_fields
from fusion_memory.retrieval.reranker import HTTPReranker, Qwen3Reranker, rerank_candidates


class ModelAdapterTests(unittest.TestCase):
    def test_rerank_candidates_normalizes_large_raw_scores(self) -> None:
        class FixedScaleReranker:
            def score(self, query: str, docs: list[str]) -> list[float]:
                return [-12.0, -3.0, 9.0]

        candidates = [
            Candidate(id="a", type="span", text="a", source="test", scores={"utility_score": 0.60}, source_span_ids=["a"]),
            Candidate(id="b", type="span", text="b", source="test", scores={"utility_score": 0.62}, source_span_ids=["b"]),
            Candidate(id="c", type="span", text="c", source="test", scores={"utility_score": 0.58}, source_span_ids=["c"]),
        ]

        reranked = rerank_candidates("query", candidates, FixedScaleReranker())

        normalized = [candidate.scores["rerank_score_normalized"] for candidate in reranked]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in normalized))
        self.assertGreaterEqual(reranked[0].scores["utility_score"], reranked[-1].scores["utility_score"])

    def test_rubric_score_retries_with_longer_timeout(self) -> None:
        class FlakyJudgeClient:
            def __init__(self) -> None:
                self.timeout_seconds = 15.0
                self.calls: list[float] = []

            def structured(self, prompt, schema, input):
                self.calls.append(self.timeout_seconds)
                if len(self.calls) < 3:
                    raise ValueError("LLM endpoint did not return a structured JSON object")
                return {"score": 1.0, "reason": "retry recovered"}

        judge = OpenAICompatibleJudgeModel(FlakyJudgeClient())
        score, reason = judge.rubric_score("question", "answer", "rubric item")

        self.assertEqual(score, 1.0)
        self.assertEqual(reason, "retry recovered")
        self.assertEqual(judge.client.calls, [15.0, 180.0, 300.0])

    def test_openai_compatible_llm_client_feeds_structured_extractor(self) -> None:
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/llm"), model="test-llm")
            extractor = StructuredLLMExtractor(client)
            memory = MemoryService(extractor=extractor)
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            result = memory.add("Please remember reports should use PostgreSQL.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))

            self.assertTrue(result.accepted_fact_ids)
            trace = memory.debug_trace(result.trace_id)
            self.assertIn("structured_llm_extractor", str(trace))
            self.assertTrue(any(call["component"] == "extractor_client" for call in trace["model_calls"]))
            llm_call = next(call for call in trace["model_calls"] if call["component"] == "extractor_client")
            self.assertEqual(llm_call["model"], "test-llm")
            self.assertEqual(llm_call["prompt_version"], "llm-extractor-v0")
            self.assertEqual(llm_call["usage"]["total_tokens"], 42)
            self.assertEqual(server.requests[-1]["path"], "/llm")
            self.assertEqual(server.requests[-1]["json"]["model"], "test-llm")
            self.assertTrue(client.calls)
            self.assertIn("latency_ms", client.calls[0])

    def test_openai_compatible_llm_client_reads_reasoning_content_json(self) -> None:
        response = _extract_structured_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": '{"answer": "ok"}',
                            "provider_specific_fields": {"reasoning_content": '{"answer": "fallback"}'},
                        }
                    }
                ]
            }
        )

        self.assertEqual(response, {"answer": "ok"})

    def test_llm_error_sanitizer_redacts_token_fragments(self) -> None:
        text = "Authentication Error. Received API Key = sk-...jVKT, token=abcdef1234567890, Bearer secret-token-value"

        sanitized = sanitize_error_text(text)

        self.assertNotIn("jVKT", sanitized)
        self.assertNotIn("abcdef1234567890", sanitized)
        self.assertNotIn("secret-token-value", sanitized)
        self.assertIn("sk-...", sanitized)

    def test_openai_compatible_llm_client_retries_http_429(self) -> None:
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(
                server.url("/rate-limit-once"),
                model="test-llm",
                retry_attempts=2,
                retry_backoff_seconds=0.0,
                min_interval_seconds=0.0,
            )

            response = client.structured(
                prompt="answer",
                schema={"type": "object"},
                input={"evidence_pack": {"source_spans": [{"content": "Atlas uses Qdrant."}]}},
            )

            self.assertEqual(response["answer"], "Qdrant")
            self.assertEqual([request["path"] for request in server.requests].count("/rate-limit-once"), 2)
            self.assertEqual(client.calls[0]["attempts"], 2)

    def test_openai_compatible_llm_client_falls_back_to_stream_when_non_stream_is_empty(self) -> None:
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/empty-then-stream"), model="test-llm")

            response = client.structured(
                prompt="answer",
                schema={"type": "object"},
                input={"question": "2+2"},
            )

        self.assertEqual(response, {"answer": "4"})
        requests = [request for request in server.requests if request["path"] == "/empty-then-stream"]
        self.assertEqual([request["json"].get("stream") for request in requests], [False, True])

    def test_answer_model_can_use_strict_llm_aggregation_items(self) -> None:
        class AggregationClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                if prompt.startswith("llm-aggregation-v0"):
                    return {
                        "items": [
                            {
                                "key": "feature:offline_mode",
                                "label": "offline mode",
                                "value": 1,
                                "included": True,
                                "count_role": "additive_item",
                                "memory_object_type": "user_intent_item",
                                "source_span_id": "s1",
                                "confidence": 0.92,
                            }
                        ]
                    }
                return {"answer": "1"}

        client = AggregationClient()
        pack = EvidencePack(
            query="How many different features did I mention across my app conversations?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "speaker": "user",
                    "content": "I want offline mode in the app.",
                    "aggregation_keys": ["feature:app"],
                }
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        answer = OpenAICompatibleAnswerModel(client, use_llm_aggregation=True).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="multi_session_reasoning",
        )

        self.assertEqual(answer, "1")
        self.assertEqual(client.calls[0]["prompt"].splitlines()[0], "llm-aggregation-v0")
        evidence_pack = client.calls[-1]["input"]["evidence_pack"]
        self.assertEqual(evidence_pack["aggregation_items"][0]["key"], "feature:offline_mode")
        self.assertFalse(evidence_pack["aggregation_telemetry"]["fallback"])
        self.assertEqual(evidence_pack["aggregation_telemetry"]["accepted_count"], 1)

    def test_event_ordering_pack_keeps_legacy_path_while_graph_shadow_is_missing(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I set up the initial project schema and local server.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
        memory.add("I implemented transaction CRUD with validation errors.", scope, datetime(2026, 6, 2, tzinfo=timezone.utc))

        pack = memory.answer_context(
            "Can you walk me through the order in which I brought up different aspects of my app development and deployment across our conversations?",
            scope,
            budget={"limit": 6, "mode": "benchmark"},
        )

        model_pack = _pack_for_model(pack)

        self.assertTrue(pack.coverage["query_type"] == "event_ordering")
        self.assertTrue(model_pack.get("timeline"))
        self.assertTrue(pack.coverage.get("event_ordering_graph") or pack.coverage.get("event_ordering_shadow"))

    def test_answer_model_falls_back_to_rule_aggregation_when_llm_fails(self) -> None:
        class FailingAggregationClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                if prompt.startswith("llm-aggregation-v0"):
                    raise RuntimeError("aggregation unavailable")
                return {"answer": "fallback"}

        client = FailingAggregationClient()
        pack = EvidencePack(
            query="How many total days did I take off or breaks to manage stress and prevent burnout across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "speaker": "user",
                    "content": "I took a one-hour yoga break because I was stressed and needed focus.",
                    "aggregation_keys": ["break:one_hour_stress_day"],
                }
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        answer = OpenAICompatibleAnswerModel(client, use_llm_aggregation=True).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="multi_session_reasoning",
        )

        evidence_pack = client.calls[-1]["input"]["evidence_pack"]
        self.assertEqual(answer, "fallback")
        self.assertEqual(evidence_pack["aggregation_items"][0]["key"], "break:one_hour_stress_day")
        self.assertTrue(evidence_pack["aggregation_telemetry"]["fallback"])
        self.assertEqual(evidence_pack["aggregation_telemetry"]["reason"], "llm_call_failed")

    def test_http_embedding_client_can_back_store_embeddings(self) -> None:
        with FakeModelServer() as server:
            embedder = HTTPEmbeddingClient(server.url("/embed"), model="test-embed")
            memory = MemoryService(embedder=embedder)
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            memory.add("I prefer Qdrant for Atlas retrieval.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
            result = memory.search("Atlas Qdrant", scope)

            self.assertTrue(result.candidates)
            trace = memory.debug_trace(result.trace_id)
            self.assertTrue(any(call["component"] == "embedder" and call["model"] == "test-embed" for call in trace["model_calls"]))
            audit = memory.audit_events(scope, event_type="memory.search")
            self.assertGreater(audit[0]["payload"]["model_calls"]["count"], 0)
            self.assertTrue(embedder.calls)
            self.assertTrue(any(request["path"] == "/embed" for request in server.requests))

    def test_http_embedding_client_can_send_dashscope_dimensions(self) -> None:
        with FakeModelServer() as server:
            embedder = HTTPEmbeddingClient(
                server.url("/embed"),
                model="text-embedding-v4",
                dimensions=1024,
                encoding_format="float",
            )

            embeddings = embedder.embed_texts(["中文检索", "dense retrieval"])

        self.assertEqual(len(embeddings), 2)
        request_payload = server.requests[-1]["json"]
        self.assertEqual(request_payload["model"], "text-embedding-v4")
        self.assertEqual(request_payload["input"], ["中文检索", "dense retrieval"])
        self.assertEqual(request_payload["dimensions"], 1024)
        self.assertEqual(request_payload["encoding_format"], "float")

    def test_http_reranker_is_used_in_balanced_mode(self) -> None:
        with FakeModelServer() as server:
            reranker = HTTPReranker(server.url("/rerank"), model="test-rerank")
            memory = MemoryService(reranker=reranker)
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            memory.add("I tested BM25 yesterday.", scope, datetime(2026, 6, 3, tzinfo=timezone.utc))
            memory.add("I added dense retrieval today.", scope, datetime(2026, 6, 5, tzinfo=timezone.utc))
            result = memory.search("dense retrieval", scope, options={"mode": "balanced"})

            trace = memory.debug_trace(result.trace_id)
            self.assertEqual(trace["rerank"]["model_version"], "http-reranker:test-rerank")
            self.assertTrue(any(call["component"] == "reranker" and call["model"] == "test-rerank" for call in trace["model_calls"]))
            self.assertTrue(reranker.calls)
            self.assertTrue(any(request["path"] == "/rerank" for request in server.requests))

    def test_http_reranker_accepts_dashscope_results_shape(self) -> None:
        with FakeModelServer() as server:
            reranker = HTTPReranker(server.url("/dashscope-rerank"), model="qwen3-rerank")

            scores = reranker.score("dense retrieval", ["dense retrieval notes", "unrelated billing notes"])

        self.assertEqual(scores, [0.91, 0.12])
        request_payload = server.requests[-1]["json"]
        self.assertEqual(request_payload["model"], "qwen3-rerank")
        self.assertEqual(request_payload["query"], "dense retrieval")
        self.assertEqual(request_payload["documents"], ["dense retrieval notes", "unrelated billing notes"])

    def test_http_reranker_sends_dashscope_options_and_accepts_output_results_shape(self) -> None:
        with FakeModelServer() as server:
            reranker = HTTPReranker(
                server.url("/dashscope-output-rerank"),
                model="qwen3-rerank",
                top_n=2,
                instruct="Given a web search query, retrieve relevant passages that answer the query.",
            )

            scores = reranker.score("dense retrieval", ["dense retrieval notes", "unrelated billing notes", "third note"])

        self.assertEqual(scores, [0.91, 0.12, 0.0])
        request_payload = server.requests[-1]["json"]
        self.assertEqual(request_payload["top_n"], 2)
        self.assertEqual(
            request_payload["instruct"],
            "Given a web search query, retrieve relevant passages that answer the query.",
        )

    def test_runtime_config_wires_http_model_adapters_from_env(self) -> None:
        with FakeModelServer() as server:
            env = {
                "FUSION_MEMORY_EMBEDDING_PROVIDER": "http",
                "FUSION_MEMORY_EMBEDDING_ENDPOINT": server.url("/embed"),
                "FUSION_MEMORY_EMBEDDING_MODEL": "env-embed",
                "FUSION_MEMORY_EMBEDDING_DIMENSION": "1024",
                "FUSION_MEMORY_RERANKER_PROVIDER": "http",
                "FUSION_MEMORY_RERANKER_ENDPOINT": server.url("/rerank"),
                "FUSION_MEMORY_RERANKER_MODEL": "env-rerank",
                "FUSION_MEMORY_RERANKER_TOP_N": "2",
                "FUSION_MEMORY_RERANKER_INSTRUCT": "Retrieve relevant memory snippets.",
                "FUSION_MEMORY_EXTRACTOR_MODE": "async",
                "FUSION_MEMORY_EXTRACTOR_ENDPOINT": server.url("/llm"),
                "FUSION_MEMORY_EXTRACTOR_MODEL": "env-extractor",
            }
            with patch.dict(os.environ, env, clear=True):
                memory = memory_service_from_env(":memory:")
                scope = Scope(workspace_id="w", user_id="u", agent_id="a")
                memory.add("Please remember reports should use PostgreSQL.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
                self.assertFalse(any(request["path"] == "/llm" for request in server.requests))
                memory.process_background_tasks(scope, limit=5)
                result = memory.search("PostgreSQL reports", scope, options={"mode": "balanced"})
                memory.close()

            self.assertTrue(result.candidates)
            paths = [request["path"] for request in server.requests]
            self.assertIn("/llm", paths)
            self.assertIn("/embed", paths)
            self.assertIn("/rerank", paths)
            self.assertTrue(any(request["json"].get("model") == "env-embed" for request in server.requests))
            self.assertTrue(any(request["json"].get("dimensions") == 1024 for request in server.requests if request["path"] == "/embed"))
            self.assertTrue(any(request["json"].get("model") == "env-rerank" for request in server.requests))
            self.assertTrue(any(request["json"].get("top_n") == 2 for request in server.requests if request["path"] == "/rerank"))
            self.assertTrue(
                any(request["json"].get("instruct") == "Retrieve relevant memory snippets." for request in server.requests if request["path"] == "/rerank")
            )
            self.assertTrue(any(request["json"].get("model") == "env-extractor" for request in server.requests))

    def test_runtime_config_requires_explicit_extractor_mode(self) -> None:
        with FakeModelServer() as server:
            env = {
                "FUSION_MEMORY_EXTRACTOR_BASE_URL": server.url(""),
                "FUSION_MEMORY_EXTRACTOR_MODEL": "env-extractor",
            }
            with patch.dict(os.environ, env, clear=True):
                memory = memory_service_from_env(":memory:")
                scope = Scope(workspace_id="w", user_id="u", agent_id="a")
                memory.add("Please remember reports should use PostgreSQL.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
                memory.close()

            self.assertFalse(any(request["path"] == "/chat/completions" for request in server.requests))

    def test_runtime_config_ignores_sync_extractor_mode(self) -> None:
        with FakeModelServer() as server:
            env = {
                "FUSION_MEMORY_EXTRACTOR_MODE": "sync",
                "FUSION_MEMORY_EXTRACTOR_BASE_URL": server.url(""),
                "FUSION_MEMORY_EXTRACTOR_MODEL": "env-extractor",
            }
            with patch.dict(os.environ, env, clear=True):
                memory = memory_service_from_env(":memory:")
                scope = Scope(workspace_id="w", user_id="u", agent_id="a")
                memory.add("Please remember reports should use PostgreSQL.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
                memory.close()

            self.assertFalse(any(request["path"] == "/chat/completions" for request in server.requests))

    def test_runtime_config_query_intent_refiner_is_off_by_default(self) -> None:
        with FakeModelServer() as server:
            env = {
                "FUSION_MEMORY_QUERY_INTENT_BASE_URL": server.url(""),
                "FUSION_MEMORY_QUERY_INTENT_MODEL": "env-router",
            }
            with patch.dict(os.environ, env, clear=True):
                memory = memory_service_from_env(":memory:")
                scope = Scope(workspace_id="w", user_id="u", agent_id="a")
                memory.search("我之前提过哪些权限控制和登录保护能力？", scope)
                memory.close()

            self.assertFalse(any(request["path"] == "/chat/completions" for request in server.requests))

    def test_runtime_config_query_intent_refiner_requires_explicit_mode(self) -> None:
        with FakeModelServer() as server:
            env = {
                "FUSION_MEMORY_QUERY_INTENT_MODE": "always",
                "FUSION_MEMORY_QUERY_INTENT_BASE_URL": server.url(""),
                "FUSION_MEMORY_QUERY_INTENT_MODEL": "env-router",
            }
            with patch.dict(os.environ, env, clear=True):
                memory = memory_service_from_env(":memory:")
                scope = Scope(workspace_id="w", user_id="u", agent_id="a")
                memory.search("我之前提过哪些权限控制和登录保护能力？", scope)
                memory.close()

            self.assertTrue(any(request["path"] == "/chat/completions" for request in server.requests))

    def test_eval_model_builder_wires_optional_llm_aggregation(self) -> None:
        args = SimpleNamespace(
            model_config_file=None,
            answer_endpoint="http://127.0.0.1:9/answer",
            answer_model="answer",
            answer_api_key=None,
            judge_endpoint=None,
            judge_model=None,
            judge_api_key=None,
            model_api_key=None,
            model_timeout_seconds=None,
            use_llm_aggregation=None,
            llm_aggregation_min_confidence=None,
        )

        with patch.dict(os.environ, {}, clear=True):
            answer_model, _judge_model = _build_eval_models(args)
            self.assertFalse(answer_model.use_llm_aggregation)
            self.assertEqual(answer_model.llm_aggregation_min_confidence, 0.70)

        with patch.dict(
            os.environ,
            {
                "FUSION_MEMORY_EVAL_USE_LLM_AGGREGATION": "true",
                "FUSION_MEMORY_EVAL_LLM_AGGREGATION_MIN_CONFIDENCE": "0.82",
            },
            clear=True,
        ):
            answer_model, _judge_model = _build_eval_models(args)
            self.assertTrue(answer_model.use_llm_aggregation)
            self.assertEqual(answer_model.llm_aggregation_min_confidence, 0.82)

        args.use_llm_aggregation = True
        args.llm_aggregation_min_confidence = 0.91
        with patch.dict(os.environ, {}, clear=True):
            answer_model, _judge_model = _build_eval_models(args)
            self.assertTrue(answer_model.use_llm_aggregation)
            self.assertEqual(answer_model.llm_aggregation_min_confidence, 0.91)

    def test_eval_model_config_file_accepts_loose_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "key.txt"
            config_path.write_text(
                '"sk-live-token"\n'
                'base_url = "https://example.test/v1"\n'
                "model_use = gpt5.4 or gpt5.5\n",
                encoding="utf-8",
            )

            config = _eval_model_config_from_file(str(config_path))

        self.assertEqual(config["api_key"], "sk-live-token")
        self.assertEqual(config["endpoint"], "https://example.test/v1/chat/completions")
        self.assertEqual(config["model"], "gpt-5.4")

    def test_eval_model_config_file_accepts_jsonish_key_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "key.txt"
            config_path.write_text(
                '"OPENAI_API_KEY": "sk-json-token"\n'
                'base_url = "https://example.test/v1"\n'
                "model_use = gpt5.4 or gpt5.5\n",
                encoding="utf-8",
            )

            config = _eval_model_config_from_file(str(config_path))

        self.assertEqual(config["api_key"], "sk-json-token")
        self.assertEqual(config["endpoint"], "https://example.test/v1/chat/completions")
        self.assertEqual(config["model"], "gpt-5.4")

    def test_eval_model_builder_uses_model_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "key.txt"
            config_path.write_text(
                '"sk-live-token"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                "model_use = gpt5.4 or gpt5.5\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model_config_file=str(config_path),
                answer_endpoint=None,
                answer_model=None,
                answer_api_key=None,
                judge_endpoint=None,
                judge_model=None,
                judge_api_key=None,
                model_api_key=None,
                model_timeout_seconds=None,
                use_llm_aggregation=None,
                llm_aggregation_min_confidence=None,
            )

            with patch.dict(os.environ, {}, clear=True):
                answer_model, judge_model = _build_eval_models(args)

        self.assertIsNotNone(answer_model)
        self.assertIsNotNone(judge_model)
        self.assertEqual(answer_model.client.endpoint, "http://127.0.0.1:9/v1/chat/completions")
        self.assertEqual(answer_model.client.api_key, "sk-live-token")
        self.assertEqual(answer_model.client.model, "gpt-5.4")

    def test_qwen_defaults_are_configured_without_required_runtime_dependency(self) -> None:
        self.assertEqual(len(DeterministicEmbedder().embed_text("Atlas")), DEFAULT_EMBEDDING_DIMENSION)
        self.assertTrue(DEFAULT_EMBEDDING_MODEL.endswith("models/Qwen3-Embedding-0.6B"))
        self.assertTrue(DEFAULT_RERANKER_MODEL.endswith("models/Qwen3-Reranker-0.6B"))
        real_import = __import__

        def block_sentence_transformers(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=block_sentence_transformers):
            with self.assertRaisesRegex(RuntimeError, "Qwen3EmbeddingClient requires optional ML dependencies"):
                Qwen3EmbeddingClient()
            with self.assertRaisesRegex(RuntimeError, "Qwen3Reranker requires optional ML dependencies"):
                Qwen3Reranker()

    def test_qwen_runtime_defaults_follow_fusion_memory_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["FUSION_MEMORY_HOME"] = tmp
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "from fusion_memory.core.config import "
                        "DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL; "
                        "print(json.dumps([DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL]))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
            )

        embedding_model, reranker_model = json.loads(output)
        self.assertEqual(
            embedding_model,
            str(Path(tmp) / "models" / "Qwen3-Embedding-0.6B"),
        )
        self.assertEqual(
            reranker_model,
            str(Path(tmp) / "models" / "Qwen3-Reranker-0.6B"),
        )

    def test_eval_answer_and_judge_models_are_pluggable(self) -> None:
        with FakeModelServer() as server:
            answer_client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            judge_client = OpenAICompatibleLLMClient(server.url("/judge"), model="judge-model")
            service = MemoryService()
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            adapter = BenchmarkAdapter(
                service,
                scope,
                answer_model=OpenAICompatibleAnswerModel(answer_client),
                judge_model=OpenAICompatibleJudgeModel(judge_client),
            )
            adapter.ingest_documents(
                [EvalDocument(id="doc1", content="Atlas retrieval uses Qdrant.", timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc))]
            )

            results = adapter.run_queries([EvalQuery(id="q1", query="What does Atlas retrieval use?", gold_answers=["Qdrant"])])
            report = adapter.report(results)

            self.assertEqual(results[0].answer, "Qdrant")
            self.assertTrue(results[0].matched_gold)
            self.assertEqual(results[0].llm_calls, 2)
            self.assertIn("answer-model", results[0].answer_model)
            self.assertIn("judge-model", results[0].judge_model)
            self.assertEqual(report["llm_calls_query"], 2.0)
            self.assertTrue(any(request["path"] == "/answer" for request in server.requests))
            self.assertTrue(any(request["path"] == "/judge" for request in server.requests))

    def test_eval_answer_model_adds_beam_category_instructions(self) -> None:
        pack = EvidencePack(
            query="Could you show me how to implement a login feature?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[{"id": "s1", "content": "Always format code snippets with syntax highlighting."}],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            answer = OpenAICompatibleAnswerModel(client).answer_with_context(
                pack.query,
                pack,
                benchmark="BEAM",
                category="instruction_following",
                metadata={"rubric": ["code blocks with syntax highlighting"]},
            )

        self.assertEqual(answer, "Qdrant")
        request_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        self.assertNotIn("benchmark", request_input)
        self.assertNotIn("category", request_input)
        self.assertNotIn("rubric", request_input)
        self.assertIn("fenced code blocks", request_input["instruction"])

    def test_eval_answer_model_surfaces_client_failure(self) -> None:
        class FailingClient:
            version = "failing"

            def structured(self, prompt, schema, input):
                raise RuntimeError("LLM endpoint returned HTTP 429: rate limited")

        pack = EvidencePack(
            query="Question",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[{"id": "s1", "content": "Evidence."}],
            conflicts=[],
            coverage={},
            debug_trace=[],
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            OpenAICompatibleAnswerModel(FailingClient()).answer_with_context("Question", pack)

    def test_eval_answer_model_does_not_send_beam_gold_metadata(self) -> None:
        pack = EvidencePack(
            query="What happened first?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[
                {
                    "id": "e1",
                    "timeline_index": 1,
                    "time_start": "2026-06-01T10:00:00+00:00",
                    "description": "Core functionality was planned.",
                    "event_type": "milestone",
                    "milestone_group": "core_functionality",
                    "source_span_ids": ["s1"],
                }
            ],
            source_spans=[
                {
                    "id": "s1",
                    "timeline_index": 1,
                    "timestamp": "2026-06-01T10:00:00+00:00",
                    "content": "The team planned core functionality before deployment.",
                }
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )
        gold_metadata = {
            "rubric": ["LLM response should contain: Core functionality"],
            "ideal_response": "Core functionality came first.",
            "source_chat_ids": ["chat-1"],
            "ordering_tested": ["1st: Core functionality", "2nd: Security and deployment"],
            "conversation_references": ["hidden-source"],
        }
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(
                pack.query,
                pack,
                benchmark="BEAM",
                category="event_ordering",
                metadata=gold_metadata,
            )

        request_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        serialized_input = json.dumps(request_input)
        self.assertIn("timeline_index", request_input["instruction"])
        self.assertIn("conversation chronology", request_input["instruction"])
        self.assertIn("timeline", request_input["evidence_pack"])
        self.assertNotIn("source_spans", request_input["evidence_pack"])
        self.assertNotIn("events", request_input["evidence_pack"])
        timeline = request_input["evidence_pack"]["timeline"]
        self.assertEqual(timeline[0]["timeline_index"], 1)
        self.assertEqual(timeline[0]["kind"], "event")
        self.assertEqual(timeline[0]["milestone_group"], "core_functionality")
        self.assertEqual(timeline[0]["source_span_ids"], ["s1"])
        self.assertNotIn("time_start", serialized_input)
        self.assertNotIn("timestamp", serialized_input)
        for key in gold_metadata:
            self.assertNotIn(key, request_input)
            self.assertNotIn(key, serialized_input)

    def test_eval_answer_model_prefers_event_ordering_coverage_anchors_over_events(self) -> None:
        pack = EvidencePack(
            query="What order did I bring up aspects?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[
                {
                    "id": "e1",
                    "timeline_index": 1,
                    "description": "A broad implementation event.",
                    "event_type": "milestone",
                    "source_span_ids": ["s3"],
                }
            ],
            source_spans=[
                {
                    "id": "s1#aspect1",
                    "original_span_id": "s1",
                    "timeline_index": 1,
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Core functionality: login and analytics",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "Core functionality: login and analytics.",
                },
                {
                    "id": "s1#aspect2",
                    "original_span_id": "s1",
                    "timeline_index": 2,
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Database schema: users and transactions",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "Database schema: users and transactions.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(
                pack.query,
                pack,
                benchmark="BEAM",
                category="event_ordering",
            )

        request_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        timeline = request_input["evidence_pack"]["timeline"]
        self.assertEqual([item["kind"] for item in timeline], ["user_introduced_aspect", "user_introduced_aspect"])
        self.assertEqual(timeline[0]["label"], "Core functionality: login and analytics")
        self.assertEqual(timeline[1]["label"], "Database schema: users and transactions")

    def test_eval_answer_model_adds_event_ordering_sequence_items_for_budget_tracker(self) -> None:
        pack = EvidencePack(
            query="Can you list the order in which I brought up different aspects of developing my personal budget tracker throughout our conversations, in order? Mention ONLY and ONLY three items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "timeline_index": 1,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "timeline_label": "Core Functionality",
                    "content": "Can you help me implement the core functionality of my budget tracker, including user authentication and expense tracking?",
                },
                {
                    "id": "s2",
                    "timeline_index": 2,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "timeline_label": "Transaction Error Handling",
                    "content": "I'm trying to fix this KeyError: 'amount' in my transaction POST handler with schema validation.",
                },
                {
                    "id": "s3",
                    "timeline_index": 3,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "timeline_label": "Security And Deployment",
                    "content": "I need security hardening before the public launch and deployment.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(pack.query, pack, benchmark="BEAM", category="event_ordering")

        request_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        sequence_items = request_input["evidence_pack"]["sequence_items"]
        self.assertIn("sequence_items", request_input["instruction"])
        self.assertEqual(
            [item["label"].lower() for item in sequence_items],
            ["core functionality", "transaction error handling", "security and deployment"],
        )

    def test_eval_answer_model_adds_event_ordering_sequence_items_for_weather_errors(self) -> None:
        pack = EvidencePack(
            query="Can you list the order in which I brought up different aspects of handling errors and promise rejections in my weather app code throughout our conversations in order? Mention ONLY and ONLY five items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "s1", "timeline_index": 1, "timeline_role": "user_aspect_anchor", "selector": "event_ordering_coverage", "content": "I want user-friendly messages for HTTP 404 and 400 invalid city errors."},
                {"id": "s2", "timeline_index": 2, "timeline_role": "user_aspect_anchor", "selector": "event_ordering_coverage", "content": "Can you implement a try-catch block around the OpenWeather API call?"},
                {"id": "s3", "timeline_index": 3, "timeline_role": "user_aspect_anchor", "selector": "event_ordering_coverage", "content": "I'm getting an Unhandled Promise Rejection warning in fetchWeatherData()."},
                {"id": "s4", "timeline_index": 4, "timeline_role": "user_aspect_anchor", "selector": "event_ordering_coverage", "content": "I need invalid city name error handling improvements with friendly messages."},
                {"id": "s5", "timeline_index": 5, "timeline_role": "user_aspect_anchor", "selector": "event_ordering_coverage", "content": "Please refine fetchWeatherData() for better error handling, promise chaining, and UX."},
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(pack.query, pack, benchmark="BEAM", category="event_ordering")

        sequence_items = _decode_model_payload(server.requests[-1]["json"])["input"]["evidence_pack"]["sequence_items"]
        self.assertEqual(len(sequence_items), 5)
        self.assertEqual([item["timeline_index"] for item in sequence_items], [1, 2, 3, 4, 5])
        labels = [item["label"].lower() for item in sequence_items]
        self.assertIn("user-friendly", labels[0])
        self.assertIn("try-catch", labels[1])
        self.assertIn("unhandled promise rejection", labels[2])
        self.assertIn("invalid city", labels[3])
        self.assertIn("ux", labels[4])

    def test_event_ordering_model_pack_adds_referenceable_episodes(self) -> None:
        pack = EvidencePack(
            query="Can you list the order in which I brought up different aspects of integrating and customizing the framework in my projects across our conversations, in order? Mention ONLY and ONLY three items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "timeline_index": 1,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I am trying to integrate Bootstrap 5.3.0 CDN into my portfolio website for responsive grid components like navbar and cards.",
                },
                {
                    "id": "s2",
                    "timeline_index": 2,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I want to integrate form-control and btn-primary styling classes along with custom CSS for consistent styling and hover effects.",
                },
                {
                    "id": "s3",
                    "timeline_index": 3,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I am trying to fix a known modal accessibility bug by upgrading from Bootstrap v5.3.0 to v5.3.1 while keeping custom modal functionality intact.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        episodes = model_pack["referenceable_episodes"]
        episode_text = json.dumps(episodes)
        graph_timeline = model_pack.get("graph_timeline")

        self.assertEqual([item["timeline_index"] for item in episodes], [1, 2, 3])
        self.assertIn("5.3.0", episode_text)
        self.assertIn("form-control", episode_text)
        self.assertIn("btn-primary", episode_text)
        self.assertIn("custom CSS", episode_text)
        self.assertIn("modal accessibility", episode_text)
        self.assertIsInstance(graph_timeline, list)
        self.assertEqual([item["timeline_index"] for item in graph_timeline], [1, 2, 3])
        self.assertTrue(all(item.get("kind") == "graph_candidate" for item in graph_timeline))
        self.assertEqual([item["timeline_index"] for item in model_pack["timeline"]], [1, 2, 3])

    def test_event_ordering_referenceable_episodes_keep_support_option_details(self) -> None:
        pack = EvidencePack(
            query="Can you list the order in which I brought up different strategies and support options for managing my workload throughout our conversations in order? Mention ONLY and ONLY five items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "mentor",
                    "timeline_index": 1,
                    "speaker": "user",
                    "candidate_source": "event_ordering_episode_recall",
                    "content": "I've got a weekly call with a veteran producer and I asked her for advice on how to manage my schedule better.",
                },
                {
                    "id": "assistant",
                    "timeline_index": 2,
                    "speaker": "user",
                    "candidate_source": "event_ordering_episode_recall+l0_raw_hybrid",
                    "content": "I'm thinking of asking my new part-time assistant who I hired for 20 hours/week at $25/hour after a mentor recommended hiring one to help me manage my schedule better.",
                },
                {
                    "id": "audience",
                    "timeline_index": 3,
                    "speaker": "user",
                    "candidate_source": "event_ordering_episode_recall",
                    "content": "I had a review meeting and she suggested focusing on audience engagement strategies while balancing marketing prep.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        episode_text = json.dumps(model_pack["referenceable_episodes"])

        self.assertIn("part-time assistant", episode_text)
        self.assertIn("20 hours/week", episode_text)
        self.assertIn("$25/hour", episode_text)

    def test_eval_answer_model_keeps_component_episode_before_adjacent_topic_drift(self) -> None:
        pack = EvidencePack(
            query="Can you list the order in which I brought up different aspects of implementing the city autocomplete feature across our conversations, in order? Mention ONLY and ONLY five items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "autocomplete",
                    "timeline_index": 1,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I'm trying to implement city autocomplete using OpenWeather's Geocoding API v1 with a 300ms debounce delay.",
                },
                {
                    "id": "response-time",
                    "timeline_index": 2,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "What about handling cases where the API response time exceeds 300ms?",
                },
                {
                    "id": "rapid-typing",
                    "timeline_index": 3,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "What if the user types quickly and the debounce delay is not enough?",
                },
                {
                    "id": "responsive-ui",
                    "timeline_index": 4,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I'm trying to implement a responsive design for my weather app using CSS Grid and Flexbox.",
                },
                {
                    "id": "invalid-city",
                    "timeline_index": 5,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I'm trying to handle errors for invalid city names with user-friendly HTTP 404 and 400 messages.",
                },
                {
                    "id": "deployment",
                    "timeline_index": 6,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I need to set up a custom domain in GitHub Pages for the weather app.",
                },
                {
                    "id": "rate-limit",
                    "timeline_index": 7,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I'm trying to handle the OpenWeather API rate limit and API calls with a simple counter per minute and per day.",
                },
                {
                    "id": "frontend-stack",
                    "timeline_index": 8,
                    "timeline_role": "user_aspect_anchor",
                    "selector": "event_ordering_coverage",
                    "speaker": "user",
                    "content": "I'm deciding between pure JavaScript and React 18.2 for my frontend.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        sequence_items = _pack_for_model(pack)["sequence_items"]
        labels = [item["label"].lower() for item in sequence_items]

        self.assertEqual([item["timeline_index"] for item in sequence_items], [1, 2, 3, 5, 7])
        self.assertIn("city autocomplete", labels[0])
        self.assertIn("api response", labels[1])
        self.assertIn("debounce", labels[2])
        self.assertTrue(all("responsive" not in label for label in labels))
        self.assertTrue(all("frontend" not in label for label in labels))

    def test_event_ordering_compact_label_uses_context_when_request_shell_is_underspecified(self) -> None:
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "understand how this will affect my financial planning",
                "I started using the YNAB app on Sept 2, and I synced it with my bank accounts.",
            ),
            "YNAB bank accounts sync",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "make sure I'm making the right decision",
                "I'm considering automating my savings to avoid manual transfers and reduce the temptation to spend.",
            ),
            "savings automation",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "minimize these fees or should I consider other payment methods",
                "I've been using PayPal for my freelance payments and the fees are averaging 3% per transaction.",
            ),
            "PayPal fees",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "understand how this feature affects the overall",
                "I have researched that the Dunk Low leather upper is full-grain, which improves durability by 25%, can you help me understand how this feature affects the overall sneaker lifespan?",
            ),
            "Dunk Low leather upper durability",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "protect my new Nike Dunk Low from",
                "What is the best way to protect my new Nike Dunk Low from rain, considering the 30% chance of rain at the festival?",
            ),
            "Nike Dunk Low rain protection",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "does the sneaker protector spray need to",
                "Does the sneaker protector spray need to be reapplied regularly, or is one application enough?",
            ),
            "sneaker protector spray reapplication",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "404 Not Found error",
                'I am having trouble with a "404 Not Found" error on my favicon.ico file, and I think the path in index.html is wrong.',
            ),
            "favicon 404 Not Found fix",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "how can I highlight",
                "I am kinda curious, how can I highlight my hands-on problem-solving skills in a cover letter?",
            ),
            "cover letter hands-on problem-solving skills highlight",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "my age, I'm 65, and I don't",
                "I am kinda worried about my age, I am 65, and I do not know if that is going to be a problem in the competitive job market, can you help me with this?",
            ),
            "age competitive job market concern",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "at Montserrat Film Festival in 2004, for",
                "I am thinking of reaching out to my close friend Leslie, who I met at Montserrat Film Festival in 2004, for some advice on networking at Caribbean media events.",
            ),
            "Leslie networking at Caribbean media events advice",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "I'm thinking of reaching out to my close friend Leslie, who I met at Montserrat Film Festival in 2004, for some advice on networking at Cari",
                "I'm thinking of reaching out to my close friend Leslie, who I met at Montserrat Film Festival in 2004, for some advice on networking at Caribbean Creative Hub, since she's been a great mentor to me for 20 years.",
            ),
            "Leslie networking at Caribbean Creative Hub advice",
        )
        self.assertEqual(
            _event_ordering_compact_aspect_label(
                "my portfolio, Greg told me to update",
                "I am kinda worried about my portfolio, Greg told me to update it by April 1, what should I do to make it stand out?",
            ),
            "portfolio update",
        )

    def test_eval_answer_model_sends_temporal_role_annotations(self) -> None:
        pack = EvidencePack(
            query="How many weeks are between completion and deployment?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "content": "Features complete on January 15, 2024.",
                    "temporal_mentions": [
                        {
                            "text": "January 15, 2024",
                            "normalized_date": "2024-01-15",
                            "role": "feature_finish_date",
                            "role_confidence": 0.88,
                            "context": "Features complete on January 15, 2024.",
                        }
                    ],
                    "temporal_roles": ["feature_finish_date"],
                }
            ],
            conflicts=[],
            coverage={"query_type": "temporal_lookup"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(
                pack.query,
                pack,
                benchmark="BEAM",
                category="temporal_reasoning",
            )

        request_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        span = request_input["evidence_pack"]["source_spans"][0]
        self.assertEqual(span["temporal_roles"], ["feature_finish_date"])
        self.assertEqual(span["temporal_mentions"][0]["normalized_date"], "2024-01-15")

    def test_eval_answer_model_guides_multi_session_aggregation(self) -> None:
        pack = EvidencePack(
            query="How many total ways did I mention across my questions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "speaker": "user",
                    "content": "I mentioned arranging 3 objects with 3! equals 6 ways and 4C2 / 52C2 = 6/1326 for cards.",
                    "aggregation_keys": ["ways:arrange_objects"],
                    "aggregation_signal": 0.9,
                }
            ],
            conflicts=[],
            coverage={"query_type": "temporal_lookup"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(
                pack.query,
                pack,
                benchmark="BEAM",
                category="multi_session_reasoning",
                metadata={"rubric": ["hidden"]},
            )

        request_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        self.assertIn("multi-session reasoning", request_input["instruction"])
        self.assertIn("Do not sum every number", request_input["instruction"])
        self.assertIn("aggregation_keys", request_input["evidence_pack"]["source_spans"][0])
        aggregation_items = request_input["evidence_pack"]["aggregation_items"]
        self.assertTrue(any(item["included"] and item["value"] == 6 for item in aggregation_items))
        self.assertTrue(any(not item["included"] and item["value"] == 1326 for item in aggregation_items))
        self.assertNotIn("rubric", json.dumps(request_input))

    def test_eval_answer_model_filters_combinatorics_items_to_requested_domains(self) -> None:
        pack = EvidencePack(
            query="How many total ways did I mention for arranging or choosing balls and cards across my questions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "balls-user",
                    "speaker": "user",
                    "history_index": 1,
                    "content": (
                        "I asked about 3 different colored balls: 3! equals 6 ways to arrange them, "
                        "and 3C2 equals 3 ways to choose 2 balls."
                    ),
                    "aggregation_keys": ["ways:arrange_objects", "ways:choose_balls"],
                },
                {
                    "id": "balls-assistant",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": "The number of ways to arrange 3 balls is 3! = 6, and choosing 2 balls is C(3,2) = 3.",
                    "aggregation_keys": ["ways:arrange_objects", "ways:choose_balls"],
                },
                {
                    "id": "cards-user",
                    "speaker": "user",
                    "history_index": 3,
                    "content": "For cards, I used combinations like 4C2 / 52C2 = 6/1326 for drawing 2 aces.",
                    "aggregation_keys": ["ways:choose_cards"],
                },
                {
                    "id": "unrelated",
                    "speaker": "user",
                    "history_index": 4,
                    "content": "Separately, I practiced 5C3 = 10 with generic objects.",
                    "aggregation_keys": ["ways:choose_objects"],
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        aggregation_items = model_pack["aggregation_items"]
        included = [item for item in aggregation_items if item["included"]]
        excluded = {item["key"]: item.get("reason") for item in aggregation_items if not item["included"]}

        self.assertEqual(model_pack["aggregation_summary"]["by_count_role"]["additive_value"]["value_sum"], 15)
        self.assertEqual({(item["key"], item["value"]) for item in included}, {("ways:arrange_balls", 6), ("ways:choose_balls", 3), ("ways:choose_aces_cards", 6)})
        self.assertIn("ways:choose_objects", excluded)
        self.assertEqual(excluded["ways:choose_objects"], "outside_requested_combinatorics_domain")

    def test_eval_answer_model_probability_calculations_suppress_generic_requests(self) -> None:
        pack = EvidencePack(
            query="In my questions about tossing coins and rolling dice, how many different probability calculations did I try to confirm?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "coin-die",
                    "speaker": "user",
                    "content": "Can you help me confirm probability calculations for getting heads = 1/2 and rolling a 4 = 1/6?",
                    "aggregation_keys": ["request:confirm_probability_calculations"],
                },
                {
                    "id": "die-greater",
                    "speaker": "user",
                    "content": "I want to verify if rolling a number greater than 4 is 2/6 = 1/3.",
                    "aggregation_keys": ["calculation:die_greater_than_4"],
                },
                {
                    "id": "example",
                    "speaker": "user",
                    "content": "I also mentioned both heads = 1/4 as background, not something I asked to confirm.",
                    "aggregation_keys": ["request:background"],
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        included = {item["key"] for item in model_pack["aggregation_items"] if item["included"]}
        excluded = {item["key"]: item.get("reason") for item in model_pack["aggregation_items"] if not item["included"]}

        self.assertEqual(included, {"calculation:coin_heads", "calculation:die_roll_4", "calculation:die_greater_than_4"})
        self.assertNotIn("request:confirm_probability_calculations", included)
        self.assertEqual(excluded["calculation:two_coin_both_heads"], "educational_example_not_confirmed")

    def test_eval_answer_model_uses_generic_aggregation_keys(self) -> None:
        pack = EvidencePack(
            query="How many different planning areas did I mention across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "resume",
                    "speaker": "user",
                    "content": "I focused on adapting my resume to international standards.",
                    "aggregation_keys": ["area:resume_international_standards"],
                    "aggregation_signal": 0.7,
                },
                {
                    "id": "portfolio",
                    "speaker": "user",
                    "content": "I also wanted to improve my portfolio project selection.",
                    "aggregation_keys": ["area:portfolio_project_selection"],
                    "aggregation_signal": 0.7,
                },
                {
                    "id": "portfolio-dupe",
                    "speaker": "assistant",
                    "content": "Portfolio project selection remains one area.",
                    "aggregation_keys": ["area:portfolio_project_selection"],
                    "aggregation_signal": 0.6,
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(
                pack.query,
                pack,
                benchmark="BEAM",
                category="multi_session_reasoning",
            )

        request_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        aggregation_items = request_input["evidence_pack"]["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}
        self.assertEqual(
            included,
            {
                "area:resume_international_standards": "resume international standards",
                "area:portfolio_project_selection": "portfolio project selection",
            },
        )

    def test_eval_answer_model_filters_generic_items_for_area_queries(self) -> None:
        pack = EvidencePack(
            query="How many different planning areas did I mention across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "one",
                    "speaker": "user",
                    "content": "I focused on adapting my resume to international standards.",
                    "aggregation_keys": ["area:resume_international_standards", "item:resume_keyword_density"],
                },
                {
                    "id": "two",
                    "speaker": "user",
                    "content": "I also wanted to improve my portfolio project selection.",
                    "aggregation_keys": ["area:portfolio_project_selection", "item:highlight_award_projects"],
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(pack.query, pack, benchmark="BEAM", category="multi_session_reasoning")

        aggregation_items = _decode_model_payload(server.requests[-1]["json"])["input"]["evidence_pack"]["aggregation_items"]
        included_keys = {item["key"] for item in aggregation_items if item["included"]}
        self.assertEqual(included_keys, {"area:resume_international_standards", "area:portfolio_project_selection"})

    def test_eval_answer_model_extracts_generic_quoted_list_and_action_items(self) -> None:
        pack = EvidencePack(
            query="How many unique titles, checklist items, and selected options did I mention across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "quoted",
                    "speaker": "user",
                    "content": 'I planned "Alpha Roadmap" and "Beta Launch" for the next review.',
                },
                {
                    "id": "bullets",
                    "speaker": "user",
                    "content": "My checklist now includes:\n- stakeholder notes\n- migration dry run",
                },
                {
                    "id": "action",
                    "speaker": "user",
                    "content": "I also selected the blue deployment lane after comparing options.",
                },
                {
                    "id": "dupe",
                    "speaker": "assistant",
                    "content": 'You mentioned "Alpha Roadmap" again while summarizing.',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(pack.query, pack, benchmark="BEAM", category="multi_session_reasoning")

        aggregation_items = _decode_model_payload(server.requests[-1]["json"])["input"]["evidence_pack"]["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}
        self.assertIn("title:alpha_roadmap", included)
        self.assertIn("title:beta_launch", included)
        self.assertNotIn("title:stakeholder_notes", included)
        self.assertNotIn("title:migration_dry_run", included)
        self.assertNotIn("title:blue_deployment_lane", included)
        self.assertEqual(sum(1 for item in aggregation_items if item["key"] == "title:alpha_roadmap" and item["included"]), 1)

    def test_eval_answer_model_allows_series_and_genre_items_together(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres did I mention across conversations?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "series", "speaker": "user", "content": 'I mentioned series like "The Expanse" and "The Broken Earth".'},
                {"id": "genre", "speaker": "user", "content": "I also mentioned genres including fantasy and space opera."},
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}

        self.assertIn("title:the_expanse", included)
        self.assertIn("title:the_broken_earth", included)
        self.assertTrue(any(key.startswith("genre:") for key in included))

    def test_eval_answer_model_uses_recommendation_group_count_without_expanding_assistant_titles(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres did I mention across conversations?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "request",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "Can you recommend three fiction series that fit my library budget?",
                },
                {
                    "id": "assistant-list",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": 'Here are three good fits:\n1. "The Broken Earth"\n2. "The Expanse"\n3. "The Murderbot Diaries"',
                },
                {
                    "id": "later",
                    "speaker": "user",
                    "history_index": 8,
                    "content": 'I later mentioned that "The Vorkosigan Saga" sounded like a good science fiction series.',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        aggregation_items = model_pack["aggregation_items"]
        summary = model_pack["aggregation_summary"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        group_hints = [item for key, item in included.items() if key.startswith("group_count:series:")]
        self.assertEqual(len(group_hints), 1)
        self.assertEqual(group_hints[0]["value"], 3)
        self.assertEqual(group_hints[0]["memory_object_type"], "assistant_recommendation_group")
        self.assertEqual(group_hints[0]["count_role"], "candidate_group_count")
        self.assertIn("title:the_vorkosigan_saga", included)
        self.assertEqual(included["title:the_vorkosigan_saga"]["memory_object_type"], "user_intent_item")
        self.assertEqual(included["title:the_vorkosigan_saga"]["count_role"], "additive_item")
        self.assertNotIn("title:the_broken_earth", included)
        self.assertNotIn("title:the_expanse", included)
        self.assertNotIn("title:the_murderbot_diaries", included)
        self.assertEqual(summary["by_count_role"]["candidate_group_count"]["value_sum"], 3)
        self.assertEqual(summary["by_count_role"]["additive_item"]["value_sum"], 2)
        self.assertIn("science fiction", summary["by_count_role"]["additive_item"]["labels"])
        self.assertEqual(group_hints[0]["label"], "3 recommended series")
        self.assertEqual(
            [item["count_role"] for item in summary["primary_count_candidates"][:2]],
            ["additive_item", "candidate_group_count"],
        )
        formulas = {item["formula"]: item for item in model_pack["aggregation_answer_candidates"]}
        self.assertNotIn("distinct_union_count", formulas)
        self.assertIn("dedupe_mixed_count_roles", formulas)

    def test_eval_answer_model_infers_recommendation_group_count_from_headings(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres have I mentioned wanting to explore?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "request",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "Can you suggest some must-read fiction series that fit my budget?",
                },
                {
                    "id": "assistant-list",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": 'Here are a few suggestions:\n\n### "The Kingkiller Chronicle"\nDetails...\n\n### "The Mistborn Trilogy"\nDetails...\n\n### "The Lies of Locke Lamora"\nDetails...',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}
        group_hints = [item for key, item in included.items() if key.startswith("group_count:series:")]

        self.assertEqual(len(group_hints), 1)
        self.assertEqual(group_hints[0]["value"], 3)
        self.assertNotIn("title:the_kingkiller_chronicle", included)
        self.assertNotIn("title:the_mistborn_trilogy", included)

    def test_eval_answer_model_filters_exploratory_title_items_by_intent(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres have I mentioned wanting to explore?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "explore",
                    "speaker": "user",
                    "content": '"The Lies of Locke Lamora" sounds really interesting because it blends fantasy and historical fiction.',
                },
                {
                    "id": "completed",
                    "speaker": "user",
                    "content": 'I finished "The Expanse" and already completed the first three books.',
                },
                {
                    "id": "list",
                    "speaker": "user",
                    "content": 'My current reading list includes "The Stormlight Archive" and "The Wheel of Time".',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}
        excluded = {item["key"]: item.get("reason") for item in aggregation_items if not item["included"]}

        self.assertIn("title:the_lies_of_locke_lamora", included)
        self.assertIn("genre:fantasy", included)
        self.assertIn("genre:historical_fiction", included)
        self.assertEqual(excluded["title:the_expanse"], "not_exploratory_or_already_completed")
        self.assertEqual(excluded["title:the_stormlight_archive"], "not_exploratory_or_already_completed")

    def test_eval_answer_model_excludes_purchase_review_from_exploratory_titles(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres have I mentioned wanting to explore?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "purchase-review",
                    "speaker": "user",
                    "content": 'I debated the financial decision of spending $18 on "Leviathan Wakes" and wondered if it was worth it after exceeding my budget.',
                },
                {
                    "id": "explore",
                    "speaker": "user",
                    "content": '"The Lies of Locke Lamora" sounds really interesting, and I want to explore it next.',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}
        excluded = {item["key"]: item for item in aggregation_items if not item["included"]}

        self.assertIn("title:the_lies_of_locke_lamora", included)
        self.assertNotIn("title:leviathan_wakes", included)
        self.assertEqual(excluded["title:leviathan_wakes"]["reason"], "not_exploratory_purchase_or_budget_review")
        self.assertEqual(excluded["title:leviathan_wakes"]["count_role"], "excluded")

    def test_eval_answer_model_keeps_genre_interest_separate_from_completed_titles(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres have I mentioned wanting to explore?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "mixed-intent",
                    "speaker": "user",
                    "content": (
                        'I just finished "The Nightingale", but I am also interested in historical fiction '
                        "and looking for a new series that combines fantasy and historical fiction."
                    ),
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}
        excluded = {item["key"]: item.get("reason") for item in aggregation_items if not item["included"]}

        self.assertIn("genre:historical_fiction", included)
        self.assertIn("genre:fantasy", included)
        self.assertEqual(excluded["title:the_nightingale"], "not_exploratory_or_already_completed")

    def test_eval_answer_model_uses_assistant_genre_echo_without_expanding_recommendations(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres have I mentioned wanting to explore?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "assistant-echo",
                    "speaker": "assistant",
                    "content": (
                        'Exploring sci-fi subgenres to better appreciate "The Expanse" is a great idea. '
                        'Here are some series you might enjoy: "Dune", "Hyperion", and "The Culture".'
                    ),
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertIn("genre:sci_fi", included)
        self.assertEqual(included["genre:sci_fi"]["memory_object_type"], "assistant_supported_item")
        self.assertNotIn("title:dune", included)
        self.assertNotIn("title:hyperion", included)

    def test_eval_answer_model_summarizes_additive_aggregation_items(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 6-7?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "plan", "speaker": "user", "content": 'For April 6-7, I planned "Soul", "Coco", and "Moana".'},
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        summary = model_pack["aggregation_summary"]

        self.assertEqual(summary["included_count"], 3)
        self.assertEqual(summary["by_count_role"]["additive_item"]["value_sum"], 3)
        self.assertEqual(summary["primary_count_candidates"][0]["count_role"], "additive_item")

    def test_pack_for_model_filters_only_weak_generic_aggregation_items(self) -> None:
        pack = EvidencePack(
            query="How many different AI vendors or tools have I mentioned using or customizing for hiring automation?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "generic",
                    "speaker": "user",
                    "content": "I'm thinking of automating the hiring process and need advice on whether that is a good idea.",
                    "aggregation_keys": ["generic:hiring_process", "generic:advice_whether_good_idea"],
                }
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        self.assertNotIn("aggregation_items", model_pack)
        self.assertNotIn("aggregation_summary", model_pack)

    def test_eval_answer_model_extracts_vendor_tool_items(self) -> None:
        pack = EvidencePack(
            query="How many different AI vendors or tools have I mentioned using or customizing for hiring automation?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "signalcraft",
                    "speaker": "user",
                    "content": "I've begun customizing an AI algorithm with vendor SignalCraft since July 1 to reduce age bias.",
                },
                {
                    "id": "screenwise",
                    "speaker": "assistant",
                    "content": "You've already seen a 45% reduction in screening time with ScreenWise, and your current AI tools include ScreenWise and SignalCraft.",
                },
                {
                    "id": "template",
                    "speaker": "assistant",
                    "content": "Research reputable AI vendors such as ScreenWise, SignalCraft, TalentOrbit, etc. before choosing a tool.",
                },
                {
                    "id": "generic",
                    "speaker": "user",
                    "content": "I'm thinking about whether AI tools are a good idea for hiring automation.",
                    "aggregation_keys": ["generic:ai_tools_good_idea"],
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        aggregation_items = model_pack["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}

        self.assertEqual(
            included,
            {
                "vendor_tool:signalcraft": "signalcraft",
                "vendor_tool:screenwise": "screenwise",
            },
        )
        self.assertFalse(any(item["key"] == "generic:ai_tools_good_idea" and item["included"] for item in aggregation_items))
        self.assertNotIn("vendor_tool:talentorbit", included)

    def test_eval_answer_model_uses_exact_candidates_for_vendor_tool_items(self) -> None:
        pack = EvidencePack(
            query="How many different AI vendors or tools have I mentioned using or customizing for hiring automation?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "generic",
                    "speaker": "user",
                    "content": "I'm thinking about whether AI tools are a good idea for hiring automation.",
                    "aggregation_keys": ["generic:ai_tools_good_idea"],
                },
            ],
            conflicts=[],
            coverage={
                "query_type": "multi_session_reasoning",
                "exact_answer_candidates": [
                    {
                        "source_span_id": "exact-tool",
                        "speaker": "user",
                        "content": "I've begun customizing an AI algorithm with vendor SignalCraft since July 1.",
                    }
                ],
            },
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}

        self.assertEqual(included, {"vendor_tool:signalcraft": "signalcraft"})

    def test_eval_answer_model_does_not_count_people_as_vendor_tools(self) -> None:
        pack = EvidencePack(
            query="How many different AI vendors or tools have I mentioned using or customizing for hiring automation?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "person",
                    "speaker": "assistant",
                    "content": "Celebrating the success of your AI pilot with Maya is a good way to acknowledge the achievement.",
                },
                {
                    "id": "tool",
                    "speaker": "assistant",
                    "content": "Your current AI tools include ScreenWise.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}

        self.assertEqual(included, {"vendor_tool:screenwise": "screenwise"})
        self.assertNotIn("vendor_tool:maya", included)

    def test_eval_answer_model_excludes_generic_items_outside_query_date_scope(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 6-7 and April 8?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "in-range", "speaker": "user", "content": 'For April 6-7, I planned "Soul" and "Coco".'},
                {"id": "also-in-range", "speaker": "user", "content": 'For April 8, I added "Paddington 2".'},
                {"id": "out-range", "speaker": "user", "content": 'For April 12, I planned "Arrival".'},
                {"id": "missing-date", "speaker": "user", "content": 'I also planned "Moonlight" for a later movie night.'},
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}
        excluded = {item["key"]: item.get("reason") for item in aggregation_items if not item["included"]}

        self.assertEqual(included, {"title:soul", "title:coco", "title:paddington_2"})
        self.assertEqual(excluded["title:arrival"], "outside_query_date_scope")
        self.assertEqual(excluded["title:moonlight"], "missing_query_date_scope")

    def test_eval_answer_model_includes_adjacent_assistant_titles_for_date_scoped_plan(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 6-7?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "request",
                    "speaker": "user",
                    "history_index": 1,
                    "content": 'For April 6-7, I am planning a family movie marathon and already chose "Soul".',
                },
                {
                    "id": "response",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": 'A good final list would include "Coco", "Moana", and "Paddington 2".',
                },
                {
                    "id": "later",
                    "speaker": "assistant",
                    "history_index": 8,
                    "content": 'For another weekend, consider "Arrival".',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}

        self.assertEqual(included, {"title:soul", "title:coco", "title:moana", "title:paddington_2"})

    def test_eval_answer_model_projects_date_scope_across_nearby_turns(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 8?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "request",
                    "speaker": "user",
                    "turn_id": "session:msg110",
                    "content": 'What movies should we watch on April 8, considering "Coco" and "Paddington 2"?',
                },
                {
                    "id": "choice",
                    "speaker": "user",
                    "turn_id": "session:msg112",
                    "content": 'I think "Moana" and "Zootopia" sound perfect for Michelle and Francis.',
                },
                {
                    "id": "other-date",
                    "speaker": "user",
                    "turn_id": "session:msg116",
                    "content": 'For April 12, I planned "Arrival".',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}
        excluded = {item["key"]: item.get("reason") for item in aggregation_items if not item["included"]}

        self.assertEqual(included, {"title:coco", "title:paddington_2", "title:moana", "title:zootopia"})
        self.assertEqual(excluded["title:arrival"], "outside_query_date_scope")

    def test_eval_answer_model_uses_adjacent_recommendation_group_for_date_scoped_query(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 8?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "planned",
                    "speaker": "user",
                    "turn_id": "session:msg10",
                    "history_index": 10,
                    "content": 'I have 2 movies planned for April 8: "River Quest" and "Garden Bears".',
                },
                {
                    "id": "request",
                    "speaker": "user",
                    "turn_id": "session:msg11",
                    "history_index": 11,
                    "content": "What other movies would you recommend for April 8?",
                },
                {
                    "id": "recommendations",
                    "speaker": "assistant",
                    "turn_id": "session:msg12",
                    "history_index": 12,
                    "candidate_source": "adjacent_exact_answer_support",
                    "content": (
                        'Here are some movie recommendations:\n'
                        '1. **"Sky Boats"**\n'
                        '2. **"Moon Kitchen"**\n'
                        '3. **"City Parade"**'
                    ),
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        included = {item["key"]: item for item in model_pack["aggregation_items"] if item["included"]}

        group_items = [item for key, item in included.items() if key.startswith("group_count:movie:")]
        self.assertEqual(len(group_items), 1)
        self.assertEqual(group_items[0]["value"], 3)
        self.assertEqual(group_items[0]["count_role"], "candidate_group_count")
        self.assertIn("title:river_quest", included)
        self.assertIn("title:garden_bears", included)
        self.assertNotIn("title:sky_boats", included)
        self.assertNotIn("title:moon_kitchen", included)
        formulas = {item["formula"]: item for item in model_pack["aggregation_answer_candidates"]}
        self.assertEqual(formulas["distinct_union_count"]["answer_value"], 5)
        self.assertEqual(
            formulas["distinct_union_count"]["component_values"],
            {"base_unique_count": 2, "candidate_group_count": 3, "explicit_overlap": 0},
        )
        self.assertIn("dedupe_mixed_count_roles", formulas)
        self.assertIsNone(formulas["dedupe_mixed_count_roles"]["answer_value"])
        self.assertEqual(
            formulas["dedupe_mixed_count_roles"]["component_values"],
            {"additive_item": 2, "candidate_group_count": 3},
        )

    def test_eval_answer_model_builds_distinct_union_candidate_for_partial_count_and_recommendations(self) -> None:
        pack = EvidencePack(
            query="How many unique movies have I planned to watch across April 6-7 and April 8?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "partial",
                    "speaker": "user",
                    "turn_id": "session:msg20",
                    "history_index": 20,
                    "content": 'For April 6-7, I finalized 8 movies including "Soul", "Coco", and "Paddington 2".',
                },
                {
                    "id": "request",
                    "speaker": "user",
                    "turn_id": "session:msg21",
                    "history_index": 21,
                    "content": "Can you suggest five movies for April 8?",
                },
                {
                    "id": "recommendations",
                    "speaker": "assistant",
                    "turn_id": "session:msg22",
                    "history_index": 22,
                    "content": (
                        'Here are five movies for April 8:\n'
                        '1. **"Moana"**\n'
                        '2. **"Zootopia"**\n'
                        '3. **"Encanto"**\n'
                        '4. **"Tangled"**\n'
                        '5. **"Turning Red"**'
                    ),
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        candidates = _pack_for_model(pack)["aggregation_answer_candidates"]
        formulas = {item["formula"]: item for item in candidates}

        self.assertEqual(formulas["distinct_union_count"]["answer_value"], 13)
        self.assertEqual(
            formulas["distinct_union_count"]["component_values"],
            {"base_unique_count": 8, "candidate_group_count": 5, "explicit_overlap": 0},
        )

    def test_eval_answer_model_returns_deterministic_distinct_union_count(self) -> None:
        class UnexpectedAnswerClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                return {"answer": "8"}

        pack = EvidencePack(
            query="How many unique movies have I planned to watch across April 6-7 and April 8?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "partial",
                    "speaker": "user",
                    "history_index": 20,
                    "content": 'For April 6-7, I finalized 8 movies including "Soul", "Coco", and "Paddington 2".',
                },
                {
                    "id": "request",
                    "speaker": "user",
                    "history_index": 21,
                    "content": "Can you suggest five movies for April 8?",
                },
                {
                    "id": "recommendations",
                    "speaker": "assistant",
                    "history_index": 22,
                    "content": (
                        'Here are five movies for April 8:\n'
                        '1. **"Moana"**\n'
                        '2. **"Zootopia"**\n'
                        '3. **"Encanto"**\n'
                        '4. **"Tangled"**\n'
                        '5. **"Turning Red"**'
                    ),
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )
        client = UnexpectedAnswerClient()

        answer = OpenAICompatibleAnswerModel(client).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="multi_session_reasoning",
        )

        self.assertIn("13 unique items", answer)
        self.assertIn("base count 8", answer)
        self.assertIn("additional recommendation group 5", answer)
        self.assertEqual(client.calls, [])

    def test_eval_answer_model_does_not_use_union_count_for_synthesis_question(self) -> None:
        class AnswerClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                return {"answer": "Use the subscriptions and rentals according to the budget evidence."}

        pack = EvidencePack(
            query="Considering my subscriptions and snack budget, how can I optimize my monthly entertainment spending?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "partial",
                    "speaker": "user",
                    "history_index": 20,
                    "content": 'For April 6-7, I finalized 8 movies including "Soul", "Coco", and "Paddington 2".',
                },
                {
                    "id": "request",
                    "speaker": "user",
                    "history_index": 21,
                    "content": "Can you suggest five movies for April 8?",
                },
                {
                    "id": "recommendations",
                    "speaker": "assistant",
                    "history_index": 22,
                    "content": 'Here are five movies for April 8: "Moana", "Zootopia", "Encanto", "Tangled", "Turning Red".',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        answer = OpenAICompatibleAnswerModel(AnswerClient()).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="multi_session_reasoning",
        )

        self.assertNotIn("unique items", answer)
        self.assertIn("subscriptions", answer)

    def test_eval_answer_model_uses_direct_assistant_schedule_but_not_strategy_examples(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 8?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "schedule",
                    "speaker": "assistant",
                    "history_index": 1,
                    "content": 'Suggested schedule for April 8: 10:00 AM "Moana", 11:45 AM "Zootopia".',
                },
                {
                    "id": "strategy",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": 'Strategies to manage the schedule: consider "Finding Nemo" and "Paddington 2" as examples.',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}

        self.assertEqual(included, {"title:moana", "title:zootopia"})

    def test_eval_answer_model_does_not_project_single_add_question_to_assistant_titles(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 6?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "request",
                    "speaker": "user",
                    "history_index": 1,
                    "content": 'For April 6, should I add "Klaus" to the watchlist?',
                },
                {
                    "id": "response",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": 'You could also consider "Toy Story 4" and "Finding Nemo".',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}

        self.assertEqual(included, {"title:klaus"})

    def test_eval_answer_model_uses_count_hint_for_partial_title_lists(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 6-7?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "partial",
                    "speaker": "user",
                    "history_index": 1,
                    "content": 'For April 6-7, I finalized 8 movies including "Soul" and "Coco".',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertEqual(len(included), 1)
        hint = next(item for key, item in included.items() if key.startswith("count_hint:movies:"))
        self.assertEqual(hint["value"], 8)

    def test_eval_answer_model_excludes_negated_title_items(self) -> None:
        pack = EvidencePack(
            query="How many unique movies did I plan for April 6-7?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "negative",
                    "speaker": "user",
                    "history_index": 1,
                    "content": 'For April 6-7, I excluded "Joker" because Michelle was uncomfortable with scary scenes.',
                },
                {
                    "id": "positive",
                    "speaker": "user",
                    "history_index": 2,
                    "content": 'For April 6-7, I planned "Soul".',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}
        excluded = {item["key"]: item.get("reason") for item in aggregation_items if not item["included"]}

        self.assertEqual(included, {"title:soul"})
        self.assertEqual(excluded["title:joker"], "title_excluded_or_rejected_in_context")

    def test_eval_answer_model_derives_generic_items_from_first_person_actions(self) -> None:
        pack = EvidencePack(
            query="How many different planning areas did I mention across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "one", "speaker": "user", "content": "I focused on adapting my resume to international standards."},
                {"id": "two", "speaker": "user", "content": "I also wanted to improve my portfolio project selection."},
                {"id": "assistant", "speaker": "assistant", "content": "Portfolio project selection remains one area."},
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            OpenAICompatibleAnswerModel(client).answer_with_context(pack.query, pack, benchmark="BEAM", category="multi_session_reasoning")

        aggregation_items = _decode_model_payload(server.requests[-1]["json"])["input"]["evidence_pack"]["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}
        self.assertEqual(
            included,
            {
                "area:resume_international_standards": "resume international standards",
                "area:portfolio_project_selection": "portfolio project selection",
            },
        )

    def test_eval_answer_model_truncates_generic_item_at_new_first_person_clause(self) -> None:
        pack = EvidencePack(
            query="How many different areas have I focused on updating or improving?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "portfolio",
                    "speaker": "user",
                    "content": "I'm updating my portfolio and I noticed the number of mentees I've worked with has increased to 7.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}

        self.assertEqual(included, {"area:portfolio": "portfolio"})
        self.assertNotIn("area:portfolio_noticed_number_mentees_worked", included)

    def test_eval_answer_model_extracts_planning_system_items_for_reminder_queries(self) -> None:
        pack = EvidencePack(
            query="How many different types of reminders or plans have I mentioned using to manage my tasks and family events?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "todoist",
                    "speaker": "assistant",
                    "content": "Using Todoist to manage both your daily tasks and weekend plans can help; set reminders before family events.",
                },
                {
                    "id": "calendar",
                    "speaker": "assistant",
                    "content": "Sync your family appointments and school events in Google Calendar so reminders stay visible.",
                },
                {
                    "id": "asana",
                    "speaker": "assistant",
                    "content": "For pilot deadlines, use Asana templates to create planning sessions and share your plans with the team.",
                },
                {
                    "id": "tactic",
                    "speaker": "user",
                    "content": "I implemented batching for emails and calls on Mondays and Fridays.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertEqual(set(included), {"plan_system:todoist", "plan_system:google_calendar", "plan_system:asana"})
        self.assertTrue(all(item["memory_object_type"] == "assistant_supported_item" for item in included.values()))
        self.assertNotIn("generic:batching", included)

    def test_eval_answer_model_extracts_roles_and_security_features_without_generic_project_features(self) -> None:
        pack = EvidencePack(
            query="How many different user roles and security features am I trying to implement across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "mvp",
                    "speaker": "user",
                    "content": (
                        "1. User Authentication - Registration - Login - Logout. "
                        "2. Transaction Management - Add Income - Add Expense - View Transactions. "
                        "Nov 16 - Dec 15: implement user authentication."
                    ),
                    "aggregation_keys": [
                        "feature:user_authentication",
                        "feature:transaction_management",
                        "feature:add_income",
                        "feature:nov_16_dec_15",
                    ],
                },
                {
                    "id": "auth",
                    "speaker": "user",
                    "content": "I need to implement user registration with hashed passwords and session login using Werkzeug.security.",
                },
                {
                    "id": "role",
                    "speaker": "user",
                    "content": "I added a role-based access control stub for future multi-user support, but currently all users have the 'user' role.",
                },
                {
                    "id": "lockout",
                    "speaker": "user",
                    "content": "I'm trying to implement account lockout after 5 failed login attempts using Redis for rate limiting.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertEqual(
            set(included),
            {
                "security_feature:authentication",
                "security_feature:password_hashing",
                "security_feature:session_management",
                "role:user",
                "security_feature:role_based_access_control",
                "security_feature:account_lockout_rate_limiting",
            },
        )
        self.assertFalse(any(key.startswith("feature:") for key in included))
        self.assertTrue(all(item["count_role"] == "additive_item" for item in included.values()))

    def test_eval_answer_model_uses_query_intent_for_multilingual_role_security_aggregation(self) -> None:
        pack = EvidencePack(
            query="我在所有会话里一共提到过几个用户角色和安全功能？",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "auth",
                    "speaker": "user",
                    "content": "I need to implement user registration with hashed passwords and session login using Werkzeug.security.",
                },
                {
                    "id": "role",
                    "speaker": "user",
                    "content": "I added a role-based access control stub, but currently all users have the 'user' role.",
                },
                {
                    "id": "crud",
                    "speaker": "user",
                    "content": "I also implemented transaction CRUD and monthly analytics for the dashboard.",
                    "aggregation_keys": ["feature:transaction_crud", "feature:monthly_analytics"],
                },
            ],
            conflicts=[],
            coverage={
                "query_type": "multi_session_reasoning",
                "query_intent": {
                    "schema_version": "query-intent-v1",
                    "language": "zh",
                    "answer_shape": "count",
                    "evidence_scope": "multi_session",
                    "speaker_scope": "user",
                    "object_types": ["role", "security_feature"],
                    "aggregation": {"operation": "count", "distinct": False},
                },
            },
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        included = {item["key"] for item in model_pack["aggregation_items"] if item["included"]}

        self.assertEqual(model_pack["query_intent"]["language"], "zh")
        self.assertIn("security_feature:password_hashing", included)
        self.assertIn("security_feature:session_management", included)
        self.assertIn("security_feature:role_based_access_control", included)
        self.assertIn("role:user", included)
        self.assertFalse(any(key.startswith("feature:") for key in included))

    def test_eval_answer_model_does_not_treat_generic_plans_as_planning_system_query(self) -> None:
        pack = EvidencePack(
            query=(
                "How many different application types am I planning to use my personal statement for, "
                "and which roles or plans did I mention that might affect my visa application choice?"
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "reminders",
                    "speaker": "assistant",
                    "content": "Use Todoist, Google Calendar, and Asana to manage tasks, events, reminders, and deadlines.",
                },
                {
                    "id": "application",
                    "speaker": "user",
                    "content": "I want to reuse my personal statement for academic, scholarship, and visa applications.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"] for item in aggregation_items if item["included"]}

        self.assertFalse(any(key.startswith("plan_system:") for key in included))

    def test_multi_session_pack_exposes_schema_column_aggregation_items(self) -> None:
        pack = EvidencePack(
            query="How many new columns did I want to add to the transactions table across my requests?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "turn_id": "t1",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "My transactions table currently has id, user_id, type, amount, and date.",
                },
                {
                    "id": "s2",
                    "turn_id": "t2",
                    "speaker": "user",
                    "history_index": 2,
                    "content": "I want to add a category column to the transactions table for reporting.",
                },
                {
                    "id": "s3",
                    "turn_id": "t3",
                    "speaker": "user",
                    "history_index": 3,
                    "content": "I'm trying to implement an Alembic migration script to add a 'notes' TEXT column to the transactions table without downtime.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertEqual(set(included), {"column:category", "column:notes"})
        self.assertTrue(all(item["count_role"] == "additive_item" for item in included.values()))

    def test_eval_answer_model_extracts_application_type_items_for_personal_statement_queries(self) -> None:
        pack = EvidencePack(
            query="How many different application types am I planning to use my personal statement for?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "hypothetical",
                    "speaker": "assistant",
                    "content": "Think about the purpose of your personal statement. Is it for a job application, a grant proposal, or another type of submission?",
                },
                {
                    "id": "multipurpose",
                    "speaker": "assistant",
                    "content": "Creating a multi-purpose personal statement that works for academic, visa, and grant applications can be streamlined.",
                },
                {
                    "id": "scholarship",
                    "speaker": "user",
                    "content": "I need the personal statement ready before the scholarship deadline and the visa application deadline.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertEqual(
            set(included),
            {
                "application_type:academic",
                "application_type:visa",
                "application_type:grant",
                "application_type:scholarship",
            },
        )
        self.assertNotIn("application_type:job", included)

    def test_eval_answer_model_uses_query_scoped_area_focus_items(self) -> None:
        pack = EvidencePack(
            query="How many different areas have I focused on updating or improving based on my messages about my resume, portfolio, and salary negotiation?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "resume-status",
                    "speaker": "user",
                    "content": "I'm kinda worried that my resume won't be ready by April 10, can you help me get started?",
                    "aggregation_keys": ["area:resume_won_ready_april"],
                },
                {
                    "id": "salary",
                    "speaker": "user",
                    "content": "I'm thinking of asking for a $10,000 salary increase based on my new resume and portfolio; can you give advice on how to negotiate that?",
                },
                {
                    "id": "portfolio",
                    "speaker": "user",
                    "content": "I chose to highlight 5 award-winning projects in my portfolio after Alexis advised focusing on storytelling impact.",
                    "aggregation_keys": ["area:highlight_award_winning_projects_portfolio"],
                },
                {
                    "id": "leadership",
                    "speaker": "user",
                    "content": "I'm trying to update my resume to highlight my leadership skills in remote work settings.",
                    "aggregation_keys": ["area:leadership_skills_remote_work_settings", "area:know_effectively_convey_experience"],
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}

        self.assertEqual(
            included,
            {
                "area:resume_update": "resume update",
                "area:salary_negotiation": "salary negotiation",
                "area:portfolio_project_selection": "portfolio project selection",
                "area:remote_leadership_skills": "remote leadership skills",
            },
        )
        self.assertNotIn("area:resume_won_ready_april", included)
        self.assertNotIn("area:know_effectively_convey_experience", included)

    def test_eval_answer_model_groups_app_feature_concerns_without_metadata_noise(self) -> None:
        pack = EvidencePack(
            query="How many different features or concerns did I mention wanting to handle across my weather app conversations?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "responsive",
                    "speaker": "user",
                    "content": "I'm trying to implement a responsive design for my weather app using CSS Grid and Flexbox, targeting mobile devices.",
                    "aggregation_keys": ["feature:grid_container_adapts_different_screen_sizes_orientations"],
                },
                {
                    "id": "invalid",
                    "speaker": "user",
                    "content": "I'm trying to handle errors for invalid city names and display user-friendly messages for HTTP 404 and 400 status codes.",
                    "aggregation_keys": ["feature:display_user_friendly_messages"],
                },
                {
                    "id": "metadata",
                    "speaker": "user",
                    "content": '"name": "weather-app", "version": "1.0.0", "scripts": {"deploy": "gh-pages -d build"}',
                    "aggregation_keys": ["feature:name", "feature:version", "feature:scripts", "feature:deploy"],
                },
                {
                    "id": "quota",
                    "speaker": "user",
                    "content": "I'm trying to handle the API rate limit and improve response time with caching and retries.",
                    "aggregation_keys": ["feature:calls_minute_1000_calls_day_rate_limits"],
                },
                {
                    "id": "autocomplete",
                    "speaker": "user",
                    "content": "I'm working on city autocomplete and weather display behavior for the app.",
                    "aggregation_keys": ["feature:autocomplete_feature"],
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item["label"] for item in aggregation_items if item["included"]}

        self.assertEqual(
            included,
            {
                "feature:responsive_ui": "responsive ui",
                "feature:user_visible_error_handling": "user visible error handling",
                "feature:api_operational_limits": "api operational limits",
                "feature:weather_lookup_interaction": "weather lookup interaction",
            },
        )
        self.assertNotIn("feature:name", included)
        self.assertNotIn("feature:version", included)
        self.assertNotIn("feature:deploy", included)

    def test_eval_answer_model_adds_financial_impact_items(self) -> None:
        pack = EvidencePack(
            query="How will increasing our grocery budget while taking on the freelance contract affect my ability to support Ashlee's medical bills and still meet my savings goals?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "medical-old",
                    "speaker": "user",
                    "history_index": 3,
                    "content": "Ashlee's medical bills are around $200 monthly, and I need to manage this expense without affecting my other financial goals.",
                },
                {
                    "id": "medical-new",
                    "speaker": "user",
                    "history_index": 7,
                    "content": "Ashlee revealed increased medical expenses of $350/month during our May 20 visit.",
                },
                {
                    "id": "contract",
                    "speaker": "assistant",
                    "history_index": 10,
                    "content": "The freelance contract with Natalie is $8,000 over 4 months, which equates to $2,000 per month.",
                },
                {
                    "id": "grocery",
                    "speaker": "user",
                    "history_index": 12,
                    "content": "I've agreed with Alexis on a $500 monthly joint budget for groceries starting Sept 1, which is up from $400.",
                },
                {
                    "id": "savings",
                    "speaker": "assistant",
                    "history_index": 13,
                    "content": "You can still work toward your $2,000 emergency fund goal by June 30 if you keep the monthly savings target visible.",
                },
                {
                    "id": "car-savings",
                    "speaker": "user",
                    "history_index": 15,
                    "content": "I'm trying to save $5,000 for a family car by Dec 31, 2026, and I've increased my monthly savings to $200.",
                },
                {
                    "id": "mixed-expenses",
                    "speaker": "assistant",
                    "history_index": 14,
                    "content": (
                        "This change, along with the potential freelance contract, might affect your planning. "
                        "Current expenses: rent is $1,200/month and utilities are $150/month."
                    ),
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        impacts = model_pack["financial_impacts"]
        by_subject = {(item["subject_key"], item["amount"]): item for item in impacts}

        self.assertEqual(by_subject[("financial:ashlee_medical_bills", "$350")]["period"], "monthly")
        self.assertEqual(by_subject[("financial:ashlee_medical_bills", "$350")]["impact_role"], "expense_obligation")
        self.assertEqual(by_subject[("financial:grocery_budget", "$500")]["impact_role"], "budget_change")
        self.assertEqual(by_subject[("financial:grocery_budget", "$400")]["current_state"], "prior_or_baseline")
        self.assertEqual(by_subject[("financial:freelance_contract", "$2,000")]["direction"], "inflow")
        self.assertEqual(by_subject[("financial:emergency_fund", "$2,000")]["direction"], "target")
        self.assertEqual(by_subject[("financial:savings_goal", "$5,000")]["period"], "unspecified")
        self.assertEqual(by_subject[("financial:savings_goal", "$200")]["period"], "monthly")
        self.assertNotIn(("financial:freelance_contract", "$1,200"), by_subject)
        self.assertNotIn(("financial:freelance_contract", "$150"), by_subject)

        summary = model_pack["financial_summary"]
        self.assertEqual(summary["monthly_inflows"][0]["subject_key"], "financial:freelance_contract")
        self.assertEqual(summary["monthly_inflows"][0]["amount_number"], 2000.0)
        self.assertEqual(summary["monthly_outflows"][0]["subject_key"], "financial:ashlee_medical_bills")
        self.assertEqual(summary["monthly_outflows"][0]["amount_number"], 350.0)
        self.assertEqual(summary["budget_changes"][0]["subject_key"], "financial:grocery_budget")
        self.assertEqual(summary["budget_changes"][0]["delta_number"], 100.0)
        self.assertEqual(summary["monthly_net_after_obligations_and_budget_changes"]["amount_number"], 1550.0)
        self.assertEqual(summary["monthly_net_after_obligations_and_budget_changes"]["interpretation"], "positive")

    def test_eval_answer_model_suppresses_generic_items_for_stress_break_totals(self) -> None:
        pack = EvidencePack(
            query="How many total days did I take off or breaks to manage stress and prevent burnout across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "two-hour",
                    "speaker": "user",
                    "content": "On May 15, I took a two-hour break to reset after a long writing sprint.",
                },
                {
                    "id": "one-hour",
                    "speaker": "user",
                    "content": "I took a one-hour yoga break because I was near burnout and needed focus.",
                },
                {
                    "id": "generic-days",
                    "speaker": "user",
                    "content": "My essay days have been intense, and I mentioned days on the calendar.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertNotIn("break:two_hour_stress_break", included)
        self.assertIn("break:one_hour_stress_day", included)
        self.assertFalse(any(key.startswith("item:") for key in included))

    def test_eval_answer_model_keeps_explicit_stress_break_totals(self) -> None:
        pack = EvidencePack(
            query="How many total days did I take off or breaks to manage stress and prevent burnout across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "two-hour",
                    "speaker": "user",
                    "content": "On May 15, I took a two-hour break because I was stressed and needed focus.",
                },
                {
                    "id": "full-days",
                    "speaker": "user",
                    "content": "I took two full days off on July 13-14 to prevent burnout and maintain focus.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertEqual(included["break:two_hour_stress_break"]["value"], 1)
        self.assertEqual(included["break:full_days_off"]["value"], 2)

    def test_eval_answer_model_ignores_assistant_echo_for_stress_break_totals(self) -> None:
        pack = EvidencePack(
            query="How many total days did I take off or breaks to manage stress and prevent burnout across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "assistant-echo",
                    "speaker": "assistant",
                    "content": "Long breaks like the 2-hour break you took can be effective for resetting and resting your mind.",
                },
                {
                    "id": "one-hour",
                    "speaker": "user",
                    "content": "I took a one-hour yoga break because I was stressed and needed focus.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertNotIn("break:two_hour_stress_break", included)
        self.assertIn("break:one_hour_stress_day", included)

    def test_eval_answer_model_scans_late_keyed_aggregation_spans(self) -> None:
        filler = [
            {
                "id": f"filler-{index}",
                "speaker": "assistant",
                "content": f"General planning note {index} with no concrete stress break item.",
            }
            for index in range(45)
        ]
        pack = EvidencePack(
            query="How many total days did I take off or breaks to manage stress and prevent burnout across my sessions?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                *filler,
                {
                    "id": "late-full-days",
                    "speaker": "user",
                    "content": "I took two full days off on July 13-14 to prevent burnout and maintain focus.",
                    "aggregation_keys": ["break:full_days_off"],
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        aggregation_items = _pack_for_model(pack)["aggregation_items"]
        included = {item["key"]: item for item in aggregation_items if item["included"]}

        self.assertEqual(included["break:full_days_off"]["value"], 2)

    def test_eval_answer_model_extracts_score_improvement_items(self) -> None:
        pack = EvidencePack(
            query="How much did my accuracy improve between the two times I mentioned my scores on area calculation problems and special lines in triangles?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "special-lines",
                    "speaker": "user",
                    "content": "My quiz score improved from 78% to 88% on special lines and area formulas.",
                },
                {
                    "id": "area",
                    "speaker": "user",
                    "content": "My accuracy in area calculation problems improved from 70% to 90% after completing 12 problems.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "temporal_lookup"},
            debug_trace=[],
        )
        model_pack = _pack_for_model(pack)
        aggregation_items = model_pack["aggregation_items"]
        candidates = model_pack["aggregation_answer_candidates"]
        included = [item for item in aggregation_items if item["included"]]
        excluded = [item for item in aggregation_items if not item["included"]]
        self.assertEqual([(item["label"], item["value"]) for item in included], [("70% to 90% area calculation improvement", 20)])
        self.assertTrue(any(item["label"] == "78% to 88% improvement" for item in excluded))
        formulas = {item["formula"]: item for item in candidates}
        self.assertEqual(formulas["delta_between_values"]["answer_value"], 20)
        self.assertEqual(formulas["delta_between_values"]["unit"], "percentage_points")

    def test_eval_answer_model_returns_deterministic_delta_between_values(self) -> None:
        class UnexpectedAnswerClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                return {"answer": "I am unsure."}

        pack = EvidencePack(
            query="How much did my accuracy improve on area calculation problems?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "area",
                    "speaker": "user",
                    "content": "My accuracy in area calculation problems improved from 70% to 90% after completing 12 problems.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        client = UnexpectedAnswerClient()
        answer = OpenAICompatibleAnswerModel(client).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="multi_session_reasoning",
        )

        self.assertEqual(answer, "20 percentage points, from 70% to 90%.")
        self.assertEqual(client.calls, [])

    def test_eval_answer_model_builds_distinct_slot_values_for_shoe_size_updates(self) -> None:
        pack = EvidencePack(
            query="How many different shoe sizes have I mentioned across my messages?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "order",
                    "speaker": "user",
                    "content": "I just placed an order for Ultraboost size 11 on Finish Line's website.",
                },
                {
                    "id": "reorder",
                    "speaker": "user",
                    "content": "I'm returning the Adidas Ultraboost size 11 and I've already reordered size 11.5.",
                },
                {
                    "id": "try-on",
                    "speaker": "user",
                    "content": "I tried the New Balance 990v5 and the size 10.5 was a perfect fit.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        formulas = {item["formula"]: item for item in _pack_for_model(pack)["aggregation_answer_candidates"]}

        self.assertEqual(formulas["distinct_slot_values"]["answer_value"], 2)
        self.assertEqual(formulas["distinct_slot_values"]["labels"], ["size 11", "size 11.5"])

    def test_eval_answer_model_builds_grouped_exploration_count_from_contexts(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres have I mentioned wanting to explore across my conversations?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "store",
                    "speaker": "assistant",
                    "content": (
                        "With a budget of $120 for print editions from Montserrat Books on Main Street, "
                        "Combination 1: \"The Kingkiller Chronicle\" ($30-$40), \"The Mistborn Trilogy\" "
                        "($30-$40), and \"The Broken Empire\" ($30-$40). Total: Approximately $90-$120."
                    ),
                },
                {
                    "id": "chat",
                    "speaker": "user",
                    "content": "I'm invited to co-host a live chat on sci-fi series, so I want to understand the genre.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        formulas = {item["formula"]: item for item in _pack_for_model(pack)["aggregation_answer_candidates"]}

        self.assertEqual(formulas["grouped_distinct_count"]["answer_value"], 4)
        self.assertEqual(len(formulas["grouped_distinct_count"]["labels"]), 2)
        self.assertTrue(any("3 titles" in label for label in formulas["grouped_distinct_count"]["labels"]))
        self.assertTrue(any("1 genre" in label for label in formulas["grouped_distinct_count"]["labels"]))

    def test_eval_answer_model_demotes_recommendation_groups_for_user_mentioned_scope(self) -> None:
        pack = EvidencePack(
            query="How many different book series or genres have I mentioned wanting to explore across my conversations?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "request",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "Can you recommend seven fiction series that blend fantasy and historical elements?",
                },
                {
                    "id": "assistant-list",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": (
                        'Here are seven series: "A", "B", "C", "D", "E", "F", and "G".'
                    ),
                },
                {
                    "id": "mentioned",
                    "speaker": "user",
                    "history_index": 8,
                    "content": 'I mentioned wanting to explore "The Poppy War" and historical fiction next.',
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        formulas = {item["formula"]: item for item in _pack_for_model(pack)["aggregation_answer_candidates"]}

        self.assertLess(formulas["candidate_group_count"]["confidence"], 0.5)
        self.assertNotIn("distinct_union_count", formulas)

    def test_eval_answer_model_returns_deterministic_deadline_pair(self) -> None:
        class UnexpectedAnswerClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                return {"answer": "I am unsure."}

        pack = EvidencePack(
            query="What are the two different patent filing deadlines I need to meet?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "provisional",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "I aim to file a provisional patent by June 1, 2024.",
                },
                {
                    "id": "assistant-plan",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": "Confirm the timeline for filing the provisional patent application by May 2, 2024.",
                },
                {
                    "id": "non-provisional",
                    "speaker": "user",
                    "history_index": 3,
                    "content": "My non-provisional patent filing deadline is set for November 10, 2024.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        client = UnexpectedAnswerClient()
        answer = OpenAICompatibleAnswerModel(client).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="multi_session_reasoning",
        )

        self.assertEqual(
            answer,
            "June 1, 2024 for the provisional patent; November 10, 2024 for the non-provisional patent filing.",
        )
        self.assertEqual(client.calls, [])

    def test_eval_answer_model_builds_grouped_asset_count_without_planning_actions(self) -> None:
        pack = EvidencePack(
            query="How many specific assets or items have I mentioned across my conversations that are part of my estate planning?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "home",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "I started listing my assets on March 1, and my $350,000 home on 45 Coral Bay Rd is a big part of that.",
                },
                {
                    "id": "review",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": (
                        "Assets listed: Home, savings account, film equipment, vehicle, and digital assets. "
                        "Also review executor choice and guardianship separately."
                    ),
                },
                {
                    "id": "action",
                    "speaker": "user",
                    "history_index": 3,
                    "content": "I decided to update my will digitally and finalized Douglas as executor.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        formulas = {item["formula"]: item for item in _pack_for_model(pack)["aggregation_answer_candidates"]}

        self.assertEqual(formulas["grouped_distinct_count"]["answer_value"], 5)
        labels = " ".join(formulas["grouped_distinct_count"]["labels"])
        self.assertIn("home", labels)
        self.assertIn("savings account", labels)
        self.assertNotIn("executor choice", labels)

    def test_eval_answer_model_returns_direct_where_met_extraction(self) -> None:
        fields = exact_answer_operator_fields(
            "Where did I say I met Laura?",
            "Laura met me on set at Blue Horizon Studios in 2019.",
            speaker="user",
        )
        self.assertEqual(fields["answer_value"], "on set at Blue Horizon Studios in 2019")
        pronoun_fields = exact_answer_operator_fields(
            "Where did I say I met Laura?",
            "Laura recommended the mixer, and she met me on set at Blue Horizon Studios in 2019.",
            speaker="user",
        )
        self.assertEqual(pronoun_fields["answer_value"], "on set at Blue Horizon Studios in 2019")

        class UnexpectedAnswerClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                return {"answer": "wrong"}

        pack = EvidencePack(
            query="Where did I say I met Laura?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "factual_exact",
                "exact_answer_candidates": [
                    {
                        "source_span_id": "s1",
                        "speaker": "user",
                        "content": "Laura met me on set at Blue Horizon Studios in 2019.",
                        "answer_type": "location",
                        "answer_value": "on set at Blue Horizon Studios in 2019",
                        "confidence": 0.91,
                        "extraction_formula": "where_met_relation",
                    }
                ],
            },
            debug_trace=[],
        )

        answer = OpenAICompatibleAnswerModel(UnexpectedAnswerClient()).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="information_extraction",
        )

        self.assertEqual(answer, "on set at Blue Horizon Studios in 2019.")

    def test_eval_answer_model_returns_prior_probability_extraction(self) -> None:
        class UnexpectedAnswerClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                return {"answer": "wrong"}

        pack = EvidencePack(
            query="What probability did I mention for drawing a certain card from the deck before we started discussing drawing two cards?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "factual_exact",
                "exact_answer_candidates": [
                    {
                        "source_span_id": "s1",
                        "speaker": "user",
                        "content": "Before drawing two cards, I said the probability of drawing an ace on the first draw was 4/52.",
                        "answer_type": "probability",
                        "answer_value": "4/52",
                        "confidence": 0.88,
                        "extraction_formula": "prior_probability_before_sequence",
                    }
                ],
            },
            debug_trace=[],
        )

        answer = OpenAICompatibleAnswerModel(UnexpectedAnswerClient()).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="information_extraction",
        )

        self.assertEqual(answer, "4/52.")

    def test_eval_answer_model_filters_assistant_plan_systems_for_user_mentioned_scope(self) -> None:
        pack = EvidencePack(
            query="How many different types of reminders or plans have I mentioned using to manage my tasks and family events?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "user-tools",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "My task completion rate increased by 25% since I started using Todoist and syncing it with Google Calendar.",
                },
                {
                    "id": "assistant-template",
                    "speaker": "assistant",
                    "history_index": 2,
                    "content": "You could use Asana templates and the Reminders app to create planning sessions.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        included = [item for item in model_pack["aggregation_items"] if item["included"]]
        excluded = [item for item in model_pack["aggregation_items"] if not item["included"]]

        self.assertEqual([item["label"] for item in included], ["todoist", "google calendar"])
        self.assertTrue(any(item["label"] == "asana" for item in excluded))

    def test_eval_answer_model_does_not_use_deadline_pair_for_synthesis_question(self) -> None:
        class AnswerClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                return {"answer": "Prioritize the cover letter, Zoom call preparation, and interview practice in that order."}

        pack = EvidencePack(
            query="Considering my cover letter deadlines and Zoom call, how should I prioritize preparation?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "cover",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "I need to submit my cover letter by April 14, 2024.",
                },
                {
                    "id": "zoom",
                    "speaker": "user",
                    "history_index": 2,
                    "content": "My Zoom call with the creative director is scheduled for April 15, 2024.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        answer = OpenAICompatibleAnswerModel(AnswerClient()).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="multi_session_reasoning",
        )

        self.assertIn("Prioritize", answer)
        self.assertNotIn("April 14, 2024 for", answer)

    def test_eval_answer_model_ignores_budget_approval_date_for_deadline_pair(self) -> None:
        pack = EvidencePack(
            query="What are the two different filing deadlines I need to meet?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "provisional",
                    "speaker": "user",
                    "history_index": 1,
                    "content": "I aim to file a provisional application by June 1, 2024.",
                },
                {
                    "id": "budget",
                    "speaker": "user",
                    "history_index": 2,
                    "content": "I have $12,000 approved for the non-provisional filing and PCT application by October 15, 2024.",
                },
                {
                    "id": "non-provisional",
                    "speaker": "user",
                    "history_index": 3,
                    "content": "I've got a deadline to meet for my non-provisional filing, which is set for November 10, 2024.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "multi_session_reasoning"},
            debug_trace=[],
        )

        formulas = {item["formula"]: item for item in _pack_for_model(pack)["aggregation_answer_candidates"]}

        self.assertEqual(
            formulas["deadline_pair"]["component_values"],
            {"provisional": "June 1, 2024", "non_provisional": "November 10, 2024"},
        )
        self.assertNotIn("October 15, 2024", " ".join(formulas["deadline_pair"]["labels"]))

    def test_event_ordering_sequence_label_prefers_short_phase_text(self) -> None:
        label = _event_ordering_sequence_label(
            {
                "label": "Here's my current CSS code",
                "text": "Here's my current CSS code: I want to optimize it further, can you help me identify areas where I can remove redundant selectors and consolidate media queries?",
            }
        )
        self.assertRegex(label.lower(), r"remov(?:e|ing) redundant selectors")
        self.assertNotIn("here's my current css code", label.lower())

    def test_event_ordering_sequence_label_extracts_action_from_low_information_shell(self) -> None:
        accessibility = _event_ordering_sequence_label(
            {
                "label": "make sure I'm doing it correctly to avoid any accessibility issues",
                "text": "Can you make sure I'm doing it correctly to avoid any accessibility issues?",
            }
        )
        favicon = _event_ordering_sequence_label(
            {
                "label": "this by providing an example of how to correctly link to the favicon",
                "text": "Could you explain this by providing an example of how to correctly link to the favicon?",
            }
        )

        self.assertIn("accessibility issues", accessibility.lower())
        self.assertNotIn("doing it correctly", accessibility.lower())
        self.assertIn("link to the favicon", favicon.lower())
        self.assertNotIn("providing an example", favicon.lower())

    def test_event_ordering_compact_aspect_label_removes_request_shell(self) -> None:
        label = _event_ordering_compact_aspect_label(
            "I'm trying to implement transaction CRUD endpoints and validation errors, can you help?",
        )

        self.assertIn("transaction CRUD", label)
        self.assertIn("implementation", label.lower())
        self.assertNotIn("trying", label.lower())
        self.assertNotIn("can you", label.lower())

    def test_event_ordering_compact_aspect_label_handles_decisions_and_concerns(self) -> None:
        decision = _event_ordering_compact_aspect_label(
            "I need to decide between a client-side cache and server-side cache for city autocomplete.",
        )
        frontend_decision = _event_ordering_compact_aspect_label(
            "I'm trying to decide between using pure JavaScript or React 18.2 for my frontend, but I chose vanilla JS.",
        )
        concern = _event_ordering_compact_aspect_label(
            "hmm, what if the user types quickly and the debounce delay isn't enough?",
        )

        self.assertIn("city autocomplete decision", decision.lower())
        self.assertNotIn("decide between", decision.lower())
        self.assertIn("javascript", frontend_decision.lower())
        self.assertIn("decision", frontend_decision.lower())
        self.assertNotEqual(frontend_decision.lower(), "decide between")
        self.assertIn("debounce", concern.lower())
        self.assertIn("concern", concern.lower())
        self.assertNotIn("types type", concern.lower())

    def test_event_ordering_compact_aspect_label_removes_how_can_request_shell(self) -> None:
        label = _event_ordering_compact_aspect_label(
            "achieve: How can I use Sass to create a reusable CSS component",
        )

        self.assertIn("Sass", label)
        self.assertIn("reusable CSS", label)
        self.assertNotIn("achieve", label.lower())
        self.assertNotIn("how can", label.lower())

    def test_event_ordering_sequence_items_use_compact_action_labels(self) -> None:
        pack = EvidencePack(
            query="What order did I bring up the project work? Mention ONLY and ONLY three items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "timeline_index": 1,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Can you help me initialize a Flask project with the local server?",
                    "content": "Can you help me initialize a Flask project with the local server?",
                },
                {
                    "id": "s2",
                    "timeline_index": 2,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "I want to implement transaction CRUD endpoints with validation.",
                    "content": "I want to implement transaction CRUD endpoints with validation.",
                },
                {
                    "id": "s3",
                    "timeline_index": 3,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Then I configured deployment settings for the service.",
                    "content": "Then I configured deployment settings for the service.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        labels = [item["label"] for item in _pack_for_model(pack)["sequence_items"]]

        self.assertEqual(len(labels), 3)
        self.assertIn("initialization", labels[0].lower())
        self.assertIn("implementation", labels[1].lower())
        self.assertIn("configuration", labels[2].lower())
        self.assertFalse(any("can you" in label.lower() or "i want" in label.lower() for label in labels))

    def test_event_ordering_sequence_items_focus_specific_feature_when_enough_evidence_exists(self) -> None:
        pack = EvidencePack(
            query=(
                "Can you list the order in which I brought up different aspects of implementing "
                "the city autocomplete feature throughout our conversations? Mention ONLY and ONLY three items."
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "timeline_index": 1,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "city autocomplete setup",
                    "content": "I'm trying to implement city autocomplete using a geocoding API.",
                },
                {
                    "id": "s2",
                    "timeline_index": 2,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "autocomplete dropdown debounce",
                    "content": "I want the autocomplete dropdown to debounce fast typing.",
                },
                {
                    "id": "s3",
                    "timeline_index": 3,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "responsive grid layout",
                    "content": "I need the weather app cards to use a responsive grid layout.",
                },
                {
                    "id": "s4",
                    "timeline_index": 4,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "invalid city message handling",
                    "content": "I need friendly invalid city messages when autocomplete cannot find a match.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        sequence_items = _pack_for_model(pack)["sequence_items"]
        labels = " ".join(item["label"].lower() for item in sequence_items)

        self.assertEqual(len(sequence_items), 3)
        self.assertIn("autocomplete", labels)
        self.assertIn("invalid city", labels)
        self.assertNotIn("responsive grid", labels)

    def test_event_ordering_anchor_timeline_ignores_assistant_plan_as_user_phase(self) -> None:
        pack = EvidencePack(
            query="What order did I bring up the deployment work? Mention ONLY and ONLY two items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "a1",
                    "timeline_index": 1,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "User Authentication",
                    "content": "Sure, let's break it down. Components: 1. User Authentication 2. Transactions. Does this breakdown work for you?",
                },
                {
                    "id": "u1",
                    "timeline_index": 2,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Render deployment setup",
                    "content": "I started setting up Render deployment.",
                },
                {
                    "id": "u2",
                    "timeline_index": 3,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Gunicorn worker configuration",
                    "content": "Then I configured Gunicorn workers for deployment.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        labels = " ".join(item["label"].lower() for item in model_pack["sequence_items"])
        anchor_labels = " ".join(str(item.get("label", "")).lower() for item in model_pack["anchor_timeline"])

        self.assertNotIn("user authentication", labels)
        self.assertNotIn("user authentication", anchor_labels)
        self.assertIn("render deployment", labels)
        self.assertIn("gunicorn", labels)

    def test_event_ordering_sequence_items_use_aspect_hints_for_non_code_timeline(self) -> None:
        pack = EvidencePack(
            query=(
                "Can you walk me through the order in which I brought up different aspects "
                "of using AI in our hiring process? Mention ONLY and ONLY four items."
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "timeline_index": 1,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "I'm worried about using AI for hiring because I'm not sure it can recognize soft skills.",
                },
                {
                    "id": "s2",
                    "timeline_index": 2,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "Ok cool, do I need to look into any specific AI hiring tools that are known for their fairness and transparency?",
                },
                {
                    "id": "s3",
                    "timeline_index": 3,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "I think starting with a pilot program makes sense for resume screening, efficiency, and diversity of our candidate pool.",
                },
                {
                    "id": "s4",
                    "timeline_index": 4,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "My friend Maya suggested AI hiring over lunch, and I value her opinion about bias considerations.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        sequence_items = _pack_for_model(pack)["sequence_items"]
        labels = [item["label"].lower() for item in sequence_items]

        self.assertEqual(len(sequence_items), 4)
        self.assertEqual([item["timeline_index"] for item in sequence_items], [1, 2, 3, 4])
        self.assertTrue(all(label for label in labels))
        self.assertIn("ai", labels[0])
        self.assertIn("soft skills", labels[0])
        self.assertTrue("fairness" in labels[1] or "transparency" in labels[1])
        self.assertTrue("screening" in labels[2] or "diversity" in labels[2])
        self.assertIn("maya", labels[3])

    def test_event_ordering_raw_chronology_fallback_replaces_drifting_sequence_items(self) -> None:
        pack = EvidencePack(
            query=(
                "Can you walk me through the order in which I brought up different personal "
                "and work-related challenges during our chats? Mention ONLY and ONLY four items."
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "a1",
                    "timeline_index": 1,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Burnout and stress management",
                    "content": "I've identified early signs of burnout, such as fatigue, irritability, and sleep issues at work.",
                },
                {
                    "id": "a2",
                    "timeline_index": 2,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Vacation and unplugging",
                    "content": "David and I are going on a weekend getaway, and I want to unplug because I am worried about burnout.",
                },
                {
                    "id": "a3",
                    "timeline_index": 3,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Partner connection planning",
                    "content": "I'm nervous about planning our anniversary dinner and choosing the right menu around David's favorites.",
                },
                {
                    "id": "a4",
                    "timeline_index": 4,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "timeline_label": "Mental health over extra income",
                    "content": "I want to prioritize mental health over extra income.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        self.assertIn("raw_chronology_items", model_pack)
        self.assertEqual(len(model_pack["raw_chronology_items"]), 4)
        self.assertEqual([item["timeline_index"] for item in model_pack["sequence_items"]], [1, 2, 3, 4])
        self.assertEqual(
            [item["label"].lower() for item in model_pack["sequence_items"]],
            [
                "burnout and stress management",
                "vacation and unplugging",
                "partner connection planning",
                "mental health over extra income",
            ],
        )

    def test_event_ordering_sequence_output_sort_key_uses_timeline_before_source_id(self) -> None:
        records = [
            {
                "source_uri": "beam:100k:18:document:15",
                "turn_id": "turn-15",
                "timeline_index": 15,
                "source_span_id": "span-z",
            },
            {
                "source_uri": "beam:100k:18:document:2",
                "turn_id": "turn-2",
                "timeline_index": 2,
                "source_span_id": "span-a",
            },
            {
                "source_uri": "beam:100k:18:document:10",
                "turn_id": "turn-10",
                "timeline_index": 10,
                "source_span_id": "span-c",
            },
            {
                "source_uri": "beam:100k:18:document:6",
                "turn_id": "turn-6",
                "timeline_index": 6,
                "source_span_id": "span-b",
            },
        ]

        ordered = sorted(records, key=_event_ordering_sequence_output_sort_key)

        self.assertEqual([item["timeline_index"] for item in ordered], [2, 6, 10, 15])

    def test_event_ordering_chronology_rescue_scores_out_of_window_profile_updates(self) -> None:
        query = (
            "Can you list the order in which I brought up different aspects of improving "
            "my professional profile and resume throughout our conversations?"
        )

        score = _event_ordering_chronology_rescue_score(
            query,
            "I used Textio to identify 15 action verbs for my resume and Applicant Tracking System readiness.",
            "user",
        )
        assistant_plan_score = _event_ordering_chronology_rescue_score(
            query,
            "Sure, let's break it down. Components: resume update, portfolio polish, and networking milestones.",
            "user",
        )

        self.assertGreaterEqual(score, 0.26)
        self.assertEqual(assistant_plan_score, 0.0)

    def test_event_ordering_query_scoped_phase_selector_filters_topic_drift(self) -> None:
        pack = EvidencePack(
            query=(
                "Can you walk me through the order in which I brought up different aspects "
                "of using AI in our hiring process across our conversations? Mention ONLY and ONLY four items."
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "soft",
                    "timeline_index": 1,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "I'm worried about using AI for hiring because I'm not sure it can recognize soft skills.",
                },
                {
                    "id": "automation",
                    "timeline_index": 2,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "I'm considering automating the hiring process with an AI screening workflow.",
                },
                {
                    "id": "pilot",
                    "timeline_index": 3,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "I think starting with a pilot program makes sense for resume screening and candidate pool diversity.",
                },
                {
                    "id": "burnout",
                    "timeline_index": 4,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "I'm worried about burnout and stress management after too many production meetings.",
                },
                {
                    "id": "fairness",
                    "timeline_index": 5,
                    "speaker": "user",
                    "selector": "event_ordering_coverage",
                    "timeline_role": "user_aspect_anchor",
                    "content": "Do I need to look into AI hiring tools known for fairness and transparency?",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        labels = [item["label"].lower() for item in _pack_for_model(pack)["sequence_items"]]

        self.assertEqual(len(labels), 4)
        self.assertTrue(all("burnout" not in label and "stress" not in label for label in labels))
        self.assertTrue(any("soft skills" in label for label in labels))
        self.assertTrue(any("screening" in label or "candidate" in label for label in labels))

    def test_event_ordering_typed_aspects_cover_personal_work_challenges(self) -> None:
        pack = EvidencePack(
            query=(
                "Can you walk me through the order in which I brought up different personal "
                "and work-related challenges during our chats? Mention ONLY and ONLY four items."
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "burnout",
                    "timeline_index": 1,
                    "speaker": "user",
                    "content": "I've identified early signs of burnout, such as fatigue, irritability, and sleep issues at work.",
                },
                {
                    "id": "getaway",
                    "timeline_index": 2,
                    "speaker": "user",
                    "content": "David and I are going on a weekend getaway, and I want to unplug because I am worried about burnout.",
                },
                {
                    "id": "anniversary",
                    "timeline_index": 3,
                    "speaker": "user",
                    "content": "I'm nervous about planning our anniversary dinner and choosing the right menu around David's favorites.",
                },
                {
                    "id": "surprise",
                    "timeline_index": 4,
                    "speaker": "user",
                    "content": "I want to plan a surprise celebration and return the favor after David supported me.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        labels = [item["label"].lower() for item in _pack_for_model(pack)["sequence_items"]]

        self.assertEqual(len(labels), 4)
        self.assertIn("burnout", labels[0])
        self.assertTrue("vacation" in labels[1] or "unplug" in labels[1] or "getaway" in labels[1])
        self.assertIn("anniversary", labels[2])
        self.assertIn("surprise", labels[3])

    def test_event_ordering_cluster_label_prefers_chronological_representative(self) -> None:
        from fusion_memory.eval.model_adapters import _event_ordering_cluster_label

        label = _event_ordering_cluster_label(
            [
                "Here's my current CSS code",
                "fix a known modal accessibility bug in my Bootstrap project by upgrading from v5.3.0 to",
            ],
            [
                "I'm trying to refactor my CSS code by removing redundant selectors and consolidating media queries. Here's my current CSS code.",
                "I'm trying to fix a known modal accessibility bug in my Bootstrap project.",
            ],
        )

        self.assertRegex(label.lower(), r"remov(?:e|ing) redundant selectors")
        self.assertNotIn("bootstrap", label.lower())

    def test_event_ordering_phase_clusters_split_on_topic_shift(self) -> None:
        from fusion_memory.eval.model_adapters import _event_ordering_phase_clusters

        anchors = [
            {"timeline_index": 1, "label": "initial project setup", "content": "I set up the Flask app and local server.", "conversation_content": "I set up the Flask app and local server."},
            {"timeline_index": 2, "label": "database schema", "content": "I added the users and transactions tables.", "conversation_content": "I added the users and transactions tables."},
            {"timeline_index": 3, "label": "render deployment", "content": "I configured Render and Gunicorn deployment.", "conversation_content": "I configured Render and Gunicorn deployment."},
            {"timeline_index": 4, "label": "integration tests", "content": "I expanded integration coverage for auth and transactions.", "conversation_content": "I expanded integration coverage for auth and transactions."},
        ]

        clusters = _event_ordering_phase_clusters("Mention ONLY and ONLY three items.", anchors)

        self.assertGreaterEqual(len(clusters), 3)
        self.assertLessEqual(len(clusters), 4)
        self.assertEqual(clusters[0]["timeline_start"], 1)
        self.assertIn("initial project setup", " ".join(clusters[0]["candidate_labels"]).lower())

    def test_event_ordering_milestone_selection_prefers_source_diversity(self) -> None:
        from fusion_memory.eval.model_adapters import _event_ordering_select_milestones

        candidates = [
            {"milestone_group": "initial_project_setup", "timeline_index": 1, "source_span_id": "s1"},
            {"milestone_group": "transaction_crud_implementation", "timeline_index": 2, "source_span_id": "s2"},
            {"milestone_group": "deployment_configuration", "timeline_index": 5, "source_span_id": "s5"},
            {"milestone_group": "deployment_and_test_improvements", "timeline_index": 5, "source_span_id": "s5"},
            {"milestone_group": "integration_test_coverage", "timeline_index": 5, "source_span_id": "s5"},
            {"milestone_group": "security_auth", "timeline_index": 6, "source_span_id": "s6"},
            {"milestone_group": "transaction_error_handling", "timeline_index": 3, "source_span_id": "s3"},
        ]

        selected = _event_ordering_select_milestones(
            "Walk me through my app development and deployment. Mention ONLY five items.",
            candidates,
            5,
        )

        source_ids = [item["source_span_id"] for item in selected]
        self.assertEqual(len(selected), 5)
        self.assertLessEqual(source_ids.count("s5"), 1)
        self.assertIn("s3", source_ids)
        self.assertEqual(
            [item["timeline_index"] for item in selected],
            sorted(item["timeline_index"] for item in selected),
        )

    def test_eval_answer_model_adds_strict_abstention_and_contradiction_instructions(self) -> None:
        pack = EvidencePack(
            query="Question",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[{"id": "s1", "content": "Evidence."}],
            conflicts=[],
            coverage={},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            model = OpenAICompatibleAnswerModel(client)
            model.answer_with_context("Missing?", pack, benchmark="BEAM", category="abstention")
            model.answer_with_context("Contradiction?", pack, benchmark="BEAM", category="contradiction_resolution")

        abstention_input = _decode_model_payload(server.requests[-2]["json"])["input"]
        contradiction_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        self.assertIn("does not contain that information", abstention_input["instruction"])
        self.assertIn("contradictory claims", contradiction_input["instruction"])
        self.assertIn("conflict_claims", contradiction_input["instruction"])

    def test_pack_for_model_includes_conflict_claim_contents(self) -> None:
        pack = EvidencePack(
            query="Have I used Excel to track expenses?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s-pos",
                    "speaker": "user",
                    "content": "I have been using Excel to track my daily expenses since March 1.",
                },
                {
                    "id": "s-neg",
                    "speaker": "user",
                    "content": "I have never used Excel for tracking expenses before.",
                },
            ],
            conflicts=[
                {
                    "type": "claim_polarity_buckets",
                    "positive_source_span_ids": ["s-pos"],
                    "negative_source_span_ids": ["s-neg"],
                    "uncertain_source_span_ids": [],
                    "note": "Buckets organize retrieved raw claims by surface polarity.",
                }
            ],
            coverage={"query_type": "contradiction_resolution"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        self.assertIn("conflict_claims", model_pack)
        self.assertEqual(model_pack["conflict_claims"][0]["positive"][0]["source_span_id"], "s-pos")
        self.assertIn("using Excel", model_pack["conflict_claims"][0]["positive"][0]["claim"])
        self.assertEqual(model_pack["conflict_claims"][0]["negative"][0]["source_span_id"], "s-neg")

    def test_pack_for_model_adds_conflict_resolution_candidate_from_current_support(self) -> None:
        pack = EvidencePack(
            query="Have I been using Excel to track my daily expenses?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s-pos",
                    "speaker": "user",
                    "content": "I've been using Excel to track my daily expenses since March 1.",
                },
                {
                    "id": "s-neg",
                    "speaker": "user",
                    "content": "I've never used Excel for tracking expenses, can you help me get started?",
                },
                {
                    "id": "s-current",
                    "speaker": "user",
                    "content": "I'm frustrated with tracking my daily expenses for 3 months now, but I persisted.",
                },
                {
                    "id": "s-keep",
                    "speaker": "assistant",
                    "content": "Keep using Excel to track daily spending and review it weekly.",
                },
            ],
            conflicts=[
                {
                    "type": "claim_polarity_buckets",
                    "positive_source_span_ids": ["s-pos"],
                    "negative_source_span_ids": ["s-neg"],
                    "uncertain_source_span_ids": ["s-current", "s-keep"],
                }
            ],
            coverage={"query_type": "contradiction_resolution"},
            debug_trace=[],
        )

        conflict = _pack_for_model(pack)["conflict_claims"][0]

        self.assertEqual(conflict["resolution_candidate"]["resolved_answer"], "yes")
        self.assertGreater(conflict["resolution_candidate"]["support_counts"]["positive_or_current"], 1)

    def test_conflict_claims_prioritize_query_specific_user_claims(self) -> None:
        pack = EvidencePack(
            query="Have I ever invited Mason or Michael to join any family movie events?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "generic-positive",
                    "speaker": "assistant",
                    "candidate_source": "contradiction_claim_positive",
                    "content": "Family movie events can be a great way to include guests and plan snacks.",
                },
                {
                    "id": "direct-positive",
                    "speaker": "user",
                    "candidate_source": "contradiction_claim_uncertain+exact_filter",
                    "content": "I've invited Mason and Michael to join the April 7 afternoon session, but Michael declined.",
                },
                {
                    "id": "direct-negative",
                    "speaker": "user",
                    "candidate_source": "contradiction_claim_negative+exact_filter",
                    "content": "I've never invited Mason or Michael to any family movie events.",
                },
            ],
            conflicts=[
                {
                    "type": "claim_polarity_buckets",
                    "positive_source_span_ids": ["generic-positive"],
                    "negative_source_span_ids": ["direct-negative"],
                    "uncertain_source_span_ids": ["direct-positive"],
                }
            ],
            coverage={"query_type": "contradiction_resolution"},
            debug_trace=[],
        )

        conflict = _pack_for_model(pack)["conflict_claims"][0]

        self.assertEqual(conflict["positive"][0]["source_span_id"], "direct-positive")
        self.assertEqual(conflict["negative"][0]["source_span_id"], "direct-negative")

    def test_conflict_claims_rescue_query_grounded_user_claims_from_source_spans(self) -> None:
        pack = EvidencePack(
            query="Do I usually feel anxious about my grammar accuracy after receiving feedback?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "weak-positive",
                    "speaker": "user",
                    "content": "I've recently integrated the ProWritingAid desktop app on May 21.",
                },
                {
                    "id": "direct-positive",
                    "speaker": "user",
                    "content": "I'm feeling kinda anxious about my grammar accuracy after Joseph's feedback on Feb 28.",
                },
                {
                    "id": "direct-negative",
                    "speaker": "user",
                    "content": "I've never felt anxious about grammar accuracy after any feedback.",
                },
            ],
            conflicts=[
                {
                    "type": "claim_polarity_buckets",
                    "positive_source_span_ids": ["weak-positive"],
                    "negative_source_span_ids": ["direct-negative"],
                    "uncertain_source_span_ids": [],
                }
            ],
            coverage={"query_type": "contradiction_resolution"},
            debug_trace=[],
        )

        conflict = _pack_for_model(pack)["conflict_claims"][0]

        self.assertEqual(conflict["positive"][0]["source_span_id"], "direct-positive")
        self.assertIn("anxious about my grammar accuracy", conflict["positive"][0]["claim"])
        self.assertEqual(conflict["negative"][0]["source_span_id"], "direct-negative")

    def test_conflict_claims_do_not_resolve_help_requests_as_completed_facts(self) -> None:
        pack = EvidencePack(
            query="Have I obtained an API key for this project?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "trying-settings",
                    "speaker": "user",
                    "content": "I'm trying to update my API key settings to reflect the new daily quota of 1,200 calls per day.",
                },
                {
                    "id": "restrict-key",
                    "speaker": "user",
                    "content": "I'm trying to restrict my API key usage to the weather domain to avoid exposing it in client code.",
                },
                {
                    "id": "direct-negative",
                    "speaker": "user",
                    "content": "I've never actually obtained an API key for this project, so I'm not sure how to proceed.",
                },
            ],
            conflicts=[
                {
                    "type": "claim_polarity_buckets",
                    "positive_source_span_ids": ["trying-settings"],
                    "negative_source_span_ids": ["direct-negative"],
                    "uncertain_source_span_ids": ["restrict-key"],
                }
            ],
            coverage={"query_type": "contradiction_resolution"},
            debug_trace=[],
        )

        conflict = _pack_for_model(pack)["conflict_claims"][0]

        self.assertNotIn("resolution_candidate", conflict)
        self.assertEqual(conflict["negative"][0]["claim_role"], "direct_negative")
        self.assertIn(conflict["positive"][0]["claim_role"], {"planned_or_intended", "current_state_explicit"})

    def test_conflict_claims_use_query_clause_for_mixed_positive_negative_sentence(self) -> None:
        pack = EvidencePack(
            query="Have I ever met Kyle or been to any sneaker expos?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "mixed-positive",
                    "speaker": "user",
                    "content": "I met Kyle back in 2018 at a sneaker expo in Bridgetown, Barbados, but I've never tried Nike Air Max before.",
                },
                {
                    "id": "direct-negative",
                    "speaker": "user",
                    "content": "I've never met anyone like Kyle or been to sneaker expos.",
                },
            ],
            conflicts=[
                {
                    "type": "claim_polarity_buckets",
                    "positive_source_span_ids": ["mixed-positive"],
                    "negative_source_span_ids": ["direct-negative"],
                    "uncertain_source_span_ids": [],
                }
            ],
            coverage={"query_type": "contradiction_resolution"},
            debug_trace=[],
        )

        conflict = _pack_for_model(pack)["conflict_claims"][0]

        self.assertEqual(conflict["positive"][0]["source_span_id"], "mixed-positive")
        self.assertEqual(conflict["positive"][0]["claim_polarity"], "positive")
        self.assertEqual(conflict["positive"][0]["claim_role"], "past_experience_explicit")
        self.assertIn("met Kyle back in 2018", conflict["positive"][0]["claim"])
        self.assertNotIn("never tried Nike Air Max", conflict["positive"][0]["claim"])

    def test_answer_model_passes_grounded_conflict_claims_to_llm(self) -> None:
        pack = EvidencePack(
            query="Have I been using Excel to track my daily expenses?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "s-pos", "speaker": "user", "content": "I've been using Excel to track my daily expenses since March 1."},
                {"id": "s-neg", "speaker": "user", "content": "I've never used Excel for tracking expenses, can you help me get started?"},
                {"id": "s-current", "speaker": "user", "content": "I'm frustrated with tracking my daily expenses for 3 months now, but I persisted."},
            ],
            conflicts=[
                {
                    "type": "claim_polarity_buckets",
                    "positive_source_span_ids": ["s-pos"],
                    "negative_source_span_ids": ["s-neg"],
                    "uncertain_source_span_ids": ["s-current"],
                }
            ],
            coverage={"query_type": "contradiction_resolution"},
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            model = OpenAICompatibleAnswerModel(client)
            model.answer_with_context(pack.query, pack, benchmark="BEAM", category="contradiction_resolution")

        payload = _decode_model_payload(server.requests[-1]["json"])["input"]
        claims = payload["evidence_pack"]["conflict_claims"][0]

        self.assertIn("positive", claims)
        self.assertIn("negative", claims)
        self.assertIn("resolution_candidate", claims)
        self.assertIn("I've been using Excel to track my daily expenses", claims["positive"][0]["claim"])
        self.assertIn("never used Excel", claims["negative"][0]["claim"])

    def test_pack_for_model_adds_summary_coverage_matrix(self) -> None:
        pack = EvidencePack(
            query="Summarize how my fiction book budgeting decisions evolved.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "budget",
                    "speaker": "user",
                    "content": 'I allocated $120 for print fiction series from Montserrat Books.',
                },
                {
                    "id": "decision",
                    "speaker": "user",
                    "content": 'I ordered the "Outlander" paperback box set on March 5 and chose audiobooks for new releases.',
                },
            ],
            conflicts=[],
            coverage={"query_type": "summarization"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        self.assertIn("summary_highlights", model_pack)
        self.assertIn("summary_coverage", model_pack)
        facets = model_pack["summary_coverage"]["facets"]
        self.assertIn("money_or_budget", facets)
        self.assertIn("named_item", facets)
        self.assertIn("$120", facets["money_or_budget"][0]["content"])
        must_cover = model_pack["summary_coverage"]["must_cover_highlights"]
        self.assertTrue(any("$120" in row["content"] for row in must_cover))
        self.assertTrue(any("Outlander" in row["content"] for row in must_cover))
        must_mentions = model_pack["summary_coverage"]["must_mention_points"]
        self.assertTrue(any("$120 budget" in point and "Montserrat Books" in point for point in must_mentions))
        self.assertTrue(any("Outlander" in point and "recent" in point for point in must_mentions))

    def test_summary_highlights_rescue_budget_decision_tail_items(self) -> None:
        spans = [
            {
                "id": f"generic-{i}",
                "speaker": "assistant",
                "content": (
                    f"On March {i + 1}, recommended fiction series {i} with 12 books, audiobook options, "
                    "and historical fantasy themes for winter evenings."
                ),
            }
            for i in range(14)
        ]
        spans.append(
            {
                "id": "witcher-contest",
                "speaker": "assistant",
                "content": (
                    'Given your current book budget, entering the "The Witcher" fan fiction contest is constrained: '
                    "your $35 monthly book budget has $28 already spent, leaving a $7 remaining budget, so the entry "
                    "fee and limited remaining funds make the decision tight."
                ),
            }
        )
        pack = EvidencePack(
            query="Summarize how my plans and decisions around choosing and budgeting for fiction books evolved.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=spans,
            conflicts=[],
            coverage={"query_type": "summarization"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        contents = [
            row["content"]
            for row in model_pack["summary_coverage"]["must_cover_highlights"]
        ]
        points = model_pack["summary_coverage"]["must_mention_points"]

        self.assertTrue(any("Witcher" in content and "$7" in content for content in contents))
        self.assertTrue(any("Witcher" in point and "limited remaining funds" in point for point in points))

    def test_summary_must_mentions_use_full_highlight_set(self) -> None:
        spans = [
            {
                "id": f"generic-{i}",
                "speaker": "assistant",
                "content": (
                    f"On March {i + 1}, recommended historical fiction series {i} with 12 books, "
                    "events, deadlines, and audiobook options for winter evenings."
                ),
            }
            for i in range(18)
        ]
        spans.extend(
            [
                {
                    "id": "budget",
                    "speaker": "user",
                    "content": "I've allocated $120 for book purchases this winter and want print editions from Montserrat Books.",
                },
                {
                    "id": "poppy",
                    "speaker": "user",
                    "content": 'I bought a $25 boxed set of "The Poppy War" trilogy for my winter reading challenge.',
                },
            ]
        )
        pack = EvidencePack(
            query="Summarize how my plans and decisions around choosing and budgeting for fiction books evolved.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=spans,
            conflicts=[],
            coverage={"query_type": "summarization"},
            debug_trace=[],
        )

        points = _pack_for_model(pack)["summary_coverage"]["must_mention_points"]

        self.assertTrue(any("$120 budget" in point for point in points))
        self.assertTrue(any("Poppy War" in point and "reading challenge" in point for point in points))

    def test_pack_for_model_exposes_summary_must_mention_points_without_answer_template(self) -> None:
        pack = EvidencePack(
            query="Summarize how my plans and decisions around choosing and budgeting for fiction books evolved.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "budget", "speaker": "user", "content": "I've allocated $120 for print editions from Montserrat Books."},
                {"id": "poppy", "speaker": "user", "content": 'I bought a $25 boxed set of "The Poppy War" trilogy for my winter reading challenge.'},
                {
                    "id": "format",
                    "speaker": "user",
                    "content": "I prioritize print editions for series I plan to reread, while preferring audiobooks for new releases.",
                },
                {
                    "id": "witcher",
                    "speaker": "assistant",
                    "content": 'Entering the "The Witcher" fan fiction contest is constrained by your limited remaining funds.',
                },
            ],
            conflicts=[],
            coverage={"query_type": "summarization"},
            debug_trace=[],
        )
        model_pack = _pack_for_model(pack)
        points = model_pack["summary_coverage"]["must_mention_points"]

        self.assertTrue(any("$120 budget" in point for point in points))
        self.assertTrue(any("Poppy War" in point for point in points))
        self.assertTrue(any("print editions" in point for point in points))
        self.assertTrue(any("Witcher" in point for point in points))

    def test_summary_must_mentions_filter_assistant_boilerplate(self) -> None:
        pack = EvidencePack(
            query="Summarize how my project planning decisions evolved.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "boilerplate",
                    "speaker": "assistant",
                    "content": (
                        "Here are a few final points to reinforce your decision and help you move forward with confidence: "
                        "keep the plan simple and review progress weekly."
                    ),
                },
                {
                    "id": "decision",
                    "speaker": "user",
                    "content": (
                        "I decided on March 12 to use the lightweight project plan, cut the budget to $500, "
                        "and review the timeline every 2 weeks."
                    ),
                },
            ],
            conflicts=[],
            coverage={"query_type": "summarization"},
            debug_trace=[],
        )

        points = _pack_for_model(pack)["summary_coverage"]["must_mention_points"]

        self.assertFalse(any("Here are a few final points" in point for point in points))
        self.assertTrue(any("March 12" in point and "$500" in point for point in points))

    def test_eval_answer_model_includes_temporal_history_and_resolution_structures(self) -> None:
        pack = EvidencePack(
            query="What is my current target in days?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "s1",
                    "speaker": "user",
                    "content": "The launch moved to March 20, 2026.",
                    "temporal_mentions": [
                        {
                            "text": "March 20, 2026",
                            "normalized_date": "2026-03-20",
                            "role": "deadline_date",
                            "role_confidence": 0.9,
                            "context": "The launch moved to March 20, 2026.",
                        }
                    ],
                    "temporal_roles": ["deadline_date"],
                },
                {
                    "id": "s2",
                    "speaker": "user",
                    "content": "Current target is 18 days.",
                    "value_mentions": [{"type": "duration", "text": "18 days", "context": "Current target is 18 days."}],
                },
                {
                    "id": "s3",
                    "speaker": "user",
                    "content": "I hit a CSS layout issue on the page.",
                },
                {
                    "id": "s4",
                    "speaker": "assistant",
                    "content": "Try adjusting the flex settings and reducing overflow.",
                },
            ],
            conflicts=[],
            coverage={
                "query_type": "summarization",
                "temporal_candidates": [
                    {
                        "source_span_id": "s1",
                        "speaker": "user",
                        "timeline_index": 2,
                        "role": "deadline_date",
                        "normalized_date": "2026-03-20",
                        "context": "The launch moved to March 20, 2026.",
                        "confidence": 0.9,
                    }
                ],
                "value_history": [
                    {
                        "source_span_id": "s1",
                        "speaker": "user",
                        "timeline_index": 1,
                        "recency_rank": 3,
                        "value_type": "duration",
                        "value": "10 days",
                        "context": "Old target was 10 days.",
                        "subject_key": "subject:target",
                        "current": False,
                        "query_overlap": 2,
                    },
                    {
                        "source_span_id": "s2",
                        "speaker": "user",
                        "timeline_index": 3,
                        "recency_rank": 1,
                        "value_type": "duration",
                        "value": "18 days",
                        "context": "Current target is 18 days.",
                        "subject_key": "subject:target",
                        "current": True,
                        "query_overlap": 2,
                    }
                ],
                "resolution_pairs": [
                    {
                        "issue_span_id": "s3",
                        "issue": "I hit a CSS layout issue on the page.",
                        "resolution_span_id": "s4",
                        "resolution": "Try adjusting the flex settings and reducing overflow.",
                    }
                ],
                "summary_clusters": [
                    {
                        "cluster_key": "project-a",
                        "representative_span_id": "s3",
                        "representative": "I hit a CSS layout issue on the page.",
                        "span_count": 3,
                    }
                ],
                "instruction_constraints": ["exact_count_or_scope", "format_constraint"],
            },
            debug_trace=[],
        )
        with FakeModelServer() as server:
            client = OpenAICompatibleLLMClient(server.url("/answer"), model="answer-model")
            model = OpenAICompatibleAnswerModel(client)
            model.answer_with_context(pack.query, pack, benchmark="BEAM", category="temporal_reasoning")
            model.answer_with_context(pack.query, pack, benchmark="BEAM", category="knowledge_update")
            model.answer_with_context(pack.query, pack, benchmark="BEAM", category="summarization")
            model.answer_with_context(pack.query, pack, benchmark="BEAM", category="instruction_following")

        temporal_input = _decode_model_payload(server.requests[-4]["json"])["input"]
        history_input = _decode_model_payload(server.requests[-3]["json"])["input"]
        summary_input = _decode_model_payload(server.requests[-2]["json"])["input"]
        instruction_input = _decode_model_payload(server.requests[-1]["json"])["input"]
        self.assertIn("temporal_candidates", temporal_input["evidence_pack"])
        self.assertIn("value_history", history_input["evidence_pack"])
        self.assertIn("value_state_summary", history_input["evidence_pack"])
        self.assertIn("value_history_summary", history_input["evidence_pack"])
        self.assertEqual(history_input["evidence_pack"]["value_history"][0]["subject_key"], "subject:target")
        self.assertEqual(history_input["evidence_pack"]["value_state_summary"]["resolved_value"], "18 days")
        self.assertEqual(history_input["evidence_pack"]["value_history_summary"]["current_candidates"][0]["value"], "18 days")
        self.assertIn("resolution_pairs", summary_input["evidence_pack"])
        self.assertIn("summary_clusters", summary_input["evidence_pack"])
        self.assertIn("summary_highlights", summary_input["evidence_pack"])
        self.assertIn("summary_coverage", summary_input["evidence_pack"])
        self.assertIn("coverage-first structure", summary_input["instruction"])
        self.assertTrue(summary_input["evidence_pack"]["summary_highlights"][0]["facets"])
        self.assertIn("facets", summary_input["evidence_pack"]["summary_coverage"])
        self.assertEqual(instruction_input["evidence_pack"]["instruction_constraints"], ["exact_count_or_scope", "format_constraint"])

    def test_pack_for_model_extracts_instruction_answer_requirements(self) -> None:
        pack = EvidencePack(
            query="When was my meetings at East Janethaven Library? Please use MM/DD/YYYY.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "format",
                    "speaker": "user",
                    "content": "For meeting reminders, please format dates as MM/DD/YYYY.",
                }
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        requirements = _pack_for_model(pack)["answer_requirements"]["must_satisfy"]

        self.assertTrue(any(item["type"] == "date_format" and "MM/DD/YYYY" in item["requirement"] for item in requirements))

    def test_pack_for_model_extracts_instruction_version_requirements_when_requested(self) -> None:
        pack = EvidencePack(
            query="Which versions of the libraries and dependencies should I use?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "tools",
                    "speaker": "assistant",
                    "content": "Use Google Drive version 95.0, Dropbox v184.4, and OneDrive 24.1 if exact software versions matter.",
                }
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        requirements = _pack_for_model(pack)["answer_requirements"]["must_satisfy"]

        self.assertTrue(any(item["type"] == "version_detail" for item in requirements))

    def test_pack_for_model_does_not_require_versions_for_generic_tools(self) -> None:
        pack = EvidencePack(
            query="What are some popular tools I can use to organize and manage my digital files?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "tools",
                    "speaker": "assistant",
                    "content": "Use tools like calendar apps to stay organized.",
                }
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        requirements = _pack_for_model(pack).get("answer_requirements", {}).get("must_satisfy", [])

        self.assertFalse(any(item["type"] == "version_detail" for item in requirements))

    def test_pack_for_model_does_not_treat_place_library_as_version_requirement(self) -> None:
        pack = EvidencePack(
            query="When was my meetings at East Janethaven Library?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "place",
                    "speaker": "user",
                    "content": "I'm stressed about this book club meeting at East Janethaven Library on July 18.",
                },
                {
                    "id": "software-adjacent",
                    "speaker": "assistant",
                    "content": "Use tools like calendar apps to stay organized.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        requirements = _pack_for_model(pack)["answer_requirements"]["must_satisfy"]

        self.assertFalse(any(item["type"] == "version_detail" for item in requirements))

    def test_pack_for_model_adds_direct_date_candidates_for_instruction_dates(self) -> None:
        pack = EvidencePack(
            query="When is the final submission due? Please use MM/DD/YYYY.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "other-deadlines",
                    "speaker": "user",
                    "content": "The scholarship deadline is May 15, 2024, and the visa application is due June 1, 2024.",
                },
                {
                    "id": "submission",
                    "speaker": "user",
                    "content": "I've completed 3 drafts, but I'm unsure if my final version is ready for submission by May 12, 2024.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)
        candidates = model_pack["direct_date_answer_candidates"]

        self.assertEqual(candidates[0]["answer_value"], "2024-05-12")
        self.assertEqual(candidates[0]["date_mm_dd_yyyy"], "05/12/2024")

    def test_answer_model_formats_high_confidence_instruction_date_candidate(self) -> None:
        class DateClient:
            def structured(self, prompt, schema, input):
                return {"answer": "wrong"}

        pack = EvidencePack(
            query="When is the final submission due? Please use MM/DD/YYYY.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "other-deadlines",
                    "speaker": "user",
                    "content": "The scholarship deadline is May 15, 2024, and the visa application is due June 1, 2024.",
                },
                {
                    "id": "submission",
                    "speaker": "user",
                    "content": "I've completed 3 drafts, but I'm unsure if my final version is ready for submission by May 12, 2024.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        answer = OpenAICompatibleAnswerModel(DateClient()).answer_with_context(
            pack.query,
            pack,
            benchmark="BEAM",
            category="instruction_following",
        )

        self.assertEqual(answer, "05/12/2024")

    def test_pack_for_model_projects_preference_date_format_to_answer_requirements(self) -> None:
        pack = EvidencePack(
            query="When was my meetings at East Janethaven Library?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "format-rule",
                    "speaker": "user",
                    "content": "Always format dates in MM/DD/YYYY when I ask about scheduling details.",
                },
                {
                    "id": "meeting",
                    "speaker": "user",
                    "content": "The book club meeting at East Janethaven Library is on July 18.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        requirements = _pack_for_model(pack)["answer_requirements"]["must_satisfy"]

        self.assertTrue(any(item["type"] == "date_format" and "MM/DD/YYYY" in item["requirement"] for item in requirements))

    def test_pack_for_model_projects_supported_version_instruction_for_tools(self) -> None:
        pack = EvidencePack(
            query="What are some popular tools I can use to organize and manage my digital files?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "version-rule",
                    "speaker": "assistant",
                    "content": "I'll include software version details when discussing digital asset management tools. WillMaker Pro Version: 2024.1.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        requirements = _pack_for_model(pack)["answer_requirements"]["must_satisfy"]

        self.assertTrue(any(item["type"] == "version_detail" for item in requirements))

    def test_pack_for_model_extracts_instruction_platform_and_explanation_requirements(self) -> None:
        movie_pack = EvidencePack(
            query="What movies would you recommend for me to watch?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "streaming", "speaker": "assistant", "content": "Check Netflix, Disney+, and Hulu availability before recommending movies."}
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )
        legal_pack = EvidencePack(
            query="What do I need to include to make sure my wishes are legally valid?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {"id": "legal", "speaker": "assistant", "content": "Explain terms such as executor, beneficiary, witness, and notarized affidavit."}
            ],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        movie_requirements = _pack_for_model(movie_pack)["answer_requirements"]["must_satisfy"]
        legal_requirements = _pack_for_model(legal_pack)["answer_requirements"]["must_satisfy"]

        self.assertTrue(any(item["type"] == "platform_detail" for item in movie_requirements))
        self.assertTrue(any(item["type"] == "explanation_depth" for item in legal_requirements))

    def test_pack_for_model_extracts_instruction_security_requirements(self) -> None:
        pack = EvidencePack(
            query="What should I know about keeping my information safe when using online services?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={"query_type": "instruction"},
            debug_trace=[],
        )

        requirements = _pack_for_model(pack)["answer_requirements"]["must_satisfy"]

        self.assertTrue(any(item["type"] == "security_detail" for item in requirements))
        self.assertFalse(any(item["type"] == "platform_detail" for item in requirements))

    def test_pack_for_model_extracts_deep_preference_constraints(self) -> None:
        spans = [
            {
                "id": f"s{i}",
                "speaker": "assistant",
                "content": f"Generic writing advice {i}: set goals and review your draft.",
            }
            for i in range(30)
        ]
        spans.append(
            {
                "id": "pref-time",
                "speaker": "user",
                "content": "I prefer writing in the mornings between 7-9 AM because I am most focused then.",
                "recency_rank": 1,
            }
        )
        spans.append(
            {
                "id": "pref-short",
                "speaker": "user",
                "content": "I prefer editing in short bursts, 30 minutes at a time, rather than marathon sessions.",
                "recency_rank": 2,
            }
        )
        pack = EvidencePack(
            query="Can you help me plan my writing sessions for the upcoming week?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=spans,
            conflicts=[],
            coverage={"query_type": "factual_exact"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        labels = [item["label"] for item in model_pack["preference_constraints"]]
        self.assertIn("preferred time window: 7-9 AM", labels)
        self.assertIn("short session length: 30 minutes", labels)
        checklist = model_pack["preference_requirement_checklist"]
        requirements = [item["requirement"] for item in checklist["must_satisfy"]]
        self.assertIn("Respect the timing preference: preferred time window: 7-9 AM.", requirements)
        self.assertIn("Use the explicit short-session length: short session length: 30 minutes.", requirements)
        self.assertLessEqual(len(model_pack["source_spans"]), 20)

    def test_pack_for_model_extracts_durable_instruction_constraints(self) -> None:
        pack = EvidencePack(
            query="How should I approach editing my draft, explain dependent event probability, and discuss social norms?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "tree",
                    "speaker": "user",
                    "content": "Always include a tree diagram when explaining dependent probability problems without replacement.",
                },
                {
                    "id": "split",
                    "speaker": "user",
                    "content": "Always use Scrivener's split-screen mode for editing when I ask about draft revisions.",
                },
                {
                    "id": "culture",
                    "speaker": "user",
                    "content": "Always include cultural context when I ask about social norms and meeting expectations.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "factual_exact"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        labels = {item["label"] for item in model_pack["preference_constraints"]}
        self.assertIn("include a tree diagram for dependent probability problems", labels)
        self.assertIn("use Scrivener split-screen mode for draft revisions", labels)
        self.assertIn("include cultural context and cross-cultural variation for social norms", labels)

    def test_pack_for_model_uses_coverage_preference_constraints(self) -> None:
        pack = EvidencePack(
            query="How should I break up my editing sessions this week?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "generic",
                    "speaker": "assistant",
                    "content": "Use clear editing goals and a consistent schedule.",
                }
            ],
            conflicts=[],
            coverage={
                "query_type": "preference",
                "preference_constraints": [
                    {
                        "type": "session_length",
                        "label": "short session length: 30 minutes",
                        "score": 3.2,
                        "source_span_id": "pref-user",
                    }
                ],
            },
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        labels = [item["label"] for item in model_pack["preference_constraints"]]
        self.assertIn("short session length: 30 minutes", labels)
        requirements = [item["requirement"] for item in model_pack["preference_requirement_checklist"]["must_satisfy"]]
        self.assertIn("Use the explicit short-session length: short session length: 30 minutes.", requirements)

    def test_pack_for_model_extracts_generic_preference_facets(self) -> None:
        pack = EvidencePack(
            query=(
                "Can you suggest editing steps, sneaker options, document organization, patent materials, "
                "candidate choice, daily routine, family movie options, and expense tracking?"
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "editing-ai",
                    "speaker": "assistant",
                    "content": "Step 1: Initial edits with AI tools like Grammarly and ProWritingAid, then use Jasper AI tone calibration.",
                },
                {
                    "id": "sneaker-style",
                    "speaker": "assistant",
                    "content": "Choose sneakers with a sleek and modern design in neutral colors like black, white, or gray.",
                },
                {
                    "id": "will-tool",
                    "speaker": "assistant",
                    "content": "WillMaker Pro 2024.1 helps with small will changes, and electronic signatures support updates.",
                },
                {
                    "id": "patent-media",
                    "speaker": "assistant",
                    "content": "For patent materials, include detailed drawings and video demos that show the invention working.",
                },
                {
                    "id": "candidate",
                    "speaker": "assistant",
                    "content": "Douglas is known for strong organizational abilities and may be the best fit for executor responsibilities.",
                },
                {
                    "id": "routine",
                    "speaker": "assistant",
                    "content": "Create a consistent morning routine and keep important tasks at a fixed time each day.",
                },
                {
                    "id": "reviews",
                    "speaker": "assistant",
                    "content": "Pick family-friendly movies with positive family reviews and strong audience ratings.",
                },
                {
                    "id": "reading-balance",
                    "speaker": "assistant",
                    "content": "Balancing standalone novels with series gives a good mix of both types of books.",
                },
                {
                    "id": "budget-simple-tools",
                    "speaker": "assistant",
                    "content": "Use an Excel spreadsheet for expense tracking and keep full control with a customizable manual setup.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "preference"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        labels = {item["label"] for item in model_pack["preference_constraints"]}
        self.assertIn("start editing with AI-assisted tools when supported", labels)
        self.assertIn("use AI/tool support for tone calibration or tone consistency", labels)
        self.assertIn("prefer sleek, modern sneaker styling", labels)
        self.assertIn("prefer neutral colors such as black or gray", labels)
        self.assertIn("use digital will updating tools when making future document changes", labels)
        self.assertIn("prefer electronic update and filing workflows when supported", labels)
        self.assertIn("include detailed drawings", labels)
        self.assertIn("include video demos or multimedia demonstrations when supported", labels)
        self.assertIn("consider Douglas when their organizational or reliability strengths are supported", labels)
        self.assertIn("use consistent timing for recurring routine activities", labels)
        self.assertIn("prefer options with positive family or audience reviews", labels)
        self.assertIn("balance recommendations between standalone novels and series", labels)
        self.assertIn("prefer simple spreadsheet or manual tracking tools", labels)
        self.assertIn("avoid overcomplicated or specialized budgeting platforms when simple tracking is requested", labels)
        checklist = model_pack["preference_requirement_checklist"]
        must_satisfy = [item["requirement"] for item in checklist["must_satisfy"]]
        must_avoid = [item["requirement"] for item in checklist["must_avoid"]]
        self.assertIn(
            "Name the supported candidate and rationale: consider Douglas when their organizational or reliability strengths are supported.",
            must_satisfy,
        )
        self.assertIn(
            "Do not recommend this avoided option or approach: avoid overcomplicated or specialized budgeting platforms when simple tracking is requested.",
            must_avoid,
        )

    def test_value_history_summary_prioritizes_query_units(self) -> None:
        pack = EvidencePack(
            query="How many total hours have I spent studying probability basics?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "factual_exact",
                "value_history": [
                    {"value": "8 problems", "value_type": "count", "current": True, "query_overlap": 4, "context": "8 problems"},
                    {"value": "4 hours", "value_type": "duration", "current": True, "query_overlap": 4, "context": "4 hours studying"},
                ],
            },
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        summary = model_pack["value_history_summary"]
        self.assertEqual(summary["target_value_types"][0], "duration")
        self.assertEqual(summary["current_candidates"][0]["value"], "4 hours")

    def test_value_history_summary_prefers_exact_count_unit_over_generic_item(self) -> None:
        pack = EvidencePack(
            query="How many project cards are included in my gallery using Bootstrap 5.3.0?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "knowledge_update",
                "value_history": [
                    {
                        "value": "6 items",
                        "value_type": "count",
                        "current": True,
                        "query_overlap": 4,
                        "recency_rank": 1,
                        "speaker": "user",
                        "context": "I defined MVP features including a project gallery with 6 items.",
                    },
                    {
                        "value": "10 cards",
                        "value_type": "count",
                        "current": True,
                        "query_overlap": 5,
                        "speaker": "user",
                        "context": "I've added two new projects, so now I have a total of 10 cards using Bootstrap 5.3.0.",
                    },
                ],
            },
            debug_trace=[],
        )

        summary = _pack_for_model(pack)["value_history_summary"]

        self.assertEqual(summary["current_candidates"][0]["value"], "10 cards")
        self.assertEqual(summary["resolved_current_value"], "10 cards")

    def test_value_history_summary_prefers_user_current_value_over_assistant_plan_values(self) -> None:
        pack = EvidencePack(
            query="How many hours of overtime have I tracked most recently?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "knowledge_update",
                "value_history": [
                    {
                        "value": "12 hours",
                        "value_type": "duration",
                        "current": True,
                        "query_overlap": 4,
                        "recency_rank": 1,
                        "speaker": "assistant",
                        "update_marker_strength": 1.2,
                        "context": "Given that you did 12 hours of overtime in February, your target for March would be 6 hours.",
                    },
                    {
                        "value": "6 hours",
                        "value_type": "duration",
                        "current": True,
                        "query_overlap": 4,
                        "recency_rank": 1,
                        "speaker": "assistant",
                        "update_marker_strength": 1.2,
                        "context": "Your target for March would be to limit overtime to 6 hours or less.",
                    },
                    {
                        "value": "4 hours",
                        "value_type": "duration",
                        "current": True,
                        "query_overlap": 3,
                        "recency_rank": 20,
                        "speaker": "user",
                        "update_marker_strength": 1.0,
                        "context": "I've managed to get overtime down to just 4 hours in March, which is a huge accomplishment.",
                    },
                ],
            },
            debug_trace=[],
        )

        summary = _pack_for_model(pack)["value_history_summary"]

        self.assertEqual(summary["current_candidates"][0]["value"], "4 hours")
        self.assertEqual(summary["resolved_current_value"], "4 hours")

    def test_pack_for_model_aligns_history_resolved_value_to_state_transition(self) -> None:
        pack = EvidencePack(
            query="How long does the probate process usually take in Montserrat?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "knowledge_update",
                "value_history": [
                    {
                        "value": "6-9 months",
                        "value_type": "duration",
                        "current": True,
                        "query_overlap": 5,
                        "slot_overlap": 4,
                        "speaker": "assistant",
                        "history_index": 20,
                        "context": "The probate process in Montserrat typically takes 6-9 months.",
                    },
                    {
                        "value": "5-7 months",
                        "value_type": "duration",
                        "current": False,
                        "query_overlap": 5,
                        "slot_overlap": 4,
                        "speaker": "user",
                        "history_index": 40,
                        "context": "The probate process was shortened to 5-7 months after recent legal reforms.",
                    },
                ],
            },
            debug_trace=[],
        )

        summary = _pack_for_model(pack)["value_history_summary"]

        self.assertEqual(summary["resolved_current_value"], "5-7 months")
        self.assertEqual(summary["secondary_current_value"], "6-9 months")
        self.assertTrue(summary["preferred_current_candidate"]["superseded_by_state_summary"])

    def test_event_ordering_uses_project_milestone_sequence_items(self) -> None:
        pack = EvidencePack(
            query="Can you walk me through the order in which I brought up different aspects of my app development and deployment? Mention ONLY and ONLY five items.",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[
                {
                    "id": "setup",
                    "speaker": "user",
                    "timeline_index": 1,
                    "content": "I need help with the initial project setup, local server, and database schema for my app.",
                },
                {
                    "id": "crud",
                    "speaker": "user",
                    "timeline_index": 2,
                    "content": "Now I am implementing transaction CRUD endpoints including POST /transactions and create_transaction.",
                },
                {
                    "id": "deploy",
                    "speaker": "user",
                    "timeline_index": 3,
                    "content": "I need deployment configuration for Render with gunicorn, port 10000, and environment variables.",
                },
                {
                    "id": "tests",
                    "speaker": "user",
                    "timeline_index": 4,
                    "content": "I am working on integration tests and endpoint coverage for the API test suite.",
                },
                {
                    "id": "improve",
                    "speaker": "user",
                    "timeline_index": 5,
                    "content": "After deployment, I want deployment and test improvements with expanded additional tests.",
                },
            ],
            conflicts=[],
            coverage={"query_type": "event_ordering"},
            debug_trace=[],
        )

        model_pack = _pack_for_model(pack)

        labels = [item["label"] for item in model_pack["sequence_items"]]
        self.assertEqual(
            labels,
            [
                "initial project setup",
                "transaction CRUD implementation",
                "deployment configuration",
                "integration test coverage",
                "deployment and test improvements",
            ],
        )

    def test_eval_answer_model_adds_temporal_answer_candidate_pairs(self) -> None:
        pack = EvidencePack(
            query="How many days do I have between scheduling the meeting and the start of the testing period for my project?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "temporal_reasoning",
                "temporal_candidates": [
                    {
                        "source_span_id": "plan",
                        "speaker": "assistant",
                        "role": "start_date",
                        "normalized_date": "2024-03-25",
                        "context": "Week 3 (March 25 - March 31, 2024): Core Development and Initial Testing.",
                        "confidence": 0.7,
                        "query_overlap": 1,
                    },
                    {
                        "source_span_id": "meeting",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2024-03-15",
                        "context": "I'm trying to schedule a meeting for March 15, 2024, at 09:00 CET.",
                        "confidence": 0.5,
                        "query_overlap": 1,
                    },
                    {
                        "source_span_id": "deadline",
                        "speaker": "user",
                        "role": "completion_date",
                        "normalized_date": "2024-04-05",
                        "context": "The project has a deadline set for MVP completion by April 5, 2024, to allow two weeks for testing and deployment.",
                        "confidence": 0.78,
                        "query_overlap": 2,
                    },
                ],
            },
            debug_trace=[],
        )

        temporal_pairs = _pack_for_model(pack)["temporal_answer_candidates"]

        self.assertEqual(temporal_pairs[0]["start_date"], "2024-03-15")
        self.assertEqual(temporal_pairs[0]["end_date"], "2024-04-05")
        self.assertEqual(temporal_pairs[0]["day_difference"], 21)
        self.assertEqual(temporal_pairs[0]["start_label"], "meeting_date")
        self.assertEqual(temporal_pairs[0]["end_label"], "testing_or_deployment_start")

    def test_eval_answer_model_binds_temporal_endpoint_to_local_date_clause(self) -> None:
        context = (
            "I started working with Ashlee on September 1, 2026, and I met her at her office "
            "on September 10, 2026 to review the draft."
        )
        pack = EvidencePack(
            query="How many days do I have between my meeting with Ashlee and the patent response deadline?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "temporal_reasoning",
                "temporal_candidates": [
                    {
                        "source_span_id": "ashlee",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2026-09-01",
                        "context": context,
                        "confidence": 0.5,
                        "query_overlap": 2,
                    },
                    {
                        "source_span_id": "ashlee",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2026-09-10",
                        "context": context,
                        "confidence": 0.5,
                        "query_overlap": 2,
                    },
                    {
                        "source_span_id": "deadline",
                        "speaker": "user",
                        "role": "deadline_date",
                        "normalized_date": "2026-10-01",
                        "context": "The patent response deadline is due October 1, 2026.",
                        "confidence": 0.86,
                        "query_overlap": 2,
                    },
                ],
            },
            debug_trace=[],
        )

        temporal_pairs = _pack_for_model(pack)["temporal_answer_candidates"]

        self.assertEqual(temporal_pairs[0]["start_date"], "2026-09-10")
        self.assertEqual(temporal_pairs[0]["end_date"], "2026-10-01")
        self.assertEqual(temporal_pairs[0]["day_difference"], 21)

    def test_eval_answer_model_prefers_planned_event_over_later_scheduled_event(self) -> None:
        pack = EvidencePack(
            query="How many days passed between when I planned the peer review and when I completed the final code review for my project?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "temporal_reasoning",
                "temporal_candidates": [
                    {
                        "source_span_id": "planned",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2024-04-02",
                        "explicit_year": True,
                        "context": "I'm planning a peer review for April 2, 2024, focused on semantic HTML.",
                        "confidence": 0.5,
                        "query_overlap": 2,
                    },
                    {
                        "source_span_id": "scheduled",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2024-04-15",
                        "explicit_year": True,
                        "context": "I'm getting ready for the scheduled peer review on April 15, 2024.",
                        "confidence": 0.5,
                        "query_overlap": 2,
                    },
                    {
                        "source_span_id": "complete",
                        "speaker": "user",
                        "role": "completion_date",
                        "normalized_date": "2024-05-03",
                        "explicit_year": True,
                        "context": "I completed the final code review on May 3, 2024.",
                        "confidence": 0.78,
                        "query_overlap": 3,
                    },
                ],
            },
            debug_trace=[],
        )

        temporal_pairs = _pack_for_model(pack)["temporal_answer_candidates"]

        self.assertEqual(temporal_pairs[0]["start_date"], "2024-04-02")
        self.assertEqual(temporal_pairs[0]["end_date"], "2024-05-03")
        self.assertEqual(temporal_pairs[0]["day_difference"], 31)

    def test_temporal_answer_candidates_align_implicit_endpoint_year_to_explicit_endpoint(self) -> None:
        pack = EvidencePack(
            query="How many months are there between when I planned to reach my daily walking goal and the festival I’m preparing my sneaker outfit for?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "temporal_reasoning",
                "temporal_candidates": [
                    {
                        "source_span_id": "goal",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2024-04-15",
                        "explicit_year": True,
                        "context": "I planned to reach my daily walking goal by April 15, 2024.",
                        "confidence": 0.5,
                        "query_overlap": 5,
                    },
                    {
                        "source_span_id": "festival",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2026-08-22",
                        "explicit_year": False,
                        "context": "I've got a festival coming up on August 22 and need a sneaker outfit.",
                        "confidence": 0.5,
                        "query_overlap": 4,
                    },
                ],
            },
            debug_trace=[],
        )

        temporal_pairs = _pack_for_model(pack)["temporal_answer_candidates"]

        self.assertEqual(temporal_pairs[0]["start_date"], "2024-04-15")
        self.assertEqual(temporal_pairs[0]["end_date"], "2024-08-22")

    def test_temporal_completion_endpoint_can_use_range_start(self) -> None:
        pack = EvidencePack(
            query=(
                "How many days passed between when I started my 30-day editing challenge "
                "and when I completed the 15-day clarity editing challenge?"
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "temporal_reasoning",
                "temporal_candidates": [
                    {
                        "source_span_id": "start",
                        "speaker": "user",
                        "role": "start_date",
                        "normalized_date": "2026-04-02",
                        "context": "I entered a 30-day editing challenge starting April 2.",
                        "confidence": 0.7,
                        "query_overlap": 3,
                    },
                    {
                        "source_span_id": "clarity",
                        "speaker": "user",
                        "role": "start_date",
                        "normalized_date": "2026-05-10",
                        "range_endpoint": "range_start",
                        "context": "I completed that 15-day clarity editing challenge from May 10 to May 25.",
                        "confidence": 0.7,
                        "query_overlap": 4,
                    },
                    {
                        "source_span_id": "clarity",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2026-05-25",
                        "range_endpoint": "range_end",
                        "context": "I completed that 15-day clarity editing challenge from May 10 to May 25.",
                        "confidence": 0.7,
                        "query_overlap": 4,
                    },
                ],
            },
            debug_trace=[],
        )

        temporal_pairs = _pack_for_model(pack)["temporal_answer_candidates"]

        self.assertEqual(temporal_pairs[0]["start_date"], "2026-04-02")
        self.assertEqual(temporal_pairs[0]["end_date"], "2026-05-10")
        self.assertEqual(temporal_pairs[0]["day_difference"], 38)

    def test_temporal_user_goal_deadline_beats_assistant_example_deadline(self) -> None:
        pack = EvidencePack(
            query=(
                "How many days were there between when I planned to complete my prior art search "
                "and when I aimed to file my provisional patent?"
            ),
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "temporal_reasoning",
                "temporal_candidates": [
                    {
                        "source_span_id": "search",
                        "speaker": "user",
                        "role": "completion_date",
                        "normalized_date": "2024-04-10",
                        "explicit_year": True,
                        "context": "I plan to complete my prior art search by April 10, 2024.",
                        "confidence": 0.78,
                        "query_overlap": 4,
                    },
                    {
                        "source_span_id": "assistant_plan",
                        "speaker": "assistant",
                        "role": "deadline_date",
                        "normalized_date": "2024-05-02",
                        "explicit_year": True,
                        "context": "Example Timeline: Filing Deadline: File the provisional patent application by May 2, 2024.",
                        "confidence": 0.86,
                        "query_overlap": 4,
                    },
                    {
                        "source_span_id": "user_goal",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2024-05-15",
                        "explicit_year": True,
                        "context": "This affects my decision to file a provisional patent by May 15, 2024.",
                        "confidence": 0.5,
                        "query_overlap": 3,
                    },
                ],
            },
            debug_trace=[],
        )

        temporal_pairs = _pack_for_model(pack)["temporal_answer_candidates"]

        self.assertEqual(temporal_pairs[0]["start_date"], "2024-04-10")
        self.assertEqual(temporal_pairs[0]["end_date"], "2024-05-15")
        self.assertEqual(temporal_pairs[0]["day_difference"], 35)

    def test_eval_answer_model_does_not_deterministically_answer_generic_temporal_pair(self) -> None:
        class RecordingClient:
            timeout_seconds = 15.0

            def __init__(self) -> None:
                self.calls = 0

            def structured(self, prompt, schema, input):
                self.calls += 1
                return {"answer": "model answer"}

        pack = EvidencePack(
            query="How many days are there between when my friend suggested using AI and my upcoming webinar?",
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage={
                "query_type": "temporal_reasoning",
                "temporal_candidates": [
                    {
                        "source_span_id": "start",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2026-03-01",
                        "context": "Carla suggested using AI over lunch on March 1.",
                        "confidence": 0.5,
                        "query_overlap": 5,
                    },
                    {
                        "source_span_id": "wrong",
                        "speaker": "user",
                        "role": "mentioned_date",
                        "normalized_date": "2026-07-11",
                        "context": "The AI budget was approved on July 11.",
                        "confidence": 0.5,
                        "query_overlap": 4,
                    },
                ],
            },
            debug_trace=[],
        )
        client = RecordingClient()

        answer = OpenAICompatibleAnswerModel(client).answer_with_context(pack.query, pack, benchmark="BEAM", category="temporal_reasoning")

        self.assertEqual(answer, "model answer")
        self.assertEqual(client.calls, 1)

    def test_deterministic_temporal_answer_allows_tiny_margin_rounding_error(self) -> None:
        answer = _deterministic_temporal_answer(
            "How many days passed between when I planned the peer review and when I completed the final code review for my project?",
            {
                "temporal_answer_candidates": [
                    {
                        "start_date": "2024-04-02",
                        "end_date": "2024-05-03",
                        "day_difference": 31,
                        "confidence": 0.97,
                        "start_label": "planned_event_date",
                        "end_label": "completion_date",
                        "score": 16.255,
                        "start_context": "I planned the peer review for April 2, 2024.",
                        "end_context": "I completed the final code review on May 3, 2024.",
                    },
                    {
                        "start_date": "2024-04-01",
                        "end_date": "2024-05-03",
                        "day_difference": 32,
                        "confidence": 0.97,
                        "start_label": "planned_event_date",
                        "end_label": "completion_date",
                        "score": 15.505000000000003,
                    },
                ]
            },
        )

        self.assertEqual(answer, "31 days, from 2024-04-02 to 2024-05-03.")

    def test_deterministic_temporal_answer_allows_direct_generic_event_pair_with_strong_margin(self) -> None:
        answer = _deterministic_temporal_answer(
            "How many days are there between when my friend Carla suggested using AI for hiring over lunch and my upcoming webinar on AI ethics in hiring?",
            {
                "temporal_answer_candidates": [
                    {
                        "start_date": "2026-03-01",
                        "end_date": "2026-03-20",
                        "day_difference": 19,
                        "confidence": 0.70,
                        "start_label": "start_event",
                        "end_label": "event_date",
                        "score": 17.7,
                        "start_context": "My friend Carla suggested using AI for hiring over lunch on March 1.",
                        "end_context": "I registered for the webinar on AI ethics in hiring scheduled for March 20.",
                    },
                    {
                        "start_date": "2026-02-20",
                        "end_date": "2026-03-20",
                        "day_difference": 28,
                        "confidence": 0.70,
                        "start_label": "start_event",
                        "end_label": "event_date",
                        "score": 16.012,
                    },
                ]
            },
        )

        self.assertEqual(answer, "19 days, from 2026-03-01 to 2026-03-20.")

    def test_deterministic_temporal_answer_keeps_ambiguous_endpoint_block(self) -> None:
        answer = _deterministic_temporal_answer(
            "How many days are there between when my friend suggested using AI and my upcoming webinar?",
            {
                "temporal_answer_candidates": [
                    {
                        "start_date": "2026-03-01",
                        "end_date": "2026-03-20",
                        "day_difference": 19,
                        "confidence": 0.70,
                        "start_label": "start_event",
                        "end_label": "event_date",
                        "score": 17.7,
                        "start_context": "My friend suggested using AI on March 1.",
                        "end_context": "The webinar on AI is scheduled for March 20.",
                    },
                    {
                        "start_date": "2026-03-01",
                        "end_date": "2026-04-10",
                        "day_difference": 40,
                        "confidence": 0.70,
                        "start_label": "start_event",
                        "end_label": "event_date",
                        "score": 15.9,
                        "start_context": "My friend suggested using AI on March 1.",
                        "end_context": "The webinar on AI moved to April 10.",
                    },
                ]
            },
        )

        self.assertIsNone(answer)

    def test_cli_benchmark_accepts_eval_model_endpoints(self) -> None:
        with FakeModelServer() as server, tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = _write_official_beam_fixture(tmp_path)
            db = tmp_path / "fm.sqlite3"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fusion_memory.cli",
                    "--db",
                    str(db),
                    "--workspace-id",
                    "w",
                    "--user-id",
                    "u",
                    "--agent-id",
                    "a",
                    "run-beam",
                    str(dataset),
                    "--split",
                    "small",
                    "--answer-endpoint",
                    server.url("/answer"),
                    "--answer-model",
                    "answer-model",
                    "--judge-endpoint",
                    server.url("/judge"),
                    "--judge-model",
                    "judge-model",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )
            data = json.loads(proc.stdout)
            self.assertGreaterEqual(data["report"]["accuracy"], 1.0)
            self.assertEqual(data["report"]["llm_calls_query"], 2.0)
            self.assertIn("answer-model", data["report"]["answer_model"])
            self.assertIn("judge-model", data["report"]["judge_model"])

    def test_cli_benchmark_accepts_eval_model_env(self) -> None:
        with FakeModelServer() as server, tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = _write_official_beam_fixture(tmp_path)
            db = tmp_path / "fm.sqlite3"
            env = {
                **{key: value for key, value in os.environ.items() if not key.startswith("FUSION_MEMORY_")},
                "FUSION_MEMORY_EVAL_BASE_URL": server.url(""),
                "FUSION_MEMORY_EVAL_ANSWER_MODEL": "env-answer-model",
                "FUSION_MEMORY_EVAL_JUDGE_MODEL": "env-judge-model",
            }
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fusion_memory.cli",
                    "--db",
                    str(db),
                    "--workspace-id",
                    "w",
                    "--user-id",
                    "u",
                    "--agent-id",
                    "a",
                    "run-beam",
                    str(dataset),
                    "--split",
                    "small",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            data = json.loads(proc.stdout)
            self.assertGreaterEqual(data["report"]["accuracy"], 1.0)
            self.assertEqual(data["report"]["llm_calls_query"], 2.0)
            self.assertIn("env-answer-model", data["report"]["answer_model"])
            self.assertIn("env-judge-model", data["report"]["judge_model"])
            self.assertTrue(all(request["path"] == "/chat/completions" for request in server.requests if "json" in request))


class FakeModelServer:
    def __enter__(self) -> "FakeModelServer":
        self.requests: list[dict[str, Any]] = []
        requests_ref = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler_self) -> None:
                length = int(handler_self.headers.get("Content-Length", "0"))
                payload = json.loads(handler_self.rfile.read(length).decode("utf-8"))
                requests_ref.append({"path": handler_self.path, "json": payload})
                if handler_self.path == "/rate-limit-once" and sum(1 for request in requests_ref if request["path"] == "/rate-limit-once") == 1:
                    body = json.dumps({"error": "rate limited"}).encode("utf-8")
                    handler_self.send_response(429)
                    handler_self.send_header("Content-Type", "application/json")
                    handler_self.send_header("Retry-After", "0")
                    handler_self.send_header("Content-Length", str(len(body)))
                    handler_self.end_headers()
                    handler_self.wfile.write(body)
                    return
                if handler_self.path == "/empty-then-stream":
                    if payload.get("stream"):
                        body = (
                            'data: {"choices":[{"delta":{"content":"{\\"answer\\":"}}]}\n\n'
                            'data: {"choices":[{"delta":{"content":"\\"4\\"}"}}]}\n\n'
                            "data: [DONE]\n\n"
                        ).encode("utf-8")
                    else:
                        body = (
                            'data: {"object":"chat.completion.chunk","choices":[],"usage":{"completion_tokens":0}}\n\n'
                            "data: [DONE]\n\n"
                        ).encode("utf-8")
                    handler_self.send_response(200)
                    handler_self.send_header("Content-Type", "text/event-stream")
                    handler_self.send_header("Content-Length", str(len(body)))
                    handler_self.end_headers()
                    handler_self.wfile.write(body)
                    return
                if handler_self.path in {"/llm", "/chat/completions", "/rate-limit-once"}:
                    data = _decode_model_payload(payload)
                    request_input = data["input"]
                    if "evidence_pack" in request_input:
                        response = {
                            "choices": [{"message": {"content": json.dumps({"answer": "Qdrant"})}}],
                            "usage": {"total_tokens": 7},
                        }
                    elif "rubric_item" in request_input:
                        rubric_item = str(request_input["rubric_item"]).lower()
                        answer = str(request_input.get("candidate_answer", "")).lower()
                        score = 1.0 if "qdrant" in rubric_item and "qdrant" in answer else 0.0
                        response = {
                            "choices": [{"message": {"content": json.dumps({"score": score, "reason": "fixture rubric score"})}}],
                            "usage": {"total_tokens": 5},
                        }
                    elif "candidate_answer" in request_input:
                        answer = request_input["candidate_answer"]
                        response = {
                            "choices": [{"message": {"content": json.dumps({"matched": "qdrant" in answer.lower()})}}],
                            "usage": {"total_tokens": 5},
                        }
                    elif "deterministic_intent" in request_input:
                        response = {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "intent": {
                                                    "language": "zh",
                                                    "answer_shape": "unordered_list",
                                                    "evidence_scope": "multi_session",
                                                    "speaker_scope": "user",
                                                    "target_terms": ["权限控制", "登录保护"],
                                                    "object_types": ["security_feature"],
                                                    "temporal": {
                                                        "requires_time": False,
                                                        "requires_order": False,
                                                        "requires_duration": False,
                                                        "order_direction": "unknown",
                                                        "endpoint_roles": [],
                                                        "time_expressions": [],
                                                    },
                                                    "aggregation": {
                                                        "operation": "count_distinct",
                                                        "distinct": True,
                                                        "target_terms": ["security_feature"],
                                                        "unit_terms": [],
                                                    },
                                                    "needs_current_state": False,
                                                    "needs_conflict_check": False,
                                                    "confidence": 0.88,
                                                    "route_reasons": ["fixture_router"],
                                                }
                                            }
                                        )
                                    }
                                }
                            ],
                            "usage": {"total_tokens": 9},
                        }
                    else:
                        source_id = request_input["spans"][0]["span_id"]
                        response = {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            {
                                                "facts": [
                                                    {
                                                        "text": "User prefers PostgreSQL for reports.",
                                                        "subject": "user",
                                                        "predicate": "prefers",
                                                        "object": "PostgreSQL for reports",
                                                        "category": "preference",
                                                        "confidence": 0.91,
                                                        "salience": 0.84,
                                                        "source_span_ids": [source_id],
                                                    }
                                                ]
                                            }
                                        )
                                    }
                                }
                            ],
                            "usage": {"total_tokens": 42},
                        }
                elif handler_self.path == "/embed":
                    texts = payload["input"]
                    response = {"embeddings": [_embedding(text) for text in texts], "usage": {"total_tokens": len(texts)}}
                elif handler_self.path == "/rerank":
                    docs = payload["documents"]
                    response = {"scores": [float(index + 1) for index, _ in enumerate(docs)]}
                elif handler_self.path == "/dashscope-rerank":
                    response = {
                        "results": [
                            {"index": 1, "relevance_score": 0.12},
                            {"index": 0, "relevance_score": 0.91},
                        ],
                        "usage": {"total_tokens": 2},
                    }
                elif handler_self.path == "/dashscope-output-rerank":
                    response = {
                        "output": {
                            "results": [
                                {"index": 1, "relevance_score": 0.12},
                                {"index": 0, "relevance_score": 0.91},
                            ]
                        },
                        "usage": {"total_tokens": 2},
                    }
                elif handler_self.path == "/answer":
                    response = {
                        "choices": [{"message": {"content": json.dumps({"answer": "Qdrant"})}}],
                        "usage": {"total_tokens": 7},
                    }
                elif handler_self.path == "/judge":
                    data = _decode_model_payload(payload)
                    request_input = data["input"]
                    if "rubric_item" in request_input:
                        rubric_item = str(request_input["rubric_item"]).lower()
                        answer = str(request_input.get("candidate_answer", "")).lower()
                        score = 1.0 if "qdrant" in rubric_item and "qdrant" in answer else 0.0
                        response = {
                            "choices": [{"message": {"content": json.dumps({"score": score, "reason": "fixture rubric score"})}}],
                            "usage": {"total_tokens": 5},
                        }
                    else:
                        answer = request_input["candidate_answer"]
                        response = {
                            "choices": [{"message": {"content": json.dumps({"matched": "qdrant" in answer.lower()})}}],
                            "usage": {"total_tokens": 5},
                        }
                else:
                    response = {"error": "unknown path"}
                body = json.dumps(response).encode("utf-8")
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "application/json")
                handler_self.send_header("Content-Length", str(len(body)))
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                return None

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    def url(self, path: str) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}{path}"


def _embedding(text: str) -> list[float]:
    qdrant = 1.0 if "qdrant" in text.lower() else 0.0
    atlas = 1.0 if "atlas" in text.lower() else 0.0
    length = min(1.0, len(text.split()) / 20)
    return [qdrant, atlas, length]


def _decode_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload["messages"][-1]["content"]
    if isinstance(content, str) and "\n" in content:
        content = content.split("\n", 1)[1]
    return json.loads(content)


def _write_official_beam_fixture(base: Path) -> Path:
    chat_dir = base / "chats" / "100K" / "1"
    questions_dir = chat_dir / "probing_questions"
    questions_dir.mkdir(parents=True)
    (chat_dir / "chat.json").write_text(
        json.dumps(
            [
                {
                    "batch_number": 1,
                    "turns": [
                        [
                            {
                                "role": "user",
                                "id": 1,
                                "time_anchor": "March-15-2024",
                                "content": "Atlas retrieval uses Qdrant.",
                            }
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (questions_dir / "probing_questions.json").write_text(
        json.dumps(
            {
                "information_extraction": [
                    {
                        "question": "What does Atlas retrieval use?",
                        "answer": "Qdrant",
                        "rubric": ["LLM response should contain: Qdrant"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return base


if __name__ == "__main__":
    unittest.main()
