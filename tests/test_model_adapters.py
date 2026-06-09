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
from typing import Any
from unittest.mock import patch

from fusion_memory import MemoryService, Scope
from fusion_memory.core.config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from fusion_memory.core.embedding import DeterministicEmbedder, HTTPEmbeddingClient, Qwen3EmbeddingClient
from fusion_memory.core.llm import OpenAICompatibleLLMClient
from fusion_memory.core.runtime_config import memory_service_from_env
from fusion_memory.eval.adapter import BenchmarkAdapter, EvalDocument, EvalQuery
from fusion_memory.eval.model_adapters import OpenAICompatibleAnswerModel, OpenAICompatibleJudgeModel
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor
from fusion_memory.retrieval.reranker import HTTPReranker, Qwen3Reranker


class ModelAdapterTests(unittest.TestCase):
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

    def test_runtime_config_wires_http_model_adapters_from_env(self) -> None:
        with FakeModelServer() as server:
            env = {
                "FUSION_MEMORY_EMBEDDING_PROVIDER": "http",
                "FUSION_MEMORY_EMBEDDING_ENDPOINT": server.url("/embed"),
                "FUSION_MEMORY_EMBEDDING_MODEL": "env-embed",
                "FUSION_MEMORY_RERANKER_PROVIDER": "http",
                "FUSION_MEMORY_RERANKER_ENDPOINT": server.url("/rerank"),
                "FUSION_MEMORY_RERANKER_MODEL": "env-rerank",
                "FUSION_MEMORY_EXTRACTOR_ENDPOINT": server.url("/llm"),
                "FUSION_MEMORY_EXTRACTOR_MODEL": "env-extractor",
            }
            with patch.dict(os.environ, env, clear=True):
                memory = memory_service_from_env(":memory:")
                scope = Scope(workspace_id="w", user_id="u", agent_id="a")
                memory.add("Please remember reports should use PostgreSQL.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
                result = memory.search("PostgreSQL reports", scope, options={"mode": "balanced"})
                memory.close()

            self.assertTrue(result.candidates)
            paths = [request["path"] for request in server.requests]
            self.assertIn("/llm", paths)
            self.assertIn("/embed", paths)
            self.assertIn("/rerank", paths)
            self.assertTrue(any(request["json"].get("model") == "env-embed" for request in server.requests))
            self.assertTrue(any(request["json"].get("model") == "env-rerank" for request in server.requests))
            self.assertTrue(any(request["json"].get("model") == "env-extractor" for request in server.requests))

    def test_runtime_config_accepts_extractor_base_url(self) -> None:
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

            self.assertTrue(any(request["path"] == "/chat/completions" for request in server.requests))

    def test_qwen_defaults_are_configured_without_required_runtime_dependency(self) -> None:
        self.assertEqual(len(DeterministicEmbedder().embed_text("Atlas")), DEFAULT_EMBEDDING_DIMENSION)
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(DEFAULT_RERANKER_MODEL, "Qwen/Qwen3-Reranker-0.6B")
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

    def test_cli_benchmark_accepts_eval_model_endpoints(self) -> None:
        with FakeModelServer() as server, tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "dataset.json"
            db = tmp_path / "fm.sqlite3"
            dataset.write_text(
                json.dumps(
                    {
                        "documents": [{"id": "doc1", "content": "Atlas retrieval uses Qdrant.", "speaker": "user"}],
                        "queries": [{"id": "q1", "query": "What does Atlas retrieval use?", "gold_answers": ["Qdrant"]}],
                    }
                ),
                encoding="utf-8",
            )
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
                    "run-benchmark",
                    str(dataset),
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
            self.assertEqual(data["report"]["answer_match_rate"], 1.0)
            self.assertEqual(data["report"]["llm_calls_query"], 2.0)
            self.assertIn("answer-model", data["report"]["answer_model"])
            self.assertIn("judge-model", data["report"]["judge_model"])


class FakeModelServer:
    def __enter__(self) -> "FakeModelServer":
        self.requests: list[dict[str, Any]] = []
        requests_ref = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler_self) -> None:
                length = int(handler_self.headers.get("Content-Length", "0"))
                payload = json.loads(handler_self.rfile.read(length).decode("utf-8"))
                requests_ref.append({"path": handler_self.path, "json": payload})
                if handler_self.path in {"/llm", "/chat/completions"}:
                    span_id = payload["messages"][1]["content"]
                    data = json.loads(span_id)
                    source_id = data["input"]["spans"][0]["span_id"]
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
                elif handler_self.path == "/answer":
                    response = {
                        "choices": [{"message": {"content": json.dumps({"answer": "Qdrant"})}}],
                        "usage": {"total_tokens": 7},
                    }
                elif handler_self.path == "/judge":
                    data = json.loads(payload["messages"][1]["content"])
                    answer = data["input"]["candidate_answer"]
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


if __name__ == "__main__":
    unittest.main()
