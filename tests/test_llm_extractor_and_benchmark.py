from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fusion_memory import MemoryService, Scope
from fusion_memory.eval.adapter import BenchmarkAdapter, load_dataset
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor


class LLMExtractorAndBenchmarkTests(unittest.TestCase):
    def test_structured_llm_extractor_can_be_injected(self) -> None:
        class EchoSourceClient:
            def __init__(self) -> None:
                self.calls = []

            def structured(self, prompt, schema, input):
                self.calls.append({"prompt": prompt, "schema": schema, "input": input})
                span_id = input["spans"][0]["span_id"]
                return {
                    "facts": [
                        {
                            "local_id": "f0",
                            "text": "User prefers PostgreSQL for reports.",
                            "subject": "user",
                            "predicate": "prefers",
                            "object": "PostgreSQL for reports",
                            "category": "preference",
                            "confidence": 0.91,
                            "salience": 0.84,
                            "source_span_ids": [span_id],
                        }
                    ]
                }

        client = EchoSourceClient()
        extractor = StructuredLLMExtractor(client)
        memory = MemoryService(extractor=extractor)
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        result = memory.add("Please remember reports should use PostgreSQL.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
        self.assertTrue(result.accepted_fact_ids)
        trace = memory.debug_trace(result.trace_id)
        self.assertIn("structured_llm_extractor", str(trace))
        self.assertTrue(client.calls)

    def test_dataset_loader_and_benchmark_report_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "id": "doc1",
                                "content": "User said Atlas now uses Qdrant.",
                                "timestamp": "2026-06-01T00:00:00+00:00",
                                "speaker": "user",
                            }
                        ],
                        "queries": [
                            {
                                "id": "q1",
                                "query": "What does Atlas use?",
                                "gold_answers": ["Qdrant"],
                                "category": "factual_exact",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            docs, queries = load_dataset(path)
            self.assertEqual(len(docs), 1)
            self.assertEqual(len(queries), 1)
            memory = MemoryService()
            adapter = BenchmarkAdapter(memory, Scope(workspace_id="w", user_id="u", agent_id="a"))
            ingest = adapter.ingest_dataset(path)
            results = adapter.run_queries(adapter.build_queries(path))
            report = adapter.report(results)
            self.assertEqual(ingest["documents"], 1)
            self.assertEqual(report["retrieval_match_rate"], 1.0)
            self.assertIn("factual_exact", report["by_category"])

    def test_cli_run_benchmark_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "dataset.json"
            db = tmp_path / "fm.sqlite3"
            dataset.write_text(
                json.dumps(
                    {
                        "documents": [{"id": "doc1", "content": "User prefers Qdrant for Atlas.", "speaker": "user"}],
                        "queries": [{"id": "q1", "query": "What does user prefer for Atlas?", "gold_answers": ["Qdrant"]}],
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
                ],
                cwd="/home/wwb/fusion-memory",
                check=True,
                text=True,
                capture_output=True,
            )
            data = json.loads(proc.stdout)
            self.assertEqual(data["report"]["retrieval_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
