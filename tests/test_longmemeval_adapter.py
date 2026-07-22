from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fusion_memory import MemoryService, Scope
from fusion_memory.core.llm import StaticLLMClient
from fusion_memory.eval.longmemeval_adapter import LongMemEvalAdapter, load_longmemeval_dataset
from fusion_memory.eval.model_adapters import OpenAICompatibleAnswerModel, OpenAICompatibleJudgeModel


class LongMemEvalAdapterTests(unittest.TestCase):
    def test_longmemeval_adapter_runs_official_shape_and_tracks_answer_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_longmemeval_fixture(Path(tmp), split="dev")
            service = MemoryService()
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            adapter = LongMemEvalAdapter(service, scope, split="dev")

            items = adapter.load_items(dataset, split="dev")
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].question_id, "q1")
            self.assertEqual(items[0].answer_session_ids, ["s_answer"])

            output = adapter.run_dataset(dataset, split="dev", ablate=True)
            report = output["report"]

            self.assertEqual(output["ingest"]["benchmark"], "LongMemEval")
            self.assertEqual(output["ingest"]["haystack_sessions"], 3)
            self.assertEqual(report["benchmark"], "LongMemEval")
            self.assertEqual(report["split"], "dev")
            self.assertGreaterEqual(report["retrieval_match_rate"], 0.5)
            self.assertGreaterEqual(report["answer_session_hit_rate"], 0.5)
            self.assertIn("single-session-user", report["question_type_mapping"])
            self.assertIsNotNone(report["abstention_accuracy"])
            answer = next(item for item in report["answers"] if item["question_id"] == "q1")
            self.assertIn("s_answer", answer["retrieved_session_ids"])
            self.assertEqual(set(output["ablation"]), {"retrieval_modes", "components"})

    def test_loader_accepts_single_json_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "longmemeval.json"
            path.write_text(json.dumps([_longmem_record("q1")]), encoding="utf-8")
            items = load_longmemeval_dataset(path, split="dev")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].haystack_session_ids, ["s_answer", "s_distractor"])

    def test_longmemeval_reports_injected_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_longmemeval_fixture(Path(tmp), split="dev")
            service = MemoryService()
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            adapter = LongMemEvalAdapter(
                service,
                scope,
                split="dev",
                answer_model=OpenAICompatibleAnswerModel(StaticLLMClient({"answer": "Qdrant"})),
                judge_model=OpenAICompatibleJudgeModel(StaticLLMClient({"matched": True})),
            )

            output = adapter.run_dataset(dataset, split="dev")

            self.assertGreater(output["report"]["llm_calls_query"], 0.0)
            self.assertIn("llm_answer", output["report"]["answer_model"])
            self.assertIn("llm_judge", output["report"]["judge_model"])

    def test_longmemeval_defaults_to_balanced_cross_session_retrieval(self) -> None:
        class CaptureService(MemoryService):
            def __init__(self) -> None:
                super().__init__()
                self.answer_context_budgets: list[dict[str, Any]] = []

            def answer_context(self, query, scope, budget=None):
                self.answer_context_budgets.append(dict(budget or {}))
                return super().answer_context(query, scope, budget=budget)

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_longmemeval_fixture(Path(tmp), split="dev")
            service = CaptureService()
            adapter = LongMemEvalAdapter(service, Scope(user_id="u"), split="dev")
            item = adapter.load_items(dataset, split="dev")[0]

            adapter.answer_item(item)

        self.assertEqual(
            service.answer_context_budgets[0],
            {"mode": "balanced", "allow_cross_session": True},
        )
        self.assertNotIn("query_type_hint", service.answer_context_budgets[0])

    def test_longmemeval_ablation_uses_only_product_modes(self) -> None:
        class CaptureAdapter(LongMemEvalAdapter):
            def __init__(self) -> None:
                super().__init__(MemoryService(), Scope(user_id="u"), split="dev")
                self.budgets: list[dict[str, Any]] = []

            def run_items(self, items, budget=None):
                self.budgets.append(dict(budget or {}))
                return []

        adapter = CaptureAdapter()

        report = adapter.run_ablation([])

        self.assertEqual(set(report), {"fast", "balanced"})
        self.assertEqual(adapter.budgets, [{"mode": "fast"}, {"mode": "balanced"}])


def _write_longmemeval_fixture(base: Path, split: str) -> Path:
    (base / f"{split}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_longmem_record("q1")),
                json.dumps(
                    {
                        "question_id": "q_abs",
                        "question_type": "single-session-user_abs",
                        "question": "What is the user's Kubernetes cluster name?",
                        "answer": "Not mentioned.",
                        "question_date": "2026-06-04T00:00:00+00:00",
                        "haystack_session_ids": ["s_no_answer"],
                        "haystack_dates": ["2026-06-03T00:00:00+00:00"],
                        "haystack_sessions": [
                            [
                                {"role": "user", "content": "I prefer PostgreSQL for reports."},
                                {"role": "assistant", "content": "I will use that for report examples."},
                            ]
                        ],
                        "answer_session_ids": [],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    return base


def _longmem_record(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "What does Atlas retrieval use?",
        "answer": "Qdrant",
        "question_date": "2026-06-03T00:00:00+00:00",
        "haystack_session_ids": ["s_answer", "s_distractor"],
        "haystack_dates": ["2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Atlas retrieval now uses Qdrant."},
                {"role": "assistant", "content": "Noted."},
            ],
            [
                {"role": "user", "content": "Reports should use PostgreSQL."},
                {"role": "assistant", "content": "Understood."},
            ],
        ],
        "answer_session_ids": ["s_answer"],
    }


if __name__ == "__main__":
    unittest.main()
