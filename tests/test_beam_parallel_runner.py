from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "beam_parallel_runner.py"
    spec = importlib.util.spec_from_file_location("beam_parallel_runner_test_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load runner module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BeamParallelRunnerResumeTests(unittest.TestCase):
    def test_default_dataset_is_environment_or_relative_path(self) -> None:
        runner = _load_runner_module()
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(runner._default_beam_dataset(), "datasets/BEAM")
        with patch.dict("os.environ", {"BEAM_DATASET": "/data/BEAM"}, clear=True):
            self.assertEqual(runner._default_beam_dataset(), "/data/BEAM")

    def test_resume_loads_all_worker_partials_and_prefers_completed_records(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            partial_dir = Path(tmp)
            (partial_dir / "worker_0.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "q1", "answer_failed": False, "score": 1.0}),
                        json.dumps({"query_id": "q2", "answer_failed": True, "score": 0.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (partial_dir / "worker_1.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "q2", "answer_failed": False, "score": 0.8}),
                        json.dumps({"query_id": "q3", "answer_failed": True, "score": 0.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = runner._load_resumable_partial_records(partial_dir, {"q1", "q2", "q3"})

        self.assertEqual({record["query_id"] for record in records}, {"q1", "q2", "q3"})
        self.assertEqual(runner._completed_query_ids(records), {"q1", "q2"})
        q2 = next(record for record in records if record["query_id"] == "q2")
        self.assertFalse(q2["answer_failed"])
        self.assertEqual(q2["score"], 0.8)

    def test_completed_retry_record_wins_even_when_older_failure_is_in_later_worker_file(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            partial_dir = Path(tmp)
            (partial_dir / "worker_0.jsonl").write_text(
                json.dumps({"query_id": "q1", "answer_failed": False, "score": 1.0}) + "\n",
                encoding="utf-8",
            )
            (partial_dir / "worker_9.jsonl").write_text(
                json.dumps({"query_id": "q1", "answer_failed": True, "score": 0.0}) + "\n",
                encoding="utf-8",
            )

            records = runner._load_resumable_partial_records(partial_dir, {"q1"})

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["query_id"], "q1")
        self.assertFalse(records[0]["answer_failed"])
        self.assertEqual(records[0]["score"], 1.0)

    def test_merge_and_order_preserve_selected_query_sequence(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            partial_dir = Path(tmp)
            (partial_dir / "worker_0.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "q2", "answer_failed": False, "score": 0.8}),
                        json.dumps({"query_id": "q4", "answer_failed": False, "score": 0.3}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (partial_dir / "worker_1.jsonl").write_text(
                json.dumps({"query_id": "q1", "answer_failed": False, "score": 1.0}) + "\n",
                encoding="utf-8",
            )

            records = runner._load_resumable_partial_records(partial_dir, {"q1", "q2", "q4"})
            order = {qid: idx for idx, qid in enumerate(["q1", "q2", "q4"])}
            records.sort(key=lambda item: order.get(item["query_id"], 10**9))

        self.assertEqual([record["query_id"] for record in records], ["q1", "q2", "q4"])

    def test_completed_query_ids_ignore_worker_failure_sentinels(self) -> None:
        runner = _load_runner_module()
        records = [
            {"query_id": "q1", "answer_failed": False, "score": 1.0},
            {"worker_failed": True, "error": "boom"},
            {"query_id": "q2", "answer_failed": True, "score": 0.0},
            {"query_id": "q3", "judge_reason": "rubric scoring failed after retries: timeout", "score": 0.0},
        ]
        self.assertEqual(runner._completed_query_ids(records), {"q1"})
        self.assertEqual(
            runner._merge_partial_records(records),
            [
                {"query_id": "q1", "answer_failed": False, "score": 1.0},
                {"query_id": "q2", "answer_failed": True, "score": 0.0},
                {"query_id": "q3", "judge_reason": "rubric scoring failed after retries: timeout", "score": 0.0},
            ],
        )

    def test_is_completed_record_requires_successful_answer(self) -> None:
        runner = _load_runner_module()
        self.assertTrue(runner._is_completed_record({"query_id": "q1", "answer_failed": False}))
        self.assertFalse(runner._is_completed_record({"query_id": "q2", "answer_failed": True}))
        self.assertFalse(runner._is_completed_record({"query_id": "q3", "judge_failed": True}))
        self.assertFalse(
            runner._is_completed_record({"query_id": "q4", "judge_reason": "rubric scoring failed after retries: 429"})
        )
        self.assertFalse(runner._is_completed_record({"worker_failed": True, "error": "boom"}))

    def test_retryable_query_ids_include_answer_failed_and_missing_queries(self) -> None:
        runner = _load_runner_module()
        records = [
            {"query_id": "q1", "answer_failed": False, "score": 1.0},
            {"query_id": "q2", "answer_failed": True, "score": 0.0},
        ]
        queries = [
            SimpleNamespace(id="q1"),
            SimpleNamespace(id="q2"),
            SimpleNamespace(id="q3"),
        ]

        self.assertEqual(runner._retryable_query_ids(records, queries), {"q2", "q3"})
        self.assertFalse(runner._all_queries_completed(records, queries))

        completed_records = [
            {"query_id": "q1", "answer_failed": False, "score": 1.0},
            {"query_id": "q2", "answer_failed": False, "score": 0.5},
            {"query_id": "q3", "answer_failed": False, "score": 0.0},
        ]
        self.assertTrue(runner._all_queries_completed(completed_records, queries))

    def test_from_result_can_select_only_answer_failed_queries(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "report": {
                            "answers": [
                                {"query_id": "q1", "category": "a", "score": 0.0, "answer_failed": False},
                                {"query_id": "q2", "category": "a", "score": 0.0, "answer_failed": True},
                                {"query_id": "q3", "category": "b", "score": 1.0, "answer_failed": True},
                                {"query_id": "q4", "category": "b", "score": 0.25, "answer_failed": False},
                                {"query_id": "q5", "category": "b", "score": 0.0, "judge_failed": True},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            ids = runner._query_ids_from_result(
                str(result_path),
                score_lt=None,
                per_category=None,
                answer_failed_only=True,
            )

        self.assertEqual(ids, ["q2", "q3", "q5"])

    def test_query_ids_file_accepts_newline_comma_and_comments(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text("q1\nq2,q3  # grouped retry\n\n# comment\nq4\n", encoding="utf-8")

            ids = runner._ids_from_file(str(path))

        self.assertEqual(ids, ["q1", "q2", "q3", "q4"])

    def test_from_result_low_score_selection_still_includes_answer_failures(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "report": {
                            "answers": [
                                {"query_id": "q1", "category": "a", "score": 0.75, "answer_failed": False},
                                {"query_id": "q2", "category": "a", "score": 0.0, "answer_failed": True},
                                {"query_id": "q3", "category": "b", "score": 0.25, "answer_failed": False},
                                {"query_id": "q4", "category": "b", "score": 0.0, "judge_reason": "rubric scoring failed after retries: timeout"},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            ids = runner._query_ids_from_result(
                str(result_path),
                score_lt=0.5,
                per_category=None,
                answer_failed_only=False,
            )

        self.assertEqual(ids, ["q2", "q3", "q4"])

    def test_from_result_falls_back_to_partials_when_report_answers_empty(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps({"report": {"answers": []}}), encoding="utf-8")
            partial_dir = result_path.with_suffix(".partials")
            partial_dir.mkdir()
            (partial_dir / "worker_0.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "q1", "category": "a", "score": 1.0, "answer_failed": False}),
                        json.dumps({"query_id": "q2", "category": "a", "score": 0.0, "answer_failed": True}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ids = runner._query_ids_from_result(
                str(result_path),
                score_lt=None,
                per_category=None,
                answer_failed_only=True,
            )

        self.assertEqual(ids, ["q2"])

    def test_resume_skips_malformed_partial_jsonl_lines(self) -> None:
        runner = _load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            partial_dir = Path(tmp)
            (partial_dir / "worker_0.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "q1", "answer_failed": False, "score": 1.0}),
                        '{"query_id": "q_bad", ',
                        json.dumps({"query_id": "q2", "answer_failed": True, "score": 0.0}),
                        json.dumps({"query_id": "q3", "judge_reason": "rubric scoring failed after retries: timeout", "score": 0.0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = runner._load_resumable_partial_records(partial_dir, {"q1", "q2", "q3", "q_bad"})

        self.assertEqual([record["query_id"] for record in records], ["q1", "q2", "q3"])
        self.assertEqual(runner._completed_query_ids(records), {"q1"})

    def test_runner_repo_root_points_to_memory_repo(self) -> None:
        runner = _load_runner_module()
        self.assertEqual(runner.REPO_ROOT.name, "memory")
        self.assertTrue((runner.REPO_ROOT / "fusion_memory").is_dir())

    def test_query_chunk_uses_eval_engine_for_full_pack(self) -> None:
        runner = _load_runner_module()
        query_scope = SimpleNamespace(session_id="beam:100k:1")
        pack = SimpleNamespace()
        retrieval_engine = SimpleNamespace(answer_context=MagicMock(return_value=pack))
        result = SimpleNamespace(query_id="q1", answer_failed=False, score=1.0)
        adapter = SimpleNamespace(
            retrieval_engine=retrieval_engine,
            _beam_scope=MagicMock(return_value=query_scope),
            answer_query=MagicMock(return_value=result),
        )
        service = SimpleNamespace(
            close=MagicMock(),
            answer_context=MagicMock(side_effect=AssertionError("production answer_context must not be called")),
        )

        def jsonable(value):
            if value is result:
                return {"query_id": "q1", "answer_failed": False, "score": 1.0}
            if value is pack:
                return {"query": "full pack"}
            raise AssertionError(f"unexpected jsonable value: {value!r}")

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            runner, "_memory_service", return_value=service
        ), patch.object(
            runner, "_build_eval_models", return_value=(SimpleNamespace(), SimpleNamespace())
        ), patch.object(
            runner, "BeamAdapter", return_value=adapter
        ), patch.object(
            runner, "_load_resumable_partial_records", return_value=[]
        ), patch.object(
            runner, "_append_partial_record"
        ), patch.object(
            runner, "_jsonable", side_effect=jsonable
        ):
            records = runner._query_chunk(
                [
                    {
                        "id": "q1",
                        "query": "What memory applies?",
                        "category": "information_extraction",
                    }
                ],
                "/unused",
                "100k",
                "w",
                "u",
                "a",
                None,
                None,
                "sqlite://unused",
                {},
                0,
                tmp,
                True,
                3,
                None,
            )

        self.assertEqual(records[0]["full_evidence_pack"], {"query": "full pack"})
        retrieval_engine.answer_context.assert_called_once_with(
            "What memory applies?",
            query_scope,
            "information_extraction",
            budget={"mode": "balanced"},
        )
        service.answer_context.assert_not_called()

    def test_query_help_exposes_retry_and_llm_aggregation_switches(self) -> None:
        path = Path(__file__).resolve().parents[1] / "tools" / "beam_parallel_runner.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(path),
                "--workspace",
                "w",
                "--output",
                "/tmp/out.json",
                "query",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("--answer-failure-retries", proc.stdout)
        self.assertIn("--use-llm-aggregation", proc.stdout)

    def test_runtime_runner_shim_uses_maintained_runner(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".runtime" / "beam_tools" / "beam_parallel_runner.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(path),
                "--workspace",
                "w",
                "--output",
                "/tmp/out.json",
                "query",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("--answer-failure-retries", proc.stdout)
        self.assertIn("--answer-failed-only", proc.stdout)


if __name__ == "__main__":
    unittest.main()
