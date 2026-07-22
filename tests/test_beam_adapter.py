from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fusion_memory import MemoryService, Scope
from fusion_memory.core.models import EvidencePack
from fusion_memory.eval.adapter import BenchmarkAdapter, EvalQuery
from fusion_memory.eval.beam.engine import BeamRetrievalEngine
from fusion_memory.eval.beam.query_planner import BeamQueryPlanner
from fusion_memory.eval.beam_adapter import BeamAdapter, _event_ordering_score
from fusion_memory.retrieval.context import (
    ProductQueryPlan,
    ProviderKind,
    RetrievalResult,
)


@dataclass(frozen=True)
class BeamEngineCall:
    query: str
    scope: Scope
    category: str | None
    budget: dict[str, Any]


class CaptureBeamEngine:
    def __init__(self) -> None:
        self.calls: list[BeamEngineCall] = []

    def answer_context(
        self,
        query: str,
        scope: Scope,
        category: str | None,
        budget: dict[str, Any] | None = None,
    ) -> EvidencePack:
        self.calls.append(BeamEngineCall(query, scope, category, dict(budget or {})))
        return EvidencePack(
            query=query,
            answer_policy="answer_with_evidence_or_abstain",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[{"id": "span-1", "content": "There is contradictory information."}],
            conflicts=[],
            coverage={"query_type": category},
            debug_trace=[],
        )


class CaptureProductEngine:
    def __init__(self) -> None:
        self.calls = []

    def search_with_plan(self, context, request, plan):
        self.calls.append((context, request, plan))
        return RetrievalResult(
            candidates=(),
            coverage={"degraded": False},
            trace={"stages": ["selection"]},
            plan=plan,
        )


class CapturePackBuilder:
    def __init__(self) -> None:
        self.calls = []

    def build(self, context, request, result, token_budget):
        self.calls.append((context, request, result, token_budget))
        return EvidencePack(
            query=request.query,
            answer_policy="abstain_if_not_supported",
            current_views=[],
            entity_profiles=[],
            facts=[],
            events=[],
            source_spans=[],
            conflicts=[],
            coverage=dict(result.coverage),
            debug_trace=[dict(result.trace)],
        )


class BeamAdapterTests(unittest.TestCase):
    def test_beam_query_planner_decorates_product_plan_by_category(self) -> None:
        expected = {
            "event_ordering": {ProviderKind.CHRONOLOGY, ProviderKind.TEMPORAL},
            "temporal_reasoning": {ProviderKind.TEMPORAL},
            "contradiction_resolution": {ProviderKind.LEXICAL, ProviderKind.VECTOR},
            "knowledge_update": {ProviderKind.TEMPORAL, ProviderKind.LEXICAL},
            "multi_session_reasoning": {ProviderKind.LEXICAL, ProviderKind.VECTOR},
            "preference_following": {ProviderKind.LEXICAL, ProviderKind.ENTITY},
            "instruction_following": {ProviderKind.LEXICAL},
            "information_extraction": {ProviderKind.LEXICAL, ProviderKind.VECTOR},
            "summarization": {ProviderKind.LEXICAL, ProviderKind.VECTOR},
            "abstention": {ProviderKind.LEXICAL, ProviderKind.VECTOR},
        }
        planner = BeamQueryPlanner()

        for category, required_providers in expected.items():
            with self.subTest(category=category):
                plan = planner.plan("What memory applies?", category, 7)
                planned_providers = {request.kind for request in plan.provider_requests}

                self.assertIsInstance(plan, ProductQueryPlan)
                self.assertTrue(required_providers.issubset(planned_providers))
                self.assertTrue(plan.use_reranker)
                self.assertFalse(hasattr(plan, "category"))
                self.assertNotIn(category, repr(plan))

    def test_beam_retrieval_engine_applies_eval_limits_after_product_pack(self) -> None:
        product_engine = CaptureProductEngine()
        pack_builder = CapturePackBuilder()
        engine = BeamRetrievalEngine(
            product_engine=product_engine,
            pack_builder=pack_builder,
        )
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="beam:100k:1")

        pack = engine.answer_context(
            "What happened first?",
            scope,
            "event_ordering",
            {"limit": 3, "token_budget": 1200},
        )

        context, request, plan = product_engine.calls[0]
        self.assertTrue(context.include_session)
        self.assertEqual(request.limit, 24)
        self.assertEqual(request.mode, "balanced")
        self.assertEqual(pack_builder.calls[0][3], 24000)
        self.assertIs(pack_builder.calls[0][2].plan, plan)
        self.assertNotIn("event_ordering", repr(plan))
        self.assertNotIn("event_ordering", repr(pack.debug_trace))
        self.assertEqual(pack.coverage["benchmark"], "BEAM")
        self.assertEqual(pack.coverage["benchmark_category"], "event_ordering")
        self.assertEqual(pack.coverage["query_type"], "event_ordering")

    def test_beam_retrieval_engine_uses_fifty_result_floor_outside_ordering(self) -> None:
        product_engine = CaptureProductEngine()
        engine = BeamRetrievalEngine(
            product_engine=product_engine,
            pack_builder=CapturePackBuilder(),
        )

        engine.answer_context(
            "What does Atlas use?",
            Scope(user_id="u", session_id="beam:100k:1"),
            "information_extraction",
            {"limit": 12, "token_budget": 30000},
        )

        self.assertEqual(product_engine.calls[0][1].limit, 50)

    def test_beam_adapter_loads_official_chat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_official_beam_fixture(Path(tmp))
            adapter = BeamAdapter(MemoryService(), Scope(workspace_id="w", user_id="u", agent_id="a"), split="small")

            ingest = adapter.ingest_dataset(dataset, split="small")
            queries = adapter.build_queries(dataset, split="small")

            self.assertEqual(ingest["documents"], 2)
            self.assertEqual(len(queries), 3)
            self.assertEqual(queries[0].category, "information_extraction")
            self.assertIn("Qdrant", queries[0].gold_answers[0])
            instruction_query = next(query for query in queries if query.category == "instruction_following")
            self.assertTrue(instruction_query.gold_answers)
            self.assertIn("syntax highlighting", instruction_query.gold_answers[0])

    def test_beam_adapter_runs_split_and_records_answer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_official_beam_fixture(Path(tmp))
            service = MemoryService()
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            adapter = BeamAdapter(service, scope, split="small")

            output = adapter.run_dataset(dataset, split="small", ablate=True)
            report = output["report"]

            self.assertEqual(output["ingest"]["benchmark"], "BEAM")
            self.assertEqual(report["benchmark"], "BEAM")
            self.assertEqual(report["split"], "small")
            self.assertIn("scoring", report)
            self.assertIn("judge_failures", report)
            self.assertIn("information_extraction", report["query_type_mapping"])
            self.assertEqual(report["evidence_pack_trace_coverage"], 1.0)
            self.assertTrue(report["answers"][0]["evidence_pack"]["source_span_ids"])
            self.assertIn("source_span_quota_met", report["answers"][0])
            self.assertIn("coverage_insufficient", report["answers"][0])
            self.assertIn("answer_model", report["answers"][0])
            self.assertIn("judge_model", report["answers"][0])
            self.assertIn("llm_calls", report["answers"][0])
            self.assertEqual(set(output["ablation"]), {"retrieval_modes"})

    def test_beam_adapter_passes_category_context_to_answer_model(self) -> None:
        class ContextAnswer:
            version = "context-answer"

            def __init__(self) -> None:
                self.calls = []

            def answer_with_context(self, query, pack, *, benchmark=None, category=None, metadata=None):
                self.calls.append({"benchmark": benchmark, "category": category, "metadata": metadata})
                return "Qdrant"

        class AlwaysMatchJudge:
            version = "always-match"

            def score(self, answer, gold_answers):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_official_beam_fixture(Path(tmp))
            answer_model = ContextAnswer()
            service = MemoryService()
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            adapter = BeamAdapter(service, scope, split="small", answer_model=answer_model, judge_model=AlwaysMatchJudge())
            adapter.ingest_dataset(dataset, split="small")
            query = next(item for item in adapter.build_queries(dataset, split="small") if item.category == "instruction_following")

            adapter.answer_query(query)

        self.assertEqual(answer_model.calls[0]["benchmark"], "BEAM")
        self.assertEqual(answer_model.calls[0]["category"], "instruction_following")
        self.assertEqual(answer_model.calls[0]["metadata"], {})

    def test_beam_adapter_routes_category_only_to_eval_engine(self) -> None:
        class CaptureService(MemoryService):
            def answer_context(self, query, scope, budget=None):
                raise AssertionError("BeamAdapter must use the eval-owned retrieval engine")

        class StaticAnswer:
            version = "static-answer"

            def answer_with_context(self, query, pack, *, benchmark=None, category=None, metadata=None):
                return "There is contradictory information."

        class AlwaysMatchJudge:
            version = "always-match"

            def rubric_score(self, query, answer, rubric_item):
                return 1.0, "ok"

        service = CaptureService()
        engine = CaptureBeamEngine()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        adapter = BeamAdapter(
            service,
            scope,
            split="100k",
            answer_model=StaticAnswer(),
            judge_model=AlwaysMatchJudge(),
            retrieval_engine=engine,
        )
        result = adapter.answer_query(
            EvalQuery(
                id="beam:100k:1:contradiction_resolution:0",
                query="Have I used Excel for tracking expenses?",
                gold_answers=["There is contradictory information."],
                category="contradiction_resolution",
                metadata={"rubric": ["LLM response should contain: There is contradictory information."]},
            )
        )

        self.assertEqual(result.query_type, "contradiction_resolution")
        self.assertEqual(engine.calls[0].category, "contradiction_resolution")
        self.assertEqual(engine.calls[0].scope.session_id, "beam:100k:1")
        self.assertNotIn("query_type_hint", repr(engine.calls[0]))

    def test_generic_ablation_uses_only_product_modes(self) -> None:
        class CaptureAdapter(BenchmarkAdapter):
            def __init__(self) -> None:
                super().__init__(MemoryService(), Scope(user_id="u"))
                self.budgets: list[dict[str, Any]] = []

            def run_queries(self, queries, budget=None):
                self.budgets.append(dict(budget or {}))
                return []

        adapter = CaptureAdapter()

        report = adapter.run_ablation([])

        self.assertEqual(set(report), {"fast", "balanced"})
        self.assertEqual(adapter.budgets, [{"mode": "fast"}, {"mode": "balanced"}])

    def test_beam_adapter_scopes_official_queries_to_their_chat_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_official_beam_fixture(Path(tmp), include_second_chat=True)
            service = MemoryService()
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            adapter = BeamAdapter(service, scope, split="100k")
            adapter.ingest_dataset(dataset, split="100k")
            query = next(
                item
                for item in adapter.build_queries(dataset, split="100k")
                if item.id == "beam:100k:1:information_extraction:0"
            )

            result = adapter.answer_query(query)

        self.assertTrue(result.retrieved_source_span_ids)
        spans = [service.get(span_id, "span") for span_id in result.retrieved_source_span_ids]
        self.assertTrue(all(span and span.scope.session_id == "beam:100k:1" for span in spans))
        self.assertFalse(any(span and span.scope.session_id == "beam:100k:2" for span in spans))

    def test_beam_adapter_reports_answer_model_failures(self) -> None:
        class FailingAnswer:
            version = "failing-answer"

            def answer_with_context(self, query, pack, *, benchmark=None, category=None, metadata=None):
                raise RuntimeError("LLM endpoint returned HTTP 429: rate limited")

        class NeverCalledJudge:
            version = "never-called"

            def rubric_score(self, query, answer, rubric_item):
                raise AssertionError("judge should not be called when answer generation fails")

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _write_official_beam_fixture(Path(tmp))
            service = MemoryService()
            scope = Scope(workspace_id="w", user_id="u", agent_id="a")
            adapter = BeamAdapter(service, scope, split="small", answer_model=FailingAnswer(), judge_model=NeverCalledJudge())
            adapter.ingest_dataset(dataset, split="small")
            query = adapter.build_queries(dataset, split="small")[0]

            result = adapter.answer_query(query)
            report = adapter.report([result])

        self.assertTrue(result.answer_failed)
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.matched_gold)
        self.assertIn("answer generation failed", result.judge_reason)
        self.assertEqual(report["answer_failures"]["count"], 1)
        self.assertEqual(report["judge_failures"]["count"], 0)

    def test_event_ordering_score_aligns_ordinals_and_descriptive_items(self) -> None:
        reference = [
            "1st: Core functionality",
            "2nd: Transaction error handling",
            "3rd: Security and deployment",
        ]
        system = [
            "Core functionality: planning the Flask app and SQLite schema.",
            "Transaction error handling: implementing validation and error handling.",
            "Security and deployment: adding password hashing before deployment.",
        ]

        self.assertEqual(_event_ordering_score(reference, system), 1.0)

    def test_event_ordering_score_does_not_overmatch_short_labels(self) -> None:
        reference = [
            "1st: Core functionality",
            "2nd: Transaction error handling",
            "3rd: Security and deployment",
        ]
        system = [
            "Core functionality",
            "Transaction error handling",
            "Security",
        ]

        self.assertLess(_event_ordering_score(reference, system), 1.0)

    def test_cli_run_beam_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
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
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )
            data = json.loads(proc.stdout)
            self.assertEqual(data["report"]["benchmark"], "BEAM")
            self.assertEqual(data["report"]["split"], "small")
            self.assertIn("accuracy", data["report"])
            self.assertNotIn("retrieval_match_rate", data["report"])


def _write_official_beam_fixture(base: Path, *, include_second_chat: bool = False) -> Path:
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
                                "content": "I prefer Qdrant for Atlas retrieval.",
                            },
                            {
                                "role": "assistant",
                                "id": 2,
                                "content": "Noted that Atlas retrieval should use Qdrant.",
                            },
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
                    }
                ],
                "abstention": [
                    {
                        "question": "What database was never mentioned?",
                        "ideal_response": "The chat does not mention that database.",
                        "rubric": [
                            "LLM response should contain: The chat does not mention that database."
                        ],
                    }
                ],
                "instruction_following": [
                    {
                        "question": "Could you show me how to implement a login feature?",
                        "instruction_being_tested": "Always format all code snippets with syntax highlighting when I ask about implementation details.",
                        "rubric": [
                            "LLM response should contain: code blocks with syntax highlighting"
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if include_second_chat:
        chat2_dir = base / "chats" / "100K" / "2"
        questions2_dir = chat2_dir / "probing_questions"
        questions2_dir.mkdir(parents=True)
        (chat2_dir / "chat.json").write_text(
            json.dumps(
                [
                    {
                        "batch_number": 1,
                        "turns": [
                            [
                                {
                                    "role": "user",
                                    "id": 1,
                                    "time_anchor": "March-16-2024",
                                    "content": "I prefer Pinecone for Atlas retrieval.",
                                }
                            ]
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        (questions2_dir / "probing_questions.json").write_text(
            json.dumps(
                {
                    "information_extraction": [
                        {
                            "question": "What does Atlas retrieval use?",
                            "answer": "Pinecone",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    return base


if __name__ == "__main__":
    unittest.main()
