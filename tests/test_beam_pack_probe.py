from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools.beam_pack_probe as probe
from tools.beam_pack_probe import _compact_mapping, _selected_ids


class BeamPackProbeTests(unittest.TestCase):
    def test_selected_ids_accepts_csv_file_and_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text("q2,q3 # batch\n\nq4\n", encoding="utf-8")

            ids = _selected_ids("q1,q2", str(path))

        self.assertEqual(ids, ["q1", "q2", "q3", "q4"])

    def test_compact_mapping_bounds_large_values(self) -> None:
        compact = _compact_mapping(
            {
                "small": "ok",
                "items": list(range(20)),
                "nested": {str(index): index for index in range(20)},
                "later": "x",
            },
            max_items=3,
        )

        self.assertEqual(compact["small"], "ok")
        self.assertEqual(compact["items"], list(range(12)))
        self.assertEqual(len(compact["nested"]), 12)
        self.assertTrue(compact["_truncated"])
        self.assertNotIn("later", compact)

    def test_build_probe_uses_eval_engine_for_beam_category(self) -> None:
        query = SimpleNamespace(
            id="beam:100k:1:information_extraction:0",
            query="What memory applies?",
            category="information_extraction",
        )
        pack = SimpleNamespace(
            facts=[],
            events=[],
            source_spans=[],
            current_views=[],
            entity_profiles=[],
            coverage={"query_type": "information_extraction"},
        )
        query_scope = SimpleNamespace(session_id="beam:100k:1")
        retrieval_engine = SimpleNamespace(answer_context=MagicMock(return_value=pack))
        adapter = SimpleNamespace(
            retrieval_engine=retrieval_engine,
            _beam_scope=MagicMock(return_value=query_scope),
        )
        service = SimpleNamespace(
            close=MagicMock(),
            answer_context=MagicMock(side_effect=AssertionError("production answer_context must not be called")),
        )
        args = SimpleNamespace(
            dataset="/unused",
            split="100k",
            workspace="w",
            user_id="u",
            agent_id="a",
            run_id=None,
            session_id=None,
            db="sqlite://unused",
            query_ids=query.id,
            query_ids_file=None,
            include_full_pack=False,
        )

        with patch.object(probe, "_load_official_beam_dataset", return_value=(None, [query])), patch.object(
            probe, "memory_service_from_env", return_value=service
        ), patch.object(
            probe, "BeamAdapter", return_value=adapter
        ), patch.object(
            probe, "_pack_for_model", return_value={}
        ):
            report = probe.build_probe(args)

        self.assertEqual(report["query_count"], 1)
        retrieval_engine.answer_context.assert_called_once_with(
            "What memory applies?",
            query_scope,
            "information_extraction",
            budget={"mode": "balanced"},
        )
        service.answer_context.assert_not_called()


if __name__ == "__main__":
    unittest.main()
