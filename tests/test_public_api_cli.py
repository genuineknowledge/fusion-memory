from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fusion_memory import MemoryService, Scope


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class PublicApiCliTests(unittest.TestCase):
    def test_views_profiles_refresh_and_getters(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a")
        first = memory.add("I prefer Qdrant for Atlas retrieval.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Always give concise technical answers.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("Please keep concise answers but include implementation tradeoffs.", scope, ts("2026-06-03T10:00:00+00:00"))

        views = memory.get_current_views(scope, view_type="current_preferences")
        self.assertTrue(views)
        self.assertIn("Qdrant", views[0].text)

        refreshed_views = memory.refresh_current_views(scope, affected_fact_ids=first.accepted_fact_ids)
        self.assertTrue(refreshed_views)
        self.assertTrue(set(first.accepted_fact_ids).intersection(refreshed_views[0].source_fact_ids))

        profiles = memory.get_entity_profile("u", scope, profile_type="communication_style")
        self.assertTrue(profiles)
        self.assertGreaterEqual(profiles[0].support_count, 2)

        refreshed_profiles = memory.refresh_entity_profiles(scope, affected_entity_ids=["u"])
        self.assertTrue(refreshed_profiles)
        self.assertEqual(refreshed_profiles[0].entity_id, "u")

    def test_timeline_and_event_compare(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w", user_id="u", agent_id="a", session_id="s")
        memory.add("I tested BM25 yesterday.", scope, ts("2026-06-03T12:00:00+00:00"))
        memory.add("I added dense retrieval today.", scope, ts("2026-06-05T12:00:00+00:00"))

        events = memory.timeline(None, scope)
        self.assertGreaterEqual(len(events), 2)
        self.assertLess(events[0].time_start, events[1].time_start)

        dense_events = memory.timeline("dense", scope, start="2026-06-05T00:00:00+00:00")
        self.assertEqual(len(dense_events), 1)
        self.assertIn("dense retrieval", dense_events[0].description)

        comparison = memory.compare_events(events[0].event_id, events[1].event_id)
        self.assertEqual(comparison["relation"], "before")
        self.assertIn(comparison["basis"], {"event_edge", "time_start"})

        reverse = memory.compare_events(events[1], events[0])
        self.assertEqual(reverse["relation"], "after")

    def test_cli_public_commands_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fm.sqlite3"
            add_preference = self._run_cli(
                db,
                [
                    "add",
                    "I prefer Qdrant for Atlas retrieval.",
                    "--time",
                    "2026-06-01T10:00:00+00:00",
                ],
            )
            self._run_cli(
                db,
                [
                    "add",
                    "Always give concise technical answers.",
                    "--time",
                    "2026-06-02T10:00:00+00:00",
                ],
            )
            self._run_cli(
                db,
                [
                    "add",
                    "Please keep concise answers but include implementation tradeoffs.",
                    "--time",
                    "2026-06-03T10:00:00+00:00",
                ],
            )
            self._run_cli(
                db,
                [
                    "add",
                    "I tested BM25 yesterday.",
                    "--time",
                    "2026-06-04T10:00:00+00:00",
                ],
            )

            span_id = add_preference["span_ids"][0]
            fact_id = add_preference["accepted_fact_ids"][0]
            trace_id = add_preference["trace_id"]

            span = self._run_cli(db, ["get", span_id, "--type", "span"])
            self.assertEqual(span["span_id"], span_id)

            fact = self._run_cli(db, ["get", fact_id, "--type", "fact"])
            self.assertEqual(fact["fact_id"], fact_id)

            history = self._run_cli(db, ["history", "--entity", "Qdrant"])
            self.assertTrue(history["facts"])

            trace = self._run_cli(db, ["debug-trace", trace_id])
            self.assertEqual(trace["operation"], "add")

            audit = self._run_cli(db, ["audit", "--type", "memory.add"])
            self.assertTrue(audit)
            self.assertTrue(any(event["event_type"] == "memory.add" and event["trace_id"] == trace_id for event in audit))

            timeline = self._run_cli(db, ["timeline", "--entity", "BM25"])
            self.assertTrue(timeline)
            self.assertIn("BM25", timeline[0]["description"])

            views = self._run_cli(db, ["views", "--type", "current_preferences"])
            self.assertTrue(views)
            self.assertIn("Qdrant", views[0]["text"])

            profiles = self._run_cli(db, ["profiles", "u", "--type", "communication_style"])
            self.assertTrue(profiles)
            self.assertGreaterEqual(profiles[0]["support_count"], 2)

            encoding_report = self._run_cli(db, ["report", "encoding"])
            self.assertGreater(encoding_report["total"], 0)

            profile_report = self._run_cli(db, ["report", "profiles"])
            self.assertGreaterEqual(profile_report["source_coverage"], 0.0)

            for content in [
                "Atlas uses Qdrant for retrieval.",
                "Reports use PostgreSQL.",
            ]:
                self._run_cli(
                    db,
                    [
                        "add",
                        content,
                        "--time",
                        "2026-06-05T10:00:00+00:00",
                        ],
                    )

            pending_tasks = self._run_cli(db, ["tasks", "--status", "pending"])
            self.assertEqual(len(pending_tasks), 1)
            self.assertEqual(pending_tasks[0]["task_type"], "refresh_session_summary")

            processed = self._run_cli(db, ["tasks", "--process", "--limit", "5"])
            self.assertEqual(processed["processed_count"], 1)
            self.assertEqual(processed["status_counts"], {"succeeded": 1})

            summary = self._run_cli(db, ["summaries", "--refresh"])
            self.assertEqual(summary["span_type"], "summary")
            self.assertIn("Atlas", summary["content"])

            summaries = self._run_cli(db, ["summaries"])
            self.assertTrue(any(item["span_id"] == summary["span_id"] for item in summaries))

    def _run_cli(self, db: Path, args: list[str]):
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
                "--session-id",
                "s",
                *args,
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            text=True,
            capture_output=True,
        )
        return json.loads(proc.stdout)


if __name__ == "__main__":
    unittest.main()
