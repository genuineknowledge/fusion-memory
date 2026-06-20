from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.rule_audit import build_rule_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


class RuleAuditTests(unittest.TestCase):
    def test_build_rule_audit_counts_hits_contribution_and_drops(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {"rule_id": "current_value.stale_history_marker", "contributed_candidate_id": "c1", "stage": "filter"}
                ],
                "paths": {"hybrid": {"sources": ["l3_current_view"]}},
                "coverage": {"dropped_high_signal_candidates": [{"candidate_id": "c1"}]},
            },
            {
                "query_id": "q2",
                "rule_hits": [
                    {"rule_id": "current_value.stale_history_marker", "contributed_candidate_id": None, "stage": "filter"}
                ],
                "paths": {"hybrid": {"sources": []}},
                "coverage": {},
            },
        ]

        audit = build_rule_audit(records)

        row = next(item for item in audit if item["rule_id"] == "current_value.stale_history_marker")
        self.assertEqual(row["hit_count"], 2)
        self.assertEqual(row["contribution_count"], 1)
        self.assertEqual(row["dropped_count"], 1)

    def test_build_rule_audit_combines_top_level_and_nested_rule_hits(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {"rule_id": "rule.top_only", "contributed_candidate_id": "c1"},
                    {"rule_id": "rule.duplicate", "contributed_candidate_id": "c2"},
                ],
                "coverage": {
                    "rule_hits": [
                        {"rule_id": "rule.nested_only", "contributed_candidate_id": "c3"},
                        {"rule_id": "rule.duplicate", "contributed_candidate_id": "c2"},
                    ],
                    "dropped_high_signal_candidates": [{"candidate_id": "c3"}],
                },
                "paths": {"hybrid": {"sources": ["source_a"]}},
            }
        ]

        audit = build_rule_audit(records)

        self.assertEqual([row["rule_id"] for row in audit], ["rule.duplicate", "rule.nested_only", "rule.top_only"])
        duplicate_row = next(item for item in audit if item["rule_id"] == "rule.duplicate")
        self.assertEqual(duplicate_row["hit_count"], 1)
        self.assertEqual(duplicate_row["contribution_count"], 1)

        nested_row = next(item for item in audit if item["rule_id"] == "rule.nested_only")
        self.assertEqual(nested_row["hit_count"], 1)
        self.assertEqual(nested_row["contribution_count"], 1)
        self.assertEqual(nested_row["dropped_count"], 1)

    def test_build_rule_audit_reads_nested_rule_hits_and_recommendations(self) -> None:
        records = [
            {
                "query_id": "q1",
                "coverage": {
                    "rule_hits": [
                        {
                            "rule_id": "event_ordering.taxonomy.domain_label_match",
                            "contributed_candidate_id": "c1",
                            "metadata": {"category": "taxonomy_candidate"},
                        },
                        {
                            "rule_id": "event_ordering.legacy.tie_breaker",
                            "contributed_candidate_id": None,
                        },
                    ],
                    "dropped_high_signal_candidates": [{"candidate_id": "c1"}],
                },
                "paths": {"hybrid": {"sources": ["l3_current_view", "taxonomy"]}},
            }
        ]

        audit = build_rule_audit(records)

        taxonomy_row = next(item for item in audit if item["rule_id"] == "event_ordering.taxonomy.domain_label_match")
        self.assertEqual(taxonomy_row["query_count"], 1)
        self.assertEqual(taxonomy_row["candidate_sources"], ["l3_current_view", "taxonomy"])
        self.assertEqual(taxonomy_row["recommendation"], "migrate_to_taxonomy")
        self.assertEqual(taxonomy_row["cleanup_phase"], "first_pass")
        self.assertEqual(taxonomy_row["cleanup_action"], "migrate_to_taxonomy")
        self.assertFalse(taxonomy_row["safe_to_delete"])

        legacy_row = next(item for item in audit if item["rule_id"] == "event_ordering.legacy.tie_breaker")
        self.assertEqual(legacy_row["recommendation"], "legacy_shadow")
        self.assertEqual(legacy_row["cleanup_action"], "keep_shadow")
        self.assertFalse(legacy_row["safe_to_delete"])

    def test_build_rule_audit_marks_event_ordering_legacy_rules_as_legacy_shadow(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {"rule_id": "event_ordering.legacy.tie_breaker", "contributed_candidate_id": "c9"},
                    {"rule_id": "event_ordering.legacy.unused", "contributed_candidate_id": None},
                ],
                "paths": {"hybrid": {"sources": ["timeline"]}},
                "coverage": {},
            }
        ]

        audit = build_rule_audit(records)

        legacy_row = next(item for item in audit if item["rule_id"] == "event_ordering.legacy.tie_breaker")
        self.assertEqual(legacy_row["recommendation"], "legacy_shadow")
        self.assertEqual(legacy_row["cleanup_action"], "keep_shadow")
        unused_row = next(item for item in audit if item["rule_id"] == "event_ordering.legacy.unused")
        self.assertEqual(unused_row["recommendation"], "legacy_shadow")
        self.assertEqual(unused_row["cleanup_action"], "keep_shadow")
        self.assertFalse(unused_row["safe_to_delete"])

    def test_rule_audit_marks_duplicate_no_contribution_rules_for_first_pass_cleanup(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [{"rule_id": "rule.alpha", "contributed_candidate_id": "c1", "stage": "filter"}],
                "coverage": {},
                "paths": {"hybrid": {"sources": ["s"]}},
            },
            {
                "query_id": "q2",
                "rule_hits": [
                    {
                        "rule_id": "rule.alpha_duplicate",
                        "contributed_candidate_id": None,
                        "stage": "filter",
                        "metadata": {"duplicate_of": "rule.alpha"},
                    }
                ],
                "coverage": {},
                "paths": {"hybrid": {"sources": []}},
            },
        ]

        audit = build_rule_audit(records)
        duplicate = next(row for row in audit if row["rule_id"] == "rule.alpha_duplicate")

        self.assertEqual(duplicate["duplicate_of"], "rule.alpha")
        self.assertEqual(duplicate["cleanup_phase"], "first_pass")
        self.assertEqual(duplicate["cleanup_action"], "delete_duplicate")
        self.assertTrue(duplicate["safe_to_delete"])

    def test_cli_writes_deterministic_json_and_csv_for_object_input(self) -> None:
        payload = {
            "records": [
                {
                    "query_id": "q2",
                    "rule_hits": [
                        {"rule_id": "rule.beta", "contributed_candidate_id": None},
                        {"rule_id": "rule.alpha", "contributed_candidate_id": "c1"},
                    ],
                    "coverage": {"dropped_high_signal_candidates": [{"candidate_id": "c1"}]},
                    "paths": {"hybrid": {"sources": ["source_b", "source_a"]}},
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "replay.json"
            output_path = tmp_path / "audit.json"
            csv_path = tmp_path / "audit.csv"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/rule_audit.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--csv",
                    str(csv_path),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            audit = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["rule_id"] for row in audit], ["rule.alpha", "rule.beta"])
            self.assertEqual(audit[0]["candidate_sources"], ["source_a", "source_b"])
            self.assertEqual(audit[0]["recommendation"], "keep")
            self.assertEqual(audit[1]["recommendation"], "delete_candidate")

            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "rule_id",
                        "ability",
                        "hit_count",
                        "query_count",
                        "contribution_count",
                        "negative_impact_count",
                        "dropped_count",
                        "candidate_sources",
                        "evidence_inputs",
                        "recommendation",
                        "duplicate_of",
                        "cleanup_phase",
                        "cleanup_action",
                        "safe_to_delete",
                    ],
                )
                rows = list(reader)

            self.assertEqual([row["rule_id"] for row in rows], ["rule.alpha", "rule.beta"])
            self.assertEqual(rows[0]["candidate_sources"], "source_a;source_b")
            self.assertEqual(rows[0]["evidence_inputs"], str(input_path))

    def test_cli_merges_multiple_replay_inputs_and_marks_evidence_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            first = tmp / "event.json"
            second = tmp / "current.json"
            out = tmp / "audit.json"
            first.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "query_id": "q1",
                                "rule_hits": [{"rule_id": "rule.keep", "contributed_candidate_id": "c1"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "query_id": "q2",
                                "rule_hits": [{"rule_id": "rule.drop", "contributed_candidate_id": None}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "tools/rule_audit.py", "--input", str(first), "--input", str(second), "--output", str(out)],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual({row["rule_id"] for row in rows}, {"rule.keep", "rule.drop"})
            self.assertTrue(all(row["evidence_inputs"] for row in rows))

    def test_cli_can_write_json_without_csv(self) -> None:
        payload = {
            "records": [
                {
                    "query_id": "q-json",
                    "rule_hits": [{"rule_id": "rule.alpha", "contributed_candidate_id": "c1"}],
                    "coverage": {},
                    "paths": {"hybrid": {"sources": ["source_a"]}},
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "replay.json"
            output_path = tmp_path / "audit.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/rule_audit.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            audit = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["rule_id"] for row in audit], ["rule.alpha"])

    def test_cli_writes_deterministic_json_and_csv_for_top_level_list_input(self) -> None:
        payload = [
            {
                "query_id": "q3",
                "rule_hits": [
                    {"rule_id": "rule.delta", "contributed_candidate_id": "c4"},
                ],
                "coverage": {
                    "rule_hits": [
                        {"rule_id": "rule.gamma", "contributed_candidate_id": None},
                    ],
                    "dropped_high_signal_candidates": [{"candidate_id": "c4"}],
                },
                "paths": {"hybrid": {"sources": ["source_c"]}},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "replay.json"
            output_path = tmp_path / "audit.json"
            csv_path = tmp_path / "audit.csv"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/rule_audit.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--csv",
                    str(csv_path),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            audit = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["rule_id"] for row in audit], ["rule.delta", "rule.gamma"])
            self.assertEqual(audit[0]["dropped_count"], 1)
            self.assertEqual(audit[1]["recommendation"], "delete_candidate")

            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["rule_id"] for row in rows], ["rule.delta", "rule.gamma"])
            self.assertEqual(rows[0]["candidate_sources"], "source_c")
            self.assertEqual(rows[0]["evidence_inputs"], str(input_path))


if __name__ == "__main__":
    unittest.main()
