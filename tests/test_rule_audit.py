from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tools.rule_audit as rule_audit
from tools.rule_audit import build_rule_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


class RuleAuditTests(unittest.TestCase):
    def _build_provider_audit(self, records: list[dict[str, object]]) -> list[dict[str, object]]:
        builder = getattr(rule_audit, "build_provider_audit", None)
        self.assertIsNotNone(builder, "build_provider_audit() must be exported")
        return builder(records)

    def test_build_provider_audit_aggregates_top_level_and_coverage_provider_summary(self) -> None:
        records = [
            {
                "query_id": "q1",
                "_audit_input": "first.json",
                "pipeline_trace": {
                    "pipeline_layers": {
                        "CandidateRecall": {
                            "provider_summary": [
                                {
                                    "provider_id": "raw_provider",
                                    "source_family": "raw_provider",
                                    "output_count": 2,
                                    "output_source_counts": {"l0_raw_hybrid": 2},
                                    "production_default": True,
                                    "shadow_only": False,
                                    "graph_related": False,
                                }
                            ]
                        }
                    }
                },
            },
            {
                "query_id": "q1",
                "_audit_input": "second.json",
                "coverage": {
                    "pipeline_trace": {
                        "pipeline_layers": {
                            "CandidateRecall": {
                                "provider_summary": [
                                    {
                                        "provider_id": "raw_provider",
                                        "source_family": "raw_provider",
                                        "output_count": 3,
                                        "output_source_counts": {"l0_raw_hybrid": 1, "raw_span": 2},
                                        "production_default": True,
                                        "shadow_only": False,
                                        "graph_related": False,
                                    }
                                ]
                            }
                        }
                    }
                },
            },
            {
                "query_id": "q2",
                "_audit_input": "second.json",
                "coverage": {
                    "pipeline_trace": {
                        "pipeline_layers": {
                            "CandidateRecall": {
                                "provider_summary": [
                                    {
                                        "provider_id": "entity_graph",
                                        "source_family": "entity_graph",
                                        "output_count": 1,
                                        "output_source_counts": {"entity_graph": 1},
                                        "production_default": False,
                                        "shadow_only": True,
                                        "graph_related": True,
                                    }
                                ]
                            }
                        }
                    }
                },
            },
        ]

        audit = self._build_provider_audit(records)

        self.assertEqual([row["provider_id"] for row in audit], ["entity_graph", "raw_provider"])
        raw_row = next(row for row in audit if row["provider_id"] == "raw_provider")
        self.assertEqual(raw_row["source_family"], "raw_provider")
        self.assertEqual(raw_row["hit_count"], 2)
        self.assertEqual(raw_row["query_count"], 1)
        self.assertEqual(raw_row["output_count"], 5)
        self.assertEqual(raw_row["output_source_counts"], {"l0_raw_hybrid": 3, "raw_span": 2})
        self.assertEqual(raw_row["evidence_inputs"], ["first.json", "second.json"])
        self.assertTrue(raw_row["production_default"])
        self.assertFalse(raw_row["shadow_only"])
        self.assertFalse(raw_row["graph_related"])

        graph_row = next(row for row in audit if row["provider_id"] == "entity_graph")
        self.assertEqual(graph_row["query_count"], 1)
        self.assertEqual(graph_row["output_count"], 1)
        self.assertFalse(graph_row["production_default"])
        self.assertTrue(graph_row["shadow_only"])
        self.assertTrue(graph_row["graph_related"])

    def test_build_provider_audit_deduplicates_same_provider_summary_across_trace_locations(self) -> None:
        pipeline_trace = {
            "pipeline_layers": {
                "CandidateRecall": {
                    "provider_summary": [
                        {
                            "provider_id": "raw_provider",
                            "source_family": "raw_provider",
                            "output_count": 2,
                            "output_source_counts": {"l0_raw_hybrid": 2},
                            "production_default": True,
                            "shadow_only": False,
                            "graph_related": False,
                        }
                    ]
                }
            }
        }
        records = [
            {
                "query_id": "q1",
                "pipeline_trace": pipeline_trace,
                "coverage": {"pipeline_trace": pipeline_trace},
            }
        ]

        audit = self._build_provider_audit(records)

        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["hit_count"], 1)
        self.assertEqual(audit[0]["output_count"], 2)
        self.assertEqual(audit[0]["output_source_counts"], {"l0_raw_hybrid": 2})

    def test_build_provider_audit_hashes_unsafe_dimensions_and_excludes_raw_text(self) -> None:
        records = [
            {
                "query_id": "private question zinc-sparrow-17",
                "_audit_input": "raw-input.json",
                "pipeline_trace": {
                    "pipeline_layers": {
                        "CandidateRecall": {
                            "provider_summary": [
                                {
                                    "provider_id": "private provider PostgreSQL",
                                    "source_family": "数据库 selected",
                                    "output_count": 1,
                                    "output_source_counts": {"candidate text zinc-sparrow-17": 1},
                                    "production_default": True,
                                    "shadow_only": False,
                                    "graph_related": False,
                                    "sample_text": "candidate text zinc-sparrow-17",
                                    "raw_query": "private question zinc-sparrow-17",
                                }
                            ]
                        }
                    }
                },
            }
        ]

        audit = self._build_provider_audit(records)

        self.assertEqual(len(audit), 1)
        row = audit[0]
        self.assertEqual(len(row["provider_id"]), 12)
        self.assertEqual(len(row["source_family"]), 12)
        self.assertTrue(all(len(source) == 12 for source in row["output_source_counts"]))
        self.assertEqual(row["output_count"], 1)
        output_text = json.dumps(audit, sort_keys=True)
        self.assertNotIn("PostgreSQL", output_text)
        self.assertNotIn("数据库", output_text)
        self.assertNotIn("zinc-sparrow-17", output_text)
        self.assertNotIn("private question", output_text)
        self.assertNotIn("candidate text", output_text)

    def test_build_provider_audit_keeps_raw_source_family_readable(self) -> None:
        records = [
            {
                "query_id": "q1",
                "_audit_input": "replay.json",
                "pipeline_trace": {
                    "pipeline_layers": {
                        "CandidateRecall": {
                            "provider_summary": [
                                {
                                    "provider_id": "raw_span",
                                    "source_family": "raw",
                                    "output_count": 2,
                                    "output_source_counts": {"l0_raw": 1, "topic_scope_raw": 1},
                                    "production_default": True,
                                    "shadow_only": False,
                                    "graph_related": False,
                                }
                            ]
                        }
                    }
                },
            }
        ]

        audit = self._build_provider_audit(records)

        self.assertEqual(audit[0]["provider_id"], "raw_span")
        self.assertEqual(audit[0]["source_family"], "raw")
        self.assertEqual(audit[0]["output_source_counts"], {"l0_raw": 1, "topic_scope_raw": 1})

    def test_cli_writes_provider_json_and_csv_without_changing_rule_outputs(self) -> None:
        payload = {
            "records": [
                {
                    "query_id": "q1",
                    "rule_hits": [{"rule_id": "rule.alpha", "contributed_candidate_id": "c1"}],
                    "pipeline_trace": {
                        "pipeline_layers": {
                            "CandidateRecall": {
                                "provider_summary": [
                                    {
                                        "provider_id": "raw_provider",
                                        "source_family": "raw_provider",
                                        "output_count": 2,
                                        "output_source_counts": {"l0_raw_hybrid": 2},
                                        "production_default": True,
                                        "shadow_only": False,
                                        "graph_related": False,
                                    }
                                ]
                            }
                        }
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "replay.json"
            rule_output = tmp / "rule-audit.json"
            provider_output = tmp / "provider-audit.json"
            provider_csv = tmp / "provider-audit.csv"
            input_path.write_text(json.dumps(payload), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/rule_audit.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(rule_output),
                    "--provider-output",
                    str(provider_output),
                    "--provider-csv",
                    str(provider_csv),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rule_rows = json.loads(rule_output.read_text(encoding="utf-8"))
            self.assertEqual([row["rule_id"] for row in rule_rows], ["rule.alpha"])

            provider_rows = json.loads(provider_output.read_text(encoding="utf-8"))
            self.assertEqual([row["provider_id"] for row in provider_rows], ["raw_provider"])
            self.assertEqual(provider_rows[0]["evidence_inputs"], [str(input_path)])

            with provider_csv.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "provider_id",
                        "source_family",
                        "hit_count",
                        "query_count",
                        "output_count",
                        "output_source_counts",
                        "evidence_inputs",
                        "production_default",
                        "shadow_only",
                        "graph_related",
                    ],
                )
                csv_rows = list(reader)
            self.assertEqual(csv_rows[0]["provider_id"], "raw_provider")
            self.assertEqual(csv_rows[0]["output_source_counts"], "l0_raw_hybrid:2")
            self.assertEqual(csv_rows[0]["evidence_inputs"], str(input_path))

    def test_cli_provider_output_coerces_non_finite_numeric_counts_without_traceback(self) -> None:
        payload = """
{
  "records": [
    {
      "query_id": "q1",
      "pipeline_trace": {
        "pipeline_layers": {
          "CandidateRecall": {
            "provider_summary": [
              {
                "provider_id": "raw_provider",
                "source_family": "raw_provider",
                "output_count": 1e309,
                "output_source_counts": {"l0_raw_hybrid": 1e309},
                "production_default": true,
                "shadow_only": false,
                "graph_related": false
              }
            ]
          }
        }
      }
    }
  ]
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "replay.json"
            rule_output = tmp / "rule-audit.json"
            provider_output = tmp / "provider-audit.json"
            input_path.write_text(payload, encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/rule_audit.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(rule_output),
                    "--provider-output",
                    str(provider_output),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            combined_output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, msg=combined_output)
            self.assertNotIn("Traceback", combined_output)
            self.assertNotIn("OverflowError", combined_output)
            provider_rows = json.loads(provider_output.read_text(encoding="utf-8"))
            self.assertEqual(provider_rows[0]["output_count"], 0)
            self.assertEqual(provider_rows[0]["output_source_counts"], {"l0_raw_hybrid": 0})

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
        self.assertEqual(taxonomy_row["cleanup_blockers"], ["domain_label_taxonomy_migration_required"])
        self.assertFalse(taxonomy_row["safe_to_delete"])

        legacy_row = next(item for item in audit if item["rule_id"] == "event_ordering.legacy.tie_breaker")
        self.assertEqual(legacy_row["recommendation"], "legacy_shadow")
        self.assertEqual(legacy_row["cleanup_action"], "keep_protected")
        self.assertEqual(legacy_row["cleanup_blockers"], ["protected:legacy_event_ordering_fallback"])
        self.assertFalse(legacy_row["safe_to_delete"])

    def test_build_rule_audit_includes_provider_and_lifecycle_dimensions(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "current_value.stale_history_marker",
                        "contributed_candidate_id": "c1",
                        "provider_id": "views",
                        "lifecycle_stage": "selected",
                        "lifecycle_reason": "views",
                        "impact": "selected",
                    }
                ],
                "candidate_lifecycle": {
                    "records": [
                        {
                            "candidate_id": "c1",
                            "candidate_source": "l3_current_view",
                            "stage": "selected",
                            "reason_code": "views",
                        }
                    ]
                },
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "current_value.stale_history_marker")

        self.assertEqual(row["provider_ids"], ["views"])
        self.assertEqual(row["lifecycle_stages"], ["selected"])
        self.assertEqual(row["lifecycle_reasons"], ["views"])

    def test_build_rule_audit_hashes_unsafe_provider_and_lifecycle_dimensions(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "current_value.stale_history_marker",
                        "contributed_candidate_id": "c1",
                        "provider_id": "private provider PostgreSQL",
                        "lifecycle_stage": "数据库 selected",
                        "lifecycle_reason": "current database is PostgreSQL",
                        "impact": "selected",
                    }
                ],
                "pipeline_trace": {
                    "candidate_lifecycle": {
                        "records": [
                            {
                                "candidate_id": "c1",
                                "stage": "selected from private text",
                                "reason_code": "raw reason PostgreSQL",
                            }
                        ]
                    }
                },
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "current_value.stale_history_marker")

        self.assertTrue(all(len(item) == 12 for item in row["provider_ids"]))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_stages"] if item != "selected"))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_reasons"]))
        self.assertNotIn("PostgreSQL", repr(row))
        self.assertNotIn("数据库", repr(row))
        self.assertNotIn("private text", repr(row))

    def test_build_rule_audit_hashes_raw_looking_provider_and_lifecycle_dimensions(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "rule.raw_values",
                        "provider_id": "postgres://user@example.com/db",
                        "lifecycle_stage": "/var/lib/private/current.txt",
                        "lifecycle_reason": "acct-1234@example.com",
                        "impact": "selected",
                    }
                ],
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "rule.raw_values")

        self.assertTrue(all(len(item) == 12 for item in row["provider_ids"]))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_stages"]))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_reasons"]))
        self.assertNotIn("postgres://", repr(row))
        self.assertNotIn("/var/lib", repr(row))
        self.assertNotIn("@example.com", repr(row))

    def test_build_rule_audit_hashes_unknown_identifier_like_dimensions(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "rule.identifier_like_values",
                        "provider_id": "source_private_project",
                        "lifecycle_stage": "l3_private_goal",
                        "lifecycle_reason": "event_ordering_secret",
                        "impact": "selected",
                    }
                ],
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "rule.identifier_like_values")

        self.assertTrue(all(len(item) == 12 for item in row["provider_ids"]))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_stages"]))
        self.assertTrue(all(len(item) == 12 for item in row["lifecycle_reasons"]))
        self.assertNotIn("source_private_project", repr(row))
        self.assertNotIn("event_ordering_secret", repr(row))
        self.assertNotIn("l3_private_goal", repr(row))

    def test_build_rule_audit_correlates_lifecycle_with_raw_candidate_ids_before_hashing(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "rule.raw_candidate",
                        "contributed_candidate_id": "candidate id/用户@example.com",
                        "provider_id": "views",
                    }
                ],
                "candidate_lifecycle": {
                    "records": [
                        {
                            "candidate_id": "candidate id/用户@example.com",
                            "stage": "selected",
                            "reason_code": "views",
                        }
                    ]
                },
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "rule.raw_candidate")

        self.assertEqual(row["lifecycle_stages"], ["selected"])
        self.assertEqual(row["lifecycle_reasons"], ["views"])

    def test_build_rule_audit_reads_coverage_candidate_lifecycle_records(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "rule.coverage_lifecycle",
                        "contributed_candidate_id": "c1",
                    }
                ],
                "coverage": {
                    "candidate_lifecycle": {
                        "records": [
                            {
                                "candidate_id": "c1",
                                "stage": "rescued",
                                "reason_code": "event_ordering_coverage",
                            }
                        ]
                    }
                },
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "rule.coverage_lifecycle")

        self.assertEqual(row["lifecycle_stages"], ["rescued"])
        self.assertEqual(row["lifecycle_reasons"], ["event_ordering_coverage"])

    def test_build_rule_audit_marks_known_cli_governance_rules_protected(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {"rule_id": "current_value.stale_history_marker"},
                    {"rule_id": "exact_match.cjk_phrase"},
                    {"rule_id": "event_ordering.legacy.unused"},
                    {"rule_id": "event_ordering.legacy_rescue"},
                    {
                        "rule_id": "event_ordering.taxonomy.domain_label_match",
                        "metadata": {"category": "taxonomy_candidate"},
                    },
                ],
            }
        ]

        audit = build_rule_audit(records)
        by_rule = {str(row["rule_id"]): row for row in audit}

        self.assertTrue(by_rule["current_value.stale_history_marker"]["protected"])
        self.assertEqual(
            by_rule["current_value.stale_history_marker"]["protected_reason"],
            "high_precision_current_value",
        )
        self.assertFalse(by_rule["current_value.stale_history_marker"]["safe_to_delete"])
        self.assertTrue(by_rule["exact_match.cjk_phrase"]["protected"])
        self.assertEqual(by_rule["exact_match.cjk_phrase"]["protected_reason"], "chinese_recall_precision")
        self.assertFalse(by_rule["exact_match.cjk_phrase"]["safe_to_delete"])
        self.assertTrue(by_rule["event_ordering.legacy.unused"]["protected"])
        self.assertEqual(by_rule["event_ordering.legacy.unused"]["protected_reason"], "legacy_event_ordering_fallback")
        self.assertEqual(by_rule["event_ordering.legacy.unused"]["cleanup_action"], "keep_protected")
        self.assertEqual(
            by_rule["event_ordering.legacy.unused"]["cleanup_blockers"],
            ["protected:legacy_event_ordering_fallback"],
        )
        self.assertFalse(by_rule["event_ordering.legacy.unused"]["safe_to_delete"])
        self.assertTrue(by_rule["event_ordering.legacy_rescue"]["protected"])
        self.assertFalse(by_rule["event_ordering.legacy_rescue"]["safe_to_delete"])
        self.assertEqual(by_rule["event_ordering.taxonomy.domain_label_match"]["recommendation"], "migrate_to_taxonomy")
        self.assertEqual(by_rule["event_ordering.taxonomy.domain_label_match"]["cleanup_action"], "migrate_to_taxonomy")
        self.assertEqual(
            by_rule["event_ordering.taxonomy.domain_label_match"]["cleanup_blockers"],
            ["domain_label_taxonomy_migration_required"],
        )
        self.assertFalse(by_rule["event_ordering.taxonomy.domain_label_match"]["safe_to_delete"])

    def test_cleanup_gate_blocks_protected_zero_contribution_rules(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "current_value.stale_history_marker",
                        "contributed_candidate_id": None,
                        "protected": True,
                        "protected_reason": "high_precision_current_value",
                    }
                ],
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "current_value.stale_history_marker")

        self.assertFalse(row["safe_to_delete"])
        self.assertEqual(row["cleanup_action"], "keep_protected")
        self.assertEqual(row["cleanup_blockers"], ["protected:high_precision_current_value"])

    def test_explicit_protected_unknown_rule_blocks_safe_delete(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "rule.explicitly_protected",
                        "contributed_candidate_id": None,
                        "protected": True,
                        "protected_reason": "custom private reason",
                    }
                ],
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "rule.explicitly_protected")

        self.assertTrue(row["protected"])
        self.assertEqual(len(row["protected_reason"]), 12)
        self.assertFalse(row["safe_to_delete"])
        self.assertEqual(row["cleanup_action"], "keep_protected")
        self.assertEqual(row["cleanup_blockers"], [f"protected:{row['protected_reason']}"])
        self.assertNotIn("custom private reason", repr(row))

    def test_cleanup_gate_marks_exact_duplicate_unprotected_rule_safe_to_delete(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "rule.alpha_duplicate",
                        "contributed_candidate_id": None,
                        "metadata": {"duplicate_of": "rule.alpha"},
                    }
                ],
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "rule.alpha_duplicate")

        self.assertEqual(row["cleanup_action"], "delete_duplicate")
        self.assertEqual(row["cleanup_phase"], "first_pass")
        self.assertTrue(row["safe_to_delete"])

    def test_cleanup_gate_keeps_observation_only_rules(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {"rule_id": "multi_condition.query_token_match", "impact": "observed"},
                    {"rule_id": "zh_recall.cjk_exact_match", "impact": "observed"},
                    {
                        "rule_id": "taxonomy.alias_match",
                        "impact": "observed",
                        "metadata": {"category": "taxonomy_candidate"},
                    },
                ],
            }
        ]

        audit = build_rule_audit(records)
        by_rule = {row["rule_id"]: row for row in audit}

        for rule_id in ("multi_condition.query_token_match", "zh_recall.cjk_exact_match", "taxonomy.alias_match"):
            self.assertEqual(by_rule[rule_id]["cleanup_action"], "keep_observation")
            self.assertEqual(by_rule[rule_id]["cleanup_blockers"], ["observation_only_rule"])
            self.assertFalse(by_rule[rule_id]["safe_to_delete"])

    def test_cleanup_gate_migrates_domain_label_rules_even_when_duplicate(self) -> None:
        records = [
            {
                "query_id": "q1",
                "rule_hits": [
                    {
                        "rule_id": "event_ordering.taxonomy.domain_label_match",
                        "contributed_candidate_id": None,
                        "metadata": {
                            "category": "taxonomy_candidate",
                            "duplicate_of": "event_ordering.taxonomy.domain_label",
                        },
                    }
                ],
            }
        ]

        audit = build_rule_audit(records)
        row = next(item for item in audit if item["rule_id"] == "event_ordering.taxonomy.domain_label_match")

        self.assertEqual(row["cleanup_action"], "migrate_to_taxonomy")
        self.assertEqual(row["cleanup_blockers"], ["domain_label_taxonomy_migration_required"])
        self.assertFalse(row["safe_to_delete"])

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
        self.assertEqual(legacy_row["cleanup_action"], "keep_protected")
        self.assertEqual(legacy_row["cleanup_blockers"], ["protected:legacy_event_ordering_fallback"])
        unused_row = next(item for item in audit if item["rule_id"] == "event_ordering.legacy.unused")
        self.assertEqual(unused_row["recommendation"], "legacy_shadow")
        self.assertEqual(unused_row["cleanup_action"], "keep_protected")
        self.assertEqual(unused_row["cleanup_blockers"], ["protected:legacy_event_ordering_fallback"])
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
            self.assertTrue(all(len(item) == 12 for item in audit[0]["candidate_sources"]))
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
                        "provider_ids",
                        "lifecycle_stages",
                        "lifecycle_reasons",
                        "recommendation",
                        "protected",
                        "protected_reason",
                        "duplicate_of",
                        "cleanup_phase",
                        "cleanup_action",
                        "safe_to_delete",
                        "cleanup_blockers",
                    ],
                )
                rows = list(reader)

            self.assertEqual([row["rule_id"] for row in rows], ["rule.alpha", "rule.beta"])
            self.assertNotIn("source_a", rows[0]["candidate_sources"])
            self.assertNotIn("source_b", rows[0]["candidate_sources"])
            self.assertEqual(rows[0]["evidence_inputs"], str(input_path))

    def test_cli_csv_includes_rule_governance_columns(self) -> None:
        payload = {
            "records": [
                {
                    "query_id": "q1",
                    "rule_hits": [
                        {
                            "rule_id": "event_ordering.legacy_rescue",
                            "provider_id": "event_ordering_coverage",
                            "lifecycle_stage": "rescued",
                            "lifecycle_reason": "event_ordering_coverage",
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "replay.json"
            output_path = tmp / "audit.json"
            csv_path = tmp / "audit.csv"
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
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("provider_ids", reader.fieldnames)
                self.assertIn("lifecycle_stages", reader.fieldnames)
                self.assertIn("lifecycle_reasons", reader.fieldnames)
                self.assertIn("protected", reader.fieldnames)
                self.assertIn("protected_reason", reader.fieldnames)

    def test_cli_hashes_unknown_identifier_like_dimensions(self) -> None:
        payload = {
            "records": [
                {
                    "query_id": "q1",
                    "rule_hits": [
                        {
                            "rule_id": "rule.identifier_like_values",
                            "provider_id": "source_private_project",
                            "lifecycle_stage": "l3_private_goal",
                            "lifecycle_reason": "event_ordering_secret",
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "replay.json"
            output_path = tmp / "audit.json"
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
            output_text = output_path.read_text(encoding="utf-8")
            audit = json.loads(output_text)
            row = audit[0]
            self.assertTrue(all(len(item) == 12 for item in row["provider_ids"]))
            self.assertTrue(all(len(item) == 12 for item in row["lifecycle_stages"]))
            self.assertTrue(all(len(item) == 12 for item in row["lifecycle_reasons"]))
            self.assertNotIn("source_private_project", output_text)
            self.assertNotIn("event_ordering_secret", output_text)
            self.assertNotIn("l3_private_goal", output_text)

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

    def test_cli_reports_safe_error_for_malformed_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "malformed.json"
            output_path = tmp_path / "audit.json"
            input_path.write_text('{"records": [', encoding="utf-8")

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

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("Error: failed to load input", result.stderr)
            self.assertIn("Fix the JSON syntax or provide a readable replay JSON file.", result.stderr)
            combined_output = result.stdout + result.stderr
            self.assertNotIn("Traceback", combined_output)
            self.assertNotIn("json.decoder", combined_output)
            self.assertNotIn('File "', combined_output)

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
            self.assertNotEqual(rows[0]["candidate_sources"], "source_c")
            self.assertEqual(len(rows[0]["candidate_sources"]), 12)
            self.assertEqual(rows[0]["evidence_inputs"], str(input_path))


if __name__ == "__main__":
    unittest.main()
