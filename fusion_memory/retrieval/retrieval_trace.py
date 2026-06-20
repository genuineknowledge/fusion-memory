from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PIPELINE_LAYER_ORDER = (
    ("QueryUnderstanding", "query_understanding"),
    ("CandidateRecall", "candidate_recall"),
    ("CandidateFusion", "candidate_fusion"),
    ("EvidencePackBuilder", "evidence_output"),
)


@dataclass
class RetrievalTraceBuilder:
    query_type: str
    mode: str
    _sections: dict[str, Any] = field(default_factory=dict)

    def query_understanding(self, *, language: str, intent: str, features: list[str]) -> None:
        self._sections["query_understanding"] = {
            "language": language,
            "intent": intent,
            "features": list(features),
        }

    def candidate_recall(self, *, source_counts: dict[str, int]) -> None:
        self._sections["candidate_recall"] = {
            "source_counts": dict(source_counts),
        }

    def candidate_fusion(self, *, selected_sources: list[str], dropped_count: int) -> None:
        self._sections["candidate_fusion"] = {
            "selected_sources": list(selected_sources),
            "dropped_count": int(dropped_count),
        }

    def evidence_output(self, *, source_span_count: int, coverage_insufficient: bool) -> None:
        self._sections["evidence_output"] = {
            "source_span_count": int(source_span_count),
            "coverage_insufficient": bool(coverage_insufficient),
        }

    def pipeline_layers(self) -> dict[str, object]:
        return {
            layer_name: self._sections.get(section_name, {})
            for layer_name, section_name in PIPELINE_LAYER_ORDER
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "mode": self.mode,
            "pipeline_layers": self.pipeline_layers(),
            **self._sections,
        }
