from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fusion_memory.core.models import Candidate, QueryPlan
from fusion_memory.retrieval.utility_scorer import feature_vector


NUMERIC_FEATURES = [
    "rrf_score",
    "semantic_score",
    "bm25_score",
    "entity_overlap",
    "temporal_fit",
    "graph_proximity",
    "view_or_profile_prior",
    "source_quality",
    "utility_score",
    "quota_selected",
]


@dataclass
class UtilityTrainingReport:
    total_examples: int
    used_examples: int
    positive_examples: int
    negative_examples: int
    loss: float
    accuracy: float
    ndcg_at_10: float = 0.0
    mrr: float = 0.0


@dataclass
class LogisticUtilityScorer:
    learning_rate: float = 0.2
    epochs: int = 200
    l2: float = 0.001
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    trained: bool = False
    version: str = "logistic-v0"

    def fit(self, examples: list[dict[str, Any]]) -> UtilityTrainingReport:
        rows: list[tuple[dict[str, float], int]] = []
        grouped_rows: dict[str, list[tuple[dict[str, float], int]]] = {}
        for example in examples:
            label = example.get("label")
            if label not in {"useful", "not_useful"}:
                continue
            features = self._encode_features(example.get("features", {}))
            encoded_label = 1 if label == "useful" else 0
            rows.append((features, encoded_label))
            group_key = str(example.get("query_id") or example.get("query_text") or "default")
            grouped_rows.setdefault(group_key, []).append((features, encoded_label))
        if not rows:
            self.trained = False
            return UtilityTrainingReport(len(examples), 0, 0, 0, 0.0, 0.0)
        self.feature_names = sorted({name for features, _ in rows for name in features})
        self.weights = {name: self.weights.get(name, 0.0) for name in self.feature_names}
        self.bias = self.bias or 0.0
        for _ in range(self.epochs):
            for features, label in rows:
                pred = self._predict_encoded(features)
                error = pred - label
                self.bias -= self.learning_rate * error
                for name in self.feature_names:
                    value = features.get(name, 0.0)
                    grad = error * value + self.l2 * self.weights[name]
                    self.weights[name] -= self.learning_rate * grad
        self.trained = True
        loss = 0.0
        correct = 0
        positives = 0
        negatives = 0
        for features, label in rows:
            pred = min(max(self._predict_encoded(features), 1e-6), 1 - 1e-6)
            loss += -(label * math.log(pred) + (1 - label) * math.log(1 - pred))
            predicted_label = 1 if pred >= 0.5 else 0
            correct += int(predicted_label == label)
            positives += int(label == 1)
            negatives += int(label == 0)
        ndcg_at_10, mrr = self._ranking_metrics(grouped_rows)
        return UtilityTrainingReport(
            total_examples=len(examples),
            used_examples=len(rows),
            positive_examples=positives,
            negative_examples=negatives,
            loss=loss / len(rows),
            accuracy=correct / len(rows),
            ndcg_at_10=ndcg_at_10,
            mrr=mrr,
        )

    def predict_candidate(self, candidate: Candidate, plan: QueryPlan) -> float:
        if not self.trained:
            return candidate.scores.get("utility_score", 0.0)
        return self._predict_encoded(self._encode_features(feature_vector(candidate, plan)))

    def rank_shadow(self, candidates: list[Candidate], plan: QueryPlan) -> list[dict[str, Any]]:
        ranked = [
            {
                "id": candidate.id,
                "type": candidate.type,
                "hand_score": candidate.scores.get("utility_score", 0.0),
                "shadow_score": self.predict_candidate(candidate, plan),
            }
            for candidate in candidates
        ]
        ranked.sort(key=lambda item: item["shadow_score"], reverse=True)
        return ranked

    def save(self, path: str | Path) -> None:
        data = {
            "version": self.version,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "weights": self.weights,
            "bias": self.bias,
            "feature_names": self.feature_names,
            "trained": self.trained,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LogisticUtilityScorer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            learning_rate=data["learning_rate"],
            epochs=data["epochs"],
            l2=data["l2"],
            weights={key: float(value) for key, value in data["weights"].items()},
            bias=float(data["bias"]),
            feature_names=list(data["feature_names"]),
            trained=bool(data["trained"]),
            version=data.get("version", "logistic-v0"),
        )

    def _predict_encoded(self, features: dict[str, float]) -> float:
        z = self.bias + sum(self.weights.get(name, 0.0) * features.get(name, 0.0) for name in self.feature_names)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def _encode_features(self, features: dict[str, Any]) -> dict[str, float]:
        encoded: dict[str, float] = {}
        for name in NUMERIC_FEATURES:
            value = features.get(name, 0.0)
            if isinstance(value, (int, float)):
                encoded[name] = float(value)
        candidate_type = str(features.get("candidate_type", "unknown"))
        query_type = str(features.get("query_type", "unknown"))
        encoded[f"candidate_type={candidate_type}"] = 1.0
        encoded[f"query_type={query_type}"] = 1.0
        return encoded

    def _ranking_metrics(self, grouped_rows: dict[str, list[tuple[dict[str, float], int]]]) -> tuple[float, float]:
        ndcgs: list[float] = []
        reciprocal_ranks: list[float] = []
        for rows in grouped_rows.values():
            if not any(label == 1 for _, label in rows):
                continue
            ranked = sorted(
                [(self._predict_encoded(features), label) for features, label in rows],
                key=lambda item: item[0],
                reverse=True,
            )
            dcg = _dcg([label for _, label in ranked[:10]])
            ideal = _dcg(sorted((label for _, label in rows), reverse=True)[:10])
            ndcgs.append(dcg / ideal if ideal else 0.0)
            reciprocal_ranks.append(next((1.0 / index for index, (_, label) in enumerate(ranked, start=1) if label == 1), 0.0))
        if not ndcgs:
            return 0.0, 0.0
        return sum(ndcgs) / len(ndcgs), sum(reciprocal_ranks) / len(reciprocal_ranks)


def _dcg(labels: list[int]) -> float:
    return sum((2**label - 1) / math.log2(index + 1) for index, label in enumerate(labels, start=1))
