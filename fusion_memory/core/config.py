from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RAW_EVIDENCE_QUOTAS = {
    "factual_exact": 2,
    "temporal_lookup": 4,
    "event_ordering": 6,
    "contradiction_resolution": 4,
    "knowledge_update": 4,
    "multi_session_reasoning": 4,
    "abstention": 3,
    "summarization": 6,
    "preference": 1,
    "instruction": 1,
    "assistant_reference": 2,
}


def _default_home() -> Path:
    env_home = os.getenv("FUSION_MEMORY_HOME")
    if env_home:
        return Path(env_home).expanduser()
    system = platform.system().lower()
    if system == "windows":
        return (
            Path(os.getenv("APPDATA", str(Path.home() / "AppData" / "Roaming")))
            / "FusionMemory"
        )
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / "FusionMemory"
    return (
        Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        / "fusion-memory"
    )


DEFAULT_MODEL_ROOT = _default_home() / "models"
DEFAULT_EMBEDDING_MODEL = str(DEFAULT_MODEL_ROOT / "Qwen3-Embedding-0.6B")
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_RERANKER_MODEL = str(DEFAULT_MODEL_ROOT / "Qwen3-Reranker-0.6B")


@dataclass
class MemoryConfig:
    storage_backend: str = "sqlite"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    reranker_model: str = DEFAULT_RERANKER_MODEL
    extractor_model: str | None = None
    extractor_prompt_version: str = "llm-extractor-v0"
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 150
    session_window_size: int = 6
    min_window_spans: int = 3
    session_summary_min_spans: int = 6
    session_summary_max_source_spans: int = 40
    session_summary_max_chars: int = 1400
    auto_session_summary_tasks: bool = True
    fact_accept_confidence: float = 0.70
    fact_quarantine_confidence: float = 0.45
    salience_threshold: float = 0.35
    novelty_threshold: float = 0.25
    duplicate_similarity_threshold: float = 0.92
    relation_accept_confidence: float = 0.75
    event_accept_confidence: float = 0.70
    retrieval_top_k_per_source: int = 30
    rrf_k: int = 60
    mmr_lambda: float = 0.72
    answer_context_budget_tokens: int = 8000
    retrieval_output_n: int = 12
    balanced_mode_rerank_top_n: int = 50
    benchmark_mode_rerank_top_n: int = 20
    evidence_span_summary_chars: int = 500
    local_answer_summary_chars: int = 360
    raw_evidence_quotas: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_RAW_EVIDENCE_QUOTAS))

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = MemoryConfig()
