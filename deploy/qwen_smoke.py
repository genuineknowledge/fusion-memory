from __future__ import annotations

import argparse
import os

from fusion_memory.core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from fusion_memory.core.embedding import Qwen3EmbeddingClient
from fusion_memory.retrieval.reranker import Qwen3Reranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test local Qwen embedding and reranker adapters")
    parser.add_argument("--embedding-model", default=os.getenv("FUSION_MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--reranker-model", default=os.getenv("FUSION_MEMORY_RERANKER_MODEL", DEFAULT_RERANKER_MODEL))
    parser.add_argument("--device", default=os.getenv("FUSION_MEMORY_QWEN_DEVICE"))
    parser.add_argument("--cache-dir", default=os.getenv("FUSION_MEMORY_MODEL_CACHE"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("FUSION_MEMORY_QWEN_BATCH_SIZE", "2")))
    args = parser.parse_args()

    model_kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    embedder = Qwen3EmbeddingClient(
        model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        model_kwargs=model_kwargs,
    )
    vectors = embedder.embed_texts(["Atlas uses Qdrant.", "Reports should use PostgreSQL."])
    print({"embedding_count": len(vectors), "embedding_dimension": len(vectors[0]), "embedding_version": embedder.version})

    reranker = Qwen3Reranker(
        model=args.reranker_model,
        device=args.device,
        batch_size=args.batch_size,
        model_kwargs=model_kwargs,
    )
    scores = reranker.score("What does Atlas use?", ["Atlas uses Qdrant.", "Reports should use PostgreSQL."])
    print({"rerank_scores": scores, "reranker_version": reranker.version})


if __name__ == "__main__":
    main()
