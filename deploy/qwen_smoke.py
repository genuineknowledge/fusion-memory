from __future__ import annotations

from fusion_memory.core.embedding import Qwen3EmbeddingClient
from fusion_memory.retrieval.reranker import Qwen3Reranker


def main() -> None:
    embedder = Qwen3EmbeddingClient(device=None, batch_size=2)
    vectors = embedder.embed_texts(["Atlas uses Qdrant.", "Reports should use PostgreSQL."])
    print({"embedding_count": len(vectors), "embedding_dimension": len(vectors[0]), "embedding_version": embedder.version})

    reranker = Qwen3Reranker(device=None, batch_size=2)
    scores = reranker.score("What does Atlas use?", ["Atlas uses Qdrant.", "Reports should use PostgreSQL."])
    print({"rerank_scores": scores, "reranker_version": reranker.version})


if __name__ == "__main__":
    main()
