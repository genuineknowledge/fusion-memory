# Qwen3 + Postgres Deployment

Current target:

- Storage: Postgres + pgvector
- Embedding: `Qwen/Qwen3-Embedding-0.6B`
- Embedding dimension: 1024
- Reranker: `Qwen/Qwen3-Reranker-0.6B`
- Extractor: memory-owned configurable LLM extractor

## Environment Status

Checked on 2026-06-09:

- Docker and Docker Compose are installed.
- No visible NVIDIA runtime: `nvidia-smi` is not available.
- Python is currently 3.14.3.
- `torch`, `transformers`, `sentence_transformers`, and `vllm` are not installed.
- `/home` has about 8.8 GB free after cleanup.

This is enough for Postgres smoke and code verification. It is not enough for a comfortable local Qwen model deployment. For local CPU smoke, create a Python 3.11/3.12 environment and keep at least 15-25 GB free for model/dependency caches.

## Start Postgres

```bash
cd /home/wwb/fusion-memory
docker compose -f deploy/docker-compose.postgres.yml up -d
python -m fusion_memory.cli migrate-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
python -m fusion_memory.cli verify-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
```

The migration now uses `vector(1024)` for `evidence_spans`, `memory_facts`, and `entity_profiles`.

## Install Qwen Dependencies

Use Python 3.11 or 3.12. Python 3.14 is not recommended for the ML stack.

```bash
cd /home/wwb/fusion-memory
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[postgres,qwen]"
```

If Python 3.12 is not available on this machine, install it first or run the Qwen services on a separate model host.

## Qwen Smoke

```bash
cd /home/wwb/fusion-memory
. .venv/bin/activate
python deploy/qwen_smoke.py
```

Expected:

- `embedding_dimension` is `1024`.
- reranker returns two numeric scores.

CPU smoke may be slow. Production traffic should use a GPU-backed model host or an HTTP model service.

## Memory Service Wiring

```python
from fusion_memory import MemoryService
from fusion_memory.core.embedding import Qwen3EmbeddingClient
from fusion_memory.core.llm import OpenAICompatibleLLMClient
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor
from fusion_memory.retrieval.reranker import Qwen3Reranker

extractor_client = OpenAICompatibleLLMClient(
    "https://your-llm-provider/v1/chat/completions",
    model="extractor-model",
    api_key="...",
)

memory = MemoryService(
    "postgresql://fusion:fusion@localhost:5432/fusion_memory",
    storage_backend="postgres",
    embedder=Qwen3EmbeddingClient(),
    reranker=Qwen3Reranker(),
    extractor=StructuredLLMExtractor(extractor_client),
)
```

The extractor should be owned by the memory system. Agent models can pass hints or candidate context, but canonical fact/event/profile writes should still go through the memory extractor, source validation, and EncodingGate.
