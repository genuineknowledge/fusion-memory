# Qwen3 + Postgres Deployment

Current target:

- Storage: Postgres + pgvector
- Embedding: Fusion Memory home-local `models/Qwen3-Embedding-0.6B`
- Embedding dimension: 1024
- Reranker: Fusion Memory home-local `models/Qwen3-Reranker-0.6B`
- Extractor: memory-owned configurable LLM extractor

## Prerequisites

- Python 3.11 or 3.12.
- PostgreSQL with the `pgvector` extension, or Docker Compose for the bundled
  development database.
- Optional local Qwen runtime dependencies if you do not use hosted HTTP model
  APIs.
- Optional GPU for lower local model latency. CPU works for smoke tests but is
  slower.

The installer downloads the default local model directories from ModelScope into
the Fusion Memory home models directory. The repository does not carry the model
weights. Override `FUSION_MEMORY_HOME` when you need the models and SQLite data
under a specific writable directory:

```bash
export FUSION_MEMORY_HOME="$HOME/.local/share/fusion-memory"
```

The remaining production-specific inputs are the LLM extractor endpoint/model
and API key, if you choose to enable the extractor.

## Start Postgres

Docker path:

```bash
cd /path/to/fusion-memory
docker compose -f deploy/docker-compose.postgres.yml up -d
python -m fusion_memory.cli migrate-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
python -m fusion_memory.cli verify-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
```

Rootless Docker can fail on some filesystems while registering the pgvector image
layer. A local Postgres fallback looks like this:

```bash
cd /path/to/fusion-memory
conda activate fusion-memory-qwen
conda install -y -c conda-forge postgresql pgvector
mkdir -p .runtime/postgres-data .runtime/postgres-run
initdb -D .runtime/postgres-data --auth=trust --username=fusion --no-locale --encoding=UTF8
pg_ctl -D .runtime/postgres-data -l .runtime/postgres.log -o "-p 55432 -k $PWD/.runtime/postgres-run" start
createdb -h 127.0.0.1 -p 55432 -U fusion fusion_memory
psql -h 127.0.0.1 -p 55432 -U fusion -d fusion_memory -c 'create extension if not exists vector;'
python -m fusion_memory.cli migrate-postgres postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory
python -m fusion_memory.cli verify-postgres postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory
```

The migration now uses `vector(1024)` for `evidence_spans`, `memory_facts`, and `entity_profiles`.

## Install Qwen Runtime

Use the project installer. It installs Fusion Memory as a `uv tool` with
uv-managed Python 3.12, downloads Qwen model weights from ModelScope, and avoids
using an agent runtime such as MSYS2 Python for the ML stack.

```bash
cd /path/to/fusion-memory
sh install.sh
fusion-memory install-check --force --json
fusion-memory doctor --json
```

Windows PowerShell:

```powershell
Set-Location C:\path\to\fusion-memory
.\install.ps1
fusion-memory install-check --force --json
fusion-memory doctor --json
```

## Qwen Smoke

```bash
cd /path/to/fusion-memory
python deploy/qwen_smoke.py \
  --embedding-model "${FUSION_MEMORY_HOME:-$HOME/.local/share/fusion-memory}/models/Qwen3-Embedding-0.6B" \
  --reranker-model "${FUSION_MEMORY_HOME:-$HOME/.local/share/fusion-memory}/models/Qwen3-Reranker-0.6B" \
  --device "${FUSION_MEMORY_QWEN_DEVICE:-cpu}" \
  --cache-dir "${FUSION_MEMORY_HOME:-$HOME/.local/share/fusion-memory}/models"
```

Expected:

- `embedding_dimension` is `1024`.
- reranker returns two numeric scores.

CPU smoke may be slow. Production traffic should use a GPU-backed model host or an HTTP model service.

## Runtime Environment Variables

```bash
source deploy/fusion-memory.env.example
```

The example enables local Qwen embedding/reranking and leaves the LLM extractor unset. When both `FUSION_MEMORY_EXTRACTOR_ENDPOINT` and `FUSION_MEMORY_EXTRACTOR_BASE_URL` are unset, the service keeps using the local rule-based extractor. Keep real extractor API keys in an ignored local file such as `deploy/fusion-memory.local.env`.

For OpenAI-compatible providers, either set the full endpoint:

```bash
export FUSION_MEMORY_EXTRACTOR_ENDPOINT=https://provider.example/v1/chat/completions
```

or set a base URL and let runtime config append `/chat/completions`:

```bash
export FUSION_MEMORY_EXTRACTOR_BASE_URL=https://provider.example/v1
```

## Aliyun DashScope HTTP Models

Fusion Memory can use DashScope hosted models through the existing HTTP
embedding/reranker adapters. Store the real key in an ignored local env file,
not in `deploy/fusion-memory.env.example`.

Embedding:

```bash
export DASHSCOPE_API_KEY=...
export FUSION_MEMORY_EMBEDDING_PROVIDER=http
export FUSION_MEMORY_EMBEDDING_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
export FUSION_MEMORY_EMBEDDING_API_KEY=$DASHSCOPE_API_KEY
export FUSION_MEMORY_EMBEDDING_MODEL=text-embedding-v4
export FUSION_MEMORY_EMBEDDING_DIMENSION=1024
export FUSION_MEMORY_EMBEDDING_ENCODING_FORMAT=float
```

Reranker:

```bash
export FUSION_MEMORY_RERANKER_PROVIDER=http
export FUSION_MEMORY_RERANKER_ENDPOINT=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
export FUSION_MEMORY_RERANKER_API_KEY=$DASHSCOPE_API_KEY
export FUSION_MEMORY_RERANKER_MODEL=qwen3-rerank
export FUSION_MEMORY_RERANKER_TOP_N=20
export FUSION_MEMORY_RERANKER_INSTRUCT="Given a memory query, retrieve relevant memory snippets."
```

The embedding endpoint is OpenAI-compatible and returns `data[].embedding`.
The rerank endpoint returns ranked `results[]` with `index` and
`relevance_score`; some DashScope rerank APIs wrap that array under
`output.results`. Fusion Memory accepts both shapes and restores scores to the
original document order before reranking candidates.

## Memory Service Wiring

```python
from fusion_memory.core.runtime_config import memory_service_from_env

memory = memory_service_from_env(
    "postgresql://fusion:fusion@localhost:5432/fusion_memory",
    storage_backend="postgres",
)
```

The extractor should be owned by the memory system. Agent models can pass hints or candidate context, but canonical fact/event/profile writes should still go through the memory extractor, source validation, and EncodingGate.

## Persistent HTTP Service

```bash
cd /path/to/fusion-memory
source deploy/fusion-memory.local.env
python -m fusion_memory.server \
  --host "$FUSION_MEMORY_SERVER_HOST" \
  --port "$FUSION_MEMORY_SERVER_PORT" \
  --db "$FUSION_MEMORY_DB" \
  --storage-backend "$FUSION_MEMORY_STORAGE_BACKEND"
```

Available endpoints:

- `GET /health`
- `POST /add`
- `POST /search`
- `POST /answer-context`

The process keeps one `MemoryService` instance alive, so local Qwen embedding and
reranker models are loaded once at startup and reused by later requests.
