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
- NVIDIA H100 GPUs are visible with `nvidia-smi`.
- System `python3` is 3.13.5.
- Conda `env1` has `torch 2.8.0+cu128` and was used as an initial fallback smoke environment.
- Python 3.12 conda env `fusion-memory-qwen` is now the target local environment.
- `fusion-memory-qwen` has `torch 2.8.0+cu128`, `transformers 5.10.2`, `sentence-transformers 5.5.1`, and `psycopg2-binary 2.9.12`.
- `pip check` passes in `fusion-memory-qwen`.
- `/public` has ample free space; model cache target is `/public/home/wwb/model/fusion-memory`.
- `hf-mirror.com` is reachable from this host; `huggingface.co` is not reliably reachable.
- Qwen local model weights are downloaded:
  - `/public/home/wwb/model/fusion-memory/Qwen3-Embedding-0.6B`
  - `/public/home/wwb/model/fusion-memory/Qwen3-Reranker-0.6B`

This is enough for Postgres smoke, code verification, and local Qwen embedding/reranker smoke. The remaining deployment gap is the production LLM extractor endpoint/model/API key.

## Start Postgres

Docker path:

```bash
cd /public/home/wwb/memory
docker compose -f deploy/docker-compose.postgres.yml up -d
python -m fusion_memory.cli migrate-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
python -m fusion_memory.cli verify-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
```

Rootless Docker can fail on some filesystems while registering the pgvector image
layer. The current host uses a conda-backed local Postgres fallback:

```bash
cd /public/home/wwb/memory
conda activate fusion-memory-qwen
conda install -y -c conda-forge postgresql pgvector
mkdir -p .runtime/postgres-data .runtime/postgres-run
initdb -D .runtime/postgres-data --auth=trust --username=fusion --no-locale --encoding=UTF8
pg_ctl -D .runtime/postgres-data -l .runtime/postgres.log -o "-p 55432 -k /public/home/wwb/memory/.runtime/postgres-run" start
createdb -h 127.0.0.1 -p 55432 -U fusion fusion_memory
psql -h 127.0.0.1 -p 55432 -U fusion -d fusion_memory -c 'create extension if not exists vector;'
python -m fusion_memory.cli migrate-postgres postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory
python -m fusion_memory.cli verify-postgres postgresql://fusion:fusion@127.0.0.1:55432/fusion_memory
```

The migration now uses `vector(1024)` for `evidence_spans`, `memory_facts`, and `entity_profiles`.

## Install Qwen Dependencies

Use Python 3.11 or 3.12. Python 3.14 is not recommended for the ML stack.

```bash
cd /public/home/wwb/memory
/public/home/wwb/anaconda3/bin/conda create -y -n fusion-memory-qwen python=3.12 pip
conda activate fusion-memory-qwen
pip install --no-deps -e . psycopg2-binary sentence-transformers transformers huggingface-hub tokenizers safetensors numpy scipy scikit-learn tqdm pyyaml regex requests filelock fsspec jinja2 networkx sympy typing-extensions
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
pip install tokenizers==0.22.2 typer click hf-xet httpx certifi charset-normalizer idna urllib3 joblib narwhals threadpoolctl rich shellingham annotated-doc anyio httpcore h11 markdown-it-py mdurl pygments
pip check
```

The explicit install sequence avoids the resolver selecting a newer CUDA 13 PyTorch build and keeps the environment aligned with the existing H100/CUDA 12.8 runtime.

The model weights were downloaded with:

```bash
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-Embedding-0.6B",
    local_dir="/public/home/wwb/model/fusion-memory/Qwen3-Embedding-0.6B",
)
snapshot_download(
    "Qwen/Qwen3-Reranker-0.6B",
    local_dir="/public/home/wwb/model/fusion-memory/Qwen3-Reranker-0.6B",
)
PY
```

## Qwen Smoke

```bash
cd /public/home/wwb/memory
conda activate fusion-memory-qwen
FUSION_MEMORY_MODEL_CACHE=/public/home/wwb/model/fusion-memory \
python deploy/qwen_smoke.py \
  --embedding-model /public/home/wwb/model/fusion-memory/Qwen3-Embedding-0.6B \
  --reranker-model /public/home/wwb/model/fusion-memory/Qwen3-Reranker-0.6B \
  --device cuda:0 \
  --cache-dir /public/home/wwb/model/fusion-memory
```

Expected:

- `embedding_dimension` is `1024`.
- reranker returns two numeric scores.

CPU smoke may be slow. Production traffic should use a GPU-backed model host or an HTTP model service.

Current smoke result in `fusion-memory-qwen`:

- `embedding_dimension` is `1024`.
- reranker returns two numeric scores.
- Runtime config smoke successfully loads local Qwen models and runs `add/search`.

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
cd /public/home/wwb/memory
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
