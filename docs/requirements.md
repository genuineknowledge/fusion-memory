# Fusion Memory Requirements

Date: 2026-06-09

This document records the current product, architecture, deployment, and
acceptance requirements for the Fusion Memory project. It is intended to travel
with the repository when the project is moved to a new server.

## Goal

Build a general-purpose Agent memory system that combines:

- auditable raw evidence retention,
- ADD-only actionable fact memory,
- temporal/event reasoning,
- current-state views and entity/personality profiles,
- benchmark-aware multi-source retrieval and evidence packing.

The memory system should remain usable layer by layer. Each layer must expose
stable APIs, persist debuggable state, and be testable independently.

## Target Deployment

The currently selected production-oriented configuration is:

| Component | Requirement |
|---|---|
| Primary database | Postgres 16 |
| Vector extension | pgvector |
| Vector dimension | 1024 |
| Embedding model | Fusion Memory home-local `models/Qwen3-Embedding-0.6B` |
| Reranker model | Fusion Memory home-local `models/Qwen3-Reranker-0.6B` |
| Extractor | Memory-owned configurable LLM extractor |
| Local fallback | SQLite, deterministic 1024-dimensional embedder, rule extractor, lexical reranker |
| Benchmark answer/judge | Configurable OpenAI-compatible answer and judge models |

The extractor is owned by the memory system. Agent models may provide hints or
context, but canonical fact/event/profile writes must pass through source
validation and EncodingGate.

## Environment Requirements

Minimum for local code verification:

- Python 3.11, 3.12, or compatible runtime for the project tests.
- Docker and Docker Compose for Postgres/pgvector smoke.
- No external model service is required for the default unittest suite.

Recommended for local Qwen model smoke:

- Python 3.11 or 3.12.
- `torch`, `transformers`, `sentence-transformers`.
- At least 15-25 GB free disk for dependencies and model cache.
- GPU is recommended for useful throughput. CPU is acceptable only for smoke.

Current source machine status before migration:

- Docker and Docker Compose are available.
- Postgres/pgvector container verified successfully.
- Python is 3.14.3.
- Qwen ML dependencies are not installed.
- No visible NVIDIA runtime.
- `/home` has about 8.5 GB free after cleanup.

## Storage Requirements

Postgres/pgvector is the source of truth for production deployment.

Required persisted objects:

- `evidence_spans`
- `memory_facts`
- `fact_relations`
- `events`
- `event_edges`
- `current_views`
- `entity_profiles`
- `entities`
- `encoding_decisions`
- `retrieval_utility_examples`
- `debug_traces`
- `audit_events`
- `background_tasks`

Vector columns in the Postgres migration must remain `vector(1024)` while using
Qwen3-Embedding-0.6B. If the embedding model changes to a different dimension,
the migration, indexes, and all stored embeddings must be rebuilt.

## Model Requirements

Embedding:

- Default target: Fusion Memory home-local `models/Qwen3-Embedding-0.6B`.
- Output dimension: 1024.
- Must support batch embedding.
- Must expose model version and call telemetry where possible.
- Must support either local adapter or HTTP embedding endpoint.

Reranker:

- Default target: Fusion Memory home-local `models/Qwen3-Reranker-0.6B`.
- Used in Balanced and Benchmark retrieval modes.
- Must expose model version and call telemetry where possible.
- Must support either local adapter or HTTP reranker endpoint.

LLM extractor:

- Must be configured and versioned by the memory system.
- Must return structured facts/events/relations/profile candidates.
- Must include source span attribution.
- Must be validated before EncodingGate decisions.
- Must record prompt version, model version, latency, and usage where available.

Benchmark answer/judge:

- Must be independently configurable.
- Must record model versions and LLM calls/query.
- Required for leaderboard-grade BEAM/LongMemEval reports.

## Functional Requirements

Layer 0 Runtime/Foundation:

- Validate write/read scopes.
- Enforce session-isolated reads by default.
- Provide explicit cross-session opt-in.
- Support product-provided `Authorizer`.
- Persist debug traces and audit events.
- Track model call telemetry.

Layer 1 Evidence Store:

- Preserve raw conversation/document/tool evidence.
- Support turn, window, document chunk, tool result, and summary spans.
- Support dense and sparse retrieval.
- Keep all derived memories linked to source spans.

Layer 2 Extraction + EncodingGate:

- Extract candidates instead of writing directly.
- Gate fact/event/relation/profile promotion.
- Support accept, merge, update_relation, quarantine, and reject decisions.
- Persist decisions for audit and reporting.

Layer 3 Fact Ledger:

- Facts are ADD-only.
- Updates must be represented with relations such as `supersedes`.
- Every fact must keep source span attribution.

Layer 4 Temporal/Event Graph:

- Persist events and event edges.
- Normalize relative times against session time.
- Support timeline and event comparison APIs.

Layer 5 Views/Profile:

- Maintain current-state views.
- Maintain entity/personality profiles only when repeated support exists.
- Profiles must not override explicit current-state or historical evidence.

Layer 6 Retrieval Pack:

- Use multi-source recall across raw spans, facts, events, views, and profiles.
- Enforce raw evidence quotas by query type.
- Apply RRF/MMR and optional reranking.
- Build token-budgeted evidence packs.

Layer 7 Retrieval Utility Scorer:

- Collect weak-label utility examples.
- Train a dependency-free local scorer.
- Report accuracy, NDCG@10, and MRR.
- Run in shadow mode before any ranking replacement.

Layer 8 Benchmark/Product Integration:

- Support generic JSON/JSONL benchmark datasets.
- Support BEAM-style local benchmark runs.
- Support LongMemEval-style local benchmark runs.
- Report per-category scores, quota hit rate, token estimates, latencies, model versions, failure samples, and evidence pack traces.

## Evaluation Requirements

Local verification:

```bash
cd /path/to/fusion-memory
python -Werror::ResourceWarning -m unittest discover -s tests -v
python -m compileall -q fusion_memory tests deploy
```

Postgres smoke:

```bash
cd /path/to/fusion-memory
docker compose -f deploy/docker-compose.postgres.yml up -d
python -m fusion_memory.cli migrate-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
python -m fusion_memory.cli verify-postgres postgresql://fusion:fusion@localhost:5432/fusion_memory
```

Qwen smoke, after installing optional dependencies:

```bash
cd /path/to/fusion-memory
source deploy/fusion-memory.env.example
python deploy/qwen_smoke.py \
  --embedding-model "$FUSION_MEMORY_EMBEDDING_MODEL" \
  --reranker-model "$FUSION_MEMORY_RERANKER_MODEL" \
  --device "$FUSION_MEMORY_EMBEDDING_DEVICE" \
  --cache-dir "$FUSION_MEMORY_MODEL_CACHE"
```

Expected Qwen smoke result:

- embedding dimension is 1024,
- reranker returns numeric scores.

Benchmark evaluation:

- Run generic benchmark fixtures for local regression.
- Run BEAM small/dev before 1M/10M.
- Run LongMemEval dev before full test.
- Use fixed answer and judge models for comparable reports.
- Record answer model, judge model, tokens/query, LLM calls/query, latency, and failure samples.

## Migration Checklist For New Server

1. Clone/pull this repository.
2. Install Python 3.11 or 3.12.
3. Run the default unittest suite.
4. Start Postgres/pgvector with `deploy/docker-compose.postgres.yml`.
5. Run migration and `verify-postgres`.
6. Install `.[postgres,qwen]` or reproduce the pinned conda environment described in `docs/deployment-qwen-postgres.md`.
7. Source `deploy/fusion-memory.env.example` and replace only the LLM extractor endpoint/model/API key when the provider is chosen.
7. Run `python deploy/qwen_smoke.py`.
8. Configure the LLM extractor endpoint/model/API key.
9. Configure benchmark answer/judge models.
10. Run BEAM/LongMemEval small or dev splits.

## Open Requirements

- Choose and configure the production LLM extractor model.
- Add a unified config/env loader for embedding/reranker/extractor endpoints.
- Add CLI flags or config-file support for runtime embedding/reranker/extractor selection.
- Add embedding reindex/backfill tooling.
- Run official BEAM 1M/10M and LongMemEval evaluation on the new server.
- Calibrate Retrieval Utility Scorer with real benchmark/dev/replay labels.
- Integrate the product identity provider through `Authorizer`.
