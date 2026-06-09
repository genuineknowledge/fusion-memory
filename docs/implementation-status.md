# Implementation Status

Date: 2026-06-09

This project is a runnable local MVP of Fusion Memory. It implements the
execution-plan module boundaries and core data flow, but it is not yet the final
production implementation described in the architecture document.

## Implemented

- Layer 0 Runtime/Foundation
  - Scope validation.
  - Read ScopeGuard requiring a business scope for search/history/timeline/report-facing reads.
  - Pluggable product `Authorizer` hook for add/search/answer-context/history/audit/report/view/profile/timeline entry points.
  - Session-isolated read behavior by default, with explicit `allow_cross_session` opt-in.
  - Centralized `MemoryConfig` for default parameters.
  - Deterministic local embedder aligned to the Qwen3 1024-dimensional deployment schema for dependency-free tests.
  - Dependency-free HTTP embedding adapter.
  - Dependency-free HTTP reranker adapter.
  - Optional local Qwen3 embedding and reranker adapters.
  - OpenAI-compatible structured LLM client for hosted/local extraction endpoints.
  - Per-operation model call telemetry in debug traces and audit payloads, including model/version, latency, usage, prompt version, and batch size when adapters expose it.
  - Debug trace persistence.
  - Scope-aware object-id reads for spans, facts, events, debug traces, and event comparison while preserving unscoped local debug compatibility.
  - Append-only `audit_events` for operation replay.

- Layer 1 Evidence Store
  - SQLite `evidence_spans`.
  - Scope-isolated add/get/search.
  - Local hybrid scoring with sparse keyword overlap and deterministic dense vectors.
  - SQLite FTS5 sparse indexes for evidence, facts, events, and profiles, with keyword fallback if FTS5 is unavailable.
  - Document chunking with overlap.
  - Session window spans.
  - Refreshable session summary spans that remain in L0 raw evidence and are skipped by extraction.
  - Background `background_tasks` queue for automatic long-session summary refresh.
  - Persistent `entities` registry.
  - `PostgresEvidenceRepository` for production-schema evidence span insert/get/list/duplicate/search.

- Layer 2 Extraction + EncodingGate
  - Rule-based extractor candidate schema.
  - Injectable structured LLM extractor interface.
  - Structured extractor can use the OpenAI-compatible HTTP LLM client.
  - Source-span validation for LLM extractor outputs.
  - Rule-based temporal normalizer.
  - EncodingGate decisions: `accept`, `merge`, `update_relation`, `quarantine`, `reject`.
  - `encoding_decisions` persisted for audit.
  - Encoding report API/CLI with decision counts, accept source coverage, and optional labeled precision.
  - `PostgresRuntimeRepository` can persist/list production-schema encoding decisions.

- Layer 3 Fact Ledger
  - ADD-only `memory_facts`.
  - `fact_relations`, including `supersedes`.
  - Source span attribution enforced by gate and tests.
  - `PostgresFactRepository` for production-schema fact insert/get/list/search and fact relation CRUD.

- Layer 4 Temporal/Event Graph
  - `events`.
  - Session-local `before` event edges.
  - Relative time support for `today`, `yesterday`, `tomorrow`, `last/this/next week`, `last/this/next month`.
  - Weekday support for `this Friday` / `next Friday` style references.
  - Explicit ISO date and month-name date parsing.
  - Unknown temporal expressions stay `time_source=unknown` instead of silently using current system time.
  - Explicit `before`/`after` statements write `event_edges` when a target event can be matched.
  - Public `timeline()` and `compare_events()` helpers.
  - `PostgresEventRepository` for production-schema event insert/get/list/search and event-edge CRUD.

- Layer 5 Views/Profile Layer
  - `current_views`.
  - `entity_profiles`.
  - Profile generation requires repeated support.
  - Public CurrentView/Profile getters and refreshers.
  - `PostgresViewProfileRepository` for production-schema CurrentView/Profile/Entity CRUD and profile/entity search.

- Layer 6 Retrieval Pack
  - Query planner.
  - Multi-source candidate generation.
  - Scope/session-aware candidate generation.
  - Raw evidence quota.
  - RRF, weighted utility score, MMR.
  - Fast/Balanced/Benchmark retrieval modes.
  - Pluggable reranker interface with local lexical reranker.
  - Pluggable HTTP reranker adapter.
  - Evidence pack builder with abstention policy.
  - Token budget enforcement for source spans.

- Layer 7 Retrieval Utility Scorer
  - Weak-label utility examples are collected in `retrieval_utility_examples`.
  - Dependency-free logistic scorer can train from collected examples.
  - Training report includes accuracy, NDCG@10, and MRR.
  - Search debug traces include shadow ranking when a scorer is trained.
  - `PostgresRuntimeRepository` can persist/list production-schema retrieval utility examples.

- Layer 8 Benchmark/Product Integration
  - Minimal retrieval-match benchmark adapter.
  - BEAM-specific adapter and `run-beam` CLI with split tracking.
  - LongMemEval-specific adapter and `run-longmemeval` CLI with question-scoped haystack ingestion, answer-session hit/recall metrics, and abstention accuracy.
  - JSON/JSONL dataset loader.
  - Local extractive answer model and lexical judge skeleton.
  - Optional OpenAI-compatible answer model and semantic judge model adapters for benchmark runs.
  - Per-category local retrieval/answer report with quota hit rate, latency, token estimates, model versions, and failure samples.
  - Benchmark reports include per-query LLM call counts and average LLM calls per query.
  - BEAM report records query-type mapping and per-answer evidence pack summaries.
  - Fast/Balanced/Benchmark retrieval-mode ablation report.
  - L0/L0+L1/L0+L1+L2/Full source-component ablation report.
  - Benchmark report includes encoding/profile coverage reports.
  - CLI for local add/search/answer-context/get/history/debug-trace/audit/timeline/views/profiles/report.
  - CLI `run-benchmark`, `run-beam`, and `run-longmemeval` accept answer/judge model endpoint options.
  - CLI `audit` command for operation audit events.
  - CLI `tasks` command for listing and processing background tasks.

- Production storage boundary
  - Postgres/pgvector migration SQL is present at `fusion_memory/storage/migrations/postgres/001_init.sql`.
  - Postgres schema now uses `vector(1024)` for Qwen3-Embedding-0.6B alignment.
  - `PostgresMigrationRunner` and `migrate-postgres` CLI can apply the production schema.
  - `PostgresEvidenceRepository` can use the production `evidence_spans` table for Layer 1 CRUD/search.
  - `PostgresFactRepository` can use production `memory_facts` and `fact_relations` for Layer 3 CRUD/search.
  - `PostgresEventRepository` can use production `events` and `event_edges` for Layer 4 CRUD/search.
  - `PostgresViewProfileRepository` can use production `current_views`, `entity_profiles`, and `entities`.
  - `PostgresRuntimeRepository` can use production `encoding_decisions`, `retrieval_utility_examples`, `debug_traces`, `audit_events`, and `background_tasks`.
  - `PostgresMemoryStore` provides a `SQLiteMemoryStore`-compatible facade over the production repositories.
  - `MemoryService(..., storage_backend="postgres")` selects the Postgres facade; `store=` remains available for direct product/test injection.
  - `verify-postgres` CLI and `FUSION_MEMORY_POSTGRES_DSN`-gated integration test can run a live Postgres add/search/answer-context/task/audit smoke.
  - Postgres schema uses text ids to match Fusion's prefixed local ids (`span_*`, `fact_*`, `trace_*`).
  - Postgres schema includes `background_tasks` for worker-compatible production scheduling.

## Pending

- Provision a Python 3.11/3.12 ML runtime for local Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B, or point the HTTP model adapters at a GPU-backed model service.
- Wire the model adapters to chosen production providers and validate latency/cost/token accounting on real traffic.
- Execute `verify-postgres` against a live Postgres/pgvector instance in this environment or CI and record the result.
- Add real BM25/Tantivy/OpenSearch backend or Postgres FTS ranking.
- Validate Retrieval Utility Scorer on real benchmark/replay data before enabling it for ranking.
- Configure leaderboard-grade BEAM/LongMemEval answer and judge models, then validate on official small/dev data.
- Extend temporal parsing beyond the current rule subset if production data needs quarters, deadlines, recurring events, or locale-specific dates.
- Wire the pluggable `Authorizer` to the product-specific authz/tenant identity provider.

## Verification

Current local verification command:

```bash
cd /home/wwb/fusion-memory
python -Werror::ResourceWarning -m unittest discover -s tests -v
python -m compileall -q fusion_memory tests
```

Last verified: 2026-06-09, 69 unittest cases passing/skipped as expected plus compileall. Manual Docker Postgres/pgvector verifier also passed locally.
