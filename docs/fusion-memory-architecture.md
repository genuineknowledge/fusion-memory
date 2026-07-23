# Fusion Memory Architecture

Updated: 2026-07-23

The long historical architecture note is archived at
`docs/archive/fusion-memory-architecture-20260616-long.md`. This file is the
active product architecture reference.

## Objective

Fusion Memory is a multi-user product memory service for agents. Production
retrieval is optimized for stable evidence recall, explicit failure behavior,
and user isolation. BEAM is an evaluation profile, not a production retrieval
mode or a source of production query categories.

## Runtime Flow

1. `MemoryService` validates authorization and builds a `SearchRequest` plus a
   request-local `RetrievalContext`.
2. `ProductQueryPlanner` produces a typed `ProductQueryPlan`; invalid model
   refinement falls back to the deterministic product plan.
3. `ProductProviderRegistry` executes the requested providers against the
   repository boundary.
4. The engine performs reciprocal-rank fusion, optional reranking, utility
   scoring, and MMR selection exactly once.
5. The product trace records stage durations, provider counts, and allowlisted
   failure codes without raw query or memory text.
6. `ProductEvidencePackBuilder` projects authorized selected evidence into the
   typed answer-context sections.

Production supports only `fast` and `balanced` search modes.

## Layer Model

| Layer | Responsibility | Current State |
| --- | --- | --- |
| L0 evidence | Raw turns, documents, tool results, provenance | Durable source of truth; always user-scoped. |
| L1 facts | Add-only facts linked to source spans | Implemented. |
| L2 events | Time, ordering, updates, contradictions | Implemented with repository-owned chronology data. |
| L3 views/profiles | Current state and long-lived entity/profile summaries | Implemented; views are supplemental evidence, not authorization roots. |
| L4 product retrieval | Planning, provider recall, fusion, rerank, MMR, trace | Owned by `fusion_memory/retrieval/`; no benchmark mode. |
| L5 typed packs | Timeline, value history, temporal, aggregation, conflict, summary, instruction | Built from authorized product results and supporting spans. |
| L6 eval profiles | Benchmark-specific planning, prompts, deterministic consumers | Isolated under `fusion_memory/eval/`; BEAM-specific code is under `eval/beam/`. |

## Product Providers

The default product engine installs five providers:

| Provider | Purpose |
| --- | --- |
| `vector` | Semantic source-span recall through the repository embedding boundary. |
| `lexical` | Exact/BM25-style span, view, fact, and profile recall. |
| `temporal` | Date, interval, and temporal-relation evidence. |
| `entity` | Entity-linked facts, profiles, and source support. |
| `chronology` | Ordered event/aspect evidence from persisted chronology data. |

Providers receive a repository, request, plan, and request-local context. They
do not call private `MemoryService` helpers.

## Ownership Boundaries

- `fusion_memory/api/service.py` is a facade for authorization, ingestion,
  request construction, trace persistence, audit events, and public APIs.
- `fusion_memory/retrieval/query_planner.py` owns the product query plan.
- `fusion_memory/retrieval/providers/` owns product recall sources.
- `fusion_memory/retrieval/product_engine.py` owns one-pass execution and
  degradation semantics.
- `fusion_memory/retrieval/evidence_pack.py` owns the product pack projection;
  section algorithms remain in their focused retrieval modules.
- `fusion_memory/storage/` owns persistence and repository operations. Storage
  exceptions are not converted into partial retrieval success.
- Production modules under `api`, `retrieval`, and `mcp_runtime.py` must not
  import `fusion_memory.eval`.

## Scope And Isolation

- Authentication establishes the canonical `user_id`.
- Workspace and session are provenance for writes, not isolation boundaries
  between memories owned by the same user.
- Reads with only `Scope(user_id=...)` search all workspaces and sessions for
  that user.
- Different users never share candidates, traces, or persisted objects.
- PostgreSQL writes for one user are serialized with an advisory transaction
  lock; different users can execute concurrently.

## Failure Semantics

- A model-backed provider may return a retryable `model_unavailable` failure;
  successful providers still contribute candidates and coverage is marked
  degraded.
- Reranker endpoint failure returns the pre-rerank selection and records
  `reranker_unavailable`.
- If all planned providers fail, the engine raises `RetrievalUnavailable`.
- PostgreSQL backend failures propagate and do not return partial candidates.
- Expired deadlines stop execution before the next provider boundary.
- Trace and audit payloads contain counts, hashed identifiers, durations, and
  allowlisted error codes, never raw query/candidate text or credentials.

## BEAM Evaluation Boundary

`BeamRetrievalEngine` decorates the product plan inside
`fusion_memory/eval/beam/`. `BeamAdapter` wraps the generic answer model with
`OpenAICompatibleBeamAnswerModel`, whose category instructions and
deterministic answer behavior also live under `eval/beam/`.

Production accepts no `benchmark` mode, no BEAM category on
`ProductQueryPlan`, and no qid/gold-answer/named-scenario routing. Historical
BEAM artifacts remain useful evaluation evidence but are not runtime
dependencies.

## Validation Policy

Merge gates are:

- product query cases and same-user/different-user scope tests;
- provider/reranker degradation and storage/deadline fault tests;
- PostgreSQL concurrency and deployed MCP integration when configured;
- architecture scans for deleted legacy symbols, benchmark mode, private
  service callbacks, and production-to-eval imports;
- the full automated pytest suite and the `service.py` 1200-line budget.

BEAM scoring is diagnostic and is run only when explicitly requested. Ordinary
BEAM adapter/profile unit tests remain part of the full pytest suite, but the
2026-07-23 retrieval refactor did not separately invoke the BEAM smoke command
or run a benchmark/scoring job.

## Remaining Risks

- `retrieval/structured_annotations.py`, `event_ordering_*`, and
  `aggregation_pack.py` remain large heuristic-heavy modules. New product
  behavior should prefer typed repository/provider contracts over query-shaped
  rules.
- The deployed MCP/PostgreSQL isolation gate requires an active service URL and
  two configured user tokens; unit tests cannot substitute for that final
  deployment check.
