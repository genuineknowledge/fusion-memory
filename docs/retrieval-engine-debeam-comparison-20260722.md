# Retrieval Engine De-BEAM Comparison

Date: 2026-07-23

## Scope

This report compares the product retrieval behavior before and after the
retrieval-engine de-BEAM refactor.

- Baseline commit: `a9983dbf612fcf50b70222be9e4a6789f87ef84f`
- Product-gate commit: `faf37d91598b1e2f6d4dd9c358eb7f9b1f673c04`
- Branch: `refactor/retrieval-engine-debeam`
- Runtime: Python 3.13.5, x86_64, in-memory SQLite, deterministic local
  embedding behavior, rule extraction, and lexical cross-encoder reranking.
- Credentials/model tokens: not used by this comparison.

The current `tests/test_product_retrieval_cases.py` was copied into a temporary
`git archive` of the baseline commit so both revisions executed the same six
product assertions.

## Architecture Delta

| Area | Baseline `a9983db` | Product-gate `faf37d9` |
| --- | --- | --- |
| `api/service.py` | 4175 lines; retrieval/rescue policy centered in the facade | 1175 lines; facade delegates to the product engine |
| Query contract | Benchmark/category-shaped legacy paths coexisted with product behavior | `ProductQueryPlan` has product intent, providers, ordering, and reranker choice only |
| Recall | Service/private-helper callbacks and legacy provider pipeline | Five repository-backed product providers |
| Selection | Multiple rescue/preservation paths | One pass: fusion, optional rerank, utility, MMR |
| Production modes | Included benchmark-shaped behavior | Only `fast` and `balanced` |
| BEAM ownership | Category logic mixed into generic eval/retrieval paths | Planner/model behavior confined to `fusion_memory/eval/beam/` |
| Failure contract | Mixed fallback behavior | Explicit provider degradation; storage failures propagate |
| Isolation gate | Covered indirectly | Same user cross-session/workspace and different-user gates are explicit |

## Product Results

Both revisions passed all six cases:

- preference evidence (`Qdrant`);
- deployment deadline (`July 30`);
- incident chronology evidence (`mitigation`);
- exact internal code (`ZINC-42`);
- same-user cross-workspace/session visibility;
- different-user isolation.

| Revision | Result | Total pytest time | Slowest cases |
| --- | ---: | ---: | --- |
| Baseline `a9983db` | 6/6 passed | 0.41 s | 0.04 s preference; 0.03 s chronology; 0.02 s deadline |
| Product-gate `faf37d9` | 6/6 passed | 0.20 s | 0.04 s preference; remaining reported cases about 0.01 s |

These small in-memory timings are regression indicators, not production
latency benchmarks. The meaningful result is that the refactor retained target
evidence coverage and isolation while removing centralized rescue execution.

## Automated Validation

- Full suite after Task 13: `994 passed, 9 skipped, 12 subtests passed`.
- Product/fault/concurrency gate: `29 passed`.
- Product/fault/concurrency plus non-integration MCP checks: `32 passed, 4 deselected`.
- Production architecture scans: no benchmark mode, legacy private callbacks,
  or production import from `fusion_memory.eval`.
- `fusion_memory/api/service.py`: 1175 lines (limit: 1200).

The deployed MCP/PostgreSQL integration tests were collected but skipped
because `FUSION_MEMORY_E2E_URL`, `FUSION_MEMORY_E2E_TOKEN_A`, and
`FUSION_MEMORY_E2E_TOKEN_B` were not configured.

## BEAM Status

No separately invoked BEAM smoke command, benchmark runner, or full benchmark
run was performed for this refactor, by user direction. The full `pytest -q`
validation does include ordinary BEAM adapter/profile unit tests; those tests do
not produce benchmark scores or run artifacts. Therefore this report does not
claim a new total/category score or delta. The retained historical baselines
remain:

- `0.7751916960517531`
- `0.7676505254168324`

Their artifacts and category breakdown remain documented in `AGENTS.md` and
`docs/beam-100k-final-evaluation-report-20260617.md`.
