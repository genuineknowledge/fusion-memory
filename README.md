# Fusion Memory

Fusion Memory 是一个面向 Agent 的通用记忆系统本地 MVP。它基于：

- 新手安装：[docs/quickstart.md](docs/quickstart.md)
- Agent adapters: [docs/agent-adapters.md](docs/agent-adapters.md)
- Error guide: [docs/errors.md](docs/errors.md)
- [docs/fusion-memory-architecture.md](docs/fusion-memory-architecture.md)
- [docs/fusion-memory-execution-plan.md](docs/fusion-memory-execution-plan.md)
- [docs/requirements.md](docs/requirements.md)

当前实现目标是把架构拆成可单独运行、单独测试、单独替换的层级模块。默认测试运行不依赖外部服务：本地使用 SQLite、确定性 embedding、规则 extractor、本地 reranker、本地 benchmark answer/judge stub。

当前部署目标已经确定为 Postgres + pgvector、Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B，以及 Memory 系统自管的可配置 LLM extractor。代码已经提供本地 Qwen 适配器和 HTTP model fallback，但生产 endpoint、API key、GPU/CPU 运行环境、成本/延迟策略仍需要配置和验证。

## 当前状态

项目状态如下：

- 本地 MVP 已实现 Layer 0-8 的主流程。
- SQLite 后端可直接运行 add/search/answer-context/history/timeline/views/profiles/report/benchmark。
- Postgres/pgvector schema、migration runner、repository facade 已实现；schema 已切到 Qwen3-Embedding-0.6B 对齐的 `vector(1024)`。
- 新增 Qwen3 本地适配器：`Qwen3EmbeddingClient` 和 `Qwen3Reranker`，通过 optional `qwen` 依赖安装。
- BEAM/LongMemEval 本地 harness 可 ingest/retrieve/answer/score，并支持 ablation report。
- Benchmark answer/judge 已支持 OpenAI-compatible endpoint 注入。
- Retrieval Utility Scorer 支持弱标签收集、训练、NDCG@10/MRR shadow report，但还没有真实 replay/benchmark 数据校准。
- 默认 extractor 仍是规则实现，不代表最终抽取质量；生产路径应注入 Memory 系统自管的 `StructuredLLMExtractor`。
- 生产部署可以选择本地 Qwen 模型或 HTTP 模型服务；LLM extractor endpoint 仍留待生产配置填写。详见 `docs/deployment-qwen-postgres.md`。

最近验证：

```bash
cd /path/to/fusion-memory
PYTHONDONTWRITEBYTECODE=1 python -Werror::ResourceWarning -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q fusion_memory tests
```

测试套件默认不需要外部服务；需要真实 Postgres、Qwen 或 Agent runtime 的 smoke 时，按对应文档设置环境变量后再运行。

## 快速开始

面向新手的默认 SQLite 本地服务：

```bash
git clone https://github.com/genuineknowledge/fusion-memory.git
cd fusion-memory
sh install.sh
fusion-memory init --json
fusion-memory start --json
fusion-memory doctor --json
export PSI_MEMORY_BASE_URL=http://127.0.0.1:8700
```

If port `8700` is already in use, `fusion-memory start --json` tries the next available local port and returns the actual `url`; set `PSI_MEMORY_BASE_URL` to that returned URL before starting the agent workspace.

仓库自带两个本地向量模型目录：`models/Qwen3-Embedding-0.6B` 和
`models/Qwen3-Reranker-0.6B`。安装脚本不会从其他地方下载模型；它会安装
基础包，并尽力安装完整运行依赖（`.[postgres,qwen]`），包括 Postgres adapter、
本地 Qwen adapter 以及 PyTorch/Transformers 相关依赖。安装检查会确认仓库内
模型文件存在，并运行一次最小 Qwen embedding/reranker smoke test。默认本地配置是
SQLite + repo-local Qwen embedding/reranker。Postgres/pgvector 是可选生产配置，
不是本地默认安装门槛。当模型文件齐全，但 Python/Qwen runtime 或当前硬件无法
加载/运行本地 Qwen 模型时，安装会 fallback 到
`compromised` 本地模式：SQLite + 内置轻量 embedding/reranker 可以继续试用，
但检索质量是 compromised 的。安装完成后会提示提供 API key；推荐阿里云
DashScope，例如设置 `DASHSCOPE_API_KEY`，再通过向导或环境变量接入 API provider。
如果模型文件缺失，安装检查会返回 not ready，并提示重新运行
`pip install -e ".[postgres,qwen]"`。如果模型目录中只有 Git LFS pointer 文件，
先安装 Git LFS 并在仓库中运行 `git lfs pull` 拉取真实权重，再重新运行
`fusion-memory install-check --force`。

`install.sh` / `install.ps1` 安装完成后会自动运行 `fusion-memory install-check`。
如需手动配置数据库、embedding、reranker、extractor/router，可设置
`FUSION_MEMORY_USE_WIZARD=1` 后重新运行安装脚本。API key 只通过环境变量读取，
不写入配置文件。

常用维护命令：

```bash
fusion-memory doctor
fusion-memory backup
fusion-memory upgrade --dry-run
fusion-memory stop
```

psi-agent/Haitun 自动持久化：

```bash
fusion-memory sync-haitun-history \
  --workspace /path/to/haitun-workspace \
  --session-id <session-id> \
  --background --json
fusion-memory status-haitun-history-watcher \
  --workspace /path/to/haitun-workspace \
  --session-id <session-id> \
  --json
```

这条命令不改 agent core。它读取 `histories/<session-id>.jsonl`，把新的
user/assistant turn 写入 Fusion Memory `/add`，并用本地 state 文件去重。

默认 workspace tools 只有在 agent 调用 `memory_add` 时才写入；启动
`sync-haitun-history --background` 后，history 才会持续自动写入。

```bash
cd /path/to/fusion-memory
python -m unittest discover -s tests
```

最小 Python 用法：

```python
from datetime import datetime, timezone
from fusion_memory import MemoryService, Scope

memory = MemoryService()
scope = Scope(workspace_id="w1", user_id="u1", agent_id="a1", session_id="s1")

memory.add("I prefer Qdrant for Atlas retrieval.", scope, datetime.now(timezone.utc))
pack = memory.answer_context("What do I currently prefer for Atlas?", scope)
print(pack.current_views)
```

最小 CLI 用法：

```bash
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a add "I prefer Qdrant for Atlas retrieval."
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a search "What do I prefer for Atlas?"
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a answer-context "What do I prefer for Atlas?"
```

带 `session_id` 时，读接口默认只读当前 session；需要跨 session 时显式传 `--allow-cross-session` 或 API 参数 `allow_cross_session=True`。

## 已实现层级

- Layer 0 Runtime/Foundation：Scope 校验、ScopeGuard、Authorizer、配置、审计、debug trace、模型调用 telemetry、后台任务。
- Layer 1 Evidence Store：`evidence_spans`、chunk/window/summary span、SQLite FTS5、实体 registry、原文 get/search。
- Layer 2 Extraction + EncodingGate：规则候选抽取、结构化 LLM extractor 接口、source-span 校验、accept/merge/update_relation/quarantine/reject 决策。
- Layer 3 Fact Ledger：ADD-only `memory_facts`、`fact_relations`、`supersedes`。
- Layer 4 Temporal/Event Graph：`events`、`event_edges`、相对时间/weekday/ISO/month-name temporal normalizer、timeline/compare_events。
- Layer 5 Views/Profile：`current_views`、`entity_profiles`、重复证据支持的 profile 生成、getter/refresh/report。
- Layer 6 Retrieval Pack：query planner、多源召回、raw evidence quota、RRF/MMR、Fast/Balanced/Benchmark 模式、token-budgeted evidence pack。
- Layer 7 Retrieval Utility Scorer：弱标签样本、dependency-free logistic scorer、shadow ranking、accuracy/NDCG@10/MRR report。
- Layer 8 Benchmark/Product Integration：通用 JSON/JSONL benchmark、BEAM adapter、LongMemEval adapter、ablation report、answer/judge model endpoint。
- Production storage boundary：Postgres/pgvector migration、Postgres repositories、`MemoryService(..., storage_backend="postgres")` facade、`verify-postgres` smoke。

## 主要 API

```python
memory.add(input, scope, session_time=None, metadata=None)
memory.search(query, scope, options=None)
memory.answer_context(query, scope, budget=None)
memory.get(object_id, object_type=None, scope=None)
memory.history(scope, entity=None, fact_id=None, allow_cross_session=False)
memory.timeline(entity, scope, start=None, end=None, allow_cross_session=False)
memory.compare_events(event_a, event_b, scope=None)
memory.get_current_views(scope, view_type=None, allow_cross_session=False)
memory.refresh_current_views(scope)
memory.get_entity_profile(entity_id, scope, profile_type=None, allow_cross_session=False)
memory.refresh_entity_profiles(scope)
memory.refresh_session_summary(scope)
memory.get_session_summaries(scope)
memory.list_background_tasks(scope, status=None)
memory.process_background_tasks(scope, limit=10)
memory.encoding_report(scope, labels=None)
memory.profile_report(scope, labels=None)
memory.train_utility_scorer()
```

读接口要求至少有一个业务 scope：`workspace_id`、`user_id`、`agent_id` 或 `run_id`。产品侧可以通过 `Authorizer` 接入真实租户/身份权限。

## CLI 命令

常用命令：

```bash
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a add "..."
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a search "..."
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a answer-context "..."
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a history --entity Atlas
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a timeline --entity Atlas
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a views --type current_preferences
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a profiles u --type communication_style
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a report encoding
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a report profiles
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a train-utility --save-model utility-model.json
```

Benchmark：

```bash
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a run-benchmark dataset.json --ablate
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a run-beam beam_dataset_dir --split small --ablate
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a run-longmemeval longmemeval_dir --split dev --ablate
```

Benchmark answer/judge 使用 OpenAI-compatible endpoint：

```bash
python -m fusion_memory.cli --db fusion-memory.sqlite3 --workspace-id w --user-id u --agent-id a run-beam beam_dataset_dir --split 1m \
  --answer-endpoint http://localhost:8000/v1/chat/completions --answer-model answer-model \
  --judge-endpoint http://localhost:8000/v1/chat/completions --judge-model judge-model \
  --model-api-key "$MODEL_API_KEY"
```

Postgres：

```bash
python -m fusion_memory.cli migrate-postgres postgresql://user:pass@localhost:5432/fusion_memory
python -m fusion_memory.cli verify-postgres postgresql://user:pass@localhost:5432/fusion_memory
```

常驻 HTTP service wrapper：

```bash
source deploy/fusion-memory.local.env
python -m fusion_memory.server \
  --host "$FUSION_MEMORY_SERVER_HOST" \
  --port "$FUSION_MEMORY_SERVER_PORT" \
  --db "$FUSION_MEMORY_DB" \
  --storage-backend "$FUSION_MEMORY_STORAGE_BACKEND"
```

该 wrapper 在进程启动时构造一个 `MemoryService`，因此本地 Qwen embedding/reranker 模型只加载一次，后续 `/add`、`/search`、`/answer-context` 请求复用同一组模型实例。

## 模型配置状态

当前模型相关能力是“接口已实现，生产配置未完成”。

| 模块 | 当前默认 | 已有适配器 | 缺少的生产配置 |
|---|---|---|---|
| Embedding | `DeterministicEmbedder(dimensions=1024)`，只适合本地测试 | `Qwen3EmbeddingClient`、`HTTPEmbeddingClient(endpoint, model, api_key)`，CLI/API 可通过 `memory_service_from_env` 读取 `FUSION_MEMORY_EMBEDDING_*` | 生产需要选择本地 Qwen 或 HTTP provider，并配置 timeout/retry、成本记录、历史 reindex/backfill |
| SQLite 向量 | JSON text 存储 dense vector，维度随 embedder 输出 | 可注入自定义 embedder | 无需 schema 维度，但需要重算历史 embedding 的 reindex/backfill 工具 |
| Postgres/pgvector | migration 固定 `vector(1024)` | `PostgresMemoryStore(..., embedder=...)` | 需要 live DSN 验证；如果未来换非 1024 维 embedding，必须改 pgvector 维度并重建 HNSW index |
| LLM extractor | 默认 `RuleBasedExtractor` | `StructuredLLMExtractor(OpenAICompatibleLLMClient(...))`，支持 `FUSION_MEMORY_EXTRACTOR_BASE_URL` 或完整 `FUSION_MEMORY_EXTRACTOR_ENDPOINT` | 需要继续校准抽取 prompt/schema version、retry、成本/延迟预算 |
| Reranker | 默认 `LexicalCrossEncoderReranker` | `Qwen3Reranker`、`HTTPReranker(endpoint, model, api_key)`，CLI/API 可通过 `memory_service_from_env` 读取 `FUSION_MEMORY_RERANKER_*` | 生产需要选择本地 Qwen 或 HTTP provider，并配置 timeout/fallback、top_n、成本/延迟策略 |
| Benchmark answer | 默认 `LocalExtractiveAnswerModel` | `OpenAICompatibleAnswerModel` | 需要 leaderboard-grade answer 模型配置，并在官方 BEAM/LongMemEval split 上验证 |
| Benchmark judge | 默认 `LexicalContainsJudge` | `OpenAICompatibleJudgeModel` | 需要 semantic judge 模型和判分 prompt 配置；需要和 leaderboard 评测口径对齐 |
| Utility scorer | 默认未训练，收集弱标签后可训练 | `LogisticUtilityScorer` | 需要真实 benchmark/dev/replay 标签校准，再决定是否从 shadow ranking 切到主排序 |

### 生产模型注入示例

```python
from fusion_memory import MemoryService
from fusion_memory.core.embedding import Qwen3EmbeddingClient
from fusion_memory.core.llm import OpenAICompatibleLLMClient
from fusion_memory.ingestion.llm_extractor import StructuredLLMExtractor
from fusion_memory.retrieval.reranker import Qwen3Reranker

llm = OpenAICompatibleLLMClient(
    "https://your-provider.example/v1/chat/completions",
    model="extractor-model",
    api_key="...",
)
extractor = StructuredLLMExtractor(llm)

embedder = Qwen3EmbeddingClient()
reranker = Qwen3Reranker()

memory = MemoryService(
    "fusion-memory.sqlite3",
    extractor=extractor,
    embedder=embedder,
    reranker=reranker,
)
```

Postgres 使用同样的模型注入方式：

```python
memory = MemoryService(
    "postgresql://user:pass@localhost:5432/fusion_memory",
    storage_backend="postgres",
    extractor=extractor,
    embedder=embedder,
    reranker=reranker,
)
```

注意：当前 Postgres migration 是 `vector(1024)`，和 Qwen3-Embedding-0.6B 对齐。未来如果换成不同维度的 embedding，必须同步修改 schema 和索引，并设计历史数据 embedding backfill。

## 配置缺口汇报

需要补齐的配置项按优先级如下：

1. LLM extractor 配置：模型名、endpoint、API key、抽取 prompt/schema version、温度、token 限制、错误重试、成本/延迟统计。
2. Embedding provider 生产策略：当前本机 Qwen 路径已配置，仍需 timeout、retry、历史 reindex/backfill 和成本记录。
3. Reranker 生产策略：当前本机 Qwen 路径已配置，仍需 top_n、timeout、fallback、batch size 和成本/延迟统计。
4. Benchmark answer/judge 配置：BEAM 1M/10M 与 LongMemEval 使用的 answer model、judge model、prompt、官方数据路径、输出保存路径。
5. Balanced/benchmark rerank 上线策略：何时启用、top_n、timeout、fallback 策略。
6. Postgres 配置：DSN、HNSW index 参数、migration 环境、CI 中的 live smoke。
7. Auth 配置：产品侧 principal、workspace/user/agent 权限映射、拒绝策略和审计字段。
8. Utility scorer 数据配置：weak label 来源、人工/dev 标签路径、训练/验证 split、上线阈值。

短期最关键的是 LLM extractor。当前默认规则 extractor 能跑通架构，但抽取质量不会代表生产效果。

## 项目结构

```text
fusion_memory/
  api/service.py                  # MemoryService 主入口
  core/                           # Scope、config、auth、embedding、LLM client、数据模型
  ingestion/                      # normalizer、extractor、EncodingGate、views、temporal
  retrieval/                      # planner、quota、RRF/MMR、rerank、evidence pack、utility scorer
  storage/                        # SQLite store、Postgres repositories、migration/verifier
  eval/                           # generic benchmark、BEAM、LongMemEval、answer/judge adapters
tests/                            # unittest 覆盖本地 MVP 和 Postgres repository facade
docs/implementation-status.md     # 更细的实现状态清单
```

## 待办

- 接入真实 embedding provider，并解决 pgvector 维度和历史 backfill。
- 接入 Memory 系统自管的真实 LLM extractor，替换默认规则 extractor 的生产路径。
- 将常驻 HTTP service wrapper 接入产品侧 API gateway 或进程管理器。
- 在 live Postgres/pgvector 环境执行 `verify-postgres`。
- 用官方 BEAM small/dev/1m/10m 和 LongMemEval 数据跑正式报告。
- 用真实 replay/dev 数据校准 Retrieval Utility Scorer，再评估是否进入主排序。
- 对 temporal parser 扩展 quarter/deadline/recurring/locales。
- 将 `Authorizer` 接入产品身份和租户权限系统。
