# Fusion Memory Retrieval Engine 去 BEAM 中心化设计

- 日期: 2026-07-22
- 状态: Ready for review
- 范围: Fusion Memory 生产检索架构、`MemoryService` 瘦身、BEAM 评测隔离
- 分支: `refactor/retrieval-engine-debeam`

## 1. 背景

Fusion Memory 已具备 PostgreSQL/pgvector 持久化、模型服务池、MCP 鉴权和跨 session/workspace 的个人长期记忆能力。当前 MCP 请求链为：

```text
MCP bearer auth
  -> token subject becomes user_id
  -> FusionMemoryRuntime
  -> bounded worker + PostgreSQL transaction
  -> per-request MemoryService
  -> PostgreSQL/pgvector + model pools
```

写入保留 workspace/session provenance；读取以认证后的 `user_id` 为可见性边界，因此同一用户可跨 session/workspace 召回，不同用户互相隔离。不同用户请求可以并行；同一用户的写入继续通过 PostgreSQL advisory lock 串行化。

当前主要架构问题不在存储或 MCP，而在生产检索核心：

- `fusion_memory/api/service.py` 约 4175 行，其中约 2500 行承担候选补偿、保留、过滤和题型策略。
- `fusion_memory/api/service_helpers.py` 约 1773 行，已经承接大量从 service 抽出的检索启发式，但没有形成清晰边界。
- `retrieval/providers/` 和 `retrieval/pipeline.py` 只是部分拆分，provider 仍通过 `context.service._xxx()` 调用 `MemoryService` 私有方法。
- `BeamAdapter` 向生产 `search()` 注入 `mode="benchmark"` 和 `query_type_hint`。
- `QueryPlanner`、scoring、evidence pack 和 `MemoryService.search()` 直接理解 BEAM category，并运行 category-specific rescue/preservation chain。

结果是生产产品语义、评测题型策略和 API 编排互相缠绕。继续在此基础上增加团队记忆，会把个人记忆、团队权威记录、权限和评测逻辑耦合到同一服务模型中。

本设计先完成个人记忆检索核心的边界治理。团队记忆将在后续作为独立领域模型设计和实现。

## 2. 已确认决策

### 2.1 产品优先

Fusion Memory 是产品记忆服务，不是 BEAM 专用检索器。生产架构以真实产品查询、用户隔离和运行可靠性为约束；BEAM 只用于评测和诊断。

### 2.2 兼容边界

本次重构保持以下外部行为稳定：

- 公共服务 API 及 MCP 工具的产品语义。
- bearer token subject 到 `user_id` 的身份绑定。
- 同一用户跨 session/workspace 互通。
- 不同用户严格隔离。
- 写入事务、删除、历史、视图和 provenance 语义。
- 不同用户请求并行及同一用户写入串行化。

本次重构不要求以下内部行为逐条兼容：

- 旧检索结果的逐条一致性。
- 旧候选排序和分数的逐项一致性。
- BEAM category 对生产检索的特殊补偿行为。
- 缺少产品案例与回归测试依据的历史启发式。

BEAM 分数在重构前后保留对比，但只作为诊断数据，不约束生产模块边界。

### 2.3 渐进替换架构，不渐进继承启发式

采用独立 Retrieval Engine 的渐进替换方案。迁移按小步完成，但旧启发式不会被默认原样搬入新模块：每组逻辑必须明确归为通用产品能力、BEAM 评测能力或删除项。

## 3. 目标

- 将 `MemoryService` 收敛为公共 API facade 和应用服务编排层。
- 建立独立、可替换、可测试的生产 `ProductRetrievalEngine`。
- 让 provider/policy 依赖明确的 repository/model port，而不是依赖 `MemoryService`。
- 从生产 API、MCP schema 和核心 QueryPlanner 中移除 BEAM category 与 `benchmark` mode。
- 将 BEAM 保留为 `fusion_memory/eval/beam/` 下显式、可选的评测 profile。
- 建立小而稳定的产品检索流水线，允许基于真实产品案例继续演进。
- 为后续独立的团队记忆模型提供干净的服务和检索边界。

## 4. 非目标

- 本阶段不实现团队记忆、共享池、团队权限或飞书发布。
- 不扩展现有个人 `Scope` 来承载团队权威记录。
- 不进行数据库数据迁移。
- 不更改 MCP bearer token 的用户身份模型。
- 不重写写入、删除、历史和视图领域逻辑，除非是接入新检索边界所必需的最小调整。
- 不以恢复旧 BEAM 分数为理由向生产路径增加新的 category heuristic。
- 不在生产请求上默认双跑新旧引擎，避免模型和数据库负载翻倍；对比通过离线/测试 harness 完成。

## 5. 目标架构

```text
MCP Runtime / other product adapters
                 |
                 v
           MemoryService
           - public API facade
           - identity/context validation
           - write/delete/history/view orchestration
           - call RetrievalEngine
                 |
                 v
       ProductRetrievalEngine
           - product QueryPlanner
           - candidate providers
           - generic policies
           - fusion/rerank/selection
           - retrieval trace
                 |
                 v
       Repository / Model Ports
           - PostgreSQL / pgvector
           - embedding / reranker
           - entity / chronology stores

BEAM runner
    -> BeamAdapter
       -> MemoryService for ingestion/cleanup
       -> BeamRetrievalEngine for evaluation queries
          -> BeamQueryPlanner
          -> shared generic providers/ports
          -> eval-only BEAM policies
```

依赖方向必须始终从入口指向领域组件和基础设施接口。以下依赖被禁止：

- provider/policy 持有 `MemoryService`。
- provider/policy 调用 `MemoryService._private_method()`。
- 生产模块 import `fusion_memory.eval`。
- MCP 或生产 `search()` 根据 BEAM category 选择策略。

## 6. 核心契约

### 6.1 MemoryService

`MemoryService` 保留现有公共入口，对搜索只承担：

1. 校验认证上下文和产品请求。
2. 构造 `RetrievalContext` 与 `SearchRequest`。
3. 在现有事务边界内调用注入的 `RetrievalEngine`。
4. 将 `RetrievalResult` 映射为稳定的公共返回结构。

它不再决定候选源、题型、rescue、preservation 或最终候选组成。

### 6.2 RetrievalContext

`RetrievalContext` 是不可变的运行上下文，至少包含：

```python
@dataclass(frozen=True)
class RetrievalContext:
    user_id: str
    now: datetime
    trace_id: str
    deadline: datetime | None
```

`user_id` 必须来自鉴权层，检索引擎和 provider 均不能覆盖、推断或自动分配。workspace/session 只作为候选 provenance 存在，不成为读取隔离边界。

### 6.3 SearchRequest

`SearchRequest` 只包含产品可解释的查询约束：

```python
@dataclass(frozen=True)
class SearchRequest:
    query: str
    limit: int
    time_range: TimeRange | None = None
    include_trace: bool = False
```

正式接口中不包含 `mode="benchmark"`、`query_type_hint` 或 BEAM category metadata。

### 6.4 QueryPlan

产品 QueryPlanner 输出声明式能力计划，而不是 benchmark 题型：

```python
@dataclass(frozen=True)
class QueryPlan:
    provider_requests: tuple[ProviderRequest, ...]
    time_range: TimeRange | None
    entity_constraints: tuple[EntityConstraint, ...]
    ordering: OrderingMode
    use_reranker: bool
    result_limit: int
```

`ProviderRequest` 描述候选源、预算和显式约束。Planner 可以解析时间、实体、精确术语和顺序需求，但不能输出 `contradiction_resolution`、`multi_session_reasoning` 等 BEAM category。

### 6.5 Candidate 与 RetrievalResult

所有 provider 输出统一 `Candidate`：

- stable memory/span ID
- source kind
- content or structured payload reference
- normalized provenance
- raw provider score
- timestamps
- structured reason codes

`RetrievalResult` 包含最终候选、可选 evidence pack 和结构化 trace。分数归一化、过滤、融合与选择只处理统一候选契约。

### 6.6 Repository / Model Ports

检索组件只依赖面向能力的接口，例如：

- `MemorySearchRepository`
- `EntitySearchRepository`
- `ChronologyRepository`
- `EmbeddingPort`
- `RerankerPort`

具体 PostgreSQL、pgvector 和模型池实现由 runtime 组装。端口不得暴露服务 facade，也不得允许 provider 绕过 `user_id` 可见性约束。

## 7. 产品检索流水线

```text
SearchRequest
  -> QueryPlanner
  -> CandidateProviders
  -> Normalize / Deduplicate
  -> Visibility and constraint filters
  -> Score fusion
  -> Optional rerank
  -> Final selection
  -> RetrievalResult + Trace
```

### 7.1 Query planning

Planner 只识别产品能力信号：

- semantic recall
- lexical/exact recall
- explicit time range or recency
- explicit entities
- explicit chronology/order request
- result budget and reranker need

Planner 不负责在查询后期发起补偿检索，也不根据评测类别声明 must-preserve candidate。

### 7.2 Candidate providers

首版生产 provider 控制在五类：

| Provider | 职责 | 不负责 |
| --- | --- | --- |
| `VectorProvider` | 语义相似召回 | 题型判断、最终排序 |
| `LexicalProvider` | 名称、术语、原句等精确匹配 | 语义扩展、答案模板 |
| `TemporalProvider` | 最近记忆和明确时间范围 | event-ordering 答案构造 |
| `EntityProvider` | 围绕显式识别实体召回 | 全局 topic rescue |
| `ChronologyProvider` | 明确时间线请求的事件序列 | BEAM event-ordering 专用补偿 |

Provider 可以并行执行，但共享统一 deadline、并发上限和候选预算。

### 7.3 Normalize, filter, fusion, rerank

- 按 stable ID 和规范化内容去重。
- 强制 `user_id` 可见性、删除状态、时间范围和显式实体约束。
- 使用单一、记录得分组成的归一化与融合方法。
- reranker 最多执行一次，不在 rerank 后继续多轮 rescue/filter。
- 最终选择限制重复内容并保留 raw evidence、来源和时间信息。
- 同分结果使用稳定 tie-breaker，保证相同输入与模型结果下可复现。

### 7.4 Heuristic policy

允许的启发式：

- 解析用户明确表达的时间、实体和精确约束。
- 归一化 provider 输出。
- 抑制重复、已删除或明显低价值候选。
- 在有产品案例和回归测试时实现通用 chronology、state update 或 contradiction 能力。

禁止的启发式：

- 根据 BEAM category 启动特殊 rescue chain。
- 最终选择后多轮“保底补回”。
- 通过 query-shaped regex、答案模板或 named scenario 恢复评测分数。
- 没有产品案例和测试的 preservation rule。

## 8. BEAM 评测隔离

BEAM 移入显式评测 profile：

```text
fusion_memory/eval/beam/
  adapter.py
  engine.py
  query_planner.py
  policies/
  factory.py
```

### 8.1 BeamAdapter

`BeamAdapter` 使用 `MemoryService` 完成数据写入和清理，使用独立注入的 `BeamRetrievalEngine` 回答评测查询。它不再调用：

```python
service.search(..., budget={"mode": "benchmark", "query_type_hint": query.category})
```

### 8.2 BeamQueryPlanner

只有 `BeamQueryPlanner` 理解 BEAM category。它可以把 category 转换成标准 provider 请求，或启用 eval-only policy。category 不进入生产 QueryPlanner、scoring、evidence pack 或 MCP schema。

### 8.3 Reuse boundary

BEAM 可以复用：

- repository/model ports
- stable candidate contracts
- generic providers
- generic fusion/rerank/trace components

BEAM 不得要求生产引擎暴露 benchmark mode，也不得通过私有 service 方法复用逻辑。

### 8.4 Baseline preservation

仓库 `AGENTS.md` 中记录的两次 BEAM 100K 基线继续保留：

- `0.7751916960517531`
- `0.7676505254168324`

重构前后生成 category 与总分对比报告。分数变化用于定位被删除或迁移的能力，但不是破坏生产边界的理由。现有保留评测 artifacts 不删除、不覆盖。

## 9. 旧启发式处置规则

迁移前建立 inventory，记录每组逻辑的调用方、依赖、产品案例、BEAM category、测试和最终去向。每组只能选择以下一种结果：

### 9.1 泛化为产品能力

适用于有明确用户场景、能用产品语言解释且有回归案例的逻辑。它被改写为 provider、policy 或 selector，并依赖公开端口。

### 9.2 迁入 BEAM profile

适用于只由 BEAM category、评测 pack 或评测答案行为触发的逻辑。它保留在 `eval/beam/`，不参与生产 runtime 组装。

### 9.3 删除

适用于与其他补偿链重复、缺少产品依据、不可解释或只为恢复局部分数存在的逻辑。

禁止把旧方法整体移动到 `service_helpers.py`、`retrieval/utils.py` 或新的大型 policy 文件。文件拆分必须对应稳定职责和公开契约。

## 10. `service.py` 瘦身边界

完成后的 `service.py` 目标约 1200 行以内，但职责边界优先于机械行数。

保留内容：

- 构造和依赖注入接口。
- `add`、`ingest_turn`、`search`、`delete`、`history`、`view` 等公共服务入口。
- 事务内的应用服务编排。
- 公共输入输出映射。

迁出内容：

- query type/category 分支。
- raw/topic/aggregation/event-ordering 候选补偿。
- post-rerank preservation。
- 分数融合、候选生命周期和最终选择。
- 只为检索 trace 服务的阶段实现。

`service_helpers.py` 不作为迁出逻辑的默认承接点。其现有内容按相同 inventory 规则拆分或删除；最终只允许保留真正跨 API 编排复用、且不属于 retrieval domain 的小型 helper。

## 11. 失败处理与降级

### 11.1 Identity and storage failures

- 鉴权层先确定 `user_id`；缺失或无效身份直接拒绝。
- PostgreSQL、事务或可见性校验失败时，请求整体失败。
- 不允许在身份边界不确定时返回部分结果。

### 11.2 Provider failures

- 单个 provider 或模型服务失败时，记录结构化失败并允许其他 provider 继续。
- 例如 embedding/reranker 不可用时，可降级到 lexical/temporal/entity 结果。
- 所有被计划的有效候选源均失败时返回稳定的检索错误。
- 降级只减少能力，不触发隐藏的 heuristic rescue chain。

### 11.3 Planner failures

Planner 输出无效计划时使用固定的安全默认计划，例如受预算约束的 lexical + vector recall；默认计划不依赖 query category。

### 11.4 Deadline and cancellation

- provider 共享请求 deadline。
- 请求取消或超时后停止未完成的模型调用和 provider 工作。
- 数据库 transaction/session 必须按现有 runtime 规则释放。

### 11.5 MCP errors and process recovery

MCP 返回稳定、结构化的错误类型。服务进程崩溃拉起、远程连接断开后的本地正常运行与恢复继续由现有部署和监督机制负责，不下沉到 Retrieval Engine。

## 12. 并发模型

- 不同用户的读取和写入请求继续通过 bounded worker 并行执行。
- 同一用户写入继续使用 PostgreSQL advisory lock 串行化。
- 读取不因 session/workspace 不同而隔离，也不获取同一用户写锁。
- 单次检索内部 provider 可并行，但受统一的 per-request 并发上限限制，避免本地长期模型服务过载。
- repository port 在既有 request transaction 内执行，不允许 provider 自行创建不受 runtime 管理的长事务。

## 13. Trace 与可观测性

每次检索生成结构化 trace，至少记录：

- planner 输出与默认计划降级原因。
- 每个 provider 的预算、耗时、候选数和错误类型。
- 去重、可见性、时间和实体过滤数量。
- 分数归一化、融合、rerank 和最终选择 reason code。
- 请求总耗时和 deadline/cancellation 状态。

默认日志不记录查询全文、记忆正文、token 或模型服务凭据。调试所需内容使用 trace ID、candidate ID、计数和已脱敏 reason code 表达。

## 14. 迁移顺序

### Phase 0: 基线和 inventory

- 固定公共 API、用户隔离、跨 session/workspace、删除、历史和视图的 characterization tests。
- 保存产品案例、性能和 BEAM 诊断基线。
- 建立旧启发式 inventory 和处置结论。

### Phase 1: 新契约

- 增加 `RetrievalContext`、`SearchRequest`、`QueryPlan`、`Candidate`、`RetrievalResult` 和 port 接口。
- 用契约测试固定身份、事务和候选 provenance 语义。
- 默认生产行为暂不切换。

### Phase 2: ProductRetrievalEngine

- 实现五类基础 provider。
- 实现统一 normalize、filter、fusion、rerank、selection 和 trace。
- 通过测试/离线 harness 对比旧引擎，不在正式生产请求上默认双跑。

### Phase 3: 逐组迁移或删除

- 按 inventory 处理 service/service_helpers 中的检索逻辑。
- 有产品价值的改写为通用组件。
- BEAM-only 逻辑迁入 eval profile。
- 其余删除。

### Phase 4: 生产切换

- `MemoryService.search()` 只调用 `ProductRetrievalEngine`。
- 开发期间允许临时 legacy 开关用于离线对比和快速回退。
- 正式收尾前删除 legacy 开关和旧检索实现，避免永久维护双路径。

### Phase 5: BEAM adapter 切换

- `BeamAdapter` 改用独立 `BeamRetrievalEngine`。
- 移除生产 `query_type_hint`、`benchmark` mode 和 category 分支。
- 生成重构前后 BEAM 对比报告。

### Phase 6: 清理与文档

- 删除无调用方 helper、兼容参数和重复 trace。
- 更新 active architecture 文档和 MCP schema 文档。
- 验证生产包无 `fusion_memory.eval` 依赖。

## 15. 测试策略

### 15.1 Public contract tests

- `add/search/delete/history/view` 输入输出兼容。
- MCP 产品模式 schema 不包含 `benchmark`。
- 认证 subject 始终成为唯一读取 `user_id`。

### 15.2 Isolation and concurrency tests

- 同一用户跨 session/workspace 可以召回。
- 不同用户在并发读写下不能互相召回。
- 同一用户并发写入保持 advisory-lock 语义。
- provider 并发不会越过 request deadline 或数据库事务边界。

### 15.3 Retrieval unit and integration tests

- Product QueryPlanner 不输出 BEAM category。
- 每个 provider 的输入、预算、失败和候选契约。
- normalize、deduplicate、visibility filter、fusion、rerank、selection。
- 稳定 tie-break 和 trace reason code。
- raw evidence 与 provenance 在最终结果中可追溯。

### 15.4 Fault-injection tests

- embedding 服务不可用。
- reranker 服务不可用。
- 单 provider 超时或抛错。
- PostgreSQL 失败或事务回滚。
- 请求取消和 deadline 超时。
- planner 输出非法计划并触发安全默认计划。

### 15.5 Product case suite

建立少量真实产品查询案例，至少覆盖：

- 长期事实和偏好。
- 明确实体与精确术语。
- 时间范围与最近状态。
- 状态更新/冲突的通用产品语义。
- 事件顺序与来源追溯。

每个案例定义目标 evidence 的 top-k 期望和禁止出现的跨用户结果。产品案例失败是合并阻断项。

### 15.6 BEAM diagnostics

- BEAM adapter/profile 可以独立运行。
- 记录总分和 category 变化。
- 不以 BEAM 分数为由恢复生产 category branch。

## 16. 完成标准

- `service.py` 约 1200 行以内，且不含候选补偿链和 BEAM 分支。
- provider/policy 不持有或回调 `MemoryService`。
- 生产代码不 import `fusion_memory.eval`。
- `benchmark`、BEAM category 和专用 rescue 逻辑只存在于 `eval/beam/`、评测 runner 及相关测试/文档。
- MCP 不再公开 `benchmark` mode。
- 所有公共契约、用户隔离、跨 session/workspace、事务、并发和故障降级测试通过。
- 产品案例集全部通过。
- 对相同硬件和模型记录重构前后延迟、候选组成和召回对比；显著退化有明确处置结论。
- 保留的 BEAM artifacts 和基线未被删除或覆盖。
- 不需要数据库数据迁移。

## 17. 回滚策略

迁移阶段以小提交保持可回滚：

- 新契约和新引擎先作为未接管生产的独立路径加入。
- 生产切换提交与旧逻辑删除提交分开。
- 若切换后产品级隔离、正确性或稳定性失败，可回退生产切换提交。
- legacy 开关仅在迁移期存在；完成验收后删除，避免形成长期双实现。
- 数据模型不迁移，因此回滚不需要数据库恢复操作。

## 18. 后续团队记忆边界

团队记忆不是个人 `Scope` 的新枚举值。后续设计至少包含独立的：

- `TeamMemoryService`
- 团队权威记录数据模型
- 成员、角色和来源权限模型
- 发布/撤回/修订状态机
- `TeamRetrievalEngine`

团队检索可以复用通用候选、模型和 trace 接口，但必须使用独立的授权上下文和 repository。个人记忆的 `user_id` 可见性规则不能被团队查询绕过，团队记录也不能通过个人 `memory_add` 写入。

本次重构的价值是先让个人产品检索成为一个有明确契约的模块，使后续团队模型可以与其并列，而不是继续扩大 `MemoryService` 和个人 Scope。
