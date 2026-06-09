# Fusion Memory 架构拆解

日期：2026-06-08；更新：2026-06-09

目标：设计一个自研通用 Agent 记忆系统，在使用体验和 benchmark 表现上同时优于 Mem0 和 MemPalace。系统暂命名为 **Fusion Memory**。

本文对齐已有两份源码复盘文档，并在每个自研对象旁标明它借鉴或类似哪一方：

- `mem0-architecture-analysis.md`：Mem0 是 fact ledger，主 truth 是 LLM 抽取后的 memory fact。
- `mempalace-architecture-analysis.md`：MemPalace 是 evidence palace，主 truth 是原文 drawer chunk。

Fusion Memory 的设计目标不是简单拼接两者，而是把两者优秀部分分层吸收。这里的“借鉴”表示保留该方案的核心行为，“自研补强”表示 Mem0 和 MemPalace 都没有完整原生实现，需要我们升成一等对象：

```text
L0 Evidence Layer       借鉴 MemPalace，不丢原文
L1 Fact Ledger          借鉴 Mem0，保存 Agent 可直接使用的 facts
L2 Temporal/Event Graph 自研补强，建模时间、事件、顺序、更新、矛盾
L3 Views/Profile Layer  自研补强，给 Agent 低延迟读取当前状态和长期画像
L4 Retrieval/Answer Pack 自研补强，按问题类型融合召回和证据打包
```

核心原则：

- **原文永远可回溯**：所有 facts、events、views 都必须能回链 source span。
- **事实只 ADD，不隐式覆盖**：状态变化通过 `supersedes`、`contradicts`、`valid_to` 表达。
- **时间是一等对象**：相对时间必须锚定到 session time，事件顺序必须可查询。
- **检索是多源融合，不是单路 vector search**：dense、BM25、entity、graph、view 都可以独立召回。
- **派生记忆需要升格门控**：extractor 输出不能直接污染 L1/L2/L3，必须经过 EncodingGate 判断是否升格、隔离或仅保留原文。
- **原文证据必须保底入池**：temporal、contradiction、abstention、event ordering 等问题必须设置 raw evidence quota，防止派生结构错误后无法回查原文。
- **检索质量要可学习**：手写权重只作为冷启动，后续由 Retrieval Utility Scorer 用弱标签和 dev/benchmark 数据校准。
- **Agent 体验优先**：常用偏好、指令、项目状态、任务状态要能低延迟读取，不让每次回答都从原文重推。

## 0. 术语说明

| 术语 | 含义 | 对齐来源 |
|---|---|---|
| Evidence / 证据 | 可回到原文的内容，例如聊天原句、工具输出、文档片段 | 借鉴 MemPalace drawer |
| Span / 片段 | 一段可独立索引的证据单位，可以是一轮消息、文档 chunk、工具结果或摘要 | 类似 MemPalace drawer，但覆盖 Agent trace |
| Source span | 某个 fact/event/view 的原文来源，用于审计和重新抽取 | Mem0 缺失，MemPalace 强项 |
| Fact ledger / 事实账本 | 只追加长期事实，不把原文作为主事实；适合 Agent 直接使用 | 借鉴 Mem0 memory facts |
| ADD-only | 新事实只追加，不隐式覆盖旧事实；变化用关系边表达 | 借鉴 Mem0 新版方向 |
| Current view / 当前状态视图 | 从历史 facts 和 relations 折叠出来的“当前偏好/指令/任务状态” | 自研补强 |
| Entity profile / 实体画像 | 围绕用户、项目、Agent、工具或组织沉淀长期风格、偏好和稳定属性 | 借鉴 True Memory Pro L0 Personality，融入 Fusion L3 |
| Personality view / 风格画像 | 用户或 Agent 的长期沟通风格、决策偏好、工作习惯 | 借鉴 True Memory Pro L0 Personality |
| Encoding gate / 升格门控 | 判断 extractor 输出是否写入 L1/L2/L3，或只保留 L0 原文 | 借鉴 True Memory Pro encoding gate，自研规则化 |
| Event graph / 事件图 | 把事件、时间、顺序、更新、矛盾显式建成可查询结构 | 自研补强 |
| Dense retrieval | 用 embedding 向量找语义相近内容 | Mem0/MemPalace 都使用 |
| Sparse/BM25 retrieval | 用关键词、专名、数字、日期等词面匹配召回 | Mem0/MemPalace 都有类似能力 |
| Rerank | 初召回后再次精排，提升最终证据相关性 | Mem0 可选 reranker，MemPalace 有 BM25 rerank |
| Evidence pack | 给最终回答模型的一组压缩证据，不是最终答案 | 自研补强 |
| Retrieval trace | 记录每条证据来自哪一路召回、为什么入选 | Mem0 explain 和 MemPalace provenance 的扩展 |
| Retrieval utility score | 候选证据对当前问题的预计可用性分数 | 借鉴 True Memory Pro salience scorer，自研训练和校准 |

## 1. 系统定位

Fusion Memory 面向通用 Agent，而不是单纯文档 RAG。它要同时服务四类问题：

| 问题类型 | 例子 | 需要的记忆能力 |
|---|---|---|
| 当前状态 | 用户现在偏好 PostgreSQL 还是 MySQL？ | latest view + supersedes chain |
| 历史事实 | 用户什么时候第一次提到 Atlas？ | raw evidence + time index |
| Agent 行为 | 上次你建议我怎么做检索？ | assistant/agent spans + action facts |
| 时间推理 | 哪个事件先发生？后来有没有变更？ | event graph + temporal reasoning |

Mem0 对第一类和部分第三类强；MemPalace 对第二类强；两者都没有完整覆盖第四类。

## 2. 总体架构

```text
Conversation / Tool Trace / Document
  -> Ingestion Normalizer
  -> L0 Evidence Store
  -> Extractor Pipeline
      -> Fact Extractor
      -> Event Extractor
      -> Preference/Instruction Extractor
      -> Tool/Agent Action Extractor
  -> EncodingGate
  -> L1 Fact Ledger
  -> L2 Temporal/Event Graph
  -> L3 Views/Profile Layer

Query
  -> Query Planner
  -> Multi-source Candidate Generation
      -> L0 raw spans
      -> L1 facts
      -> L2 graph paths
      -> L3 views
      -> L3 entity/personality profiles
      -> BM25 / entity / exact filters
  -> Raw Evidence Quota
  -> Retrieval Utility Scorer
  -> Reranker
  -> Evidence Pack Builder
  -> Answer Context / API Result
```

系统对外提供两类主 API：

```text
add(input, scope, session_time, metadata)
search(query, scope, options)
answer_context(query, scope, budget)
get(memory_id / span_id / event_id)
history(entity/fact/session)
debug_trace(query)
```

`search` 返回结构化候选；`answer_context` 返回可直接放入 LLM prompt 的证据包；`debug_trace` 用于 benchmark 和产品调试。

## 3. 核心对象

| 对象 | 借鉴/类似对象 | 主存储 | 作用 |
|---|---|---|---|
| EvidenceSpan | 借鉴 MemPalace drawer；比 drawer 多 speaker/action 语义 | `evidence_spans` | 原文 turn/chunk/tool result，是所有记忆的证据源 |
| Fact | 借鉴 Mem0 memory；增加结构化字段和 source spans | `memory_facts` | Agent 可直接使用的长期事实 |
| Event | 类似 MemPalace KG 的 temporal triple 场景，但自动从对话/tool trace 抽取 | `events` | 行为、状态变化、决定、会议、deadline |
| Relation | 借鉴 Mem0 `linked_memory_ids` 意图和 MemPalace KG 边；落成持久关系 | `fact_relations` / `event_edges` | supersedes、contradicts、before、after、updates |
| Entity | 借鉴 Mem0 entity boost 和 MemPalace known_entities registry | `entities` | canonical entity、alias、类型、scope |
| CurrentView | Mem0/MemPalace 都没有原生等价物；类似“当前 profile cache” | `current_views` | 当前偏好、指令、项目、任务、procedures |
| EntityProfile | 借鉴 True Memory Pro L0 Personality；比 CurrentView 更偏长期画像 | `entity_profiles` | 用户、项目、Agent、工具的长期属性、风格、稳定偏好 |
| EncodingDecision | True Memory Pro EncodingGate 的 Fusion 化 | `encoding_decisions` | 记录 extractor 输出为何升格、隔离、合并或丢弃 |
| RetrievalUtilityExample | 自研补强 | `retrieval_utility_examples` | 训练和校准 retrieval utility scorer |
| RetrievalTrace | 扩展 Mem0 explain 和 MemPalace provenance | logs / debug table | 解释每条结果来自哪一路召回、如何融合和精排 |

## 4. Scope 模型

Fusion Memory 继承 Mem0 的 scope 思路。当前设计面向单产品/单工作区原型，先用用户、Agent、任务和会话隔离，避免把尚未证明必要的多租户字段提前放进核心 schema。

```text
workspace_id
user_id
agent_id
run_id
session_id
app_id
```

最低要求：

- 写入必须至少提供一个业务主体：`user_id`、`agent_id`、`workspace_id` 或 `run_id`。
- 检索默认在同一组 scope 内进行，除非调用方显式允许跨 workspace 或跨 user。
- `session_id` 用于短期上下文和 event ordering，不等价于长期 scope。
- `agent_id` 用于保存 Agent 行为、procedural memory 和 agent-specific rules。

推荐 session key：

```text
workspace_id=<w>&user_id=<u>&agent_id=<a>&run_id=<r>&session_id=<s>
```

## 5. L0 Evidence Layer

L0 的目标是解决 Mem0 最大短板：默认不长期保存完整原文。它的存储单位借鉴 MemPalace 的 drawer，但不是只服务文档 chunk，而是扩展为 Agent 场景下的 EvidenceSpan。

表：`evidence_spans`

```text
span_id uuid
workspace_id / user_id / agent_id / run_id / session_id
turn_id
speaker: user|assistant|agent|tool|system|document
content
content_hash
timestamp
source_uri
line_start / line_end
parent_span_id
span_type: turn|window|tool_result|document_chunk|summary
entities[]
topics[]
embedding_dense
embedding_sparse
created_at
```

字段说明：

- `span_id`：证据片段 id。它类似 MemPalace drawer id，是后续 fact/event/view 回链原文的核心。
- `speaker`：谁产生了这段内容。它回答“这是用户说的、助手说的、工具返回的，还是系统/文档提供的”。
- `span_type`：这段内容是什么形态。它回答“这是原始一轮对话、长文档 chunk、工具结果、会话窗口，还是摘要”。
- `parent_span_id`：派生来源。例如一个 summary span 来自多个原文 turn，一个 document chunk 来自原始 document，一个 window span 来自一组 turn。MVP 可先用单值；需要表达多来源时升级为 `parent_span_ids[]`。
- `embedding_dense`：语义向量，用于“意思相近”的召回。
- `embedding_sparse`：关键词/BM25 向量或索引，用于专名、日期、数字、工具名等词面召回。

`speaker` 和 `span_type` 都需要保留，因为它们表达不同维度：

```text
speaker=user, span_type=turn              用户原话
speaker=assistant, span_type=turn         助手建议
speaker=tool, span_type=tool_result       工具输出
speaker=document, span_type=document_chunk 文档片段
speaker=user, span_type=summary           用户相关会话摘要
```

写入行为：

1. 对 chat 保留 turn 级 span。
2. 对长文档生成 chunk span，默认 800-1200 tokens，overlap 120-180 tokens。
3. 对长 session 生成 window span，便于 summarization 和 multi-session retrieval。
4. 对 tool result 保留原始输入、输出、状态码、时间。
5. 所有 L1/L2/L3 派生对象必须带 `source_span_ids`。

speaker-owned evidence：

- `user_owned`：用户陈述和偏好。
- `assistant_owned`：助手建议、解释、承诺。
- `agent_action_owned`：Agent 实际执行的动作、工具结果、文件变更。
- `system_owned`：系统指令和配置，不默认作为用户事实。

这比 Mem0 更可审计，也比 MemPalace 更适合 Agent，因为它保留 speaker/action 语义。

## 6. L1 Fact Ledger

L1 继承 Mem0 的核心优势：把对话抽成 Agent 可直接使用的长期 facts。

表：`memory_facts`

```text
fact_id uuid
workspace_id / user_id / agent_id / run_id
subject
predicate
object
text
category: preference|profile|instruction|procedure|event_fact|commitment|assistant_statement|agent_action|tool_result|world_fact
polarity: positive|negative|unknown
confidence 0..1
salience 0..1
observed_at
valid_from
valid_to nullable
source_span_ids[]
linked_fact_ids[]
embedding_dense
embedding_sparse
hash
created_at
```

`category` 字段用于告诉系统这条事实属于哪种记忆。它不是为了展示，而是为了检索、视图生成和冲突处理：

- `preference`：用户偏好，例如喜欢 PostgreSQL。
- `instruction`：长期指令，例如以后回答要简洁。
- `profile`：用户画像，例如职业、地点、关系。
- `procedure`：Agent 做事流程，类似 Mem0 procedural memory。
- `commitment`：用户或 Agent 承诺要做的事。
- `assistant_statement`：助手给出的建议或计划，不能误当成用户偏好。
- `agent_action`：Agent 已经执行过的动作。
- `tool_result`：工具调用结果。
- `world_fact`：与用户无关但本任务需要保存的背景事实。

MVP 不需要过细分类，建议先用 `preference/instruction/profile/project_state/commitment/assistant_suggestion/agent_action/tool_result/general_fact`。分类能提高 preference、instruction、current-state 类问题的稳定性；如果分类置信度低，应该落到 `general_fact`，不要硬分。

写入规则：

- 只 ADD 新 fact，不隐式改写旧 fact。
- 每条 fact 必须有 source span；没有证据的抽取结果丢弃或进入 low-confidence quarantine。
- 如果新 fact 更新旧 fact，写 relation，不直接覆盖旧 fact。
- 对 preference、instruction、commitment、agent_action 使用专门 extractor，避免通用抽取过宽。

推荐 LLM 输出 schema：

```json
{
  "facts": [
    {
      "local_id": "f0",
      "text": "User prefers PostgreSQL for new backend services.",
      "subject": "user",
      "predicate": "prefers",
      "object": "PostgreSQL for new backend services",
      "category": "preference",
      "confidence": 0.86,
      "source_span_ids": ["span_1"],
      "valid_from": "2026-06-08T10:00:00Z"
    }
  ],
  "relations": [
    {
      "type": "supersedes",
      "from_local_id": "f0",
      "to_fact_id": "fact_old"
    }
  ]
}
```

对比 Mem0 的改进：

| Mem0 当前行为 | Fusion Memory 改进 |
|---|---|
| 只真正使用 LLM 输出的 `text` 和 `attributed_to` | 使用结构化 subject/predicate/object/category/source/relation |
| `linked_memory_ids` 不落成图 | 写入 `fact_relations` |
| exact hash 去重 | exact hash + SimHash/MinHash + embedding dedup + relation check |
| facts 不强制带证据 | facts 必须带 source_span_ids |

## 7. L2 Temporal/Event Graph

L2 是 Fusion Memory 用来超过 Mem0 BEAM 10M 的关键层。

Mem0 有 history 和 entity store，MemPalace 有 KG、tunnel、Palace Graph，但它们都没有把“对话中的事件、时间、顺序、更新、矛盾”自动抽成统一可查询图。L2 的设计就是补这个缺口。

表：`events`

```text
event_id uuid
workspace_id / user_id / agent_id / run_id / session_id
event_type: user_action|agent_action|assistant_suggestion|tool_result|state_change|decision|deadline|meeting|preference_change
participants[]
description
time_start
time_end
time_granularity: exact|day|week|month|relative|unknown
time_source: explicit|relative_resolved|session_time|inferred
source_span_ids[]
fact_ids[]
confidence
created_at
```

表：`event_edges`

```text
edge_id uuid
from_event_id
to_event_id
edge_type: before|after|during|causes|enables|blocks|updates|resolves|mentioned_with
source_span_ids[]
confidence
```

表：`fact_relations`

```text
relation_id uuid
from_fact_id
to_fact_id
relation_type: linked_to|supersedes|contradicts|supports|derived_from|same_as
source_span_ids[]
confidence
```

为什么同时需要 `fact_relations` 和 `event_edges`：

- `fact_relations` 描述事实之间的语义关系，例如新偏好覆盖旧偏好、两个事实互相矛盾、两个 facts 是同义改写。
- `event_edges` 描述事件之间的时间、流程或因果关系，例如 A 发生在 B 之前、A 导致 B、A 解决了 B。

例子：

```text
fact_relations:
  "User prefers Qdrant for Atlas retrieval"
  supersedes
  "User prefers Chroma for Atlas retrieval"

event_edges:
  "User tested Chroma on Monday"
  before
  "User switched Atlas to Qdrant on Friday"
```

BEAM 的 contradiction/current-state 问题更依赖 `fact_relations`；event_ordering/temporal_reasoning 更依赖 `event_edges`。

Temporal normalizer：

- 规则优先解析 `today`、`yesterday`、`last week`、`next month`、ISO date、slash date、month names。
- 规则无法覆盖时，用 LLM 输出结构化时间，不允许直接生成自由文本。
- 相对时间必须锚定到 `session_time`，不能用当前系统时间替代历史对话时间。

示例：

```text
User said on 2026-06-08: "I switched Atlas from Chroma to Qdrant last week."
```

L2 写入：

```text
event_type = state_change
description = "Atlas retrieval backend switched from Chroma to Qdrant"
time_start = week of 2026-06-01
edge = updates(old_event_or_fact)
source_span_ids = [...]
```

## 8. L3 Views/Profile Layer

L3 解决产品体验：Agent 不应该每次都从全历史临时推理当前状态，也不应该把长期画像混在一次性事实检索里。L3 分为 Current-State Views 和 Entity/Profile Views 两类。

### 8.1 Current-State Views

表：`current_views`

```text
view_id uuid
workspace_id / user_id / agent_id
view_type: current_preferences|standing_instructions|active_projects|open_commitments|recent_agent_actions|procedural_memory|profile
subject
text
state_json
source_fact_ids[]
source_event_ids[]
source_span_ids[]
confidence
updated_at
expires_at nullable
```

生成规则：

- 从 L1 facts 和 L2 relations 折叠。
- `supersedes` 后的旧 fact 不进入 current view，但保留为历史。
- `contradicts` 未解决时降低 confidence，并要求回答时带冲突证据。
- view 可异步更新；关键写入可以同步更新。

CurrentView 借鉴的是 Mem0 fact 可直接使用的优点，但它不是 Mem0 原生能力。Mem0 搜索后仍需要临时判断哪个 fact 最新；CurrentView 把这个判断提前物化，作为低延迟“当前状态缓存”。MemPalace 的 closet 更像 source pointer，不是 current-state view。

典型 view：

```json
{
  "view_type": "current_preferences",
  "subject": "user",
  "text": "User currently prefers PostgreSQL for new backend services.",
  "source_fact_ids": ["fact_1", "fact_2"],
  "source_span_ids": ["span_7"],
  "confidence": 0.91
}
```

### 8.2 EntityProfile / PersonalityView

EntityProfile 负责长期画像，不替代 CurrentView。CurrentView 关注“现在应该按哪个状态执行”，EntityProfile 关注“这个主体长期是什么样”。它主要学习 True Memory Pro 的 L0 Personality 思路，但必须保持 source attribution 和可重算。

表：`entity_profiles`

```text
profile_id uuid
workspace_id / user_id / agent_id / run_id
entity_id
entity_type: user|agent|project|tool|organization|document_source
profile_type: personality|preference_pattern|work_style|communication_style|domain_expertise|stable_attributes
text
state_json
source_fact_ids[]
source_event_ids[]
source_span_ids[]
confidence
support_count
last_observed_at
updated_at
expires_at nullable
embedding_dense
embedding_sparse
```

生成规则：

- 只从多次出现、跨 session 稳定、或用户明确要求记住的信息生成 profile。
- 单次事实优先留在 L1 fact，不直接升格为 profile。
- 与 CurrentView 冲突时，CurrentView 优先；profile 只能作为 prior，不得覆盖明确的新偏好。
- profile 必须记录 `support_count` 和来源，支持后台重算。

典型 profile：

```json
{
  "entity_type": "user",
  "profile_type": "communication_style",
  "text": "User tends to prefer concise technical answers with implementation details and explicit tradeoffs.",
  "support_count": 8,
  "source_fact_ids": ["fact_10", "fact_32"],
  "source_span_ids": ["span_91", "span_147"],
  "confidence": 0.84
}
```

## 9. 写入流程

Fusion Memory 的写入行为：

```text
add(input, scope, session_time, metadata)
  1. validate scope and normalize messages/documents/tool traces
  2. assign session_id, turn_id, span_id
  3. write raw turns/chunks/tool results to L0 evidence_spans
  4. build short session windows and speaker-owned spans
  5. retrieve nearby facts/events/spans as extraction context
  6. run structured extractors
  7. normalize entities and dates
  8. run EncodingGate on facts/events/relations/profile candidates
  9. write accepted L1 facts
  10. write accepted L2 events and relations
  11. update affected L3 current views and entity profiles
  12. quarantine low-confidence candidates for audit/retry
  13. emit debug trace and audit records
```

Extractor pipeline：

| Extractor | 输入 | 输出 |
|---|---|---|
| Fact extractor | new spans + related facts | profile/preference/instruction facts |
| Event extractor | new spans + session time | events + event edges |
| Agent action extractor | assistant/tool/agent spans | suggestions/actions/tool_result facts |
| Profile extractor | accepted facts + repeated evidence | entity/personality profile candidates |
| Temporal normalizer | raw time mentions | normalized dates/intervals |
| Relation detector | new facts + related old facts | supersedes/contradicts/linked_to |

失败策略：

- L0 写入失败：整个 add 失败，因为没有证据不能写派生记忆。
- L1/L2 抽取失败：保留 L0，标记 pending extraction，可后台重试。
- EncodingGate 拒绝升格：不丢 L0，只记录 `encoding_decisions`，必要时进入 quarantine。
- relation 不确定：先写 `linked_to` 或 low-confidence relation，不强行 supersede。

### 9.1 EncodingGate

EncodingGate 位于 extractor 之后、写入 L1/L2/L3 之前。它的目标不是提高召回，而是减少派生记忆污染。

表：`encoding_decisions`

```text
decision_id uuid
workspace_id / user_id / agent_id / run_id / session_id
candidate_type: fact|event|relation|current_view|entity_profile
candidate_json
source_span_ids[]
decision: accept|merge|update_relation|quarantine|reject
reason_codes[]
scores_json
matched_existing_ids[]
created_at
```

冷启动规则：

```text
accept if:
  confidence >= 0.70
  and source_span_ids not empty
  and salience >= 0.35
  and novelty >= 0.25

quarantine if:
  confidence in [0.45, 0.70)
  or relation_type is supersedes/contradicts with confidence < 0.75
  or temporal normalization is inferred and query-critical

reject if:
  source_span_ids empty
  or speech_act is greeting/backchannel/acknowledgement
  or duplicate_score >= 0.92 and no new relation
  or candidate is assistant/system content incorrectly attributed to user
```

建议特征：

- `confidence`：extractor 自报置信度，必须被校准，不能单独决定写入。
- `novelty`：与同 scope 现有 facts/events 的 embedding distance、SimHash/MinHash、gzip delta。
- `salience`：是否是偏好、指令、承诺、状态变化、决策、时间事件、工具结果。
- `source_quality`：source span 是否是用户/agent/tool 原文，是否来自 summary，是否有明确时间。
- `speaker_attribution`：用户事实不能从 assistant/system 句子中直接生成。
- `temporal_quality`：相对时间是否可锚定，是否有明确 session_time。
- `relation_confidence`：supersedes/contradicts/update 是否有旧事实和新事实双方证据。

EncodingGate 输出不是简单丢弃。被拒绝的候选仍可通过 L0 raw evidence 被检索到；被 quarantine 的候选可后台重跑或人工检查；被 merge 的候选写入 `same_as` 或更新 relation，不重复污染 L1。

## 10. 检索流程

检索先做 query planning。

Query plan schema：

```json
{
  "query_type": "temporal_lookup",
  "entities": ["Atlas", "Qdrant"],
  "time_constraints": [{"type": "relative", "text": "last week"}],
  "speaker_focus": "user|assistant|agent|tool|any",
  "needs_current_state": false,
  "needs_source_evidence": true
}
```

query_type：

```text
factual_exact
preference
instruction
assistant_reference
agent_action
temporal_lookup
event_ordering
multi_session_reasoning
knowledge_update
contradiction_resolution
abstention
summarization
```

Candidate generation 至少跑六路：

```text
A. L1 fact dense search
B. L1 fact BM25 / sparse search
C. L0 raw evidence dense + BM25 search
D. L2 graph expansion by entity/time/session/event edges
E. L3 current view lookup
F. L3 entity/personality profile lookup
```

关键改进：BM25、raw evidence、graph、view 都能独立召回候选，不再像 Mem0 那样只让 semantic results 决定候选池。

### 10.1 Raw Evidence Quota

Raw Evidence Quota 是检索阶段的硬规则，发生在 candidate generation 之后、最终打包之前。它保证 L0 原文证据不会被派生事实、view 或 profile 完全挤掉。

默认 quota：

| query_type | source_spans 最低数量 | 规则 |
|---|---:|---|
| factual_exact | 2 | 至少保留 BM25 或 exact 命中的原文 |
| temporal_lookup | 4 | 至少覆盖不同时间点，按 time_start 排序 |
| event_ordering | 6 | 至少覆盖候选事件链的前后节点 |
| contradiction_resolution | 4 | 至少覆盖冲突双方的新旧证据 |
| knowledge_update | 4 | 至少覆盖旧状态、新状态和更新触发 span |
| multi_session_reasoning | 4 | 至少覆盖两个 session，除非证据只有一个 session |
| abstention | 3 | 若不足 quota，answer_policy 必须倾向 abstain |
| summarization | 6 | MMR 保留多时间点和多主题原文 |
| preference/instruction | 1 | 当前 view 可优先，但必须带至少一条 source span |

quota 不是盲目塞上下文。候选原文仍需通过 scope、speaker、时间和基本相关性过滤；如果过滤后不足 quota，Evidence Pack 记录 `coverage_insufficient=true`，不能让 answer model 猜。

## 11. Fusion scoring

先做 RRF 合并，再按 query_type 调权。手写权重是冷启动策略；系统进入 benchmark/dev 校准后，逐步切换到 Retrieval Utility Scorer。

RRF 是 Reciprocal Rank Fusion，意思是“按多个召回列表的名次做融合”。它不强依赖不同检索器的原始分数是否可比，只看某条候选在各路召回中排得靠不靠前。直觉是：如果一条证据同时被 dense、BM25、graph 都排在前面，它就更可靠。

基础分：

```text
score =
  0.28 * semantic_score
  + 0.22 * bm25_score
  + 0.16 * entity_overlap
  + 0.16 * temporal_fit
  + 0.12 * graph_proximity
  + 0.06 * view_or_profile_prior
```

各项含义：

- `semantic_score`：语义相似度，来自 dense embedding。
- `bm25_score`：关键词匹配分，适合专名、日期、数字、工具名。
- `entity_overlap`：查询实体和候选实体的重合度。
- `temporal_fit`：候选时间是否满足 query 的时间约束。
- `graph_proximity`：候选和 query entities/events 在关系图里的距离。
- `view_or_profile_prior`：候选是否来自 CurrentView、EntityProfile，或被它们支持。CurrentView 只对“当前偏好/当前指令/当前项目状态”类问题重要；EntityProfile 只作为长期画像 prior，历史查询中应降低或关闭。

动态调权：

| query_type | 权重变化 |
|---|---|
| temporal_lookup | 提高 temporal_fit 和 source_span recency |
| event_ordering | 提高 event edge 和 chronological consistency |
| preference/instruction | 提高 current view、entity profile、supersedes chain、confidence |
| assistant_reference | 强制 assistant/agent/tool spans 入池 |
| contradiction_resolution | 同时召回 old/new/conflicting facts |
| abstention | 增加 coverage 和 confidence threshold |

MMR 去冗余：

- 相同 source span 的重复 facts 降权。
- 同一事实的多个 paraphrase 只保留最高分。
- temporal query 保留不同时间点证据，不能只保留最近证据。

动态调权策略：

1. 先用规则和轻量 LLM 给 query 分类。
2. 根据 `query_type` 选择权重模板。
3. 召回阶段也按模板决定哪些源必须入池。
4. rerank 前用 MMR 控制重复证据。

例子：

```text
current preference query:
  提高 view_prior、supersedes chain、recency
  降低纯 semantic 权重

event ordering query:
  提高 temporal_fit、event_edges、chronological consistency
  降低 view_prior

exact extraction query:
  提高 BM25、entity_overlap
  保留 raw source spans
```

MMR 是 Maximal Marginal Relevance，最大边际相关性。它解决 top results 重复的问题：候选既要和 query 相关，也要和已选证据保持差异。公式直觉是：

```text
next = relevance_to_query - similarity_to_already_selected
```

在 temporal、summarization、contradiction 类问题中，MMR 可以避免证据包全是同一段 paraphrase，保留不同时间点或冲突双方。

### 11.1 Retrieval Utility Scorer

Retrieval Utility Scorer 负责判断“某个候选对当前问题是否有用”。它替代长期手调权重，但不替代 RRF、quota 和 hard filters。

冷启动实现：

```text
utility_score = sigmoid(
  w1 * rrf_score
  + w2 * semantic_score
  + w3 * bm25_score
  + w4 * entity_overlap
  + w5 * temporal_fit
  + w6 * graph_proximity
  + w7 * view_or_profile_prior
  + w8 * source_quality
  + w9 * contradiction_coverage
  + w10 * query_type_match
)
```

训练数据表：`retrieval_utility_examples`

```text
example_id uuid
query_id
query_text
query_type
candidate_id
candidate_type: span|fact|event|view|profile
features_json
label: useful|not_useful|unknown
label_source: weak_rule|llm_judge|human|benchmark_gold
answer_correct nullable
created_at
```

弱标签生成：

- benchmark gold source 命中或 gold answer 可由候选直接支持：`useful`。
- 同 scope 但实体、时间、speaker 全不匹配：`not_useful`。
- contradiction query 中只覆盖冲突一方：`unknown` 或低权重 `not_useful`。
- abstention query 中没有支持证据但候选被高分召回：负例。
- LLM evidence judge 只能用于 dev/benchmark 模式，不进入产品在线路径。

上线策略：

1. MVP 使用手写权重。
2. 收集 `retrieval_utility_examples` 和 debug trace。
3. 用 logistic regression / LightGBM 做离线校准。
4. 新 scorer 先 shadow run，只记录排序差异。
5. 在 dev/benchmark 和真实 query replay 上胜出后再设为 Balanced 默认。

## 12. Rerank 和 Evidence Pack

三档模式：

| 模式 | 行为 | 用途 |
|---|---|---|
| Fast | RRF + weighted score + MMR | 产品默认低延迟 |
| Balanced | cross-encoder rerank top 50 -> top 12 | 高质量搜索 |
| Benchmark | LLM judge top 20 evidence，输出选择理由 | BEAM/LongMemEval |

Evidence pack schema：

```json
{
  "query": "...",
  "answer_policy": "answer_with_evidence_or_abstain",
  "current_views": [],
  "facts": [],
  "events": [],
  "source_spans": [],
  "entity_profiles": [],
  "conflicts": [],
  "coverage": {"source_span_quota_met": true},
  "debug_trace": []
}
```

打包规则：

- 最终上下文默认 6k-8k tokens。
- 每条 fact/event 必须带 source span 摘要和 id。
- temporal query 必须按时间排序。
- contradiction query 必须同时包含冲突双方。
- abstention query 如果证据不足，明确告诉 answer model 不要猜。
- preference/instruction query 可以优先放 CurrentView，但必须附带 source span 或 profile 支持。
- EntityProfile 只能作为长期 prior，不能覆盖 query 中明确要求的历史证据。

## 13. 与 Mem0 和 MemPalace 对齐

| 能力 | Mem0 | MemPalace | Fusion Memory |
|---|---|---|---|
| 主 truth | 抽取 fact | 原文 drawer | L0 原文 + L1 fact 双 truth，fact 必须回链原文 |
| 原文保留 | 默认最近 10 条短期消息 | 长期 drawer | 长期 evidence spans |
| Agent 可用事实 | 强 | 弱，需要额外抽取 | 强，结构化 fact ledger |
| 时间/事件 | 弱 | KG 旁路，普通 mine 不自动生成 | 一等 event graph |
| 当前状态 | 弱，需要检索后推断 | 弱，需要原文推断 | materialized views |
| 长期画像 | 有 profile 类 fact，但不独立成层 | 需要从原文推断 | entity/personality profiles，作为可回溯 prior |
| 升格门控 | 主要依赖 extractor 和去重 | 原文主导，派生少 | EncodingGate 控制 fact/event/relation/profile 污染 |
| 检索候选 | semantic 主导，BM25/entity 加权 | drawer 原文召回 + BM25 | dense/BM25/raw/graph/view 多源独立召回 |
| 原文保底 | 不稳定 | 强 | query-type raw evidence quota |
| 检索学习 | 可选 reranker | BM25/图召回为主 | retrieval utility scorer 可校准 |
| 审计 | history | drawer provenance | source span + audit + relation trace |
| Benchmark 弱项 | BEAM 10M temporal/event/multi-session | 缺 fact/state layer | 定向补强两者弱点 |

## 14. BEAM 提升机制

Mem0 BEAM 10M 公开弱项：

```text
temporal_reasoning: 16.3
event_ordering: 20.2
multi_session_reasoning: 26.1
contradiction_resolution: 32.5
```

Fusion Memory 对应机制：

| BEAM 类别 | 机制 |
|---|---|
| temporal_reasoning | relative time normalization + event time index + temporal rerank |
| event_ordering | event_edges before/after/during + chronological evidence pack |
| multi_session_reasoning | session windows + graph expansion + source spans |
| contradiction_resolution | supersedes/contradicts relations + old/new evidence |
| information_extraction | raw evidence fallback + BM25 independent recall |
| summarization | session summaries + MMR evidence pack |
| abstention | evidence confidence + coverage check |
| preference_following | current view + entity/personality profile |

目标不是每类都赢，而是保住 Mem0 的 preference/instruction 强项，并在 10M 的结构性弱项上拿主要增量。

## 15. 使用体验目标

产品体验要优于二者：

- 用户明确说“记住”：立即写 L1 fact 和 L0 evidence。
- 用户问“我之前说过什么”：优先给 source span，可回溯原文。
- 用户问“现在应该按哪个偏好”：优先读 current view，并说明如果存在冲突。
- 用户问“上次你建议了什么”：检索 assistant/agent spans，不污染成用户事实。
- Agent 执行工具：保存 tool result 和 agent action fact，后续可查“我做过什么”。
- Debug UI 可以展示：命中 fact、命中原文、关系边、时间解析、最终证据包。

## 16. 存储选型

MVP：

```text
Postgres
  evidence_spans
  memory_facts
  events
  fact_relations
  event_edges
  current_views
  entity_profiles
  encoding_decisions
  retrieval_utility_examples
  entities

pgvector
  dense vectors

Postgres FTS / Tantivy / SQLite FTS5
  BM25 sparse retrieval
```

高吞吐版本：

```text
Qdrant / Milvus: dense + sparse vector
Postgres: metadata, relations, views, audit
OpenSearch / Tantivy: BM25
Redis: hot current views and query cache
```

## 17. 评测计划

Ablation：

| Variant | 目的 |
|---|---|
| Mem0 baseline | 官方对照 |
| L0 only | 测原文 evidence 召回上限 |
| L1 only | 测 fact ledger 基线 |
| L0 + L1 | 测抽取遗漏恢复 |
| L0 + L1 + EncodingGate | 测 memory 污染下降、事实准确率和召回损失 |
| L0 + L1 + L2 | 测 temporal/event/multi-session |
| Full with L3/L4/Profile | 测最终体验和 benchmark |
| Full + Utility Scorer | 测学习式排序相对手写权重的收益 |

报告指标：

```text
overall score
per-category score
tokens/query
latency p50/p95
LLM calls/query
evidence recall
abstention precision
debug trace coverage
encoding accept/reject precision
raw evidence quota hit rate
utility scorer AUC/NDCG/MRR
```

防过拟合：

- 固定 query planner prompt 和 rerank prompt 后跑正式集。
- 只在 dev/small 上调权。
- 报告失败案例，不只报 overall。
- 所有 benchmark answer 都必须能追踪到 evidence pack。

## 18. MVP 路线

Week 1：L0 + L1

- evidence span store。
- Mem0-style ADD-only extraction。
- EncodingGate v0：source attribution、speech-act filter、duplicate filter。
- facts 强制 source attribution。
- dense + BM25 检索。

Week 2：Query planner + fusion retrieval

- query type classifier。
- 六路候选召回。
- raw evidence quota v0。
- RRF + weighted scoring + MMR。
- debug trace。

Week 3：L2 temporal/event graph

- events / event_edges / fact_relations schema。
- relative time normalizer。
- supersedes / contradicts detector。
- temporal evidence pack。

Week 4：L3 views/profile

- current_preferences。
- standing_instructions。
- active_projects。
- open_commitments。
- procedural memory view。
- entity_profiles / personality_views。

Week 5：Benchmark hardening

- BEAM adapter。
- per-category ablation。
- retrieval_utility_examples 采集。
- Retrieval Utility Scorer shadow run。
- latency/token budget 优化。
- answer context packing。

## 19. 风险

- LLM 抽取 event/relation 错误会导致结构化幻觉，所以必须有 source span 和 confidence。
- EncodingGate 如果过严会损伤 recall，所以必须单独报告 accept/reject precision 和下游 QA 变化。
- 多源检索容易引入噪声，必须依赖 query planner 和 MMR 控制证据包。
- Retrieval Utility Scorer 可能过拟合 benchmark，所以必须保留手写权重 fallback 和真实 query replay。
- current view 如果更新错误，会影响产品体验，因此必须可回溯、可重算。
- BEAM 提升不等于真实体验提升，必须同时测真实 Agent 工作流。
- MemPalace 的 retrieval recall 不能直接等价为 QA score，不能把原文召回强当作最终回答强。

## 20. 最终架构判断

Fusion Memory 的主体不是 Mem0，也不是 MemPalace，而是一个分层系统：

```text
MemPalace-style L0 makes memory auditable.
Mem0-style L1 makes memory actionable.
Fusion L2 makes memory temporal and relational.
Fusion L3 makes memory usable in real-time Agent loops.
Fusion L4 makes retrieval benchmark-aware without losing product traceability.
```

目标体验：

- 像 Mem0 一样能直接记住用户偏好和 Agent 行为。
- 像 MemPalace 一样能回到原文证据。
- 比两者都更擅长时间、事件、多 session、更新和矛盾。
- 每个回答都能解释“为什么召回这些记忆”。

这才是同时超过 Mem0 和 MemPalace 的可行路线。
