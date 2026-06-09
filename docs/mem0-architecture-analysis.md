# Mem0 架构拆解

本文基于本机 `/home/wwb/memory/mem0` 源码复核，目标不是复述函数调用，而是说明 Mem0 在写入、保存、检索、更新和删除记忆时实际做了什么行为，以及这些行为对 Agent 记忆系统设计意味着什么。

复核版本信息：

- GitHub `mem0ai/mem0` 的 `main` commit：`366945965df43aa7084be98d1b5073b62a20b431`
- `pyproject.toml` package 版本：`mem0ai 2.0.4`
- 重点源码位置：`mem0/memory/main.py`、`mem0/memory/storage.py`、`mem0/configs/prompts.py`、`mem0/utils/entity_extraction.py`、`mem0/utils/lemmatization.py`、`mem0/utils/scoring.py`、`mem0/vector_stores/qdrant.py`、`mem0/configs/base.py`

核心结论：Mem0 的主事实来源不是完整原始对话，而是 LLM 从输入消息里抽取出来的简短 memory fact。原始消息只作为下一轮抽取的近邻上下文被短期保留，默认每个 scope 保存最近 10 条。SQLite history 记录 ADD、UPDATE、DELETE 的审计过程；entity store 和 BM25 只是搜索加权信号，不构成事实图谱，也不替代主 memory collection。

## 1. 系统总体行为

Mem0 写入一段对话时，会先确认这段记忆属于哪个隔离范围，例如某个用户、某个 Agent 或某次运行。随后它把输入消息规范化成标准聊天消息，并根据 `infer` 参数选择两种完全不同的写入语义：

- `infer=True` 是默认模式。系统会把新消息、同一 scope 最近保存的消息、以及向量库中语义相近的旧记忆一起交给 LLM，让 LLM 只抽取值得长期保存的事实。最后保存的是 LLM 输出的短事实句，而不是原始对话。
- `infer=False` 是原句保存模式。系统不让 LLM 解释或压缩输入，而是把每条非 system message 的内容直接作为一个 memory point 写入向量库。

检索时，Mem0 不是在原始对话中查找答案。它只在已经写入的 memory facts 上做语义召回；如果 Qdrant collection 支持 BM25 sparse vector，就额外计算关键词匹配分；如果 entity store 中存在与查询实体相近的实体点，就给这些实体链接的 memory 加权。最终候选集仍然来自语义召回，BM25 和实体只影响排序，不会单独把一个没有被语义召回命中的 memory 拉进结果集。

## 2. 主数据对象

| 对象 | 存储位置 | 系统实际保存什么 | 架构含义 |
|---|---|---|---|
| Memory | 主 vector collection，默认 `mem0` | LLM 抽取或原句写入的记忆文本、embedding、hash、词形还原文本、时间戳、scope 和自定义 metadata | 这是 Mem0 的主 truth。回答和搜索围绕这些 fact 展开 |
| Message | SQLite `messages` 表 | 同一 scope 的最近聊天消息，默认最多 10 条 | 这是下一轮抽取的短期上下文，不是长期 transcript |
| History | SQLite `history` 表 | 每条 memory 的 ADD、UPDATE、DELETE 审计记录 | 用于追踪生命周期，不参与搜索排序 |
| Entity | 辅助 vector collection，默认 `<collection>_entities` | 从 memory 文本抽出的实体、实体类型、链接到的 memory id 列表 | 用于 search boost，不是知识图谱 |
| Scope | payload、filter、SQLite session key | `user_id`、`agent_id`、`run_id` 中至少一个 | 决定写入隔离、检索隔离和短期消息上下文隔离 |

Memory 搜索结果里的 `memory` 字段来自 payload 的 `data`。`user_id`、`agent_id`、`run_id`、`actor_id`、`role` 会被提升到结果顶层；`data`、`hash`、`created_at`、`updated_at`、`text_lemmatized`、`attributed_to` 等核心字段不会混入 metadata。

## 3. 初始化时 Mem0 组装了哪些能力

创建 `Memory()` 时，系统会根据配置创建四类组件：

- 一个 embedding provider，用于把新增记忆、查询文本和实体文本转成向量。
- 一个 vector store provider，默认是 Qdrant，本地路径默认 `/tmp/qdrant`，主 collection 默认 `mem0`。
- 一个 LLM provider，默认使用 OpenAI，用于把对话压缩成长期事实，以及生成 procedural memory。
- 一个 SQLite manager，默认数据库路径是 `~/.mem0/history.db`，里面同时放 history 和短期 messages。

reranker 是可选组件，只有配置后且 search 请求启用 rerank 时才使用。

entity store 是懒加载的。也就是说，系统不会在初始化时马上创建实体 collection；只有写入或搜索真的需要实体索引时，才复制主 vector store 的配置，把 collection 名改成 `<主 collection>_entities`。在 Qdrant embedded 场景下，它会复用主 Qdrant client，避免同一路径被多个 client 锁住。

## 4. Scope、metadata 和过滤语义

Mem0 要求每次写入或搜索至少携带一个隔离标识：`user_id`、`agent_id` 或 `run_id`。这些标识会被 trim 和校验，不能为空，也不能含空白字符。

写入时，系统会同时生成两份信息：

- 一份写入 memory payload，成为这条记忆长期携带的 metadata。
- 一份用于查询的 filter，后续读取旧记忆、读取实体索引、搜索结果时都用它限制范围。

`user_id`、`agent_id`、`run_id` 会同时进入 payload 和 filter。`actor_id` 更像“消息是谁说的”或“动作是谁发起的”这类事件属性，它在部分路径中进入 query filters 或 per-message metadata，但不是与 `user_id` 同级的主隔离键。

短期消息上下文使用一个由 scope 拼出来的字符串作为 session key。Mem0 会把存在的 scope key 按固定顺序拼接，例如同时有 agent 和 user 时，形成类似：

```text
agent_id=agent_research&user_id=u_alice
```

这意味着同一用户在不同 agent 下、同一 agent 在不同 run 下，可以拥有不同的最近消息上下文。

## 5. 输入消息如何被理解

Mem0 接受三类输入：

- 字符串会被理解成一条 user message。
- 单个 dict 会被包成单元素消息列表。
- `list[dict]` 会作为多轮消息进入处理。

其他类型会被拒绝。`memory_type` 目前只显式接受 `procedural_memory`。如果传入 agent scope 且 memory type 是 procedural memory，系统会进入专门的流程，让 LLM 总结“这个 Agent 以后应该如何做事”。否则走普通事实记忆流程。

如果配置启用 vision，系统会把图片类消息交给 LLM 解析并规范化；未启用 vision 时，也会执行普通的消息规范化，确保后续看到的是统一的消息结构。

## 6. `infer=False`：把输入当作记忆本身

当调用方关闭推断时，Mem0 不会让 LLM 判断哪些内容值得记，也不会读取相似旧记忆，更不会建立实体链接或保存最近 messages。它的行为非常直接：

1. 系统逐条检查输入消息，跳过格式不完整的消息，也跳过 system message。
2. 对每条可保存的消息，复制调用方传入的 metadata。
3. 把消息角色写入 payload 的 `role` 字段；如果消息有 `name`，把它写成 `actor_id`。
4. 用整条 `content` 生成 embedding。
5. 生成新的 UUID，把原文、hash、词形还原文本、创建时间、更新时间和 metadata 一起写入主 vector collection。
6. 在 SQLite history 中写一条 ADD 审计。
7. 返回刚刚创建的 memory id、原文、事件类型和角色信息。

这个模式适合“我明确要求你记住这句话”的场景。但它仍然不是 transcript 存储系统：每条 message 会变成一个 memory point，而不是被放入可回放的对话日志。

## 7. `infer=True`：默认长期事实抽取流程

默认写入模式的重点是“从新对话里抽取长期有价值的 facts”。它不是把对话整体塞进向量库，也不是对旧记忆做复杂合并。实际行为可以理解为九个阶段。

### 7.1 先收集本轮抽取所需上下文

系统先根据当前 scope 找到 SQLite 中同一 session 最近保存的消息，最多 10 条。这些消息按时间恢复成自然顺序后，会被放进 LLM prompt，帮助模型理解新输入延续了什么上下文。

同时，系统会把本轮输入消息压成一段可供 embedding 和 prompt 使用的文本。这个文本既用于寻找相似旧记忆，也作为“new messages”交给 LLM。

关键点：最近 messages 是抽取辅助材料。它们不是主记忆，不会在 search 里被召回，也默认只保留很短窗口。

### 7.2 再找相似旧记忆，提醒 LLM 不要重复抽取

系统用本轮输入文本生成查询 embedding，并在同一 scope 的主 memory collection 中搜索最相近的 10 条旧记忆。搜索出来的旧记忆不会直接被修改，而是被整理成一个短列表交给 LLM。

为了降低 LLM 编造真实 UUID 的风险，Mem0 不把真实 memory id 暴露给模型，而是把旧记忆映射成本轮 prompt 内部的局部整数 id，例如：

```json
[
  {"id": "0", "text": "Alice prefers green tea."},
  {"id": "1", "text": "Alice works on Atlas."}
]
```

源码中确实保存了局部 id 到真实 UUID 的映射，但当前 additive 写入路径不会把 LLM 输出的 `linked_memory_ids` 落到 memory payload，也不会形成可查询的事实关系图。因此这个列表主要用于“避免重复抽取”和“让 LLM 理解已有事实”。

### 7.3 让 LLM 只输出新增事实

Mem0 使用 additive extraction prompt。系统提示词强调：只新增值得长期保存的事实，不要重写完整对话。如果当前只有 agent scope、没有 user scope，系统会追加 agent 视角说明，使抽取结果更偏向“Agent 的记忆”而不是“用户画像”。

用户提示词会包含五类信息：

- 当前 scope 下找到的相似旧记忆。
- 本轮新增消息。
- 同一 scope 的最近消息上下文。
- 调用方传入的自定义抽取指令。
- 当前日期和观察日期。

LLM 被要求返回 JSON object。Mem0 期望 JSON 里有 `memory` 数组，每个元素至少包含 `text`，也可以包含 `attributed_to` 和 `linked_memory_ids`。后续保存时，系统只真正使用 `text` 和 `attributed_to`。

如果 LLM 调用失败，系统直接返回空结果。如果 LLM 返回不是干净 JSON，系统会尝试去掉代码块并提取 JSON；仍然失败时，本轮不写入新 memory。

### 7.4 没有抽取结果时仍然保存短期消息

如果 LLM 没抽出任何 memory，Mem0 不会写主 vector collection，也不会写 history。但它仍然会把本轮消息保存到 SQLite `messages` 表中，作为下一轮抽取时可能有用的上下文。

这个行为很重要：短期消息窗口可以让下一轮输入“接着刚才说的”时被理解，但这不代表这些消息已经成为长期记忆。

### 7.5 对抽取出来的事实批量生成 embedding

当 LLM 返回一组事实文本后，Mem0 会先过滤掉没有 `text` 的条目，然后尝试批量生成 embedding。批量失败时，它会逐条重试。单条 embedding 仍然失败的事实会被跳过，不会写入向量库。

这一步说明 Mem0 的长期记忆必须可向量检索。没有 embedding 的 fact 不进入主存储。

### 7.6 用 exact hash 做去重

Mem0 会对每条候选事实文本计算 MD5 hash。它只做精确文本级去重：

- 如果这个 hash 已经出现在刚才召回的 top 10 旧记忆里，就跳过。
- 如果这个 hash 已经出现在本轮 LLM 输出的其他事实里，也跳过。

这意味着语义重复但措辞不同的事实不会被 hash 去重自动发现。例如 “Alice likes green tea.” 和 “Alice prefers green tea.” 是不同 hash，是否避免重复主要依赖 LLM 在抽取阶段的判断。

### 7.7 组装 memory payload 并写入主向量库

每条通过去重的事实都会变成一个新的 memory point。系统为它生成 UUID，payload 至少包含：

```text
data = LLM 抽取的事实文本
text_lemmatized = 面向 BM25 的词形还原文本
hash = md5(data)
created_at = 当前 UTC 时间，除非 metadata 已给出
updated_at = created_at
user_id / agent_id / run_id
custom metadata
attributed_to = LLM 输出该字段时写入
```

主向量写入优先使用批量插入。批量插入失败时，系统会逐条 fallback。默认 Qdrant 会保存 dense vector；如果 collection 有 BM25 sparse slot 且 BM25 编码可用，还会同时保存基于 `text_lemmatized` 的 sparse vector。

### 7.8 写入 history 审计

成功准备写入的每条 memory 都会获得一条 ADD history。审计记录保存 memory id、新 memory 文本、事件类型、创建时间和删除标记。这个 history 表不参与搜索，它的作用是让调用方能查某条 memory 的生命周期。

如果批量 history 写入失败，系统会逐条重试。

### 7.9 抽实体并维护实体辅助索引

主 memory 写入后，Mem0 会尝试从新 facts 中抽实体。实体抽取依赖 spaCy；如果 spaCy 不可用，抽取结果为空，写入主 memory 仍然成功，只是没有 entity boost 能力。

实体维护行为如下：

1. 系统对本轮所有新 memory 文本批量抽实体。
2. 用实体文本的小写形式做全局去重，把同一实体链接到本轮所有相关 memory id。
3. 为每个唯一实体生成 embedding。
4. 在同一 scope 的 entity collection 中搜索最相近的旧实体。
5. 如果 top1 相似度达到 0.95 或以上，系统认为它就是同一个实体，于是更新旧实体 payload 里的 `linked_memory_ids`。
6. 如果没有足够相近的旧实体，系统新建一个 entity point，payload 保存实体文本、实体类型、链接的 memory id 列表和 scope。

entity store 的内容像这样：

```json
{
  "data": "Alice",
  "entity_type": "PROPER",
  "linked_memory_ids": ["memory_uuid_1", "memory_uuid_2"],
  "user_id": "u_alice",
  "agent_id": "agent_research"
}
```

这不是知识图谱。它没有 predicate、关系边、有效时间区间或事实置信度。它只是让搜索时“查询里出现 Alice”能够给链接到 Alice 的 memory 加一点排序权重。

### 7.10 最后保存本轮消息

无论本轮抽取出多少 memory，只要走的是 `infer=True` 路径，系统都会把本轮规范化后的 messages 插入 SQLite `messages` 表。插入后，它会删除同一 session_scope 下超过最近 10 条的旧消息。

因此，Mem0 对原始对话的默认态度是“短期上下文缓存”，不是“长期证据层”。

## 8. Entity 抽取规则

实体抽取不是 LLM 完成的，而是 spaCy + 规则完成的。系统会从文本里识别几类实体：

- `PROPER`：由首字母大写的专有名词、名词或形容词组成的短语。规则允许 `of`、`the`、`in`、`and` 等连接词，但会避免把句首普通大写误判成实体。
- `QUOTED`：双引号或部分单引号中的短文本。
- `COMPOUND`：名词短语中带 compound 或特定形容词修饰的多词实体。
- `NOUN`：在特定上下文中具有 compound modifier 的名词 head lemma。

抽取后还会清理 markdown、编号、尾冒号和格式残留。同一实体文本如果被多种规则识别，会按 `PROPER > COMPOUND > QUOTED > NOUN` 保留优先级更高的类型。被更长实体包含的短实体会被删除，减少搜索 boost 的噪声。

## 9. BM25 文本如何生成

Mem0 会为每条 memory 额外生成 `text_lemmatized`。生成过程是：把文本转小写，用 spaCy 做词形还原，跳过标点和停用词，只保留字母数字 token 的 lemma。

如果某个词以 `ing` 结尾且 lemma 不同，系统会同时保留原词，用来缓解 `meeting` 和 `meet` 这类词在检索中的歧义。

如果 spaCy 不可用，系统直接把原文本作为 `text_lemmatized`。这会降低 BM25 质量，但不会影响 dense semantic search。

## 10. 存储结构

### 10.1 主 memory collection

默认 Qdrant collection 名为 `mem0`。每个 point 的语义是“一条长期记忆事实”。典型内容如下：

```text
id = uuid4
dense vector = embedding(data)
sparse bm25 vector = BM25(text_lemmatized)，如果 collection 支持
payload = {
  data,
  hash,
  text_lemmatized,
  created_at,
  updated_at,
  user_id / agent_id / run_id,
  actor_id / role,
  attributed_to,
  custom metadata
}
```

如果旧 collection 没有 BM25 sparse slot，关键词搜索会返回空，语义搜索仍然正常。

### 10.2 Entity collection

默认实体 collection 名为 `<主 collection>_entities`，例如 `mem0_entities`。每个 point 的语义是“一个用于搜索加权的实体索引项”。

```text
id = uuid4
vector = embedding(entity_text)
payload = {
  data,
  entity_type,
  linked_memory_ids,
  user_id / agent_id / run_id
}
```

它只知道某个实体链接了哪些 memory，不知道这些 memory 如何陈述该实体。

### 10.3 SQLite history 和 messages

SQLite 默认路径是 `~/.mem0/history.db`。它包含两张核心表。

`history` 表保存 memory 生命周期：

```sql
history(
  id TEXT PRIMARY KEY,
  memory_id TEXT,
  old_memory TEXT,
  new_memory TEXT,
  event TEXT,
  created_at DATETIME,
  updated_at DATETIME,
  is_deleted INTEGER,
  actor_id TEXT,
  role TEXT
)
```

`messages` 表保存短期上下文：

```sql
messages(
  id TEXT PRIMARY KEY,
  session_scope TEXT,
  role TEXT,
  content TEXT,
  name TEXT,
  created_at DATETIME
)
```

保存消息时，系统只保留同一 `session_scope` 最近 10 条。

## 11. 搜索行为

`search` 要求调用方把 `user_id`、`agent_id` 或 `run_id` 放在 `filters` 里。Mem0 会拒绝把这些 scope id 作为 search 的顶层参数传入。这样做可以让搜索 API 的过滤语义保持统一。

搜索可以携带简单等值过滤，也可以携带增强操作符，例如等于、不等于、范围、包含、大小写不敏感包含、集合包含，以及 `AND`、`OR`、`NOT` 逻辑组合。Mem0 会把这些过滤条件转换成 vector store provider 能理解的通用格式，再由具体 provider 翻译成底层查询条件。

### 11.1 先生成三类查询信号

对一条 query，Mem0 会同时准备三种检索信号：

- 原始 query 的 dense embedding，用于语义召回。
- query 的词形还原文本，用于 BM25 sparse search。
- query 中抽取出的实体，用于 entity store 搜索和加权。

这三种信号的角色不同：dense embedding 决定候选集，BM25 和 entity boost 只影响候选集内部排序。

### 11.2 语义召回决定候选池

系统会在主 memory collection 中按 dense vector 搜索，并做 over-fetch。实际拉取数量是 `max(top_k * 4, 60)`。例如用户只要 top 5，Mem0 仍然先取最多 60 条语义候选，再在内部重新打分。

如果某条 memory 没有被语义搜索召回，即使它在 BM25 中关键词命中，或者它链接了 query entity，也不会进入最终候选池。这是 Mem0 搜索设计里最重要的限制之一。

### 11.3 BM25 只提供额外排序分

如果 Qdrant collection 有 BM25 sparse slot，系统会用 query 的词形还原文本做关键词搜索。返回的 raw BM25 score 会被 sigmoid 归一化到 0 到 1 之间。

归一化参数会根据 query 长度变化。短 query 的 midpoint 较低，长 query 的 midpoint 较高，避免长查询天然获得过高关键词分。

如果 collection 没有 BM25 slot、BM25 编码失败或底层查询报错，关键词分直接缺席，搜索继续使用语义分和实体分。

### 11.4 Entity boost 提供实体相关性奖励

如果 query 中抽到了实体，系统最多取前 8 个去重实体，为它们生成 embedding，然后在 entity collection 中搜索同 scope 下的近似实体。

只有相似度达到 0.5 的实体命中才会产生 boost。boost 会分配给实体 payload 中 `linked_memory_ids` 指向的 memory。计算时会考虑一个惩罚项：某个实体链接的 memory 越多，它对单条 memory 的加权越弱。这样可以避免过宽泛实体支配排序。

同一条 memory 被多个实体命中时，只保留最高 boost。entity boost 的最大设计权重是 0.5。

### 11.5 最终打分和结果格式

Mem0 对每个语义候选先检查语义分是否低于 threshold。低于阈值的候选直接丢弃，哪怕它有 BM25 或 entity boost。

通过阈值后，系统把可用信号相加：

```text
raw_score = semantic_score + bm25_score + entity_boost
```

然后根据本条候选实际可用的最大分归一化，得到不超过 1 的 final score。最终按 final score 降序截取 top_k。

如果 `explain=True`，结果会包含 `semantic_score`、`bm25_score`、`entity_boost`、`raw_score`、`max_possible_score`、`final_score` 和 threshold，方便观察为什么某条 memory 排在前面。

如果配置了 reranker 且 search 请求开启 rerank，Mem0 会在上述排序之后再调用 reranker。reranker 失败时，系统记录 warning 并保留原排序。

最终结果形态类似：

```json
{
  "results": [
    {
      "id": "uuid",
      "memory": "Alice prefers green tea.",
      "hash": "md5",
      "created_at": "2026-...",
      "updated_at": "2026-...",
      "score": 0.82,
      "user_id": "u_alice",
      "agent_id": "agent_research",
      "metadata": {"source": "chat"},
      "score_details": {
        "semantic_score": 0.71,
        "bm25_score": 0.82,
        "entity_boost": 0.41,
        "final_score": 0.776
      }
    }
  ]
}
```

## 12. get、get_all 和 history

按 id 读取 memory 时，Mem0 直接从主 vector store 取 point，并把 payload 格式化成 API 返回对象。它不会回查 SQLite messages，也不会重建原始对话上下文。

按 filter 列出 memory 时，系统要求至少有一个 scope id，然后从 vector store 列出匹配 payload 的 points。

查看 history 时，系统读取 SQLite `history` 表，并按创建时间和更新时间排序。history 展示的是某条 memory 的审计记录，例如何时 ADD、何时 UPDATE、何时 DELETE。它不是搜索索引，也不是事实版本链查询引擎。

## 13. update、delete 和 reset 的语义

### 13.1 update 是直接改写主 memory

手动 update 某条 memory 时，系统会读取旧 payload，保留旧的 `created_at` 和已有 scope 字段，然后用新文本替换 `data`，重新计算 hash、词形还原文本和更新时间，并重新生成 embedding 写回 vector store。

随后系统在 history 中写 UPDATE，记录 old memory 和 new memory。实体索引也会被同步调整：先从旧文本抽出的实体链接中移除该 memory id，再根据新文本重新抽实体并链接。

这不是 temporal append。update 会改变主 memory point 的当前内容，只把旧内容留在 history 审计中。

### 13.2 delete 是物理删除主 memory

删除某条 memory 时，系统会先确认主 vector store 中存在该 id，然后删除这个 point，并在 history 中写 DELETE，`is_deleted=1`。

实体索引会同步移除这个 memory id。如果某个 entity 不再链接任何 memory，系统会删除该 entity point。

删除不是“把事实标记为某个时间后失效”。被删 memory 不再参与 search。

### 13.3 delete_all 和 reset

`delete_all` 必须提供至少一个 scope filter。系统先列出匹配的 memories，再逐条删除。

`reset` 会重置主 vector store；如果 entity store 已经初始化，也会重置 entity store 并清空引用。SQLite 方面，当前同步/异步实现显式 drop history 并重建 manager。`SQLiteManager.reset()` 自身能 drop history 和 messages，但 `Memory.reset()` 的调用路径需要注意实现差异，不能简单假设所有短期 messages 在所有版本路径下都被同样清理。

## 14. Procedural memory

Procedural memory 是单独的写入模式，只有当 `memory_type="procedural_memory"` 且传入 `agent_id` 时触发。它的目标不是保存用户事实，而是总结“Agent 以后执行类似任务时应遵循的步骤、偏好或流程”。

系统会把一段 procedural system prompt、本轮 messages，以及一句“Create procedural memory of the above conversation.” 一起交给 LLM。LLM 返回的 summary 会作为一条普通 memory 写入主 vector collection，只是 payload 额外带上：

```text
memory_type = procedural_memory
```

因此 procedural memory 的存储、history 和检索仍复用普通 memory 机制。区别在于抽取 prompt 和语义用途。

## 15. 端到端例子

### 15.1 第一次写入用户事实

调用方输入：

```python
from mem0 import Memory

m = Memory()

m.add(
    [
        {"role": "user", "name": "Alice", "content": "I am Alice. I prefer green tea and I work on Atlas."},
        {"role": "assistant", "content": "Noted."}
    ],
    user_id="u_alice",
    agent_id="agent_research",
    metadata={"source": "chat"},
)
```

系统会把这次写入隔离在 `u_alice + agent_research` 这个 scope 下。第一次写入时，SQLite 里没有最近消息，主 memory collection 里也没有相似旧记忆。LLM 看到新消息后，可能抽取两条长期事实：

```json
{
  "memory": [
    {"text": "Alice prefers green tea.", "attributed_to": "Alice"},
    {"text": "Alice works on Atlas.", "attributed_to": "Alice"}
  ]
}
```

Mem0 会为这两条事实分别生成 embedding、hash、词形还原文本和 UUID，然后写入主 collection。payload 大致是：

```json
{
  "data": "Alice prefers green tea.",
  "hash": "...",
  "text_lemmatized": "alice prefer green tea",
  "created_at": "2026-...Z",
  "updated_at": "2026-...Z",
  "user_id": "u_alice",
  "agent_id": "agent_research",
  "source": "chat",
  "attributed_to": "Alice"
}
```

随后 history 写两条 ADD。实体抽取可能识别出 `Alice` 和 `Atlas`，系统会在 `mem0_entities` 中创建或更新实体点，把它们链接到相关 memory id。最后，本轮 user 和 assistant 消息会进入 SQLite messages，作为同 scope 下一次抽取的近邻上下文。

### 15.2 第二次写入时如何避免完全重复

调用方继续输入：

```python
m.add(
    "Alice still prefers green tea, and Atlas now uses Qdrant BM25.",
    user_id="u_alice",
    agent_id="agent_research",
)
```

这次系统会先取出上一轮保存的最近消息，也会在主 collection 中找到相似旧记忆，例如 `Alice prefers green tea.` 和 `Alice works on Atlas.`。这些旧记忆会被放进 prompt，提醒 LLM 不要重复抽取。

如果 LLM 仍然输出完全相同的 `Alice prefers green tea.`，Mem0 会因为 hash 命中旧记忆而跳过。如果 LLM 输出新事实 `Atlas uses Qdrant BM25.`，这条会被写入新的 memory point。实体索引中已有的 `Atlas` 如果相似度足够高，就会追加链接到这条新 memory。

### 15.3 搜索时多信号如何合并

调用方搜索：

```python
m.search(
    "What does Alice use for Atlas retrieval?",
    filters={"user_id": "u_alice", "agent_id": "agent_research"},
    top_k=5,
    explain=True,
)
```

系统会用原始 query 做语义搜索，取最多 60 条候选；用词形还原后的 query 做 BM25 搜索；从 query 中抽出 `Alice` 和 `Atlas` 后，在实体索引中寻找相近实体，并给这些实体链接的 memory 加 boost。

如果 `Atlas uses Qdrant BM25.` 被语义召回命中，它可能同时得到 BM25 分和 Atlas 实体 boost，因此排到前面。返回结果会解释各个分量：

```json
{
  "results": [
    {
      "id": "mem_3_uuid",
      "memory": "Atlas uses Qdrant BM25.",
      "score": 0.78,
      "user_id": "u_alice",
      "agent_id": "agent_research",
      "score_details": {
        "semantic_score": 0.71,
        "bm25_score": 0.82,
        "entity_boost": 0.41,
        "raw_score": 1.94,
        "max_possible_score": 2.5,
        "final_score": 0.776,
        "threshold": 0.1
      }
    }
  ]
}
```

但如果这条 memory 没有进入语义候选池，BM25 或实体索引本身不会把它单独召回。

### 15.4 update、history 和 delete

如果调用方手动修正一条 memory：

```python
m.update("mem_3_uuid", "Atlas uses Qdrant BM25 and reranking.", metadata={"source": "manual_fix"})
```

系统会把主 vector point 改成新文本，重新 embedding，并在 history 中记录 old/new。实体链接也会按新文本重建。随后调用：

```python
m.history("mem_3_uuid")
```

能看到这条 memory 从 ADD 到 UPDATE 的审计记录。

如果调用：

```python
m.delete("mem_3_uuid")
```

主 memory point 会被删除，不再参与搜索；history 中会留下 DELETE；entity store 中对应的 linked id 会被移除。

### 15.5 原句保存和 procedural memory

关闭推断：

```python
m.add(
    {"role": "user", "name": "Alice", "content": "Raw note: deploy Atlas on Friday."},
    user_id="u_alice",
    infer=False,
)
```

系统不会调用 LLM，不会保存最近 messages，也不会写 entity store。它会直接把整句 `Raw note: deploy Atlas on Friday.` 写成一条 memory，payload 带 `role=user` 和 `actor_id=Alice`。

创建 procedural memory：

```python
m.add(
    [
      {"role": "user", "content": "When summarizing Atlas meetings, list decisions first and risks second."},
      {"role": "assistant", "content": "Understood."}
    ],
    agent_id="agent_research",
    memory_type="procedural_memory",
)
```

系统会让 LLM 总结这段对话中可复用的操作规程，然后把总结文本作为 memory 写入，并用 `memory_type=procedural_memory` 标记。后续可以按 `agent_id` 和 `memory_type` 过滤检索。

## 16. 与 MemPalace 的关键差异

| 维度 | Mem0 | MemPalace |
|---|---|---|
| 主 truth | LLM 抽取后的 memory fact | 原始文本 chunk / drawer |
| 原始消息 | 默认只保留同 scope 最近 10 条，用于下一轮抽取 | 长期保存原文证据 |
| 写入重点 | 从对话中抽长期事实，并把事实写成向量点 | 把文件、聊天、chunk 按 provenance 保存 |
| 去重 | top 10 相似旧记忆的 exact hash + 本批 exact hash | 更偏 source/chunk id 幂等 |
| 检索增强 | dense 召回基础上叠加 BM25 分、entity boost、可选 reranker | 原文召回、source boost、BM25 rerank、FTS fallback |
| 审计 | SQLite history 记录 memory 生命周期 | drawer provenance 和更强证据回溯 |
| 图结构 | entity store 是搜索加权索引 | KG/temporal triple 更接近事实导航层 |

Mem0 更像“事实账本 + 搜索加权”，适合 Agent 长期偏好、用户画像、指令、简短事实。MemPalace 更像“原文证据库 + 可回溯检索”，适合要求不丢原文、能追踪来源、能回答长上下文细节的问题。

## 17. 对自研 Agent 记忆系统的启发

可以直接借鉴 Mem0 的部分：

- 用 scope 严格隔离用户、Agent 和 run。
- 默认采用 additive fact 写入，避免把旧事实过早合并导致信息丢失。
- 给每条 memory 保存 hash、创建时间、更新时间、metadata 和 history。
- 在 dense retrieval 之外叠加 BM25 和实体 boost，并提供 explain 分数，便于调试检索质量。
- 把 procedural memory 作为 Agent 行为偏好单独标记，而不是混进用户事实。

需要补强的部分：

- 如果系统需要可审计证据，必须长期保存原始对话、工具结果和 source span。Mem0 默认不会帮你做这一层。
- 如果要回答 temporal reasoning 问题，需要显式建模事件、时间区间、先后关系和事实失效关系。Mem0 的 ADD history 和 entity store 不等价于 temporal graph。
- 如果要防语义重复和冲突，需要在 exact hash 之外增加语义去重、contradicts/supersedes 关系和 materialized current view。
- 如果希望关键词或实体能独立召回，不能照搬 Mem0 当前“semantic candidates only”的候选池策略，需要做真正的 multi-source fusion，例如 RRF 或 MMR。
- 如果删除语义需要保留历史真相，应使用 temporal invalidation，而不是只做主 memory point 的物理删除。
