# Fusion Memory 执行方案

日期：2026-06-09

目标：把 `fusion-memory-architecture.md` 落成可执行工程计划。标准是每一层都能单独运行、单独测试、单独上线；上层能力只依赖下层稳定 API，不直接依赖下层内部实现。

## 1. 分层原则

每一层必须满足三个条件：

- 有独立 API，可以被业务或测试直接调用。
- 有独立数据表或持久对象，能被 debug 和回放。
- 有独立测试集和验收指标，失败时能定位到本层。

分层如下：

```text
Layer 0 Runtime/Foundation
  scope、配置、模型适配、任务队列、审计日志

Layer 1 Evidence Store
  原文 evidence_spans，dense/BM25 索引，get/search span

Layer 2 Extraction + EncodingGate
  extractor 输出候选，EncodingGate 决定 accept/quarantine/reject

Layer 3 Fact Ledger
  memory_facts、fact_relations，ADD-only facts，可直接 search facts

Layer 4 Temporal/Event Graph
  events、event_edges、temporal normalizer，支持时间和顺序查询

Layer 5 Views/Profile Layer
  current_views、entity_profiles、personality_views，支持低延迟状态和长期画像

Layer 6 Retrieval Pack
  query planner、多源召回、raw evidence quota、RRF/MMR、evidence pack

Layer 7 Retrieval Utility Scorer
  retrieval_utility_examples、弱标签、离线训练、shadow run、排序替换

Layer 8 Benchmark/Product Integration
  BEAM/LongMemEval adapters、answer_context API、debug UI、真实 query replay
```

实现顺序必须按层推进，但每层完成后都要能产生可用价值。例如 Layer 1 做完后就是一个可审计原文检索系统；Layer 3 做完后就是一个 Mem0-style fact memory；Layer 6 做完后才是完整 Fusion retrieval。

## 2. 建议代码结构

语言不限，但建议先用 Python 做 MVP，因为 benchmark harness、LLM 调用、数据处理和离线训练更快。后续可把在线 retrieval 服务迁移到 Go/TS/Rust。

```text
fusion_memory/
  core/
    scope.py
    ids.py
    config.py
    clock.py
    audit.py
  storage/
    db.py
    migrations/
    repositories/
      evidence_repo.py
      fact_repo.py
      event_repo.py
      view_repo.py
      profile_repo.py
      encoding_repo.py
      utility_repo.py
  models/
    embeddings.py
    rerankers.py
    llm.py
  ingestion/
    normalizer.py
    window_builder.py
    extractors.py
    temporal_normalizer.py
    encoding_gate.py
    pipeline.py
  retrieval/
    query_planner.py
    candidate_generators.py
    raw_evidence_quota.py
    rrf.py
    scoring.py
    mmr.py
    evidence_pack.py
    utility_scorer.py
  api/
    service.py
    schemas.py
  eval/
    datasets/
    beam_adapter.py
    longmemeval_adapter.py
    labels.py
    reports.py
  tests/
```

MVP 技术栈：

- Postgres 16：主元数据、关系、审计。
- pgvector：dense embedding。
- Postgres FTS：BM25/tsvector 冷启动；如果效果不够再换 Tantivy/OpenSearch。
- Redis 可选：只用于 hot current_views 和 query cache，不进入第一版强依赖。
- Embedding：先用成本低、部署简单的模型；模型接口必须可替换。
- Reranker：第一版可选；Layer 6 Balanced 模式再接 cross-encoder。

## 3. 公共 API

所有层共享这些外部 API。

```python
class MemoryService:
    def add(self, input, scope, session_time, metadata=None) -> AddResult: ...
    def search(self, query, scope, options=None) -> SearchResult: ...
    def answer_context(self, query, scope, budget=None) -> EvidencePack: ...
    def get(self, object_id, object_type=None) -> MemoryObject: ...
    def history(self, scope, entity=None, fact_id=None, session_id=None) -> HistoryResult: ...
    def debug_trace(self, trace_id) -> DebugTrace: ...
```

最小返回结构：

```python
class AddResult:
    span_ids: list[str]
    accepted_fact_ids: list[str]
    accepted_event_ids: list[str]
    updated_view_ids: list[str]
    updated_profile_ids: list[str]
    quarantined_candidate_ids: list[str]
    trace_id: str

class SearchResult:
    candidates: list[Candidate]
    trace_id: str
    coverage: dict

class EvidencePack:
    query: str
    answer_policy: str
    current_views: list[dict]
    entity_profiles: list[dict]
    facts: list[dict]
    events: list[dict]
    source_spans: list[dict]
    conflicts: list[dict]
    coverage: dict
    debug_trace: list[dict]
```

## 4. Layer 0 Runtime/Foundation

目标：所有后续模块共享统一 scope、id、配置、审计、模型适配。

### 4.1 Scope

Scope 字段：

```text
workspace_id nullable
user_id nullable
agent_id nullable
run_id nullable
session_id nullable
app_id nullable
```

规则：

- `add` 至少要求 `workspace_id/user_id/agent_id/run_id` 中一个非空。
- `session_id` 只表示一次会话或任务，不是长期隔离边界。
- 默认检索不能跨 workspace 或 user。
- `agent_id` 用于 assistant/agent 行为和 procedural memory。

验收测试：

- 缺少所有主体字段时 `add` 失败。
- 默认 `search` 不返回其他 user/workspace 的数据。
- 显式 `allow_cross_session=true` 可以跨 session，但仍不跨 user/workspace。

### 4.2 模型适配

接口：

```python
class Embedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

class Reranker:
    def score(self, query: str, docs: list[str]) -> list[float]: ...

class LLMClient:
    def structured(self, prompt: str, schema: dict, input: dict) -> dict: ...
```

要求：

- 所有模型调用必须记录 model name、latency、token/cost、prompt version。
- 测试环境必须支持 fake model，保证单元测试不依赖外部 API。
- prompt version 要写进 audit/debug trace。

## 5. Layer 1 Evidence Store

目标：长期保留原文，支持原文级 dense/BM25 检索。做完这一层后，系统已经能作为可审计 evidence search 单独使用。

### 5.1 Schema

表：`evidence_spans`

```sql
span_id uuid primary key,
workspace_id text,
user_id text,
agent_id text,
run_id text,
session_id text,
turn_id text,
speaker text not null,
span_type text not null,
content text not null,
content_hash text not null,
timestamp timestamptz,
source_uri text,
line_start int,
line_end int,
parent_span_id uuid,
entities jsonb default '[]',
topics jsonb default '[]',
embedding_dense vector,
search_tsv tsvector,
metadata jsonb default '{}',
created_at timestamptz not null default now()
```

索引：

```sql
create index evidence_scope_idx on evidence_spans(workspace_id, user_id, agent_id, run_id, session_id);
create index evidence_timestamp_idx on evidence_spans(timestamp);
create index evidence_hash_idx on evidence_spans(content_hash);
create index evidence_search_idx on evidence_spans using gin(search_tsv);
create index evidence_embedding_idx on evidence_spans using hnsw (embedding_dense vector_cosine_ops);
```

### 5.2 API

```python
add_spans(input, scope, session_time, metadata) -> list[EvidenceSpan]
get_span(span_id) -> EvidenceSpan
search_spans(query, scope, mode="hybrid", filters=None, limit=20) -> list[SpanCandidate]
```

### 5.3 写入规则

- chat 按 turn 保存。
- 长文档按 800-1200 tokens chunk，overlap 120-180 tokens。
- tool result 必须保留 tool name、arguments、status、stdout/stderr 或结构化返回。
- assistant/agent/tool/system/document 不能混成 user evidence。
- content_hash 完全重复时跳过重复写入，但仍可记录 audit link。

### 5.4 测试

单元测试：

- 同一输入重复 add 不产生重复 span。
- speaker 和 span_type 正确保存。
- parent_span_id 可回查。
- scope filter 生效。

检索测试：

- BM25 能命中专名、日期、数字、工具名。
- dense 能命中同义表达。
- hybrid 结果包含 `source=BM25|dense|hybrid` 和分数。

验收：

- `add` 1000 turns 后，随机抽样 100 条能通过 `get_span` 原文还原。
- exact/BM25 专名查询 recall@20 >= 0.90。
- dense paraphrase 查询 recall@20 >= 0.75，具体阈值可随模型调整。

## 6. Layer 2 Extraction + EncodingGate

目标：extractor 只产候选，EncodingGate 决定是否升格。做完这一层后，可以独立审计“为什么某条候选被写入或拒绝”。

### 6.1 Candidate Schema

extractor 输出统一候选格式：

```python
class MemoryCandidate:
    local_id: str
    candidate_type: Literal["fact", "event", "relation", "current_view", "entity_profile"]
    text: str
    structured: dict
    confidence: float
    source_span_ids: list[str]
    extractor_name: str
    prompt_version: str
```

### 6.2 Extractors

第一版只实现四个：

- `FactExtractor`：preference、instruction、profile、commitment、assistant_statement、agent_action、tool_result、general_fact。
- `EventExtractor`：state_change、decision、deadline、meeting、agent_action、tool_result。
- `TemporalNormalizer`：把 today/yesterday/last week 等相对时间锚定到 session_time。
- `RelationDetector`：linked_to、same_as、supersedes、contradicts、supports。

ProfileExtractor 放到 Layer 5，但候选格式复用这里。

### 6.3 EncodingGate

输入：

```text
new candidates
source spans
nearby existing facts/events/profiles
scope
session_time
```

输出：

```text
accept
merge
update_relation
quarantine
reject
```

冷启动评分：

```python
salience = rule_speech_act_score + category_prior + source_quality
novelty = max(embedding_distance, minhash_distance, gzip_delta)
duplicate_score = max(hash_match, simhash_similarity, embedding_similarity)
```

决策规则：

- `accept`：source 存在，confidence >= 0.70，salience >= 0.35，novelty >= 0.25。
- `merge`：duplicate_score >= 0.92 且没有状态变化。
- `update_relation`：新旧事实冲突或更新，relation_confidence >= 0.75。
- `quarantine`：confidence 0.45-0.70，或 relation/temporal 不确定。
- `reject`：无 source、寒暄、重复、speaker attribution 错误。

### 6.4 Schema

表：`encoding_decisions`

```sql
decision_id uuid primary key,
workspace_id text,
user_id text,
agent_id text,
run_id text,
session_id text,
candidate_type text not null,
candidate_json jsonb not null,
source_span_ids jsonb not null default '[]',
decision text not null,
reason_codes jsonb not null default '[]',
scores_json jsonb not null default '{}',
matched_existing_ids jsonb not null default '[]',
created_at timestamptz not null default now()
```

### 6.5 测试

单元测试：

- 寒暄、确认、空泛反馈被 reject。
- 明确“记住我喜欢 X”被 accept。
- assistant 建议不会被错误归因为 user preference。
- 同义重复进入 merge，不重复写 fact。
- “我不再用 Chroma，改用 Qdrant”产生 supersedes 候选或 relation。

验收：

- 人工标注 200 条 conversation turn，EncodingGate accept precision >= 0.85。
- reject 样本中真实应写入比例 <= 0.10。
- 所有 accept 候选 source_span_ids 非空。

## 7. Layer 3 Fact Ledger

目标：保存 Agent 可直接使用的长期 facts。做完这一层后，就是一个可用的 fact memory 系统。

### 7.1 Schema

表：`memory_facts`

```sql
fact_id uuid primary key,
workspace_id text,
user_id text,
agent_id text,
run_id text,
subject text,
predicate text,
object text,
text text not null,
category text not null,
polarity text default 'unknown',
confidence double precision not null,
salience double precision not null,
observed_at timestamptz,
valid_from timestamptz,
valid_to timestamptz,
source_span_ids jsonb not null,
linked_fact_ids jsonb default '[]',
embedding_dense vector,
search_tsv tsvector,
hash text,
metadata jsonb default '{}',
created_at timestamptz not null default now()
```

表：`fact_relations`

```sql
relation_id uuid primary key,
from_fact_id uuid not null,
to_fact_id uuid not null,
relation_type text not null,
source_span_ids jsonb not null default '[]',
confidence double precision not null,
created_at timestamptz not null default now()
```

### 7.2 API

```python
add_facts(accepted_candidates) -> list[Fact]
search_facts(query, scope, filters=None, limit=20) -> list[FactCandidate]
get_fact(fact_id) -> Fact
get_fact_relations(fact_id) -> list[FactRelation]
```

### 7.3 规则

- facts ADD-only。
- UPDATE 不改旧 fact 文本，只写 `supersedes` 或设置 `valid_to`。
- relation 必须有双方 fact 和 source span。
- `assistant_statement`、`agent_action` 不能进入用户 profile/current preference。

### 7.4 测试

- 新偏好写入 fact。
- 偏好变化写入新 fact + supersedes relation。
- 旧 fact 仍可 history 查到。
- 当前检索默认降低 superseded fact，历史查询仍能召回。
- fact search 支持 dense 和 BM25。

验收：

- fact extraction precision >= 0.80。
- source attribution coverage = 1.0。
- duplicate fact rate <= 0.10。

## 8. Layer 4 Temporal/Event Graph

目标：补 temporal reasoning、event ordering、multi-session reasoning。做完这一层后，可以独立回答“何时发生、哪个先发生、后来怎么变化”。

### 8.1 Schema

表：`events`

```sql
event_id uuid primary key,
workspace_id text,
user_id text,
agent_id text,
run_id text,
session_id text,
event_type text not null,
participants jsonb default '[]',
description text not null,
time_start timestamptz,
time_end timestamptz,
time_granularity text,
time_source text,
source_span_ids jsonb not null default '[]',
fact_ids jsonb default '[]',
confidence double precision not null,
created_at timestamptz not null default now()
```

表：`event_edges`

```sql
edge_id uuid primary key,
from_event_id uuid not null,
to_event_id uuid not null,
edge_type text not null,
source_span_ids jsonb not null default '[]',
confidence double precision not null,
created_at timestamptz not null default now()
```

### 8.2 Temporal Normalizer

规则优先：

- `today/yesterday/tomorrow` 以 session_time 为锚。
- `last week/next month/this Friday` 转为 interval。
- ISO date 和显式月份直接解析。
- 无法解析时输出 `time_source=unknown`，不能用当前系统时间替代。

### 8.3 API

```python
search_events(query, scope, filters=None, limit=20) -> list[EventCandidate]
timeline(entity, scope, start=None, end=None) -> list[Event]
compare_events(event_a, event_b) -> TemporalRelation
```

### 8.4 测试

- 相对时间根据 session_time 解析。
- 同一 session 内事件默认有顺序。
- 明确 before/after 语句写 event_edges。
- state_change 事件关联对应 facts。
- event ordering 查询保留多个时间点 source spans。

验收：

- 时间解析准确率 >= 0.85。
- 明确顺序关系 extraction precision >= 0.80。
- event_ordering 小测试集 answer accuracy 比 L1-only 提升 >= 15 points。

## 9. Layer 5 Views/Profile Layer

目标：低延迟提供当前状态和长期画像。做完这一层后，产品可以直接读取当前偏好、长期指令、项目状态、用户风格。

### 9.1 Current Views

表：`current_views`

```sql
view_id uuid primary key,
workspace_id text,
user_id text,
agent_id text,
view_type text not null,
subject text not null,
text text not null,
state_json jsonb not null default '{}',
source_fact_ids jsonb default '[]',
source_event_ids jsonb default '[]',
source_span_ids jsonb default '[]',
confidence double precision not null,
updated_at timestamptz not null default now(),
expires_at timestamptz
```

生成逻辑：

- 从 active facts + fact_relations 折叠。
- 被 superseded 的旧 fact 不进入 current view。
- unresolved contradiction 降低 confidence，并保留 conflicts。
- view 可后台重算，关键 preference/instruction 可同步更新。

### 9.2 EntityProfile / PersonalityView

表：`entity_profiles`

```sql
profile_id uuid primary key,
workspace_id text,
user_id text,
agent_id text,
run_id text,
entity_id text not null,
entity_type text not null,
profile_type text not null,
text text not null,
state_json jsonb not null default '{}',
source_fact_ids jsonb default '[]',
source_event_ids jsonb default '[]',
source_span_ids jsonb default '[]',
confidence double precision not null,
support_count int not null default 1,
last_observed_at timestamptz,
updated_at timestamptz not null default now(),
expires_at timestamptz,
embedding_dense vector,
search_tsv tsvector
```

生成规则：

- 单次事实不升格 profile。
- 至少满足 `support_count >= 2` 或用户显式要求长期记住。
- profile 是 retrieval prior，不是最终事实裁判。
- 当用户新偏好与 profile 冲突，CurrentView 和最新 source span 优先。

### 9.3 API

```python
get_current_views(scope, view_type=None) -> list[CurrentView]
refresh_current_views(scope, affected_fact_ids=None) -> list[CurrentView]
get_entity_profile(entity_id, scope, profile_type=None) -> list[EntityProfile]
refresh_entity_profiles(scope, affected_entity_ids=None) -> list[EntityProfile]
```

### 9.4 测试

- 偏好变化后 current view 指向最新偏好。
- unresolved contradiction 降低 view confidence。
- profile 需要多次支持才生成。
- profile 不覆盖明确最新偏好。
- source_fact_ids/source_span_ids 可回溯。

验收：

- current preference 查询 p95 < 100ms，不含 LLM。
- current view source coverage = 1.0。
- profile 生成 precision >= 0.80。

## 10. Layer 6 Retrieval Pack

目标：完整 query-time memory。做完这一层后，系统可以作为产品 answer_context 和 benchmark retrieval 使用。

### 10.1 Query Planner

输入 query，输出：

```json
{
  "query_type": "temporal_lookup",
  "entities": ["Atlas", "Qdrant"],
  "time_constraints": [{"type": "relative", "text": "last week"}],
  "speaker_focus": "user",
  "needs_current_state": false,
  "needs_source_evidence": true,
  "must_include_sources": ["raw_evidence", "events"]
}
```

冷启动 query_type 用规则 + 小模型/LLM：

- 含“现在/当前/以后按哪个” -> preference/instruction/current state。
- 含“什么时候/先后/之前/之后/last/first” -> temporal/event。
- 含“有没有说过/具体是什么” -> factual_exact。
- 含“不知道/没有提到时”或 benchmark abstention 类 -> abstention。

### 10.2 Candidate Generators

每一路 generator 独立返回候选：

```python
class Candidate:
    id: str
    type: Literal["span", "fact", "event", "view", "profile"]
    text: str
    source: str
    scores: dict
    source_span_ids: list[str]
    metadata: dict
```

Generators：

- L0 raw dense。
- L0 raw BM25。
- L1 fact dense。
- L1 fact BM25。
- L2 graph expansion。
- L3 current view lookup。
- L3 entity/profile lookup。
- exact/entity/time filters。

### 10.3 Raw Evidence Quota

实现：

```python
def enforce_raw_evidence_quota(query_plan, candidates, scope) -> QuotaResult:
    required = quota_for(query_plan.query_type)
    eligible_spans = filter_raw_spans(candidates, query_plan)
    if len(eligible_spans) < required.min_count:
        backfill = search_spans(..., limit=required.min_count - len(eligible_spans))
    return selected_spans, coverage
```

必须写入 trace：

- quota required。
- quota selected。
- backfill queries。
- coverage_insufficient。

### 10.4 Scoring / RRF / MMR

流程：

```text
1. hard filter by scope/speaker/time/security
2. RRF merge candidate lists
3. compute feature vector
4. cold-start weighted score or utility_score
5. enforce raw evidence quota
6. MMR diversify
7. optional cross-encoder rerank
8. build evidence pack
```

### 10.5 Evidence Pack Builder

规则：

- 默认 6k-8k tokens。
- source_spans 放原文摘要，不直接塞全量长文。
- temporal/event ordering 按时间排序。
- contradiction 必须同时包含冲突双方。
- abstention quota 不足时 `answer_policy=abstain_if_not_supported`。
- profile 只作为 prior，不能单独支撑事实答案。

### 10.6 测试

单元测试：

- 每种 query_type 生成合理 plan。
- 每路 generator 可单独调用。
- raw quota 在 temporal/contradiction/abstention 生效。
- MMR 不全选同一 source span。
- evidence pack token budget 不超。

集成测试：

- 构造 5-session 用户偏好变化，查询当前偏好返回最新 view + 原文。
- 构造冲突事实，查询冲突时返回双方 source。
- 构造 event ordering，返回按时间排序 evidence。
- 构造 unknown query，证据不足时 abstain。

验收：

- answer_context p95 在 Fast 模式 < 800ms，不含外部 LLM。
- Balanced 模式 p95 < 3s，不含 answer LLM。
- raw evidence quota hit rate >= 0.95。
- debug_trace coverage = 1.0。

## 11. Layer 7 Retrieval Utility Scorer

目标：逐步替代手写权重。第一版只 shadow，不影响线上排序。

### 11.1 Feature Vector

候选特征：

```text
rrf_score
semantic_score
bm25_score
entity_overlap
temporal_fit
graph_proximity
view_or_profile_prior
source_quality
source_recency
speaker_match
query_type_match
candidate_type
source_span_count
confidence
salience
relation_support
contradiction_coverage
quota_selected
```

### 11.2 Label Sources

表：`retrieval_utility_examples`

```sql
example_id uuid primary key,
query_id text,
query_text text not null,
query_type text,
candidate_id text not null,
candidate_type text not null,
features_json jsonb not null,
label text not null,
label_source text not null,
answer_correct boolean,
created_at timestamptz not null default now()
```

弱标签：

- gold source 或 gold answer 直接支持候选 -> useful。
- scope 内但实体/时间/speaker 全不匹配 -> not_useful。
- 被 evidence pack 选中且 answer 正确 -> useful 弱正例。
- 被高分选中但 answer 错误，且 judge reason 指向证据错误 -> not_useful。
- contradiction 只覆盖一方 -> unknown 或低权负例。

### 11.3 训练

第一版：

- Logistic Regression，可解释、便于看特征。
- 指标：AUC、NDCG@10、MRR、Recall@quota。
- 按 query_type 分桶报告。

第二版：

- LightGBM / XGBoost。
- 加 calibration，输出可解释 confidence。

上线：

1. 只记录，不排序。
2. shadow 排序，与手写排序比较。
3. dev/benchmark query replay 胜出后，在 Balanced 模式启用。
4. Fast 模式保留手写权重 fallback。

### 11.4 测试

- 特征生成稳定，不因缺字段报错。
- 模型文件版本可回滚。
- shadow 结果写入 trace。
- 同一输入同一模型版本排序确定。

验收：

- NDCG@10 比手写权重提升 >= 5%。
- 不降低 raw evidence quota hit rate。
- per-category 无明显回退，尤其 temporal/contradiction/abstention。

## 12. Layer 8 Benchmark/Product Integration

目标：同时验证 benchmark 和真实产品工作流。

### 12.1 Benchmark Adapter

实现：

```python
class BenchmarkAdapter:
    def ingest_dataset(self, dataset_path, split, scope) -> IngestReport: ...
    def build_queries(self, dataset_path, split) -> list[BenchmarkQuery]: ...
    def answer_query(self, query) -> BenchmarkAnswer: ...
    def score(self, answers) -> BenchmarkReport: ...
```

BEAM adapter 必须记录：

- split：100k/500k/1m/10m。
- query_type 映射。
- evidence_pack。
- answer model。
- judge model。
- tokens/query。
- retrieval latency。
- per-category score。

### 12.2 Product Replay

真实 query replay 集至少包含：

- 当前偏好。
- 历史事实。
- agent/tool 行为。
- 时间和顺序。
- 矛盾和更新。
- 不应回答的问题。
- 长期风格/画像。

### 12.3 报告

每次评测输出：

```text
overall score
per-category score
latency p50/p95
tokens/query
LLM calls/query
evidence recall
raw evidence quota hit rate
encoding accept/reject precision
utility scorer NDCG/MRR
debug trace coverage
failure samples
```

## 13. 开发里程碑

### Milestone 1：Evidence Store 可用

范围：

- Scope。
- evidence_spans。
- add/get/search spans。
- dense + BM25。
- audit trace。

验收：

- 能独立做原文检索。
- 专名/日期/数字 BM25 recall@20 达标。
- 所有 span 可按 scope 隔离。

### Milestone 2：Extraction + EncodingGate 可用

范围：

- Fact/Event/Relation candidate schema。
- FactExtractor v0。
- TemporalNormalizer v0。
- EncodingGate v0。
- encoding_decisions。

验收：

- 明确事实被 accept。
- 寒暄/重复/错误 speaker attribution 被 reject。
- 不确定候选进入 quarantine。

### Milestone 3：Fact Ledger 可用

范围：

- memory_facts。
- fact_relations。
- ADD-only facts。
- fact search。

验收：

- 能作为 Mem0-style fact memory 单独使用。
- supersedes/contradicts relation 可查询。
- facts 100% 回链 source_span_ids。

### Milestone 4：Temporal/Event Graph 可用

范围：

- events。
- event_edges。
- timeline API。
- event ordering query。

验收：

- 相对时间锚定正确。
- before/after 查询可独立回答。
- temporal 小集明显优于 L1-only。

### Milestone 5：Views/Profile Layer 可用

范围：

- current_views。
- entity_profiles。
- refresh views/profiles。

验收：

- current preference p95 < 100ms。
- profile 不由单次事实误生成。
- 新偏好覆盖旧 profile prior。

### Milestone 6：Retrieval Pack 可用

范围：

- QueryPlanner。
- multi-source generators。
- raw evidence quota。
- RRF/MMR。
- EvidencePack。

验收：

- answer_context 可直接放入 LLM prompt。
- temporal/contradiction/abstention raw quota 生效。
- debug_trace 解释每条证据来源。

### Milestone 7：Utility Scorer Shadow

范围：

- feature extraction。
- retrieval_utility_examples。
- weak labels。
- logistic regression。
- shadow ranking。

验收：

- NDCG/MRR 报告生成。
- shadow 不影响线上结果。
- 有明确是否替换手写权重的结论。

### Milestone 8：Benchmark Hardening

范围：

- BEAM adapter。
- per-category reports。
- ablation。
- failure analysis。

验收：

- 完成 L0、L1、L0+L1+Gate、L0+L1+L2、Full、Full+Utility ablation。
- 所有 answer 有 evidence_pack trace。
- 输出可复跑报告。

## 14. 测试矩阵

| 层 | 单测 | 集成测试 | 指标 |
|---|---|---|---|
| L0 Foundation | scope、fake model、audit | add/search 隔离 | scope leak = 0 |
| L1 Evidence | chunk、hash、BM25、dense | 原文检索 | recall@20 |
| L2 Gate | salience、novelty、speaker | extract -> gate | accept precision |
| L3 Facts | ADD-only、relations | preference update | source coverage |
| L4 Events | time parse、edges | timeline/order | temporal accuracy |
| L5 Views/Profile | refresh、conflict | current state/profile | p95、precision |
| L6 Retrieval | planner、quota、MMR | answer_context | quota hit、latency |
| L7 Utility | features、model version | shadow ranking | NDCG/MRR |
| L8 Benchmark | adapter | BEAM replay | per-category score |

## 15. 第一批人工测试样例

### Preference Update

输入：

```text
2026-06-01 User: For Atlas, I prefer Chroma because it is simple.
2026-06-08 User: We switched Atlas to Qdrant. Remember that Qdrant is now preferred.
```

期望：

- 两条 facts 均保留。
- Qdrant fact supersedes Chroma fact。
- current_view 指向 Qdrant。
- 查询“现在 Atlas 用什么”返回 Qdrant + 两条 source。
- 查询“之前是不是用过 Chroma”仍返回 Chroma source。

### Speaker Attribution

输入：

```text
Assistant: You may want to use PostgreSQL.
User: Good idea, but don't remember that as my preference yet.
```

期望：

- 不生成 user preference PostgreSQL。
- 可生成 assistant_statement。
- EncodingGate 对 user preference candidate reject，reason 包含 `speaker_attribution`。

### Temporal Ordering

输入：

```text
2026-06-03 User: I tested BM25 yesterday.
2026-06-05 User: After the BM25 test, I added dense retrieval.
```

期望：

- BM25 test time_start = 2026-06-02。
- dense retrieval event after BM25 test。
- event_ordering 查询返回两个 source spans，按时间排序。

### Contradiction

输入：

```text
User: I only want short answers.
User: For architecture reviews, give me detailed tradeoffs.
```

期望：

- 不是简单覆盖。
- relation 可为 linked_to 或 scoped exception。
- current_view 表达“默认短答；架构 review 需要详细 tradeoffs”。

### Abstention

输入：

```text
User: Remember that my database is PostgreSQL.
Query: What is my Kubernetes cluster name?
```

期望：

- evidence quota 不足。
- answer_policy = abstain_if_not_supported。
- 不用 PostgreSQL 证据硬猜 cluster name。

### EntityProfile

输入：

```text
多次对话中用户都要求：结论先行、少废话、但技术方案要可执行。
```

期望：

- 多次支持后生成 communication_style profile。
- 单次要求只进入 fact/current instruction，不直接升格 profile。
- profile 在回答风格类 query 中作为 prior。

## 16. 工程任务拆分

第一批 issue：

1. 创建 migrations：evidence_spans、audit_events。
2. 实现 ScopeGuard 和 Repository 基类。
3. 实现 EvidenceStore add/get/search。
4. 接 embedding fake + real adapter。
5. 实现 FactExtractor candidate schema 和 fake extractor tests。
6. 实现 EncodingGate v0 规则。
7. 创建 encoding_decisions 表和 trace 输出。
8. 实现 memory_facts / fact_relations repository。
9. 实现 fact dense/BM25 search。
10. 实现 TemporalNormalizer v0。
11. 实现 events/event_edges repository。
12. 实现 current_views refresh。
13. 实现 entity_profiles refresh。
14. 实现 QueryPlanner v0。
15. 实现 raw evidence quota。
16. 实现 RRF/MMR。
17. 实现 EvidencePack builder。
18. 实现 retrieval_utility_examples 采集。
19. 实现 BEAM adapter skeleton。
20. 实现 ablation report。

每个 issue 必须包含：

- 目标行为。
- 输入/输出 schema。
- migration 或接口变更。
- 单元测试。
- 集成测试或 fixture。
- debug trace 字段。

## 17. 默认参数

冷启动参数：

```text
chunk_size_tokens = 1000
chunk_overlap_tokens = 150
fact_accept_confidence = 0.70
fact_quarantine_confidence = 0.45
salience_threshold = 0.35
novelty_threshold = 0.25
duplicate_similarity_threshold = 0.92
relation_accept_confidence = 0.75
retrieval_top_k_per_source = 30
rrf_k = 60
mmr_lambda = 0.72
answer_context_budget_tokens = 8000
fast_mode_rerank = false
balanced_mode_rerank_top_n = 50
balanced_mode_output_n = 12
```

这些参数必须集中在 config，不允许散落在业务代码里。所有 benchmark report 要记录参数快照。

## 18. 不能做的事

- 不能没有 source_span_ids 就写 fact/event/view/profile。
- 不能把 assistant/system 内容默认当成用户事实。
- 不能用当前系统时间解析历史对话中的相对时间。
- 不能让 CurrentView 覆盖历史事实。
- 不能让 EntityProfile 覆盖明确的新偏好。
- 不能让 utility scorer 绕过 raw evidence quota。
- 不能只报 overall benchmark，不报 per-category 和失败样例。

## 19. 完成定义

MVP 完成标准：

- Layer 1-6 全部可用。
- Layer 7 shadow 可运行，但不要求默认上线。
- BEAM small/dev 可以完整 ingest、retrieve、answer、score。
- 所有答案可追踪到 evidence_pack。
- 四项新增能力均有可观测指标：
  - EncodingGate：accept/reject precision。
  - Retrieval Utility Scorer：NDCG/MRR shadow report。
  - Raw Evidence Quota：quota hit rate 和 coverage_insufficient。
  - EntityProfile/PersonalityView：profile precision 和 source coverage。

