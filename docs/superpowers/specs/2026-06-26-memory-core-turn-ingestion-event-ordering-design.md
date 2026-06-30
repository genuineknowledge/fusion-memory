# Fusion Memory Core Turn Ingestion 与 Event Ordering 收敛设计

- 日期: 2026-06-26
- 状态: Ready for review
- 范围: `memory/` core

## 1. 背景

当前 Fusion Memory 更偏向显式 `add()` 驱动：调用方主动挑选一段内容写入 memory。这个模型适合 tool 显式写入，但不适合 Dolphin 这类已经维护 session history、或通过外部 watcher 同步已落盘 history 的 agent。

当前问题有两类：

1. 写入侧没有“按 turn 一次 flush、内部保留原始 message 顺序”的 core 抽象。
2. 检索侧 `event_ordering` 与紧邻的 retrieval protection 已经堆出多轮 `preserve -> filter -> rescue -> filter`，行为不稳定，难以推理，也难以给上层适配器提供稳定契约。

本设计只处理这两件事的第一阶段收敛：

- 新增 core 级 turn ingestion 能力。
- 收敛 `event_ordering` 主路径及其紧邻的保护逻辑。

## 2. 目标

### 2.1 写入目标

- 支持调用方按“每轮一次 flush”或“saved-history batch 一次 flush”提交新增 message 列表。
- core 内部保留 `user / assistant / tool` 的原始顺序，不要求适配层先做归并或过滤。
- assistant / tool 的重要性降低、噪声过滤、状态变更识别都下沉到 memory core。
- 显式 `memory_add` 继续保留，不与 turn ingestion 互斥。

### 2.2 检索目标

- 默认允许跨 session 汇总到 `workspace_id + user_id + agent_id` 长期视图。
- 排序上始终 `当前 session` 优先，长期视图只作为补充召回源。
- 将 `event_ordering` 收敛为单一主路径，减少重复 rescue/filter 往返。

## 3. 非目标

- 这轮不重写整个 retrieval pipeline。
- 这轮不引入新的长期记忆产品形态或新的 UI。
- 这轮不取消显式 `add/search/answer_context` 接口。
- 这轮不做“方案 B”级别的 retrieval policy engine 重构；它只会被写入后续计划。

## 4. 核心决策

### 4.1 Turn ingestion 输入单位

采用调用方传入“本轮新增 message 列表”的方案。对 Dolphin watcher 降级适配来说，这个列表来自已落盘 history JSONL 的 batch，而不一定来自 Dolphin 内部强 turn boundary。

- 每个逻辑 batch 只 flush 一次；内部 hook 场景通常是一轮一个 batch，history watcher 场景则是一次 saved-history batch。
- 一次 flush 内仍拆成多个 raw spans。
- 顺序严格保留，至少包含：
  - `turn_index`
  - `message_index_in_turn`
  - `role`
  - `content`
  - `message_time`（若调用方提供）

调用方不需要先做 message 合并、importance 过滤或 assistant/tool 折叠。

如果调用方是文件 watcher，必须提供稳定 `turn_id`，例如：

```text
dolphin:<session-id>:lines:<start-line>-<end-line>:<batch-hash>
```

core 不依赖这个格式解析业务语义，但会保留它用于去重、排序、trace 和 event ordering。

### 4.2 Session 与长期视图

默认读取策略：

- `current session` 优先
- `workspace_id + user_id + agent_id` 长期视图补充

这不是完全统一混排。当前 session 的时间邻近性和局部对话连续性在排序中要保留更高权重。

### 4.3 Assistant / tool 降权

assistant 内容不再与 user 内容同权进入长期记忆。

基础规则：

- `user`：默认高保真保留
- `assistant`：默认降权
- `tool`：默认降权，仅在“外部状态发生变化”时提升

assistant 的高价值例外只保留两类：

1. 承诺 / 决定
2. 对外部状态的确认

tool result 的高价值例外：

- 只有 state-changing tool result 才可被提升为高价值候选
- 纯读取型、回显型、调试型输出默认低权重

降权同时发生在两层：

- 写入侧：importance hint / evidence weight
- 检索侧：candidate score / packing priority

### 4.4 错误 turn 的处理

如果一轮在 AI 或 tool 阶段中途失败，turn ingestion 仍然应写入当时已经产生的新增 messages。

也就是说：

- flush 触发点仍是“本轮结束”
- “结束”既包括正常完成，也包括以错误结束
- 不因为 assistant 缺失就丢掉本轮 user message

对只读取 history 文件的 Dolphin watcher，core 接受 `metadata["ended_with_error"] = "unknown"`。是否精确知道错误结束属于 adapter 能力，不属于 core 必填项。

## 5. Core 接口设计

新增一个 core 级入口，命名建议为 `ingest_turn(...)`。

建议签名：

```python
def ingest_turn(
    self,
    messages: list[dict[str, Any]],
    scope: Scope,
    *,
    turn_id: str | None = None,
    turn_index: int | None = None,
    session_time: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> AddResult:
    ...
```

语义：

- `messages` 是本轮新增 message 列表，不是完整 history。
- `messages` 也可以是外部 watcher 从已保存 history 中切出的 batch，但必须保持原始 message 顺序。
- `messages` 中至少支持 `role/content`，可选 `tool_name/tool_call_id/name/metadata`。
- `turn_id` 如果由调用方提供，core 必须原样用于 raw span `turn_id` 和 trace `turn_ingestion.turn_id`。
- core 负责把 message 拆成 raw spans，并写入统一的 ingestion pipeline。
- `AddResult` 继续沿用已有结构，必要时补充 `ingested_turn_span_ids` 一类字段。

## 6. 写入模型

### 6.1 Raw span 层

每条 message 转成一个 raw span，不在 adapter 层做合并。

span metadata 至少新增：

- `ingestion_kind = "turn"`
- `turn_id`
- `turn_index`
- `message_index_in_turn`
- `message_role`
- `message_kind`：
  - `user_message`
  - `assistant_message`
  - `tool_result`
- `importance_hint`
- `state_change_hint`（仅 tool）

### 6.2 Importance hint 生成

core 根据 role 和内容特征生成初始 hint：

- `user_message`: `high`
- `assistant_message`: `low`，若命中“承诺/决定”或“外部状态确认”提升为 `medium/high`
- `tool_result`: `low`，若判定为 state-changing result 提升为 `medium/high`

这一层只做轻量结构化标注，不做大规模启发式裁剪。

### 6.3 与显式 add 的关系

显式 `add()` 不移除。

差异：

- `add()`：调用方明确声明“这条值得长期记忆”
- `ingest_turn()`：调用方声明“这是本轮新增原始 history，请 core 自己判断价值”

二者在底层可复用同一写入管线，但必须保留不同的 `ingestion_kind`，以便后续检索和调试区分来源。

### 6.4 Watcher 幂等边界

`ingest_turn()` 的第一阶段不要求实现全局强幂等，因为显式 add 和已有 store 没有统一事务级幂等键。Dolphin history watcher 必须先用本地 checkpoint 避免重复提交。

core 需要做到：

- 保留调用方提供的稳定 `turn_id`。
- 将 `metadata["source"]`、`metadata["batch_hash"]`、`metadata["history_path"]`、`metadata["line_start"]`、`metadata["line_end"]` 写入 spans/trace。
- 检索和 event ordering 使用 `turn_id + message_index_in_turn` 作为顺序线索。

后续如果要加强幂等，可在 store 层增加 `(scope, ingestion_kind, turn_id, message_index_in_turn, content_hash)` 唯一约束或软去重逻辑；这不进入本阶段。

## 7. 检索排序模型

默认查询视图采用“双源但非同权”：

1. 当前 session candidates
2. 长期视图 candidates（`workspace_id + user_id + agent_id`）

排序要求：

- 先保证当前 session 中的高相关候选不被长期视图挤掉。
- 再让长期视图补齐“本 session 没有，但长期稳定存在”的偏好、事实、历史结论。
- assistant/tool 生成的候选在同等相关性下不应压过 user-origin 候选。

## 8. Event Ordering Phase 1 收敛

### 8.1 当前问题

`MemoryService.search()` 中 `event_ordering` 现在被多轮特殊逻辑穿插：

- preserve event ordering events
- preserve event ordering raw facets
- topic scope filter
- preservation runtime rescue
- post-preservation topic scope filter

这些逻辑和通用 rescue/filter 交错，导致：

- 同一类候选被多次拉回和剔除
- topic scope 与 timeline completeness 互相覆盖
- trace 很长，但不容易解释“为什么最后选了它”

### 8.2 Phase 1 目标路径

`event_ordering` 只保留一条主路径：

1. 常规 recall + score
2. event-ordering 专用 timeline / graph candidate selection
3. 一次 topic-scope gating
4. 一次 required preservation 回灌
5. 最终 pack

关键原则：

- 不再允许多轮“先 filter 掉、再救回来、再 filter 掉”
- 只有 required preservation 能在 topic-scope 之后做有限回灌
- 回灌理由必须结构化记录到 trace

### 8.3 这轮具体收敛范围

这轮只收：

- `_preserve_event_ordering_events`
- `_preserve_event_ordering_raw_facets`
- `_apply_topic_scope_filter`
- `_apply_event_ordering_post_preservation_topic_scope_filter`
- 它们在 `search()` 主流程中的交错顺序

目标不是完全删除所有 helper，而是把它们折叠成“单次 event-ordering selection + 单次 post-selection guard”。

## 9. Trace 与可调试性

新增或强化以下 trace 字段：

- `turn_ingestion`
  - turn_id
  - turn_index
  - raw_message_count
  - role_breakdown
  - promoted_assistant_count
  - promoted_tool_count
- `event_ordering_selection`
  - graph_candidates
  - timeline_representatives
  - topic_scope_dropped
  - preservation_restored

要求：

- 每个被恢复的 candidate 都要有单一明确 reason code
- 不再接受“因为经历了三轮 helper，最后留下来了但说不清哪一步决定”的状态

## 10. 对 Dolphin 适配层的契约

core 对 Dolphin-compatible adapter 的输入契约非常薄：

- adapter 提供新增 messages，可以来自 Dolphin 内部 turn hook，也可以来自外部 history watcher 的 saved-history batch
- 不要求 adapter 过滤 assistant/tool 噪声
- 不要求 adapter 预先拆 spans
- adapter 只需要保证 messages 顺序、scope 信息、以及 watcher 场景下的稳定 `turn_id`

在当前 Dolphin watcher-only 方案中，Dolphin `src` 不做任何 memory 改动。memory 侧 watcher 只做：

- history JSONL 读取
- saved-history batch 分组
- checkpoint 去重
- scope 传递

memory core 才负责：

- role-aware ingestion
- importance weighting
- long-term persistence selection

## 11. 后续计划（方案 B）

这轮完成后，把更激进的方案 B 明确放入下一阶段：

- 将 `event_ordering / topic scope / preservation` 从当前 `search()` 的过程式分支中拆出
- 形成更稳定的 stage 化 retrieval policy engine
- 让不同 query type 的特殊逻辑不再靠多轮 rescue/filter 穿插表达

这不在当前实现范围内，但必须在后续 plan 中列为 phase 2。
