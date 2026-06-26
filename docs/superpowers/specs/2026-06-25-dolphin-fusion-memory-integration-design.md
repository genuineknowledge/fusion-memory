# Dolphin-Agent × Fusion Memory 适配层设计

- 日期: 2026-06-25
- 更新: 2026-06-26
- 状态: Ready for review
- 适配层范围: `Dolphin-Agent` + `memory/integrations/dolphin-fusion-memory/`

## 1. 背景

Fusion Memory 的长期目标不再只是“模型显式调用 `memory_add` 时才写入”。在 Dolphin 场景下，session history 本身是完备的，因此除了保留显式记忆工具，还需要在 turn 结束后，从 Dolphin 当前 session 的新增 history 自动做一次持久化。

这个自动持久化不应该把噪声过滤、assistant 降权、tool 结果判断放到 Dolphin 侧。Dolphin 只负责把“本轮新增 messages”可靠交给 memory core；真正的角色权重、筛选和长期记忆策略由 memory core 统一负责。

## 2. 目标

- 保留显式 `memory_add / memory_search / memory_answer_context` 工具，让模型仍可主动读写记忆。
- 在 Dolphin session 内增加 turn 结束后的自动持久化。
- 自动持久化的数据源是当前 session history 的“本轮新增 message 列表”。
- Dolphin 不在适配层过滤 assistant/tool 噪声。
- 检索默认允许跨 session 长期视图，但由 memory core 保证 `current session` 优先。

## 3. 非目标

- Dolphin 侧不实现记忆价值判断。
- Dolphin 侧不做 spans 预拆分。
- Dolphin 侧不重写 Fusion Memory 的 retrieval policy。
- 这轮不把 Dolphin 改成“只有自动持久化，没有显式 add tool”。

## 4. 适配结构

适配层由两部分组成：

1. **显式工具层**
   - 继续通过 workspace tools 暴露 `memory_add / memory_search / memory_answer_context`
   - 面向模型显式调用

2. **自动持久化层**
   - 挂在 Dolphin session turn 生命周期上
   - 每轮结束时抽取本轮新增 messages
   - 调 memory core 的 turn ingestion 能力

显式工具和自动持久化并存，不互斥。

## 5. Dolphin 侧职责边界

Dolphin 只负责三件事：

1. 维护完整 history
2. 判断一轮何时结束
3. 计算本轮 history delta，并把 delta 原样交给 memory core

Dolphin 不负责：

- 判断 assistant 内容是否值得长期记忆
- 判断 tool 输出是否只是噪声
- 合并 message 为 summary span
- 决定跨 session 检索权重

## 6. Turn 持久化行为

### 6.1 Flush 粒度

采用“每轮只 flush 一次，但内部保留多 message 顺序”的方案。

也就是说：

- 不是每来一条 message 就立即打到 memory
- 也不是先把一整轮拼成一个大文本窗口
- 而是在 turn 结束后，一次提交一个 message list

### 6.2 提交内容

提交给 memory core 的是“本轮新增 message 列表”，不是完整 history。

列表中至少保留：

- `role`
- `content`
- 在本轮中的顺序
- session / workspace / user / agent 标识

如果存在 tool calls / tool results，可附带：

- tool name
- tool_call_id
- tool metadata

### 6.3 Turn 结束定义

以下两类情况都要触发 flush：

1. 正常完成的一轮  
   assistant 最终回复或 tool 链闭合后结束

2. 异常结束的一轮  
   AI 请求失败、tool 阶段失败、或中途终止

异常结束时也必须把当时已经产生的新增 messages 持久化，至少不能丢掉 user message。

## 7. Dolphin Hook 点

建议在 `SessionAgent.run()` 一轮执行过程中增加一个轻量的 turn delta 记录点。

高层流程：

1. 记录本轮开始前的 `history_len_before`
2. 正常执行本轮
3. 在 turn 正常结束或异常结束时，取 `history[history_len_before:]`
4. 将该 delta 作为一次 turn ingestion 提交给 memory core

这样有几个好处：

- 不依赖 channel 重放完整 history
- 不需要 Dolphin 额外拼装 summary
- 自动持久化严格跟随 session 实际落地的 history

## 8. 与显式 memory_add 的关系

显式 `memory_add` 继续保留，并仍然显示在 tool 列表中。

两者分工：

- `memory_add`：模型明确判断“这条值得长期记忆”
- auto-persist：系统保证“本轮原始历史不会漏”

这意味着：

- 没有显式 `memory_add`，依然会有自动持久化
- 有显式 `memory_add` 时，不等于关闭自动持久化
- 去重、降权、长期合并在 memory core 处理

## 9. Scope 与读取策略

Dolphin 侧在写入和读取时都需要提供统一 scope：

- `workspace_id`
- `user_id`
- `agent_id`
- `session_id`

读取默认策略由 memory core 实现为：

- `current session` 优先
- `workspace_id + user_id + agent_id` 长期视图补充

Dolphin 适配层只传 scope，不自己做跨 session 混排。

## 10. 显式工具层保留项

workspace tools 仍然保留原三件套：

- `memory_add`
- `memory_search`
- `memory_answer_context`

它们的作用不变：

- `memory_add`：主动持久化稳定事实/偏好/决定
- `memory_search`：低层原始证据检索
- `memory_answer_context`：高层问答上下文检索

自动持久化不会替代检索工具，也不会替代显式 add。

## 11. 错误处理

自动持久化必须是“非阻断”的：

- memory core 不可用时，不影响 Dolphin 当前 turn 对用户返回结果
- 失败要记日志 / trace，但不能让 session turn 失败
- Dolphin 对外错误文案保持统一，不把 memory 内部细节泄露给终端用户

## 12. 验收标准

1. Dolphin session 在不依赖模型显式调用 `memory_add` 的情况下，也能在每轮结束后自动持久化本轮新增 history。
2. 显式 `memory_add` 仍然存在于工具列表中，且行为不变。
3. 自动持久化输入给 memory core 的单位是“本轮新增 messages”，而不是 Dolphin 预聚合后的大文本块。
4. Dolphin 侧不实现 assistant/tool 噪声过滤。
5. memory 不可用时，Dolphin 正常完成 turn，只记录降级日志/trace。

## 13. 与 core spec 的关系

本文件只定义 Dolphin 适配层边界。

真正决定以下策略的是 memory core spec：

- assistant / tool 降权
- 自动持久化后哪些内容进入长期记忆
- `event_ordering` 的主路径
- `current session` 与长期视图的排序关系

对应 core 设计见：

- `memory/docs/superpowers/specs/2026-06-26-memory-core-turn-ingestion-event-ordering-design.md`
