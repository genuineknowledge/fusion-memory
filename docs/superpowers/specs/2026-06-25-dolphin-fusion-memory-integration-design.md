# Dolphin-Agent x Fusion Memory History Watcher Design

- 日期: 2026-06-25
- 更新: 2026-06-27
- 状态: Ready for review
- 适配层范围: `memory/integrations/dolphin-fusion-memory/`
- Dolphin 约束: 不修改 `Dolphin-Agent/src`

## 1. 背景

Fusion Memory 的长期目标不再只是“模型显式调用 `memory_add` 时才写入”。在 Dolphin 场景下，Dolphin 已经维护 session history，并会把部分历史落盘到 `workspace/histories/{session_id}.jsonl`。

本轮适配追求“不动 Dolphin `src`”。因此自动持久化不再挂到 `SessionAgent.run()` 的 turn 生命周期里，而是由 memory 侧提供一个外部 watcher，读取 Dolphin 已落盘的 history JSONL，并把新增历史同步到 Fusion Memory。

这意味着自动持久化语义从“捕获完整 in-memory turn delta”降级为“同步已写入 history 文件的消息”。这个边界必须写清楚，不能在文档或实现里继续假设 Dolphin 内部 hook。

## 2. 目标

- 保留显式 `memory_add / memory_search / memory_answer_context` 工具，让模型仍可主动读写记忆。
- 不修改 Dolphin `src`、不新增 Dolphin 内部 memory client、不改 `SessionAgent.run()`。
- 通过 memory 侧 watcher 监听 Dolphin `workspace/histories/{session_id}.jsonl`。
- watcher 将已落盘的新增 messages 分批提交到 memory core 的 `/ingest-turn`。
- Dolphin 不在适配层过滤 assistant/tool 噪声；watcher 也只做文件同步、分组、去重和 scope 补齐。
- 检索默认允许跨 session 长期视图，但由 memory core 保证 `current session` 优先。

## 3. 非目标

- 不在 Dolphin 侧实现记忆价值判断。
- 不在 Dolphin 侧做 spans 预拆分。
- 不重写 Dolphin history 保存策略。
- 不保证捕获未落盘的 in-memory message。
- 不把 Dolphin 改成“只有自动持久化，没有显式 add tool”。
- 不在 watcher 中实现复杂长期记忆筛选；筛选仍属于 memory core。

## 4. 适配结构

适配层由两部分组成：

1. **显式工具层**
   - 继续通过 Dolphin workspace tools 暴露 `memory_add / memory_search / memory_answer_context`
   - 面向模型显式调用
   - 请求 Fusion Memory HTTP server

2. **History watcher 层**
   - 运行在 Dolphin session 进程外
   - 读取 `workspace/histories/{session_id}.jsonl`
   - 根据 checkpoint 识别新增 JSONL messages
   - 将新增 message batch 提交到 `/ingest-turn`

显式工具和 watcher 并存，不互斥。

## 5. Dolphin 侧职责边界

Dolphin 只作为现有能力提供方：

- 加载 workspace tools
- 维护 session 内存 history
- 在既有逻辑中把 history 保存为 JSONL
- 接收 `--session-id` 以形成稳定 history 文件路径

Dolphin 不新增以下能力：

- Fusion Memory HTTP client
- turn-end hook
- memory flush task
- watcher lifecycle
- memory 错误文案

## 6. Watcher 职责

watcher 由 memory integration 提供，负责：

- 定位 Dolphin history 文件
- 周期性读取 JSONL
- 容忍整文件覆盖保存
- 跳过 malformed / partial line
- 用 checkpoint 记录已提交进度
- 对新增 messages 做轻量 turn grouping
- 生成稳定 `turn_id`
- 补齐 scope 和 metadata
- 非阻断提交 `/ingest-turn`

watcher 不负责：

- 判断 message 是否值得长期记忆
- 过滤 assistant/tool 噪声
- 汇总一轮为 summary
- 修改 Dolphin history 文件
- 启停 Dolphin session 进程

## 7. History 文件语义

数据源是 Dolphin 当前已有的 JSONL：

```text
<dolphin-workspace>/histories/<session-id>.jsonl
```

每行是一个 OpenAI-style message dict。watcher 读取时以文件内容为准，而不是以 Dolphin 内存状态为准。

推荐启动 Dolphin 时显式指定 `--session-id`，并同时设置：

```bash
PSI_MEMORY_SESSION_ID=<session-id>
```

如果没有稳定 `session_id`，watcher 可以支持“watch histories 目录中最新 JSONL”的降级模式，但该模式只用于本地调试，不作为生产默认。

## 8. Turn 分组策略

由于 watcher 看不到 Dolphin 内部 turn boundary，只能从落盘 message 序列推断 batch。

默认分组规则：

- 一个 batch 从 `role=user` 开始。
- 后续 `assistant`、`tool`、`assistant tool_calls` 等消息归入同一 batch。
- 遇到下一条 `role=user` 时关闭上一 batch。
- 文件末尾如果最后一个 batch 还没有后续 user，也可以在 debounce 窗口后提交。

debounce 默认 `1.0s`，用于等待 Dolphin 的整文件覆盖保存稳定。

这不是强 turn 语义，只是 history 文件接口下的最佳努力分组。memory core 必须能接受 batch 不完美的情况。

## 9. Checkpoint 与幂等

watcher 必须维护本地 checkpoint，默认位置：

```text
<dolphin-workspace>/.fusion-memory/dolphin-history-watcher/<session-id>.json
```

checkpoint 至少包含：

- `history_path`
- `session_id`
- `line_count`
- `file_size`
- `file_mtime_ns`
- `last_message_hash`
- `submitted_batches`

watcher 每次读取后：

1. 校验已提交前缀是否仍然匹配。
2. 如果文件只是追加或覆盖为相同前缀，提交新增 lines。
3. 如果文件被截断或前缀不匹配，重新扫描并用 message hash 跳过已提交 batch。

提交 `/ingest-turn` 时必须带稳定 `turn_id`：

```text
dolphin:<session-id>:lines:<start-line>-<end-line>:<batch-hash>
```

metadata 至少包含：

- `source = "dolphin-history-watcher"`
- `history_path`
- `line_start`
- `line_end`
- `batch_hash`
- `ended_with_error = "unknown"`

即使 memory core 当前不是严格幂等，watcher 也要通过 checkpoint 和稳定 `turn_id` 将重复提交风险降到最低。

## 10. 与显式 memory_add 的关系

显式 `memory_add` 继续保留，并仍然显示在 tool 列表中。

两者分工：

- `memory_add`：模型明确判断“这条值得长期记忆”
- history watcher：系统尽量保证“已落盘的原始 history 不漏同步”

这意味着：

- 没有显式 `memory_add`，已落盘 history 仍会由 watcher 同步
- 有显式 `memory_add` 时，不关闭 watcher
- 重叠内容的去重、降权、长期合并在 memory core 处理

## 11. Scope 与读取策略

显式工具和 watcher 使用同一组 scope：

- `workspace_id`
- `user_id`
- `agent_id`
- `session_id`
- `app_id = "dolphin"`

默认读取策略由 memory core 实现为：

- `current session` 优先
- `workspace_id + user_id + agent_id` 长期视图补充

watcher 只传 scope，不自己做跨 session 混排。

## 12. 错误处理与降级

watcher 必须是非阻断的：

- Fusion Memory 不可用时，不影响 Dolphin 当前 session。
- 提交失败时保留 checkpoint 前的未确认 batch，下次重试。
- malformed JSONL line 记录日志并跳过本轮提交，等待文件稳定后重读。
- 对外文案统一为 Dolphin 侧安全的 memory 降级文案，不泄露内部栈。

因为不修改 Dolphin `src`，以下场景不保证捕获：

- AI 请求失败后 Dolphin 没有把 user message 写入 history 文件。
- tool 阶段中断且中间消息尚未落盘。
- Dolphin 进程崩溃前只存在于内存中的 history。
- `ended_with_error` 的精确信息。

## 13. 验收标准

1. Dolphin `src` 没有 memory 相关改动。
2. Dolphin workspace 仍暴露 `memory_add / memory_search / memory_answer_context`。
3. Fusion Memory server 默认端口为 `8700`。
4. watcher 能读取指定 `workspace/histories/{session_id}.jsonl` 并提交新增 messages 到 `/ingest-turn`。
5. watcher restart 后不会重复提交已经确认的 batch。
6. history 文件整文件覆盖保存时，watcher 仍能识别新增 messages。
7. Fusion Memory 不可用时，Dolphin session 不受影响，watcher 保留待重试状态。
8. 文档明确说明 watcher 是“已落盘 history 同步”，不是 Dolphin 内部 turn hook。

## 14. 与 core spec 的关系

本文件只定义 Dolphin history watcher 适配层边界。

真正决定以下策略的是 memory core spec：

- `/ingest-turn` 输入契约
- assistant / tool 降权
- 重复内容降权和长期合并
- `event_ordering` 的主路径
- `current session` 与长期视图的排序关系

对应 core 设计见：

- `memory/docs/superpowers/specs/2026-06-26-memory-core-turn-ingestion-event-ordering-design.md`
