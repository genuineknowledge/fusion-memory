# Fusion Memory Core Turn Ingestion 与 Event Ordering Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fusion Memory core 增加按 turn ingestion 的写入入口，并把 `event_ordering` 主路径收敛成单次 selection、单次 topic-scope gating、单次 required-preservation restore。

**Architecture:** 在 `MemoryService` 上新增 `ingest_turn()`，把调用方提交的“本轮新增 messages”或 watcher 的 saved-history batch 转成现有 `normalize_input()` 可消费的 message list，并补齐 `turn_id / turn_index / message_index_in_turn / message_role / importance_hint / state_change_hint` 等 metadata。随后复用既有 `add()` 写入管线，并在写入 trace 上追加 `turn_ingestion` 结构化调试块。检索侧只收 `event_ordering` 与紧邻保护逻辑：把现有多轮 preserve/filter/rescue 交错压成一条主路径，并把最终覆盖信息统一写入 `coverage["event_ordering_selection"]`。

**Tech Stack:** Python 3.12+, `unittest`, `pytest`, `http.server`, Fusion Memory `MemoryService`, SQLite/Postgres store trace APIs.

## Global Constraints

- 支持调用方按“每轮一次 flush”或“saved-history batch 一次 flush”提交新增 message 列表。
- core 内部保留 `user / assistant / tool` 的原始顺序，不要求适配层先做归并或过滤。
- core 必须原样保留调用方提供的稳定 `turn_id`，供 watcher 去重、trace 和 event ordering 使用。
- core 接受 `metadata["ended_with_error"] = "unknown"`，因为 history watcher 无法可靠知道 Dolphin 内部错误结束状态。
- assistant / tool 的重要性降低、噪声过滤、状态变更识别都下沉到 memory core。
- 显式 `memory_add` 继续保留，不与 turn ingestion 互斥。
- 默认允许跨 session 汇总到 `workspace_id + user_id + agent_id` 长期视图。
- 排序上始终 `当前 session` 优先，长期视图只作为补充召回源。
- 这轮不重写整个 retrieval pipeline。
- 这轮不取消显式 `add/search/answer_context` 接口。
- 这轮不做“方案 B”级别的 retrieval policy engine 重构；它只会被写入后续计划。

## File Map

- Modify: `/public/home/wwb/memory/fusion_memory/api/service.py`
  - 新增 `ingest_turn()` 入口。
  - 负责 turn message 到 `add()` 输入格式的转换。
  - 负责 role-aware importance/state-change hints。
  - 负责 `turn_ingestion` trace 写回。
  - 负责 `event_ordering` 主路径收敛和 coverage 输出。
- Modify: `/public/home/wwb/memory/fusion_memory/server.py`
  - 暴露 `POST /ingest-turn`。
- Modify: `/public/home/wwb/memory/tests/test_fusion_memory.py`
  - 覆盖 turn ingestion 顺序、异常 turn 保留、role-aware metadata、trace 字段。
- Modify: `/public/home/wwb/memory/tests/test_server.py`
  - 覆盖 `/ingest-turn` HTTP round-trip。
- Modify: `/public/home/wwb/memory/tests/test_event_ordering_graph.py`
  - 覆盖 event ordering selection coverage 与单次 topic-scope pass。
- Modify: `/public/home/wwb/memory/tests/test_retrieval_preservation.py`
  - 覆盖 required-preservation restore 的结构化 coverage。
- Modify: `/public/home/wwb/memory/tests/test_retrieval_pipeline.py`
  - 覆盖 search/answer_context trace 中新的 `event_ordering_selection` 可见性。

---

### Task 1: Add the `MemoryService.ingest_turn()` entry point

**Files:**
- Modify: `/public/home/wwb/memory/fusion_memory/api/service.py`
- Test: `/public/home/wwb/memory/tests/test_fusion_memory.py`

**Interfaces:**
- Produces:
  - `MemoryService.ingest_turn(messages: list[dict[str, Any]], scope: Scope, *, turn_id: str | None = None, turn_index: int | None = None, session_time: datetime | None = None, metadata: dict[str, Any] | None = None) -> AddResult`
  - Raw span metadata preserving adapter-provided `source`, `batch_hash`, `history_path`, `line_start`, and `line_end` when present.
- Consumes:
  - existing `MemoryService.add(input: Any, scope: Scope, session_time: datetime | None = None, metadata: dict[str, Any] | None = None) -> AddResult`
  - `normalize_input()` contract: list/dict entries use `role`, `content`, `turn_id`, `timestamp`, `span_type`, `metadata`

- [ ] **Step 1: Write the failing tests**

`/public/home/wwb/memory/tests/test_fusion_memory.py`
```python
class FusionMemoryTests(unittest.TestCase):
    def test_ingest_turn_persists_raw_messages_in_order(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws", user_id="u", agent_id="a", session_id="s")

        result = memory.ingest_turn(
            [
                {"role": "user", "content": "book the train to Hangzhou"},
                {"role": "assistant", "content": "I will compare the morning options."},
                {"role": "tool", "content": "draft booking created", "name": "rail_search"},
            ],
            scope,
            turn_id="turn-1",
            turn_index=1,
            session_time=ts("2026-06-26T09:00:00+00:00"),
        )

        spans = [
            span
            for span in memory.store.list_spans(scope, include_session=True)
            if span.turn_id == "turn-1" and span.span_type in {"turn", "tool_result"}
        ]

        self.assertTrue(result.span_ids)
        self.assertEqual([span.speaker for span in spans], ["user", "assistant", "tool"])
        self.assertEqual([span.metadata["message_index_in_turn"] for span in spans], [0, 1, 2])
        self.assertEqual([span.content for span in spans], [
            "book the train to Hangzhou",
            "I will compare the morning options.",
            "draft booking created",
        ])

    def test_ingest_turn_keeps_user_message_for_error_turn(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws", user_id="u", agent_id="a", session_id="s")

        memory.ingest_turn(
            [{"role": "user", "content": "delete the stale branch"}],
            scope,
            turn_id="turn-error",
            turn_index=8,
            metadata={"ended_with_error": True},
            session_time=ts("2026-06-26T09:05:00+00:00"),
        )

        spans = [
            span
            for span in memory.store.list_spans(scope, include_session=True)
            if span.turn_id == "turn-error" and span.span_type == "turn"
        ]

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].speaker, "user")
        self.assertEqual(spans[0].content, "delete the stale branch")
        self.assertTrue(spans[0].metadata["ended_with_error"])

    def test_ingest_turn_preserves_watcher_turn_id_and_source_metadata(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws", user_id="u", agent_id="a", session_id="session-1")

        memory.ingest_turn(
            [{"role": "user", "content": "remember blue"}],
            scope,
            turn_id="dolphin:session-1:lines:1-1:abcd1234",
            metadata={
                "source": "dolphin-history-watcher",
                "history_path": "/workspace/histories/session-1.jsonl",
                "line_start": 1,
                "line_end": 1,
                "batch_hash": "abcd1234",
                "ended_with_error": "unknown",
            },
            session_time=ts("2026-06-26T09:06:00+00:00"),
        )

        spans = [
            span
            for span in memory.store.list_spans(scope, include_session=True)
            if span.turn_id == "dolphin:session-1:lines:1-1:abcd1234"
        ]

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].metadata["source"], "dolphin-history-watcher")
        self.assertEqual(spans[0].metadata["batch_hash"], "abcd1234")
        self.assertEqual(spans[0].metadata["ended_with_error"], "unknown")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest tests/test_fusion_memory.py -k "ingest_turn" -v
```

Expected: FAIL with `AttributeError: 'MemoryService' object has no attribute 'ingest_turn'`.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/memory/fusion_memory/api/service.py`
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
    scope.validate_for_add()
    session_time = session_time or datetime.now(timezone.utc)
    resolved_turn_id = turn_id or f"turn_{turn_index or 0}"
    base_metadata = dict(metadata or {})
    payload_messages: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        role = str(message.get("role") or "user")
        payload_messages.append(
            {
                "role": role,
                "content": str(message.get("content") or ""),
                "turn_id": resolved_turn_id,
                "timestamp": message.get("timestamp") or session_time.isoformat(),
                "span_type": "tool_result" if role == "tool" else "turn",
                "metadata": {
                    **base_metadata,
                    **dict(message.get("metadata") or {}),
                    "ingestion_kind": "turn",
                    "turn_index": turn_index,
                    "message_index_in_turn": index,
                    "message_role": role,
                    "tool_name": message.get("name"),
                    "tool_call_id": message.get("tool_call_id"),
                },
            }
        )

    return self.add({"messages": payload_messages}, scope, session_time=session_time)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest tests/test_fusion_memory.py -k "ingest_turn" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/memory
git add fusion_memory/api/service.py tests/test_fusion_memory.py
git commit -m "feat: add turn ingestion service entry point"
```

### Task 2: Add role-aware turn metadata and trace persistence

**Files:**
- Modify: `/public/home/wwb/memory/fusion_memory/api/service.py`
- Test: `/public/home/wwb/memory/tests/test_fusion_memory.py`

**Interfaces:**
- Produces:
  - span metadata keys:
    - `message_kind`
    - `importance_hint`
    - `state_change_hint`
  - trace block:
    - `turn_ingestion.raw_message_count`
    - `turn_ingestion.role_breakdown`
    - `turn_ingestion.promoted_assistant_count`
    - `turn_ingestion.promoted_tool_count`
- Consumes:
  - `self.store.get_trace(trace_id, scope, include_session=True) -> dict[str, Any] | None`
  - `self.store.save_trace(trace_id, trace, scope) -> None | bool`

- [ ] **Step 1: Write the failing tests**

`/public/home/wwb/memory/tests/test_fusion_memory.py`
```python
class FusionMemoryTests(unittest.TestCase):
    def test_ingest_turn_marks_role_aware_importance(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws", user_id="u", agent_id="a", session_id="s")

        memory.ingest_turn(
            [
                {"role": "user", "content": "remember that I prefer aisle seats"},
                {"role": "assistant", "content": "Confirmed. I will remember your aisle-seat preference."},
                {
                    "role": "tool",
                    "content": "booking preference updated in vendor system",
                    "name": "booking_api",
                    "metadata": {"state_changed": True},
                },
            ],
            scope,
            turn_id="turn-2",
            turn_index=2,
            session_time=ts("2026-06-26T10:00:00+00:00"),
        )

        spans = [
            span
            for span in memory.store.list_spans(scope, include_session=True)
            if span.turn_id == "turn-2" and span.span_type in {"turn", "tool_result"}
        ]

        self.assertEqual(spans[0].metadata["message_kind"], "user_message")
        self.assertEqual(spans[0].metadata["importance_hint"], "high")
        self.assertEqual(spans[1].metadata["message_kind"], "assistant_message")
        self.assertIn(spans[1].metadata["importance_hint"], {"medium", "high"})
        self.assertEqual(spans[2].metadata["message_kind"], "tool_result")
        self.assertTrue(spans[2].metadata["state_change_hint"])

    def test_ingest_turn_trace_records_role_breakdown(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="ws", user_id="u", agent_id="a", session_id="s")

        result = memory.ingest_turn(
            [{"role": "user", "content": "hello"}],
            scope,
            turn_id="turn-3",
            turn_index=3,
            session_time=ts("2026-06-26T10:05:00+00:00"),
        )

        trace = memory.store.get_trace(result.trace_id, scope, include_session=True)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["turn_ingestion"]["raw_message_count"], 1)
        self.assertEqual(trace["turn_ingestion"]["role_breakdown"]["user"], 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest tests/test_fusion_memory.py -k "role_aware_importance or role_breakdown" -v
```

Expected: FAIL because `importance_hint`, `state_change_hint`, and `turn_ingestion` trace fields do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/memory/fusion_memory/api/service.py`
```python
def _turn_message_annotations(self, message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role") or "user")
    metadata = dict(message.get("metadata") or {})
    content = str(message.get("content") or "")
    lowered = content.lower()

    if role == "user":
        return {"message_kind": "user_message", "importance_hint": "high", "state_change_hint": False}
    if role == "tool":
        state_changed = bool(metadata.get("state_changed"))
        return {
            "message_kind": "tool_result",
            "importance_hint": "medium" if state_changed else "low",
            "state_change_hint": state_changed,
        }

    promoted = any(token in lowered for token in ("confirmed", "decided", "i will remember"))
    return {
        "message_kind": "assistant_message",
        "importance_hint": "medium" if promoted else "low",
        "state_change_hint": False,
    }
```

`/public/home/wwb/memory/fusion_memory/api/service.py`
```python
role_breakdown: dict[str, int] = {}
promoted_assistant_count = 0
promoted_tool_count = 0

for index, message in enumerate(messages):
    annotations = self._turn_message_annotations(message)
    role = str(message.get("role") or "user")
    role_breakdown[role] = role_breakdown.get(role, 0) + 1
    if role == "assistant" and annotations["importance_hint"] in {"medium", "high"}:
        promoted_assistant_count += 1
    if role == "tool" and annotations["state_change_hint"]:
        promoted_tool_count += 1
    payload_messages.append(
        {
            "role": role,
            "content": str(message.get("content") or ""),
            "turn_id": resolved_turn_id,
            "timestamp": message.get("timestamp") or session_time.isoformat(),
            "span_type": "tool_result" if role == "tool" else "turn",
            "metadata": {
                **base_metadata,
                **dict(message.get("metadata") or {}),
                "ingestion_kind": "turn",
                "turn_index": turn_index,
                "message_index_in_turn": index,
                "message_role": role,
                "tool_name": message.get("name"),
                "tool_call_id": message.get("tool_call_id"),
                **annotations,
            },
        }
    )
```

`/public/home/wwb/memory/fusion_memory/api/service.py`
```python
result = self.add({"messages": payload_messages}, scope, session_time=session_time)
trace = self.store.get_trace(result.trace_id, scope, include_session=True) or {}
trace["turn_ingestion"] = {
    "turn_id": resolved_turn_id,
    "turn_index": turn_index,
    "raw_message_count": len(messages),
    "role_breakdown": role_breakdown,
    "promoted_assistant_count": promoted_assistant_count,
    "promoted_tool_count": promoted_tool_count,
}
self.store.save_trace(result.trace_id, trace, scope)
return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest tests/test_fusion_memory.py -k "role_aware_importance or role_breakdown" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/memory
git add fusion_memory/api/service.py tests/test_fusion_memory.py
git commit -m "feat: add role-aware turn ingestion metadata"
```

### Task 3: Expose `POST /ingest-turn` through the HTTP server

**Files:**
- Modify: `/public/home/wwb/memory/fusion_memory/server.py`
- Test: `/public/home/wwb/memory/tests/test_server.py`

**Interfaces:**
- Produces:
  - `POST /ingest-turn`
  - request JSON fields:
    - `messages: list[dict[str, Any]]`
    - `scope: dict[str, Any]`
    - optional `turn_id`
    - optional `turn_index`
    - optional `session_time`
    - optional `metadata`
- Consumes:
  - `MemoryService.ingest_turn(...) -> AddResult`

- [ ] **Step 1: Write the failing test**

`/public/home/wwb/memory/tests/test_server.py`
```python
class ServerTests(unittest.TestCase):
    def test_ingest_turn_endpoint_roundtrip(self) -> None:
        ready = threading.Event()
        holder: dict[str, object] = {}

        def run_server() -> None:
            service = MemoryService()
            server = serve(service, host="127.0.0.1", port=0)
            holder["service"] = service
            holder["server"] = server
            ready.set()
            try:
                server.serve_forever()
            finally:
                server.server_close()
                service.close()

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=5))
        server = holder["server"]
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            body = _post_or_get(
                f"{base_url}/ingest-turn",
                {
                    "messages": [{"role": "user", "content": "plan a trip to Suzhou"}],
                    "scope": {"workspace_id": "ws", "user_id": "u", "agent_id": "a", "session_id": "s"},
                    "turn_id": "turn-http",
                    "turn_index": 4,
                },
            )
            self.assertTrue(body["span_ids"])
        finally:
            server.shutdown()
            thread.join(timeout=2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest tests/test_server.py -k "ingest_turn_endpoint_roundtrip" -v
```

Expected: FAIL with `404` or `request_failed`.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/memory/fusion_memory/server.py`
```python
elif path == "/ingest-turn":
    result = state.service.ingest_turn(
        payload.get("messages") or [],
        _scope(payload),
        turn_id=payload.get("turn_id"),
        turn_index=payload.get("turn_index"),
        session_time=_optional_datetime(payload.get("session_time")),
        metadata=payload.get("metadata"),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest tests/test_server.py -k "ingest_turn_endpoint_roundtrip" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/memory
git add fusion_memory/server.py tests/test_server.py
git commit -m "feat: expose turn ingestion over HTTP"
```

### Task 4: Collapse `event_ordering` to one selection path plus one restore pass

**Files:**
- Modify: `/public/home/wwb/memory/fusion_memory/api/service.py`
- Modify: `/public/home/wwb/memory/tests/test_event_ordering_graph.py`
- Modify: `/public/home/wwb/memory/tests/test_retrieval_preservation.py`
- Modify: `/public/home/wwb/memory/tests/test_retrieval_pipeline.py`

**Interfaces:**
- Produces:
  - `coverage["event_ordering_selection"]`
    - `graph_candidates`
    - `timeline_representatives`
    - `topic_scope_dropped`
    - `preservation_restored`
    - `topic_scope_filter_passes`
- Consumes:
  - existing event-ordering candidate helpers
  - existing runtime preservation helpers

- [ ] **Step 1: Write the failing tests**

`/public/home/wwb/memory/tests/test_event_ordering_graph.py`
```python
class EventOrderingGraphTests(unittest.TestCase):
    def test_event_ordering_pack_reports_single_topic_scope_pass(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="w-order", user_id="u", agent_id="a", session_id="s")
        memory.add("First I booked the outbound flight to Shanghai.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Then I reserved the hotel near the Bund.", scope, ts("2026-06-02T10:00:00+00:00"))
        memory.add("After that I submitted the visa application.", scope, ts("2026-06-03T10:00:00+00:00"))

        pack = memory.answer_context(
            "按时间顺序总结我的航班、酒店和签证事项。",
            scope,
            budget={"limit": 6, "mode": "benchmark"},
        )

        selection = pack.coverage["event_ordering_selection"]
        self.assertEqual(selection["topic_scope_filter_passes"], 1)
        self.assertTrue(selection["graph_candidates"])
        self.assertIn("timeline_representatives", selection)
```

`/public/home/wwb/memory/tests/test_retrieval_preservation.py`
```python
class RetrievalRegressionFixtureTests(unittest.TestCase):
    def test_event_ordering_restore_reports_structured_reason_codes(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="preserve-order", user_id="u", agent_id="a", session_id="s")
        memory.add("First I drafted the itinerary.", scope, datetime(2026, 6, 1, tzinfo=timezone.utc))
        memory.add("Then I changed the hotel to one near the station.", scope, datetime(2026, 6, 2, tzinfo=timezone.utc))
        memory.add("Finally I moved the departure to Friday night.", scope, datetime(2026, 6, 3, tzinfo=timezone.utc))

        result = memory.search(
            "按时间顺序列出我改动过的行程。",
            scope,
            {"mode": "benchmark", "limit": 6},
        )

        restored = result.coverage["event_ordering_selection"]["preservation_restored"]
        self.assertTrue(all("reason" in item for item in restored))
```

`/public/home/wwb/memory/tests/test_retrieval_pipeline.py`
```python
class RetrievalPipelineTests(unittest.TestCase):
    def test_event_ordering_selection_is_exposed_in_search_coverage(self) -> None:
        memory = MemoryService()
        scope = Scope(workspace_id="pipeline-order", user_id="u", agent_id="a", session_id="s")
        memory.add("First I prepared the migration checklist.", scope, ts("2026-06-01T10:00:00+00:00"))
        memory.add("Then I applied the schema changes.", scope, ts("2026-06-02T10:00:00+00:00"))

        result = memory.search(
            "List the migration work in chronological order.",
            scope,
            {"mode": "benchmark", "limit": 6},
        )

        self.assertIn("event_ordering_selection", result.coverage)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest \
  tests/test_event_ordering_graph.py \
  tests/test_retrieval_preservation.py \
  tests/test_retrieval_pipeline.py \
  -k "event_ordering_selection or topic_scope_passes or structured_reason_codes" -v
```

Expected: FAIL because the new `event_ordering_selection` coverage block does not exist yet, or because topic-scope passes are still reported through scattered legacy fields.

- [ ] **Step 3: Write the minimal implementation**

`/public/home/wwb/memory/fusion_memory/api/service.py`
```python
if plan.query_type == "event_ordering":
    selected = self._select_event_ordering_candidates(query, plan, scored_again, limit)
    selected, topic_scope_dropped = self._apply_event_ordering_topic_scope(
        query,
        plan,
        scored_again,
        selected,
        limit,
    )
    selected, restored = self._restore_required_event_ordering_candidates(
        scored_again,
        selected,
        limit,
    )
    coverage["event_ordering_selection"] = {
        "graph_candidates": [candidate.id for candidate in selected],
        "timeline_representatives": [
            candidate.id for candidate in selected if candidate.source.startswith("event_ordering_graph")
        ],
        "topic_scope_dropped": topic_scope_dropped,
        "preservation_restored": restored,
        "topic_scope_filter_passes": 1,
    }
```

`/public/home/wwb/memory/fusion_memory/api/service.py`
```python
def _restore_required_event_ordering_candidates(
    self,
    scored_candidates: list[Candidate],
    selected: list[Candidate],
    limit: int,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    preserved, dropped = preserve_required_candidates(scored_candidates, selected, limit=limit)
    restored = [
        {
            "candidate_id": candidate.id,
            "reason": ",".join(candidate.metadata.get("must_preserve_reason", [])),
        }
        for candidate in preserved
        if candidate not in selected and candidate.metadata.get("must_preserve_reason")
    ]
    return preserved, restored + dropped
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cd /public/home/wwb/memory
python -m pytest \
  tests/test_event_ordering_graph.py \
  tests/test_retrieval_preservation.py \
  tests/test_retrieval_pipeline.py \
  -k "event_ordering_selection or topic_scope_passes or structured_reason_codes" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /public/home/wwb/memory
git add \
  fusion_memory/api/service.py \
  tests/test_event_ordering_graph.py \
  tests/test_retrieval_preservation.py \
  tests/test_retrieval_pipeline.py
git commit -m "refactor: collapse event ordering phase 1 path"
```
