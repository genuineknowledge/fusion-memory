# MemPalace 架构复盘

本文以 GitHub `MemPalace/mempalace` 默认分支 `develop` 的 commit `939a076baf0b349e1f5b3a7e27ad1d545364f18b` 为准，按本机展开目录 `/home/wwb/memory/mempalace` 中的源码复核。核心文件：

```text
mempalace/cli.py
mempalace/miner.py
mempalace/palace.py
mempalace/searcher.py
mempalace/knowledge_graph.py
mempalace/palace_graph.py
mempalace/hallways.py
mempalace/config.py
mempalace/corpus_origin.py
mempalace/entity_registry.py
```

核心结论：MemPalace 的主 truth 是原文 chunk，即 drawer；closet、hallway、tunnel、Palace Graph 都是围绕 drawer 派生出的导航或检索信号；Knowledge Graph 是独立的 temporal triple store，不是普通 mine 自动生成的事实库。

## 1. 总览

一次普通项目导入和检索可以拆成两条主线：

```text
init <project>
  -> Pass 0 corpus origin
  -> entity discovery + registry merge
  -> room detection + mempalace.yaml
  -> optional mine

mine <project>
  -> scan files
  -> detect room / hall
  -> extract date / chunk / line range
  -> write drawers
  -> build closet pointer lines
  -> compute hallways
  -> compute topic/entity tunnels

search
  -> drawer vector candidates
  -> closet source_file boost
  -> BM25 rerank or SQLite FTS/BM25 fallback
  -> return verbatim drawer text/context
```

KG 和 Palace Graph 是旁路能力：

```text
mempalace_kg_add / KnowledgeGraph.add_triple
  -> SQLite entities + triples
  -> kg_query / timeline

build_graph / list_tunnels / follow_tunnels
  -> read drawer metadata + tunnels.json
  -> return navigation graph, not truth facts
```

## 2. 核心对象

| 对象 | 存储 | 作用 | 生成方式 |
|---|---|---|---|
| Palace | ChromaDB 数据目录 | 一个本地记忆库 | 默认由 `MempalaceConfig` 从 `~/.mempalace/config.json` 或 CLI 参数解析 |
| Wing | drawer metadata | 项目/语料大域 | init/mine 参数、项目目录名或配置 normalize 后写入 |
| Room | drawer metadata | wing 内主题分区 | `mempalace.yaml` 定义；mine 时 `detect_room()` 路由 |
| Hall | drawer metadata | 内容类型分类 | `detect_hall()` 对内容前 3000 字符按 `hall_keywords` 计分 |
| Drawer | Chroma collection `mempalace_drawers` | 原文 truth chunk | `chunk_text()` 切分后 upsert，document 是原文 |
| Closet | Chroma collection `mempalace_closets` | 轻量 topic/entity 指针 | `build_closet_lines()` 从同 source_file 前 5000 字符生成 pointer line |
| Hallway | `~/.mempalace/hallways.json` | wing 内 entity 共现边 | 同 wing drawer metadata entities 两两计数，默认 `min_count=2` |
| Tunnel | `tunnels.json` | 跨 wing/room 导航边 | explicit 手工创建；topic/entity mine 后自动刷新 |
| Knowledge Graph | `knowledge_graph.sqlite3` | typed temporal facts | 显式 `add_triple()` 写入，不由普通 mine 自动抽取 |
| Palace Graph | 运行时聚合 | room/wing/hall/tunnel 导航图 | `build_graph()` 读 drawer metadata 和 tunnels |

## 3. 初始化流程

### 3.1 Pass 0: origin.json

`cli._run_pass_zero(project_dir, palace_dir, llm_provider)` 负责生成 `<palace_path>/.mempalace/origin.json`。它不是配置文件，也不参与检索排序，而是 onboarding 审计和 entity discovery 的上下文输入。

具体实现：

1. `_gather_origin_samples()` 从项目文本文件采样，跳过 `entities.json`、`mempalace.yaml` 等系统文件。
2. `detect_origin_heuristic(samples)` 永远运行，用本地正则/格式特征判断语料是否像 AI dialogue，并给出 `likely_ai_dialogue`、`confidence`、`evidence`。
3. 如果有 LLM provider，`detect_origin_llm(_trim_samples_for_llm(samples), llm_provider)` 只补充 `primary_platform`、`user_name`、`agent_persona_names` 和证据。代码明确不让弱 LLM 覆盖 heuristic 的 `likely_ai_dialogue/confidence`。
4. 写入 JSON：

```json
{
  "schema_version": 1,
  "detected_at": "2026-...Z",
  "result": {
    "likely_ai_dialogue": true,
    "confidence": 0.9,
    "primary_platform": "claude",
    "user_name": "...",
    "agent_persona_names": ["..."],
    "evidence": ["Tier-1 heuristic: ...", "Tier-2 LLM: ..."]
  }
}
```

写文件失败只打印 warning，`init` 不会因为 origin 写入失败而中断。后续 `discover_entities(..., corpus_origin=...)` 用它区分真实用户名、agent persona 和普通文本实体。

### 3.2 Entity discovery 和 registry

init 阶段的实体来源不是单一模型抽取，而是多源合并：manifest、git authors、Claude Code cwd/project 名、prose regex、可选 LLM refinement、origin persona reclassification。确认后写两处：

- 项目内 `<project>/entities.json`：当前项目确认出的实体。
- 全局 `~/.mempalace/known_entities.json`：跨项目 known registry。

普通 category 合并策略：对每个 category 的 list 做大小写不敏感去重，已有值保留，新值追加。`topics_by_wing` 是保留字段，不按普通 category 追加；它按 wing 维护 topic list。

### 3.3 topics 和 topics_by_wing

`topics` 是全局实体类别列表，表示系统知道的主题词集合；它不记录主题来自哪个项目。

`topics_by_wing` 是 `{wing: [topic...]}` 映射，供 topic tunnel 计算。mine 或 registry merge 时，该 wing 的 topic list 会被 `_set_wing_topics()` 清洗、去重后写回；这个映射回答的是“哪个 wing 拥有哪些 topic”。

区别：

| 字段 | 粒度 | 写入策略 | 用途 |
|---|---|---|---|
| `topics` | 全局 | 大小写不敏感 union | known entity 匹配、人工查看 |
| `topics_by_wing` | 每个 wing | 对应 wing 的 topic list 单独维护 | topic tunnel 计算 shared topic |

### 3.4 Room detection

`room_detector_local.detect_rooms_from_folders/files()` 在 init 期间给出 room 候选，最终写 `<project>/mempalace.yaml`。mine 时每个文件调用 `miner.detect_room(filepath, content, rooms, project_path)`：

1. 把相对路径分段，目录名与 room name/keyword 做 `_name_matches()`，目录命中优先。
2. 文件名 stem 与 room name 做匹配。
3. 对内容前 2000 字符按每个 room 的 keywords 和 room name 计数，最高且大于 0 的 room 胜出。
4. 都没命中则 `general`。

room 是主题/空间分区，例如 `api`、`diary`、`research`。

## 4. Mine 写入流程

普通 project mine 的主路径在 `miner.process_file()` 和外层 mine loop 中完成。

### 4.1 文件扫描和 stale check

mine 会扫描项目文本文件，按配置和 ignore 规则过滤。每个文件读取后计算 normalized content version 和 `source_mtime`，结合 collection 中已有 `source_file`、`source_mtime`、`normalize_version` 判断是否跳过。重新 mine 某文件时会先删除该 `source_file` 对应旧 drawers/closets，避免旧 chunk 或旧 closet pointer 残留。

### 4.2 chunk_text

`chunk_text(content, source_file, chunk_size, chunk_overlap, min_chunk_size)` 默认来自配置，代码默认值是 `chunk_size=800`、`chunk_overlap=100`、`min_chunk_size=50`。实现细节：

1. 参数防御：`chunk_size > 0`，`chunk_overlap >= 0`，且 `chunk_overlap < chunk_size`，否则直接 `ValueError`，避免死循环。
2. `content.strip()` 后从 `start` 开始取 `start + chunk_size`。
3. 如果不是最后一段，优先向前找 `\n\n` 段落边界；找不到再找单个 `\n`。只有边界在 chunk 后半段时才切过去，避免过小 chunk。
4. chunk 长度达到 `min_chunk_size` 才保留。
5. `line_start = content.count("\n", 0, start) + 1`，`line_end = content.count("\n", 0, end) + 1`，用于 closet locator。
6. 下一段从 `end - chunk_overlap` 继续。

### 4.3 drawer id 和 metadata

project/format mine 的 drawer id 形如：

```text
drawer_<wing>_<room>_<sha256(source_file + str(chunk_index))[:24]>
```

当前摘要版曾写成带 `|` 的 recipe，但源码中 `process_file()` 和 `add_drawer()` 使用的是 `source_file + str(chunk_index)`。metadata 由 `_build_drawer_metadata()` 生成，核心字段：

```text
wing
room
source_file          # 通常是完整路径
chunk_index
added_by
filed_at             # 写入时 datetime.now().isoformat()
normalize_version
source_mtime
line_start / line_end
content_date
hall
entities
```

`content_date` 的提取顺序是：

1. 文件名 stem 中 ISO 日期或完整自然语言日期。
2. YAML frontmatter 的 `date/created/published`。
3. 正文前约 10 行中的 ISO、slash date、自然语言日期。slash date 会根据是否出现首位数字大于 12 来判断 DD/MM。
4. 文件系统 mtime。
5. 没有则 None，调用方使用 `filed_at`。

### 4.4 hall 字段如何写入

`hall` 是每个 drawer metadata 上的内容类型字段。`miner.detect_hall(content)` 读取 `MempalaceConfig().hall_keywords`，只检查内容前 3000 字符：

```text
for hall, keywords in hall_keywords:
  score = count(keyword present in content_lower)
return score 最大的 hall；无任何命中 -> general
```

注意这里不是按出现次数累加，而是每个 keyword 如果出现在内容里就加 1。hall 表示类型，例如 technical、emotional、family 等；它和 hallway 不是一回事。

### 4.5 entity 字段如何抽取和写入

每个 drawer 的 `entities` metadata 由 `_extract_entities_for_metadata(content)` 写入，值是分号拼接字符串，例如：

```text
Alice;Claude Code;MemPalace;Qdrant
```

抽取策略是两路合并：

1. 读取 `~/.mempalace/known_entities.json` 中的 known names，对整个 content 做大小写不敏感边界匹配。命中就加入。
2. 对内容前 `_ENTITY_EXTRACT_WINDOW`，即前 5000 字符，运行 i18n proper-name regex 候选抽取。

候选过滤和裁剪：

- 跳过 `_ENTITY_STOPLIST` 中的句首/泛词。
- 跳过 COCA common content words，避免 `Code`、`Line`、`Note` 这类词污染实体。
- proper-name 候选频次必须 `>=2`，长度大于 2。
- 最终排序后最多 `_ENTITY_METADATA_LIMIT` 个，源码限制为 25 个。
- 返回前按 list 截断再 join，避免把实体名截断到一半。

这个字段是 hallway/entity tunnel 的基础，也是 metadata 解释字段；它不是 KG fact。

### 4.6 closet 如何生成和存储

drawer 写完后，mine 会为同一个 `source_file` 构造 closet lines：

```text
build_closet_lines(source_file, drawer_ids, full_file_content, wing, room, drawer_metas)
-> upsert_closet_lines(closets_col, closet_id_base, lines, metadata)
```

`build_closet_lines()` 的具体规则：

1. 只看 source content 前 5000 字符。
2. `drawer_ref = ",".join(drawer_ids[:3])`，每条 pointer 最多指向该文件前三个 drawer id。
3. entity 部分复用 proper-name 候选抽取，频次 `>=2`，停用词和 COCA 过滤，最多 5 个。
4. topic 部分来自三类信号：
   - action-verb phrase：正则匹配 `built|fixed|wrote|added|pushed|tested|created|decided|migrated|reviewed|deployed|configured|removed|updated` 后跟 3-40 字符。
   - Markdown H1-H3 标题：`^#{1,3}\s+(.{5,60})$`。
   - 15-150 字符的双引号 quote，最多 3 条。
5. topic 去重保持顺序，最多 12 条。
6. 如果没有任何 topic/quote，用 fallback：`<wing>/<room>/<file_stem>`。
7. 如果首个 drawer metadata 有 `content_date` 或 `filed_at`，且有 `line_start/line_end`，pointer 格式为：

```text
topic|entity1;entity2|YYYY-MM-DD:Lstart-Lend|->drawer_a,drawer_b,drawer_c
```

否则使用兼容旧格式：

```text
topic|entity1;entity2|->drawer_a,drawer_b,drawer_c
```

`upsert_closet_lines()` 按 1500 字符贪心打包：一条 line 永不拆开；超过限制就 flush 当前 closet，id 后缀为 `_01`、`_02`。重 mine 前会 `purge_file_closets(source_file)`，删除该 source_file 的旧 closet。

## 5. hall 和 hallway 的区别

| 名称 | 类型 | 存储位置 | 生成方式 | 表达含义 |
|---|---|---|---|---|
| `hall` | drawer metadata 字段 | Chroma drawer metadata | `detect_hall()` keyword scoring | 这个 drawer 的内容类型 |
| `hallway` | entity 共现记录 | `~/.mempalace/hallways.json` | `compute_hallways_for_wing()` 统计同 wing drawer 的 entity pair | 两个 entity 在同一 wing 多次共同出现 |

`hall` 是分类标签；`hallway` 是导航边。hallway 不是 KG fact，不表示“两个 entity 有某种确定关系”，只表示它们在同 wing 的 drawers 中共同出现达到阈值。

## 6. hallway 和 tunnels

### 6.1 Hallway

`compute_hallways_for_wing(wing, col, min_count=2)` 的算法：

1. `col.get(where={"wing": wing}, include=["metadatas"])` 读取该 wing drawer metadata。
2. 跳过 `is_sentinel` 和没有有效 `entities` 的 drawer。
3. 将 `entities` 分号字符串解析成实体列表；每个 drawer 内取所有无序实体对。
4. 对 `(entity_a, entity_b)` 计数，并记录它们共同出现过的 room。
5. count `>= min_count` 才 materialize hallway。
6. hallway id：

```text
hallway_<wing>_<a>_<b>_<sha256(wing + "::" + a + "::" + b)[:8]>
```

实体名先排序，所以 `(Alice, Bob)` 与 `(Bob, Alice)` 是同一个 hallway。

### 6.2 Topic Tunnel

`topic_tunnels_for_wing(wing, topics_by_wing, min_count)` 用 `topics_by_wing` 计算当前 wing 与其他 wing 的 shared topics。实现上它为每个 other wing 构造两 wing slice，调用 `compute_topic_tunnels()`，当 normalized topic 交集数量达到阈值时，为 shared topic 写 tunnel。

endpoint 使用 synthetic room：

```text
source: { wing: wing_a, room: "topic:<topic>" }
target: { wing: wing_b, room: "topic:<topic>" }
kind: "topic"
label: "shared topic: <topic>"
```

它不要求真实 room 存在，因为 `kind != explicit` 时 `create_tunnel()` 不做 room existence 检查。

### 6.3 Entity Tunnel

`entity_tunnels_for_wing(wing, hallways)` 基于 hallway records 生成跨 wing 连接：

1. 遍历所有 hallway，`entity_a` 和 `entity_b` 都算作该 hallway 所属 wing 中出现的 entity。
2. 构造 `entity -> {normalized_wing -> display_wing}`。
3. 如果同一 entity 出现在当前 wing 和其他 wing 的 hallway 集合中，就创建 tunnel。
4. endpoint 使用 synthetic room：`entity:<name>`，kind 为 `entity`。

Entity tunnel 连接的是“这个实体在多个 wing 的 hallway 结构中都是强信号”，不是 typed relationship。

### 6.4 Explicit Tunnel

显式 tunnel 由 MCP/CLI 调 `create_tunnel(source_wing, source_room, target_wing, target_room, label, kind="explicit")` 创建。`kind="explicit"` 时会通过 Chroma collection 检查 source/target room 是否存在 drawer；不存在直接 `ValueError`。topic/entity tunnel 跳过真实 room 检查，因为它们使用 synthetic room。

所有 tunnel 都是无向去重：`_canonical_tunnel_id()` 对两个 endpoint key 排序，再 hash。再次创建同 endpoints 会更新 label/drawer ids，同时保留 `created_at` 和 dynamics 字段。

## 7. Knowledge Graph

KG 在 `knowledge_graph.py` 中实现，是 SQLite temporal triple store。普通 `mempalace mine` 不会自动从 drawer 抽 triple；只有显式调用 MCP `mempalace_kg_add` 或 Python `KnowledgeGraph.add_triple()` 才写入。

### 7.1 存储 schema

SQLite 文件通常在：

```text
<palace_path>/knowledge_graph.sqlite3
```

MCP 如果没有指定 palace，可能使用：

```text
~/.mempalace/knowledge_graph.sqlite3
```

核心表：

```text
entities(id, name, type, properties)
triples(id, subject, predicate, object,
        valid_from, valid_to, confidence,
        source_closet, source_file, source_drawer_id, adapter_name)
```

entity id 规则：

```text
name.lower().replace(" ", "_").replace("'", "")
```

predicate 规则：

```text
predicate.lower().replace(" ", "_")
```

triple id：

```text
t_<subject_id>_<predicate>_<object_id>_<sha256(valid_from + datetime.now().isoformat())[:12]>
```

### 7.2 add_triple

`add_triple(subject, predicate, obj, valid_from=None, valid_to=None, confidence=1.0, source_...)`：

1. `sanitize_iso_temporal()` 清洗 `valid_from/valid_to`。
2. 如果 `valid_to < valid_from`，抛 `ValueError`，避免产生任何 query 都看不到的倒置区间。
3. 自动 `INSERT OR IGNORE` subject/object entities。
4. 查是否已有同 subject/predicate/object 且 `valid_to IS NULL` 的 current triple；有则直接返回旧 id，不重复写。
5. 否则插入 triples，保留 confidence 和 provenance。

### 7.3 invalidate 和 query

`invalidate(subject, predicate, obj, ended=None)` 不删除 triple，而是将当前 triple 的 `valid_to` 设置为 `ended` 或当天日期。

`query_entity(name, as_of=None, direction="outgoing")`：

- `direction=outgoing` 查 `triples.subject = entity_id`。
- `direction=incoming` 查 `triples.object = entity_id`。
- `direction=both` 两边都查。
- 如果传 `as_of`，追加 temporal 条件：

```sql
valid_from <= as_of
AND (valid_to IS NULL OR valid_to >= as_of)
```

返回 subject/predicate/object、valid interval、confidence、source_closet/current 等。KG 查询适合 typed temporal fact，例如“某关系在某天是否成立”“Alice 当前负责哪个项目”。没有 KG 结果不代表 drawer 原文里没有相关内容。

## 8. Palace Graph

Palace Graph 不是 KG。`palace_graph.build_graph(col, config)` 从 drawer metadata 聚合导航图：

1. 分批读取 collection metadatas。
2. 对每个 drawer 读取 `room`、`wing`、`hall`、`date`。
3. `room != general` 且 wing 存在时，聚合：

```text
nodes[room].wings.add(wing)
nodes[room].halls.add(hall)
nodes[room].dates.add(date)
nodes[room].count += 1
```

4. 对出现在多个 wing 的同名 room 生成 passive edge，表示“这些 wing 里都有这个 room”。
5. 另有 `list_tunnels()`、`follow_tunnels()` 读取 `tunnels.json`，把 explicit/topic/entity tunnel 作为导航边。

Palace Graph 的作用是告诉 agent “可以从哪个 wing/room 跳到哪个 wing/room”，不是回答事实。典型用法是先用 graph/tunnel 缩小探索范围，再在目标 wing/room 上跑 drawer search。

## 9. 检索流程

### 9.1 CLI search

`searcher.search(query, palace_path, wing=None, room=None, n_results=5)`：

1. 先检查 palace path 是否存在、是否有 `chroma.sqlite3`、collection 是否初始化。
2. `build_where_filter(wing, room)` 构造 Chroma metadata filter。
3. collection query：`query_texts=[query]`，`n_results=n_results`，include documents/metadatas/distances。
4. 将结果组装成 hits 后调用 `_hybrid_rank(hits, query)`，用本地 BM25 对候选重排。
5. 打印原文 drawer text。

### 9.2 MCP/programmatic search_memories

`searcher.search_memories()` 是 MCP 主路径，目标是返回 drawer 原文，而不是 closet 文本。

主流程：

```text
validate candidate_strategy
if vector_disabled -> _bm25_only_via_sqlite
open drawer collection
query drawers: n_results * 3
query closets: n_results * 2
closet best-per-source_file -> source boost
merge optional BM25-only union candidates
_hybrid_rank candidates
expand source_file context if needed
format public result
```

closet boost 具体规则：

- closet query 找到的是 closet documents，但不会直接作为最终结果返回。
- 对每个 closet result 读取 metadata `source_file`，取每个 source_file 的 best closet distance/rank。
- `CLOSET_RANK_BOOSTS = [0.40, 0.25, 0.15, 0.08, 0.04]`。
- 只有 closet distance `<= 1.5` 才加 boost。
- drawer candidate 如果同 `source_file` 有 closet 命中：

```text
effective_distance = max(distance - closet_boost, 0)
matched_via = drawer+closet
closet_preview = closet document 前 200 字符
```

BM25 rerank：候选集中计算 BM25 raw score，归一化后和 vector similarity 组合。文档中不要把 closet 说成 gate；它只是 ranking signal。

### 9.3 SQLite FTS/BM25 fallback

当 vector search 禁用或 Chroma/HNSW 不可用时，`_bm25_only_via_sqlite()` 直接打开 `<palace_path>/chroma.sqlite3`：

1. 读取 Chroma 的 `embedding_fulltext_search` FTS5 trigram index。
2. query token 长度必须 `>=3`；用 `OR` 拼成 FTS MATCH。
3. wing/room filter 通过 `embedding_metadata` EXISTS 子查询过滤。
4. 如果 query 没有可用 token，回退到最近 row window，然后本地 BM25 排序。
5. 返回 `matched_via=bm25_sqlite`、`fallback=bm25_only_via_sqlite`。

## 10. 什么时候用 KG 查询，什么时候用 PG 查询

| 问题类型 | 首选 | 原因 |
|---|---|---|
| “Alice 现在负责哪个项目？” | KG query | typed predicate + temporal current |
| “2025-01-01 时 Max 是否 child_of Alice？” | KG query with `as_of` | 需要 temporal validity |
| “哪些 wing 都提到 payment topic？” | Palace Graph / topic tunnels | 需要跨 wing 导航 |
| “从 project_a/api 能跳到哪里？” | `follow_tunnels()` | 需要 tunnel endpoints |
| “找原文证据” | drawer search | truth 是 drawer 原文 |
| “某实体相关资料在哪些项目出现？” | entity tunnel + drawer search | tunnel 定位范围，search 找证据 |

KG 查询返回结构化事实；PG/tunnel 查询返回导航关系；drawer search 返回原文证据。三者不要混用成同一个 truth 层。

## 11. Wake-up layers

`layers.py` 的 wake-up 是上下文组装，不等同于完整 `mempalace_search`：

| 层 | 来源 | 注意 |
|---|---|---|
| L0 Identity | `~/.mempalace/identity.txt` | 用户身份文本 |
| L1 Essential Story | drawer collection 中按 importance/weight/recency/top drawers 取候选 | 普通 mine 不保证写 importance/weight |
| L2 On-Demand | wing/room filter 下的局部 drawers | 用于局部上下文 |
| L3 Deep Search | 简化 drawer query | 不查 closet，不做 closet boost/BM25/FTS fallback |

## 12. 存储位置汇总

| 内容 | 路径/collection | 说明 |
|---|---|---|
| drawers | `<palace_path>/chroma.sqlite3` / `mempalace_drawers` | document 是原文 chunk，metadata 是检索/导航字段 |
| closets | 同一 Chroma / `mempalace_closets` | document 是 topic pointer lines |
| KG | `<palace_path>/knowledge_graph.sqlite3` 或 `~/.mempalace/knowledge_graph.sqlite3` | entities/triples temporal fact |
| hallways | `~/.mempalace/hallways.json` | wing 内 entity pair 共现 |
| tunnels | `MempalaceConfig.tunnel_file`，通常 `dirname(<palace_path>)/tunnels.json`，fallback `~/.mempalace/tunnels.json` | explicit/topic/entity tunnels |
| known registry | `~/.mempalace/known_entities.json` | people/projects/topics/topics_by_wing 等 |
| project manifest | `<project>/mempalace.yaml` | wing/rooms/keywords |
| project entities | `<project>/entities.json` | 当前项目确认实体 |
| origin | `<palace_path>/.mempalace/origin.json` | corpus origin detection audit |

## 13. 完整端到端例子

这个例子覆盖 init、drawer、closet、hall、entity、hallway、topic tunnel、entity tunnel、explicit tunnel、KG、PG 和 search。字段值里的 hash/id 是示意，真实值由源码生成。

### 13.1 输入项目

项目 `/tmp/atlas-notes`：

```text
/tmp/atlas-notes/
  mempalace.yaml
  entities.json
  research/2026-05-01-memory-kernel.md
  engineering/2026-05-02-qdrant-bm25.md
```

`mempalace.yaml`：

```yaml
wing: atlas_notes
rooms:
  - name: research
    keywords: [memory, agent, kernel]
  - name: engineering
    keywords: [qdrant, bm25, deployment]
```

`entities.json`：

```json
{
  "people": ["Alice", "Bob"],
  "projects": ["Atlas", "Memory Kernel"],
  "topics": ["hybrid search", "entity linking"]
}
```

`research/2026-05-01-memory-kernel.md`：

```markdown
---
date: 2026-05-01
---
# Memory Kernel design
Alice and Bob reviewed Memory Kernel with Alice twice.
They decided hybrid search should combine drawer truth and lightweight pointers.
"Keep the raw drawer as truth, never the summary."
```

### 13.2 init 后

`mempalace init /tmp/atlas-notes --yes`：

- 写 `/tmp/atlas-notes/mempalace.yaml` 和 `/tmp/atlas-notes/entities.json`，如果已有则复用/确认。
- merge 到 `~/.mempalace/known_entities.json`：

```json
{
  "people": ["Alice", "Bob"],
  "projects": ["Atlas", "Memory Kernel"],
  "topics": ["hybrid search", "entity linking"],
  "topics_by_wing": {
    "atlas_notes": ["hybrid search", "entity linking"]
  }
}
```

- 写 `<palace_path>/.mempalace/origin.json`，记录语料是否像 AI dialogue、平台/persona 证据。

### 13.3 mine research 文件

对 `research/2026-05-01-memory-kernel.md`：

- `detect_room()`：路径 segment `research` 命中 room `research`。
- `_extract_content_date()`：frontmatter `date` 得到 `2026-05-01`。
- `detect_hall()`：内容前 3000 字符按关键词命中，例如若 config 中 `technical` 包含 design/reviewed，则 hall=`technical`；否则 `general`。
- `_extract_entities_for_metadata()`：
  - known registry 命中 Alice、Bob、Memory Kernel。
  - proper-name 候选 Alice 出现两次，也加入。
  - 最终 metadata `entities="Alice;Bob;Memory Kernel"`。
- `chunk_text()`：短文只产生一个 chunk：

```text
id: drawer_atlas_notes_research_<hash24>
document: 原始 markdown chunk
metadata:
  wing: atlas_notes
  room: research
  source_file: /tmp/atlas-notes/research/2026-05-01-memory-kernel.md
  chunk_index: 0
  line_start: 1
  line_end: 7
  content_date: 2026-05-01
  hall: technical/general
  entities: Alice;Bob;Memory Kernel
```

- `build_closet_lines()` 生成：

```text
memory kernel design|Alice;Bob;Memory Kernel|2026-05-01:L1-L7|->drawer_atlas_notes_research_<hash24>
decided hybrid search should combine drawer truth|Alice;Bob;Memory Kernel|2026-05-01:L1-L7|->drawer_atlas_notes_research_<hash24>
"Keep the raw drawer as truth, never the summary."|Alice;Bob;Memory Kernel|2026-05-01:L1-L7|->drawer_atlas_notes_research_<hash24>
```

这些 lines 被 packed 到 closet：

```text
id: closet_atlas_notes_research_<source_hash>_01
metadata: {wing, room, source_file, ...}
document: 上面多行 pointer text
```

### 13.4 hallways 和 tunnels

假设 engineering 文件也包含 Alice、Bob、Qdrant，并且 Alice/Bob 在两个 drawer 中共同出现：

`compute_hallways_for_wing("atlas_notes")` 得到：

```json
{
  "id": "hallway_atlas_notes_Alice_Bob_<hash8>",
  "wing": "atlas_notes",
  "entity_a": "Alice",
  "entity_b": "Bob",
  "count": 2,
  "rooms": ["research", "engineering"]
}
```

如果另一个 wing `mem0_notes` 的 `topics_by_wing` 也有 `hybrid search`，mine 后 topic tunnel：

```json
{
  "kind": "topic",
  "source": {"wing": "atlas_notes", "room": "topic:hybrid search"},
  "target": {"wing": "mem0_notes", "room": "topic:hybrid search"},
  "label": "shared topic: hybrid search"
}
```

如果 `Alice` 也出现在 `mem0_notes` 的 hallway 中，entity tunnel：

```json
{
  "kind": "entity",
  "source": {"wing": "atlas_notes", "room": "entity:Alice"},
  "target": {"wing": "mem0_notes", "room": "entity:Alice"},
  "label": "shared entity: Alice"
}
```

显式 tunnel 示例：

```text
mempalace_create_tunnel(
  source_wing="atlas_notes", source_room="research",
  target_wing="mem0_notes", target_room="architecture",
  label="Compare drawer-truth design with fact-memory design"
)
```

因为 kind 是 explicit，源码会检查 `atlas_notes/research` 和 `mem0_notes/architecture` 是否有真实 drawer。

### 13.5 KG 写入和查询

普通 mine 不会自动写 KG。需要显式写：

```python
kg.add_triple(
    "Alice",
    "reviewed",
    "Memory Kernel",
    valid_from="2026-05-01",
    confidence=0.9,
    source_file="/tmp/atlas-notes/research/2026-05-01-memory-kernel.md",
    source_drawer_id="drawer_atlas_notes_research_<hash24>",
)
```

SQLite 中：

```text
entities: alice, memory_kernel
triples: t_alice_reviewed_memory_kernel_<hash12>
  subject=alice
  predicate=reviewed
  object=memory_kernel
  valid_from=2026-05-01
  valid_to=NULL
  confidence=0.9
```

查询：

```text
mempalace_kg_query(entity="Alice", direction="outgoing", as_of="2026-06-01")
```

返回 Alice 在 2026-06-01 仍 current 的 outgoing facts。若后来调用：

```python
kg.invalidate("Alice", "reviewed", "Memory Kernel", ended="2026-06-15")
```

则 `as_of="2026-06-01"` 仍能查到，`as_of="2026-07-01"` 不再查到 current fact，但 timeline 还能看到历史。

### 13.6 Search 和 PG 查询

查询 “raw drawer truth hybrid search”：

1. drawer vector 搜到 research drawer。
2. closet vector 搜到包含 `drawer truth` 的 pointer line，对同 source_file 加 boost。
3. BM25 在候选原文中发现 `drawer`、`truth`、`hybrid search` 词面强命中。
4. 返回 drawer 原文，不返回 closet line：

```json
{
  "text": "# Memory Kernel design\nAlice and Bob reviewed...",
  "wing": "atlas_notes",
  "room": "research",
  "source_file": "2026-05-01-memory-kernel.md",
  "matched_via": "drawer+closet",
  "closet_preview": "memory kernel design|Alice;Bob...",
  "bm25_score": 3.42
}
```

PG 查询 “从 atlas_notes/research 能去哪里”：

```text
follow_tunnels("atlas_notes", "research")
```

返回 explicit tunnel 到 `mem0_notes/architecture`，但不会返回事实答案。agent 应再对 `mem0_notes/architecture` 运行 drawer search 获取原文证据。

## 14. 限制和设计启发

- 普通 mine 不自动抽 KG triple；KG 质量依赖显式写入或 adapter。
- drawer 是 truth；closet、hallway、tunnel、Palace Graph 都是辅助索引或导航结构。
- hallway/entity tunnel 只表示 entity 共现强信号，不表示 typed relationship。
- topic/entity tunnel 使用 synthetic room，不能当真实 room。
- closet locator 是粗粒度 source_file/date/line 指针，不是精确 span 标注。
- KG query 适合结构化事实，drawer search 适合证据，PG/tunnel 适合探索路径。
