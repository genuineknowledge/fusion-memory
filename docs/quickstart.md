# Fusion Memory 快速开始

这是面向新手的默认安装方式。

## 1. 安装

Linux / macOS:

```bash
git clone https://github.com/genuineknowledge/fusion-memory.git
cd fusion-memory
sh install.sh
```

Windows PowerShell:

```powershell
cd C:\path\to\memory
.\install.ps1
```

安装完成后会自动运行 `fusion-memory install-check`。安装脚本会把 Fusion Memory
安装成 `uv tool`，由 uv 管理独立 Python 3.12 runtime；不会使用 agent 自身的
Python 环境。脚本会通过 ModelScope 下载两个本地向量模型到 Fusion Memory home
的 `models/` 目录，再安装完整 Qwen 运行依赖，包括 Postgres adapter、本地 Qwen
adapter 以及 PyTorch/Transformers 相关依赖。

条件满足时会配置：

- 数据库：默认 SQLite，本地数据库文件来自初始化配置。
- Embedding：默认 Fusion Memory home-local `models/Qwen3-Embedding-0.6B`。
- Reranker：默认 Fusion Memory home-local `models/Qwen3-Reranker-0.6B`。
- Extractor/router：默认内置规则；高级用户可选 OpenAI-compatible API。
- Query router：默认关闭；需要复杂查询路由时再开启 API。
- Postgres/pgvector：可选生产配置，适合大量数据、多用户或需要 pgvector 索引的部署。

如果模型下载失败、文件仍是 Git LFS pointer、或 Qwen runtime 依赖不可用，安装检查会
返回 not ready，并给出失败步骤和日志路径；不会要求安装 Git LFS，也不会静默降级到
`local_test`。CPU-only 机器是支持路径，不能把“没有 CUDA/GPU”当成安装失败原因；
当模型文件齐全，但 Python/Qwen ML runtime 不可导入，或 Qwen 本地 smoke test 在当前运行时无法加载/运行两个本地 Qwen 模型时，安装会 fallback 到 `compromised` 本地模式：
SQLite + 内置轻量 embedding/reranker 可以继续试用，但当前 memory 功能是
compromised 的。安装完成后需要提供 API key 才能接入更完整的模型能力；推荐阿里云
DashScope，设置：

```bash
export DASHSCOPE_API_KEY=<your-api-key>
```

API key 不会写入配置文件。向导只保存环境变量名，例如
`FUSION_MEMORY_MODEL_API_KEY`。启动服务前把真实 key 放到环境变量里即可。

如需进入手动向导：

```bash
FUSION_MEMORY_USE_WIZARD=1 sh install.sh
```

无人值守安装使用默认检测流程：

```bash
FUSION_MEMORY_SKIP_WIZARD=1 sh install.sh
```

### Recommended first run

Run:

```bash
fusion-memory init --json
fusion-memory start --json
fusion-memory doctor --json
export PSI_MEMORY_BASE_URL=http://127.0.0.1:8700
```

If port `8700` is already in use, `fusion-memory start --json` tries the next available local port and returns the actual `url`; set `PSI_MEMORY_BASE_URL` to that returned URL before starting the agent workspace.

The default first run uses SQLite plus local Qwen models downloaded by the installer.

Local test mode uses SQLite and built-in lightweight models. Use it only as a
temporary fallback when Qwen model dependencies or hardware are not ready:

```bash
fusion-memory init --local-test --json
```

## 2. 启动

```bash
fusion-memory start
```

## 3. 检查状态

```bash
fusion-memory status
```

For machine-readable readiness, use:

```bash
fusion-memory doctor --json
```

The doctor report includes `postgres_connection`, `pgvector`,
`embedding_dependency`, `embedding_readiness`, `reranker_dependency`,
`reranker_readiness`, `service`, and `port` checks, plus a `next_step`.

## 4. 安装 Agent 适配

```bash
fusion-memory install-agent --target all
```

如果失败，运行：

```bash
fusion-memory doctor
```

## 5. 接入 psi-agent

启动 memory 服务后，在 psi-agent 中设置：

Linux / macOS:

```bash
export PSI_MEMORY_BASE_URL=http://127.0.0.1:8700
```

Windows PowerShell:

```powershell
$env:PSI_MEMORY_BASE_URL = "http://127.0.0.1:8700"
```

Windows cmd:

```bat
set PSI_MEMORY_BASE_URL=http://127.0.0.1:8700
```

然后使用带 Fusion Memory tools 的 psi-agent workspace，例如
`examples/haitun-workspace`。当前 agent main 通过 workspace tools 接入，
不需要额外的 agent core memory flag。

## 6. 自动持久化 history

默认情况下，workspace tools 只有在 agent 调用 `memory_add` 时才会写入
Fusion Memory。要让会话 history 持续自动写入，不需要改 agent core；启动
一个 Fusion Memory 侧的同步进程即可。

后台启动 workspace history watcher：

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

一次性回填：

```bash
fusion-memory sync-haitun-history \
  --workspace /path/to/haitun-workspace \
  --session-id <session-id> \
  --once --json
```

同步命令只读取 user/assistant turn，写入 Fusion Memory `/add`，并记录本地
state 文件，重复运行不会重复写入。不要在 WebUI/agent 前台工具里运行不带
`--once` 或 `--background` 的长驻 sync；那会一直等待 watcher 结束。

## 7. 常见问题

- 启动失败：先运行 `fusion-memory doctor`
- 端口被占用：修改本地配置文件里的端口
- Postgres 不可用：启动 Postgres，确认 pgvector 已安装，再运行 `fusion-memory doctor`
- Qwen 模型不可用：安装 Qwen 依赖或确认本地模型缓存/路径，再运行 `fusion-memory doctor`
- API 模型不可用：确认向导里填写的 API key 环境变量已经设置
- 想备份：运行 `fusion-memory backup`
- 升级前检查备份/回滚计划：运行 `fusion-memory upgrade --dry-run --json`
