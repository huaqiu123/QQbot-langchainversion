# Small Agent 后端

基于 **FastAPI + LangChain `create_agent` + DeepSeek** 的可联网问答服务。

接收一个问题，Agent 自主判断是否需要联网检索，多轮调用搜索工具后汇总出**带引用来源**的中文回答。

> 本轮只做 Agent 后端，**不接入 NapCat / OneBot**。后续群消息只需转发到 `/v1/agent/ask` 即可。

---

## 目录结构

```
small_agent/
├── app/
│   ├── main.py        # FastAPI 入口与路由
│   ├── config.py      # 配置（pydantic-settings）
│   ├── schemas.py     # 请求/响应模型（接口契约）
│   ├── llm.py         # ChatDeepSeek 构建（超参集中地）
│   ├── search.py      # 搜索 Provider（DuckDuckGo / Tavily / Serper）
│   ├── tools.py       # @tool 定义的 web_search（content + artifact）
│   ├── prompts.py     # 提示词
│   └── agent.py       # create_agent 封装 + 事件流适配
├── requirements.txt
├── .env.example
└── TASKS.md           # 实现任务清单
```

依赖方向严格单向：`main → agent → (llm, tools, prompts) → (search, config)`

---

## 快速开始

```bash
cd small_agent

# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
# 编辑 .env，至少填入 DEEPSEEK_API_KEY

# 4. 启动
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后：

- 交互式文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

---

## 配置说明

**必填**

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API Key，[申请地址](https://platform.deepseek.com/api_keys) |

**模型**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_MODEL` | `deepseek-chat` | ⚠️ **不要改成 `deepseek-reasoner`**，R1 不支持 function calling，Agent 会直接失效 |
| `LLM_TEMPERATURE` | `0.3` | |
| `LLM_MAX_TOKENS` | `2048` | |
| `LLM_TIMEOUT` | `60` | 单次模型调用超时（秒） |
| `LLM_MAX_RETRIES` | `2` | |

**搜索**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo`（免 Key）\| `tavily` \| `serper` |
| `SEARCH_TOP_K` | `5` | 每次检索返回条数 |
| `SEARCH_TIMEOUT` | `15` | 单次检索超时（秒） |
| `DDGS_BACKENDS` | `bing,brave,duckduckgo` | ddgs 的搜索后端，**逗号分隔、按顺序依次尝试**，命中即返回 |
| `TAVILY_API_KEY` | — | 选 tavily 时必填 |
| `SERPER_API_KEY` | — | 选 serper 时必填 |

> 指定了 Provider 但缺 Key 时会自动降级到 ddgs 并打警告日志。
>
> **关于 `DDGS_BACKENDS`**：`ddgs` 是元搜索库，可指定后端。实测在国内网络下
> `duckduckgo` / `startpage` 后端会超时，而 `bing` / `brave` 正常，因此默认顺序是
> `bing,brave,duckduckgo`。若你的网络环境不同，可自行调整顺序或替换为
> `google` / `mojeek` / `yahoo` / `yandex` 等。全部后端都失败时返回空结果并打警告日志，
> Agent 会如实告知用户「未检索到」而不是编造答案。

**Agent 与服务**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RECURSION_LIMIT` | `12` | 图循环上限，超限会降级返回而非报错 |
| `MAX_HISTORY_MESSAGES` | `10` | 传入模型的历史消息条数上限 |
| `REQUEST_TIMEOUT` | `120` | 单次 `/ask` 总超时（秒） |
| `AGENT_API_KEY` | 空 | 留空则不鉴权；填了之后需带 `X-API-Key` 请求头 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | |
| `LANGSMITH_TRACING` 等 | `false` | 可选链路追踪 |

---

## 接口

### `POST /v1/agent/ask`

```bash
curl -X POST http://127.0.0.1:8000/v1/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "DeepSeek V3 是什么时候发布的？"}'
```

响应：

```json
{
  "answer": "DeepSeek-V3 于 2024 年 12 月发布……[1]",
  "sources": [
    { "title": "...", "url": "https://...", "snippet": "..." }
  ],
  "search_queries": ["DeepSeek V3 发布时间"],
  "iterations": 2,
  "model": "deepseek-chat",
  "usage": { "prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500 }
}
```

**请求字段**

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `question` | ✅ | 用户问题 |
| `session_id` | | 会话标识，本轮仅透传不持久化 |
| `history` | | `[{role, content}]`，由调用方维护；`system` 角色会被忽略 |
| `use_search` | | `false` 时禁用检索，Agent 退化为纯对话；缺省表示允许 |
| `extra_context` | | 附加到问题后的补充上下文 |

### `POST /v1/agent/stream`（SSE）

```bash
curl -N -X POST http://127.0.0.1:8000/v1/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "今天有什么科技新闻？"}'
```

事件流：

```
data: {"type":"search","query":"科技新闻","hits":5}
data: {"type":"delta","content":"根据"}
data: {"type":"delta","content":"最新消息……"}
data: {"type":"done","answer":"完整答案","sources":[...],"search_queries":[...],"iterations":2,"usage":{...}}
```

异常时：`data: {"type":"error","message":"..."}`

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model": "deepseek-chat",
  "search_provider": "duckduckgo",
  "agent_impl": "create_agent",
  "graph_ready": true,
  "llm_configured": true
}
```

> 未配置 `DEEPSEEK_API_KEY` 时服务仍会启动，但 `status=degraded`、`graph_ready=false`，
> 调用 `/v1/agent/ask` 返回 503 并附带具体原因。

---

## 实现要点

### 引用来源怎么拿到

工具用 `response_format="content_and_artifact"`，把输出分成两条路：

```python
@tool(response_format="content_and_artifact")
async def web_search(query: str) -> tuple[str, dict]:
    results = await provider.search(query)
    content = format_results(results)                  # → 模型可见，用于产出 [1][2] 引用
    artifact = {"query": query, "sources": [...]}      # → 模型不可见，供程序读取
    return content, artifact
```

事后遍历图状态里的 `ToolMessage.artifact` 即可还原结构化来源，无需自建收集管道。

### 工具失败不会炸图

`@wrap_tool_call` 中间件兜底：检索异常时返回一条 `ToolMessage` 说明「无法检索」，
模型会据此如实告知用户，而不是整个请求 500。

### 历史消息裁剪

按 `MAX_HISTORY_MESSAGES` 在**输入组装阶段**截断，而不是靠中间件改状态——
行为确定、易于调试。

---

## 与 NapCat 对接（预留）

本轮未实现，后续接入方式：

1. 后端保持现状，暴露 `/v1/agent/ask`
2. NapCat 侧（`connection/`）接收 OneBot 群消息事件，判断是否 @机器人
3. 命中则 POST 到 `/v1/agent/ask`，把 `answer` 通过 `send_group_msg` 下发回群
4. 配置 `AGENT_API_KEY` 并在请求头带 `X-API-Key`，避免接口被随意调用

多轮上下文由 NapCat 侧维护 `history` 并在请求中回传。

---

## 尚未实现

- **T9 `StateGraph` 备选编排**：当前只有 `create_agent` 主线。若后续需要显式控制
  「先判断是否联网 → 再检索 → 再生成」的流程，再补 `app/graph.py`
- 会话持久化（Checkpointer / Redis / Postgres）
- 向量库 RAG
- `/v1/agent/stream` 的总超时控制（目前只有非流式接口有 `REQUEST_TIMEOUT`）
