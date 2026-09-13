# Small Agent 后端实现 —— 任务清单（LangChain 版）

> **范围**：只实现 **Agent 后端**（问题 → 自主搜索 → 汇总 → 回答），不接入 NapCat / OneBot。
> **技术栈**：FastAPI + LangChain 1.x（`create_agent`）+ LangGraph + DeepSeek（`langchain-deepseek`）+ 可插拔搜索源。
> **状态**：✅ **已实现**（所有核心功能已完成编码并启动验证通过）

---

## 一、目标与范围

### 1.1 本轮目标

做一个独立的 HTTP 服务，接收一个问题，由 DeepSeek 驱动 Agent **自主判断是否需要联网、多轮检索、汇总后产出带引用的中文回答**。

后续 NapCat 接入时，只需把群消息转发到本服务的 `/v1/agent/ask` 即可，无需改动 Agent 层。

### 1.2 环境

| 项     | 值                                                       |
| ------ | -------------------------------------------------------- |
| Python | `3.11.2`                                                 |
| pip    | `25.0.1`                                                 |
| 搜索库 | `ddgs`（`duckduckgo-search` 于 2025 年中更名，9.x 版本） |

### 1.3 明确不做（本轮）

- 不接 NapCat / OneBot，不做 WebSocket 事件接收
- 不做 Redis / Kafka 消息队列
- 不接 PostgreSQL / 向量库（RAG 留作后续）
- 不用 Checkpointer / Memory 做会话持久化（`history` 由调用方传入）
- 不引入 LangServe / LangGraph Platform 部署层
- 不做 Docker 编排
- 不做多 Agent 编排（Sub-agent / Supervisor）
- 不做 `StateGraph` 手写图备选（仅使用 `create_agent` / `create_react_agent` 主线）

---

## 二、技术选型与关键决策

### 2.1 选型表

| 决策点         | 选择                                                             | 理由                                                                                                                                                                              |
| -------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent 构建方式 | **`class Agent` 封装 `create_agent` / `create_react_agent`**     | 用类而非函数，构造时自动编译两张图，对外提供 `run()` / `stream()` 统一接口；底层兼容 LangChain 1.x 与旧版 `langgraph.prebuilt`                                                    |
| 版本策略       | 兼容 `langchain>=1.0` 和 `langgraph.prebuilt` 两条路径           | 通过 `_resolve_create_agent()` 自动探测可用实现，不锁定版本                                                                                                                       |
| 模型接入       | 官方 `langchain-deepseek` 的 `ChatDeepSeek`，逃生口 `ChatOpenAI` | 比 `ChatOpenAI(base_url=...)` 更地道；未安装时自动降级                                                                                                                            |
| 模型构建位置   | 独立 `app/llm.py`                                                | `create_agent` 只接受 `model`，**不接受** `temperature` / `max_tokens` / `timeout` / `base_url`，这些超参必须由模型实例承载；抽成独立模块便于集中管理超参、切换厂商实现、单独单测 |
| 工具定义       | `langchain_core.tools.@tool` 装饰器                              | docstring 自动生成 schema，交给 `create_agent` 绑定，比手写 JSON Schema 更稳                                                                                                      |
| 中间件扩展     | **只使用 `@wrap_tool_call`** 异常兜底                            | 历史裁剪在 `_build_messages()` 中完成（更确定、易调试），不依赖 `@before_model`；去除 `@after_model` 日志中间件，直接在 `run()` 中打日志                                          |
| 结构化来源收集 | `@tool(response_format="content_and_artifact")`                  | LangChain 原生机制：工具返回 `(content, artifact)` 元组，`content` 给模型、`artifact` 留在 `ToolMessage` 里不进模型上下文，天然解决「来源丢失」                                   |
| 搜索 Provider  | 自建（`httpx` / `ddgs`）                                         | 框架自带搜索工具只返回纯文本，拿不到 title / url / snippet，无法产出引用                                                                                                          |
| 流式输出       | `graph.astream(stream_mode=["messages", "updates"])`             | 分别拿 token 增量与工具调用事件，映射成统一 SSE                                                                                                                                   |
| 会话记忆       | 本轮不用 Checkpointer，`history` 由调用方传入                    | 避免引入持久化依赖，后续接 Redis / Postgres 时再加                                                                                                                                |
| 搜索禁用机制   | **双图编译**（`graph` + `graph_plain`）                          | 编译两张物理上不同的图，`use_search=False` 时切换到无工具图，物理上杜绝模型调用工具；比"靠提示词劝模型"更可靠                                                                     |

### 2.2 已知坑与对策

| 坑                                                      | 后果                                                           | 对策                                                                                                                                                                    |
| ------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deepseek-reasoner`（R1）**不支持工具调用与结构化输出** | Agent 直接失效                                                 | 只用 `deepseek-chat`（V3）；`llm.py` 在启动时通过 `validate_model()` 校验模型名并告警                                                                                   |
| `create_agent` 不接受超参                               | `temperature` 等无法配置                                       | 由 `llm.py` 构造实例后传入（见 2.1）                                                                                                                                    |
| `ToolMessage.content` 会进模型上下文                    | 结构化来源若塞进 content，既污染上下文、又可能被模型改写或遗漏 | 用 `response_format="content_and_artifact"`，来源放 `artifact`                                                                                                          |
| `artifact` 在持久化时可能丢失                           | 将来接 Checkpointer 后来源取不到                               | 本轮不挂 Checkpointer，`artifact` 只存活于内存状态；后续持久化需验证序列化配置                                                                                          |
| `create_agent` 图循环可能不收敛                         | 请求悬挂 / 报错                                                | 配置 `recursion_limit`，捕获 `GraphRecursionError` 降级返回 `RECURSION_ANSWER`                                                                                          |
| 国内网络下 ddgs 的部分后端超时                          | 检索恒为空，Agent 答不了实时问题                               | `DuckDuckGoProvider` 按 `DDGS_BACKENDS` 顺序**多后端降级**（依次尝试 bing / brave / duckduckgo 等）；另可切换 Tavily / Serper；检索失败不炸图（`@wrap_tool_call` 兜底） |
| `ToolException` 传播到图外                              | 整个请求变成 HTTP 500                                          | 工具内异常转 `ToolException`，配合 `@wrap_tool_call` 中间件捕获后返回友好的 `ToolMessage`                                                                               |

### 2.3 目录结构（实际）

```
small_agent/
├── app/
│   ├── __init__.py        # 版本号 0.1.0
│   ├── main.py            # FastAPI 入口与路由（CORSMiddleware + 鉴权 + 异常映射）
│   ├── config.py          # 配置（pydantic-settings + lru_cache 单例）
│   ├── schemas.py         # 请求/响应模型（Pydantic）
│   ├── llm.py             # ChatDeepSeek 构建（超参集中地 + 模型校验 + 版本回退）
│   ├── search.py          # 搜索 Provider（DuckDuckGo / Tavily / Serper）
│   ├── tools.py           # @tool 定义的 web_search（content + artifact 双通道）
│   ├── prompts.py         # 系统提示词
│   └── agent.py           # Agent 类（组图 + 双图编译 + 执行 + 事件流适配）
├── .env.example           # 配置模板
├── .gitignore
├── requirements.txt
├── README.md
└── TASKS.md               # 本文档
```

> **注**：原先规划的 `graph.py`（StateGraph 备选）**未实现** —— 项目仅使用 `create_agent` / `create_react_agent` 主线，未手写 LangGraph `StateGraph`。

---

## 三、模块职责边界

严格单向依赖，不互相渗透：

| 模块         | 只负责                                                   | 不负责                       |
| ------------ | -------------------------------------------------------- | ---------------------------- |
| `config.py`  | 读环境变量、给默认值、提供派生判断方法                   | 不构造任何客户端             |
| `llm.py`     | 配置 → 模型实例 + 启动期校验                             | 不碰工具、不碰图、不碰提示词 |
| `search.py`  | 关键词 → 结构化 `SearchResult`                           | 不感知 LangChain             |
| `tools.py`   | `SearchResult` → LangChain Tool（含 content + artifact） | 不构造模型、不建图           |
| `prompts.py` | 纯文本提示词                                             | 无逻辑                       |
| `agent.py`   | 组双图、消息组装、执行（run/stream）、结果解析           | 不碰模型超参、不碰 HTTP      |
| `main.py`    | HTTP 路由、鉴权、异常映射、生命周期                      | 不含业务逻辑                 |

依赖方向：`main → agent → (llm, tools, prompts) → (search, config)`，无反向依赖。

---

## 四、任务清单

### T1 项目骨架 ✅
- [x] 建立 `small_agent/` 目录结构与 `app/` 包
- [x] `requirements.txt`（fastapi, uvicorn, pydantic, pydantic-settings, httpx, ddgs, langchain, langchain-deepseek, langgraph）
- [x] `.env.example`（对照第六节配置清单）
- [x] `.gitignore`（`.env`, `__pycache__`, `*.pyc`, `.venv`）

### T2 配置层 `app/config.py` ✅
- [x] 基于 `pydantic-settings` 的 `Settings` 类 + `get_settings()`（`lru_cache` 单例）
- [x] 从 `small_agent/.env` 读取（用 `Path(__file__)` 定位，不依赖启动目录）
- [x] 字段名小写 → 环境变量大写自动映射（`case_sensitive=False`）
- [x] 额外新增字段：`search_timeout`（搜索超时）、`ddgs_backends`（多后端降级配置）
- [x] 派生判断方法：
  - `is_llm_configured()` — 是否配置了 API Key
  - `model_supports_tools()` — 模型是否支持 function calling（白名单判断）
  - `effective_search_provider()` — 实际生效的搜索源（缺 Key 自动降级）
  - `ddgs_backend_list()` — 解析逗号分隔的后端名列表

> **与原始设计的不同**：
> - 没有实现 `is_search_configured()`，改用 `effective_search_provider()` 替代
> - 新增 `model_supports_tools()` 和 `ddgs_backend_list()` 方法
> - 新增 `search_timeout`、`ddgs_backends` 配置字段

### T3 数据模型 `app/schemas.py` ✅
- [x] `ChatMessage`：`role`(system\|user\|assistant) / `content`
- [x] `AskRequest`：`question`(必填) / `session_id` / `history` / `use_search` / `extra_context`
- [x] `Source`：`title` / `url` / `snippet`
- [x] `AskResponse`：`answer` / `sources` / `search_queries` / `iterations` / `model` / `usage`
- [x] `HealthResponse`：`status` / `model` / `search_provider` / `agent_impl` / `graph_ready` / `llm_configured`

### T4 搜索 Provider 层 `app/search.py` ✅
- [x] `SearchResult` 数据类（`title` / `url` / `snippet`）+ `to_dict()` 序列化
- [x] `SearchProvider` 抽象基类，统一 `async def search(query, top_k) -> list[SearchResult]`
- [x] `DuckDuckGoProvider`：`ddgs`，用 `asyncio.to_thread` 包装同步库；兼容旧包名 `duckduckgo_search` 回退；支持按 `DDGS_BACKENDS` 顺序多后端降级
- [x] `TavilyProvider`：`httpx.AsyncClient` 调 `https://api.tavily.com/search`
- [x] `SerperProvider`：`httpx.AsyncClient` 调 `https://google.serper.dev/search`
- [x] `build_search_provider(settings)` 工厂：指定 Provider 但缺 Key 时降级到 DuckDuckGo + 日志告警
- [x] 所有 Provider 内部吞掉网络异常，返回空列表

### T5 模型构建层 `app/llm.py` ✅
- [x] `build_chat_model(settings) -> BaseChatModel`：ChatDeepSeek 实例构造，绑定 temperature / max_tokens / timeout / max_retries
- [x] 优先 `ChatDeepSeek`；逃生口：`ChatOpenAI(model=..., base_url="https://api.deepseek.com")`
- [x] 启动校验：`DEEPSEEK_API_KEY` 缺失 → 明确 `RuntimeError`（被 `main.py` 捕获后挂到 `app.state.agent_error`）
- [x] 启动校验：`validate_model()` — 模型为 `deepseek-reasoner` 且工具有值 → `logger.warning` 告警
- [x] 暴露 `model_name()` 供 `/health` 与响应体使用
- [x] `model_name()` 做 `lru_cache` 缓存

### T6 工具层 `app/tools.py` ✅
- [x] `@tool(response_format="content_and_artifact")` 定义 `web_search(query: str) -> tuple[str, dict]`
- [x] content（给模型）：编号文本 `[1] {title}\n{url}\n{snippet}`
- [x] artifact（给程序）：`{"query": query, "sources": [{title, url, snippet}, ...]}`，不进模型上下文
- [x] `build_tools(settings, provider)` — 通过闭包固定搜索依赖与条数，**不**直接按 `use_search` 开关返回空列表（由 Agent 编译时控制）
- [x] 工具内异常转 `ToolException`，配合中间件兜底

> **与原始设计的不同**：
> - `build_tools` 签名是 `(settings, provider)` 而非 `(settings, tools)`
> - 不在这里根据 `use_search` 返回 `[]`，而是由 Agent 编译时通过 `_compile([])` 控制

### T7 提示词 `app/prompts.py` ✅
- [x] `SYSTEM_PROMPT`（中文）包含：
  - 何时**必须**搜索：实时信息、新闻、具体数据、事实核查、不确定的内容
  - 何时**可以**直接答：常识、闲聊、代码/数学推理
  - 必须基于检索结果作答，信息不足时如实说明「未找到足够信息」，禁止编造
  - 引用规范：用 `[1] [2]` 标注，序号与检索结果编号一致
  - 允许同一问题多次检索 / 拆分关键词
  - 输出语言与风格：简体中文、简洁、必要时分点
- [x] 以 `create_agent(system_prompt=SYSTEM_PROMPT)` 接入

### T8 Agent 核心 `app/agent.py`（重点 · 主线）✅
> **使用 `class Agent` 封装，不自写 ReAct 循环、不手拼 tool 消息。**

- [x] **`class Agent`**:
  - `__init__(settings, provider)` — 构造时同时编译两张图（`graph` + `graph_plain`）
  - `_resolve_create_agent()` — 自动探测 `langchain.agents.create_agent` 或回退 `langgraph.prebuilt.create_react_agent`
  - `_compile(tools)` — 根据可用 API 编译一张图
  - `_pick_graph(use_search)` — 请求级选择带/不带工具的图

- [x] **中间件**（只使用 `@wrap_tool_call`）：
  - `handle_tool_errors` — 工具异常兜底，返回 `ToolMessage` 错误文本而非抛异常

- [x] **`_build_messages(question, history, extra_context)`** — 消息组装 + 历史裁剪：
  - 按 `max_history_messages` 截断（不是通过 `@before_model` 中间件）
  - 忽略 `system` 角色（系统提示词由 `create_agent` 统一注入）
  - `extra_context` 附加到问题末尾

- [x] **`run(...) -> AgentResult`**：
  - 选图 → 组消息 → `graph.ainvoke()` → 解析结果
  - `_final_answer()` — 从后往前找第一条不带 `tool_calls` 的 AIMessage
  - `_collect_sources()` — 遍历 ToolMessage.artifact，按 URL 去重
  - `_collect_usage()` — 累加所有 AIMessage.usage_metadata
  - 捕获 `GraphRecursionError` → 返回 `RECURSION_ANSWER` 兜底话术
  - 结构化日志：`用时=%.2fs 轮次=%d 检索=%d 来源=%d`

- [x] **`stream(...) -> AsyncIterator[dict]`**：
  - `astream(stream_mode=["messages", "updates"])`
  - `messages` 模式 → 过滤 `AIMessageChunk`（无 tool_call_chunks）→ 产出 `delta` 事件
  - `updates` 模式 → 捕获 `ToolMessage.artifact` → 产出 `search` 事件
  - 累加 token 用量（stream 过程中增量汇总）
  - 结束 → `done`；异常 → `error`

- [x] **辅助函数**：
  - `_text_of(content)` — 归一化 content 类型（兼容 str / content blocks 列表）
  - `_read_artifact(message)` — 从 ToolMessage.artifact 安全取数
  - `_iter_updates(payload)` — 归一化 updates 模式载荷

> **与原始设计的不同**：
> - **`build_agent` 函数 → `class Agent` 类**：用类替代函数，构造时自动完成全部初始化
> - **双图编译**：自动编译 `graph`（带工具）+ `graph_plain`（无工具），而非运行时动态组装
> - **历史裁剪在 `_build_messages()` 中完成**，不使用 `@before_model` 中间件
> - **日志直接在 `run()` 中记录**，不使用 `@after_model` 中间件
> - **`configurable.collector` 不再使用**，改为直接从 `ToolMessage.artifact` 事后遍历收集

### T9 备选编排 `app/graph.py`（StateGraph）❌ 未实现
> **此任务标记为「可选进阶」，实际编码中已跳过**。项目只使用 `create_agent`/`create_react_agent` 主线，不手写 `StateGraph`。当前 `create_agent` 的 ReAct 循环已满足所有需求。

### T10 FastAPI 服务 `app/main.py` ✅
- [x] **`lifespan`**：初始化全局单例（`SearchProvider` + `Agent` 图），`try/except` 包住 Agent 构造实现**带病启动**
- [x] **`_configure_tracing()`**：LangSmith 追踪配置（setdefault 让进程环境变量优先）
- [x] **`_get_agent(request)`**：取出 Agent 实例，未就绪时返回 503
- [x] **`GET /`**：服务根路径（服务名 + 版本号 + docs 链接）
- [x] **`GET /health`**：返回 `HealthResponse`（配置层用 `get_settings()`，运行层用 `getattr` 兜底）
- [x] **`POST /v1/agent/ask`**：非流式问答，`asyncio.wait_for` 超时保护（504 超时 / 502 上游异常）
- [x] **`POST /v1/agent/stream`**：SSE 流式问答（`StreamingResponse` + `text/event-stream`）
- [x] **`require_api_key`**：可选鉴权依赖（`X-API-Key` 头 vs `AGENT_API_KEY` 配置）
- [x] **`CORSMiddleware`** + 结构化日志 + 请求耗时记录

### T11 文档与验证 ✅
- [x] `README.md`：安装、配置、启动命令、curl 示例
- [x] 依赖安装并启动验证（服务已运行，`/health` 返回 `"status": "ok"`）
- [x] 接口测试（`/v1/agent/ask` 已测试，DeepSeek 正常响应，DuckDuckGo 搜索触发成功）

---

## 五、接口契约

### 5.1 `POST /v1/agent/ask`

```jsonc
// 请求
{
  "question": "DeepSeek V3 是什么时候发布的？",
  "session_id": "optional-any-string",
  "history": [ { "role": "user", "content": "..." } ],
  "use_search": true,          // 省略则用服务端默认（允许）
  "extra_context": "可选补充上下文"
}
```

```jsonc
// 响应
{
  "answer": "……详见 [1][2]",
  "sources": [ { "title": "...", "url": "...", "snippet": "..." } ],
  "search_queries": ["DeepSeek V3 发布时间"],
  "iterations": 2,
  "model": "deepseek-chat",
  "usage": { "prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500 }
}
```

### 5.2 `POST /v1/agent/stream`（SSE）

```
data: {"type":"search","query":"DeepSeek V3 发布时间","hits":5}
data: {"type":"delta","content":"DeepSeek"}
data: {"type":"delta","content":" V3 发布于……"}
data: {"type":"done","answer":"完整答案","sources":[...],"usage":{...}}
```

异常时：`data: {"type":"error","message":"..."}`

### 5.3 `GET /health`

```jsonc
{
  "status": "ok",
  "model": "deepseek-chat",
  "search_provider": "duckduckgo",
  "agent_impl": "create_agent",
  "graph_ready": true,
  "llm_configured": true
}
```

---

## 六、配置项清单（`.env`）

| 变量                   | 默认值                  | 说明                                           |
| ---------------------- | ----------------------- | ---------------------------------------------- |
| `DEEPSEEK_API_KEY`     | —                       | **必填**，`ChatDeepSeek` 自动读取              |
| `DEEPSEEK_MODEL`       | `deepseek-chat`         | 必须支持工具调用，**勿用 `deepseek-reasoner`** |
| `LLM_TEMPERATURE`      | `0.3`                   |                                                |
| `LLM_MAX_TOKENS`       | `2048`                  |                                                |
| `LLM_TIMEOUT`          | `60`                    | 单次模型调用超时（秒）                         |
| `LLM_MAX_RETRIES`      | `2`                     |                                                |
| `SEARCH_PROVIDER`      | `duckduckgo`            | `duckduckgo` \| `tavily` \| `serper`           |
| `SEARCH_TOP_K`         | `5`                     | 每次检索返回条数                               |
| `SEARCH_TIMEOUT`       | `15.0`                  | 搜索请求超时（秒）                             |
| `DDGS_BACKENDS`        | `bing,brave,duckduckgo` | ddgs 多后端降级顺序（逗号分隔）                |
| `TAVILY_API_KEY`       | —                       | 选 tavily 时必填                               |
| `SERPER_API_KEY`       | —                       | 选 serper 时必填                               |
| `RECURSION_LIMIT`      | `12`                    | 图循环上限                                     |
| `MAX_HISTORY_MESSAGES` | `10`                    | 传入模型的历史消息条数上限（`<=0` 不限制）     |
| `REQUEST_TIMEOUT`      | `120`                   | 单次 `/ask` 总超时（秒）                       |
| `AGENT_API_KEY`        | 空                      | 鉴权 Key，为空则不鉴权                         |
| `LANGSMITH_TRACING`    | `false`                 | 可选链路追踪                                   |
| `LANGSMITH_API_KEY`    | —                       |                                                |
| `LANGSMITH_PROJECT`    | `small-agent`           |                                                |
| `HOST`                 | `0.0.0.0`               |                                                |
| `PORT`                 | `8000`                  |                                                |

> **与原始设计的差异**：
> - 新增 `SEARCH_TIMEOUT`、`DDGS_BACKENDS` 字段
> - 移除 `AGENT_IMPL`（已无 StateGraph 切换需求）
> - `MAX_HISTORY_MESSAGES` 支持 `<=0` 表示不限制

---

## 七、已决定的待确认项

以下为项目初期列出的待确认项，**已在实际编码中确定**：

| 待确认项       | 决定                                                                           |
| -------------- | ------------------------------------------------------------------------------ |
| 代码位置       | `small_agent/` 目录下                                                          |
| 默认搜索源     | `duckduckgo`（ddgs 免 Key，国内网络按 `DDGS_BACKENDS` 多后端降级）             |
| 流式接口       | ✅ 已实现 SSE `/v1/agent/stream`                                                |
| 鉴权           | ✅ 已实现 `X-API-Key` 可选鉴权                                                  |
| `AGENT_IMPL`   | 默认且仅使用 `create_agent` / `create_react_agent`，**不实现 StateGraph 备选** |
| LangSmith 追踪 | ✅ 已实现（通过 `_configure_tracing()` 支持，需额外 Key）                       |