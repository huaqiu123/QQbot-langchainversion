# Agent V3：持久化记忆管理改造 Plan

## 架构概览

本次改造**不需要新增模块**，改动集中在现有的 4 个 Python 文件中，外加 1 个新的消息裁剪函数。

### 组件划分

| 组件 | 文件 | 改动类型 | 职责 |
|------|------|---------|------|
| Checkpointer 工厂 | `main.py`（lifespan） | 修改 | 启动时创建 `AsyncSqliteSaver`，建表，注入 Agent |
| Agent 核心 | `agent.py` | 修改 | 接收 checkpointer，传给 `create_agent`；用 `user_id` 替代 `history`；新增消息裁剪逻辑 |
| 配置 | `config.py` | 修改 | 新增 `checkpoint_db_path` 配置项 |
| 请求模型 | `schemas.py` | 修改 | `AskRequest` 新增必填 `user_id`，删除 `history`、`session_id` |
| 路由 | `main.py`（路由层） | 修改 | 从请求中取 `user_id` 传给 Agent，去掉 `history` 参数 |

### 数据流

```
HTTP Request { user_id, question, use_search, extra_context }
  │
  ▼
main.py: 取 payload.user_id → agent.run(user_id=..., question=..., ...)
  │
  ▼
agent.py:
  1. 知识库检索 → knowledge_text
  2. 构造本轮 HumanMessage（含 knowledge_text + extra_context）
  3. config = {"configurable": {"thread_id": user_id}, "recursion_limit": ...}
  4. graph.ainvoke({"messages": [HumanMessage(...)]}, config=config)
     │
     ├── [checkpointer] 从 SQLite 加载该 thread 的全部历史
     ├── [trim_middleware] 裁剪到最近 max_history_messages 条
     ├── LLM 调用（可能多轮 tool calling）
     ├── [checkpointer] 把本轮新消息写入 SQLite
     └── 返回完整 state（含全部历史）
  5. 解析最终回答 → AgentResult
  │
  ▼
main.py: AgentResult → AskResponse → HTTP 200
```

---

## 核心数据结构

### AskRequest（修改）

删除 `history` 和 `session_id`，新增必填 `user_id`：

```python
class AskRequest(BaseModel):
    question: str      = Field(..., min_length=1, description="用户问题")
    user_id: str       = Field(..., min_length=1, description="用户标识，用于隔离不同用户的对话历史")
    use_search: Optional[bool] = Field(default=None, description="是否允许联网；None 表示用服务端默认（允许）")
    extra_context: Optional[str] = Field(default=None, description="附加到问题后的补充上下文")
    # 删除: history, session_id
```

### Settings（修改）

```python
# ---------- 记忆 ----------
checkpoint_db_path: str = "data/checkpoint.db"   # 新增
```

`max_history_messages` 语义从"裁剪调用方传入的 history"变为"裁剪 checkpointer 加载后的消息列表"，其余字段不变。

### Agent 构造器（修改）

```python
class Agent:
    def __init__(
        self,
        settings: Settings,
        search_provider: SearchProvider,
        knowledge_base: KnowledgeBase | None = None,
        checkpointer: Any | None = None,          # 新增
    ) -> None:
```

`checkpointer` 在 `_compile` 中传给 `create_agent`。

### Agent.run() 签名（修改）

```python
async def run(
    self,
    question: str,
    user_id: str,                                 # 替代 history
    use_search: Optional[bool] = None,
    extra_context: Optional[str] = None,
) -> AgentResult:
```

内部 config 构造：

```python
config = {
    "configurable": {"thread_id": user_id},       # user_id → thread_id 映射
    "recursion_limit": self.settings.recursion_limit,
}
```

### Agent.stream() 签名（修改）

```python
async def stream(
    self,
    question: str,
    user_id: str,                                 # 替代 history
    use_search: Optional[bool] = None,
    extra_context: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
```

---

## 模块设计

### [模块 A] 配置层 — `config.py`

**职责：** 新增 `checkpoint_db_path` 配置项，其余字段不变。

**对外接口：** `get_settings() -> Settings` 不变，调用方无感知。

**依赖：** 无（标准 pydantic-settings）。

**改动量：** +1 行。

---

### [模块 B] 数据模型层 — `schemas.py`

**职责：** `AskRequest` 新增必填 `user_id`，删除 `history` 和 `session_id`。

**对外接口：**

| 模型 | 改动 |
|------|------|
| `AskRequest` | `+ user_id: str`（必填），`- history`，`- session_id` |
| `ChatMessage` | 保留（可能有其他内部用途） |
| 其余 | 不变 |

**依赖：** 无（纯 pydantic 模型）。

**改动量：** ~5 行。

---

### [模块 C] Agent 核心 — `agent.py`

**职责：** 改动最大的模块。核心变化：
- 接收 checkpointer，传入 `create_agent`
- 新增消息裁剪逻辑
- `run()` / `stream()` 签名改为 `user_id` 替代 `history`
- `_build_messages` 重命名为 `_build_input`，去掉历史拼接

**对外接口：**

| 方法 | 改动 |
|------|------|
| `Agent.__init__` | `+ checkpointer` 参数 |
| `Agent._compile` | 传给 `create_agent(checkpointer=...)` |
| `Agent.run` | `history` → `user_id`，内部构造 `thread_id` config |
| `Agent.stream` | 同上 |
| `_build_messages` → `_build_input` | 只注入知识库 + extra_context，返回 `[HumanMessage]` |
| `trim_messages`（新增） | 消息裁剪函数，保留最近 `max_history_messages` 条 |

**依赖：** `config.py`（Settings）、`tools.py`、`prompts.py`、`llm.py`、`schemas.py`、langgraph checkpointer。

**改动量：** ~40 行。

---

### [模块 D] HTTP 服务层 — `main.py`

**职责：** 
- lifespan 中创建 `AsyncSqliteSaver`，建表，注入 Agent
- ask / stream 路由从 `payload.user_id` 取参数，删除 `history` 相关代码

**对外接口：**

| 路由 | 改动 |
|------|------|
| `POST /v1/agent/ask` | `agent.run(history=...)` → `agent.run(user_id=payload.user_id, ...)` |
| `POST /v1/agent/stream` | 同上 |
| `/health` | 可新增 `checkpointer_ready` 字段（可选） |

**依赖：** `agent.py`、`config.py`、`schemas.py`、langgraph checkpoint sqlite aio。

**改动量：** ~20 行。

---

## 模块交互

### 1. 启动时装配（main.py lifespan）

```
get_settings()
  │
  ├─→ build_search_provider(settings)
  ├─→ build_embeddings(settings) → MessageStore + KnowledgeBase(...)
  ├─→ AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path)
  │     └─→ checkpointer.setup()                    # 建表 + WAL
  │
  └─→ Agent(settings, provider, knowledge_base, checkpointer)
        └─→ Agent._compile(tools) → create_agent(..., checkpointer=checkpointer)
```

### 2. 非流式请求（POST /v1/agent/ask）

```
main.ask(payload)
  │
  ├─→ _get_agent(request)                           # 取出 Agent 单例
  ├─→ agent.run(
  │       question=payload.question,
  │       user_id=payload.user_id,
  │       use_search=payload.use_search,
  │       extra_context=payload.extra_context,
  │   )
  │   │
  │   ├─→ _knowledge_search(question)                # 查知识库（如有）
  │   ├─→ _build_input(question, extra_context, knowledge_text)
  │   │     └─→ 返回 [HumanMessage("【内部知识库】...\n【用户问题】...")]
  │   │
  │   ├─→ graph.ainvoke(
  │   │       {"messages": [HumanMessage(...)]},
  │   │       config={"configurable": {"thread_id": user_id}, ...}
  │   │   )
  │   │   │
  │   │   ├─→ [checkpointer] 从 SQLite 加载该 thread 的全部历史
  │   │   ├─→ [trim_messages] 裁剪到最近 N 条
  │   │   ├─→ LLM 调用（可能多轮 tool calling）
  │   │   ├─→ [checkpointer] 把本轮新消息写入 SQLite
  │   │   └─→ 返回完整 state（含全部历史）
  │   │
  │   ├─→ _final_answer(state["messages"])           # 提取最终 AIMessage
  │   ├─→ _collect_sources(...)                      # 汇总检索来源
  │   └─→ _collect_usage(...)                        # 汇总 token 用量
  │
  └─→ AgentResult → AskResponse → HTTP 200
```

### 3. 流式请求（POST /v1/agent/stream）

流程与非流式一致，用 `graph.astream(stream_mode=["messages", "updates"])` 替代 `graph.ainvoke`。事件生成逻辑不变。

### 4. Checkpointer 内部交互

```
graph.ainvoke({"messages": [new_msg]}, config)
  │
  ├─ 1. checkpointer.get(config)           # SELECT ... WHERE thread_id = ?
  │     └─ 返回该 thread 的历史 state（如有）
  │
  ├─ 2. 合并：历史 messages + 新 messages → 完整消息列表
  │
  ├─ 3. trim_messages（LLM 调用前裁剪）
  │
  ├─ 4. LLM 执行（agent 节点 + tools 节点循环）
  │
  └─ 5. checkpointer.put(config, new_state) # INSERT/UPDATE 完整 state
        └─ 每个 checkpoint 是一个序列化的完整 state 快照
```

---

## 文件组织

```
agent/
├── app/
│   ├── config.py          — 修改：+ checkpoint_db_path
│   ├── schemas.py         — 修改：AskRequest +user_id, -history, -session_id
│   ├── agent.py           — 修改：+checkpointer, +trim, user_id 替代 history
│   ├── main.py            — 修改：lifespan 创建 AsyncSqliteSaver，路由适配
│   ├── tools.py           — 不变
│   ├── prompts.py         — 不变
│   ├── llm.py             — 不变
│   ├── search.py          — 不变
│   ├── knowledge.py       — 不变
│   ├── database.py        — 不变
│   └── __init__.py        — 不变
├── v3/
│   ├── spec.md            — 已写入
│   ├── plan.md            — 本文档
│   └── 需求.md            — 原始需求
├── requirements.txt       — 修改：+ langgraph-checkpoint-sqlite（如需）
└── data/
    └── checkpoint.db      — 新增：运行时自动创建
```

---

## 技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Checkpointer 类型 | `AsyncSqliteSaver` | Agent 使用 `ainvoke`/`astream`，必须用异步版；SQLite 零依赖、免运维 |
| `user_id` → `thread_id` 映射 | 直接赋值，不做 hash/编码 | 简单透明，调用方传入的原始 user_id 就是 SQLite 中的 key |
| 消息裁剪实现 | `trim_messages` 函数，作为 middleware 挂到 agent 节点 | `create_agent` 的 middleware 机制天然支持在 LLM 调用前改写消息；比改 state reducer 更可控、不影响 checkpointer 存储 |
| `_build_messages` 处理 | 重命名为 `_build_input`，只做注入不做拼接 | 历史归 checkpointer 管，本方法职责从"拼历史 + 注入"变为"纯注入" |
| `max_history_messages <= 0` | 视为不限制，跳过裁剪 | 保持与旧配置语义一致 |
| `/health` 变更 | 不新增 checkpointer 字段 | 改动最小；checkpointer 异常会在 Agent 初始化时暴露（`agent_error`） |
| `ChatMessage` 模型 | 保留 | 可能有其他内部用途（如 knowledge 模块），不冒险删除 |
| checkpointer 传入 `create_agent` | 显式 `checkpointer=` 参数 | `create_agent` 原生支持该参数；若版本不兼容则回退到 `**kwargs` |