# Agent V3：持久化记忆管理改造 Tasks

## 文件清单

| 操作 | 文件 | 职责 |
|------|------|------|
| 修改 | `agent/app/config.py` | 新增 `checkpoint_db_path` 配置项 |
| 修改 | `agent/app/schemas.py` | `AskRequest` 新增必填 `user_id`，删除 `history`、`session_id` |
| 修改 | `agent/app/agent.py` | 接收 checkpointer、注入 create_agent；用 `user_id` 替代 `history`；新增消息裁剪中间件 |
| 修改 | `agent/app/main.py` | lifespan 创建 `AsyncSqliteSaver` 并注入 Agent；路由改用 `user_id` |
| 修改 | `agent/requirements.txt` | 如需要，新增 `langgraph-checkpoint-sqlite` 依赖（可能已内置） |

---

## T1: config.py — 新增 checkpoint_db_path

**文件：** `agent/app/config.py`
**依赖：** 无

**步骤：**

1. 在 `Settings` 类的「记忆」区域下方新增一行配置
2. 字段名 `checkpoint_db_path`，类型 `str`，默认值 `"data/checkpoint.db"`

**验证：** `from app.config import get_settings; print(get_settings().checkpoint_db_path)` → 输出 `"data/checkpoint.db"`

---

## T2: schemas.py — AskRequest 字段重构

**文件：** `agent/app/schemas.py`
**依赖：** 无（与 T1 可并行）

**步骤：**

1. 删除 `history: Optional[List[ChatMessage]]` 字段及其 Field 定义
2. 删除 `session_id: Optional[str]` 字段
3. 新增 `user_id: str = Field(..., min_length=1, description="用户标识，用于隔离不同用户的对话历史")`

**验证：**

- `AskRequest(question="hi")` → Pydantic ValidationError（缺少 user_id）
- `AskRequest(question="hi", user_id="u1")` → 构造成功
- `AskRequest(question="hi", user_id="u1", history=[...])` → Pydantic ValidationError（多余字段）

---

## T3: agent.py — 核心改造

### T3.1: 构造器 + _compile 适配

**文件：** `agent/app/agent.py`
**依赖：** T1, T2

**步骤：**

1. `__init__` 新增参数 `checkpointer = None`，保存为 `self.checkpointer`
2. `_compile` 方法中，调用 `create_agent` 时追加 `checkpointer=self.checkpointer`

**验证：** `Agent(settings, search_provider, checkpointer=mock_checkpointer)` 不报错

---

### T3.2: _build_messages → _build_input

**文件：** `agent/app/agent.py`
**依赖：** T3.1

**步骤：**

1. 将 `_build_messages(self, question, history, extra_context, knowledge_text)` 重命名为 `_build_input(self, question, extra_context, knowledge_text)`
2. 删除方法体内 history 遍历/拼接代码（约 3 行）
3. 方法直接返回 `[HumanMessage(content=assembled_text)]`

**验证：** 调用 `_build_input("hi", "ctx", "kb")` 返回一个 `HumanMessage`，内容包含 "hi"、"ctx"、"kb"

---

### T3.3: run() + stream() 签名改造

**文件：** `agent/app/agent.py`
**依赖：** T3.2

**步骤：**

1. `run()` 签名：`history` 参数替换为 `user_id: str`
2. `stream()` 签名：同上
3. 两方法内部，`config` 字典新增 `"configurable": {"thread_id": user_id}`
4. `_build_messages` 调用改为 `_build_input`
5. `await graph.ainvoke/astream` 的 `config` 参数指向完整 config dict

**验证：** `await agent.run("hi", user_id="u1")` 正常执行并返回 `AgentResult`

---

### T3.4: 消息裁剪中间件（新增）

**文件：** `agent/app/agent.py`
**依赖：** T3.1

**步骤：**

1. 在 `_compile` 方法中或 `agent.py` 顶层，新增 `_trim_messages` 函数，签名：`_trim_messages(state, *, max_count: int)`
2. 逻辑：当 `max_count > 0` 时，`state["messages"][-max_count:]`；否则原样返回
3. 将该函数挂载为 `create_agent` 的 middleware 列表成员
4. 若 `create_agent` 不支持 middleware 参数，改为在编译后手动包装 agent 节点的 before_model 钩子

**验证：** `max_history_messages=2` 时，发 5 轮对话，观察 LLM 实际收到的上下文 ≤ 2 轮

---

## T4: main.py — 生命周期 + 路由适配

**文件：** `agent/app/main.py`
**依赖：** T3（Agent 签名变更后，main.py 才能编译通过）

### T4.1: lifespan 装配

**步骤：**

1. 在 lifespan 函数中，在 `KnowledgeBase` 初始化之后、`Agent` 构造之前，插入 checkpointer 创建逻辑
2. `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`
3. `checkpointer = AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path)`
4. `await checkpointer.setup()`
5. `Agent(..., checkpointer=checkpointer)`

**验证：** 启动服务无报错，`data/checkpoint.db` 文件自动生成

---

### T4.2: 路由适配

**步骤：**

1. `ask` 路由：`agent.run(history=...)` → `agent.run(user_id=payload.user_id, ...)`
2. `stream` 路由：同上
3. 删除路由中 `history` 相关的参数传递代码

**验证：** 发 `POST /v1/agent/ask {"question":"hi","user_id":"u1"}` → 200 正常响应

---

## T5: requirements.txt — 依赖声明

**文件：** `agent/requirements.txt`
**依赖：** 无

**步骤：**

1. 实测 `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver` 是否能 import
2. 若不能，添加 `langgraph-checkpoint-sqlite>=2.0.0`；若能则跳过

**验证：** `pip install -r requirements.txt` 成功，import 不报错

---

## 执行顺序

```
T1 ────────────────────────────────────────┐
T2 ────────────────────────────────────────┤
                                            │
         ┌──────────────────────────────────┘
         ▼
       T3.1 → T3.2 → T3.3
         │               │
         ▼               │
       T3.4              │
         │               │
         └───────┬───────┘
                 ▼
                T4 → T5
```

> T1 和 T2 可并行；T3.4 与 T3.2/T3.3 可并行；T5 可在任意时间完成。