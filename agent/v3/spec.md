# Agent V3：持久化记忆管理改造 Spec

## 背景

当前项目 (`agent/app/`) 使用 LangChain `create_agent` 构建 Agent，但记忆管理方式是 **"调用方自行维护 `history` 列表"**：每次请求时，调用方（QQ 机器人）必须把本轮之前所有的 `user`/`assistant` 消息打包进 `history` 字段回传。这导致：

- 调用方负担重（需要自己存对话、拼历史）
- 服务重启后历史全部丢失
- 历史数据散落在调用方，不在服务端统一管理

LangGraph 提供了内置的 **checkpointer** 机制，可以用 SQLite 持久化每个会话的完整消息历史，`create_agent` 原生支持。切换后调用方只需传一个 `user_id`，其余全由框架自动处理。

## 目标

- 用 LangGraph SQLite checkpointer **替代**手动 `history` 管理
- API 层面：`session_id` 改名为 `user_id`，改为**必填**，删除 `history` 字段
- 每个 `user_id` 对应一个独立会话，消息持久化到 SQLite，重启不丢失
- Checkpointer 存全量消息，LLM 调用时通过中间件只保留最近 N 条（受 `max_history_messages` 控制）
- 现有功能不受影响：联网搜索、知识库 RAG、流式输出、`extra_context` 注入

## 功能需求

- **F1: SQLite Checkpointer 集成**：Agent 图编译时注入 `AsyncSqliteSaver`，使 LangGraph 自动按 `user_id` 持久化和恢复会话状态。启动时自动建表，使用 WAL 模式。

- **F2: API 字段变更**：`AskRequest` 中新增 `user_id: str`（必填），删除 `history` 和 `session_id` 字段。`question`、`use_search`、`extra_context` 保持不变。

- **F3: Agent 执行适配**：`Agent.run()` 和 `Agent.stream()` 不再接收 `history` 参数，改为接收 `user_id`。内部将 `user_id` 映射为 `config["configurable"]["thread_id"]`，历史由 checkpointer 自动加载和保存。

- **F4: 消息窗口裁剪**：Checkpointer 存储全量消息，但每次调用 LLM 前通过中间件只保留最近 `max_history_messages` 条（`<= 0` 时不限制）。

- **F5: 向后不兼容**：旧调用方如仍传递 `history` 字段，FastAPI 返回 422。不提供兼容模式。

- **F6: `_build_messages` 精简**：移除历史消息拼接逻辑，只保留知识库注入、extra_context 注入和当前消息构造。

- **F7: 配置新增**：`Settings` 中新增 `checkpoint_db_path: str = "data/checkpoint.db"`。

## 非功能需求

- **N1: 性能**：Checkpointer 读写不应成为瓶颈，SQLite + WAL 模式足够支撑 QQ 机器人的并发量。图编译次数不变（仍是启动时一次）。

- **N2: 数据持久性**：checkpoint 数据存储在 `data/checkpoint.db`，服务重启后历史不丢失。数据库文件路径可通过配置修改。

- **N3: 兼容性**：`max_history_messages` 语义保持不变（`> 0` 限制条数，`<= 0` 不限制）。现有的 `recursion_limit`、`request_timeout` 等配置不受影响。现有功能（搜索、知识库 RAG、流式输出、health check）行为不变。

- **N4: 可观测性**：Checkpointer 初始化状态在 `/health` 中体现。旧的 `history` 字段被删除后，调用方传了会得到明确的 422 错误。

## 不做的事

- 不保留 `history` 兼容模式（旧字段直接删除）
- 不提供对话历史查询 API（留给后续）
- 不迁移旧数据（不提供旧 `history` 到 checkpointer 的迁移工具）
- 不做 checkpointer 读写加锁/并发控制（依赖 SQLite WAL 模式自带能力）
- 不改变 `/v1/agent/ask` 和 `/v1/agent/stream` 的 URL

## 验收标准

| 编号 | 验收项 | 验证方式 |
|------|--------|---------|
| AC1 | `AskRequest` 中 `user_id` 为必填，不传返回 422 | 发送缺少 `user_id` 的请求，观察响应 |
| AC2 | `history` 字段已删除，传了返回 422 | 发送携带 `history` 的请求，观察响应 |
| AC3 | 同一 `user_id` 连续两次请求，第二次能记住第一次的上下文 | 第一次问"我叫老王"，第二次问"我叫什么"，模型回答含"老王" |
| AC4 | 不同 `user_id` 之间对话历史隔离 | user_A 和 user_B 各问不同内容，互不干扰 |
| AC5 | 服务重启后历史不丢失 | 发请求 → 重启服务 → 再发请求，模型仍记得之前的内容 |
| AC6 | 知识库注入仍生效 | 配置知识库后提问，模型回答引用了知识库内容 |
| AC7 | `extra_context` 注入仍生效 | 传 `extra_context`，模型回答体现了注入内容 |
| AC8 | 流式输出正常 | `/v1/agent/stream` 正常返回 SSE 事件流，包含 `delta`/`search`/`done` |
| AC9 | `/health` 返回正常 | `graph_ready=true`，无报错 |
| AC10 | `max_history_messages` 限制生效 | 设置限制为 2，发 5 轮对话后检查 LLM 实际看到的上下文不超过限制 |