# Agent V3：持久化记忆管理改造 Checklist

> 每一项通过运行代码或观察行为来验证，聚焦系统行为。

## 实现完整性

- [ ] **config.py** — `Settings.checkpoint_db_path` 字段已添加，默认值为 `"data/checkpoint.db"`（验证：`grep checkpoint_db_path agent/app/config.py` 有输出）
- [ ] **schemas.py** — `AskRequest` 中 `user_id` 为必填字段，`history` 和 `session_id` 已删除（验证：`grep user_id agent/app/schemas.py` 有 `min_length=1`；`grep history agent/app/schemas.py` 无 `AskRequest` 附近的匹配）
- [ ] **agent.py** — `Agent.__init__` 接受 `checkpointer` 参数（验证：`grep "checkpointer" agent/app/agent.py` 匹配到 `__init__` 签名）
- [ ] **agent.py** — `_build_messages` 已重命名为 `_build_input`，不再接收 `history` 参数（验证：`grep "_build_messages" agent/app/agent.py` 无匹配；`grep "_build_input"` 有匹配）
- [ ] **agent.py** — `run()` 和 `stream()` 签名中 `history` 已替换为 `user_id`（验证：`grep "def run\|def stream" agent/app/agent.py` 签名含 `user_id` 不含 `history`）
- [ ] **agent.py** — 消息裁剪中间件 `_trim_messages` 已实现并在 LLM 调用前挂载（验证：`grep "_trim_messages\|trim_messages" agent/app/agent.py` 有匹配）
- [ ] **main.py** — lifespan 中创建了 `AsyncSqliteSaver` 并调用 `setup()`（验证：`grep "AsyncSqliteSaver\|checkpointer.setup" agent/app/main.py` 有匹配）
- [ ] **main.py** — ask/stream 路由传 `user_id=payload.user_id` 而非 `history`（验证：`grep "user_id" agent/app/main.py` 在路由函数中有引用）

## 集成

- [ ] lifespan 装配链路完整：`get_settings` → `AsyncSqliteSaver` → `checkpointer.setup()` → `Agent(..., checkpointer=...)`（验证：启动服务，日志无报错，FastAPI 正常监听端口）
- [ ] `config["configurable"]["thread_id"]` 由 `user_id` 直接赋值，未做额外编码（验证：运行一次请求，检查 agent.py 中 config 构造逻辑）
- [ ] `create_agent` 调用时 `checkpointer` 参数已传入（验证：`grep "create_agent" agent/app/agent.py` 所在行含 `checkpointer`）
- [ ] 知识库检索结果通过 `_build_input` 注入到本轮 `HumanMessage`（验证：启用知识库，提问后模型回答引用了知识库内容）
- [ ] `extra_context` 通过 `_build_input` 注入到本轮 `HumanMessage`（验证：传 `extra_context`，模型回答体现了注入上下文）

## 编译与测试

- [ ] 项目无 import 错误（验证：`cd agent && python -c "from app.main import app"` 无异常）
- [ ] `requirements.txt` 依赖完整（验证：`pip install -r agent/requirements.txt` 无 Missing 错误）
- [ ] `/health` 端点正常，`graph_ready=true`，无 `agent_error`（验证：`curl http://localhost:8000/health`）
- [ ] 配置文件 `data/checkpoint.db` 在首次请求后自动生成（验证：`ls -la agent/data/checkpoint.db` 存在）

## 端到端场景

### 场景 E1：多轮对话记忆与隔离（AC3 + AC4）

1. 用户 u1 问"我今天心情很好，记住我开心" → 200，模型回应
2. 用户 u1 问"我今天心情怎么样" → 200，模型回答能关联到"开心"
3. 用户 u2 问"我今天心情怎么样" → 200，模型不知道（不同用户隔离）
4. 用户 u1 再问"然后呢" → 200，模型仍记得 u1 的上下文

### 场景 E2：服务重启后持久化（AC5）

1. 用户 u1 问"我叫张三，请记住" → 200
2. 重启服务（`Ctrl+C` 再启动）
3. 用户 u1 问"我叫什么" → 200，模型回答"张三"

### 场景 E3：消息窗口限制生效（AC10）

1. 设置 `max_history_messages=2`
2. 用户 u1 连续发送 5 条不同话题的消息
3. 观察：模型的行为只看得到最近 2 轮对话，更早的信息已被裁剪

### 场景 E4：流式输出正常（AC8）

1. 发 `POST /v1/agent/stream {"question":"你好","user_id":"u1"}` → SSE 流返回
2. 事件类型包含 `delta`、`done`
3. 如启用搜索，事件类型包含 `search`

### 场景 E5：API 校验（AC1 + AC2）

1. 发 `POST /v1/agent/ask {"question":"hi"}` → 422，错误提示缺少 `user_id`
2. 发 `POST /v1/agent/ask {"question":"hi","user_id":"u1","history":[]}` → 422

### 场景 E6：知识库 + extra_context 注入（AC6 + AC7）

1. 配置知识库，提问后模型回答引用了知识库内容
2. 传 `extra_context`，模型回答体现了注入的上下文

### 场景 E7：Health Check（AC9）

1. 发 `GET /health` → 200，`graph_ready=true`，无 `agent_error`