# Agent V2 — RAG 知识库增强 Checklist

## 实现完整性

- [ ] `chromadb` 和 `langchain-chroma` 已加入依赖并可导入（验证：`python -c "import chromadb; import langchain_chroma"` 不报错）
- [ ] `config.py` 新增 8 个知识库配置项，均有默认值（验证：`python -c "from app.config import get_settings; s=get_settings(); print(s.embedding_api_key)"` 输出空字符串）
- [ ] `llm.py` 提供 `build_embeddings()` 函数，未配置 Key 时返回 None（验证：无 Key 时调用返回 None）
- [ ] `database.py` 实现 `MessageStore` 类，提供 `insert()` / `get_by_id()` / `count()`（验证：单元测试通过）
- [ ] `knowledge.py` 实现 `KnowledgeBase` 类，提供 `add_texts()` / `search()` / `get_chunk_count()`（验证：集成测试通过）
- [ ] `prompts.py` 新增 `build_system_prompt()`，无知识库上下文时退回 V1 原 prompt（验证：`build_system_prompt(None) == SYSTEM_PROMPT`）
- [ ] `agent.py` 的 `Agent.__init__` 接受 `knowledge_base` 参数，默认为 None（验证：`Agent(settings, provider, knowledge_base=None)` 构造成功）
- [ ] `agent.py` 的 `_build_messages()` 自动检索知识库并注入 prompt（验证：注入内容出现在 system prompt 中）
- [ ] `schemas.py` 新增 `KnowledgeIngestRequest` / `KnowledgeIngestResponse` / `KnowledgeStatusResponse`（验证：Pydantic 模型构造正常）
- [ ] `main.py` 的 lifespan 中初始化知识库，失败不影响 Agent 启动（验证：启动后 `/health` 返回 `status: ok`）
- [ ] `main.py` 新增 `split_text()` 分块函数（验证：空文本返回空列表，长文本正确分块）
- [ ] `main.py` 注册 `POST /knowledge/ingest` 和 `GET /knowledge/status` 路由（验证：启动后路由可访问）
- [ ] `main.py` 的 `/health` 响应体含 `knowledge` 字段（验证：`GET /health` 返回体中存在 `knowledge` 键）
- [ ] `.env.example` 补充知识库配置项（验证：文件内容含 `EMBEDDING_API_KEY` 等行）

## 集成

- [ ] Agent 能通过 `knowledge_base` 参数接收 `KnowledgeBase` 实例并使用其检索结果（验证：集成测试中注入知识库后回答包含知识库内容）
- [ ] `MessageStore.insert()` 返回的 ID 能通过 Chroma metadata 中的 `source_id` 关联回溯（验证：Chroma 检索结果项包含 `source_id` 字段）
- [ ] Chroma 写入失败时 SQLite 不留下孤立的无关联记录（验证：异常路径抛出 502，不污染数据）
- [ ] `POST /knowledge/ingest` 配置了 `Depends(require_api_key)` 鉴权（验证：配置 Key 后无 Key 请求返回 401）
- [ ] `GET /knowledge/status` 未配置鉴权，可公开访问（验证：无需 Key 即可调用）
- [ ] 知识库检索异常被 `try/except` 吞掉，不穿透到 LangGraph 执行图（验证：Agent 在知识库异常时仍正常联网回答）

## 编译与测试

- [ ] 依赖安装无冲突（验证：`pip install -r requirements.txt` 成功）
- [ ] MessageStore + split_text 单元测试全部通过（验证：`pytest tests/test_knowledge.py -v` 全部绿色）
- [ ] 知识库为空时 `search()` 返回空列表（验证：集成测试通过）
- [ ] 未配置 Embedding Key 时知识库不可用，但应用不崩溃（验证：兼容性测试通过）
- [ ] `knowledge_base=None` 时 Agent 正常构造（验证：兼容性测试通过）

## 端到端场景

### 场景 1：知识库写入 → 检索 → 回答

**步骤：**

1. 配置 `EMBEDDING_API_KEY`，启动服务
2. 调用 `POST /knowledge/ingest` 写入一条消息：

```json
{"content": "2024 年 12 月 26 日，DeepSeek 发布了 DeepSeek-V3 模型", "metadata": {"group_id": "g1"}}
```

3. 等待响应，确认 `chunk_count > 0`
4. 调用 `GET /knowledge/status`，确认 `chunk_count > 0` 且 `source_count > 0`
5. 调用 `POST /v1/agent/ask` 提问：

```json
{"question": "DeepSeek-V3 是什么时候发布的？"}
```

6. 检查响应：

- `answer` 包含「2024 年 12 月 26 日」或「2024 年 12 月」
- 回答不依赖联网搜索（模型基于知识库上下文作答）

**预期结果：** Agent 从知识库获取信息并正确回答

### 场景 2：知识库不足时自动联网搜索

**步骤：**

1. 保留已有知识库，调用 `POST /v1/agent/ask` 提问时效性问题：

```json
{"question": "今天有什么科技新闻？"}
```

2. 检查响应：

- `answer` 包含联网搜索结果
- `sources` 列表不为空（含引用来源）

**预期结果：** Agent 自动调用 `web_search` 联网补充

### 场景 3：未配置知识库时退化到 V1

**步骤：**

1. 清空 `EMBEDDING_API_KEY`，重启服务
2. 调用 `GET /health`，确认 `status: ok` 且 `knowledge.available: false`
3. 调用 `POST /knowledge/ingest`，确认返回 503
4. 调用 `POST /v1/agent/ask` 提问常规问题，确认正常联网回答

**预期结果：** 知识库不可用不影响 V1 核心功能

### 场景 4：鉴权保护

**步骤：**

1. 配置 `AGENT_API_KEY=test-key` 和 `EMBEDDING_API_KEY=sk-xxx`，重启服务
2. 不带 `X-API-Key` 头调用 `POST /knowledge/ingest`，确认返回 401
3. 带错误 Key 调用 `POST /knowledge/ingest`，确认返回 401
4. 带正确 Key 调用 `POST /knowledge/ingest`，确认返回 200

**预期结果：** 知识库写入接口受鉴权保护，与 V1 机制一致