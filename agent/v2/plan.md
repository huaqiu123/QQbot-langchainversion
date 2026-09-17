# Agent V2 — RAG 知识库增强 Plan

## 架构概览

V2 在 V1 基础上新增两个模块、修改四个现有模块：

```
                    ┌─────────────────────────────┐
                    │        main.py              │
                    │  ┌───────────────────────┐  │
                    │  │  /knowledge/ingest    │  │
                    │  │  /knowledge/status    │  │
                    │  │  /v1/agent/ask (改造) │  │
                    │  │  /v1/agent/stream(改造)│  │
                    │  └──────────┬────────────┘  │
                    └─────────────┼────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
  │   knowledge.py  │   │    agent.py     │   │  database.py    │
  │  【新增】       │   │  【改造】       │   │  【新增】       │
  │                 │   │                 │   │                 │
  │ - add_texts()   │   │ _build_messages │   │ - insert()      │
  │ - search()      │──▶│ 中自动检索知识库 │   │ - get_by_id()   │
  │ - get_chunk_cnt │   │ 并注入 prompt   │   │ - count()       │
  └────────┬────────┘   └─────────────────┘   └────────┬────────┘
           │                                           │
           ▼                                           ▼
  ┌─────────────────┐                          ┌─────────────────┐
  │   llm.py        │                          │    SQLite      │
  │  【改造】       │                          │   (文件存储)    │
  │ build_embeddings│                          │  data/messages  │
  └────────┬────────┘                          └─────────────────┘
           │
           ▼
  ┌─────────────────┐
  │  SiliconFlow    │
  │  /v1/embeddings │
  └─────────────────┘
```

### 核心流程变化

```
用户提问
    │
    ▼
① Agent._build_messages()
    ├── knowledge_base.search(question)  ← 新增：自动检索知识库
    ├── 命中且高于阈值 → 注入 RAG 上下文到 system prompt
    └── 未命中或低于阈值 → 不注入，走纯 V1 流程
    │
    ▼
② graph.ainvoke({messages})
    ├── 知识库有内容 → 模型优先基于知识库回答
    └── 知识库不足 → 模型自主调用 web_search 联网补充
    │
    ▼
③ 返回结果
```

## 核心数据结构

### 新增 Pydantic 模型（`schemas.py`）

```python
class KnowledgeIngestRequest(BaseModel):
    """写入知识库的请求体。"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeIngestResponse(BaseModel):
    """写入成功响应。"""
    chunk_count: int
    total_chunks: int


class KnowledgeStatusResponse(BaseModel):
    """知识库状态响应。"""
    available: bool
    chunk_count: int
    source_count: int


class KnowledgeSearchResult(BaseModel):
    """单条检索结果（仅内部使用，不序列化为 API 响应）。"""
    content: str
    source_id: int
    similarity: float
    metadata: Dict[str, Any]
```

## 核心接口定义

### 新增接口

#### `POST /knowledge/ingest`

写入文本到知识库。

**请求头**：

| 头             | 必填 | 说明                          |
| -------------- | ---- | ----------------------------- |
| `Content-Type` | 是   | `application/json`            |
| `X-API-Key`    | 条件 | 配置了 `AGENT_API_KEY` 时必须 |

**请求体**（`KnowledgeIngestRequest`）：

```json
{
    "content": "DeepSeek-V3 于 2024 年 12 月发布，是一款强大的 MoE 模型。",
    "metadata": {
        "group_id": "123456789",
        "sender_id": "user_001",
        "msg_id": "abc123"
    }
}
```

| 字段       | 类型   | 必填 | 说明                               |
| ---------- | ------ | ---- | ---------------------------------- |
| `content`  | string | 是   | 要写入的文本内容（`min_length=1`） |
| `metadata` | object | 否   | 来源元数据，默认 `{}`              |

**成功响应 `200`**（`KnowledgeIngestResponse`）：

```json
{
    "chunk_count": 3,
    "total_chunks": 42
}
```

| 字段           | 类型 | 说明                     |
| -------------- | ---- | ------------------------ |
| `chunk_count`  | int  | 本次写入生成的文档块数量 |
| `total_chunks` | int  | 写入后知识库总块数       |

**错误响应**：

| 状态码 | 场景                                     | body                                                   |
| ------ | ---------------------------------------- | ------------------------------------------------------ |
| `401`  | 鉴权失败（配置了 Key 但未传或传错）      | `{"detail": "Invalid or missing API Key"}`             |
| `422`  | 请求体验证失败（如 `content` 为空）      | Pydantic 默认校验错误                                  |
| `503`  | 知识库未配置（`EMBEDDING_API_KEY` 未设） | `{"detail": "知识库未配置：EMBEDDING_API_KEY 未设置"}` |

---

#### `GET /knowledge/status`

查询知识库当前状态。

**请求头**：无需鉴权（与 `/health` 一致，不要求 `X-API-Key`）

**成功响应 `200`**（`KnowledgeStatusResponse`）：

```json
{
    "available": true,
    "chunk_count": 42,
    "source_count": 15
}
```

| 字段           | 类型 | 说明                                        |
| -------------- | ---- | ------------------------------------------- |
| `available`    | bool | 知识库是否可用（即 Embedding Key 是否配置） |
| `chunk_count`  | int  | Chroma 中文档块总数                         |
| `source_count` | int  | SQLite 中原文片段总数                       |

未配置 Embedding Key 时的响应：

```json
{
    "available": false,
    "chunk_count": 0,
    "source_count": 0
}
```

---

### 改造接口（V1 升级）

#### `POST /v1/agent/ask`

**请求体新增字段**：无。V2 不新增请求字段，知识库检索为自动行为。

**行为变化**：
- 自动检索知识库，将命中结果注入 system prompt
- 响应格式与 V1 完全一致（`AgentAskResponse`）

**向后兼容**：
- 知识库为空/未配置 → 行为与 V1 完全一致

#### `POST /v1/agent/stream`

- 行为变化同上（自动检索并注入）
- 流式响应格式不变

#### `GET /health`

**变更**：响应中新增 `knowledge` 字段反映知识库状态。

**V2 响应体**：

```json
{
    "status": "ok",
    "version": "2.0.0",
    "knowledge": {
        "available": true,
        "chunk_count": 42,
        "source_count": 15
    }
}
```

知识库未配置时：

```json
{
    "status": "ok",
    "version": "2.0.0",
    "knowledge": {
        "available": false,
        "chunk_count": 0,
        "source_count": 0
    }
}
```

注意：知识库不可用时 `status` 仍为 `"ok"`，满足 N1 带病启动。

---

### 未变更接口

| 路径 | 方法 | 说明         |
| ---- | ---- | ------------ |
| `/`  | GET  | 欢迎页，不变 |

---

## 模块设计

### 1. `knowledge.py`（新增）

```python
class KnowledgeBase:
    """知识库，封装 Chroma 向量检索。"""

    def __init__(self, settings: Settings, embedding_func):
        self.settings = settings
        self.chroma_path = settings.chroma_persist_dir
        self.collection_name = "qqbot_knowledge"
        self.threshold = settings.knowledge_similarity_threshold
        self.embedding_func = embedding_func
        self._collection: Collection | None = None

    def _get_collection(self) -> Collection:
        """延迟加载 Chroma 集合并持久化。"""
        ...

    def add_texts(self, texts: list[str],
                  metadatas: list[dict] | None = None) -> int:
        """写入文本块并返回写入的块数。"""
        ...

    def search(self, query: str, top_k: int = 5
               ) -> list[KnowledgeSearchResult]:
        """检索与 query 最相关的结果（高于阈值才返回）。"""
        ...

    def get_chunk_count(self) -> int:
        """获取 Chroma 中文档块总数。"""
        ...

    @property
    def available(self) -> bool:
        """知识库是否可用（即 embedding_func 是否配置）。"""
        return self.embedding_func is not None
```

**关键设计**：
- 延迟加载：`_get_collection()` 在首次使用时才初始化 Chroma
- 阈值过滤：`search()` 返回前过滤掉低于 `knowledge_similarity_threshold` 的结果
- 没有 `delete`/`update` 方法

### 2. `database.py`（新增）

```python
class MessageStore:
    """SQLite 存储消息原文与元数据。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """建表（如不存在）。"""
        ...

    def insert(self, content: str, metadata: dict | None = None) -> int:
        """插入一条原文记录，返回自增 ID。"""
        ...

    def get_by_id(self, source_id: int) -> str | None:
        """根据 ID 查询原文（供溯源使用）。"""
        ...

    def count(self) -> int:
        """获取总记录数。"""
        ...
```

**Schema**：
```sql
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT    NOT NULL,
    metadata    TEXT,               -- JSON 字符串
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
```

**Chroma vs SQLite 分工**：

| 维度     | Chroma                  | SQLite           |
| -------- | ----------------------- | ---------------- |
| 存储内容 | 向量 + 文本块（分块后） | 原始完整文本     |
| 核心用途 | 相似度检索              | 原文溯源 + 计数  |
| 检索方式 | 向量相似度              | 根据 ID 精确查询 |
| 关联关系 | metadata 中存 source_id | 自增 ID 作为主键 |

### 3. `llm.py`（改造）

新增 Embedding 客户端构造函数：

```python
def build_embeddings(settings: Settings) -> Embeddings | None:
    """构造 Embeddings 客户端。

    未配置 EMBEDDING_API_KEY 时返回 None，外层 knowledge.py
    据此将 available 置为 False。
    """
    if not settings.embedding_api_key:
        return None

    return SiliconFlowEmbeddings(
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
    )
```

### 4. `config.py`（改造）

新增 8 个配置项：

```python
# 知识库配置
EMBEDDING_API_KEY: str = Field("", description="SiliconFlow Embedding API Key")
EMBEDDING_MODEL: str = Field("BAAI/bge-m3", description="Embedding 模型名")
EMBEDDING_BASE_URL: str = Field(
    "https://api.siliconflow.cn/v1", description="Embedding API 地址"
)
CHROMA_PERSIST_DIR: str = Field("data/chroma", description="Chroma 持久化目录")
KNOWLEDGE_SIMILARITY_THRESHOLD: float = Field(
    0.7, description="知识库检索相似度阈值"
)
KNOWLEDGE_SEARCH_TOP_K: int = Field(5, description="检索返回的最大结果数")
KNOWLEDGE_CHUNK_SIZE: int = Field(256, description="文本分块大小（字符数）")
KNOWLEDGE_CHUNK_OVERLAP: int = Field(32, description="文本分块重叠字符数")
```

### 5. `prompts.py`（改造）

V1 已有静态 `AGENT_SYSTEM_PROMPT`。V2 新增运行时包装函数：

```python
AGENT_SYSTEM_PROMPT = """你是一个智能助手..."""  # V1 不变

def build_system_prompt(knowledge_context: str | None = None) -> str:
    """根据是否有知识库检索结果，构造最终的 system prompt。

    - 有知识库内容：注入上下文，引导模型优先基于知识库回答
    - 无知识库内容：退回纯 V1 prompt
    """
    if knowledge_context:
        return f"""{AGENT_SYSTEM_PROMPT}

【知识库参考信息】
以下是从知识库中检索到的相关内容，请**优先参考**这些信息回答问题。
如果参考信息足以回答，直接给出答案；如果不完整或不确定，请使用
web_search 工具联网搜索补充。

{knowledge_context}
"""
    return AGENT_SYSTEM_PROMPT
```

### 6. `agent.py`（改造）

核心改动：`_build_messages` 中新增知识库检索注入。

```python
class Agent:
    def __init__(self, ..., knowledge_base: KnowledgeBase | None = None):
        ...
        # V2 新增
        self.knowledge_base = knowledge_base

    def _build_messages(self, question: str) -> list[BaseMessage]:
        """构造消息列表。V2 改动：自动检索知识库并注入 system prompt。"""
        # V2 新增：检索知识库
        knowledge_context = None
        if self.knowledge_base and self.knowledge_base.available:
            results = self.knowledge_base.search(question)
            if results:
                knowledge_context = "\n\n".join(
                    f"[来源 {i+1}] {r.content}"
                    for i, r in enumerate(results)
                )

        # V2 改动：动态构造 system prompt
        system_prompt = build_system_prompt(knowledge_context)

        # V1 原有逻辑不变
        messages = [SystemMessage(content=system_prompt)]
        if self.chat_history and len(self.chat_history) > 0:
            ...
        messages.append(HumanMessage(content=question))
        return messages
```

### 7. `main.py`（改造）

新增路由注册 + 知识库初始化。

```python
# V2 新增：知识库初始化
knowledge_base = None
message_store = None
if embedding_func := build_embeddings(settings):
    message_store = MessageStore("data/messages.db")
    knowledge_base = KnowledgeBase(settings, embedding_func)

# Agent 初始化（V2 改动：传入 knowledge_base）
agent = Agent(
    settings=settings,
    tools=tools,
    chat_history=[],
    knowledge_base=knowledge_base,
)

# V2 新增：知识库路由
@app.post("/knowledge/ingest")
async def ingest_knowledge(
    req: KnowledgeIngestRequest,
    request: Request,
):
    if not knowledge_base or not knowledge_base.available:
        raise HTTPException(status_code=503, detail="知识库未配置或不可用")
    source_id = message_store.insert(req.content, req.metadata)
    chunks = split_text(req.content)
    metadatas = [
        {"source_id": source_id, "chunk_index": i, **req.metadata}
        for i in range(len(chunks))
    ]
    chunk_count = knowledge_base.add_texts(chunks, metadatas)
    return KnowledgeIngestResponse(
        chunk_count=chunk_count,
        total_chunks=knowledge_base.get_chunk_count(),
    )

@app.get("/knowledge/status")
async def knowledge_status():
    if not knowledge_base:
        return KnowledgeStatusResponse(available=False, chunk_count=0, source_count=0)
    return KnowledgeStatusResponse(
        available=knowledge_base.available,
        chunk_count=knowledge_base.get_chunk_count(),
        source_count=message_store.count(),
    )
```

## 数据流

### 数据流 1：知识库写入

```
NapCat / 调用方
    │  POST /knowledge/ingest
    │  {"content": "...", "metadata": {...}}
    ▼
main.py (鉴权通过)
    ├─ (1) message_store.insert(content, metadata)  → SQLite, 返回 source_id
    ├─ (2) split_text(content)                       → ["块1", "块2", ...]
    ├─ (3) build_embeddings().embed_documents(blocks) → 向量列表
    └─ (4) knowledge_base.add_texts(blocks, metadatas) → Chroma
    ▼
返回 {chunk_count, total_chunks}
```

### 数据流 2：问答（V2 流程）

```
用户 → POST /v1/agent/ask
       → agent._build_messages(question)
           ├─ (1) knowledge_base.search(question)
           │       Chroma.query → 过滤 < 0.7 → 剩余结果
           ├─ (2) 有结果? → build_system_prompt(knowledge_context)
           │       无结果? → build_system_prompt(None) → 纯 V1
           └─ (3) messages = [SystemMessage(system_prompt), HumanMessage(question)]
       → graph.ainvoke({messages})
           ├─ 知识库足够 → 直接基于知识库回答
           └─ 知识库不足 → 模型自主调 web_search 联网补充
       → 返回 {answer, sources}
```

### 数据流 3：未配置 Embedding Key

```
用户 → POST /v1/agent/ask
       → knowledge_base is None → 跳过检索
       → build_system_prompt(None) = AGENT_SYSTEM_PROMPT
       → graph.ainvoke({messages})  ← 纯 V1 联网搜索
```

## 配置变更

| 环境变量                         | 默认值                            | 说明                          |
| -------------------------------- | --------------------------------- | ----------------------------- |
| `EMBEDDING_API_KEY`              | `""`（空）                        | SiliconFlow Embedding API Key |
| `EMBEDDING_MODEL`                | `"BAAI/bge-m3"`                   | Embedding 模型名              |
| `EMBEDDING_BASE_URL`             | `"https://api.siliconflow.cn/v1"` | Embedding API 基础地址        |
| `CHROMA_PERSIST_DIR`             | `"data/chroma"`                   | Chroma 持久化目录             |
| `KNOWLEDGE_SIMILARITY_THRESHOLD` | `0.7`                             | 相似度阈值                    |
| `KNOWLEDGE_SEARCH_TOP_K`         | `5`                               | 检索返回最大结果数            |
| `KNOWLEDGE_CHUNK_SIZE`           | `256`                             | 文本分块大小（字符数）        |
| `KNOWLEDGE_CHUNK_OVERLAP`        | `32`                              | 文本分块重叠字符数            |

**向后兼容**：旧 `.env` 无任何 Embedding 配置 → 知识库不可用，Agent V1 正常运行。

## 安全与边界处理

| 编号 | 场景                   | 处理方式                                |
| ---- | ---------------------- | --------------------------------------- |
| SEC1 | Embedding Key 未配置   | `ingest` 返回 503，Agent V1 正常运行    |
| SEC2 | 写入空内容             | Pydantic 校验 `min_length=1`，返回 422  |
| SEC3 | 写入超长内容           | 自动分块，每块上限由 `CHUNK_SIZE` 控制  |
| SEC4 | 相似度低于阈值         | 结果**完全不注入** prompt，退回联网搜索 |
| SEC5 | Chroma/SQLite 写入异常 | 先写 SQLite，Chroma 失败时回滚 SQLite   |
| SEC6 | 并发写入               | SQLite WAL 模式，暂不加锁（写入频率低） |
| SEC7 | 检索结果为空           | 退回纯 V1 prompt，无额外开销            |
| SEC8 | Embedding API 调用失败 | 返回 502，不污染知识库状态              |

## 关键设计决策

| 编号 | 决策             | 选型                            | 理由                                                       |
| ---- | ---------------- | ------------------------------- | ---------------------------------------------------------- |
| DEC1 | 知识库集成方式   | **Prompt 注入** 非 Tool calling | 检索是每轮必做，减少一次模型推理；知识库内容对全部工具透明 |
| DEC2 | 存储方案         | **Chroma + SQLite**             | Chroma 做向量检索，SQLite 做原文溯源和计数统计             |
| DEC3 | 分块时机         | **写入时分块**                  | 减少 Embedding API 调用次数                                |
| DEC4 | KB 注入方式      | **依赖注入** 非 Agent 工具      | 保持 tools 语义不变，知识库检索对模型透明                  |
| DEC5 | Embedding 初始化 | **延迟初始化**                  | 满足 N1 带病启动，Key 未配置时不影响 V1                    |
| DEC6 | 编辑/删除        | **本轮不做**                    | 当前场景只追加不修改                                       |

## 依赖变更

### 新增

| 包                 | 用途                    | 版本约束  |
| ------------------ | ----------------------- | --------- |
| `chromadb`         | Chroma 向量数据库客户端 | `>=1.5.0` |
| `langchain-chroma` | LangChain Chroma 封装   | `>=0.1.0` |

### 无变更

- `langchain-community`（V1 已引入，含 SiliconFlowEmbeddings）
- `langchain-core`、`langgraph`、`httpx`、`pydantic`、`duckduckgo_search`

## 任务拆分

| ID  | 任务                                                   | 耗时  | 依赖       |
| --- | ------------------------------------------------------ | ----- | ---------- |
| T1  | 新增依赖安装                                           | 10min | -          |
| T2  | `config.py` — 新增知识库配置项                         | 10min | -          |
| T3  | `llm.py` — 新增 `build_embeddings()` 工厂函数          | 15min | T2         |
| T4  | `database.py` — SQLite 消息存储模块                    | 20min | T2         |
| T5  | `knowledge.py` — Chroma 知识库模块                     | 40min | T2, T3     |
| T6  | `prompts.py` — 新增 `build_system_prompt()` 运行时构造 | 15min | -          |
| T7  | `agent.py` — 集成知识库检索到 `_build_messages()`      | 30min | T5, T6     |
| T8  | `schemas.py` — 新增知识库请求/响应模型                 | 15min | -          |
| T9  | `main.py` — 新增知识库初始化 + 路由注册 + 鉴权         | 30min | T4, T5, T8 |
| T10 | 实现文本分块函数 `split_text()`                        | 15min | -          |
| T11 | 知识库写入/查询单元测试                                | 20min | T4, T5     |
| T12 | Agent RAG 集成测试                                     | 25min | T7, T9     |
| T13 | 降级兼容性测试                                         | 15min | T9         |
| T14 | `.env.example` 补充新配置                              | 5min  | T2         |

### 关键路径

```
T1 → T2 → T3 → T5 → T7 → T9 → T12
                    ↓
                T6 ↗      T11
```

T2 完成后 T3、T4、T8 可并行。T5 完成后 T6、T10 不影响主流程，可并行。

**总计预估工时**：~7 人时

## 执行策略

### 推荐执行顺序

1. **T1**（安装依赖）→ **T2**（配置项）→ **T8**（schemas，独立，可并行）
2. **T3**（Embedding 工厂）→ **T5**（KnowledgeBase 核心）
3. **T4**（SQLite）→ **T10**（分块函数，短任务）
4. **T6**（prompt 构造，独立）→ **T7**（Agent 集成）
5. **T9**（main.py 路由注册）
6. **T14**（.env.example）
7. **T11 → T12 → T13**（测试验证）