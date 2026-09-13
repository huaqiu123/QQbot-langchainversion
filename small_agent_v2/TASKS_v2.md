# Small Agent V2 — RAG 知识库增强计划

> **基础**：基于 V1（FastAPI + LangChain `create_agent` + DeepSeek + 可插拔搜索源）进行增量改造。
> **核心目标**：在 V1 的即时搜索能力之上，增加**持久化知识库**，实现「先查库、再搜网」的问答流程。
> **状态**：规划中

---

## 一、目标与范围

### 1.1 V2 目标

在 V1 基础上增加 **RAG（Retrieval-Augmented Generation）** 能力，使 Agent 能够：

1. **接收消息并持久化**：提供一个接口，调用方可以往知识库中写入消息（如群聊记录、文档片段等）；
2. **先查库再回答**：用户提问时，先从知识库中检索相关历史消息/文档；
3. **命中则回答**：检索到足够相关的信息时，基于知识库内容生成回答；
4. **未命中则搜索网络**：知识库中未找到相关信息时，回退到 V1 的联网搜索能力；
5. **知识库可增量更新**：新的消息持续写入，无需重建索引。

### 1.2 V2 新增特性总览

```
┌─────────────────────────────────────────────────────────┐
│                    V2 Agent 工作流程                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户提问 ──→ ① 向量检索知识库 ──→ 命中? ──→ ② 基于知识库回答 │
│                          │               │               │
│                      未命中               │               │
│                          │               │               │
│                          ▼               │               │
│                    ③ 联网搜索(V1) ───────┘               │
│                                                         │
│  消息写入 ──→ ④ 分块 + 向量化 ──→ 存入向量数据库          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.3 与 V1 的关系

|      维度      | V1                                   | V2                                                   |
| :------------: | :----------------------------------- | :--------------------------------------------------- |
|    联网搜索    | ✅ 核心能力                           | ✅ **保留**（作为知识库未命中时的回退）               |
|     知识库     | ❌ 无                                 | ✅ **新增**                                           |
|   消息持久化   | ❌ 无                                 | ✅ **新增**                                           |
|    接口数量    | 3 个（`/ask`, `/stream`, `/health`） | 5 个（新增 `/knowledge/ingest`, `/knowledge/query`） |
| Agent 决策逻辑 | 模型自主判断是否搜索                 | **新增 RAG 路由**：先查库 → 未命中再搜索             |
|    架构依赖    | 无数据库                             | **新增向量数据库**（如 Chroma / FAISS）              |

### 1.4 明确不做（本轮）

- 不做文件上传/PDF 解析（消息以纯文本形式写入）
- 不做知识库的自动过期/清理（由调用方控制写入内容）
- 不做多租户/权限隔离
- 不做 Embedding 模型微调（使用通用 Embedding API）
- 不做知识库的手动编辑/删除接口（仅支持追加写入）

---

## 二、技术选型与关键决策

### 2.1 新增技术选型

| 决策点             | 选择                                               | 理由                                                                                                                         |
| ------------------ | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **向量数据库**     | **Chroma**（本地文件存储）                         | 轻量级、零运维、无需独立部署服务；数据存储在本地文件，适合 QQ 机器人这种中小规模场景；Python 原生 API，与 LangChain 集成最好 |
| **Embedding 模型** | **DeepSeek Embeddings** 或 **`text2vec` 本地模型** | 优先使用 DeepSeek 的 Embedding API（统一厂商，减少依赖）；备选本地 `text2vec-large-chinese`（离线可用，中文效果好）          |
| **文本分块**       | LangChain `RecursiveCharacterTextSplitter`         | 按中文标点（。！？）和换行符递归分割，兼顾语义完整性与块大小                                                                 |
| **向量检索方式**   | 相似度阈值 + Top-K                                 | 设置相似度阈值（如 `0.75`），高于阈值认为命中；返回 Top-K 条（默认 3 条）                                                    |
| **消息存储**       | 同时存入向量库 + SQLite                            | 向量库存语义向量用于检索，SQLite 存原始消息用于溯源和展示                                                                    |
| **RAG 集成方式**   | **改造 Agent 的 ReAct 循环**                       | 在 prompt 中注入检索到的知识库上下文，让模型基于知识库内容回答；或新增一条「知识库工具」让 Agent 自主决定是否调用            |

### 2.2 文本分块策略

```
原始消息
  "DeepSeek-V3 是一款高性能大语言模型，于 2024 年 12 月发布。
   它支持工具调用和多轮对话。"
         │
         ▼  RecursiveCharacterTextSplitter
         │
  ┌──────┴──────┐
  │ Chunk 1     │  "DeepSeek-V3 是一款高性能大语言模型，于 2024 年 12 月发布。"
  │ Chunk 2     │  "它支持工具调用和多轮对话。"
  └─────────────┘
         │
         ▼  Embedding 模型
         │
  ┌──────┴──────┐
  │ Vector 1    │  [0.023, -0.145, 0.678, ...]
  │ Vector 2    │  [-0.101, 0.234, 0.512, ...]
  └─────────────┘
         │
         ▼  存入 Chroma
```

### 2.3 RAG 工作流程（详细）

```
用户提问 "DeepSeek 什么时候发布的？"
     │
     ▼
① 向量化问题 → Chroma 检索
     │
     ├── 命中（相似度 ≥ 0.75）────→ ② 组装 Prompt：
     │                              「基于以下知识库内容回答：
     │                               [1] DeepSeek-V3 于 2024 年 12 月发布...
     │                               问题：DeepSeek 什么时候发布的？」
     │                              │
     │                              ▼
     │                          ③ 调用 DeepSeek 生成回答 ──→ 返回给用户
     │
     └── 未命中 ──→ ④ 走 V1 联网搜索流程 ──→ 返回给用户
```

### 2.4 目录结构（规划）

```
small_agent_v2/               # V2 项目根目录
│
├── small_agent/              # V1 完整代码（直接复制，增量修改）
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # 新增路由
│   │   ├── config.py         # 新增 Embedding/Chroma 配置
│   │   ├── schemas.py        # 新增请求/响应模型
│   │   ├── llm.py            # 新增 build_embeddings()
│   │   ├── search.py         # 不变
│   │   ├── tools.py          # 不变
│   │   ├── prompts.py        # 修改：增加 RAG 上下文注入
│   │   ├── agent.py          # 修改：增加 RAG 检索步骤
│   │   ├── knowledge.py      # 【新增】知识库管理（分块 + 向量化 + 检索）
│   │   └── database.py       # 【新增】SQLite 消息存储
│   ├── knowledge_base/       # Chroma 持久化目录（自动生成）
│   ├── .env
│   ├── .env.example
│   ├── requirements.txt      # 新增 chromadb, sentence-transformers 等
│   ├── README.md
│   └── TASKS.md
│
├── .gitignore
└── TASKS_v2.md               # 本文档
```

---

## 三、模块修改清单

### 3.1 新增模块

| 模块               | 职责                               | 关键类/函数                                                                     |
| ------------------ | ---------------------------------- | ------------------------------------------------------------------------------- |
| `app/knowledge.py` | 知识库核心：文本分块、向量化、检索 | `KnowledgeBase.add_texts()` — 批量写入<br>`KnowledgeBase.search()` — 相似度检索 |
| `app/database.py`  | SQLite 消息持久化                  | `MessageStore.insert()` — 写入消息<br>`MessageStore.query_history()` — 查询历史 |

### 3.2 修改模块

| 模块             | 修改内容                                                                                                                                                    |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/config.py`  | 新增 `embedding_model`、`embedding_dim`、`chroma_persist_dir`、`knowledge_top_k`、`knowledge_min_score` 等配置项；新增 `is_embedding_configured()` 判断     |
| `app/schemas.py` | 新增 `IngestRequest`（写入请求体）、`IngestResponse`（写入响应体）、`KnowledgeQuery`（知识库查询请求）                                                      |
| `app/llm.py`     | 新增 `build_embeddings()` — 构建 Embedding 模型实例                                                                                                         |
| `app/prompts.py` | 修改 `SYSTEM_PROMPT` — 增加「优先使用知识库内容，知识库不足时再搜索网络」的指令；新增 `RAG_CONTEXT_TEMPLATE` — 知识库上下文注入模板                         |
| `app/agent.py`   | 修改 `Agent.__init__()` — 接受 `KnowledgeBase` 参数；修改 `Agent.run()` — 在调用图之前先执行知识库检索；新增 `Agent._rag_retrieve()` — RAG 检索与上下文组装 |
| `app/main.py`    | 新增 `/knowledge/ingest` — 消息写入接口；新增 `/knowledge/query` — 知识库检索接口（可选）；修改 `lifespan` — 初始化 `KnowledgeBase` 和 `MessageStore`       |

---

## 四、任务清单

### T1 环境与依赖
- [ ] 新增依赖：`chromadb`、`sentence-transformers`（或 `langchain-chroma`）
- [ ] 复制 V1 完整代码到 `small_agent_v2/small_agent/`
- [ ] 创建 `knowledge_base/` 目录（Chroma 持久化目录，已配置 `.gitignore`）

### T2 配置层 `app/config.py` — 新增配置项
- [ ] 新增 `embedding_model: str = "text2vec-large-chinese"` — Embedding 模型名
- [ ] 新增 `embedding_dim: int = 1024` — 向量维度
- [ ] 新增 `chroma_persist_dir: str = "knowledge_base"` — Chroma 持久化目录
- [ ] 新增 `knowledge_top_k: int = 3` — 知识库检索返回 Top-K
- [ ] 新增 `knowledge_min_score: float = 0.75` — 相似度阈值，低于此值视为未命中
- [ ] 新增 `chunk_size: int = 512` — 文本分块大小（字符数）
- [ ] 新增 `chunk_overlap: int = 64` — 文本分块重叠
- [ ] 新增 `is_embedding_configured()` — 判断 Embedding 模型是否就绪

### T3 数据模型 `app/schemas.py` — 新增请求/响应模型
- [ ] `IngestRequest`：
  ```jsonc
  {
    "messages": [
      {
        "content": "DeepSeek-V3 于 2024 年 12 月发布",
        "metadata": {
          "source": "group_chat",      // 来源标识
          "group_id": "123456",        // 可选的群 ID
          "sender": "user_abc",        // 可选的发送者
          "timestamp": "2024-12-20T10:00:00Z"  // 可选的时间戳
        }
      }
    ]
  }
  ```
- [ ] `IngestResponse`：
  ```jsonc
  {
    "status": "ok",
    "chunks_count": 12,       // 实际写入的分块数
    "message_count": 3        // 原始消息数
  }
  ```
- [ ] `KnowledgeQueryRequest`（可选，用于调试/管理）：
  ```jsonc
  {
    "query": "DeepSeek 发布时间",
    "top_k": 5
  }
  ```
- [ ] `KnowledgeQueryResponse`：
  ```jsonc
  {
    "results": [
      {
        "content": "DeepSeek-V3 于 2024 年 12 月发布",
        "score": 0.89,
        "metadata": { ... }
      }
    ]
  }
  ```

### T4 知识库模块 `app/knowledge.py` — **核心新增**
- [ ] `KnowledgeBase` 类：
  - `__init__(settings, embed_model)` — 初始化 Chroma 客户端和 Embedding 模型
  - `add_texts(texts: List[str], metadatas: List[Dict]) -> int` — 批量分块、向量化、写入
    - 内部使用 `RecursiveCharacterTextSplitter` 分块
    - 调用 Embedding 模型生成向量
    - 存入 Chroma 集合
    - 返回写入的分块数
  - `search(query: str, top_k: int, min_score: float) -> List[Dict]` — 相似度检索
    - 向量化查询文本
    - Chroma 相似度搜索
    - 按 `min_score` 阈值过滤
    - 返回 `[{content, score, metadata}, ...]`
  - `count() -> int` — 返回知识库中的文档总数（用于 `/health`）
  - `_split_text(text: str) -> List[str]` — 文本分块（使用 `RecursiveCharacterTextSplitter`，分隔符优先中文标点）

### T5 模型构建 `app/llm.py` — 新增 Embedding 构建
- [ ] `build_embeddings(settings) -> Embeddings`：
  - 优先使用 DeepSeek Embeddings API（通过 OpenAI 兼容接口 `/v1/embeddings`）
  - 备选本地 `HuggingFaceEmbeddings(model_name="text2vec-large-chinese")`
  - 提供单例缓存
- [ ] `validate_embeddings(settings)` — 启动期校验 Embedding 模型是否可用

### T6 提示词 `app/prompts.py` — 修改系统提示词
- [ ] 修改 `SYSTEM_PROMPT`，增加 RAG 相关指令：
  - 「你具备知识库检索能力。当用户提问时，优先使用已提供的知识库上下文作答」
  - 「如果知识库内容不足以完整回答，可以补充使用联网搜索获取实时信息」
  - 「标注引用时，知识库来源标注为 `[K1][K2]`，联网搜索来源标注为 `[1][2]`」
- [ ] 新增 `RAG_CONTEXT_TEMPLATE`：
  ```
  [知识库相关条目]
  {chunks}
  
  请优先基于以上知识库内容回答。如果知识库内容不足以回答，请使用联网搜索补充。
  ```

### T7 Agent 核心 `app/agent.py` — 修改问答流程
- [ ] 修改 `Agent.__init__()`：
  - 新增参数 `knowledge_base: Optional[KnowledgeBase] = None`
  - 存储实例变量 `self.knowledge_base`
- [ ] 修改 `Agent.run()`：
  - 在 `_build_messages()` 之前插入 RAG 检索逻辑
  - 如果 `self.knowledge_base` 存在，先执行知识库检索
  - 检索到命中结果 → 注入到消息的 `extra_context`
  - 未命中 → 沿用 V1 逻辑（走联网搜索）
- [ ] 新增 `Agent._rag_retrieve(question: str) -> Optional[str]`：
  - 调用 `self.knowledge_base.search()`
  - 结果按 `min_score` 过滤
  - 命中 → 用 `RAG_CONTEXT_TEMPLATE` 格式化后返回
  - 未命中 → 返回 `None`（触发联网搜索）
- [ ] 修改 `Agent.stream()`（同上，保持流式接口也支持 RAG）

### T8 FastAPI 服务 `app/main.py` — 新增路由 + 修改生命周期
- [ ] 修改 `lifespan`：
  - 初始化 `build_embeddings(settings)`
  - 初始化 `KnowledgeBase(settings, embed_model)`
  - 挂载到 `app.state.knowledge_base`
  - 失败时不阻断服务（带病启动），`/health` 暴露 Embedding 状态
- [ ] 新增 `GET /knowledge/status` — 知识库状态（文档总数、就绪状态）
- [ ] 新增 `POST /knowledge/ingest` — **知识库写入接口**：
  - 接收 `IngestRequest`
  - 调用 `knowledge_base.add_texts()` 写入
  - 返回 `IngestResponse`
- [ ] 新增 `POST /knowledge/query`（可选，调试用）— 知识库检索接口
- [ ] 修改 `/health`：新增 `knowledge_ready`、`knowledge_doc_count` 字段
- [ ] 修改 `/v1/agent/ask` 和 `/v1/agent/stream`：如果启用了知识库，自动执行 RAG 检索

### T9 配置项清单（新增/变更）

| 变量                  | 默认值                   | 说明                     |
| --------------------- | ------------------------ | ------------------------ |
| `EMBEDDING_MODEL`     | `text2vec-large-chinese` | Embedding 模型名         |
| `EMBEDDING_DIM`       | `1024`                   | 向量维度（需与模型匹配） |
| `CHROMA_PERSIST_DIR`  | `knowledge_base`         | Chroma 持久化目录        |
| `KNOWLEDGE_TOP_K`     | `3`                      | 知识库检索返回 Top-K     |
| `KNOWLEDGE_MIN_SCORE` | `0.75`                   | 相似度阈值               |
| `CHUNK_SIZE`          | `512`                    | 文本分块大小（字符）     |
| `CHUNK_OVERLAP`       | `64`                     | 分块重叠（字符）         |

### T10 文档与验证
- [ ] 更新 `README.md`：新增知识库使用说明
- [ ] 测试场景：
  - **知识库命中**：先写入"DeepSeek-V3 于 2024 年 12 月发布"，再问"DeepSeek V3 什么时候发布的" → 应从知识库回答，不带联网引用格式
  - **知识库未命中**：问"今天天气如何" → 走联网搜索
  - **混合场景**：知识库部分命中，联网补充

---

## 五、接口契约（新增）

### 5.1 `POST /knowledge/ingest`

向知识库写入消息。消息会被自动分块、向量化后存入 Chroma。

```jsonc
// 请求
{
  "messages": [
    {
      "content": "DeepSeek-V3 是一款高性能大语言模型，于 2024 年 12 月发布。",
      "metadata": {
        "source": "group_chat",
        "group_id": "123456",
        "timestamp": "2024-12-20T10:00:00Z"
      }
    }
  ]
}
```

```jsonc
// 响应
{
  "status": "ok",
  "chunks_count": 2,
  "message_count": 1
}
```

### 5.2 `GET /knowledge/status`

知识库状态查询。

```jsonc
// 响应
{
  "ready": true,
  "doc_count": 156,
  "embedding_model": "text2vec-large-chinese"
}
```

### 5.3 `POST /v1/agent/ask`（修改）

原有字段不变，新增 RAG 行为：

- 如果启用了知识库（`KNOWLEDGE_MIN_SCORE > 0`），在 **`answer`** 中使用 `[K1][K2]` 标注知识库来源
- **`sources`** 中新增 `source_type` 字段区分「`knowledge`」和「`web`」

```jsonc
// 响应（知识库命中时）
{
  "answer": "DeepSeek-V3 于 2024 年 12 月发布[K1]。",
  "sources": [
    {
      "title": "群聊记录",
      "url": "",
      "snippet": "DeepSeek-V3 是一款高性能大语言模型，于 2024 年 12 月发布。",
      "source_type": "knowledge"     // 新增字段
    }
  ],
  "search_queries": [],
  "iterations": 1,
  "model": "deepseek-chat",
  "usage": { "prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250 }
}
```

---

## 六、数据库设计

### 6.1 Chroma 集合结构

|          字段          |  类型  | 说明                                    |
| :--------------------: | :----: | --------------------------------------- |
|          `id`          |  自动  | Chroma 自动生成的唯一 ID                |
|         `text`         |  文档  | 分块后的文本内容                        |
|      `embedding`       |  向量  | 文本对应的嵌入向量                      |
|   `metadata.source`    | 字符串 | 来源标识（如 `group_chat`, `document`） |
|  `metadata.group_id`   | 字符串 | 群 ID                                   |
|  `metadata.timestamp`  | 字符串 | 原始消息时间戳                          |
| `metadata.original_id` | 字符串 | 对应 SQLite 中的原始消息 ID             |

### 6.2 SQLite 表结构

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,           -- 原始消息内容
    source TEXT DEFAULT '',          -- 来源标识
    group_id TEXT DEFAULT '',        -- 群 ID
    sender TEXT DEFAULT '',          -- 发送者
    timestamp TEXT DEFAULT '',       -- 时间戳
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,     -- 关联的原始消息 ID
    chunk_index INTEGER NOT NULL,    -- 分块序号
    content TEXT NOT NULL,            -- 分块内容
    chroma_id TEXT DEFAULT '',        -- Chroma 中的向量 ID
    FOREIGN KEY (message_id) REFERENCES messages(id)
);
```

---

## 七、执行顺序建议

```
第一阶段（基础设施）
  T1 环境与依赖 ──→ T2 配置层 ──→ T3 数据模型
     │
     ▼
第二阶段（核心能力）
  T4 知识库模块（KnowledgeBase 实现） ──→ T5 Embedding 构建
     │
     ▼
第三阶段（集成）
  T6 提示词修改 ──→ T7 Agent 改造 ──→ T8 FastAPI 路由
     │
     ▼
第四阶段（验证）
  T9 文档与验证
```