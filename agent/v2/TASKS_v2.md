# Small Agent V2 — RAG 知识库增强计划

> **基础**：`agent/`（原 `small_agent/`，已重命名）下的 V1 —— FastAPI + LangChain 1.x `create_agent` + DeepSeek + 可插拔搜索源。
> **核心目标**：在 V1 的即时搜索能力之上，增加**持久化知识库**，实现「先查库、再搜网」的问答流程。
> **开发方式**：**在现有 `agent/` 上原地增量改造** —— 不新建 `small_agent_v2/` 副本目录，V1 代码不为 V2 保留副本。
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
|    接口数量    | 4 个（`/`、`/health`、`/v1/agent/ask`、`/v1/agent/stream`） | 7 个（新增 `/knowledge/ingest`、`/knowledge/status`、`/knowledge/query`） |
| Agent 决策逻辑 | 模型自主判断是否搜索                 | **新增 RAG 路由**：先查库 → 未命中再搜索             |
|    架构依赖    | 无数据库                             | **新增向量库（Chroma）+ 关系库（SQLite）**           |
|   V1 实现状态  | ✅ 已实现并启动验证                    | —                                                     |

### 1.4 明确不做（本轮）

- 不做文件上传/PDF 解析（消息以纯文本形式写入）
- 不做知识库的自动过期/清理（由调用方控制写入内容）
- 不做多租户/权限隔离
- 不做 Embedding 模型微调（使用通用 Embedding API）
- **不安装本地 Embedding 模型**（不引入 `sentence-transformers` / PyTorch，统一走 SiliconFlow 线上 API）
- 不做知识库的手动编辑/删除接口（仅支持追加写入）

---

## 二、技术选型与关键决策

### 2.1 新增技术选型

| 决策点             | 选择                                               | 理由                                                                                                                         |
| ------------------ | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **向量数据库**     | **Chroma**（本地文件存储）                         | 轻量级、零运维、无需独立部署服务；数据存储在本地文件，适合 QQ 机器人这种中小规模场景；Python 原生 API，与 LangChain 集成最好 |
| **Embedding 模型** | ✅ **SiliconFlow 线上 API**（`BAAI/bge-m3`，1024 维） | DeepSeek 官方不提供 Embedding 接口（详见 T5）；SiliconFlow 提供 OpenAI 兼容的 `/v1/embeddings`，**纯 HTTP 调用，无需安装 PyTorch（省约 2GB）**；`bge-m3` 支持 8192 token 上下文，中文效果好 |
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

### 2.4 目录结构（当前实际 + V2 改造标记）

```
QQbot/                          # 仓库根（git 仓库）
├── .gitignore                  # 根级忽略规则（已就绪，含 langbot/）
│
├── agent/                      # ★ V1 代码所在，V2 在此原地改造
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # 【改】新增知识库路由 + lifespan 挂载
│   │   ├── config.py           # 【改】新增 Embedding / Chroma / SQLite 配置
│   │   ├── schemas.py          # 【改】新增 Ingest / Query 模型
│   │   ├── llm.py              # 【改】新增 build_embeddings()
│   │   ├── prompts.py          # 【改】增加 RAG 上下文注入
│   │   ├── agent.py            # 【改】增加 RAG 检索步骤
│   │   ├── search.py           # 不变
│   │   ├── tools.py            # 不变
│   │   ├── knowledge.py        # 【新增】知识库（分块 + 向量化 + 检索）
│   │   └── database.py         # 【新增】SQLite 消息存储
│   ├── knowledge_base/         # 【新增】Chroma 持久化目录（自动生成，gitignore）
│   ├── data/                   # 【新增】SQLite 文件目录（gitignore）
│   ├── .env                    # 已有（不纳入版本控制）
│   ├── .env.example            # 【改】补充 V2 配置项
│   ├── .gitignore              # 【改】追加 knowledge_base/、data/
│   ├── requirements.txt        # 【改】新增 chromadb / langchain-chroma 等
│   ├── README.md               # 【改】补充知识库使用说明
│   ├── TASKS_v1.md             # V1 任务清单（已实现，归档）
│   └── TASKS_v2.md             # 本文档
│
├── connection/                 # NapCat / OneBot 协议层（预留，当前为空）
├── deep-research-report.md     # 技术选型调研报告
└── langbot/                    # 第三方参考代码，已加入 .gitignore，不纳入版本控制
```

> **路径约定**：本文档后续凡写 `app/xxx.py`，均指 `agent/app/xxx.py`。

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
| `app/agent.py`   | **V1 现状**：已有 `__init__(settings, search_provider)`、`run()`、`stream()`、`_build_messages()`、`_pick_graph()`、`_final_answer()`。<br>**V2 改动**：`__init__` 增加 `knowledge_base` 参数；`run()` 与 `stream()` 在 `_build_messages()` **之前**插入 RAG 检索；新增 `_rag_retrieve()` — 检索与上下文组装 |
| `app/main.py`    | **V1 现状**：已有 4 个路由（`/`、`/health`、`/v1/agent/ask`、`/v1/agent/stream`），`lifespan` 中装配 `SearchProvider` 与 `Agent`。<br>**V2 改动**：新增 `/knowledge/ingest`、`/knowledge/status`、`/knowledge/query`；`lifespan` 中追加初始化 `KnowledgeBase` 与 `MessageStore`（沿用 V1 的**带病启动**策略，失败不阻断服务） |

---

## 四、任务清单

### T1 环境与依赖
- [ ] ~~复制 V1 完整代码~~ → **改为在现有 `agent/` 上原地增量开发**，不新建副本目录
- [ ] 改造前先跑通基线：`cd agent && uvicorn app.main:app`，确认 V1 仍可启动（`/health` 返回 `ok`）
- [ ] 新增依赖：`chromadb`、`langchain-chroma`、`langchain-openai`（Embedding 走 HTTP 调用，**不需要** `sentence-transformers` / PyTorch）
- [ ] 创建 `agent/knowledge_base/`（Chroma 持久化目录）
- [ ] 创建 `agent/data/`（SQLite 文件目录）
- [ ] 在 `agent/.gitignore` 中追加 `knowledge_base/`、`data/`、`*.sqlite3`

### T2 配置层 `app/config.py` — 新增配置项

> V1 已有配置项（`deepseek_*` / `llm_*` / `search_*` / `ddgs_*` / `recursion_limit` /
> `max_history_messages` / `request_timeout` / `langsmith_*` / `agent_api_key` / `host` / `port`）
> 与下列新增项**无命名冲突**，直接追加即可。

- [ ] 新增 Embedding 相关（**SiliconFlow 线上 API**）：
  - `embedding_api_key: str = ""` — **必填**，SiliconFlow 账户 API Key（申请：siliconflow.cn 控制台）
  - `embedding_base_url: str = "https://api.siliconflow.cn/v1"` — 端点（**不要**带 `/embeddings` 后缀，SDK 会自己拼）
  - `embedding_model: str = "BAAI/bge-m3"` — 模型名（备选见 T5 选型表）
  - `embedding_dim: int = 1024` — 向量维度（BGE 系列维度固定，必须与模型实际输出一致）
  - `embedding_timeout: float = 30.0` — 单次请求超时
  - `embedding_batch_size: int = 32` — 批量写入时的单请求条数
  - `embedding_dimensions: Optional[int] = None` — **仅 Qwen3 系列可指定**，BGE 系列必须留空
- [ ] 新增知识库相关：
  - `chroma_persist_dir: str = "knowledge_base"` — Chroma 持久化目录
  - `chroma_collection: str = "qqbot_knowledge"` — 集合名
  - `knowledge_top_k: int = 3` — 知识库检索返回 Top-K
  - `knowledge_min_score: float = 0.75` — 相似度阈值，低于此值视为未命中
  - `chunk_size: int = 512` — 文本分块大小（字符数）
  - `chunk_overlap: int = 64` — 文本分块重叠
- [ ] 新增存储相关：
  - `sqlite_path: str = "data/messages.db"` — SQLite 文件路径
- [ ] 新增 `is_embedding_configured()` — 判断 Embedding 是否就绪（供 `/health` 使用）
- [ ] 新增 `is_knowledge_enabled()` — 是否启用知识库（`knowledge_min_score > 0`）

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

> ✅ **已决策：走 SiliconFlow 线上 API**
>
> 背景：DeepSeek 官方 API **不提供 Embedding 接口**（查其「模型 & 价格」页，
> 只有对话模型 `deepseek-flash` / `deepseek-v4-pro`，计费项只有输入/输出 tokens，
> 文档导航中无 Embeddings 章节）。改由 SiliconFlow 提供：
> OpenAI 兼容格式、**纯 HTTP 调用，无需安装 PyTorch（省约 2GB）**。

#### 接口规格（依据 SiliconFlow API 手册）

| 项 | 值 |
| --- | --- |
| 端点 | `POST https://api.siliconflow.cn/v1/embeddings` |
| 鉴权 | `Authorization: Bearer <EMBEDDING_API_KEY>` |
| 请求体 | `{ model, input, encoding_format?, dimensions? }` |
| `input` | `string \| string[]`（批量传数组） |
| `encoding_format` | `float`（默认） \| `base64` |
| `dimensions` | **仅 `Qwen/Qwen3` 系列支持**，BGE 系列传了会报错 |
| 响应 | OpenAI 格式：`{ object, model, data:[{embedding, index}], usage:{prompt_tokens,...} }` |

#### 模型选型

| 模型 | 维度 | 最大输入 token | 说明 |
| --- | --- | --- | --- |
| **`BAAI/bge-m3`（推荐）** | 1024 | **8192** | 中文效果好、上下文长，与 `chunk_size=512` 无冲突 |
| `BAAI/bge-large-zh-v1.5` | 1024 | ⚠️ **512** | 上限仅 512 token（约 250–500 汉字），**会被 `chunk_size=512` 字符撑爆** |
| `netease-youdao/bce-embedding-base_v1` | 768 | ⚠️ 512 | 同上限制 |
| `Qwen/Qwen3-Embedding-0.6B` | 可配 64–1024 | 32768 | 便宜、上下文极长，且支持 `dimensions` 参数 |
| `Qwen/Qwen3-Embedding-4B` | 可配，最大 2560 | 32768 | 效果更好，成本更高 |
| `Qwen/Qwen3-Embedding-8B` | 可配，最大 4096 | 32768 | 最强，成本最高 |

> **为什么默认 `bge-m3` 而不是 `bge-large-zh-v1.5`**：
> 后者最大输入仅 **512 token**，而我们的 `chunk_size=512` 是**字符**。
> 中文场景下 512 字符通常超过 512 token，会直接触发 API 长度报错。
> `bge-m3` 的 8192 token 上限彻底消除这个隐患。
> （若坚持用 `bge-large-zh-v1.5`，必须把 `chunk_size` 降到约 300 字符）

#### 实现要点

- [ ] 复用 `langchain_openai.OpenAIEmbeddings`（SiliconFlow 响应即 OpenAI 格式）：

  ```python
  OpenAIEmbeddings(
      model=settings.embedding_model,          # BAAI/bge-m3
      base_url=settings.embedding_base_url,    # https://api.siliconflow.cn/v1
      api_key=settings.embedding_api_key,
      check_embedding_ctx_length=False,        # ← 关键，见下
      timeout=settings.embedding_timeout,
      # dimensions=settings.embedding_dimensions,  # 仅 Qwen3 系列才传
  )
  ```

- [ ] ⚠️ **`check_embedding_ctx_length=False` 必须显式设置**
      默认 `True` 时，LangChain 会先用 **tiktoken（OpenAI 的 tokenizer）** 对文本做
      长度检查与预切分。它和 BGE 的 tokenizer 分词结果不一致，会导致文本被错误截断
      甚至请求失败。**凡是非 OpenAI 官方端点，一律关掉这个开关。**

- [ ] 依赖：`langchain-openai` 目前躺在 `langchain-deepseek` 的依赖树里（V1 环境已装），
      但**必须在 `requirements.txt` 中显式声明**，不能依赖传递依赖（上游一旦换依赖就断）

- [ ] `build_embeddings(settings) -> Embeddings`，用 `@lru_cache` 单例
      （与 V1 的 `get_chat_model()` 同一模式）

- [ ] `validate_embeddings(settings)` — 启动期校验 `EMBEDDING_API_KEY` 是否配置
      （缺失时告警，走「带病启动」，由 `/health` 暴露）

- [ ] 可选兜底：写一个基于 `httpx` 的自定义 `Embeddings` 子类。
      好处是能复用 `search.py` 已有的连接池思路，且不依赖 `OpenAIEmbeddings` 的
      隐式行为。作为 Plan B，先不实现

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

| 变量                    | 默认值                    | 说明                                              |
| ----------------------- | ------------------------- | ------------------------------------------------- |
| `EMBEDDING_API_KEY`     | 空                              | **必填**，SiliconFlow 账户 API Key                     |
| `EMBEDDING_BASE_URL`    | `https://api.siliconflow.cn/v1` | 端点（**不要**带 `/embeddings` 后缀）                  |
| `EMBEDDING_MODEL`       | `BAAI/bge-m3`                   | 模型名（备选见 T5 选型表，**换模型必须重建集合**）      |
| `EMBEDDING_DIM`         | `1024`                          | 向量维度（BGE 系列固定，必须与模型实际输出一致）        |
| `EMBEDDING_TIMEOUT`     | `30`                            | 单次请求超时（秒）                                      |
| `EMBEDDING_BATCH_SIZE`  | `32`                            | 批量写入时单请求条数                                    |
| `EMBEDDING_DIMENSIONS`  | 空                              | **仅 `Qwen/Qwen3` 系列可填**，BGE 系列必须留空          |
| `CHROMA_PERSIST_DIR`    | `knowledge_base`          | Chroma 持久化目录                                 |
| `CHROMA_COLLECTION`     | `qqbot_knowledge`         | Chroma 集合名                                     |
| `KNOWLEDGE_TOP_K`       | `3`                       | 知识库检索返回 Top-K                              |
| `KNOWLEDGE_MIN_SCORE`   | `0.75`                    | 相似度阈值                                        |
| `CHUNK_SIZE`            | `512`                     | 文本分块大小（字符）                              |
| `CHUNK_OVERLAP`         | `64`                      | 分块重叠（字符）                                  |
| `SQLITE_PATH`           | `data/messages.db`        | SQLite 文件路径                                   |

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
  "embedding_model": "BAAI/bge-m3"
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
  T5 Embedding 构建 ──→ T4 知识库模块（KnowledgeBase 实现）
     │                   （知识库依赖 Embedding，顺序不可颠倒）
     ▼
第三阶段（集成）
  T6 提示词修改 ──→ T7 Agent 改造 ──→ T8 FastAPI 路由
     │
     ▼
第四阶段（验证）
  T10 文档与验证
  （T9 是配置项清单表，随 T2 同步维护，不是独立步骤）
```

---

## 八、待确认项与风险

### 8.1 需先决策

| #   | 问题                          | 选项                                                                                             | 影响                        |
| --- | ----------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------- |
| 1   | ~~**Embedding 来源**~~ ✅ **已决策** | **SiliconFlow 线上 API**（默认 `BAAI/bge-m3`，1024 维），纯 HTTP 调用，无需 PyTorch | 已落实到 T5 / T9 |
| 2   | **RAG 集成方式**              | A 检索结果注入 `extra_context`（实现简单、必定执行）／ B 做成 `knowledge_search` 工具让模型自主调用 | 决定 T6 提示词与 T7 改造方式 |
| 3   | **SQLite 是否必要**           | 保留（可溯源原始消息）／ 去掉（仅 Chroma，靠 metadata 承载）                                      | 是否新增 `database.py`       |
| 4   | **是否保留 `/knowledge/query`** | 保留（调试阈值用，价值高）／ 去掉（少一个接口）                                                   | 接口数量                     |
| 5   | **下一步方向**                | 先做 V2 知识库 ／ 先做 `connection/` 的 NapCat 接入                                               | 排期                        |

### 8.2 已知风险

| 风险                           | 说明                                                                                                                | 对策                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **相似度阈值难调**             | `0.75` 是经验值。不同 Embedding 模型的分数分布差异很大，同一阈值在 A 模型命中、在 B 模型可能全部未命中                | 先用 `/knowledge/query` 实测分数分布再定阈值；确保阈值可配置     |
| **Embedding 依赖外网 API**     | 每次写入与检索都要调 SiliconFlow；网络抖动或服务不可用会直接影响 `ingest` 与问答首字延迟                             | 复用搜索层的容错策略：检索失败时降级为「未命中」并走联网搜索，**不阻断问答**；`ingest` 失败则返回明确错误让调用方重试 |
| **Embedding 按 token 计费**    | 群聊持续写入会累积成本（`bge-m3` 单次最多 8192 token）                                                               | 写入前做去重与长度过滤；关注免费额度；`/knowledge/status` 暴露 `doc_count` 便于观察量级 |
| **`check_embedding_ctx_length` 陷阱** | `OpenAIEmbeddings` 默认用 tiktoken（OpenAI 的 tokenizer）预切分文本，与 BGE 分词结果不一致，可能导致内容被错误截断或直接报错 | **显式设 `check_embedding_ctx_length=False`**（见 T5） |
| **换模型必须重建集合**         | 不同模型的向量维度与语义空间不同，混用会导致检索结果错乱；维度不符时 Chroma 直接报错                                 | 选定模型后不随意更换；如需更换，清空 `knowledge_base/` 重新 ingest；把模型名写入集合 metadata 便于校验 |
| **token 上限与 `chunk_size` 错配** | `BAAI/bge-large-zh-v1.5` 等模型上限仅 512 **token**，而 `chunk_size` 单位是 512 **字符**，中文场景下后者远超前者，会触发 API 长度报错 | 默认用 `bge-m3`（8192 token）；若换用 512-token 模型，必须同步把 `chunk_size` 降到 ~300 字符 |
| **分块粒度与群消息不匹配**     | 群聊天记录通常是一句话，套用 `chunk_size=512` 会把多条无关消息混进同一个块，稀释语义                                  | 短文本（< `chunk_size`）整条入库不切分，仅长文本才走分块器        |
| **Chroma 版本兼容**            | Chroma 0.5 → 1.x 有破坏性 API 变更，`langchain-chroma` 需与 `chromadb` 版本对齐                                       | 安装时锁定版本，装完先跑一次最小读写验证                          |
| **首字延迟增加**               | 每次提问前多一次向量检索（本地模型 10–100ms，API 100–300ms）                                                          | 可接受；对延迟敏感时可对同一问题做检索结果缓存                    |
| **⚠️ DeepSeek 模型名可能已变更** | 核查官方「模型 & 价格」页时发现当前在售模型为 `deepseek-flash` / `deepseek-v4-pro`，**V1 默认的 `deepseek-chat` 未出现在列表中** | **改造前先实测 V1 能否正常调用**；若失效需更新 `DEEPSEEK_MODEL` 默认值与 `TOOL_CAPABLE_MODELS` 白名单 |
| **知识库内容质量不可控**       | 写入内容来自群聊，含噪音、口语、错别字，且可能被恶意灌入                                                              | `ingest` 接口挂 `X-API-Key` 鉴权；对写入内容做长度与频率限制      |