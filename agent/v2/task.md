# Agent V2 — RAG 知识库增强 Tasks

## 文件清单

| 操作 | 文件               | 职责                                    |
| ---- | ------------------ | --------------------------------------- |
| 修改 | `requirements.txt` | 新增 chromadb、langchain-chroma 依赖    |
| 修改 | `app/config.py`    | 新增 8 个知识库配置项                   |
| 修改 | `app/llm.py`       | 新增 `build_embeddings()` 工厂函数      |
| 新建 | `app/database.py`  | SQLite 消息存储（`MessageStore` 类）    |
| 新建 | `app/knowledge.py` | Chroma 知识库（`KnowledgeBase` 类）     |
| 修改 | `app/prompts.py`   | 新增 `build_system_prompt()` 运行时构造 |
| 修改 | `app/agent.py`     | Agent 集成知识库检索                    |
| 修改 | `app/schemas.py`   | 新增 3 个知识库请求/响应模型            |
| 修改 | `app/main.py`      | 新增路由、lifespan 初始化、/health 改造 |
| 修改 | `.env.example`     | 补充新配置示例                          |

## 执行顺序

```
T1 ─→ T2 ─→ T3 ─→ T5a ─→ T5b ─→ T7a ─→ T7b ─→ T9a ─→ T9b ─→ T9c ─→ T9d
              │              ↗         ↗
              ├──→ T4a ─→ T4b ─→ T8a ─→ T8b
              │
              └──→ T6 ─→ T10
                            ↑
                          T14(随时可做)
```

T11/T12/T13 在全部编码通过后执行。

---

## T1：新增依赖安装

**文件：** `requirements.txt`
**依赖：** 无
**步骤：**

1. 打开 `requirements.txt`，在末尾新增两行：

```
chromadb>=1.5.0
langchain-chroma>=0.1.0
```

2. 在项目根目录安装依赖：

```bash
cd agent
pip install -r requirements.txt
```

**验证：** `pip install` 成功无报错；`python -c "import chromadb; import langchain_chroma"` 不抛 ImportError

---

## T2：config.py — 新增知识库配置项

**文件：** `app/config.py`
**依赖：** T1
**步骤：**

1. 在 `Settings` 类的 `# ---------- Agent ----------` 组之前，插入一组新配置段：

```python
# ---------- 知识库（Embedding + 向量检索） ----------
embedding_api_key: str = Field(
    "", description="SiliconFlow Embedding API Key；空则不启用知识库"
)
embedding_model: str = Field(
    "BAAI/bge-m3", description="Embedding 模型名"
)
embedding_base_url: str = Field(
    "https://api.siliconflow.cn/v1", description="Embedding API 基础地址"
)
chroma_persist_dir: str = Field(
    "data/chroma", description="Chroma 持久化目录（相对于工作目录）"
)
knowledge_db_path: str = Field(
    "data/knowledge.db", description="SQLite 消息存储路径"
)
knowledge_similarity_threshold: float = Field(
    0.7, description="相似度阈值，低于此值不注入 prompt"
)
knowledge_search_top_k: int = Field(
    5, description="检索返回的最大结果数"
)
knowledge_chunk_size: int = Field(
    256, description="文本分块大小（字符数）"
)
knowledge_chunk_overlap: int = Field(
    32, description="文本分块重叠字符数"
)
```

2. 8 个字段全部在同一个 `# ---------- 知识库 ----------` 组下

**验证：**
- `python -c "from app.config import get_settings; s=get_settings(); print(s.embedding_api_key)"` 输出空字符串
- `python -c "from app.config import get_settings; s=get_settings(); print(s.knowledge_similarity_threshold)"` 输出 `0.7`
- 不修改 `.env`，服务启动不报错

---

## T3：llm.py — 新增 `build_embeddings()` 工厂函数

**文件：** `app/llm.py`
**依赖：** T2
**步骤：**

1. 在文件顶部新增 import：

```python
from langchain_core.embeddings import Embeddings
```

2. 在文件末尾新增函数：

```python
def build_embeddings(settings: Settings) -> Embeddings | None:
    """构造 Embeddings 客户端（SiliconFlow / OpenAI 兼容 API）。

    未配置 EMBEDDING_API_KEY 时返回 None，外层 knowledge.py
    据此将 available 置为 False。

    Args:
        settings: 全局配置。

    Returns:
        Embeddings 实例，或 None（未配置时）。
    """
    if not settings.embedding_api_key.strip():
        return None

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        logger.warning("langchain-openai 未安装，Embedding 功能不可用")
        return None

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )
```

**验证：**
- `python -c "from app.llm import build_embeddings; from app.config import get_settings; print(build_embeddings(get_settings()))"` 输出 `None`（Key 为空时）
- `build_embeddings` 不发起网络请求（构造函数无副作用）

---

## T4a：database.py — MessageStore 建表和 insert()

**文件：** `app/database.py`（新建）
**依赖：** T2
**步骤：**

1. 创建 `app/database.py`，写入以下代码：

```python
"""SQLite 消息原文存储，与 Chroma 互补。"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MessageStore:
    """SQLite 存储消息原文与元数据。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    content     TEXT    NOT NULL,
                    metadata    TEXT,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, content: str, metadata: Dict[str, Any] | None = None) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (content, metadata) VALUES (?, ?)",
                (content, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
```

**验证：**
- `python -c "from app.database import MessageStore; s=MessageStore(':memory:'); id=s.insert('hello', {'g':'1'}); print(id)"` 输出 `1`

---

## T4b：database.py — get_by_id() 和 count()

**文件：** `app/database.py`
**依赖：** T4a
**步骤：**

1. 在 `MessageStore` 类中追加两个方法：

```python
    def get_by_id(self, source_id: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM messages WHERE id = ?", (source_id,)
            ).fetchone()
            return str(row["content"]) if row else None

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM messages").fetchone()
            return int(row["cnt"]) if row else 0
```

**验证：**
```python
from app.database import MessageStore
s = MessageStore(":memory:")
id = s.insert("hello")
assert s.get_by_id(id) == "hello"
assert s.get_by_id(999) is None
assert s.count() == 1
```
运行不抛 AssertionError

---

## T5a：knowledge.py — KnowledgeBase 初始化 + 延迟加载

**文件：** `app/knowledge.py`（新建）
**依赖：** T2, T3
**步骤：**

1. 创建 `app/knowledge.py`，写入：

```python
"""知识库核心模块，封装 Chroma 向量检索操作。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知识库，封装 Chroma 向量检索。"""

    def __init__(
        self,
        persist_dir: str,
        embedding_func: Embeddings,
        collection_name: str = "qqbot_knowledge",
        similarity_threshold: float = 0.7,
        search_top_k: int = 5,
    ):
        self.persist_dir = persist_dir
        self.embedding_func = embedding_func
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.search_top_k = search_top_k
        self._vector_store: Chroma | None = None

    def _get_store(self) -> Chroma:
        if self._vector_store is None:
            persist_path = Path(self.persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_func,
                persist_directory=str(persist_path),
            )
        return self._vector_store

    @property
    def available(self) -> bool:
        return self.embedding_func is not None
```

**验证：**
- `python -c "from app.knowledge import KnowledgeBase; print('import ok')"` 不报错
- `_get_store()` 首次调用才创建目录和 Chroma 实例（延迟加载）

---

## T5b：knowledge.py — add_texts() + search() + get_chunk_count()

**文件：** `app/knowledge.py`
**依赖：** T5a
**步骤：**

1. 在 `KnowledgeBase` 类中追加三个方法：

```python
    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> int:
        store = self._get_store()
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        store.add_documents(documents)
        return len(texts)

    def search(self, query: str) -> List[Dict[str, Any]]:
        store = self._get_store()
        results = store.similarity_search_with_relevance_scores(
            query, k=self.search_top_k
        )
        filtered = []
        for doc, score in results:
            if score < self.similarity_threshold:
                continue
            meta = dict(doc.metadata)
            filtered.append({
                "content": doc.page_content,
                "source_id": meta.pop("source_id", None),
                "similarity": score,
                "metadata": meta,
            })
        return filtered

    def get_chunk_count(self) -> int:
        store = self._get_store()
        return store._collection.count()  # type: ignore[union-attr]
```

**验证：**
```python
from app.knowledge import KnowledgeBase
# 使用一个 mock embedding 来测试（需要 langchain 测试用 embedding）
# 或者在集成测试中用真实 Chroma 验证
kb = KnowledgeBase(...)
cnt = kb.add_texts(["hello world"], [{"source_id": 1}])
assert cnt == 1
```

注意：完整验证需要真实的 Embedding 实例，可在 T11 测试中覆盖。

---

## T6：prompts.py — 新增 `build_system_prompt()`

**文件：** `app/prompts.py`
**依赖：** 无
**步骤：**

1. 在文件末尾新增函数：

```python
def build_system_prompt(knowledge_context: str | None = None) -> str:
    """根据是否有知识库检索结果，构造最终的 system prompt。

    Args:
        knowledge_context: 从知识库检索到的相关内容（已格式化文本），
            为 None 表示未命中或知识库不可用。

    Returns:
        完整的 system prompt 字符串。
    """
    if knowledge_context:
        return f"""{SYSTEM_PROMPT}

【知识库参考信息】
以下是从历史知识中检索到的相关内容，请**优先参考**这些信息回答问题。
如果参考信息足以回答，直接给出答案；如果不完整或不确定，请使用
web_search 工具联网搜索补充。

{knowledge_context}
"""
    return SYSTEM_PROMPT
```

**验证：**
- `build_system_prompt(None) == SYSTEM_PROMPT` 返回 True
- `build_system_prompt("test")` 包含 `【知识库参考信息】` 和 `"test"`

---

## T7a：agent.py — 新增 `knowledge_base` 参数

**文件：** `app/agent.py`
**依赖：** T5b
**步骤：**

1. 在文件顶部 import 区域新增：

```python
from .knowledge import KnowledgeBase
```

2. 将 `from .prompts import SYSTEM_PROMPT` 改为：

```python
from .prompts import build_system_prompt
```

3. 在 `Agent.__init__` 的参数列表末尾新增：

```python
knowledge_base: KnowledgeBase | None = None,
```

4. 在 `__init__` 方法体中新增赋值：

```python
self.knowledge_base = knowledge_base
```

**验证：** `Agent(settings, provider, knowledge_base=None)` 构造成功（兼容 V1 调用方）

---

## T7b：agent.py — 修改 `_build_messages()` 集成知识库检索

**文件：** `app/agent.py`
**依赖：** T7a, T6
**步骤：**

1. 在 `_build_messages` 方法中，在 `messages = [SystemMessage(...)]` 之前插入知识库检索逻辑：

```python
def _build_messages(
    self,
    question: str,
    history: Optional[List[ChatMessage]] = None,
    use_search: Optional[bool] = None,
    extra_context: Optional[str] = None,
) -> List[BaseMessage]:
    # V2：检索知识库
    knowledge_context = None
    if self.knowledge_base and self.knowledge_base.available:
        try:
            results = self.knowledge_base.search(question)
            if results:
                knowledge_context = "\n\n".join(
                    f"[来源 {i+1}] {r['content']}"
                    for i, r in enumerate(results)
                )
        except Exception:
            logger.warning("知识库检索失败，跳过", exc_info=True)

    # 动态构造 system prompt
    system_prompt = build_system_prompt(knowledge_context)

    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    # ... 后续历史消息、extra_context 逻辑不变
```

2. 注意保留 V1 原有的历史消息和 extra_context 逻辑，不做其他修改

**验证：**
- 不带 `knowledge_base` 参数（或传 None）时，`_build_messages` 行为与 V1 完全一致
- 带 `knowledge_base` 且有检索结果时，`_build_messages` 返回的 system prompt 包含 `【知识库参考信息】`
- 检索抛异常时，方法不向外传播异常

---

## T8a：schemas.py — 新增 `KnowledgeIngestRequest` 和 `KnowledgeIngestResponse`

**文件：** `app/schemas.py`
**依赖：** 无
**步骤：**

1. 在文件末尾新增两个模型：

```python
class KnowledgeIngestRequest(BaseModel):
    """`POST /knowledge/ingest` 请求体。"""
    content: str = Field(..., min_length=1, description="要写入的文本内容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="来源元数据（群组 ID、发送者等）"
    )


class KnowledgeIngestResponse(BaseModel):
    """`POST /knowledge/ingest` 成功响应。"""
    chunk_count: int = 0
    total_chunks: int = 0
```

**验证：**
- `KnowledgeIngestRequest(content="test")` 构造成功
- `KnowledgeIngestRequest(content="")` 抛 ValidationError
- `KnowledgeIngestResponse(chunk_count=3, total_chunks=10)` 构造成功

---

## T8b：schemas.py — 新增 `KnowledgeStatusResponse`

**文件：** `app/schemas.py`
**依赖：** 无（与 T8a 可并行）
**步骤：**

1. 在 `KnowledgeIngestResponse` 之后新增：

```python
class KnowledgeStatusResponse(BaseModel):
    """`GET /knowledge/status` 响应体。"""
    available: bool = False
    chunk_count: int = 0
    source_count: int = 0
```

**验证：** `KnowledgeStatusResponse(available=True, chunk_count=5, source_count=2)` 构造成功

---

## T9a：main.py — lifespan 中新增知识库初始化

**文件：** `app/main.py`
**依赖：** T4b, T5b, T8b
**步骤：**

1. 在文件顶部 import 区域新增：

```python
from .database import MessageStore
from .knowledge import KnowledgeBase
from .llm import build_embeddings
from .schemas import (
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeStatusResponse,
)
```

2. 在 `lifespan` 函数中，`provider = build_search_provider(settings)` 之后插入：

```python
    # V2：知识库初始化
    knowledge_base = None
    message_store = None
    if embedding_func := build_embeddings(settings):
        try:
            message_store = MessageStore("data/messages.db")
            knowledge_base = KnowledgeBase(
                persist_dir=settings.chroma_persist_dir,
                embedding_func=embedding_func,
                similarity_threshold=settings.knowledge_similarity_threshold,
                search_top_k=settings.knowledge_search_top_k,
            )
            logger.info("知识库已就绪 chroma=%s", settings.chroma_persist_dir)
        except Exception as exc:
            logger.error("知识库初始化失败：%s", exc)

    app.state.knowledge_base = knowledge_base
    app.state.message_store = message_store
```

3. 在 Agent 构造调用中传入 `knowledge_base`：

```python
app.state.agent = Agent(
    settings, provider,
    knowledge_base=knowledge_base,  # V2 新增
)
```

**验证：** 启动服务，无 Embedding Key 时 `/health` 返回 `status: ok`，`knowledge.available: false`

---

## T9b：main.py — 改造 `/health` 路由

**文件：** `app/main.py`
**依赖：** T9a
**步骤：**

1. 修改 `health` 路由，移除 `response_model=HealthResponse`，改为直接返回字典，末尾增加 knowledge 信息：

```python
@app.get("/health", tags=["meta"])
async def health(request: Request) -> Dict[str, Any]:
    """健康检查（V2：新增 knowledge 字段）。"""
    settings = get_settings()
    agent = getattr(request.app.state, "agent", None)
    provider = getattr(request.app.state, "search_provider", None)
    kb = getattr(request.app.state, "knowledge_base", None)
    store = getattr(request.app.state, "message_store", None)

    return {
        "status": "ok" if agent else "degraded",
        "model": settings.deepseek_model,
        "search_provider": provider.name if provider else "",
        "agent_impl": agent.impl if agent else "",
        "graph_ready": agent is not None,
        "llm_configured": settings.is_llm_configured(),
        "knowledge": {
            "available": kb is not None and kb.available,
            "chunk_count": kb.get_chunk_count() if kb else 0,
            "source_count": store.count() if store else 0,
        },
    }
```

2. 在文件顶部新增 `Dict` 到 `from typing` 的导入中：

```python
from typing import Any, Dict
```

**验证：** `GET /health` 返回体中包含 `knowledge` 字段，未配置 Key 时 `knowledge.available = false`

---

## T9c：main.py — `split_text()` 工具函数

**文件：** `app/main.py`
**依赖：** 无
**步骤：**

1. 在文件末尾（路由定义之后），新增工具函数：

```python
def split_text(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    """将文本按字符数分块，带重叠。

    Args:
        text: 原始文本。
        chunk_size: 每块最大字符数。
        overlap: 相邻块重叠字符数。

    Returns:
        文本块列表。
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks
```

**验证：**
- `split_text("")` 返回 `[]`
- `split_text("short")` 返回 `["short"]`
- `split_text("a" * 300, 200, 50)` 返回长度为 2 的列表，第二块开头与第一块尾部有 50 字符重叠

---

## T9d：main.py — 新增知识库路由

**文件：** `app/main.py`
**依赖：** T9a, T9c, T8b
**步骤：**

1. 在 `_get_agent` 函数之后新增辅助函数：

```python
def _get_knowledge_base(request: Request) -> tuple[KnowledgeBase, MessageStore]:
    kb = getattr(request.app.state, "knowledge_base", None)
    store = getattr(request.app.state, "message_store", None)
    if kb is None or store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="知识库未配置：EMBEDDING_API_KEY 未设置或初始化失败",
        )
    return kb, store
```

2. 在 `stream` 路由之后新增两个路由：

```python
@app.post(
    "/knowledge/ingest",
    response_model=KnowledgeIngestResponse,
    tags=["knowledge"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_knowledge(
    payload: KnowledgeIngestRequest,
    request: Request,
) -> KnowledgeIngestResponse:
    kb, store = _get_knowledge_base(request)
    source_id = store.insert(payload.content, payload.metadata)
    try:
        chunks = split_text(
            payload.content,
            chunk_size=get_settings().knowledge_chunk_size,
            overlap=get_settings().knowledge_chunk_overlap,
        )
        metadatas = [
            {"source_id": source_id, "chunk_index": i, **payload.metadata}
            for i in range(len(chunks))
        ]
        chunk_count = kb.add_texts(chunks, metadatas)
        return KnowledgeIngestResponse(
            chunk_count=chunk_count,
            total_chunks=kb.get_chunk_count(),
        )
    except Exception:
        logger.exception("Chroma 写入失败")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="知识库写入失败",
        )


@app.get(
    "/knowledge/status",
    response_model=KnowledgeStatusResponse,
    tags=["knowledge"],
)
async def knowledge_status(request: Request) -> KnowledgeStatusResponse:
    kb = getattr(request.app.state, "knowledge_base", None)
    store = getattr(request.app.state, "message_store", None)
    if kb is None or store is None:
        return KnowledgeStatusResponse(available=False, chunk_count=0, source_count=0)
    return KnowledgeStatusResponse(
        available=kb.available,
        chunk_count=kb.get_chunk_count(),
        source_count=store.count(),
    )
```

**验证：**
- 启动服务，无 Embedding Key 时 `POST /knowledge/ingest` 返回 503
- 有 Embedding Key 时 `POST /knowledge/ingest` 返回 200 且 `chunk_count > 0`
- 有 Embedding Key 时 `GET /knowledge/status` 返回 `available: true`
- 配置了 `AGENT_API_KEY` 时，不带 Key 的 `/knowledge/ingest` 请求返回 401

---

## T10：T14 — `.env.example` 补充新配置（随时可做）

**文件：** `.env.example`
**依赖：** 无
**步骤：**

1. 在文件末尾追加：

```env
# ── 知识库（可选，不填则知识库功能不可用） ──
EMBEDDING_API_KEY=sk-xxx
# EMBEDDING_MODEL=BAAI/bge-m3
# EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
# CHROMA_PERSIST_DIR=data/chroma
# KNOWLEDGE_SIMILARITY_THRESHOLD=0.7
# KNOWLEDGE_SEARCH_TOP_K=5
# KNOWLEDGE_CHUNK_SIZE=256
# KNOWLEDGE_CHUNK_OVERLAP=32
```

**验证：** 文件内容包含上述配置行

---

## T11a：测试 — MessageStore + split_text 单元测试

**文件：** `tests/test_knowledge.py`（新增）
**依赖：** T4b, T9c
**步骤：**

1. 创建 `tests/test_knowledge.py`，编写测试：

```python
"""MessageStore 和 split_text 单元测试。"""
from app.database import MessageStore
from app.main import split_text


def test_message_store_insert_and_query():
    store = MessageStore(":memory:")
    id1 = store.insert("hello", {"group": "g1"})
    id2 = store.insert("world")
    assert id1 == 1
    assert id2 == 2
    assert store.get_by_id(1) == "hello"
    assert store.get_by_id(999) is None
    assert store.count() == 2


def test_split_text_empty():
    assert split_text("") == []


def test_split_text_short():
    assert split_text("short") == ["short"]


def test_split_text_chunking():
    text = "a" * 300
    chunks = split_text(text, 200, 50)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 200
    assert chunks[1] == "a" * 100
```

**验证：** `pytest tests/test_knowledge.py -v` 全部通过

---

## T11b：测试 — KnowledgeBase 单元测试

**文件：** `tests/test_knowledge.py`
**依赖：** T5b, T3
**步骤：**

1. 追加 `KnowledgeBase` 测试（使用临时目录和真实 SiliconFlow Embedding，或使用 mock）：

```python
def test_knowledge_base_available():
    """无 embedding 时 available 为 False。"""
    from app.knowledge import KnowledgeBase
    kb = KnowledgeBase("/tmp/test_chroma", embedding_func=None)  # type: ignore
    assert kb.available is False
```

注意：涉及真实 Embedding API 的测试需要配置 `EMBEDDING_API_KEY`，建议标记为集成测试。

**验证：** 单元测试部分通过

---

## T12：集成测试 — Agent RAG 集成

**文件：** `tests/test_rag_integration.py`（新增）
**依赖：** T9d
**步骤：**

1. 创建集成测试文件，覆盖三个场景：

```python
"""Agent RAG 集成测试。需要配置 EMBEDDING_API_KEY 才能运行。"""

import pytest
from app.database import MessageStore
from app.knowledge import KnowledgeBase
from app.llm import build_embeddings
from app.main import split_text
from app.config import get_settings


@pytest.mark.asyncio
async def test_rag_knowledge_used():
    """写入知识库后提问，回答包含知识库内容。"""
    settings = get_settings()
    if not settings.embedding_api_key:
        pytest.skip("需要 EMBEDDING_API_KEY")
    embedding = build_embeddings(settings)
    kb = KnowledgeBase("/tmp/test_chroma_rag", embedding)
    store = MessageStore(":memory:")
    # 写入
    content = "DeepSeek-V3 于 2024 年 12 月发布"
    sid = store.insert(content)
    chunks = split_text(content)
    kb.add_texts(chunks, [{"source_id": sid}])
    # 检索
    results = kb.search("DeepSeek-V3 发布时间")
    assert len(results) > 0
    assert "2024" in results[0]["content"]


@pytest.mark.asyncio
async def test_rag_empty_knowledge():
    """知识库为空时，检索返回空列表。"""
    settings = get_settings()
    if not settings.embedding_api_key:
        pytest.skip("需要 EMBEDDING_API_KEY")
    embedding = build_embeddings(settings)
    kb = KnowledgeBase("/tmp/test_chroma_empty", embedding)
    results = kb.search("随便问问")
    assert results == []
```

**验证：** `pytest tests/test_rag_integration.py -v` 通过

---

## T13：兼容性测试

**文件：** `tests/test_compat.py`（新增或追加到现有）
**依赖：** T9d
**步骤：**

1. 创建测试文件：

```python
"""降级兼容性测试。"""

import pytest
from app.database import MessageStore
from app.knowledge import KnowledgeBase
from app.llm import build_embeddings
from app.config import get_settings


def test_no_embedding_key():
    """未配置 Embedding Key 时，知识库不可用但应用正常。"""
    settings = get_settings()
    # 临时清空 Key（假设配置里没有）
    embedding = build_embeddings(settings)
    assert embedding is None  # 或用 pytest.mark.skipif


def test_knowledge_base_none_safe():
    """knowledge_base 为 None 时，Agent 正常构造。"""
    from app.agent import Agent
    from app.config import get_settings
    from app.search import build_search_provider
    settings = get_settings()
    provider = build_search_provider(settings)
    agent = Agent(settings, provider, knowledge_base=None)
    assert agent is not None
```

**验证：** `pytest tests/test_compat.py -v` 通过