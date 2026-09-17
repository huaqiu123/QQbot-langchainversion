"""知识库核心模块，封装 Chroma 向量检索操作。"""
from __future__ import annotations

import logging
import warnings
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
        """初始化知识库。

        构造时不连接 Chroma，仅保存配置。真正的向量存储连接在首次
        调用 ``_get_store()`` 时懒初始化。

        Args:
            persist_dir: Chroma 持久化目录，如 ``data/chroma``。
            embedding_func: 文本嵌入模型实例（BGE-M3 等），传 None 则不可用。
            collection_name: Chroma 集合名，同目录下可存多个独立集合。
            similarity_threshold: 检索过滤阈值，低于此相似度的结果丢弃。
            search_top_k: 每次检索最多返回的候选结果数。
        """
        # 向量数据库文件持久化目录
        self.persist_dir = persist_dir
        # 嵌入模型实例，为 None 时 knowledge_base.available 为 False
        self.embedding_func = embedding_func
        # Chroma 集合名（类似数据库的表名）
        self.collection_name = collection_name
        # 余弦相似度阈值，归一化后 [0, 1]，实际配置通常更低（如 0.5）
        self.similarity_threshold = similarity_threshold
        # 每次 search() 检索的最大候选条数
        self.search_top_k = search_top_k
        # Chroma 客户端懒初始化，首次 _get_store() 才建立连接
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

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> int:
        """向知识库添加文本块。

        Args:
            texts: 文本块列表。
            metadatas: 每个文本块对应的元数据列表。

        Returns:
            添加的文本块数量。
        """
        store = self._get_store()
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        store.add_documents(documents)
        return len(texts)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """搜索知识库，返回相似度高于阈值的结果。

        BGE-M3 余弦相似度范围 [-1, 1]，而 Chroma 内部会抛
        ``UserWarning`` 因为预期 [0, 1]。这里抑制 Warning 后手动
        归一化到 [0, 1]。

        Args:
            query: 检索查询文本。

        Returns:
            过滤后的搜索结果列表，每项含 content / source_id / similarity / metadata。
        """
        store = self._get_store()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = store.similarity_search_with_relevance_scores(
                query, k=self.search_top_k
            )
        filtered = []
        for doc, score in results:
            # BGE-M3 余弦相似度 [-1, 1] -> 归一化到 [0, 1]
            similarity = (score + 1.0) / 2.0
            similarity = max(0.0, min(1.0, similarity))
            if similarity < self.similarity_threshold:
                continue
            meta = dict(doc.metadata)
            filtered.append({
                "content": doc.page_content,
                "source_id": meta.pop("source_id", None),
                "similarity": round(similarity, 4),
                "metadata": meta,
            })
        return filtered

    def get_chunk_count(self) -> int:
        """获取知识库中的总块数。"""
        store = self._get_store()
        return store._collection.count()  # type: ignore[union-attr]