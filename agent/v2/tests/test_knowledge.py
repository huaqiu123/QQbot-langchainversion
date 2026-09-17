"""T11a: MessageStore + split_text 单元测试。"""
from __future__ import annotations

import json

import pytest

from app.database import MessageStore
from app.main import split_text


# =================================================================== #
# MessageStore
# =================================================================== #


class TestMessageStore:
    """MessageStore 的增删查计数。使用 :memory: 模式避免写磁盘。"""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.store = MessageStore(":memory:")

    def test_insert_returns_positive_id(self) -> None:
        """写入后返回自增 ID，第一次写入应为 1。"""
        row_id = self.store.insert("你好世界")
        assert row_id == 1

    def test_insert_with_metadata(self) -> None:
        """写入带 metadata 的消息后，metadata 应正确持久化。"""
        meta = {"group_id": "123", "sender_id": "u001"}
        row_id = self.store.insert("测试消息", meta)
        row = self.store._connect().execute(
            "SELECT metadata FROM messages WHERE id = ?", (row_id,)
        ).fetchone()
        assert row is not None
        assert json.loads(row["metadata"]) == meta

    def test_get_by_id_returns_content(self) -> None:
        """按 ID 查询应返回原始文本。"""
        self.store.insert("第一条消息")
        self.store.insert("第二条消息")
        content = self.store.get_by_id(2)
        assert content == "第二条消息"

    def test_get_by_id_nonexistent(self) -> None:
        """查询不存在的 ID 应返回 None。"""
        assert self.store.get_by_id(999) is None

    def test_count_returns_total(self) -> None:
        """count 应准确返回总消息条数。"""
        assert self.store.count() == 0
        self.store.insert("a")
        self.store.insert("b")
        self.store.insert("c")
        assert self.store.count() == 3

    def test_multiple_inserts_auto_increment(self) -> None:
        """连续写入时 ID 持续自增。"""
        ids = [self.store.insert(f"msg_{i}") for i in range(5)]
        assert ids == [1, 2, 3, 4, 5]


# =================================================================== #
# split_text
# =================================================================== #


class TestSplitText:
    """文本分块函数的边界测试。"""

    def test_empty_text(self) -> None:
        """空字符串返回空列表。"""
        assert split_text("") == []

    def test_shorter_than_chunk_size(self) -> None:
        """文本短于 chunk_size 时返回单块。"""
        text = "短文本"
        assert split_text(text, chunk_size=256) == [text]

    def test_exact_chunk_size(self) -> None:
        """文本刚好等于 chunk_size 时返回单块。"""
        text = "a" * 100
        assert split_text(text, chunk_size=100) == [text]

    def test_single_character_overlap(self) -> None:
        """相邻块按 overlap 重叠。"""
        # chunk_size=5, overlap=2
        # "12345" → "12345", "34567", "56789", "78901"
        text = "12345678901"
        chunks = split_text(text, chunk_size=5, overlap=2)
        assert chunks == ["12345", "34567", "56789", "78901"]

    def test_no_overlap(self) -> None:
        """overlap=0 时块不重叠。"""
        text = "aabbccddee"
        chunks = split_text(text, chunk_size=4, overlap=0)
        assert chunks == ["aabb", "ccdd", "ee"]

    def test_overlap_same_as_chunk_size(self) -> None:
        """overlap >= chunk_size 时切片仍能正确推进。"""
        text = "abcdefgh"
        chunks = split_text(text, chunk_size=4, overlap=4)
        # start=0: "abcd"; start=4: "efgh"
        assert chunks == ["abcd", "efgh"]

    def test_unicode_characters(self) -> None:
        """中文字符（多字节）按字符数切分，不按字节数。"""
        text = "你好世界这是一个测试"
        chunks = split_text(text, chunk_size=4, overlap=1)
        # "你好世界", "界这是一", "一个测试"
        assert len(chunks) == 3
        assert chunks[0] == "你好世界"

    def test_large_text(self) -> None:
        """大文本分块验证总数与首尾完整性。"""
        text = "x" * 1000
        chunks = split_text(text, chunk_size=256, overlap=32)
        # 边界验证：总块数 >= ceil(1000/256)=4
        assert len(chunks) >= 4
        assert chunks[0][0] == "x"
        assert chunks[-1][-1] == "x"