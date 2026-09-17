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
        """初始化消息存储。

        Args:
            db_path: SQLite 数据库文件路径，如 ``data/knowledge.db``。
                     若父目录不存在会自动创建。
        """
        self.db_path = db_path
        # 连接对象懒初始化，首次 _connect() 时才建立连接
        self._conn: sqlite3.Connection | None = None
        # 确保数据库文件所在父目录存在（如 data/ 不存在则创建）
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # 建表 + 开启 WAL 模式
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """获取或创建 SQLite 连接（懒初始化，单例复用）。

        首次调用时建立连接并设置 row_factory，后续调用直接返回
        缓存的连接对象，避免频繁开关连接的开销。

        Returns:
            缓存的 sqlite3.Connection 对象。
        """
        if self._conn is None:
            # 首次连接：创建连接对象并绑定到实例
            self._conn = sqlite3.connect(self.db_path)
            # 让查询结果以 sqlite3.Row 形式返回，支持按列名取值（如 row["content"]）
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """创建 messages 表并启用 WAL 日志模式。

        ``IF NOT EXISTS`` 保证幂等——多次调用不会报错或重复建表。
        WAL（Write-Ahead Logging）允许读写并发，避免写入时阻塞读取。
        """
        conn = self._connect()
        # 创建消息表：id 自增主键，content 明文，metadata 存 JSON，created_at 自动时间戳
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                content     TEXT    NOT NULL,
                metadata    TEXT,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # WAL 模式：写操作不阻塞读操作，适合 Chroma 读 + SQLite 写并发的场景
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()

    def insert(self, content: str, metadata: Dict[str, Any] | None = None) -> int:
        conn = self._connect()
        cursor = conn.execute(
            "INSERT INTO messages (content, metadata) VALUES (?, ?)",
            (content, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_id(self, source_id: int) -> str | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT content FROM messages WHERE id = ?", (source_id,)
        ).fetchone()
        return str(row["content"]) if row else None

    def count(self) -> int:
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) AS cnt FROM messages").fetchone()
        return int(row["cnt"]) if row else 0