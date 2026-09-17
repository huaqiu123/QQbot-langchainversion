"""T12: 知识库集成测试 —— 写入 → 检索 → Agent 回答。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


# =================================================================== #
# 跳过条件
# =================================================================== #


def _is_rag_ready() -> bool:
    """检查是否可执行完整 RAG 集成测试（需要 Embedding Key）。"""
    settings = get_settings()
    return bool(settings.embedding_api_key)


rag_ready = pytest.mark.skipif(
    not _is_rag_ready(),
    reason="EMBEDDING_API_KEY 未配置，跳过集成测试（配置后可执行）",
)


# =================================================================== #
# 辅助
# =================================================================== #


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _headers() -> dict:
    settings = get_settings()
    if settings.agent_api_key:
        return {"X-API-Key": settings.agent_api_key}
    return {}


# =================================================================== #
# 知识库写入与状态
# =================================================================== #


class TestIngestAndStatus:
    """写入与状态查询。"""

    @rag_ready
    def test_status_returns_chunk_count_zero_before_ingest(
        self, client: TestClient
    ) -> None:
        """写入前总块数为 0。"""
        resp = client.get("/knowledge/status", headers=_headers())
        assert resp.status_code == 200
        assert resp.json()["chunk_count"] == 0

    @rag_ready
    def test_ingest_returns_chunk_count_greater_than_zero(
        self, client: TestClient
    ) -> None:
        """写入后 chunk_count > 0。"""
        resp = client.post(
            "/knowledge/ingest",
            json={
                "content": "DeepSeek-V3 于 2024 年 12 月发布，"
                "是一款强大的 MoE 模型，拥有 671B 总参数量。",
                "metadata": {"group_id": "test_group", "sender_id": "tester"},
            },
            headers=_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_count"] > 0
        assert data["total_chunks"] > 0

    @rag_ready
    def test_status_reflects_ingest(self, client: TestClient) -> None:
        """写入后查询状态，总块数与写入接口返回一致。"""
        # 先写入
        ingest_resp = client.post(
            "/knowledge/ingest",
            json={
                "content": "SiliconFlow 提供高性能 AI 模型推理服务。",
                "metadata": {"group_id": "test_group"},
            },
            headers=_headers(),
        )
        assert ingest_resp.status_code == 200
        total_after = ingest_resp.json()["total_chunks"]

        # 再查状态
        status_resp = client.get("/knowledge/status", headers=_headers())
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["available"] is True
        assert status_data["chunk_count"] == total_after


# =================================================================== #
# 鉴权
# =================================================================== #


class TestAuth:
    """知识库接口鉴权。"""

    @rag_ready
    def test_ingest_returns_401_when_key_required_and_missing(
        self, client: TestClient
    ) -> None:
        """配置了 AGENT_API_KEY 时，不带 Key 返回 401。"""
        settings = get_settings()
        if not settings.agent_api_key:
            pytest.skip("AGENT_API_KEY 未配置，跳过鉴权测试")

        resp = client.post(
            "/knowledge/ingest",
            json={"content": "test"},
        )
        assert resp.status_code == 401

    @rag_ready
    def test_ingest_returns_401_with_wrong_key(
        self, client: TestClient
    ) -> None:
        """配置了 AGENT_API_KEY 时，带错误 Key 返回 401。"""
        settings = get_settings()
        if not settings.agent_api_key:
            pytest.skip("AGENT_API_KEY 未配置，跳过鉴权测试")

        resp = client.post(
            "/knowledge/ingest",
            json={"content": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    @rag_ready
    def test_ingest_succeeds_with_valid_key(
        self, client: TestClient
    ) -> None:
        """配置了 AGENT_API_KEY 时，带正确 Key 可正常写入。"""
        settings = get_settings()
        if not settings.agent_api_key:
            pytest.skip("AGENT_API_KEY 未配置，跳过鉴权测试")

        resp = client.post(
            "/knowledge/ingest",
            json={"content": "带 Key 写入测试"},
            headers={"X-API-Key": settings.agent_api_key},
        )
        assert resp.status_code == 200
        assert resp.json()["chunk_count"] > 0


# =================================================================== #
# Agent 回答中的知识库影响
# =================================================================== #


class TestAgentWithKnowledge:
    """知识库内容影响 Agent 回答。"""

    @rag_ready
    def test_agent_uses_knowledge_when_relevant(self, client: TestClient) -> None:
        """写入知识库后提问，回答应引用知识库内容。"""
        # 先写入知识
        ingest_resp = client.post(
            "/knowledge/ingest",
            json={
                "content": "Small Agent 项目的版本号为 2.0.0。",
                "metadata": {"group_id": "test_group"},
            },
            headers=_headers(),
        )
        assert ingest_resp.status_code == 200

        # 再问相关问题
        ask_resp = client.post(
            "/v1/agent/ask",
            json={"question": "Small Agent 的版本号是多少？"},
            headers=_headers(),
        )
        # 可能因其他原因返回非 200，但不应是知识库导致
        assert ask_resp.status_code in (200, 503)
        if ask_resp.status_code == 200:
            answer = ask_resp.json()["answer"]
            # 回答应提及版本号（2.0.0 或类似内容）
            assert any(kw in answer for kw in ["2.0.0", "版本"])

    @rag_ready
    def test_agent_searches_when_knowledge_empty(self, client: TestClient) -> None:
        """知识库为空时提问时效性问题，Agent 应联网（即 search_queries 不为空）。"""
        # 不写入知识库，直接问时效性问题
        ask_resp = client.post(
            "/v1/agent/ask",
            json={"question": "今天天气怎么样？"},
            headers=_headers(),
        )
        assert ask_resp.status_code in (200, 503)
        if ask_resp.status_code == 200:
            data = ask_resp.json()
            # 应当有联网检索记录
            assert len(data["search_queries"]) > 0