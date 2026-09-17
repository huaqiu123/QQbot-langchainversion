"""T13: V1 兼容性测试 —— 无知识库配置时行为不变。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


# =================================================================== #
# 辅助：临时覆盖配置
# =================================================================== #


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# =================================================================== #
# /health —— 无知识库配置
# =================================================================== #


class TestHealthWithoutKnowledge:
    """未配置 EMBEDDING_API_KEY 时，/health 仍正常返回。"""

    def test_health_returns_ok_or_degraded(self, client: TestClient) -> None:
        """/health 不依赖知识库，始终应返回可解析的状态。"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert isinstance(data["knowledge_available"], bool)
        assert isinstance(data["knowledge_chunks"], int)

    def test_knowledge_not_available_when_no_embedding_key(
        self, client: TestClient
    ) -> None:
        """未配置 Embedding Key 时 knowledge_available 应为 False。"""
        resp = client.get("/health")
        data = resp.json()
        # 注：如果测试环境中 .env 配了 EMBEDDING_API_KEY，这个测试会跳过
        settings = get_settings()
        if not settings.embedding_api_key:
            assert data["knowledge_available"] is False
            assert data["knowledge_chunks"] == 0


# =================================================================== #
# /knowledge/ingest —— 无知识库配置返回 503
# =================================================================== #


class TestKnowledgeEndpointsDegraded:
    """未配置 EMBEDDING_API_KEY 时，知识库接口返回 503。"""

    def test_ingest_returns_503_when_knowledge_unavailable(
        self, client: TestClient
    ) -> None:
        """知识库不可用时写入应返回 503。"""
        settings = get_settings()
        if settings.embedding_api_key:
            pytest.skip("EMBEDDING_API_KEY 已配置，跳过降级测试")

        resp = client.post(
            "/knowledge/ingest",
            json={"content": "测试内容"},
        )
        assert resp.status_code == 503
        assert "未就绪" in resp.json()["detail"]

    def test_status_returns_available_false_when_no_key(
        self, client: TestClient
    ) -> None:
        """未配 Key 时 /knowledge/status 返回 available=False。"""
        settings = get_settings()
        if settings.embedding_api_key:
            pytest.skip("EMBEDDING_API_KEY 已配置，跳过降级测试")

        resp = client.get("/knowledge/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["chunk_count"] == 0


# =================================================================== #
# /v1/agent/ask —— 兼容 V1 行为
# =================================================================== #


class TestAskCompat:
    """核心问答接口不受知识库影响。"""

    def test_ask_requires_no_embedding_key(self, client: TestClient) -> None:
        """无 Embedding Key 时 /v1/agent/ask 正常可用（可能返回 503 或 502 但因其
        他原因，而非知识库）。至少不应返回 500。"""
        settings = get_settings()
        headers = {}
        if settings.agent_api_key:
            headers["X-API-Key"] = settings.agent_api_key

        resp = client.post(
            "/v1/agent/ask",
            json={"question": "你好"},
            headers=headers,
        )
        # 可能因为 DEEPSEEK_API_KEY 未配置而 503，但不会是 500
        assert resp.status_code in (200, 422, 503)
        if resp.status_code == 503:
            assert "Agent 未就绪" in resp.json()["detail"]