"""请求 / 响应数据模型 —— 对外接口契约的唯一真实来源。

这些模型同时承担三个职责：

1. **请求校验**：FastAPI 依据类型标注自动校验入参，不合法时返回 422；
2. **响应序列化**：路由上的 ``response_model`` 决定实际下发的 JSON 结构；
3. **文档生成**：``Field(description=...)`` 会渲染到 Swagger 文档（``/docs``）。

因此**对外字段的增删改一律在这里进行**。业务层（`agent.py`）使用的是内部结构
`AgentResult`，两者在 `main.py` 中显式映射，避免内部实现细节泄漏到接口上 ——
将来 `AgentResult` 加字段不会影响对外契约的稳定性。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """一条对话历史消息。

    Attributes:
        role: 消息角色，仅接受 ``system`` / ``user`` / ``assistant``。
            实际组装时会**忽略** ``system``，因为系统提示词由 Agent 内部统一注入。
        content: 消息正文。
    """

    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""


class AskRequest(BaseModel):
    """`/v1/agent/ask` 与 `/v1/agent/stream` 共用的请求体。

    Attributes:
        question: 用户问题，必填且不能为空字符串（为空时框架自动返回 422）。
        session_id: 会话标识。本轮仅透传、不做持久化，为后续多轮会话预留。
        history: 历史消息列表，由**调用方**维护并在每次请求时回传。
        use_search: 是否允许联网检索。``None``（缺省）表示采用服务端默认（允许）；
            显式传 ``False`` 时切到不带工具的图，Agent 退化为纯对话。
        extra_context: 附加到问题末尾的补充上下文，用于注入调用方已知的信息。
    """

    question: str = Field(..., min_length=1, description="用户问题")
    session_id: Optional[str] = Field(default=None, description="会话标识，本轮仅透传不持久化")
    history: List[ChatMessage] = Field(default_factory=list, description="历史消息，由调用方维护")
    use_search: Optional[bool] = Field(default=None, description="是否允许联网；None 表示用服务端默认（允许）")
    extra_context: Optional[str] = Field(default=None, description="附加到问题后的补充上下文")


class Source(BaseModel):
    """一条引用来源。

    序号由列表下标决定：``sources[0]`` 对应正文里的 ``[1]``。

    Attributes:
        title: 结果标题。
        url: 结果链接，同时作为去重键。
        snippet: 结果摘要。
    """

    title: str = ""
    url: str = ""
    snippet: str = ""


class AskResponse(BaseModel):
    """`/v1/agent/ask` 的响应体。

    Attributes:
        answer: 最终回答正文，内部含 ``[1] [2]`` 形式的引用标注。
        sources: 检索命中的来源列表，序号与 `answer` 中的标注一一对应。
        search_queries: 本次问答实际执行过的检索关键词，按执行顺序排列。
            可能包含重复项（同一关键词被检索多次会多次记录）。
        iterations: Agent 的模型调用轮次。轮次大于 1 说明发生过工具调用。
        model: 实际使用的模型名。
        usage: token 用量，形如 ``{"prompt_tokens": .., "completion_tokens": ..,
            "total_tokens": ..}``；模型未返回用量信息时为 ``None``。
    """

    answer: str = ""
    sources: List[Source] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    iterations: int = 0
    model: str = ""
    usage: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """`/health` 的响应体，用于区分「进程活着」与「服务能干活」。

    ``status`` 与后两个布尔字段构成两层诊断：``llm_configured`` 反映**配置层**
    问题（Key 没填），``graph_ready`` 反映**运行层**问题（图编译失败）。
    两者都为 ``True`` 时 ``status`` 才是 ``ok``。

    Attributes:
        status: ``ok`` 表示 Agent 已就绪；``degraded`` 表示进程活着但不可用。
        model: 配置中的模型名。
        search_provider: 实际生效的搜索源名称；lifespan 未执行时为空字符串。
        agent_impl: 实际使用的 Agent 实现（``create_agent`` 或 ``create_react_agent``）。
        graph_ready: Agent 图是否编译成功。
        llm_configured: 是否配置了 DeepSeek API Key。
    """

    status: str = "ok"
    model: str = ""
    search_provider: str = ""
    agent_impl: str = ""
    graph_ready: bool = False
    llm_configured: bool = False
    knowledge_available: bool = False
    knowledge_chunks: int = 0


class KnowledgeIngestRequest(BaseModel):
    """`POST /knowledge/ingest` 请求体。

    Attributes:
        content: 要写入知识库的文本内容。
        metadata: 附加元数据（可选），如 ``{"group_id": "123", "sender_id": "user_001"}``。
    """

    content: str = Field(..., min_length=1, description="知识文本")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class KnowledgeIngestResponse(BaseModel):
    """`POST /knowledge/ingest` 响应体。

    Attributes:
        chunk_count: 本次写入的分块数。
        total_chunks: 知识库当前总块数。
    """

    chunk_count: int = 0
    total_chunks: int = 0


class KnowledgeStatusResponse(BaseModel):
    """`GET /knowledge/status` 响应体。

    Attributes:
        available: 知识库是否可用（配置了 Embedding API Key）。
        chunk_count: 知识库当前总块数。
    """

    available: bool = False
    chunk_count: int = 0