"""FastAPI 服务层：HTTP 路由、鉴权、异常映射、应用生命周期。

这是整个项目**最外层**的模块：它不知道 LangChain、不知道搜索、不知道 DeepSeek，
只知道「HTTP 请求进来 -> 调用 `Agent` -> HTTP 响应出去」。所有业务逻辑都在
`agent.py` 及其下层，依赖方向单向，本模块不允许被下层反向导入。

**错误码约定**（调用方据此区分「改请求」还是「重试」）：

=================  ====================================================
状态码              含义
=================  ====================================================
``401``            鉴权失败（配置了 ``AGENT_API_KEY`` 但请求未带或带错）
``422``            请求体校验失败（由 FastAPI 依据 `schemas` 自动返回）
``503``            依赖未就绪（通常是 API Key 未配置，``detail`` 含具体原因）
``504``            总耗时超过 ``REQUEST_TIMEOUT``
``502``            上游执行失败（模型或搜索侧异常）
=================  ====================================================

**三个贯穿全文件的设计原则**：

1. **带病启动**：Agent 构造失败不阻断服务，改为在 ``/health`` 暴露、在业务接口用
   503 说明原因，便于排障；
2. **状态分层**：只读配置一律走 `get_settings()`（有缓存、不依赖初始化顺序），
   只有真正的运行期对象（Agent 图、搜索源）才放 ``app.state``；
3. **异常不外泄**：所有内部异常都在这里收敛成结构化的 HTTP 响应并记录日志。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import __version__
from .agent import Agent
from .config import Settings, get_settings
from .database import MessageStore
from .knowledge import KnowledgeBase
from .llm import build_embeddings
from .schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeStatusResponse,
)
from .search import build_search_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("small_agent")


def _configure_tracing(settings: Settings) -> None:
    """按配置启用 LangSmith 链路追踪。

    LangSmith 的采集逻辑是在 LangChain 内部**直接读取环境变量**的，不经过我们的
    `Settings` 对象，因此必须把配置值倒写回 ``os.environ`` 才能生效。

    使用 ``setdefault`` 而非直接赋值，是为了让真实存在的进程环境变量优先于
    ``.env`` 文件 —— 这样部署时可以用 ``docker run -e`` 临时覆盖。

    Args:
        settings: 全局配置。
    """
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        logger.info("LangSmith 追踪已启用 project=%s", settings.langsmith_project)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期钩子：``yield`` 之前启动，之后关闭。

    这是整个服务装配**唯一**的地方。装配放在这里（而非每次请求）的原因：
    `Agent.__init__` 会构造模型客户端并编译两张 LangGraph 图，属于重量级操作，
    必须做成进程级单例，否则每条消息都要重编译一遍图。

    启动段用 ``try/except`` 包住 Agent 构造，是刻意的**带病启动**设计：
    缺少 API Key 时服务照常起来，``/health`` 返回 ``degraded``，业务接口返回
    503 并附带失败原因，而不是让整个进程启动失败只能去翻控制台日志。

    Args:
        app: FastAPI 应用实例，用于挂载 ``app.state``。

    Yields:
        None。``yield`` 即为「服务已就绪」的分界点。
    """
    settings = get_settings()
    _configure_tracing(settings)

    provider = build_search_provider(settings)
    app.state.settings = settings
    app.state.search_provider = provider

    # V2：知识库初始化
    knowledge_base = None
    message_store = None
    if embedding_func := build_embeddings(settings):
        try:
            message_store = MessageStore(settings.knowledge_db_path)
            knowledge_base = KnowledgeBase(
                persist_dir=settings.chroma_persist_dir,
                embedding_func=embedding_func,
                similarity_threshold=settings.knowledge_similarity_threshold,
                search_top_k=settings.knowledge_search_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("知识库初始化失败：%s", exc)
    app.state.knowledge_base = knowledge_base
    app.state.message_store = message_store

    try:
        app.state.agent = Agent(settings, provider, knowledge_base)
        app.state.agent_error = None
        logger.info(
            "Agent 就绪 | 实现=%s | 模型=%s | 搜索=%s | 工具数=%d",
            app.state.agent.impl,
            settings.deepseek_model,
            provider.name,
            len(app.state.agent.tools),
        )
    except Exception as exc:  # noqa: BLE001 - 允许无 Key 启动，便于查看 /health
        app.state.agent = None
        app.state.agent_error = str(exc)
        logger.error("Agent 初始化失败：%s", exc)

    yield

    app.state.agent = None


app = FastAPI(
    title="Small Agent Backend",
    version=__version__,
    description="基于 LangChain create_agent + DeepSeek 的可联网问答后端",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """可选的接口鉴权依赖。

    以 ``dependencies=[Depends(require_api_key)]`` 的形式挂在路由上，
    这类依赖的**返回值会被丢弃**，只执行函数体 —— 因此本函数返回 ``None``，
    校验失败时通过抛 `HTTPException` 中断请求。

    行为：``AGENT_API_KEY`` 为空则完全不鉴权（方便本地开发）；一旦配置，
    请求必须携带同名值的 ``X-API-Key`` 头。

    Args:
        x_api_key: 请求头 ``X-API-Key``。参数名不能含连字符，故用 ``alias`` 映射；
            ``default=None`` 使其成为可选参数（缺失时不报 422 而是走鉴权分支）。

    Raises:
        HTTPException: 401，当服务端配置了 Key 但请求未带或值不匹配时。

    Note:
        这里是普通字符串比较，非时序安全。内网自用可接受；若接口要暴露到公网，
        应改用 ``secrets.compare_digest`` 并配合限流。
    """
    settings = get_settings()
    if settings.agent_api_key and x_api_key != settings.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key 无效或缺失",
        )


def _get_agent(request: Request) -> Agent:
    """取出已装配的 Agent 实例，未就绪时统一转成 503。

    为什么用两层 ``getattr`` 而不是直接访问 ``app.state.agent``：需要同时兼容
    「属性存在但为 None」（Agent 构造失败）和「属性根本不存在」（lifespan 未执行）
    两种情况，两者都应该返回 503 而不是抛 `AttributeError`。

    Args:
        request: 当前请求。

    Returns:
        已就绪的 `Agent` 实例。

    Raises:
        HTTPException: 503，``detail`` 中会带上启动时捕获的失败原因，便于排障。
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent 未就绪：{getattr(request.app.state, 'agent_error', '未知原因')}",
        )
    return agent


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #


@app.get("/", tags=["meta"])
async def root() -> dict:
    """服务根路径，用于快速确认进程存活并找到文档入口。

    Returns:
        含服务名、版本号与 Swagger 文档地址的字典。
    """
    return {"service": "small-agent", "version": __version__, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    """健康检查，用于区分「进程活着」与「服务能干活」。

    刻意不依赖 lifespan：配置走 `get_settings()`，运行期对象用 ``getattr`` 兜底，
    因此即使生命周期钩子没跑（例如测试场景）也能正常返回，而不是抛异常。

    Args:
        request: 当前请求。

    Returns:
        `HealthResponse`。``status`` 为 ``ok``（Agent 已就绪）或
        ``degraded``（进程活着但不可用）；``llm_configured`` / ``graph_ready``
        两个布尔字段分别对应配置层与运行层，便于定位问题出在哪一层。
    """
    settings = get_settings()
    agent = getattr(request.app.state, "agent", None)
    provider = getattr(request.app.state, "search_provider", None)
    kb = getattr(request.app.state, "knowledge_base", None)

    return HealthResponse(
        status="ok" if agent else "degraded",
        model=settings.deepseek_model,
        search_provider=provider.name if provider else "",
        agent_impl=agent.impl if agent else "",
        graph_ready=agent is not None,
        llm_configured=settings.is_llm_configured(),
        knowledge_available=(kb is not None and kb.available) if kb else False,
        knowledge_chunks=kb.get_chunk_count() if (kb and kb.available) else 0,
    )


@app.post(
    "/v1/agent/ask",
    response_model=AskResponse,
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """非流式问答接口，QQ 机器人的主调用入口。

    内部可能发生多轮「模型 -> 检索 -> 模型」循环，本接口会等待全部结束
    再一次性返回，因此耗时可能达数十秒。

    Args:
        payload: 请求体，由 FastAPI 依据 `AskRequest` 自动校验。
        request: 当前请求。

    Returns:
        `AskResponse`，含回答正文、引用来源、检索关键词、轮次与 token 用量。

    Raises:
        HTTPException: 503 Agent 未就绪；504 总耗时超过 ``REQUEST_TIMEOUT``；
            502 上游（模型或搜索）执行失败。

    Note:
        ``except`` 子句的顺序不可调换 —— ``asyncio.TimeoutError`` 必须写在
        ``except Exception`` 之前，否则超时会落进通用分支而被误报为 502，
        调用方就无法区分「该重试」和「真出错了」。
    """
    agent = _get_agent(request)
    settings = get_settings()

    try:
        result = await asyncio.wait_for(
            agent.run(
                question=payload.question,
                history=payload.history,
                use_search=payload.use_search,
                extra_context=payload.extra_context,
            ),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"处理超时（>{settings.request_timeout}s）",
        ) from None
    except Exception as exc:  # noqa: BLE001
        logger.exception("问答失败")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent 执行失败：{exc}",
        ) from exc

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        search_queries=result.search_queries,
        iterations=result.iterations,
        model=result.model,
        usage=result.usage,
    )


@app.post(
    "/v1/agent/stream",
    tags=["agent"],
    dependencies=[Depends(require_api_key)],
)
async def stream(payload: AskRequest, request: Request) -> StreamingResponse:
    """SSE 流式问答接口，用于前端实时展示或调试 Agent 的决策过程。

    事件类型与字段见 `Agent.stream` 的说明（``delta`` / ``search`` / ``done`` /
    ``error``）。报文格式为标准 SSE：``data: {json}\\n\\n``，必须以**两个换行**结尾。

    Args:
        payload: 请求体，与非流式接口共用 `AskRequest`。
        request: 当前请求。

    Returns:
        `StreamingResponse`，``media_type`` 为 ``text/event-stream``。

    Raises:
        HTTPException: 503 Agent 未就绪（此时尚未开始流式输出，仍可返回正常错误码）。

    Note:
        进入流式输出后 HTTP 状态码已经发出（200），**无法再改成 500**，因此
        生成器内部捕获异常并转为一条 ``error`` 事件告知客户端。这是流式接口
        与普通接口在错误处理上的本质区别。
    """
    agent = _get_agent(request)

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in agent.stream(
                question=payload.question,
                history=payload.history,
                use_search=payload.use_search,
                extra_context=payload.extra_context,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式问答失败")
            data = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run("app.main:app", host=_settings.host, port=_settings.port, reload=False)


# --------------------------------------------------------------------------- #
# 知识库路由（V2）
# --------------------------------------------------------------------------- #


@app.post(
    "/knowledge/ingest",
    response_model=KnowledgeIngestResponse,
    tags=["knowledge"],
    dependencies=[Depends(require_api_key)],
)
async def ingest(payload: KnowledgeIngestRequest, request: Request) -> KnowledgeIngestResponse:
    """写入文本到知识库。

    将文本分块后写入 Chroma 向量库与 SQLite 原文存储。
    Embedding API Key 未配置时返回 503。

    Args:
        payload: 知识文本与元数据。
        request: 当前请求。

    Returns:
        写入的分块数与总块数。
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)
    message_store = getattr(request.app.state, "message_store", None)

    if not knowledge_base or not knowledge_base.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="知识库未就绪（未配置 EMBEDDING_API_KEY 或初始化失败）",
        )

    content = payload.content
    metadata = payload.metadata
    chunk_size = get_settings().knowledge_chunk_size
    overlap = get_settings().knowledge_chunk_overlap

    chunks = split_text(content, chunk_size, overlap)
    if not chunks:
        return KnowledgeIngestResponse(chunk_count=0, total_chunks=0)

    # 写入 SQLite
    source_id = None
    if message_store:
        source_id = message_store.insert(content, metadata)

    # 写入 Chroma，携带 source_id
    metadatas = [{**metadata, "source_id": source_id} if source_id else metadata] * len(chunks)
    knowledge_base.add_texts(chunks, metadatas)

    total = knowledge_base.get_chunk_count()
    return KnowledgeIngestResponse(chunk_count=len(chunks), total_chunks=total)


@app.get(
    "/knowledge/status",
    response_model=KnowledgeStatusResponse,
    tags=["knowledge"],
    dependencies=[Depends(require_api_key)],
)
async def knowledge_status(request: Request) -> KnowledgeStatusResponse:
    """查询知识库状态。

    Returns:
        可用状态与总块数。
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)
    if not knowledge_base or not knowledge_base.available:
        return KnowledgeStatusResponse(available=False, chunk_count=0)

    try:
        chunk_count = knowledge_base.get_chunk_count()
    except Exception:
        chunk_count = 0
    return KnowledgeStatusResponse(available=True, chunk_count=chunk_count)


def split_text(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    """将文本分割成重叠块。

    Args:
        text: 输入文本。
        chunk_size: 每块最大字符数。
        overlap: 相邻块重叠字符数。

    Returns:
        文本块列表。输入为空时返回空列表。
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