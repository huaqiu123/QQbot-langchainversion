"""搜索 Provider 层：关键词 -> 结构化 `SearchResult`。

**刻意不依赖 LangChain**，保持「可独立替换、可独立单测」。上游 `tools.py` 才负责
把这里的产出包装成 LangChain Tool。

**错误处理约定**：所有 Provider 的 ``search()`` 内部吞掉网络异常并返回空列表，
**绝不向上抛异常**。原因是单次检索失败不应该炸掉整张 Agent 图 —— 上层拿到空结果后
会告诉模型「未检索到」，由模型决定是换关键词重试还是如实告知用户。
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

#: Tavily 检索端点。
TAVILY_URL = "https://api.tavily.com/search"
#: Serper（Google 结果）检索端点。
SERPER_URL = "https://google.serper.dev/search"


@dataclass
class SearchResult:
    """单条检索结果（框架无关的中间表示）。

    Attributes:
        title: 结果标题。
        url: 结果链接，同时作为去重键。
        snippet: 结果摘要。
    """

    title: str
    url: str
    snippet: str

    def to_dict(self) -> Dict[str, str]:
        """转换为普通字典。

        用于写入 Tool 的 ``artifact``（必须可被序列化）以及最终响应体。

        Returns:
            含 ``title`` / ``url`` / ``snippet`` 三个键的字典。
        """
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchProvider(ABC):
    """搜索源抽象基类。

    所有实现必须遵守「失败返回空列表、不抛异常」的约定，
    并提供一个可读的 `name` 用于日志与 ``/health`` 展示。

    Attributes:
        name: 搜索源标识，如 ``duckduckgo`` / ``tavily`` / ``serper``。
    """

    name: str = "base"

    @abstractmethod
    async def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """执行一次检索。

        Args:
            query: 检索关键词。
            top_k: 期望返回的结果条数；``None`` 表示使用实例默认值。

        Returns:
            检索结果列表；失败或无结果时返回空列表（不抛异常）。
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# DuckDuckGo（免 Key，默认）
# --------------------------------------------------------------------------- #


def _load_ddgs() -> Callable[..., Any]:
    """动态加载 ddgs 的 ``DDGS`` 类。

    ``ddgs`` 于 2025 年由 ``duckduckgo-search`` 更名，这里做一次兼容回退，
    以支持两种包名共存的旧环境。

    Returns:
        ``DDGS`` 类对象。

    Raises:
        ImportError: 两个包名都未安装时抛出。
    """
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:  # pragma: no cover
        from duckduckgo_search import DDGS  # type: ignore[no-redef]

        return DDGS


class DuckDuckGoProvider(SearchProvider):
    """基于 ``ddgs`` 元搜索库的免 Key 检索源（默认）。

    ``ddgs`` 可指定具体搜索后端。**duckduckgo / startpage 后端在部分网络环境
    （如国内）不可达**，因此默认按 ``bing -> brave -> duckduckgo`` 的顺序依次尝试，
    命中即返回；全部失败或为空时才返回空列表。

    Attributes:
        name: 固定为 ``"duckduckgo"``。
    """

    name = "duckduckgo"

    def __init__(
        self,
        top_k: int = 5,
        backends: Optional[List[str]] = None,
        timeout: float = 15.0,
    ) -> None:
        """初始化。

        Args:
            top_k: 默认返回条数。
            backends: 后端尝试顺序；``None`` 时使用 ``["bing", "brave", "duckduckgo"]``。
            timeout: 单个后端的超时秒数。
        """
        self._top_k = top_k
        self._backends = backends or ["bing", "brave", "duckduckgo"]
        self._timeout = timeout

    async def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """执行检索。

        ``ddgs`` 是同步阻塞库，这里用 `asyncio.to_thread` 丢到线程池执行，
        避免阻塞事件循环（否则会拖垮整个服务的并发能力）。

        Args:
            query: 检索关键词。
            top_k: 期望返回条数；``None`` 时使用构造时的默认值。

        Returns:
            检索结果列表；全部后端失败时返回空列表。
        """
        limit = top_k or self._top_k
        try:
            # ddgs 是同步阻塞库，放到线程池避免堵塞事件循环
            return await asyncio.to_thread(self._search_sync, query, limit)
        except Exception as exc:  # noqa: BLE001 - 检索失败不应中断问答
            logger.warning("DuckDuckGo 检索失败 query=%r err=%s", query, exc)
            return []

    def _search_sync(self, query: str, limit: int) -> List[SearchResult]:
        """在多个 ddgs 后端之间依次降级检索（同步实现，运行于线程池）。

        Args:
            query: 检索关键词。
            limit: 返回条数上限。

        Returns:
            第一个命中非空结果的后端所返回的结果；全部失败时返回空列表。
        """
        ddgs_cls = _load_ddgs()
        last_error: Optional[Exception] = None

        for backend in self._backends:
            try:
                with ddgs_cls(timeout=self._timeout) as ddgs:
                    items = list(ddgs.text(query, max_results=limit, backend=backend))
            except Exception as exc:  # noqa: BLE001 - 换下一个后端继续试
                last_error = exc
                logger.debug("ddgs 后端 %s 失败：%s", backend, exc)
                continue

            if items:
                logger.debug("ddgs 命中后端 %s，返回 %d 条", backend, len(items))
                return [self._to_result(item) for item in items]

        if last_error is not None:
            logger.warning(
                "ddgs 全部后端（%s）均未取到结果，最后一次错误：%s",
                ",".join(self._backends),
                last_error,
            )
        return []

    @staticmethod
    def _to_result(item: Dict[str, Any]) -> SearchResult:
        """把 ddgs 的原始条目转成 `SearchResult`。

        ddgs 不同后端返回的字段名不完全一致（链接可能是 ``href`` 或 ``url``），
        这里做兼容取值。

        Args:
            item: ddgs 返回的单个结果字典。

        Returns:
            归一化后的检索结果。
        """
        return SearchResult(
            title=item.get("title") or "",
            url=item.get("href") or item.get("url") or "",
            snippet=item.get("body") or "",
        )


# --------------------------------------------------------------------------- #
# Tavily
# --------------------------------------------------------------------------- #


class TavilyProvider(SearchProvider):
    """调用 Tavily HTTP API 的检索源（需 API Key）。

    相比免 Key 的 ddgs，Tavily 返回的是经过清洗的正文摘要，质量更稳定，
    适合对检索结果质量要求较高的场景。

    Attributes:
        name: 固定为 ``"tavily"``。
    """

    name = "tavily"

    def __init__(self, api_key: str, top_k: int = 5, timeout: float = 30.0) -> None:
        """初始化。

        Args:
            api_key: Tavily API Key。
            top_k: 默认返回条数。
            timeout: 单次请求超时秒数。
        """
        self._api_key = api_key
        self._top_k = top_k
        self._timeout = timeout

    async def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """调用 Tavily ``/search`` 接口执行检索。

        Args:
            query: 检索关键词。
            top_k: 期望返回条数；``None`` 时使用构造时的默认值。

        Returns:
            检索结果列表；请求失败时返回空列表。
        """
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": top_k or self._top_k,
            "search_depth": "basic",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(TAVILY_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily 检索失败 query=%r err=%s", query, exc)
            return []

        return [
            SearchResult(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
            )
            for item in data.get("results", []) or []
        ]


# --------------------------------------------------------------------------- #
# Serper（Google 结果）
# --------------------------------------------------------------------------- #


class SerperProvider(SearchProvider):
    """调用 Serper API 获取 Google 搜索结果的检索源（需 API Key）。

    Attributes:
        name: 固定为 ``"serper"``。
    """

    name = "serper"

    def __init__(self, api_key: str, top_k: int = 5, timeout: float = 30.0) -> None:
        """初始化。

        Args:
            api_key: Serper API Key。
            top_k: 默认返回条数。
            timeout: 单次请求超时秒数。
        """
        self._api_key = api_key
        self._top_k = top_k
        self._timeout = timeout

    async def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """调用 Serper ``/search`` 接口执行检索。

        Args:
            query: 检索关键词。
            top_k: 期望返回条数；``None`` 时使用构造时的默认值。

        Returns:
            取自响应 ``organic`` 字段的检索结果列表；请求失败时返回空列表。
        """
        headers = {"X-API-KEY": self._api_key, "Content-Type": "application/json"}
        body = {"q": query, "num": top_k or self._top_k}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(SERPER_URL, json=body, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Serper 检索失败 query=%r err=%s", query, exc)
            return []

        return [
            SearchResult(
                title=item.get("title") or "",
                url=item.get("link") or "",
                snippet=item.get("snippet") or "",
            )
            for item in data.get("organic", []) or []
        ]


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #


def build_search_provider(settings: Settings) -> SearchProvider:
    """根据配置构造搜索源实例。

    配置了 Tavily / Serper 但缺少对应 Key 时，会降级为 ddgs 并打警告日志，
    保证服务始终可用（降级判断逻辑在 `Settings.effective_search_provider`）。

    Args:
        settings: 全局配置。

    Returns:
        一个已就绪的 `SearchProvider` 实例。
    """
    requested = (settings.search_provider or "").strip().lower()
    effective = settings.effective_search_provider()

    if requested and requested != effective:
        logger.warning(
            "SEARCH_PROVIDER=%s 但缺少对应 API Key，已降级为 %s", requested, effective
        )

    if effective == "tavily":
        logger.info("搜索源：Tavily")
        return TavilyProvider(settings.tavily_api_key, settings.search_top_k, settings.search_timeout)
    if effective == "serper":
        logger.info("搜索源：Serper")
        return SerperProvider(settings.serper_api_key, settings.search_top_k, settings.search_timeout)

    backends = settings.ddgs_backend_list()
    logger.info("搜索源：ddgs（后端顺序 %s）", " -> ".join(backends))
    return DuckDuckGoProvider(settings.search_top_k, backends, settings.search_timeout)
