"""Agent 核心：组图（模型 + 工具 + 提示词 + 中间件）、执行、事件流适配。

**直接依赖 LangChain 的 `create_agent`**，不自写 ReAct 循环、不手拼 tool 消息 ——
工具调用循环、消息裁剪、终止条件全部交给框架托管。

职责边界：

- 本模块**不碰模型超参**（由 `llm.py` 负责）；
- 本模块**不碰 HTTP**（由 `main.py` 负责），对外暴露的是普通异步函数与异步生成器。

对外只有两个入口：`Agent.run`（一次性返回）与 `Agent.stream`（事件流），
两者共享同一套输入组装与结果解析逻辑。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from .config import Settings
from .llm import build_chat_model, model_name, validate_model
from .prompts import build_system_prompt
from .schemas import ChatMessage, Source
from .search import SearchProvider
from .tools import build_tools

if TYPE_CHECKING:
    from .knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

try:  # LangGraph 的循环上限异常
    from langgraph.errors import GraphRecursionError
except ImportError:  # pragma: no cover

    class GraphRecursionError(Exception):  # type: ignore[no-redef]
        """langgraph 缺失时的占位，保证本模块可导入。"""


#: 触及 recursion_limit 时的兜底回答。设计要点是「承认失败 + 给出可行建议」，
#: 而不是返回一句无信息量的报错。
RECURSION_ANSWER = (
    "抱歉，这个问题需要检索的轮次过多，暂时无法给出可靠答案。"
    "可以把问题拆得更具体一些再问我。"
)


# --------------------------------------------------------------------------- #
# create_agent 解析：优先 LangChain 1.x 原生 API，回退 LangGraph 预置实现
# --------------------------------------------------------------------------- #


def _resolve_create_agent() -> Tuple[Any, str]:
    """解析可用的 `create_agent` 实现。

    LangChain 1.x 提供 ``langchain.agents.create_agent``，其底层就是编译好的
    LangGraph 图；更早的版本则只有 ``langgraph.prebuilt.create_react_agent``。
    两者的入参名不同（前者 ``system_prompt``、后者 ``prompt``），因此需要把
    「用哪个函数」和「函数名标识」一起返回。

    Returns:
        二元组 ``(构造函数, 实现标识)``，标识为 ``"create_agent"`` 或
        ``"create_react_agent"``，用于日志与 ``/health`` 的 ``agent_impl`` 字段。
    """
    try:
        from langchain.agents import create_agent  # LangChain >= 1.0

        return create_agent, "create_agent"
    except ImportError:  # pragma: no cover
        from langgraph.prebuilt import create_react_agent

        return create_react_agent, "create_react_agent"


def _build_middleware() -> List[Any]:
    """构造 Agent 中间件列表。

    当前只挂一个**工具异常兜底中间件**：工具抛异常时返回一条说明性的
    ``ToolMessage``，让模型据此如实告知用户，而不是让异常穿透整张图导致请求 500。

    若当前 LangChain 版本没有 middleware API，则返回空列表静默降级 ——
    主流程仍可运行，只是失去这层兜底（`tools.py` 内部也有异常收敛作为第二道防线）。

    Returns:
        中间件列表，可能为空。
    """
    try:
        from langchain.agents.middleware import wrap_tool_call
    except ImportError:  # pragma: no cover
        logger.debug("当前 LangChain 版本无 middleware API，跳过工具异常中间件")
        return []

    @wrap_tool_call
    async def handle_tool_errors(request: Any, handler: Any) -> Any:
        """包装工具调用，把异常转成模型可读的 ToolMessage。"""
        try:
            return await handler(request)
        except Exception as exc:  # noqa: BLE001
            tool_call = getattr(request, "tool_call", None) or {}
            logger.warning("工具执行失败 name=%s err=%s", tool_call.get("name"), exc)
            return ToolMessage(
                content=(
                    f"检索工具执行失败（{exc}）。"
                    "请基于已有信息作答，或如实告知用户暂时无法联网检索。"
                ),
                tool_call_id=tool_call.get("id", ""),
            )

    return [handle_tool_errors]


# --------------------------------------------------------------------------- #
# 消息处理辅助
# --------------------------------------------------------------------------- #


def _text_of(content: Any) -> str:
    """从消息的 ``content`` 字段提取纯文本。

    不同厂商 / 不同版本的模型返回的 content 形态不一致：可能是 ``str``，
    也可能是 content blocks 列表（``[{"type": "text", "text": "..."}]``）。
    这里统一归一化成字符串。

    Args:
        content: 消息的 ``content`` 字段，类型不定。

    Returns:
        提取出的文本（已 strip）；无法识别时返回空字符串。
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return ""


def _read_artifact(message: ToolMessage) -> Tuple[str, List[Source]]:
    """从 ``ToolMessage.artifact`` 取回结构化检索来源。

    这是 `tools.web_search` 用 ``content_and_artifact`` 拆出去的那一半数据，
    模型看不到它，因此可以安全地做二次处理而不污染上下文。

    Args:
        message: 一条工具消息。

    Returns:
        二元组 ``(检索关键词, 来源列表)``；artifact 缺失或格式异常时返回
        ``("", [])``，不会抛异常。
    """
    artifact = getattr(message, "artifact", None)
    if not isinstance(artifact, dict):
        return "", []

    query = str(artifact.get("query") or "")
    sources: List[Source] = []
    for item in artifact.get("sources") or []:
        if isinstance(item, dict):
            sources.append(
                Source(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    snippet=str(item.get("snippet") or ""),
                )
            )
    return query, sources


def _iter_updates(payload: Any) -> List[Tuple[str, Any]]:
    """把 ``stream_mode="updates"`` 的载荷归一化成 ``(节点名, 状态更新)`` 列表。

    正常情况是 ``dict``；LangGraph 在带子图等场景下可能产出其他形态，
    此处对非字典载荷返回空列表，避免上层遍历时报错。

    Args:
        payload: ``astream`` 在 updates 模式下产出的原始载荷。

    Returns:
        ``(节点名, 更新内容)`` 二元组列表。
    """
    if isinstance(payload, dict):
        return list(payload.items())
    return []


# --------------------------------------------------------------------------- #
# 结果对象
# --------------------------------------------------------------------------- #


@dataclass
class AgentResult:
    """一次问答的完整结果。

    这是**内部结构**，不直接对外暴露 —— `main.py` 会把它显式映射成
    `schemas.AskResponse`，以保证对外契约的稳定性。

    Attributes:
        answer: 最终回答正文（含 ``[1] [2]`` 引用标注）。
        sources: 去重后的引用来源，顺序即正文中的编号顺序。
        search_queries: 实际执行过的检索关键词，按执行顺序排列。
        iterations: 模型调用轮次；大于 1 说明发生过工具调用。
        model: 实际使用的模型名。
        usage: token 用量统计；模型未返回时为 ``None``。
    """

    answer: str
    sources: List[Source] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    iterations: int = 0
    model: str = ""
    usage: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    """把模型、工具、提示词组装成一张可执行的 Agent 图，并提供两种执行方式。

    实例是**重量级且线程安全可复用**的：构造时会编译两张图，因此应该作为进程级
    单例创建（见 `main.lifespan`），不要按请求创建。

    Attributes:
        settings: 全局配置。
        search_provider: 搜索源实例。
        model_name: 模型名，用于回填响应的 ``model`` 字段。
        model: 聊天模型实例。
        tools: 绑定的工具列表（可能为空）。
        impl: 实际使用的 Agent 实现标识。
        graph: 带工具的图（默认执行路径）。
        graph_plain: 不带工具的图；``use_search=False`` 时走这张，Agent 退化为纯对话。
    """

    def __init__(
        self,
        settings: Settings,
        search_provider: SearchProvider,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        """构造 Agent 并编译图。

        编译两张图的代价很低（模型与工具实例是复用的），换来的是
        「禁用检索」可以由调用方按请求控制，而不需要靠提示词劝模型别用工具。

        Args:
            settings: 全局配置。
            search_provider: 已就绪的搜索源实例。
            knowledge_base: 知识库实例（可选），传入后 Agent 会在回答前先查知识库。

        Raises:
            RuntimeError: 未配置 ``DEEPSEEK_API_KEY`` 时由 `build_chat_model` 抛出。
        """
        self.settings = settings
        self.search_provider = search_provider
        self.knowledge_base = knowledge_base
        self.model_name = model_name(settings)

        self.model = build_chat_model(settings)
        self.tools = build_tools(settings, search_provider)
        validate_model(settings, has_tools=bool(self.tools))

        self._create_fn, self.impl = _resolve_create_agent()
        self.graph = self._compile(self.tools)
        # use_search=False 时走这张不带工具的图，Agent 退化为纯对话
        self.graph_plain = self._compile([]) if self.tools else self.graph

    # ---------------- 组图 ----------------

    def _compile(self, tools: List[Any]) -> Any:
        """编译一张 Agent 图。

        Args:
            tools: 要绑定的工具列表，空列表表示纯对话模式。

        Returns:
            编译好的 LangGraph 图对象，可直接 ``ainvoke`` / ``astream``。
        """
        prompt = build_system_prompt()
        if self.impl == "create_agent":
            return self._create_fn(
                model=self.model,
                tools=tools,
                system_prompt=prompt,
                middleware=_build_middleware(),
            )
        return self._create_fn(model=self.model, tools=tools, prompt=prompt)

    def _pick_graph(self, use_search: Optional[bool]) -> Any:
        """按请求参数选择要执行的图。

        Args:
            use_search: 请求级开关。``False`` 表示显式禁用检索；
                ``None`` 或 ``True`` 都走带工具的图。

        Returns:
            带工具或不带工具的图对象。
        """
        if use_search is False and self.tools:
            return self.graph_plain
        return self.graph

    # ---------------- 知识库检索 ----------------

    async def _knowledge_search(self, question: str) -> str | None:
        """查询知识库，返回格式化后的文本块。知识库不可用或未命中时返回 None。

        Args:
            question: 用户问题。

        Returns:
            格式化知识文本，或 None。
        """
        kb = self.knowledge_base
        if kb is None or not kb.available:
            return None
        try:
            results = kb.search(question)
        except Exception as exc:
            logger.warning("知识库检索失败: %s", exc)
            return None
        if not results:
            return None
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['content']}")
        return "\n".join(lines)

    # ---------------- 输入组装 ----------------

    def _build_messages(
        self,
        question: str,
        history: Optional[List[ChatMessage]],
        extra_context: Optional[str],
        knowledge_text: str | None = None,
    ) -> List[BaseMessage]:
        """把请求参数转换成 LangChain 消息列表。

        历史裁剪在这里做（而不是靠中间件改图状态），原因是这样行为确定、易调试：
        只取最后 ``max_history_messages`` 条，且 ``<= 0`` 时视为不限制。

        Args:
            question: 用户问题。
            history: 历史消息（由调用方维护）。
            extra_context: 附加到问题末尾的补充上下文。
            knowledge_text: 知识库检索结果文本，不为空时注入到提问中。

        Returns:
            转换后的消息列表，最后一条固定为本次提问。

        Note:
            历史中的 ``system`` 角色会被**忽略** —— 系统提示词由 `create_agent`
            通过 ``system_prompt`` 参数统一注入，重复注入会干扰模型。
        """
        messages: List[BaseMessage] = []

        # 历史里的 system 角色会被忽略：system 提示词由 create_agent 统一注入
        items = history or []
        limit = self.settings.max_history_messages
        if limit > 0:
            items = items[-limit:]

        for item in items:
            if item.role == "user":
                messages.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                messages.append(AIMessage(content=item.content))

        # 知识库结果注入
        text = question.strip()
        if knowledge_text:
            text = f"【内部知识库】\n{knowledge_text}\n\n【用户问题】\n{text}"
        if extra_context and extra_context.strip():
            text = f"{text}\n\n[补充上下文]\n{extra_context.strip()}"
        messages.append(HumanMessage(content=text))
        return messages

    # ---------------- 非流式 ----------------

    async def run(
        self,
        question: str,
        history: Optional[List[ChatMessage]] = None,
        use_search: Optional[bool] = None,
        extra_context: Optional[str] = None,
    ) -> AgentResult:
        """执行一次完整问答并返回最终结果。

        图内部可能发生多轮「模型 -> 工具 -> 模型」循环，本方法会一直等到
        模型产出不带 ``tool_calls`` 的最终回答为止。

        Args:
            question: 用户问题。
            history: 历史消息，由调用方维护。
            use_search: 是否允许检索；``False`` 时切到不带工具的图。
            extra_context: 附加到问题末尾的补充上下文。

        Returns:
            聚合后的 `AgentResult`。

        Note:
            触及 ``recursion_limit`` 时不会抛异常，而是返回一条
            `RECURSION_ANSWER` 兜底话术，保证调用方始终能拿到可用响应。
        """
        graph = self._pick_graph(use_search)
        knowledge_text = await self._knowledge_search(question)
        messages = self._build_messages(question, history, extra_context, knowledge_text)
        config = {"recursion_limit": self.settings.recursion_limit}
        started = time.perf_counter()

        try:
            state = await graph.ainvoke({"messages": messages}, config=config)
        except GraphRecursionError:
            logger.warning("达到 recursion_limit=%s，强制收敛", self.settings.recursion_limit)
            return AgentResult(
                answer=RECURSION_ANSWER,
                iterations=self.settings.recursion_limit,
                model=self.model_name,
            )

        out_messages = state.get("messages", []) if isinstance(state, dict) else []
        answer = self._final_answer(out_messages)
        sources, queries = self._collect_sources(out_messages)
        usage = self._collect_usage(out_messages)
        iterations = sum(1 for m in out_messages if isinstance(m, AIMessage))

        logger.info(
            "问答完成 用时=%.2fs 轮次=%d 检索=%d 来源=%d",
            time.perf_counter() - started,
            iterations,
            len(queries),
            len(sources),
        )

        return AgentResult(
            answer=answer,
            sources=sources,
            search_queries=queries,
            iterations=iterations,
            model=self.model_name,
            usage=usage,
        )

    # ---------------- 流式 ----------------

    async def stream(
        self,
        question: str,
        history: Optional[List[ChatMessage]] = None,
        use_search: Optional[bool] = None,
        extra_context: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """以事件流的方式执行问答。

        同时订阅两种流模式，各取所需：

        - ``messages``：模型逐 token 生成的内容，转成 ``delta`` 事件；
        - ``updates``：各节点的状态更新，从中捞出 ``ToolMessage`` 的 artifact，
          转成 ``search`` 事件。

        事件契约（与 `main.py` 的 SSE 输出一致）：

        +------------+--------------------------------------------------------------+
        | ``delta``  | ``{"type", "content"}``，正文增量                            |
        +------------+--------------------------------------------------------------+
        | ``search`` | ``{"type", "query", "hits"}``，发生了一次检索                |
        +------------+--------------------------------------------------------------+
        | ``done``   | ``{"type", "answer", "sources", "search_queries",            |
        |            | "iterations", "usage"}``，正常结束                           |
        +------------+--------------------------------------------------------------+
        | ``error``  | ``{"type", "message"}``，异常终止                            |
        +------------+--------------------------------------------------------------+

        Args:
            question: 用户问题。
            history: 历史消息，由调用方维护。
            use_search: 是否允许检索；``False`` 时切到不带工具的图。
            extra_context: 附加到问题末尾的补充上下文。

        Yields:
            上述四种事件字典之一。正常情况下 ``done`` 一定是最后一个事件。

        Note:
            工具调用轮次的模型输出通常为空字符串，因此这里只推送
            「不带 tool_calls 的纯文本增量」，避免把中间态的碎文本透给前端。
        """
        graph = self._pick_graph(use_search)
        knowledge_text = await self._knowledge_search(question)
        messages = self._build_messages(question, history, extra_context, knowledge_text)
        config = {"recursion_limit": self.settings.recursion_limit}

        answer_parts: List[str] = []
        sources: List[Source] = []
        queries: List[str] = []
        seen: set = set()
        iterations = 0
        usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        usage_found = False

        try:
            async for mode, payload in graph.astream(
                {"messages": messages},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, _meta = payload if isinstance(payload, tuple) else (payload, {})
                    meta = getattr(chunk, "usage_metadata", None)
                    if isinstance(meta, dict):
                        usage_found = True
                        usage["prompt_tokens"] += int(meta.get("input_tokens") or 0)
                        usage["completion_tokens"] += int(meta.get("output_tokens") or 0)
                        usage["total_tokens"] += int(meta.get("total_tokens") or 0)

                    # 只推送「纯文本生成」的增量；带 tool_calls 的轮次没有可展示内容
                    if isinstance(chunk, AIMessageChunk) and not getattr(
                        chunk, "tool_call_chunks", None
                    ):
                        text = _text_of(chunk.content)
                        if text:
                            answer_parts.append(text)
                            yield {"type": "delta", "content": text}

                elif mode == "updates":
                    for _node, update in _iter_updates(payload):
                        if not isinstance(update, dict):
                            continue
                        for message in update.get("messages") or []:
                            if isinstance(message, ToolMessage):
                                query, items = _read_artifact(message)
                                if query:
                                    queries.append(query)
                                for item in items:
                                    key = item.url or item.title
                                    if key and key in seen:
                                        continue
                                    if key:
                                        seen.add(key)
                                    sources.append(item)
                                yield {"type": "search", "query": query, "hits": len(items)}
                            elif isinstance(message, AIMessage) and not getattr(
                                message, "tool_calls", None
                            ):
                                iterations += 1

        except GraphRecursionError:
            logger.warning("流式输出达到 recursion_limit=%s", self.settings.recursion_limit)
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式输出失败")
            yield {"type": "error", "message": str(exc)}
            return

        yield {
            "type": "done",
            "answer": "".join(answer_parts).strip(),
            "sources": [item.model_dump() for item in sources],
            "search_queries": queries,
            "iterations": iterations,
            "usage": usage if usage_found else None,
        }

    # ---------------- 结果解析 ----------------

    @staticmethod
    def _final_answer(messages: List[Any]) -> str:
        """从图输出中取出最终回答。

        从后往前找**第一条不带 ``tool_calls`` 且正文非空的 AIMessage** ——
        带 tool_calls 的是中间决策轮，不是给用户的答案。

        Args:
            messages: 图输出的完整消息列表。

        Returns:
            回答正文；找不到时返回空字符串。
        """
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                text = _text_of(message.content)
                if text:
                    return text
        return ""

    @staticmethod
    def _collect_sources(messages: List[Any]) -> Tuple[List[Source], List[str]]:
        """汇总所有工具消息里的结构化来源与检索关键词。

        按 ``url``（缺失时退化用 ``title``）去重，因为模型可能用不同关键词
        多次检索到同一个页面。

        Args:
            messages: 图输出的完整消息列表。

        Returns:
            二元组 ``(去重后的来源列表, 检索关键词列表)``。
            来源列表的顺序即正文引用编号顺序。
        """
        sources: List[Source] = []
        queries: List[str] = []
        seen: set = set()

        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            query, items = _read_artifact(message)
            if query:
                queries.append(query)
            for item in items:
                key = item.url or item.title
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                sources.append(item)

        return sources, queries

    @staticmethod
    def _collect_usage(messages: List[Any]) -> Optional[Dict[str, Any]]:
        """累加所有模型调用的 token 用量。

        多轮问答会产生多次模型调用，这里把它们加总成一次请求的总开销。

        Args:
            messages: 图输出的完整消息列表。

        Returns:
            ``{"prompt_tokens", "completion_tokens", "total_tokens"}``；
            若所有消息都没有用量信息（模型或 SDK 不支持）则返回 ``None``。
        """
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        found = False

        for message in messages:
            meta = getattr(message, "usage_metadata", None)
            if isinstance(meta, dict):
                found = True
                total["prompt_tokens"] += int(meta.get("input_tokens") or 0)
                total["completion_tokens"] += int(meta.get("output_tokens") or 0)
                total["total_tokens"] += int(meta.get("total_tokens") or 0)

        return total if found else None