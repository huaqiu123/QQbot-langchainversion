"""工具层：把 `SearchResult` 包装成 LangChain Tool。

**关键设计：使用 ``response_format="content_and_artifact"``。** 工具的返回值被拆成
两条路，互不干扰：

==================  ==========================  ==============================
返回值元组位置        去向                          用途
==================  ==========================  ==============================
``content``         ``ToolMessage.content``      **模型可见**，模型据此产出 ``[1] [2]`` 引用
``artifact``        ``ToolMessage.artifact``     **模型不可见**，供程序读取结构化来源
==================  ==========================  ==============================

这样工具只需声明自己的产出，无需感知外部有没有人在收集 —— 相比「往共享收集器里写」
的旁路方案，解耦更彻底。上游 `agent.Agent` 事后遍历图状态中的
``ToolMessage.artifact`` 即可还原结构化来源。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from langchain_core.tools import BaseTool, ToolException, tool

from .config import Settings
from .search import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


def format_results(results: List[SearchResult]) -> str:
    """把检索结果格式化成带编号的文本，供模型阅读。

    **编号顺序与 ``artifact["sources"]`` 的列表顺序严格一致** —— 这是正文中
    ``[1] [2]`` 标注能与响应体 ``sources`` 正确对应的前提，改动此处顺序会破坏引用关系。

    Args:
        results: 检索结果列表，可为空。

    Returns:
        编号文本，每条形如 ``[1] 标题\\n链接\\n摘要``，条目之间空行分隔；
        空列表时返回一句引导模型如实回答的提示语（而不是空字符串，
        避免模型面对空内容时自行补全）。
    """
    if not results:
        return "（本次检索未返回任何结果。请尝试更换关键词，或如实告知用户未能检索到相关信息。）"

    blocks = [
        f"[{index}] {item.title}\n{item.url}\n{item.snippet}"
        for index, item in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


def build_tools(settings: Settings, provider: SearchProvider) -> List[BaseTool]:
    """构造可用工具列表。

    工具通过**闭包**固定住检索依赖与条数配置，因此工具本身是无状态的，
    可以被多个请求并发复用，无需额外传参。

    Args:
        settings: 全局配置，当前仅用于读取 ``search_top_k``。
        provider: 已就绪的搜索源实例。

    Returns:
        工具列表。当前只含一个 ``web_search``；若未来需要禁用检索，
        调用方应改传空列表（`agent.Agent` 会据此编译出不带工具的图）。
    """
    top_k = settings.search_top_k

    # 注意：本函数的 docstring 会被 LangChain 用作工具描述，直接发给模型，
    # 因此内容按「给模型看的说明」来写，不要改成面向开发者的文档风格。
    @tool(response_format="content_and_artifact")
    async def web_search(query: str) -> Tuple[str, Dict[str, Any]]:
        """在互联网上检索实时信息。

        适用场景：新闻时事、价格/天气/赛程等时效性数据、需要核实的具体事实，
        或任何你不确定、知识可能已经过时的内容。
        不适用于：常识问答、代码编写、数学推导。

        Args:
            query: 简洁精准的检索关键词，不要整句照搬用户原话。
        """
        # 执行检索：内部已吞掉网络异常返回空列表，这里再兜一层防止意外异常穿透图
        try:
            results = await provider.search(query, top_k)
        except Exception as exc:  # noqa: BLE001 - 交给中间件兜底，不让异常炸掉整张图
            raise ToolException(f"检索失败：{exc}") from exc

        # content → 模型可见：带编号的文本，供模型据此产出 [1][2] 引用
        content = format_results(results)

        # artifact → 模型不可见：结构化来源，供 agent 层事后遍历收集
        artifact = {
            "query": query,
            "sources": [item.to_dict() for item in results],
        }

        # 返回 (content, artifact) 二元组，LangGraph 自动拆成两条路
        return content, artifact

    return [web_search]