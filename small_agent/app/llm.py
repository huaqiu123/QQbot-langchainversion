"""模型构建层：只负责「配置 -> 模型实例」。

**为什么需要这一层**：`create_agent` 只接受 ``model`` 一个模型相关参数，
**不接受** temperature / max_tokens / timeout / base_url 等超参 —— 官方要求
「用提供商包先构造模型实例，再把实例传给 create_agent」。因此这些超参必须在这里
绑定到实例上。

抽成独立模块（而不是内联在 agent.py）的收益：超参集中可查、切换厂商实现只需改
本文件、模型构造可脱离 Agent 单独单测。

依赖方向：``config -> llm``，本模块不碰工具、不碰图、不碰提示词。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

#: DeepSeek 的 OpenAI 兼容端点，仅在 langchain-deepseek 缺失时作为回退使用。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_chat_model(settings: Settings) -> BaseChatModel:
    """根据配置构造聊天模型实例。

    优先使用官方集成包 `langchain-deepseek` 的 ``ChatDeepSeek``（会自动读取
    ``DEEPSEEK_API_KEY``，并原生支持 tools / streaming / token 用量）；
    若该包未安装，则回退到 ``ChatOpenAI`` 并把 ``base_url`` 指向 DeepSeek ——
    两者请求协议兼容，属于逃生口。

    Args:
        settings: 全局配置，提供模型名与各项超参。

    Returns:
        可直接交给 `create_agent` 使用的聊天模型实例。

    Raises:
        RuntimeError: 未配置 ``DEEPSEEK_API_KEY`` 时抛出。错误消息中包含可执行的
            修复指引，会原样透传到 ``/health`` 与 503 响应里。
    """
    if not settings.is_llm_configured():
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请复制 .env.example 为 .env 并填入 Key 后重启服务。"
        )

    kwargs = {
        "model": settings.deepseek_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
        "api_key": settings.deepseek_api_key,
    }

    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError:  # pragma: no cover - 逃生口
        logger.warning("未安装 langchain-deepseek，回退到 ChatOpenAI + DeepSeek base_url")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**kwargs, base_url=DEEPSEEK_BASE_URL)

    return ChatDeepSeek(**kwargs)


def validate_model(settings: Settings, has_tools: bool) -> None:
    """启动期校验「模型能力」与「Agent 需求」是否匹配。

    当前只检查一项：绑定工具时，模型必须支持 function calling。
    ``deepseek-reasoner``(R1) 不支持工具调用，配错**不会报错**，只会静默不调工具、
    Agent 完全失去检索能力 —— 属于难排查的故障，因此这里主动告警。

    Args:
        settings: 全局配置。
        has_tools: 本次组图是否绑定了工具（无工具时不会触发校验）。
    """
    if has_tools and not settings.model_supports_tools():
        logger.warning(
            "DEEPSEEK_MODEL=%s 不支持 function calling，Agent 将无法调用检索工具，"
            "请改用 deepseek-chat。",
            settings.deepseek_model,
        )


def model_name(settings: Settings | None = None) -> str:
    """获取当前使用的模型名。

    Args:
        settings: 可选配置实例；不传则取全局单例。

    Returns:
        模型名，用于 ``/health`` 与响应体中的 ``model`` 字段。
    """
    return (settings or get_settings()).deepseek_model


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """获取进程内共享的聊天模型实例。

    以全局配置为入参做缓存，避免每次调用都重建客户端与底层连接池。

    Returns:
        缓存后的聊天模型实例。

    Raises:
        RuntimeError: 未配置 ``DEEPSEEK_API_KEY`` 时抛出。
    """
    return build_chat_model(get_settings())
