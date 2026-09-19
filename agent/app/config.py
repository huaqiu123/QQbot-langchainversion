"""配置层：全项目**唯一**读取环境变量 / `.env` 的地方。

本模块只做「读配置 -> 给默认值」，不构造任何客户端、不发起任何网络请求，
因此可以被任意层安全导入，不会产生副作用或循环依赖。

**配置优先级**：进程环境变量 > `.env` 文件 > 字段默认值。

**环境变量命名规则**：字段名转大写，例如 ``deepseek_api_key`` 对应
``DEEPSEEK_API_KEY``（`case_sensitive=False`，因此大小写不敏感）。

对外只暴露两个东西：

- `Settings`：配置模型本体，按用途分为 DeepSeek、搜索、Agent、LangSmith、服务五组；
- `get_settings()`：进程内单例，业务代码一律通过它取配置。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# small_agent/ 目录。用 __file__ 反推而不是 os.getcwd()，
# 这样无论从哪个目录启动服务（或跑测试），都能正确定位到 .env。
BASE_DIR = Path(__file__).resolve().parent.parent

#: 支持 function calling 的 DeepSeek 模型白名单。
#: `deepseek-reasoner`(R1) 不在此列 —— 它不支持工具调用，
#: 一旦配上会让 Agent 完全失去检索能力（且不会报错，只会静默不调工具）。
TOOL_CAPABLE_MODELS = {"deepseek-chat", "deepseek-chat-v3", "deepseek-v3"}


class Settings(BaseSettings):
    """项目全量配置。

    字段按用途分组：DeepSeek 模型、搜索、Agent 行为、LangSmith 追踪、HTTP 服务。
    所有字段都有默认值，因此「零配置」也能启动 —— 只是 Agent 会因缺少 API Key
    而处于未就绪状态（见 `is_llm_configured`），`/health` 会返回 ``degraded``。

    配置来源由 `model_config` 决定：优先读进程环境变量，其次读 `small_agent/.env`。
    未声明的多余变量（`extra="ignore"`）会被忽略，便于多服务共用同一个 `.env`。
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- DeepSeek ----------
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_timeout: float = 60.0
    llm_max_retries: int = 2

    # ---------- 搜索 ----------
    search_provider: str = "duckduckgo"
    search_top_k: int = 5
    search_timeout: float = 15.0
    # ddgs 是元搜索库，可指定后端；默认顺序兼顾国内网络可达性
    ddgs_backends: str = "bing,brave,duckduckgo"
    tavily_api_key: str = ""
    serper_api_key: str = ""

    # ---------- Agent ----------
    recursion_limit: int = 12
    max_history_messages: int = 10
    checkpoint_db_path: str = "data/checkpoint.db"
    request_timeout: float = 120.0

    # ---------- 知识库（Embedding + 向量检索） ----------
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    chroma_persist_dir: str = "data/chroma"
    knowledge_db_path: str = "data/knowledge.db"
    knowledge_similarity_threshold: float = 0.7
    knowledge_search_top_k: int = 5
    knowledge_chunk_size: int = 256
    knowledge_chunk_overlap: int = 32

    # ---------- LangSmith（可选） ----------
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "small-agent"

    # ---------- 服务 ----------
    agent_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 8000

    # ---------------- 派生属性 ----------------
    # 以下几项不是配置项，而是「由配置推导出来的判断」。
    # 放在 Settings 上是为了让判断逻辑集中，避免散落在各层。

    def is_llm_configured(self) -> bool:
        """判断是否已配置 DeepSeek API Key。

        Returns:
            已配置返回 ``True``；否则返回 ``False``。全空白字符串视为未配置。
        """
        return bool(self.deepseek_api_key.strip())

    def ddgs_backend_list(self) -> List[str]:
        """把逗号分隔的 ``DDGS_BACKENDS`` 解析成后端名列表。

        Returns:
            去除空白与空项后的后端名列表；若配置为空则回退到默认顺序
            ``["bing", "brave", "duckduckgo"]``。
        """
        backends = [item.strip() for item in (self.ddgs_backends or "").split(",") if item.strip()]
        return backends or ["bing", "brave", "duckduckgo"]

    def effective_search_provider(self) -> str:
        """计算实际生效的搜索源。

        指定的 Provider 若缺少对应 API Key，会静默降级为 ``duckduckgo``（ddgs），
        避免因为配置疏漏直接让服务不可用。降级事实由
        `search.build_search_provider` 打警告日志告知。

        Returns:
            ``"duckduckgo"`` / ``"tavily"`` / ``"serper"`` 三者之一。
        """
        provider = (self.search_provider or "").strip().lower()
        if provider == "tavily" and self.tavily_api_key.strip():
            return "tavily"
        if provider in ("serper", "google") and self.serper_api_key.strip():
            return "serper"
        return "duckduckgo"

    def model_supports_tools(self) -> bool:
        """判断当前模型是否支持 function calling。

        DeepSeek 只有 V3 系列（``deepseek-chat``）支持工具调用；
        ``deepseek-reasoner``（R1）不支持，配错会导致 Agent 无法检索。

        Returns:
            模型名在 `TOOL_CAPABLE_MODELS` 白名单内则返回 ``True``。
        """
        return self.deepseek_model.strip().lower() in TOOL_CAPABLE_MODELS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取进程内唯一的配置实例。

    用 ``lru_cache`` 做单例，好处是所有模块拿到的是同一份配置，且**不依赖
    FastAPI 的 lifespan 是否已经执行** —— 这一点很重要，早期版本在路由里读
    ``request.app.state.settings``，导致 lifespan 未运行时会抛 `AttributeError`。

    注意：测试中若需要切换环境变量，必须先调用 ``get_settings.cache_clear()``。

    Returns:
        进程内共享的 `Settings` 实例。
    """
    return Settings()