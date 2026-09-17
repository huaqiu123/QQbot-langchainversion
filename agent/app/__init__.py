"""Small Agent 后端。

一个基于 **LangChain `create_agent` + DeepSeek** 的可联网问答服务，用 FastAPI
对外提供 HTTP 接口。Agent 会自主判断是否需要联网检索，多轮调用搜索工具后
汇总出带引用来源的中文回答。

模块分层（依赖严格单向，`main` 在最外层）：

.. code-block:: text

    main.py      HTTP 路由、鉴权、异常映射、生命周期
      |
      +-- agent.py        组图 / 执行 / 事件流适配
            |
            +-- llm.py      配置 -> 模型实例（超参集中地）
            +-- tools.py    SearchResult -> LangChain Tool
            +-- prompts.py  提示词
                  |
                  +-- search.py   关键词 -> 结构化 SearchResult（不依赖 LangChain）
                  +-- config.py   唯一读取环境变量的地方
            |
            +-- schemas.py  对外接口契约（被 main 与 agent 共用）

本轮范围仅限 Agent 后端，**未接入 NapCat / OneBot**；后续群消息只需转发到
``/v1/agent/ask`` 即可，Agent 层无需改动。
"""

__version__ = "0.1.0"
