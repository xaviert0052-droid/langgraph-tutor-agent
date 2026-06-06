"""TutorState — 课程助教 Agent 的共享工作台。

所有节点围绕 State 流转：
- messages:  对话历史，通过 operator.add 累加（非覆盖）
- intent:    当前问题意图 (concept / comparison / practice / project)
- topic:     当前问题主题标识
- context_blocks: 按意图分支补充的上下文块
- answer:    模型生成的最终回答（覆盖式）
- llm_calls: 本轮调用模型次数
- steps:     执行步骤日志，通过 operator.add 累加
"""

import operator
from typing import Annotated, Optional

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class TutorState(TypedDict, total=False):
    """课程助教 Agent 的状态定义。

    字段默认覆盖（新值替换旧值），
    messages / steps 配置 operator.add 实现追加。
    """

    # 对话消息历史 — 每次追加而非覆盖
    messages: Annotated[list, operator.add]

    # 意图分类（路由依据）
    intent: str

    # 主题标识
    topic: str

    # 上下文知识块（由分支节点写入）
    context_blocks: list[str]

    # LLM 最终答案（覆盖模式）
    answer: str

    # LLM 调用次数计数
    llm_calls: int

    # 执行步骤日志 — 每次追加
    steps: Annotated[list[str], operator.add]


class HelloState(TypedDict):
    """Hello World 示例的状态定义。"""
    user_text: str
    answer: str
