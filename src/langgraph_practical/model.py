"""DeepSeek 模型接入 — 课程助教 Agent 的 LLM 节点。

支持两种模式：
  1. 真实模式：通过 ChatOpenAI（OpenAI 兼容接口）调用 DeepSeek
  2. Mock 模式：返回模拟回答，适合课堂演示 / 离线试跑

环境变量（.env 或 os.environ）：
  DEEPSEEK_API_KEY      API 密钥
  DEEPSEEK_MODEL        模型名（默认 deepseek-v4-flash）
  DEEPSEEK_BASE_URL     API 端点（默认 https://api.deepseek.com）
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI


class DeepSeekTutorModel:
    """课程助教模型封装，支持真实调用与离线模拟。"""

    def __init__(self, mock: bool = False, api_key: Optional[str] = None):
        self._mock = mock

        if mock:
            self._client = None
        else:
            api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "未设置 DEEPSEEK_API_KEY。\n"
                    "  - 请创建 .env 文件（参考 .env.example）\n"
                    "  - 或在环境变量中设置 DEEPSEEK_API_KEY\n"
                    "  - 或使用 --mock 以离线模式运行"
                )

            self._client = ChatOpenAI(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                api_key=api_key,
                base_url=os.environ.get(
                    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
                ),
                temperature=0.2,
            )

    def invoke(self, system_prompt: str, user_text: str) -> str:
        """生成完整回答（阻塞）。"""
        if self._mock:
            return self._mock_answer(user_text)
        return self._real_invoke(system_prompt, user_text)

    def stream(self, system_prompt: str, user_text: str):
        """逐 token 流式生成回答。"""
        if self._mock:
            yield self._mock_answer(user_text)
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        for chunk in self._client.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content

    # ── 内部：真实调用 ──────────────────────────────────

    def _real_invoke(self, system_prompt: str, user_text: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        response = self._client.invoke(messages)
        content: str = response.content if hasattr(response, "content") else str(response)
        return content

    # ── 内部：模拟回答 ──────────────────────────────────

    def _mock_answer(self, user_text: str) -> str:
        """根据问题关键词返回模拟回答。"""
        text = user_text.lower()

        if any(w in text for w in ["state", "状态", "reducer"]):
            return (
                "State 是 LangGraph 的共享工作台，所有节点都围绕它流转。\n\n"
                "举个生活中的例子：就像医院里每位病人都有一个病历夹，护士写一笔、"
                "医生写一笔，最后所有信息都在同一份病历里。State 就是这个病历夹。\n\n"
                "特别要注意的是 Reducer——它决定新数据怎么合并回 State。"
                "字段默认是覆盖模式（新值替换旧值），但如果配了 operator.add，"
                "就会变成追加模式。这就是同一个 thread_id 下能记住前文的原因——"
                "messages 字段是追加的，不是每轮覆盖。\n\n"
                "你还有什么想深入了解的吗？"
            )

        if any(w in text for w in ["thread", "checkpoint", "持久", "恢复"]):
            return (
                "Thread 和 Checkpoint 是 LangGraph 的两个核心概念：\n\n"
                "• Thread（thread_id）：标识这条执行线属于谁，像病历号/工单号。\n"
                "• Checkpoint：每走完关键一步保存的状态快照，像游戏存档位。\n\n"
                "State 是当前桌面（我正在做什么），Checkpoint 是每一步拍的照片"
                "（我刚才做了什么），Thread 是把这些照片串成一条会话的编号"
                "（这是谁的记录）。\n\n"
                "没有 thread_id，checkpointer 不知道该从哪段历史接着跑；"
                "没有 checkpointer，挂了就丢了，没法恢复。两者缺一不可。"
            )

        if any(w in text for w in ["edge", "边", "路由", "分支", "条件"]):
            return (
                "Edge 负责告诉系统当前步骤做完后下一步去哪。\n\n"
                "• 普通边（add_edge）：路线固定，A 做完一定去 B。\n"
                "• 条件边（add_conditional_edges）：根据当前 State 内容动态决定。\n\n"
                "课堂最喜欢的 Routing 模式本质上就是：先在一个节点里做分类判断，"
                "然后用条件边把不同结果路由到不同分支。\n\n"
                "扩展一个新分支只需要改两处：条件判断函数 + 加一个处理节点，"
                "体现了图编排的模块化优势。"
            )

        if any(w in text for w in ["workflow", "agent", "模式"]):
            return (
                "官方把工作流分成两大类：\n\n"
                "Workflow（工作流）：路径大部分预先确定，适合结构化场景。\n"
                "  - Prompt Chaining：一步接一步\n"
                "  - Parallelization：并行做再汇总\n"
                "  - Routing：先分类再路由 ← 课程助教用的就是它\n"
                "  - Orchestrator-worker：总调度分配子任务\n"
                "  - Evaluator-optimizer：生成→评价→优化\n\n"
                "Agent Loop：LLM 根据状态动态决定是否调工具/继续推理。\n\n"
                "初学者建议从 Routing 模式入手，不要急着做大 Agent。"
            )

        if any(w in text for w in ["总结", "概括", "汇总", "归纳", "要点", "复习", "summary", "recap"]):
            return (
                "好的，我来帮你总结 LangGraph 的核心要点：\n\n"
                "📌 它是什么？\n"
                "LangGraph 是 LangChain 生态的低层编排框架，不是替代 LangChain，"
                "而是站在 LangChain 组件之上做流程调度。\n\n"
                "📌 四个核心概念：\n"
                "1. State — 共享数据工作台，节点之间流转\n"
                "2. Node — 处理步骤，每个节点只做一件事\n"
                "3. Edge — 路由规则，决定下一步去哪\n"
                "4. Compile — 编译执行，把草图变可运行图\n\n"
                "📌 关键能力：\n"
                "• 条件边路由（Routing 模式）\n"
                "• Persistence 持久化（State+Checkpoint+Thread）\n"
                "• Interrupt 人工审核（暂停恢复）\n"
                "• Durable Execution 中断恢复\n\n"
                "📌 初学者建议：从 4~6 节点的小项目跑通图的节奏，从 Routing 模式入手最稳。"
            )

        # 默认回答
        return (
            "这是一个很好的问题！\n\n"
            "让我从 LangGraph 的核心思想说起：它不是'又一个聊天框架'，"
            "而是 AI 流程的调度系统。\n\n"
            "它用四个核心概念来组织复杂工作流：\n"
            "1. State（共享数据）— 节点之间流转的工作台\n"
            "2. Node（处理步骤）— 每个节点只做一件事\n"
            "3. Edge（路由规则）— 决定下一步去哪\n"
            "4. Compile（编译执行）— 把草图变成可运行图\n\n"
            "如果想深入了解某个概念，可以告诉我你想听哪个！"
        )


# ── 快捷函数 ──────────────────────────────────────────────

def build_tutor_model(mock: bool = False, api_key: Optional[str] = None) -> DeepSeekTutorModel:
    """快速构建课程助教模型。"""
    return DeepSeekTutorModel(mock=mock, api_key=api_key)
