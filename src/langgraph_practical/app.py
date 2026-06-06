"""图编排 — 课程助教 Agent 的 LangGraph 图定义。

图结构（Routing 模式 — 5 分支）：
```
                    ┌→ retrieve_concept_context ─┐
                    │                             │
                    ├→ retrieve_comparison_context │
                    │                             │
  START → analyze_question ─→ retrieve_practice_context ─→ generate_answer → END
                    │                             │
                    ├→ retrieve_project_context  ─┘
                    │
                    └→ retrieve_summary_context ─┘
```

执行流程：
  1. analyze_question：分诊台 → 判断 intent + topic
  2. 条件边：根据 intent 路由到对应分支
  3. retrieve_*：按分支补上下文（RAG 检索）
  4. generate_answer：调用 LLM 生成最终回答

运行时能力：
  - InMemorySaver checkpointer（支持 thread_id 和追问）
  - 可选 mock 模式（离线演示）
"""

from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import START, StateGraph

from langgraph_practical import rag as rag_engine
from langgraph_practical.knowledge import (
    TOPIC_CATALOG,
    topic_title,
)
from langgraph_practical.model import DeepSeekTutorModel
from langgraph_practical.state import TutorState

load_dotenv()

# ── 意图检测（简化版） ──────────────────────────────────

_INTENT_KEYWORDS: dict[str, list[str]] = {
    # 具体意图优先（总结、对比）
    "summary": ["总结一下", "概括", "汇总", "归纳", "要点", "复习", "core concepts", "summary", "recap"],
    "comparison": ["区别", "对比", "vs", "vs.", "difference", "compared", "不同于", "有什么不同"],
    "practice": ["练习", "怎么学", "上手", "练", "实践", "pract", "exercise", "demo", "动手"],
    "project": ["项目", "实战", "搭建", "工程", "落地", "project", "build", "实现"],
    # 概念最宽泛，放最后兜底
    "concept": ["什么是", "解释", "介绍", "意思", "概念", "定义", "what", "explain", "tell me"],
}


def _detect_intent(question: str) -> str:
    """基于关键词的简单意图检测。"""
    q = question.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "concept"  # 默认走概念分支


def _detect_topic(question: str) -> str:
    """基于关键词的简单主题检测。"""
    q = question.lower()
    for topic_id, title in TOPIC_CATALOG.items():
        # 检测主题名中的关键词
        words = title.lower().replace("与", " ").replace("和", " ").split()
        if any(w in q for w in words if len(w) > 1):
            return topic_id
    return "langgraph_intro"  # 默认主题


def _latest_question(state: TutorState) -> str:
    """从消息历史中提取最新一条用户问题。"""
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            return msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
        if isinstance(msg, str):
            return msg
    return ""


def _build_system_prompt(state: TutorState) -> str:
    """构建系统提示词。"""
    intent = state.get("intent", "concept")
    topic = state.get("topic", "langgraph_intro")
    ctx = state.get("context_blocks", [])
    ctx_text = "\n\n".join(ctx) if ctx else "(暂无参考资料)"

    return (
        f"你是一位耐心的 AI 课程助教，正在辅导学生关于 LangGraph 的问题。\n\n"
        f"【当前问题类型】{intent}\n"
        f"【当前主题】{topic_title(topic)}\n\n"
        f"【参考资料】\n{ctx_text}\n\n"
        f"回答要求：\n"
        f"1. 用口语化、生活化的语言讲解，避免堆砌术语\n"
        f"2. 长度控制在 300 字以内\n"
        f"3. 如果适合举例子，一定要举\n"
        f"4. 结尾可以问学生'还有什么想了解的吗'"
    )


# ── 节点定义 ─────────────────────────────────────────────

def analyze_question(state: TutorState) -> dict:
    """节点 1：分诊台 — 分析问题意图和主题。"""
    question = _latest_question(state)
    if not question:
        return {"intent": "concept", "topic": "langgraph_intro", "steps": ["收到空问题"]}

    intent = _detect_intent(question)
    topic = _detect_topic(question)
    return {
        "intent": intent,
        "topic": topic,
        "steps": [
            f"📩 收到问题：{question[:60]}{'…' if len(question) > 60 else ''}",
            f"🔍 识别意图：{intent}",
            f"📂 识别主题：{topic_title(topic)}",
        ],
    }


def _rag_retrieve(intent: str, question: str, topic: str) -> tuple[list[str], str]:
    """统一 RAG 检索：调用向量引擎获取相关上下文。"""
    try:
        results = rag_engine.retrieve(
            query=question,
            k=3,
            intent=intent,
            topic=topic,
        )
        if results:
            blocks = [text for text, score in results]
            score_str = f"(最高相似度: {results[0][1]:.3f})"
        else:
            blocks = []
            score_str = "(无匹配)"
    except Exception as e:
        blocks = []
        score_str = f"(RAG 异常: {e})"
    return blocks, score_str


_RAG_INTENT_EMOJI = {
    "concept": "📖",
    "comparison": "⚖️",
    "practice": "✏️",
    "project": "🏗️",
    "summary": "📋",
}


def retrieve_concept_context(state: TutorState) -> dict:
    """节点 2a：概念分支 — RAG 检索概念类上下文。"""
    question = _latest_question(state)
    topic = state.get("topic", "langgraph_intro")
    blocks, score = _rag_retrieve("concept", question, topic)
    return {
        "context_blocks": blocks,
        "steps": [f"📖 概念分支（RAG检索）{score}：{topic_title(topic)}"],
    }


def retrieve_comparison_context(state: TutorState) -> dict:
    """节点 2b：对比分支 — RAG 检索对比类上下文。"""
    question = _latest_question(state)
    topic = state.get("topic", "langgraph_intro")
    blocks, score = _rag_retrieve("comparison", question, topic)
    return {
        "context_blocks": blocks,
        "steps": [f"⚖️ 对比分支（RAG检索）{score}：{topic_title(topic)}"],
    }


def retrieve_practice_context(state: TutorState) -> dict:
    """节点 2c：练习分支 — RAG 检索练习类上下文。"""
    question = _latest_question(state)
    topic = state.get("topic", "langgraph_intro")
    blocks, score = _rag_retrieve("practice", question, topic)
    return {
        "context_blocks": blocks,
        "steps": [f"✏️ 练习分支（RAG检索）{score}：{topic_title(topic)}"],
    }


def retrieve_project_context(state: TutorState) -> dict:
    """节点 2d：项目分支 — RAG 检索项目类上下文。"""
    question = _latest_question(state)
    topic = state.get("topic", "langgraph_intro")
    blocks, score = _rag_retrieve("project", question, topic)
    return {
        "context_blocks": blocks,
        "steps": [f"🏗️ 项目分支（RAG检索）{score}：{topic_title(topic)}"],
    }


def retrieve_summary_context(state: TutorState) -> dict:
    """节点 2e：总结分支 — RAG 检索总结类上下文。"""
    question = _latest_question(state)
    topic = state.get("topic", "langgraph_intro")
    blocks, score = _rag_retrieve("summary", question, topic)
    return {
        "context_blocks": blocks,
        "steps": [f"📋 总结分支（RAG检索）{score}：{topic_title(topic)}"],
    }


def generate_answer(state: TutorState):
    """节点 3：生成答案 — 逐 token 流式调用 DeepSeek 模型。

    这是一个生成器函数，每个 yield 在 stream_mode="updates" 下
    变成一条独立 SSE 事件，前端可逐 token 渲染。
    """
    question = _latest_question(state)
    system_prompt = _build_system_prompt(state)

    # 信号：开始生成
    yield {"type": "generate_start"}

    # 从全局单例获取模型，逐 token 流式读取
    model: DeepSeekTutorModel = _MODEL
    answer_parts: list[str] = []
    for token in model.stream(system_prompt, question):
        answer_parts.append(token)
        yield {"type": "token", "content": token}

    # 最终：完整回答 + 步骤
    full_answer = "".join(answer_parts)
    yield {
        "answer": full_answer,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "steps": ["✅ 答案生成完成"],
    }


# ── 条件边路由函数 ──────────────────────────────────────

def route_by_intent(
    state: TutorState,
) -> Literal[
    "retrieve_concept_context",
    "retrieve_comparison_context",
    "retrieve_practice_context",
    "retrieve_project_context",
    "retrieve_summary_context",
]:
    """条件边：根据意图路由到不同分支。"""
    intent = state.get("intent", "concept")
    mapping: dict[str, str] = {
        "concept": "retrieve_concept_context",
        "comparison": "retrieve_comparison_context",
        "practice": "retrieve_practice_context",
        "project": "retrieve_project_context",
        "summary": "retrieve_summary_context",
    }
    return mapping.get(intent, "retrieve_concept_context")


# ── 图构建与编译 ─────────────────────────────────────────

_MODEL: DeepSeekTutorModel = None  # type: ignore[assignment]


def build_graph(
    mock: bool = False,
    api_key: str | None = None,
    checkpointer=None,
) -> StateGraph:
    """构建并编译课程助教图。

    Args:
        mock: 是否使用离线模拟模式（默认 False）
        api_key: DeepSeek API 密钥（默认从 .env 读取）
        checkpointer: 自定义 Checkpointer（默认 InMemorySaver）
                      传 PickleSaver 或 SqliteSaver 可实现持久化

    Returns:
        编译好的可执行应用（StateGraph.compile 的返回值）
    """
    global _MODEL
    _MODEL = DeepSeekTutorModel(mock=mock, api_key=api_key)

    # 1. 初始化图
    builder = StateGraph(TutorState)

    # 2. 注册节点
    builder.add_node("analyze_question", analyze_question)
    builder.add_node("retrieve_concept_context", retrieve_concept_context)
    builder.add_node("retrieve_comparison_context", retrieve_comparison_context)
    builder.add_node("retrieve_practice_context", retrieve_practice_context)
    builder.add_node("retrieve_project_context", retrieve_project_context)
    builder.add_node("retrieve_summary_context", retrieve_summary_context)
    builder.add_node("generate_answer", generate_answer)

    # 3. 连接边
    builder.add_edge(START, "analyze_question")

    # 条件边：根据意图路由
    builder.add_conditional_edges(
        "analyze_question",
        route_by_intent,
        {
            "retrieve_concept_context": "retrieve_concept_context",
            "retrieve_comparison_context": "retrieve_comparison_context",
            "retrieve_practice_context": "retrieve_practice_context",
            "retrieve_project_context": "retrieve_project_context",
            "retrieve_summary_context": "retrieve_summary_context",
        },
    )

    # 各分支统一汇入答案生成节点
    for branch in [
        "retrieve_concept_context",
        "retrieve_comparison_context",
        "retrieve_practice_context",
        "retrieve_project_context",
        "retrieve_summary_context",
    ]:
        builder.add_edge(branch, "generate_answer")

    # 4. 编译（挂载 checkpointer 以支持 thread + 追问）
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    return app
