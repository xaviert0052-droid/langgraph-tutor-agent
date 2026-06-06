"""知识点库 — 课程助教 Agent 的知识上下文。

按主题 (topic) 和意图 (intent) 组织上下文块。
每个主题对应一个知识点卡片，各意图分支按需提取。

当前内置主题：
  - langgraph_intro    LangGraph 入门概念
  - state              State / Reducer
  - thread_checkpoint  Thread / Checkpoint
  - edge_router        条件边与路由
  - workflow_agent      Workflow vs Agent 模式
"""

from typing import Iterator

# ── 主题目录 ──────────────────────────────────────────────
TOPIC_CATALOG: dict[str, str] = {
    "langgraph_intro": "LangGraph 入门概念",
    "state": "State 与 Reducer",
    "thread_checkpoint": "Thread 与 Checkpoint",
    "edge_router": "条件边与路由",
    "workflow_agent": "Workflow 与 Agent 模式",
    "functional_api": "Functional API",
    "interrupt": "Interrupt 与人工审核",
    "persistence": "Persistence 与 Durable Execution",
}

# ── 按主题分 intent 的知识块 ──────────────────────────────
# key = f"{topic}::{intent}"
_CONTEXT_BLOCKS: dict[str, str] = {
    # ═══ langgraph_intro ═══
    "langgraph_intro::concept": (
        "LangGraph 是 LangChain 生态中的低层编排框架。它不是替代 LangChain，"
        "而是站在 LangChain 组件之上做流程调度。核心思想是把复杂工作流组织成"
        "一张可执行、可记忆、可恢复的图。图的四个核心要素是：State（共享数据）、"
        "Node（处理步骤）、Edge（路由规则）、Compile（编译执行）。"
    ),
    "langgraph_intro::comparison": (
        "LangChain 更像工具箱/零件库——提供模型接入、提示词、输出解析、"
        "检索等基础组件；LangGraph 更像总调度台——把这些组件组织成一条"
        "可分支、可恢复、可追踪的运行流程。两者不是替代关系，常见做法是"
        "'LangChain 提供组件，LangGraph 负责编排'。"
    ),
    "langgraph_intro::practice": (
        "练习建议：1）先画流程图再写代码，把业务拆成 4~6 个离散步骤；"
        "2）从 Routing 模式入手（先分类再路由），这是最常用的图入门模式；"
        "3）跑通 Hello World 感受 State→Node→Edge→Compile 四个步骤；"
        "4）给项目增加 thread_id 体验会话延续。"
    ),
    "langgraph_intro::project": (
        "项目建议：用 Routing 模式做一个课程助教 Agent——先判断学生问题意图"
        "（概念/对比/练习/项目），不同意图走不同分支补上下文，最后调用模型生成"
        "口语化答案。不需要很大，但要跑通图的分支 + 状态 + 追问。"
    ),
    "langgraph_intro::summary": (
        "核心总结：LangGraph 是 LangChain 生态的低层编排框架，用 State（共享数据）、"
        "Node（处理步骤）、Edge（路由规则）、Compile（编译执行）四个核心概念，"
        "把复杂工作流组织成一张可执行、可记忆、可恢复的图。关键能力包括条件边路由、"
        "Persistence 持久化、Interrupt 人工审核、Durable Execution 中断恢复。"
    ),

    # ═══ state ═══
    "state::concept": (
        "State 是 LangGraph 整张图共享的数据载体。前一个节点写进去的结果，"
        "后一个节点可以继续读取和补充。它不是普通字典——每个字段都可以配置"
        "Reducer（更新规则），默认是覆盖，也可以配成追加/累加。"
        "语言上，State 让流程的每个参与者都知道当前在哪、下一步去哪、"
        "上一个节点做了什么。"
    ),
    "state::comparison": (
        "和普通 Python 字典对比：1）谁改了它？普通字典难以追踪，State 的 Reducer "
        "显式控制每次更新；2）走到哪一步？普通字典没概念，State 通过 checkpoint "
        "记录每步快照；3）出错后怎么恢复？普通字典要自己写恢复逻辑，State "
        "从 checkpoint 恢复即可。"
    ),
    "state::practice": (
        "练习：定义一个状态包含 messages（list，用 operator.add 追加）、"
        "intent（str，覆盖模式）、steps（list，追加）。写两个节点，一个写 steps，"
        "一个读 steps，观察追加效果。"
    ),
    "state::project": (
        "在课程助教 Agent 中，State 的 messages 和 steps 字段配置了累加 Reducer，"
        "所以每次节点返回新消息时会被追加到历史列表尾部——这就是为什么"
        "同一个 thread_id 下能记住前文。"
    ),
    "state::summary": (
        "State 要点总结：State 是整张图共享的数据载体，前一个节点写，后一个节点读。"
        "每个字段可以配置 Reducer（更新规则）——默认覆盖，也可配 operator.add 追加。"
        "State 让流程的每个参与者都知道当前在哪、下一步去哪、上一个节点做了什么。"
        "State（当前桌面）、Checkpoint（每步快照）、Thread（会话编号）三者配合实现持久化。"
    ),

    # ═══ thread_checkpoint ═══
    "thread_checkpoint::concept": (
        "Thread 和 Checkpoint 是一对配合使用的概念。Thread（thread_id）标识"
        "同一会话或任务的主键，Checkpoint 是每走完关键一步保存的状态快照。"
        "State 是当前桌面，Checkpoint 是每一步拍的照片，Thread 是把这些照片"
        "串成一条会话的编号。没有 thread_id，checkpointer 不知道该从哪段历史"
        "接着跑。"
    ),
    "thread_checkpoint::comparison": (
        "短期记忆 vs 长期记忆：短期记忆是 thread-scoped，跟随当前会话——"
        "像正在进行中的聊天记录；长期记忆是跨会话、跨 thread 的 namespace 记忆——"
        "像'这个用户偏好什么风格'的长期档案。短期记忆靠 State + checkpoint，"
        "长期记忆靠专门的 Memory 模块。"
    ),
    "thread_checkpoint::practice": (
        "练习：用 InMemorySaver 编译图，同一 thread_id 连续问两个问题，"
        "观察 steps 是追加还是覆盖。再换一个 thread_id 提问，观察是否"
        "和之前的历史隔离。条件反射：一旦讲到记忆/恢复/追问，立刻想到 thread_id。"
    ),
    "thread_checkpoint::project": (
        "课程助教 Agent 中，同一个 thread_id 下连续 invoke 能体现会话延续。"
        "每轮调用都会在 checkpoint 中保存执行线，所以模型能看到前文。"
    ),
    "thread_checkpoint::summary": (
        "Thread & Checkpoint 要点：Thread（thread_id）标识同一会话的主键，"
        "Checkpoint 是每步的状态快照。State=当前桌面，Checkpoint=每步照片，"
        "Thread=照片串成的相册编号。没有 thread_id，checkpointer 不知道从哪恢复。"
        "短期记忆靠 State+checkpoint，长期记忆靠 Memory 模块。"
    ),

    # ═══ edge_router ═══
    "edge_router::concept": (
        "Edge 负责把节点连接起来，告诉系统当前步骤做完后下一步去哪。"
        "普通边路线固定，条件边（Conditional Edge）根据当前 State 内容"
        "动态决定路由。节点负责做什么，边负责下一步去哪——这两件事分开，"
        "流程才会清晰。"
    ),
    "edge_router::comparison": (
        "普通边 vs 条件边：普通边用 add_edge(from, to) 定义，适合固定顺序；"
        "条件边用 add_condition(from, router_fn, mapping) 定义，适合意图分流。"
        "Routing 模式是入门图结构最常用的模式——先分类再路由。"
    ),
    "edge_router::practice": (
        "练习：写一个包含 3 个节点的图，用条件边根据 intent 字段路由到不同节点。"
        "增加一条新分支只需要改两处：条件判断函数 + 加一个节点。"
    ),
    "edge_router::project": (
        "课程助教 Agent 的核心就是 Routing 模式：analyze_question 节点先判断"
        "intent（concept/comparison/practice/project），然后条件边路由到对应的"
        "retrieve_* 分支节点补上下文，最后统一汇入 generate_answer。"
    ),
    "edge_router::summary": (
        "Edge 要点总结：Edge 连接节点，告诉系统下一步去哪。普通边（add_edge）路线固定，"
        "条件边（add_conditional_edges）根据 State 内容动态路由。"
        "节点负责'做什么'，边负责'下一步去哪'——两件事分开，流程才清晰。"
        "Routing 模式是入门最常用的图模式：先分类，再路由。"
    ),

    # ═══ workflow_agent ═══
    "workflow_agent::concept": (
        "官方把两类系统分得很清楚：Workflow 有预先确定的代码路径，"
        "Agent 的过程更动态——LLM 根据当前状态决定是否调用工具、是否继续推理。"
        "Workflow 包括 Prompt Chaining、Parallelization、Routing、"
        "Orchestrator-worker、Evaluator-optimizer 五种模式；Agent 是第六种。"
    ),
    "workflow_agent::comparison": (
        "Workflow vs Agent：路径是否预先确定（大部分确定 vs 动态决定）；"
        "工具调用方式（预定义 vs 动态选择）；可测试性（通常更强 vs 依赖评估）；"
        "入门难度（更适合新手 vs 门槛更高）。不是谁比谁高级，是不同问题"
        "有不同解决方案。"
    ),
    "workflow_agent::practice": (
        "练习：分别用 Prompt Chaining 和 Routing 两种模式实现同一个功能——"
        "比如先判断问题类型，再生成回答。感受两种模式各自适合什么场景。"
    ),
    "workflow_agent::project": (
        "课程助教 Agent 采用 Routing 模式（Workflow 的一种），而不是 Agent Loop。"
        "因为教学场景的路径高度确定——先分类再回答，不需要模型动态决定"
        "是否调工具。初学者从 Routing 模式入手最稳。"
    ),
    "workflow_agent::summary": (
        "Workflow & Agent 要点总结：Workflow 路径预先确定，适合结构化场景，"
        "包括 Prompt Chaining、Parallelization、Routing、Orchestrator-worker、"
        "Evaluator-optimizer 五种模式。Agent 由 LLM 动态决定工具调用和推理路径。"
        "不是谁比谁高级，是不同问题有不同解决方案。"
    ),

    # ═══ functional_api ═══
    "functional_api::concept": (
        "Functional API 是 LangGraph 提供的另一种构建方式。与 Graph API 显式声明 "
        "State、Node、Edge 不同，Functional API 让你用普通 Python 函数 + @entrypoint / "
        "@task 装饰器来编写工作流，底层自动推断图结构。适合从已有脚本平滑迁移的场景。"
    ),
    "functional_api::comparison": (
        "Graph API vs Functional API：Graph API 显式声明 Node/Edge/State，图结构肉眼可见，"
        "适合教学、复杂分支和可视化排错；Functional API 用普通 Python 控制流（if/for/函数调用），"
        "通过装饰器获得持久化和中断能力，适合已有代码迁移。两者底层共享同一运行时，可以混用。"
    ),
    "functional_api::practice": (
        "练习：把一个简单的多步骤函数（比如：接收文本→统计字数→判断情感→返回结果）分别用 "
        "Graph API 和 Functional API 实现一遍。感受两者在控制流表达上的差异。"
    ),
    "functional_api::project": (
        "项目建议：如果你有一个已经写好的 Python 工作流脚本（比如数据处理 pipeline），"
        "尝试用 Functional API 的 @entrypoint 装饰器把它接上 LangGraph 的持久化和中断能力。"
        "这样不用重构图结构就能获得 checkpoint 和 thread 支持。"
    ),
    "functional_api::summary": (
        "Functional API 要点：用 @entrypoint/@task 装饰器包装普通 Python 函数即可获得 "
        "LangGraph 运行时能力，无需显式构图。与 Graph API 共享底层运行时，可按需混用。"
        "Graph API=显式图编排，Functional API=隐式控制流。"
    ),

    # ═══ interrupt ═══
    "interrupt::concept": (
        "Interrupt 是 LangGraph 的人工审核/暂停恢复机制。在节点中调用 interrupt() 会暂停执行、"
        "保存当前状态快照、等待外部输入，然后通过 Command(resume=...) 恢复。它不是简单的 input() "
        "替代品，而是运行时级别的暂停-存档-恢复机制。"
    ),
    "interrupt::comparison": (
        "interrupt() vs 普通 input()：interrupt() 是运行时级别的暂停——保存 checkpoint、等待外部信号、"
        "从断点恢复。而 input() 只是阻塞当前进程，没有存档恢复能力。使用 interrupt 需要三个前提："
        "配置了 checkpointer、指定了 thread_id、interrupt 前的副作用代码必须幂等。"
    ),
    "interrupt::practice": (
        "练习：在 generate_answer 节点前插入一个 approval 节点，调用 interrupt() 等待批准。"
        "先 invoke 一次触发暂停，再用 Command(resume=True) 恢复。观察 checkpoint 中保存的中断状态。"
    ),
    "interrupt::project": (
        "项目建议：在课程助教 Agent 中加入人工审核节点——模型生成回答后不直接返回，而是先暂停，"
        "等待老师批准或修改后再输出。这在实际教育场景中很有价值：AI 先答，人工再审再放行。"
    ),
    "interrupt::summary": (
        "Interrupt 要点：interrupt() 暂停→存档→等外部信号→恢复执行。三大铁律：①依赖 checkpointer+thread_id；"
        "②恢复时节点从头重跑（不是从 interrupt 下一行继续）；③interrupt 前的副作用代码必须幂等。"
        "三种持久化级别：exit（快）、async（折中）、sync（每一步落盘）。"
    ),

    # ═══ persistence ═══
    "persistence::concept": (
        "Persistence（持久化）是 LangGraph 内置的持久化层，会在执行过程中保存图状态的 checkpoint。"
        "这意味着图不只是当场算完就没了——它可以被查看、恢复、回放。Durable Execution（持久化执行）"
        "让任务中断后还能继续，即使服务器重启也不丢失进度。"
    ),
    "persistence::comparison": (
        "普通脚本 vs LangGraph 持久化：普通脚本进程挂了就丢了，要从头重跑；LangGraph 因为有 "
        "checkpointer + thread_id + checkpoint，即使进程崩溃、服务器重启，也能从上次中断的地方继续。"
        "三种持久化级别：exit（性能最好，但过程不一定全存）、async（平衡）、sync（每一步落盘，最稳最慢）。"
    ),
    "persistence::practice": (
        "练习：用 InMemorySaver 编译图，在节点中加入 time.sleep(1) 模拟耗时操作。在运行中途中断进程，"
        "重启后用同一个 thread_id 继续 invoke，观察 steps 是从头开始还是接续之前的记录。"
    ),
    "persistence::project": (
        "项目建议：把 InMemorySaver 替换为 SQLiteSaver 或 PostgresSaver，实现跨进程/跨重启的持久化。"
        "这样即使服务器重启，用户的对话线程也不会丢失——这才是真正的 Durable Execution。"
    ),
    "persistence::summary": (
        "Persistence 要点：checkpointer + thread_id = 可恢复的执行线。State=当前桌面，Checkpoint=每步快照，"
        "Thread=相册编号。Durable Execution 三要素：配置 checkpointer、指定 thread_id、非幂等操作包进 task。"
        "三种级别：exit→async→sync（越来越稳，越来越慢）。"
    ),
}


# ── 便捷查询函数 ──────────────────────────────────────────

def all_topics() -> dict[str, str]:
    """获取所有主题：{标识: 标题}"""
    return dict(TOPIC_CATALOG)


def topic_title(topic_id: str) -> str:
    """获取主题的可读标题。"""
    return TOPIC_CATALOG.get(topic_id, topic_id)


def build_context_blocks(intent: str, topic: str) -> list[str]:
    """按意图和主题查询匹配的上下文块。

    先尝试精确匹配 topic::intent，
    再尝试用通配符 *::intent 兜底，
    最后返回空列表。
    """
    key = f"{topic}::{intent}"
    block = _CONTEXT_BLOCKS.get(key)
    if block is not None:
        return [block]

    # intent 级别的兜底知识
    fallback: dict[str, str] = {
        "concept": "概念类问题：请从定义和核心思想入手，用生活化比喻帮助理解。",
        "comparison": "对比类问题：列出对比维度表格，突出核心区别，避免只说'A比B好'。",
        "practice": "练习类问题：给出可操作的分步练习建议，附代码片段示例。",
        "project": "项目类问题：给出项目目标、技术选型建议、分阶段实施路线。",
        "summary": "总结类问题：给出简洁的核心要点总结，突出关键概念之间的关联。",
    }
    fb = fallback.get(intent)
    return [fb] if fb else []


def all_context_blocks() -> Iterator[tuple[str, str, str]]:
    """遍历所有知识块：返回 (topic, intent, content) 三元组。"""
    for key, content in _CONTEXT_BLOCKS.items():
        topic, intent = key.split("::", 1)
        yield topic, intent, content


def get_all_documents() -> list[dict]:
    """导出所有知识块为带 metadata 的文档列表，供 RAG 引擎索引。

    每条文档格式：
        {
            "id": "langgraph_intro::concept",
            "text": "知识块内容...",
            "metadata": {"topic": "langgraph_intro", "intent": "concept"}
        }
    """
    docs = []
    for key, content in _CONTEXT_BLOCKS.items():
        topic, intent = key.split("::", 1)
        docs.append({
            "id": key,
            "text": content,
            "metadata": {"topic": topic, "intent": intent},
        })
    return docs
