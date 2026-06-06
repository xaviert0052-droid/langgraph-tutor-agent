"""CLI 入口 — 课程助教 Agent 命令行工具。

用法：
    # 离线演示（不用 API Key）
    python -m langgraph_practical demo --mock

    # 真实调用（需要 .env 或环境变量）
    python -m langgraph_practical run --question "什么是 LangGraph？" --thread-id class-01

    # 查看帮助
    python -m langgraph_practical --help

    # 查看知识点库
    python -m langgraph_practical knowledge
"""

import argparse
import sys
import time
from typing import Any

from langgraph_practical.knowledge import all_context_blocks, all_topics, topic_title
from langgraph_practical.persistence import PickleSaver
from langgraph_practical.state import HelloState


def _build_checkpointer(persist: str | None) -> Any:
    """根据 --persist 参数构建 Checkpointer。

    Args:
        persist: None → InMemorySaver（默认）
                 路径字符串 → PickleSaver（持久化到文件）
    """
    if not persist:
        return None
    cp = PickleSaver(file_path=persist)
    print(f"  💾 持久化存档：{cp.checkpoint_path}")
    return cp


def _build_run_config(thread_id: str) -> dict[str, Any]:
    """构造运行配置，传入 graph.invoke() 的 config 参数。"""
    return {"configurable": {"thread_id": thread_id}}


def cmd_hello(args: argparse.Namespace) -> None:
    """运行 Hello World 示例。"""
    from langgraph.graph import END, START, StateGraph

    def say_hello(state: HelloState):
        return {"answer": f"你好，你刚才说的是：{state['user_text']}"}

    graph = StateGraph(HelloState)
    graph.add_node("say_hello", say_hello)
    graph.add_edge(START, "say_hello")
    graph.add_edge("say_hello", END)
    app = graph.compile()

    text = args.text or "LangGraph"
    result = app.invoke({"user_text": text})
    print(f"\n💬 你说：{text}")
    print(f"🤖 回答：{result['answer']}")


def cmd_knowledge(args: argparse.Namespace) -> None:  # noqa: ARG001
    """列出知识点库。"""
    print("\n📚 主题目录：")
    for tid, title in all_topics().items():
        print(f"  {tid:30s}  {title}")

    print("\n📖 知识块列表：")
    for topic, intent, content in all_context_blocks():
        preview = content[:60].replace("\n", " ")
        print(f"  {topic:25s} {intent:15s}  {preview}…")


def cmd_demo(args: argparse.Namespace) -> None:
    """运行离线演示模式。"""
    from langgraph_practical.app import build_graph

    mock = args.mock
    checkpointer = _build_checkpointer(args.persist)
    app = build_graph(mock=mock, checkpointer=checkpointer)

    sample_questions = [
        ("什么是 LangGraph？", "class-demo-1"),
        ("LangGraph 和 LangChain 有什么区别？", "class-demo-1"),
        ("State 是怎么工作的？", "class-demo-1"),
        ("怎么练习 LangGraph？", "class-demo-2"),
    ]

    print(f"\n{'='*60}")
    print("  🧑‍🏫 课程助教 Agent — 演示模式")
    print(f"  模式：{'📡 真实调用' if not mock else '🧪 离线模拟'}")
    print(f"{'='*60}")

    for question, thread_id in sample_questions:
        print(f"\n{'─'*60}")
        print(f"💬 [thread={thread_id}] {question}")
        print(f"{'─'*60}")

        config = _build_run_config(thread_id)
        start = time.time()

        result = app.invoke(
            {
                "messages": [{"role": "user", "content": question}],
                "steps": [],
            },
            config,
        )

        elapsed = time.time() - start

        print(f"🤖 回答：{result.get('answer', '(无回答)')}")
        print(f"\n📋 步骤轨迹：")
        for step in result.get("steps", []):
            print(f"   {step}")
        print(f"\n⏱ 耗时：{elapsed:.2f}s | 🤖 调用了 {result.get('llm_calls', 0)} 次模型")

    print(f"\n{'='*60}")
    print("  ✅ 演示完成")
    print(f"  💡 提示：同一 thread_id 下能记住前文（class-demo-1 的三连问）")
    print(f"{'='*60}")


def cmd_run(args: argparse.Namespace) -> None:
    """单次问题运行。"""
    from langgraph_practical.app import build_graph

    checkpointer = _build_checkpointer(args.persist)
    app = build_graph(mock=args.mock, checkpointer=checkpointer)
    config = _build_run_config(args.thread_id)

    question = args.question
    print(f"\n💬 问题：{question}")

    start = time.time()
    result = app.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "steps": [],
        },
        config,
    )
    elapsed = time.time() - start

    print(f"🤖 回答：{result.get('answer', '(无回答)')}")
    print(f"\n📋 步骤轨迹：")
    for step in result.get("steps", []):
        print(f"   {step}")
    print(f"\n⏱ 耗时：{elapsed:.2f}s | 🤖 调用了 {result.get('llm_calls', 0)} 次模型")

    # 如果有多轮历史，展示追加上下文
    steps = result.get("steps", [])
    if len(steps) > 3:
        print(f"\n💡 提示：同 thread_id 下继续提问会保留历史上下文")


def cmd_interactive(args: argparse.Namespace) -> None:
    """交互式聊天模式（支持连续追问）。"""
    from langgraph_practical.app import build_graph

    checkpointer = _build_checkpointer(args.persist)
    app = build_graph(mock=args.mock, checkpointer=checkpointer)
    thread_id = args.thread_id or f"interactive-{int(time.time())}"
    config = _build_run_config(thread_id)

    print(f"\n{'='*60}")
    print("  🧑‍🏫 课程助教 Agent — 交互模式")
    print(f"  Thread: {thread_id}")
    print(f"  输入 'quit' 或 'exit' 退出")
    print(f"{'='*60}")

    while True:
        try:
            question = input("\n💬 你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if not question:
            continue

        start = time.time()
        result = app.invoke(
            {
                "messages": [{"role": "user", "content": question}],
                "steps": [],
            },
            config,
        )
        elapsed = time.time() - start

        print(f"🤖 {result.get('answer', '(无回答)')}")
        print(f"   ⏱ {elapsed:.2f}s | 🤖 {result.get('llm_calls', 0)}次调用")
        print(f"   💡 同 thread 下继续提问可延续会话")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="🧑‍🏫 课程助教 Agent — LangGraph 入门教程实战项目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # Hello World（极简入门）
  python -m langgraph_practical hello --text "你好！"

  # 离线演示（推荐课堂先跑这条）
  python -m langgraph_practical demo --mock

  # 真实调用
  export DEEPSEEK_API_KEY=sk-xxx
  python -m langgraph_practical run --question "什么是 State？" --thread-id class-01

  # 交互式聊天
  python -m langgraph_practical chat --mock

  # 查看知识点库
  python -m langgraph_practical knowledge
        """.strip(),
    )

    sub = parser.add_subparsers(dest="command", help="子命令")

    # hello
    hello_parser = sub.add_parser("hello", help="运行 Hello World 示例")
    hello_parser.add_argument("--text", default="LangGraph", help="要说的文本")

    # demo
    demo_parser = sub.add_parser("demo", help="运行演示（多轮提问）")
    demo_parser.add_argument("--mock", action="store_true", help="离线模拟模式")
    demo_parser.add_argument(
        "--persist", nargs="?", const=".checkpoints/graph.pkl", default=None,
        help="启用持久化（默认存档路径 .checkpoints/graph.pkl）",
    )

    # run
    run_parser = sub.add_parser("run", help="单次提问")
    run_parser.add_argument("--question", required=True, help="问题内容")
    run_parser.add_argument("--thread-id", default="default", help="会话标识")
    run_parser.add_argument("--mock", action="store_true", help="离线模拟模式")
    run_parser.add_argument(
        "--persist", nargs="?", const=".checkpoints/graph.pkl", default=None,
        help="启用持久化，进程重启后同一 thread_id 可接续",
    )

    # chat
    chat_parser = sub.add_parser("chat", help="交互式聊天（支持连续追问）")
    chat_parser.add_argument("--thread-id", default=None, help="会话标识（自动生成）")
    chat_parser.add_argument("--mock", action="store_true", help="离线模拟模式")
    chat_parser.add_argument(
        "--persist", nargs="?", const=".checkpoints/graph.pkl", default=None,
        help="启用持久化，进程重启后同一 thread_id 可接续",
    )

    # knowledge
    sub.add_parser("knowledge", help="查看知识点库")

    args = parser.parse_args()

    if args.command == "hello":
        cmd_hello(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "chat":
        cmd_interactive(args)
    elif args.command == "knowledge":
        cmd_knowledge(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
