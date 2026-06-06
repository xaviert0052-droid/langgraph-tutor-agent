"""极简 LangGraph 示例 — Hello World。

演示 State → Node → Edge → Compile 四个核心步骤。
"""

from langgraph.graph import END, START, StateGraph

from langgraph_practical.state import HelloState


def say_hello(state: HelloState):
    """节点：接收状态，返回状态更新。"""
    return {"answer": f"你好，你刚才说的是：{state['user_text']}"}


def build_hello_graph():
    """构建 Hello World 图。"""
    graph = StateGraph(HelloState)
    graph.add_node("say_hello", say_hello)
    graph.add_edge(START, "say_hello")
    graph.add_edge("say_hello", END)
    return graph.compile()


def main():
    app = build_hello_graph()
    result = app.invoke({"user_text": "LangGraph"})
    print(result["answer"])


if __name__ == "__main__":
    main()
