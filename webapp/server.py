"""LangGraph 课程助教 — Web 服务器

提供 SSE (Server-Sent Events) 流式输出，前端逐节点看到进度。
"""

import json
import os
import time
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_BASE)

app = Flask(
    __name__,
    static_folder=os.path.join(_BASE, "webapp", "static"),
    static_url_path="/static",
)

CHECKPOINT_DIR = os.path.join(_BASE, ".checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sanitize(output: Any) -> dict:
    if not isinstance(output, dict):
        return {"raw": str(output)}
    safe = {}
    for k, v in output.items():
        if isinstance(v, (str, int, float, bool, list, dict)):
            safe[k] = v
        elif v is None:
            safe[k] = None
        else:
            safe[k] = str(v)
    return safe


# ── 路由 ────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/chat")
def chat():
    """SSE 流：逐节点推送 graph.stream(updates)。"""
    question = request.args.get("question", "")
    thread_id = request.args.get("thread_id", f"web-{int(time.time())}")
    mock = request.args.get("mock", "true").lower() == "true"

    if not question:
        return jsonify({"error": "question is required"}), 400

    from langgraph_practical.app import build_graph
    from langgraph_practical.persistence import PickleSaver

    cp_path = os.path.join(CHECKPOINT_DIR, f"{thread_id}.pkl")
    checkpointer = PickleSaver(file_path=cp_path)
    graph = build_graph(mock=mock, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    def generate():
        input_data = {
            "messages": [{"role": "user", "content": question}],
            "steps": [],
        }

        # 1) 先发送所有前置节点的进度
        for event in graph.stream(input_data, config, stream_mode="updates"):
            for node_name, output in event.items():
                if node_name.startswith("__") or node_name == "generate_answer":
                    continue
                yield _sse({
                    "type": "node",
                    "node": node_name,
                    "output": _sanitize(output),
                })
                time.sleep(0.15)

        # 2) generate_answer：直接调用模型 stream，逐 token 推送
        from langgraph_practical.app import _latest_question, _build_system_prompt
        from langgraph_practical.model import DeepSeekTutorModel

        state = graph.get_state(config).values
        system_prompt = _build_system_prompt(state)
        question_text = _latest_question(state)

        model: DeepSeekTutorModel = graph.checkpointer  # not the model…
        # 从全局 _MODEL 拿
        import langgraph_practical.app as app_mod
        llm = app_mod._MODEL

        # 标记 generate_answer 节点为激活
        yield _sse({"type": "node", "node": "generate_answer",
                     "output": {"steps": ["🤖 正在生成回答…"]}})

        answer_parts: list[str] = []
        for token in llm.stream(system_prompt, question_text):
            answer_parts.append(token)
            yield _sse({"type": "token", "content": token})

        full_answer = "".join(answer_parts)
        yield _sse({"type": "node", "node": "generate_answer",
                     "output": {"answer": full_answer,
                                "llm_calls": state.get("llm_calls", 0) + 1,
                                "steps": ["✅ 答案生成完成"]}})

        # 3) 完成
        yield _sse({"type": "done", "answer": full_answer,
                     "steps": state.get("steps", []) + ["✅ 答案生成完成"]})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/threads", methods=["DELETE"])
def delete_thread():
    thread_id = request.args.get("thread_id", "")
    cp_path = os.path.join(CHECKPOINT_DIR, f"{thread_id}.pkl")
    if os.path.exists(cp_path):
        os.remove(cp_path)
        return jsonify({"status": "deleted", "thread_id": thread_id})
    return jsonify({"status": "not_found", "thread_id": thread_id}), 404


if __name__ == "__main__":
    import webbrowser

    port = 5000
    print(f"\n  === 课程助教 Agent - Web 界面 ===")
    print(f"  打开浏览器访问: http://localhost:{port}")
    print("  按 Ctrl+C 停止服务器")
    print("  ===================================\n")

    webbrowser.open(f"http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
