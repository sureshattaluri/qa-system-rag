# app/api.py
from flask import Flask, request, jsonify
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph

from src.qa_system_rag.core.qa_engine import get_answer, load_text_embeddings, State, call_model

app = Flask(__name__)


# Our graph consists only of one node:
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Finally, we compile the graph with a checkpointer object.
# This persists the state, in this case in memory.
memory = MemorySaver()
ai_app = workflow.compile(checkpointer=memory)

@app.route("/qa", methods=["GET"])
def qa():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    try:
        answer = get_answer(query)
        return jsonify({"query": query, "answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["GET"])
def chat():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    try:
        config = {"configurable": {"thread_id": "abc123"}}

        result = ai_app.invoke(
            {"input": query},
            config=config,
        )
        print(result["answer"])

        return jsonify({"query": query, "result": result["answer"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/loadEmbeddings", methods=["POST"])
def load_embeddings():
    try:
        load_text_embeddings()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
