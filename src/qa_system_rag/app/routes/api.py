# app/api.py
from flask import Blueprint, request, jsonify
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph

from src.qa_system_rag.core.chat_vertex_ai import get_answer, load_text_embeddings
from src.qa_system_rag.core.chat_langchain import State, call_model
from src.qa_system_rag.core.chat_langchain_agent import agent_executor

api_bp = Blueprint('api', __name__)

# Our graph consists only of one node:
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Finally, we compile the graph with a checkpointer object.
# This persists the state, in this case in memory.
memory = MemorySaver()
chain_app = workflow.compile(checkpointer=memory)

executor = agent_executor()


@api_bp.route("/chatVertexAI", methods=["GET"])
def chat_vertex_ai():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    try:
        answer = get_answer(query)
        return jsonify({"query": query, "answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/chatLangchain", methods=["GET"])
def chat_langchain():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    try:
        config = {"configurable": {"thread_id": "abc123"}}

        result = chain_app.invoke(
            {"input": query},
            config=config,
        )
        print(result["answer"])

        return jsonify({"query": query, "result": result["answer"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/chatLangchainAgent", methods=["GET"])
def chat_langchain_agent():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400
    try:
        config = {"configurable": {"thread_id": "abc123"}}

        for event in executor.stream(
                {"messages": [HumanMessage(content=query)]},
                config=config,
                stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
        return jsonify({"query": query, "result": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/loadEmbeddings", methods=["POST"])
def load_embeddings():
    try:
        load_text_embeddings()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
