# app/api.py
from flask import Flask, request, jsonify
from src.qa_system_rag.core.qa_engine import get_answer, load_text_embeddings

app = Flask(__name__)


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


@app.route("/loadEmbeddings", methods=["POST"])
def load_embeddings():
    try:
        load_text_embeddings()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
