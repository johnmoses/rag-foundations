""" 
Basic implementation of RAG (Retrieval Augmented Generation) using Flask
"""

from flask import Flask, request, jsonify


# Mock classes to simulate RAG components
class SimpleRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=3):
        # Simple keyword matching retrieval (replace with vector search in production)
        results = [doc for doc in self.documents if query.lower() in doc.lower()]
        return results[:top_k]


class SimpleGenerator:
    def generate(self, query, context_docs):
        # Simple generation combining query and retrieved docs (replace with LLM call)
        context = " ".join(context_docs)
        return f"Answer based on query: '{query}' and context: '{context}'"


class RAGPipeline:
    def __init__(self, documents):
        self.retriever = SimpleRetriever(documents)
        self.generator = SimpleGenerator()

    def query(self, question):
        retrieved_docs = self.retriever.retrieve(question)
        response = self.generator.generate(question, retrieved_docs)
        return response


# Sample documents to serve as knowledge base
DOCUMENTS = [
    "Python is a popular programming language.",
    "Flask is a lightweight web framework for Python.",
    "Retrieval-Augmented Generation combines retrieval and generation.",
    "Vector databases store embeddings for efficient search.",
    "LangChain is a framework to build LLM applications.",
]

app = Flask(__name__)
rag_pipeline = RAGPipeline(DOCUMENTS)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    question = data["query"]
    try:
        answer = rag_pipeline.query(question)
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def health_check():
    return "RAG Flask API is running."


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
