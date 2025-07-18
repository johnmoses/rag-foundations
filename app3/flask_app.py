from flask import Flask, request, jsonify
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from glob import glob

# Define Flask
app = Flask(__name__)

# Define Milvus Client. We are using the lite package here
client = MilvusClient("milvus_rag_db_1.db")  # Or your Milvus server URI

# Model and embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
# dim = model.get_sentence_embedding_dimension()
dim = 384

# Create collection
collection_name = "my_rag_collection"
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    dimension=dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
    auto_id=True,
)

def embed(t):
    return model.encode(t, normalize_embeddings=True).tolist()

text_blocks = []
for fp in glob("data/**/*.md", recursive=True):
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        text_blocks += f.read().split("\n\n")

data = [
    {"id": i, "vector": embed(txt), "text": txt} for i, txt in enumerate(text_blocks)
]
print(f"Loaded {len(data)} documents")
print(f"First document: {data[0]}")
client.insert(collection_name=collection_name, data=data)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    query_embedding = model.encode([user_query]).tolist()
    results = client.search(
        collection_name=collection_name,
        data=query_embedding,
        limit=3,
        output_fields=["text"],
    )
    # Results is a list of lists of dicts (one list per query)
    retrieved = [hit["entity"]["text"] for hit in results[0]]
    return jsonify({"retrieved_documents": retrieved})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
