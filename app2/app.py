from pymilvus import (
    MilvusClient,
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from sentence_transformers import SentenceTransformer

# --- Config ---
MILVUS_DB_URI = "milvus_rag_db.db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
MAX_GENERATION_LENGTH = 200

# --- Step 1: Connect to Milvus ---
client = MilvusClient(MILVUS_DB_URI)
connections.connect(alias="default", uri=MILVUS_DB_URI)


# --- Step 2: Create collection with primary key if not exists ---
def create_collection():
    if COLLECTION_NAME in client.list_collections():
        return Collection(COLLECTION_NAME, using="default")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="RAG collection")
    return Collection(name=COLLECTION_NAME, schema=schema, using="default")


collection = create_collection()

# --- Step 3: Prepare in-memory documents ---
documents = [
    "Milvus is an open-source vector database built for scalable similarity search.",
    "It supports embedding-based search for images, video, and text.",
    "You can use SentenceTransformers to generate embeddings for your documents.",
    "GPT-2 is an open-source language model suitable for text generation tasks.",
]

# --- Step 4: Embed documents ---
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)


# --- Step 5: Insert data programmatically ---
def insert_data(collection, embeddings, texts):
    entities = [
        embeddings.tolist(),  # embeddings
        texts,  # texts
    ]
    collection.insert(entities)
    collection.flush()


if collection.num_entities == 0:
    insert_data(collection, doc_embeddings, documents)
else:
    print(
        f"Collection already has {collection.num_entities} entities, skipping insert."
    )

# --- Step 5.1: Create index and load collection ---
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

try:
    print("Creating index on embedding field...")
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created.")
except Exception as e:
    print(f"Index creation skipped or failed: {e}")

print("Loading collection into memory...")
collection.load()
print("Collection loaded.")


# --- Step 7: Query and generate ---
def query_and_generate(query):
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=TOP_K,
        output_fields=["text"],
        using="default",
    )
    retrieved_texts = [hit.entity.get("text") for hits in results for hit in hits]

    context = "\n\n".join(retrieved_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    print(f"\nGenerated Answer:\n{prompt}")


# --- Example usage ---
if __name__ == "__main__":
    print("RAG system ready.")
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        query_and_generate(query)
