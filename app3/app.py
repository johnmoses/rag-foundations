import os
from pymilvus import (
    MilvusClient,
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# --- Config ---
MILVUS_DB_URI = "milvus_rag_db.db"  # Local Milvus Lite DB file
COLLECTION_NAME = "rag_collection"
DOCS_FOLDER = "./data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
T5_MODEL_NAME = "t5-small"  # You can use t5-base or larger if you want
TOP_K = 3
MAX_GENERATION_LENGTH = 150

# --- Step 1: Connect to Milvus ---
print("Connecting to Milvus...")
client = MilvusClient(MILVUS_DB_URI)
connections.connect(alias="default", uri=MILVUS_DB_URI)

# --- Step 2: Create collection with primary key if not exists ---
def create_collection():
    if COLLECTION_NAME in client.list_collections():
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME, using="default")

    print(f"Creating collection '{COLLECTION_NAME}'...")
    fields = [
        FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
        ),  # Primary key with auto ID
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=384,
            description="sentence embeddings",
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="document text",
        ),
    ]
    schema = CollectionSchema(fields, description="RAG documents collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema, using="default")
    return collection


collection = create_collection()

# --- Step 3: Load local documents ---
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    docs.append(content)
    print(f"Loaded {len(docs)} documents from {folder_path}")
    return docs


documents = load_documents(DOCS_FOLDER)

# --- Step 4: Embed documents ---
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding documents...")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# --- Step 5: Insert data into Milvus ---
def insert_data(collection, embeddings, texts):
    print(f"Inserting {len(texts)} documents into Milvus...")
    entities = [
        embeddings.tolist(),  # embeddings
        texts,  # texts
    ]
    collection.insert(entities)
    collection.flush()
    print("Data inserted.")


if collection.num_entities == 0:
    insert_data(collection, doc_embeddings, documents)
else:
    print(f"Collection already has {collection.num_entities} entities, skipping insert.")

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

# --- Step 6: Setup T5 for generation ---
print("Loading T5 model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# --- Step 7: Query Milvus and generate answer ---
def query_and_generate(query):
    print(f"\nQuery: {query}")
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

    retrieved_texts = []
    for hits in results:
        for hit in hits:
            retrieved_texts.append(hit.entity.get("text"))
    print(f"Retrieved {len(retrieved_texts)} documents from Milvus.")

    # Prepare prompt for T5 generation
    context = " ".join(retrieved_texts)
    prompt = f"question: {query} context: {context}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=MAX_GENERATION_LENGTH,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated Response:\n{response}")


# --- Main interactive loop ---
if __name__ == "__main__":
    print("\n--- RAG System with T5 Ready ---")
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        query_and_generate(query)
