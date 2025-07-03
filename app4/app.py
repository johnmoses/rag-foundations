from flask import Flask, request, jsonify, render_template_string
import os
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import uuid

# Vector and embedding imports
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import numpy as np

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
import chardet

# LLM imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Text processing
import nltk
from nltk.tokenize import sent_tokenize
import re

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
except:
    pass

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and collections
embedding_model = None
llm_pipeline = None
client = None

# Milvus configuration
COLLECTION_NAME = "document_chunks"
DIMENSION = 384  # all-MiniLM-L6-v2 embedding dimension
DB_FILE = "milvus_rag_db2.db"


class RAGSystem:
    def __init__(self):
        self.setup_milvus()
        self.load_models()

    def setup_milvus(self):
        """Initialize Milvus Lite client and collection"""
        global client

        try:
            # Initialize Milvus client with local database file
            client = MilvusClient(DB_FILE)
            logger.info(f"Connected to Milvus database: {DB_FILE}")

            # Check if collection exists
            collections = client.list_collections()

            if COLLECTION_NAME not in collections:
                # Create collection schema
                schema = {
                    "collection_name": COLLECTION_NAME,
                    "dimension": DIMENSION,
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "index_params": {"nlist": 128},
                }

                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    dimension=DIMENSION,
                    metric_type="COSINE",
                    consistency_level="Strong",
                )

                logger.info(f"Created new collection: {COLLECTION_NAME}")
            else:
                logger.info(f"Using existing collection: {COLLECTION_NAME}")

        except Exception as e:
            logger.error(f"Milvus setup error: {e}")
            # Fallback to in-memory storage if Milvus fails
            self.use_fallback_storage()

    def use_fallback_storage(self):
        """Fallback to in-memory vector storage if Milvus fails"""
        global client
        client = InMemoryVectorStore()
        logger.info("Using in-memory vector storage as fallback")

    def load_models(self):
        """Load embedding and LLM models"""
        global embedding_model, llm_pipeline

        try:
            # Load embedding model
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")

            # Load LLM - using a smaller model for local deployment
            model_name = "microsoft/DialoGPT-small"  # Lightweight conversational model

            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            llm_pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if device == "cuda" else -1,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
            )

            logger.info(f"Loaded LLM: {model_name}")

        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise


class InMemoryVectorStore:
    """Fallback in-memory vector storage"""

    def __init__(self):
        self.data = []
        self.index = 0

    def insert(self, collection_name, data):
        """Insert data into in-memory storage"""
        for item in data:
            self.data.append(item)

    def search(self, collection_name, data, limit=5, output_fields=None):
        """Search in-memory storage"""
        if not self.data:
            return []

        query_vec = np.array(data[0])  # First embedding in the list
        similarities = []

        for item in self.data:
            item_vec = np.array(item["vector"])
            similarity = np.dot(query_vec, item_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(item_vec)
            )
            similarities.append((similarity, item))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top results
        results = []
        for score, item in similarities[:limit]:
            result = {
                "id": item["id"],
                "distance": 1 - score,  # Convert similarity to distance
                "entity": {
                    "text": item.get("text", ""),
                    "filename": item.get("filename", ""),
                    "chunk_index": item.get("chunk_index", 0),
                },
            }
            results.append(result)

        return results


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "doc"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath):
    """Extract text from various file formats"""
    filename = os.path.basename(filepath)
    file_ext = filename.lower().split(".")[-1]

    try:
        if file_ext == "pdf":
            return extract_text_from_pdf(filepath)
        elif file_ext in ["docx", "doc"]:
            return extract_text_from_docx(filepath)
        else:  # txt and other text files
            return extract_text_from_txt(filepath)
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return ""


def extract_text_from_pdf(filepath):
    """Extract text from PDF file"""
    text = ""
    with open(filepath, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(filepath):
    """Extract text from DOCX file"""
    doc = DocxDocument(filepath)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_txt(filepath):
    """Extract text from TXT file with encoding detection"""
    with open(filepath, "rb") as file:
        raw_data = file.read()
        encoding = chardet.detect(raw_data)["encoding"]

    with open(filepath, "r", encoding=encoding or "utf-8") as file:
        return file.read()


def chunk_text(text, max_chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def store_document_chunks(chunks, filename):
    """Store document chunks in vector database"""
    global client, embedding_model

    if not chunks:
        return False

    try:
        # Generate embeddings
        embeddings = embedding_model.encode(chunks)

        # Prepare data for insertion
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append(
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding.tolist(),
                    "text": chunk,
                    "filename": filename,
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Insert data using MilvusClient
        if hasattr(client, "insert"):
            # Using actual MilvusClient
            client.insert(collection_name=COLLECTION_NAME, data=data)
        else:
            # Using fallback storage
            client.insert(COLLECTION_NAME, data)

        logger.info(f"Stored {len(chunks)} chunks for file: {filename}")
        return True

    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        return False


def search_similar_chunks(query, top_k=5):
    """Search for similar document chunks"""
    global client, embedding_model

    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0]

        # Perform search using MilvusClient
        if hasattr(client, "search"):
            # Using actual MilvusClient
            search_results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_embedding.tolist()],
                limit=top_k,
                output_fields=["text", "filename", "chunk_index"],
            )

            # Format results from MilvusClient
            retrieved_chunks = []
            for results in search_results:
                for result in results:
                    retrieved_chunks.append(
                        {
                            "text": result["entity"].get("text", ""),
                            "filename": result["entity"].get("filename", ""),
                            "chunk_index": result["entity"].get("chunk_index", 0),
                            "score": 1
                            - result["distance"],  # Convert distance to similarity
                        }
                    )
        else:
            # Using fallback storage
            search_results = client.search(
                COLLECTION_NAME,
                [query_embedding.tolist()],
                limit=top_k,
                output_fields=["text", "filename", "chunk_index"],
            )

            # Format results from fallback storage
            retrieved_chunks = []
            for result in search_results:
                retrieved_chunks.append(
                    {
                        "text": result["entity"].get("text", ""),
                        "filename": result["entity"].get("filename", ""),
                        "chunk_index": result["entity"].get("chunk_index", 0),
                        "score": 1 - result["distance"],
                    }
                )

        return retrieved_chunks

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


def generate_response(query, context_chunks):
    """Generate response using LLM with retrieved context"""
    global llm_pipeline

    # Prepare context
    context = "\n".join(
        [chunk["text"] for chunk in context_chunks[:3]]
    )  # Use top 3 chunks

    # Create prompt
    prompt = f"""Context information:
{context}

Question: {query}

Based on the context provided above, please provide a helpful and accurate answer:"""

    try:
        # Generate response
        response = llm_pipeline(
            prompt, max_length=len(prompt.split()) + 100, num_return_sequences=1
        )

        # Extract generated text (remove the prompt)
        generated_text = response[0]["generated_text"]
        answer = generated_text[len(prompt) :].strip()

        if not answer:
            answer = "I couldn't generate a specific answer based on the provided context. Please try rephrasing your question."

        return answer

    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return "Sorry, I encountered an error while generating the response."


# Initialize RAG system
rag_system = None


@app.before_first_request
def initialize_rag():
    global rag_system
    try:
        rag_system = RAGSystem()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG System with Milvus</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .upload-section, .chat-section { flex: 1; }
        .upload-box { border: 2px dashed #ccc; padding: 20px; text-align: center; margin-bottom: 20px; }
        .chat-box { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
        .message { margin-bottom: 10px; }
        .user-message { background: #e3f2fd; padding: 8px; border-radius: 5px; }
        .bot-message { background: #f5f5f5; padding: 8px; border-radius: 5px; }
        .input-group { display: flex; gap: 10px; }
        .input-group input { flex: 1; padding: 8px; }
        .btn { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .status { margin-top: 10px; padding: 10px; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
    </style>
</head>
<body>
    <h1>RAG System with Milvus Lite</h1>
    
    <div class="status info">
        <strong>Database:</strong> Using local Milvus database file: milvus_rag_db.db
    </div>
    
    <div class="container">
        <div class="upload-section">
            <h3>Document Upload</h3>
            <div class="upload-box">
                <input type="file" id="fileInput" accept=".txt,.pdf,.docx,.doc" multiple>
                <br><br>
                <button class="btn" onclick="uploadFiles()">Upload Documents</button>
            </div>
            <div id="uploadStatus"></div>
        </div>
        
        <div class="chat-section">
            <h3>Ask Questions</h3>
            <div id="chatBox" class="chat-box"></div>
            <div class="input-group">
                <input type="text" id="questionInput" placeholder="Ask a question about your documents..." onkeypress="handleKeyPress(event)">
                <button class="btn" onclick="askQuestion()">Ask</button>
            </div>
        </div>
    </div>

    <script>
        function uploadFiles() {
            const fileInput = document.getElementById('fileInput');
            const statusDiv = document.getElementById('uploadStatus');
            
            if (fileInput.files.length === 0) {
                statusDiv.innerHTML = '<div class="status error">Please select files to upload.</div>';
                return;
            }
            
            const formData = new FormData();
            for (let file of fileInput.files) {
                formData.append('files', file);
            }
            
            statusDiv.innerHTML = '<div class="status">Uploading and processing files...</div>';
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    statusDiv.innerHTML = `<div class="status success">${data.message}</div>`;
                    fileInput.value = '';
                } else {
                    statusDiv.innerHTML = `<div class="status error">${data.message}</div>`;
                }
            })
            .catch(error => {
                statusDiv.innerHTML = `<div class="status error">Upload failed: ${error}</div>`;
            });
        }
        
        function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const chatBox = document.getElementById('chatBox');
            const question = questionInput.value.trim();
            
            if (!question) return;
            
            // Add user message
            chatBox.innerHTML += `<div class="message user-message"><strong>You:</strong> ${question}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // Clear input
            questionInput.value = '';
            
            // Add loading message
            chatBox.innerHTML += `<div class="message bot-message" id="loadingMessage"><strong>Bot:</strong> Thinking...</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({query: question})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loadingMessage').remove();
                chatBox.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> ${data.answer}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            })
            .catch(error => {
                document.getElementById('loadingMessage').remove();
                chatBox.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> Sorry, I encountered an error: ${error}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            });
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        if "files" not in request.files:
            return jsonify({"success": False, "message": "No files provided"})

        files = request.files.getlist("files")

        if not files or all(f.filename == "" for f in files):
            return jsonify({"success": False, "message": "No files selected"})

        # Create upload directory if it doesn't exist
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        processed_files = 0
        total_chunks = 0

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Extract text
                text = extract_text_from_file(filepath)

                if text.strip():
                    # Chunk text
                    chunks = chunk_text(text)

                    # Store in vector database
                    if store_document_chunks(chunks, filename):
                        processed_files += 1
                        total_chunks += len(chunks)

                # Clean up uploaded file
                os.remove(filepath)

        if processed_files > 0:
            return jsonify(
                {
                    "success": True,
                    "message": f"Successfully processed {processed_files} files with {total_chunks} chunks stored in {DB_FILE}",
                }
            )
        else:
            return jsonify({"success": False, "message": "No files could be processed"})

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"})


@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        query_text = data.get("query", "").strip()

        if not query_text:
            return jsonify({"answer": "Please provide a question."})

        # Search for relevant chunks
        relevant_chunks = search_similar_chunks(query_text, top_k=5)

        if not relevant_chunks:
            return jsonify(
                {
                    "answer": f"No relevant information found in the uploaded documents. Database: {DB_FILE}"
                }
            )

        # Generate response
        answer = generate_response(query_text, relevant_chunks)

        # Add source information
        sources = list(set([chunk["filename"] for chunk in relevant_chunks[:3]]))
        if sources:
            answer += f"\n\nSources: {', '.join(sources)}"

        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify(
            {"answer": "Sorry, I encountered an error while processing your question."}
        )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": embedding_model is not None and llm_pipeline is not None,
            "database": DB_FILE,
            "collection": COLLECTION_NAME,
        }
    )


@app.route("/stats")
def stats():
    """Get database statistics"""
    try:
        if hasattr(client, "get_collection_stats"):
            stats = client.get_collection_stats(COLLECTION_NAME)
            return jsonify(stats)
        elif hasattr(client, "data"):
            # Fallback storage stats
            return jsonify(
                {
                    "collection_name": COLLECTION_NAME,
                    "entity_count": len(client.data),
                    "database_file": DB_FILE,
                }
            )
        else:
            return jsonify({"error": "Stats not available"})
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # Create upload directory
    os.makedirs("uploads", exist_ok=True)

    # Initialize RAG system
    try:
        rag_system = RAGSystem()
        logger.info("RAG system initialized successfully")
        logger.info(f"Database file: {DB_FILE}")
        logger.info(f"Collection name: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        exit(1)

    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5001)
