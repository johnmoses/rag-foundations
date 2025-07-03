# Retrieval Autmented Generation (RAG)

Here we are going to implement a basic RAG that uses the Milvus vector database along with Large Language Models.

Here are the core components of the RAG:

- MilvusRAG Class: Main class that handles all RAG operations
- Embedding Generation: Uses SentenceTransformer to create vector embeddings
- Document Storage: Stores documents with metadata (title, category, text)
- Similarity Search: Retrieves most relevant documents based on query
- Response Generation: Combines retrieved documents to answer queries
- Command Line Interface for User Interaction
- Natural Language Processing Chatbot with Intent recognition and context awareness

Prerequisites

```bash
pip install pymilvus sentence-transformers numpy
```

Example queries

"What is machine learning?"
"Tell me about databases"
"How do neural networks work?"
"Find information about Python"

Example usage

You: What is machine learning?
🤖: Based on your question about 'what is machine learning', here's what I found:

📄 Introduction to Machine Learning (Relevance: 0.95)
📁 AI/ML

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed...

🔗 Related information:
   • Deep Learning Fundamentals (AI/ML)
   • Natural Language Processing (AI/ML)

💡 Ask me more specific questions for detailed information!
