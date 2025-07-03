import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class MilvusRAG:
    def __init__(self, db_path: str = "milvus_rag_db.db"):
        """Initialize Milvus RAG system"""
        self.client = MilvusClient(db_path)
        self.collection_name = "documents"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
    def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            if self.client.has_collection(collection_name=self.collection_name):
                print(f"Collection '{self.collection_name}' already exists")
                return
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim,
                metric_type="COSINE",
                consistency_level="Strong"
            )
            print(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert documents into Milvus collection"""
        try:
            # Prepare data for insertion
            data = []
            for i, doc in enumerate(documents):
                # Generate embedding for the document text
                embedding = self.embed_text(doc['text'])
                
                data.append({
                    "id": i,
                    "text": doc['text'],
                    "title": doc.get('title', ''),
                    "category": doc.get('category', ''),
                    "vector": embedding
                })
            
            # Insert data
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            print(f"Inserted {len(documents)} documents successfully")
            
        except Exception as e:
            print(f"Error inserting documents: {e}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Search in Milvus
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["text", "title", "category"]
            )
            
            # Format results
            formatted_results = []
            for result in results[0]:  # results is a list of lists
                formatted_results.append({
                    "id": result["id"],
                    "text": result["entity"]["text"],
                    "title": result["entity"]["title"],
                    "category": result["entity"]["category"],
                    "score": result["distance"]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def generate_response(self, query: str, top_k: int = 3) -> str:
        """Generate RAG response based on retrieved documents"""
        # Retrieve relevant documents
        relevant_docs = self.search_similar(query, top_k)
        
        if not relevant_docs:
            return "No relevant documents found."
        
        # Combine retrieved documents as context
        context = "\n\n".join([
            f"Document {i+1} (Score: {doc['score']:.3f}):\n"
            f"Title: {doc['title']}\n"
            f"Category: {doc['category']}\n"
            f"Content: {doc['text']}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Simple response generation (in practice, you'd use an LLM here)
        response = f"""Based on the retrieved documents, here's the information related to your query: "{query}"

Retrieved Context:
{context}

Summary: The most relevant document (score: {relevant_docs[0]['score']:.3f}) suggests that {relevant_docs[0]['text'][:200]}...

Note: This is a basic RAG implementation. For production use, integrate with an LLM for better response generation."""
        
        return response

def create_demo_data() -> List[Dict[str, Any]]:
    """Create demo documents for testing"""
    return [
        {
            "title": "Introduction to Machine Learning",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.",
            "category": "AI/ML"
        },
        {
            "title": "Deep Learning Fundamentals",
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.",
            "category": "AI/ML"
        },
        {
            "title": "Python Programming Basics",
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation. Python's syntax is clean and intuitive, making it an excellent choice for beginners.",
            "category": "Programming"
        },
        {
            "title": "Data Structures and Algorithms",
            "text": "Data structures are ways of organizing and storing data in a computer so that it can be accessed and modified efficiently. Common data structures include arrays, linked lists, stacks, queues, trees, and graphs. Algorithms are step-by-step procedures for solving problems.",
            "category": "Computer Science"
        },
        {
            "title": "Web Development with Flask",
            "text": "Flask is a lightweight web framework for Python that provides the basic tools needed to build web applications. It's minimalist and flexible, allowing developers to choose their preferred tools and libraries for specific functionalities.",
            "category": "Web Development"
        },
        {
            "title": "Database Design Principles",
            "text": "Database design involves creating a detailed data model of a database. Good database design ensures data integrity, reduces redundancy, and optimizes performance. Key principles include normalization, proper indexing, and defining clear relationships between entities.",
            "category": "Database"
        },
        {
            "title": "Vector Databases Overview",
            "text": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They're essential for applications involving machine learning, AI, and similarity search. Examples include Milvus, Pinecone, and Weaviate.",
            "category": "Database"
        },
        {
            "title": "Natural Language Processing",
            "text": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves tasks like text classification, sentiment analysis, machine translation, and question answering.",
            "category": "AI/ML"
        },
        {
            "title": "Cloud Computing Concepts",
            "text": "Cloud computing is the delivery of computing services over the internet. It provides scalable resources like servers, storage, databases, and software on demand. Major cloud providers include Amazon Web Services, Microsoft Azure, and Google Cloud Platform.",
            "category": "Cloud"
        },
        {
            "title": "Cybersecurity Best Practices",
            "text": "Cybersecurity involves protecting digital information, systems, and networks from threats. Best practices include using strong passwords, enabling two-factor authentication, keeping software updated, and being cautious with email attachments and links.",
            "category": "Security"
        }
    ]

def display_menu():
    """Display the main menu options"""
    print("\n" + "="*60)
    print("🤖 MILVUS RAG SYSTEM")
    print("="*60)
    print("1. 🔍 Search/Query Documents")
    print("2. 📄 View All Documents")
    print("3. ➕ Add New Document")
    print("4. 📊 Show Collection Statistics")
    print("5. 🧪 Run Demo Queries")
    print("6. 🗑️  Clear Database")
    print("7. ❌ Exit")
    print("="*60)

def get_user_input(prompt: str, input_type: str = "string") -> Any:
    """Get user input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                print("❌ Input cannot be empty. Please try again.")
                continue
            
            if input_type == "int":
                return int(user_input)
            elif input_type == "float":
                return float(user_input)
            else:
                return user_input
        except ValueError:
            print(f"❌ Please enter a valid {input_type}.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            exit(0)

def view_all_documents(rag: MilvusRAG):
    """Display all documents in the collection"""
    try:
        # Get all documents by searching with a broad query
        all_docs = rag.search_similar("", top_k=1000)  # Get many documents
        
        if not all_docs:
            print("📝 No documents found in the collection.")
            return
        
        print(f"\n📚 Found {len(all_docs)} documents:")
        print("-" * 80)
        
        for i, doc in enumerate(all_docs, 1):
            print(f"{i}. 📄 {doc['title']}")
            print(f"   📁 Category: {doc['category']}")
            print(f"   📝 Text: {doc['text'][:150]}...")
            print(f"   🆔 ID: {doc['id']}")
            print()
            
            # Pause every 5 documents for readability
            if i % 5 == 0 and i < len(all_docs):
                cont = input("Press Enter to continue or 'q' to stop: ")
                if cont.lower() == 'q':
                    break
                    
    except Exception as e:
        print(f"❌ Error viewing documents: {e}")

def add_new_document(rag: MilvusRAG):
    """Add a new document to the collection"""
    print("\n➕ Adding New Document")
    print("-" * 40)
    
    try:
        title = get_user_input("📄 Enter document title: ")
        category = get_user_input("📁 Enter document category: ")
        
        print("📝 Enter document text (press Enter twice to finish):")
        text_lines = []
        empty_line_count = 0
        
        while True:
            line = input()
            if line.strip() == "":
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
                text_lines.append(line)
        
        text = "\n".join(text_lines).strip()
        
        if not text:
            print("❌ Document text cannot be empty.")
            return
        
        # Create document and insert
        new_doc = {
            "title": title,
            "text": text,
            "category": category
        }
        
        rag.insert_documents([new_doc])
        print("✅ Document added successfully!")
        
    except Exception as e:
        print(f"❌ Error adding document: {e}")

def show_collection_stats(rag: MilvusRAG):
    """Display collection statistics"""
    try:
        # Get collection info
        stats = rag.client.describe_collection(collection_name=rag.collection_name)
        print(f"\n📊 Collection Statistics")
        print("-" * 40)
        print(f"📁 Collection Name: {rag.collection_name}")
        print(f"🔢 Vector Dimension: {rag.embedding_dim}")
        print(f"📏 Metric Type: COSINE")
        print(f"🎯 Embedding Model: all-MiniLM-L6-v2")
        
        # Try to get document count (this might vary based on Milvus version)
        try:
            # Simple way to estimate count
            docs = rag.search_similar("", top_k=10000)
            print(f"📄 Estimated Documents: {len(docs)}")
        except:
            print("📄 Document count: Unable to determine")
            
    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")

def run_demo_queries(rag: MilvusRAG):
    """Run predefined demo queries"""
    demo_queries = [
        "What is machine learning?",
        "Tell me about Python programming",
        "How do vector databases work?",
        "What are data structures?",
        "Explain cloud computing"
    ]
    
    print("\n🧪 Running Demo Queries")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        # Get similar documents
        similar_docs = rag.search_similar(query, top_k=3)
        
        if similar_docs:
            print("🔍 Top 3 similar documents:")
            for j, doc in enumerate(similar_docs, 1):
                print(f"   {j}. {doc['title']} (Score: {doc['score']:.3f})")
                print(f"      📁 {doc['category']}")
                print(f"      📝 {doc['text'][:100]}...")
                print()
        else:
            print("❌ No similar documents found.")
        
        # Ask if user wants to continue
        if i < len(demo_queries):
            cont = input("Press Enter to continue to next query or 'q' to stop: ")
            if cont.lower() == 'q':
                break

def search_documents(rag: MilvusRAG):
    """Interactive document search"""
    print("\n🔍 Document Search")
    print("-" * 30)
    
    query = get_user_input("🔎 Enter your search query: ")
    
    # Get number of results
    try:
        top_k = int(get_user_input("📊 Number of results (default 5): ") or "5")
        top_k = max(1, min(top_k, 50))  # Limit between 1 and 50
    except:
        top_k = 5
    
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 50)
    
    # Search documents
    similar_docs = rag.search_similar(query, top_k=top_k)
    
    if not similar_docs:
        print("❌ No similar documents found.")
        return
    
    print(f"📄 Found {len(similar_docs)} similar documents:")
    print()
    
    for i, doc in enumerate(similar_docs, 1):
        print(f"{i}. 📄 {doc['title']}")
        print(f"   📁 Category: {doc['category']}")
        print(f"   🎯 Similarity Score: {doc['score']:.4f}")
        print(f"   📝 Text: {doc['text'][:200]}...")
        print()
    
    # Ask if user wants detailed response
    generate_response = input("\n🤖 Generate RAG response? (y/n): ").lower().strip()
    if generate_response == 'y':
        print("\n🤖 Generating response...")
        response = rag.generate_response(query, top_k=min(3, len(similar_docs)))
        print("\n" + "="*60)
        print("🤖 RAG RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)

def clear_database(rag: MilvusRAG):
    """Clear the database"""
    print("\n🗑️  Clear Database")
    print("-" * 30)
    
    confirm = input("⚠️  Are you sure you want to clear all data? (type 'YES' to confirm): ")
    
    if confirm == "YES":
        try:
            rag.client.drop_collection(collection_name=rag.collection_name)
            print("✅ Database cleared successfully!")
            print("🔄 Recreating collection...")
            rag.create_collection()
            print("✅ Collection recreated. You can now add new documents.")
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
    else:
        print("❌ Operation cancelled.")

def initialize_system():
    """Initialize the RAG system with demo data"""
    print("🚀 Initializing Milvus RAG System...")
    
    # Initialize RAG system
    rag = MilvusRAG("milvus_rag_db.db")
    
    # Create collection
    rag.create_collection()
    
    # Check if we need to load demo data
    try:
        existing_docs = rag.search_similar("", top_k=1)
        if not existing_docs:
            print("📚 Loading demo data...")
            demo_docs = create_demo_data()
            rag.insert_documents(demo_docs)
            print("✅ Demo data loaded successfully!")
        else:
            print("📚 Found existing data in database.")
    except:
        print("📚 Loading demo data...")
        demo_docs = create_demo_data()
        rag.insert_documents(demo_docs)
        print("✅ Demo data loaded successfully!")
    
    return rag

class Intent(Enum):
    """User intent classification"""
    SEARCH = "search"
    ADD_DOCUMENT = "add_document"
    VIEW_DOCUMENTS = "view_documents"
    STATS = "stats"
    HELP = "help"
    CLEAR = "clear"
    EXIT = "exit"
    UNKNOWN = "unknown"

@dataclass
class ParsedCommand:
    """Parsed user command"""
    intent: Intent
    query: str = ""
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class SimpleNLU:
    """Simple Natural Language Understanding for command parsing"""
    
    def __init__(self):
        self.intent_patterns = {
            Intent.SEARCH: [
                r"search\s+(.+)",
                r"find\s+(.+)",
                r"look\s+for\s+(.+)",
                r"what\s+(.+)",
                r"tell\s+me\s+about\s+(.+)",
                r"explain\s+(.+)",
                r"how\s+(.+)",
                r"why\s+(.+)",
                r"when\s+(.+)",
                r"where\s+(.+)",
                r"who\s+(.+)",
                r"(.+)\?$",  # Questions ending with ?
            ],
            Intent.ADD_DOCUMENT: [
                r"add\s+document\s+(.+)",
                r"add\s+(.+)",
                r"insert\s+(.+)",
                r"create\s+document\s+(.+)",
                r"new\s+document\s+(.+)",
            ],
            Intent.VIEW_DOCUMENTS: [
                r"view\s+documents",
                r"show\s+documents",
                r"list\s+documents",
                r"see\s+all\s+documents",
                r"show\s+all",
                r"list\s+all",
            ],
            Intent.STATS: [
                r"stats",
                r"statistics",
                r"info",
                r"information",
                r"show\s+stats",
                r"collection\s+info",
            ],
            Intent.HELP: [
                r"help",
                r"how\s+to\s+use",
                r"commands",
                r"what\s+can\s+you\s+do",
                r"instructions",
            ],
            Intent.CLEAR: [
                r"clear\s+database",
                r"clear\s+all",
                r"delete\s+all",
                r"reset\s+database",
                r"remove\s+all",
            ],
            Intent.EXIT: [
                r"exit",
                r"quit",
                r"bye",
                r"goodbye",
                r"stop",
            ],
        }
    
    def parse_command(self, text: str) -> ParsedCommand:
        """Parse user input into structured command"""
        text = text.strip().lower()
        
        if not text:
            return ParsedCommand(Intent.UNKNOWN)
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if match.groups():
                        query = match.group(1).strip()
                        return ParsedCommand(intent, query=query)
                    else:
                        return ParsedCommand(intent)
        
        # If no pattern matches, treat as search query
        return ParsedCommand(Intent.SEARCH, query=text)

class ChatbotRAG:
    """Chatbot interface for RAG system"""
    
    def __init__(self, rag_system: MilvusRAG):
        self.rag = rag_system
        self.nlu = SimpleNLU()
        self.conversation_history = []
        
    def process_command(self, user_input: str) -> str:
        """Process user command and return response"""
        parsed = self.nlu.parse_command(user_input)
        
        # Add to conversation history
        self.conversation_history.append({"user": user_input, "parsed": parsed})
        
        # Route to appropriate handler
        if parsed.intent == Intent.SEARCH:
            return self.handle_search(parsed)
        elif parsed.intent == Intent.ADD_DOCUMENT:
            return self.handle_add_document(parsed)
        elif parsed.intent == Intent.VIEW_DOCUMENTS:
            return self.handle_view_documents()
        elif parsed.intent == Intent.STATS:
            return self.handle_stats()
        elif parsed.intent == Intent.HELP:
            return self.handle_help()
        elif parsed.intent == Intent.CLEAR:
            return self.handle_clear()
        elif parsed.intent == Intent.EXIT:
            return "exit"
        else:
            return self.handle_unknown(parsed)
    
    def handle_search(self, parsed: ParsedCommand) -> str:
        """Handle search queries"""
        if not parsed.query:
            return "🤔 What would you like to search for? Please provide a search query."
        
        try:
            # Search for similar documents
            similar_docs = self.rag.search_similar(parsed.query, top_k=3)
            
            if not similar_docs:
                return f"❌ I couldn't find any documents related to '{parsed.query}'. Try rephrasing your query or add more documents to the collection."
            
            # Generate response using the most relevant documents
            response = self.generate_conversational_response(parsed.query, similar_docs)
            return response
            
        except Exception as e:
            return f"❌ Sorry, I encountered an error while searching: {str(e)}"
    
    def handle_add_document(self, parsed: ParsedCommand) -> str:
        """Handle document addition"""
        if not parsed.query:
            return "📄 To add a document, please provide the format: 'add document title: [title], category: [category], text: [content]'"
        
        # Simple parsing for document addition
        doc_match = re.search(r"title:\s*(.+?),\s*category:\s*(.+?),\s*text:\s*(.+)", parsed.query, re.IGNORECASE)
        if doc_match:
            title = doc_match.group(1).strip()
            category = doc_match.group(2).strip()
            text = doc_match.group(3).strip()
            
            try:
                new_doc = {
                    "title": title,
                    "category": category,
                    "text": text
                }
                self.rag.insert_documents([new_doc])
                return f"✅ Successfully added document '{title}' to the {category} category!"
            except Exception as e:
                return f"❌ Failed to add document: {str(e)}"
        else:
            return "📄 Please use the format: 'add document title: [title], category: [category], text: [content]'"
    
    def handle_view_documents(self) -> str:
        """Handle view documents request"""
        try:
            all_docs = self.rag.search_similar("", top_k=20)
            
            if not all_docs:
                return "📚 No documents found in the collection."
            
            response = f"📚 Found {len(all_docs)} documents:\n\n"
            for i, doc in enumerate(all_docs[:10], 1):  # Show first 10
                response += f"{i}. 📄 **{doc['title']}**\n"
                response += f"   📁 Category: {doc['category']}\n"
                response += f"   📝 Preview: {doc['text'][:100]}...\n\n"
            
            if len(all_docs) > 10:
                response += f"... and {len(all_docs) - 10} more documents."
            
            return response
            
        except Exception as e:
            return f"❌ Error retrieving documents: {str(e)}"
    
    def handle_stats(self) -> str:
        """Handle statistics request"""
        try:
            docs = self.rag.search_similar("", top_k=1000)
            doc_count = len(docs)
            
            # Count categories
            categories = {}
            for doc in docs:
                cat = doc['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            response = f"📊 **Collection Statistics:**\n\n"
            response += f"📁 Collection: {self.rag.collection_name}\n"
            response += f"📄 Total Documents: {doc_count}\n"
            response += f"🔢 Vector Dimension: {self.rag.embedding_dim}\n"
            response += f"🎯 Similarity Metric: COSINE\n\n"
            
            if categories:
                response += "📂 **Categories:**\n"
                for cat, count in sorted(categories.items()):
                    response += f"   • {cat}: {count} documents\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error getting statistics: {str(e)}"
    
    def handle_help(self) -> str:
        """Handle help request"""
        return """🤖 **RAG Chatbot Help**

I'm your AI assistant for searching and managing documents. Here's what I can do:

**🔍 Search & Questions:**
- "What is machine learning?"
- "Tell me about Python"
- "How do databases work?"
- "Find information about AI"

**📄 Document Management:**
- "Add document title: ML Guide, category: AI, text: Machine learning is..."
- "Show all documents"
- "List documents"

**📊 Information:**
- "Show stats" - Collection statistics
- "Info" - Database information

**🛠️ Utilities:**
- "Clear database" - Remove all documents
- "Help" - Show this help message
- "Exit" - Quit the application

**💡 Tips:**
- Ask natural questions - I'll understand!
- I can search through all your documents
- I provide answers based on your document collection
- Be specific for better results

Just type your question or command naturally!"""
    
    def handle_clear(self) -> str:
        """Handle clear database request"""
        return "⚠️ Are you sure you want to clear the database? Type 'yes clear database' to confirm."
    
    def handle_unknown(self, parsed: ParsedCommand) -> str:
        """Handle unknown commands"""
        return f"🤔 I'm not sure what you mean by '{parsed.query}'. Type 'help' to see what I can do, or just ask me a question about your documents!"
    
    def generate_conversational_response(self, query: str, docs: List[Dict]) -> str:
        """Generate a conversational response based on retrieved documents"""
        if not docs:
            return "❌ I couldn't find any relevant information."
        
        # Get the most relevant document
        top_doc = docs[0]
        
        # Create a conversational response
        response = f"🤖 Based on your question about '{query}', here's what I found:\n\n"
        
        # Main answer from top document
        response += f"📄 **{top_doc['title']}** (Relevance: {top_doc['score']:.2f})\n"
        response += f"📁 *{top_doc['category']}*\n\n"
        
        # Extract relevant portion of text
        doc_text = top_doc['text']
        if len(doc_text) > 300:
            doc_text = doc_text[:300] + "..."
        
        response += f"{doc_text}\n\n"
        
        # Add related documents if available
        if len(docs) > 1:
            response += "🔗 **Related information:**\n"
            for doc in docs[1:3]:  # Show up to 2 more documents
                response += f"   • {doc['title']} ({doc['category']})\n"
        
        response += "\n💡 *Ask me more specific questions for detailed information!*"
        
        return response

def run_chatbot_interface():
    """Run the chatbot interface"""
    print("🤖 RAG Chatbot Interface")
    print("=" * 50)
    print("Initializing system...")
    
    # Initialize RAG system
    rag = MilvusRAG("milvus_rag_db.db")
    rag.create_collection()
    
    # Load demo data if needed
    try:
        existing_docs = rag.search_similar("", top_k=1)
        if not existing_docs:
            print("📚 Loading demo data...")
            demo_docs = create_demo_data()
            rag.insert_documents(demo_docs)
            print("✅ Demo data loaded!")
    except:
        print("📚 Loading demo data...")
        demo_docs = create_demo_data()
        rag.insert_documents(demo_docs)
        print("✅ Demo data loaded!")
    
    # Initialize chatbot
    chatbot = ChatbotRAG(rag)
    
    print("\n🎉 Welcome to the RAG Chatbot!")
    print("Ask me anything about your documents or type 'help' for commands.")
    print("Type 'exit' to quit.\n")
    
    # Main chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle clear confirmation
            if user_input.lower() == "yes clear database":
                try:
                    rag.client.drop_collection(collection_name=rag.collection_name)
                    rag.create_collection()
                    print("🤖: ✅ Database cleared successfully! You can now add new documents.")
                    continue
                except Exception as e:
                    print(f"🤖: ❌ Error clearing database: {e}")
                    continue
            
            # Process command
            response = chatbot.process_command(user_input)
            
            # Handle exit
            if response == "exit":
                print("🤖: 👋 Goodbye! Thanks for using the RAG Chatbot!")
                break
            
            # Print response
            print(f"🤖: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n🤖: 👋 Goodbye!")
            break
        except Exception as e:
            print(f"🤖: ❌ Sorry, I encountered an error: {e}")

def main():
    """Main function with interface selection"""
    print("🚀 Milvus RAG System")
    print("=" * 30)
    print("Choose your interface:")
    print("1. 🤖 Chatbot Interface (Recommended)")
    print("2. 📋 Traditional CLI Menu")
    print("3. ❌ Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                run_chatbot_interface()
                break
            elif choice == "2":
                run_traditional_cli()
                break
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Please select 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def run_traditional_cli():
    """Run the traditional CLI interface"""
    try:
        # Initialize system
        rag = initialize_system()
        
        print("\n🎉 Welcome to the Milvus RAG System!")
        print("Type your queries and get AI-powered responses based on your document collection.")
        
        while True:
            display_menu()
            
            try:
                choice = get_user_input("🎯 Select an option (1-7): ", "int")
                
                if choice == 1:
                    search_documents(rag)
                elif choice == 2:
                    view_all_documents(rag)
                elif choice == 3:
                    add_new_document(rag)
                elif choice == 4:
                    show_collection_stats(rag)
                elif choice == 5:
                    run_demo_queries(rag)
                elif choice == 6:
                    clear_database(rag)
                elif choice == 7:
                    print("\n👋 Thank you for using Milvus RAG System!")
                    break
                else:
                    print("❌ Invalid option. Please select 1-7.")
                    
            except ValueError:
                print("❌ Please enter a valid number (1-7).")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
            # Pause before showing menu again
            input("\nPress Enter to continue...")
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your dependencies and try again.")

if __name__ == "__main__":
    main()