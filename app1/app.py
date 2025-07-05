import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Step 1: Prepare knowledge base
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "The Eiffel Tower is located in Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "RAG stands for Retrieval-Augmented Generation.",
]


# Step 2: Preprocess text (tokenization)
def preprocess(text):
    return re.findall(r"\w+", text.lower())


# Step 3: Keyword matching retrieval
def keyword_match_retrieval(query, docs, top_k=2):
    query_tokens = set(preprocess(query))
    doc_scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(preprocess(doc))
        score = len(query_tokens.intersection(doc_tokens))
        doc_scores.append((score, i))
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = [docs[i] for score, i in doc_scores if score > 0][:top_k]
    return top_docs


# Step 4: Summarization function using Sumy LexRank
def summarize_text(text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)


# Step 5: Query and retrieve
query = "What is Paris?"
retrieved_docs = keyword_match_retrieval(query, documents, top_k=3)

# Step 6: Concatenate retrieved docs and summarize
combined_text = " ".join(retrieved_docs)
summary = summarize_text(combined_text, sentences_count=2)

print("Query:", query)
print("Retrieved documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i}. {doc}")
print("\nSummary:")
print(summary)
