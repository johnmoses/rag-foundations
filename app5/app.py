import json
import threading
import time
import logging
import re

from flask import Flask, request, jsonify, render_template
import yfinance as yf
from llama_cpp import Llama

from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import SentenceTransformer
from pymilvus import (
    MilvusClient,
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

# nltk.download("punkt")

MILVUS_DB_URI = "milvus_rag_db.db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

milvus_client = MilvusClient(MILVUS_DB_URI)
connections.connect(alias="default", uri=MILVUS_DB_URI)


# --- Step 2: Create collection with primary key if not exists ---
def create_collection():
    if COLLECTION_NAME in milvus_client.list_collections():
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


def embed_text(text: str):
    # Dummy example: replace with your embedding model inference
    # For example, use sentence-transformers or OpenAI embeddings
    # Here we just return a fixed-size zero vector for demo purposes
    import numpy as np

    return np.random.rand(768).tolist()


app = Flask(__name__)

# --------- Model Loading ---------
MODEL_PATH = "/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"  # <-- Update this path!
llm = Llama(model_path=MODEL_PATH)

# --------- Logging Setup ---------
logger = logging.getLogger("financial_chatbot")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("financial_chatbot.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --------- Global Evaluation Results ---------
latest_eval_results = {}

# --------- Helper Functions ---------


def generate_response(prompt, max_tokens=150, temperature=0.7):
    start_time = time.time()
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["User:", "Assistant:"],
    )
    latency = time.time() - start_time
    response_text = output["choices"][0]["text"].strip()
    logger.info(f"Model inference latency: {latency:.3f} seconds")
    return response_text, latency


def build_fewshot_cot_prompt(user_query: str) -> str:
    system_prompt = """
[INST] <<SYS>>
You are a precise financial assistant. Your task is to classify the intent of a user query into one of these categories:
- get_stock_price
- get_historical_data
- calculate_interest
- get_compliance_docs
- compare_stock_prices
- general_chat

For each query, first explain your reasoning step-by-step, then output ONLY the intent label exactly as above.

If the query is ambiguous or does not fit any category, respond with "general_chat".

Be concise and accurate.
<</SYS>>
"""

    examples = """
User query: "What is the current price of AAPL stock?"
Reasoning: The user is asking about the current price of a specific stock ticker, so the intent is to get stock price.
Intent: get_stock_price

User query: "Show me the stock history of Tesla for the last month."
Reasoning: The user requests historical stock data for Tesla, so the intent is to get historical data.
Intent: get_historical_data

User query: "Calculate interest on 1000 dollars at 5% for 2 years."
Reasoning: The user wants to calculate interest based on principal, rate, and time, so the intent is calculate_interest.
Intent: calculate_interest

User query: "Where can I find GDPR compliance documents?"
Reasoning: The user is asking for compliance documents related to GDPR, so the intent is get_compliance_docs.
Intent: get_compliance_docs

User query: "Compare the prices of MSFT and GOOG stocks."
Reasoning: The user wants to compare prices of two stocks, so the intent is compare_stock_prices.
Intent: compare_stock_prices

User query: "Hello, how are you?"
Reasoning: This is a general greeting not related to finance, so the intent is general_chat.
Intent: general_chat

User query: "{user_query}"
Reasoning:"""

    return f"{system_prompt}{examples}"


def detect_intent_llama(user_query):
    prompt = build_fewshot_cot_prompt(user_query).replace("{user_query}", user_query)
    output_text, _ = generate_response(prompt, max_tokens=50, temperature=0.0)

    valid_intents = {
        "get_stock_price",
        "get_historical_data",
        "calculate_interest",
        "get_compliance_docs",
        "compare_stock_prices",
        "general_chat",
    }
    lines = output_text.split("\n")
    intent_line = None
    for line in reversed(lines):
        line_clean = line.strip().lower().replace(".", "")
        if line_clean in valid_intents:
            intent_line = line_clean
            break

    return intent_line if intent_line else "general_chat"


# --------- Financial Helper Functions ---------


def calculate_interest(principal, rate, time):
    try:
        p = float(principal)
        r = float(rate)
        t = float(time)
        interest = (p * r * t) / 100
        return f"Calculated simple interest is ${interest:.2f}"
    except Exception as e:
        return f"Error calculating interest: {str(e)}"


def get_stock_price(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        price = ticker_obj.history(period="1d")["Close"][-1]
        return f"The current price of {ticker.upper()} is ${price:.2f}"
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {str(e)}"


def compare_stock_prices(ticker1, ticker2):
    try:
        price1 = yf.Ticker(ticker1).history(period="1d")["Close"][-1]
        price2 = yf.Ticker(ticker2).history(period="1d")["Close"][-1]
        if price1 > price2:
            return f"{ticker1.upper()} (${price1:.2f}) is priced higher than {ticker2.upper()} (${price2:.2f})"
        elif price2 > price1:
            return f"{ticker2.upper()} (${price2:.2f}) is priced higher than {ticker1.upper()} (${price1:.2f})"
        else:
            return f"Both {ticker1.upper()} and {ticker2.upper()} have the same price of ${price1:.2f}"
    except Exception as e:
        return f"Error comparing stock prices: {str(e)}"


def get_compliance_docs(topic):
    docs = {
        "gdpr": "GDPR info: https://gdpr-info.eu/",
        "sox": "Sarbanes-Oxley Act info: https://www.soxlaw.com/",
        "basel": "Basel III framework: https://www.bis.org/bcbs/basel3.htm",
    }
    return docs.get(
        topic.lower(), "Compliance document not found for the specified topic."
    )


def get_historical_data(ticker, period="1mo"):
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period)
        if hist.empty:
            return f"No historical data available for {ticker} for period {period}."
        summary = hist[["Open", "High", "Low", "Close", "Volume"]].tail(5).to_dict()
        return f"Last 5 days of {ticker.upper()} historical  {summary}"
    except Exception as e:
        return f"Error fetching historical data for {ticker}: {str(e)}"


# --- RAG Integration ---
def generate_rag_response(user_query):
    # Embed the query
    query_embedding = embed_text(user_query)

    # Retrieve relevant docs from Milvus
    retrieved_docs = milvus_client.search(query_embedding, top_k=3)

    # Combine retrieved docs as context
    context = "\n\n".join(retrieved_docs)

    prompt = f"""You are a financial assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {user_query}

    Answer:"""

    response, _ = generate_response(prompt, max_tokens=150, temperature=0.7)
    return response


# --------- Advanced Evaluation Metrics ---------


def compute_bleu(candidate, reference):
    candidate_tokens = word_tokenize(candidate)
    reference_tokens = [word_tokenize(reference)]
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        reference_tokens, candidate_tokens, smoothing_function=smoothie
    )


def evaluate_model(dataset):
    total = len(dataset)
    correct_intent = 0
    response_matches = 0
    total_latency = 0.0

    candidates = []
    references = []

    for entry in dataset:
        query = entry["query"]
        expected_intent = entry.get("expected_intent")
        expected_response = entry.get("expected_response", "").lower()

        start = time.time()
        predicted_intent = detect_intent_llama(query)
        prompt = f"User: {query}\nAssistant:"
        response, latency = generate_response(prompt, max_tokens=150, temperature=0.0)
        response_lower = response.lower()
        total_latency += latency

        candidates.append(response_lower)
        references.append(expected_response)

        if predicted_intent == expected_intent:
            correct_intent += 1

        if expected_response in response_lower:
            response_matches += 1

        logger.info(f"Eval Query: {query}")
        logger.info(
            f"Expected Intent: {expected_intent}, Predicted Intent: {predicted_intent}"
        )
        logger.info(f"Latency: {latency:.3f}s")

    intent_accuracy = correct_intent / total if total else 0
    response_accuracy = response_matches / total if total else 0
    avg_latency = total_latency / total if total else 0

    # Compute BERTScore
    P, R, F1 = bert_score(candidates, references, lang="en", rescale_with_baseline=True)
    bert_precision = P.mean().item()
    bert_recall = R.mean().item()
    bert_f1 = F1.mean().item()

    # Compute BLEU
    bleu_scores = [compute_bleu(c, r) for c, r in zip(candidates, references)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    results = {
        "total_samples": total,
        "intent_accuracy": intent_accuracy,
        "response_accuracy": response_accuracy,
        "average_latency_seconds": avg_latency,
        "bert_precision": bert_precision,
        "bert_recall": bert_recall,
        "bert_f1": bert_f1,
        "average_bleu": avg_bleu,
    }

    logger.info(f"Evaluation Summary: {results}")
    return results


# --------- Periodic Evaluation Thread ---------


def load_offline_dataset(path="offline_test_data.json"):
    with open(path, "r") as f:
        return json.load(f)


def run_periodic_evaluation(interval_hours=0.003):
    def eval_loop():
        global latest_eval_results
        while True:
            logger.info("Starting periodic evaluation...")
            dataset = load_offline_dataset()
            latest_eval_results = evaluate_model(dataset)
            logger.info("Periodic evaluation completed.")
            time.sleep(interval_hours * 3600)  # Sleep for interval

    thread = threading.Thread(target=eval_loop, daemon=True)
    thread.start()


run_periodic_evaluation()

# --------- Flask Routes ---------


@app.route("/")
def index():
    return render_template("index.html")  # Your chat UI (see below)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    intent = detect_intent_llama(user_message)

    # Use RAG for general chat or compliance docs
    if intent in ["general_chat", "get_compliance_docs"]:
        response = generate_rag_response(user_message)
    else:
        tickers = re.findall(r'\b[A-Za-z]{1,5}\b', user_message.upper())
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", user_message)

        if intent == "get_stock_price":
            ticker = tickers[0] if tickers else None
            if ticker:
                response = get_stock_price(ticker)
            else:
                response = "Please specify a valid stock ticker symbol."
        elif intent == "get_historical_data":
            ticker = tickers[0] if tickers else None
            response = get_historical_data(ticker) if ticker else "Please specify a stock ticker symbol."
        elif intent == "compare_stock_prices":
            if len(tickers) >= 2:
                response = compare_stock_prices(tickers[0], tickers[1])
            else:
                response = "Please specify two stock ticker symbols to compare."
        elif intent == "calculate_interest":
            if len(numbers) >= 3:
                response = calculate_interest(numbers[0], numbers[1], numbers[2])
            else:
                response = "Please provide principal, rate, and time for interest calculation."
        else:
            # fallback general chat
            prompt = f"User: {user_message}\nAssistant:"
            response, _ = generate_response(prompt, max_tokens=150, temperature=0.7)

    return jsonify({"intent": intent, "response": response})


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation_results():
    if latest_eval_results:
        return jsonify(latest_eval_results)
    else:
        return jsonify({"message": "Evaluation results not available yet."}), 503


@app.route("/evaluation-dashboard")
def evaluation_dashboard():
    return render_template("evaluation_dashboard.html")


# --------- Run Flask App ---------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5001)
