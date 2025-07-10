import os
import logging
from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from rag import MilvusRAG

# --- Flask app setup ---
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system
rag = MilvusRAG("milvus_rag_db.db")

# Create collection
rag.create_collection()

# Seed db
rag.seed_db()

# --- Load Llama 3B model ---
model_path = os.path.expanduser("/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")  # Update path as needed
if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

llm = Llama(model_path=model_path, n_ctx=2048, temperature=0.7)

SYSTEM_CHAT_PROMPT = "You are a helpful healthcare assistant."

def build_rag_prompt(question: str, retrieved_docs: list):
    context = "\n\n".join([doc for doc, _ in retrieved_docs])
    system_prompt = "You are an AI assistant that answers questions based on the provided context."
    user_prompt = f"""
    Use the following context to answer the question. If the answer is not contained, say you don't know.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>
    """
    return f"{system_prompt}\n{user_prompt}"

def get_llm_response(prompt: str, max_tokens=512):
    try:
        response = llm(prompt=prompt, max_tokens=max_tokens)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"LLM inference error: {e}", exc_info=True)
        return "Sorry, I am unable to process your request at the moment."


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        user_input = data.get('message')
        chat_history = data.get('history', [])  # Expect list of {role, content} dicts, optional

        if not user_input:
            return jsonify({"error": "Message is required."}), 400

        # Build prompt with optional chat history (few-shot)
        prompt = SYSTEM_CHAT_PROMPT + "\n"
        for msg in chat_history:
            role = msg.get('role', 'user').capitalize()
            content = msg.get('content', '')
            prompt += f"{role}: {content}\n"
        prompt += f"User: {user_input}\nAssistant:"

        response = get_llm_response(prompt)

        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.get_json(force=True)
        symptoms = data.get('symptoms')
        history = data.get('history', 'No significant history provided.')
        if not symptoms:
            return jsonify({"error": "Symptoms are required."}), 400

        prompt = f"""
        You are a highly skilled medical diagnostic assistant. Given patient symptoms and history, provide 2-3 possible diagnoses with brief explanations and recommended next steps.
        Always include a strong disclaimer that this is NOT a substitute for professional medical advice.

        Patient Symptoms:
        {symptoms}

        Patient History:
        {history}

        Diagnosis and Recommendations:
        """
        response = get_llm_response(prompt)

        return jsonify({
            "diagnosis": response,
            "disclaimer": "This is a preliminary diagnosis and not a substitute for professional medical advice."
        })
    except Exception as e:
        logger.error(f"Error in /diagnose endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/prescribe', methods=['POST'])
def prescribe():
    try:
        data = request.get_json(force=True)
        diagnosis = data.get('diagnosis')
        if not diagnosis:
            return jsonify({"error": "Diagnosis is required."}), 400

        prompt = f"""
        You are a medical assistant providing general information on common prescriptions based on a diagnosis.
        Always include a disclaimer that prescriptions require a licensed physician's evaluation.

        Diagnosis:
        {diagnosis}

        Common prescription options:
        """
        response = get_llm_response(prompt)

        return jsonify({
            "prescription": response,
            "disclaimer": "These are common prescription suggestions and NOT a prescription. Consult a licensed physician."
        })
    except Exception as e:
        logger.error(f"Error in /prescribe endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/rag', methods=['POST'])
def rag():
    try:
        data = request.get_json(force=True)
        question = data.get('question')
        if not question:
            return jsonify({"error": "Question is required."}), 400

        retrieved_docs = rag.search_similar(question, top_k=3)
        prompt = build_rag_prompt(question, retrieved_docs)
        answer = get_llm_response(prompt)

        return jsonify({
            "answer": answer,
            "retrieved_docs": [doc for doc, _ in retrieved_docs]
        })
    except Exception as e:
        logger.error(f"Error in /rag endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
