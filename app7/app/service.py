from llama_cpp import Llama
import json
from .rag import MilvusRAG

rag = MilvusRAG("milvus_rag_db.db")

# # Create collection
rag.create_collection()

# # Seed db
rag.seed_data()

MODEL_PATH = "/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

class LlamaModelWrapper:
    _instance = None

    def __new__(cls, model_path=MODEL_PATH):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = Llama(model_path=model_path)
        return cls._instance

    def generate(self, prompt, max_tokens=512, temperature=0.3):
        resp = self.model(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        return resp['choices'][0]['text']

llama_model = LlamaModelWrapper()

def robust_json_extract(text):
    """Extract JSON array from text even if surrounded by extra text."""
    try:
        start = text.index('[')
        end = text.rindex(']') + 1  # include closing bracket
    except ValueError:
        print("No JSON array found in LLaMA output.")
        return []

    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []

def generate_flashcards(text, max_flashcards=5):
    prompt = f"""
    You are an assistant generating educational flashcards from the input text.

    Please respond ONLY with a valid JSON array of flashcards in this exact format:

    [
    {{
        "question": "Example question?",
        "answer": "Example answer."
    }}
    ]

    DO NOT include any explanations, comments, or any other text before or after the JSON array.

    Here is the text:

    \"\"\"{text}\"\"\"
    """
    raw_output = llama_model.generate(prompt)
    print("Raw LLaMA output:", raw_output)  # Debug logging
    return robust_json_extract(raw_output)



def generate_chat_response(user_message):
    prompt = f"You are a helpful assistant. Respond conversationally to the user message:\n{user_message}"
    return llama_model.generate(prompt).strip()

# app/llama_inference.py

def generate_ai_response(prompt: str, max_tokens=150, temperature=0.3) -> str:
    """
    Call your LLM inference model here.
    Replace this stub with your actual model integration.
    """
    # Example pseudo code - replace with your actual llama or HF pipeline call:
    # response = llama_model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    # return response.strip()

    # For demo, return dummy text
    return f"AI Bot response to: {prompt}"

# print(generate_ai_response("Hello AI!"))
