from flask import Blueprint, request, jsonify, render_template, session
from flask_login import login_required
from ..services.service import rag, call_llm

chat_bp = Blueprint('chat', __name__, template_folder='../templates')

def get_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def save_chat_turn(user_msg, ai_msg):
    history = get_chat_history()
    history.append({'user': user_msg, 'ai': ai_msg})
    if len(history) > 10:
        history.pop(0)
    session['chat_history'] = history

def format_history_for_rag(history):
    """
    Convert [{'user': ..., 'ai': ...}, ...] to
    [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}, ...]
    """
    formatted = []
    for turn in history:
        formatted.append({'role': 'user', 'content': turn['user']})
        formatted.append({'role': 'assistant', 'content': turn['ai']})
    return formatted

@chat_bp.route('/', methods=['GET'])
@login_required
def chat_page():
    return render_template('chat.html')

@chat_bp.route('/message', methods=['POST'])
@login_required
def chat_message():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': "Please enter a message."})

    history = get_chat_history()
    formatted_history = format_history_for_rag(history)

    # Generate RAG-powered chat response
    ai_response = rag.chat(user_message, formatted_history, call_llm)

    save_chat_turn(user_message, ai_response)

    return jsonify({'response': ai_response})
