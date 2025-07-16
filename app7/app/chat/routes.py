from flask import Blueprint, render_template, request, redirect, url_for, jsonify, flash
from flask_login import login_required, current_user
from app import db, socketio
from app.chat.models import Room, Participant, Message
from app.auth.models import User
from app.service import llama_model, generate_ai_response
from flask_socketio import join_room, leave_room, emit
import threading


chat_bp = Blueprint('chat', __name__, template_folder='../templates/chat', static_folder='../static')

@chat_bp.route("/", methods=["GET", "POST"]) 
@login_required
def home():
    if request.method == "POST":
        room_code = request.form.get("room_code").strip()
        ai_enabled = "ai_enabled" in request.form

        if not room_code:
            flash("Room code required.", "error")
            return redirect(url_for("chat.home"))

        room = Room.query.filter_by(code=room_code).first()
        if not room:
            room = Room(code=room_code, ai_enabled=ai_enabled)
            db.session.add(room)
            db.session.commit()
        else:
            # Optionally update AI flag on existing room
            room.ai_enabled = ai_enabled
            db.session.commit()

        return redirect(url_for("chat.room", room_code=room_code))

    rooms = Room.query.order_by(Room.created_at.desc()).all()
    return render_template('chat.html', rooms=rooms)

@chat_bp.route('/room/<room_code>', methods=['GET'])
@login_required
def room(room_code):
    room = Room.query.filter_by(code=room_code).first_or_404()

    # Add user as participant if not already
    participant = Participant.query.filter_by(room=room, user_id=current_user.id).first()
    if not participant:
        participant = Participant(room=room, user_id=current_user.id)
        db.session.add(participant)
        db.session.commit()

    participants = User.query.join(Participant).filter(Participant.room==room).all()
    messages = Message.query.filter_by(room=room).order_by(Message.timestamp.asc()).all()
    return render_template('room.html', room=room, participants=participants, messages=messages)

@chat_bp.route('/create', methods=['POST'])
@login_required
def create_room():
    room_code = request.form.get('room_code', '').strip()
    if not room_code:
        flash('Room name is required.', 'error')
        return redirect(url_for('chat.home'))
    existing = Room.query.filter_by(code=room_code).first()
    if existing:
        flash('Room already exists. You can join it below.', 'warning')
        return redirect(url_for('chat.home'))
    room = Room(code=room_code)
    db.session.add(room)
    db.session.commit()
    return redirect(url_for('chat.join_room', room_code=room_code))


@chat_bp.route('/room/<room_code>/messages')
@login_required
def get_messages(room_code):
    room = Room.query.filter_by(code=room_code).first_or_404()
    messages = Message.query.filter_by(room=room).order_by(Message.timestamp.asc()).limit(100).all()
    return jsonify([m.to_dict() for m in messages])

@chat_bp.route('/join', methods=['POST'])
@login_required
def join_chat_room():  # renamed from join_room
    room_code = request.form.get('room_code', '').strip()
    if not room_code:
        flash('Room name required to join.', 'error')
        return redirect(url_for('chat.chat_home'))
    room = Room.query.filter_by(code=room_code).first()
    if not room:
        flash('Room does not exist.', 'error')
        return redirect(url_for('chat.chat_home'))
    return redirect(url_for('chat.room', room_code=room_code))


@chat_bp.route("/room/<room_code>/leave", methods=["POST"])
@login_required
def leave_chat_room(room_code):
    room = Room.query.filter_by(code=room_code).first_or_404()
    participant = Participant.query.filter_by(room=room, user_id=current_user.id).first()
    if participant:
        db.session.delete(participant)
        db.session.commit()
        flash(f"You left room {room_code}", "info")
    return redirect(url_for("chat.home"))

@chat_bp.route('/room/<room_code>/send', methods=['POST'])
@login_required
def send_message(room_code):
    room = Room.query.filter_by(code=room_code).first_or_404()
    content = request.form.get("message", "").strip()
    if not content:
        flash("Message cannot be empty.", "error")
        return redirect(url_for("chat.room", room_code=room_code))

    participant = Participant.query.filter_by(room=room, user_id=current_user.id).first()
    if not participant:
        flash("Join the room before sending messages.", "error")
        return redirect(url_for("chat.home"))

    msg = Message(room=room, user_id=current_user.id, content=content)
    db.session.add(msg)
    db.session.commit()

    # Broadcast message via websocket
    socketio.emit("message", {
        "username": current_user.username,
        "content": content,
        "timestamp": msg.timestamp.isoformat()
    }, room=room_code)

    # Async AI reply
    if room.ai_enabled:
        threading.Thread(target=send_ai_response_async, args=(room_code, content)).start()

    return redirect(url_for("chat.room", room_code=room_code))

def send_ai_response_async(room_code, user_message):
    from app import create_app
    app = create_app()
    with app.app_context():
        try:
            ai_reply = generate_ai_response(user_message)
            room = Room.query.filter_by(code=room_code).first()
            if not room:
                return
            bot_msg = Message(room=room, user_id=None, content=ai_reply)
            db.session.add(bot_msg)
            db.session.commit()
            socketio.emit("message", {
                "username": "AI Bot",
                "content": ai_reply,
                "timestamp": bot_msg.timestamp.isoformat()
            }, room=room_code)
            app.logger.info(f"AI responded in room {room_code}: {ai_reply}")
        except Exception as e:
            app.logger.error(f"Error in AI reply: {e}")

def generate_llama_response(prompt):
    # Replace with your llama_model.generate or async version
    response = llama_model.generate(prompt, max_tokens=256, temperature=0.3)
    return response.strip()


# Socket.IO events

@socketio.on('join')
@login_required
def handle_join(data):
    room_code = data.get('room')
    if room_code:
        join_room(room_code)
        emit('message', {
            'username': '',
            'content': f'{current_user.username} has joined the room.'
        }, room=room_code)

@socketio.on("leave")
@login_required
def on_leave(data):
    room_code = data.get("room")
    if not room_code:
        return
    leave_room(room_code)
    emit("message", {
        "username": "",
        "content": f"{current_user.username} has left the room."
    }, room=room_code)