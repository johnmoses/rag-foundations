from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from .models import User
from app import db, login_manager

auth_bp = Blueprint('auth', __name__, template_folder='../templates/auth')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('auth/register.html') # Show registration form

    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'Missing fields'}), 400
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'error': 'User exists'}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    # Redirect to the login page (which is now '/')
    # For a JSON response (for AJAX), send redirect URL
    return jsonify({"message": "User registered successfully!", "redirect": url_for('auth.login')}), 201

@auth_bp.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if current_user.is_authenticated:
            # If already logged in, redirect to flashcards
            return redirect(url_for('flashcards.list_packs'))
        # Otherwise, show login page
        return render_template('auth/login.html')

    # If it's a POST request to '/', it's assumed to be a login attempt
    # Handle login logic here, as previously defined
    data = request.json
    username_or_email = data.get('username_or_email')
    password = data.get('password')

    user = User.query.filter(
        (User.username == username_or_email) | (User.email == username_or_email)
    ).first()

    if user and user.check_password(password):
        login_user(user)
        return jsonify({"message": "Logged in", "redirect": url_for('flashcards.list_packs')})
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
