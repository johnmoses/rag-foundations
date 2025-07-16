from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_socketio import SocketIO

db = SQLAlchemy()
login_manager = LoginManager()
socketio = SocketIO()

def create_app():
    app = Flask(
        __name__,
    )
    app.config.from_object('config.Config')

    db.init_app(app)
    login_manager.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # Import blueprints
    from app.auth.routes import auth_bp
    from app.flashcards.routes import flashcards_bp
    from app.chat.routes import chat_bp

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(flashcards_bp, url_prefix='/flashcards')
    app.register_blueprint(chat_bp, url_prefix='/chat')

    # Custom 404 error handler
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    return app
