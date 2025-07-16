from app import db

class FlashcardPack(db.Model):
    __tablename__ = 'flashcard_packs'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_shared = db.Column(db.Boolean, default=False)
    share_token = db.Column(db.String(64), unique=True, nullable=True, index=True)

    user = db.relationship('User', backref=db.backref('flashcard_packs', lazy=True))
    flashcards = db.relationship('Flashcard', backref='pack', lazy=True, cascade='all, delete-orphan')

    def generate_share_token(self):
        import secrets
        self.share_token = secrets.token_urlsafe(32)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'is_shared': self.is_shared,
            'share_token': self.share_token
        }

class Flashcard(db.Model):
    __tablename__ = 'flashcards'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500), nullable=False)
    answer = db.Column(db.String(500), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pack_id = db.Column(db.Integer, db.ForeignKey('flashcard_packs.id'), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'pack_id': self.pack_id
        }
