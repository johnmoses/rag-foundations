from flask import Blueprint, request, jsonify, render_template, url_for
from flask_login import login_required, current_user
from app import db
from app.flashcards.models import Flashcard, FlashcardPack
from app.service import generate_flashcards, llama_model, robust_json_extract

flashcards_bp = Blueprint(
    "flashcards", __name__, template_folder="../templates/flashcards"
)


# List packs for current user
@flashcards_bp.route("/packs")
@login_required
def list_packs():
    # Fetch packs here, pass data if needed
    packs = FlashcardPack.query.filter_by(user_id=current_user.id).all()
    return render_template("flashcards/packs.html", packs=packs)


# Create new pack
@flashcards_bp.route("/packs", methods=["POST"])
@login_required
def create_pack():
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Pack name required"}), 400
    pack = FlashcardPack(name=name, user_id=current_user.id)
    db.session.add(pack)
    db.session.commit()
    return jsonify(pack.to_dict()), 201


# List flashcards in pack
@flashcards_bp.route("/packs/<int:pack_id>/flashcards", methods=["GET"])
@login_required
def get_flashcards_in_pack(pack_id):
    pack = FlashcardPack.query.filter_by(
        id=pack_id, user_id=current_user.id
    ).first_or_404()
    cards = Flashcard.query.filter_by(pack_id=pack_id, user_id=current_user.id).all()
    return jsonify([c.to_dict() for c in cards])


# Add flashcard to pack
@flashcards_bp.route("/packs/<int:pack_id>/flashcards", methods=["POST"])
@login_required
def add_flashcard_to_pack(pack_id):
    pack = FlashcardPack.query.filter_by(
        id=pack_id, user_id=current_user.id
    ).first_or_404()
    data = request.json
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    if not question or not answer:
        return jsonify({"error": "Question and answer required"}), 400
    flashcard = Flashcard(
        question=question, answer=answer, user_id=current_user.id, pack_id=pack_id
    )
    db.session.add(flashcard)
    db.session.commit()
    return jsonify(flashcard.to_dict()), 201


# Generate flashcards for a pack from text (uses your LLaMA function)
@flashcards_bp.route("/packs/<int:pack_id>/generate", methods=["POST"])
@login_required
def generate_for_pack(pack_id):
    pack = FlashcardPack.query.filter_by(
        id=pack_id, user_id=current_user.id
    ).first_or_404()

    data = request.json
    text = data.get("text", "").strip()
    num_cards = int(data.get("num_flashcards", 5))
    if not text:
        return jsonify({"error": "Input text is required"}), 400

    flashcards = generate_flashcards(text, max_flashcards=num_cards)

    if not flashcards:
        return (
            jsonify(
                {
                    "generated_flashcards": [],
                    "message": "No flashcards generated. Please try with more detailed text.",
                }
            ),
            200,
        )

    saved_cards = []
    for card in flashcards:
        q = card.get("question")
        a = card.get("answer")
        if q and a:
            existing = Flashcard.query.filter_by(
                user_id=current_user.id, question=q, answer=a, pack_id=pack_id
            ).first()
            if not existing:
                new_card = Flashcard(
                    question=q, answer=a, user_id=current_user.id, pack_id=pack_id
                )
                db.session.add(new_card)
                saved_cards.append(new_card)
            else:
                saved_cards.append(existing)

    db.session.commit()

    return (
        jsonify(
            {
                "generated_flashcards": [card.to_dict() for card in saved_cards],
                "message": f"Generated and saved {len(saved_cards)} flashcards.",
            }
        ),
        201,
    )


@flashcards_bp.route("/packs/<int:pack_id>/generate-from-topic", methods=["POST"])
@login_required
def generate_flashcards_from_topic(pack_id):
    """
    Generates flashcards based on a given topic using the LLaMA model
    and saves them to the specified flashcard pack.
    """
    pack = FlashcardPack.query.filter_by(
        id=pack_id, user_id=current_user.id
    ).first_or_404()

    data = request.json
    topic = data.get("topic", "").strip()
    num_flashcards = int(data.get("num_flashcards", 5))  # Default to 5 if not provided

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    # Ensure num_flashcards is within a reasonable range to prevent
    # excessive generation or very short/long model outputs.
    if num_flashcards < 1:
        num_flashcards = 1
    elif num_flashcards > 20:  # Cap at 20 or another suitable max
        num_flashcards = 20

    # Craft the specific prompt for topic-based flashcard generation
    # This prompt is designed to be very clear to the LLM about the task.
    prompt = f"""
    You are an expert educational assistant specializing in creating concise and effective flashcards.
    Your task is to generate {num_flashcards} flashcards based ONLY on the topic: "{topic}".

    Each flashcard must consist of a clear "question" and a concise "answer".
    The output must be a JSON array.

    Strict Output Format:
    Respond ONLY with the JSON array. Do NOT include any explanations, greetings, or other text outside the JSON.
    The JSON array should look exactly like this example (with your generated content):

    [
    {{
        "question": "What is the primary function of photosynthesis?",
        "answer": "To convert light energy into chemical energy in plants."
    }},
    {{
        "question": "Who invented the light bulb?",
        "answer": "Thomas Edison."
    }}
    ]


    Topic for Flashcards: {topic}
    Number of Flashcards to Generate: {num_flashcards}
    """

    # Call the LLaMA model for generation
    # Adjust max_tokens based on expected length of generated flashcards
    # A lower temperature (e.g., 0.3-0.5) is usually better for structured output
    raw_output = llama_model.generate(
        prompt, max_tokens=num_flashcards * 100, temperature=0.5
    )  # Estimate max_tokens

    # Robustly extract JSON from the raw model output
    generated_flashcards = robust_json_extract(raw_output)

    if not generated_flashcards:
        # If no valid flashcards were extracted, return a message to the user
        return (
            jsonify(
                {
                    "generated_flashcards": [],
                    "message": "No flashcards could be generated for this topic. Please try a different topic or be more specific.",
                }
            ),
            200,
        )

    saved_cards = []
    for card_data in generated_flashcards:
        question = card_data.get("question")
        answer = card_data.get("answer")

        if question and answer:
            # Check for existing flashcard to avoid duplicates within the pack for the current user
            existing_flashcard = Flashcard.query.filter_by(
                user_id=current_user.id,
                pack_id=pack_id,
                question=question,
                answer=answer,
            ).first()

            if not existing_flashcard:
                # Create and add new flashcard to the database
                new_flashcard = Flashcard(
                    question=question,
                    answer=answer,
                    user_id=current_user.id,
                    pack_id=pack_id,
                )
                db.session.add(new_flashcard)
                saved_cards.append(new_flashcard)
            else:
                # If a duplicate exists, just include it in the response without re-saving
                saved_cards.append(existing_flashcard)

    db.session.commit()  # Commit all new flashcards in one transaction

    # Return the list of saved flashcards as JSON response
    return (
        jsonify(
            {
                "generated_flashcards": [card.to_dict() for card in saved_cards],
                "message": f"Successfully generated and saved {len(saved_cards)} flashcards for topic '{topic}'.",
            }
        ),
        201,
    )


# Share pack: generate share token and URL
@flashcards_bp.route("/packs/<int:pack_id>/share", methods=["POST"])
@login_required
def share_pack(pack_id):
    pack = FlashcardPack.query.filter_by(
        id=pack_id, user_id=current_user.id
    ).first_or_404()
    if not pack.is_shared or not pack.share_token:
        pack.is_shared = True
        pack.generate_share_token()
        db.session.commit()
    share_url = url_for(
        "flashcards.view_shared_pack", token=pack.share_token, _external=True
    )
    return jsonify({"share_url": share_url})


# View shared pack publicly with token
@flashcards_bp.route("/shared/<string:token>")
def view_shared_pack(token):
    pack = FlashcardPack.query.filter_by(
        share_token=token, is_shared=True
    ).first_or_404()
    flashcards = Flashcard.query.filter_by(pack_id=pack.id).all()
    return render_template(
        "flashcards/shared_pack.html", pack=pack, flashcards=flashcards
    )


# Update flashcard
@flashcards_bp.route("/<int:flashcard_id>", methods=["PUT"])
@login_required
def update_flashcard(flashcard_id):
    flashcard = Flashcard.query.filter_by(
        id=flashcard_id, user_id=current_user.id
    ).first()
    if not flashcard:
        return jsonify({"error": "Flashcard not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON in request"}), 400

    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    if not answer:
        return jsonify({"error": "Answer cannot be empty"}), 400

    flashcard.question = question
    flashcard.answer = answer

    try:
        db.session.commit()
    except Exception as e:
        # Log the exception if you have logging configured
        print(f"Database commit error: {e}")
        db.session.rollback()
        return (
            jsonify({"error": "Failed to update flashcard due to a server error"}),
            500,
        )

    return jsonify(flashcard.to_dict()), 200


# Delete flashcard
@flashcards_bp.route("/<int:flashcard_id>", methods=["DELETE"])
@login_required
def delete_flashcard(flashcard_id):
    flashcard = Flashcard.query.filter_by(
        id=flashcard_id, user_id=current_user.id
    ).first_or_404()
    db.session.delete(flashcard)
    db.session.commit()
    return jsonify({"message": "Flashcard deleted"}), 200
