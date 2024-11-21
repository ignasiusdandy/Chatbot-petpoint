# app/main.py

import os
import logging
from flask import Flask, request, jsonify
from app.config import Config
from app.routes.chat import chat_bp
import threading
import time

# Import dari models.py
from app.models import (
    model_intent,
    model_ner,
    tokenizer,
    label_encoder,
    ner_label_decoder,
    vectorizer,
    tfidf_matrix,
    session_manager,
    df_utterances
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Register blueprint setelah mendefinisikan app
app.register_blueprint(chat_bp)

@app.route('/health/', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "message": "Chatbot API is running."}), 200

# Background thread untuk membersihkan sesi lama
def cleanup_sessions():
    while True:
        time.sleep(60)  # Cek setiap menit
        current_time = time.time()
        sessions_to_reset = []
        with session_manager.lock:
            for session_id, last_time in session_manager.last_activity.items():
                if current_time - last_time > session_manager.timeout:
                    sessions_to_reset.append(session_id)
        for session_id in sessions_to_reset:
            session_manager.reset_session(session_id)

# Start background thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

