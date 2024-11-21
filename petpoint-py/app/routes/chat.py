# app/routes/chat.py

from flask import Blueprint, request, jsonify
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import logging

from app.models import (
    model_intent,
    model_ner,
    tokenizer,
    label_encoder,
    ner_label_decoder,
    vectorizer,
    tfidf_matrix,
    session_manager,
    df_utterances  # Pastikan df_utterances diimpor
)

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@chat_bp.route('/chat/', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        logger.debug(f"Request data: {data}")
        if not data or 'text' not in data or 'session_id' not in data:
            logger.error(f"Invalid request data: {data}")
            return jsonify({"error": "Invalid request. 'text' and 'session_id' are required."}), 400

        user_text = data['text']
        session_id = data['session_id']

        logger.info(f"Received chat request: text='{user_text}', session_id='{session_id}'")

        # Clean the input text
        text_clean = clean_text(user_text)
        logger.debug(f"Cleaned text: '{text_clean}'")

        # Predict Intent
        if tokenizer is None or model_intent is None or label_encoder is None:
            logger.error("Tokenizer, model_intent, or label_encoder is not loaded.")
            return jsonify({"error": "Server configuration error."}), 500

        seq = tokenizer.texts_to_sequences([text_clean])
        padded_seq = pad_sequences(seq, maxlen=tfidf_matrix.shape[1], padding='post')
        logger.debug(f"Padded sequence: {padded_seq}")

        pred_intent = model_intent.predict(padded_seq)
        predicted_label = np.argmax(pred_intent, axis=1)[0]
        intent = label_encoder.inverse_transform([predicted_label])[0]
        logger.debug(f"Predicted intent: '{intent}'")

        # Predict Entities
        if model_ner is None or ner_label_decoder is None:
            logger.error("Model NER or ner_label_decoder is not loaded.")
            return jsonify({"error": "Server configuration error."}), 500

        pred_entities = model_ner.predict(padded_seq)
        pred_labels = np.argmax(pred_entities, axis=-1)[0]
        tokens = text_clean.split()
        entities = []
        for idx, label_id in enumerate(pred_labels[:len(tokens)]):
            label = ner_label_decoder.get(label_id, "O")
            if label != 'O':
                entity_type = label.split('-')[1]
                entities.append({"entity": entity_type, "value": tokens[idx]})
        logger.debug(f"Predicted entities: {entities}")

        # Get Response
        if vectorizer is None or tfidf_matrix is None:
            logger.error("Vectorizer or tfidf_matrix is not loaded.")
            return jsonify({"error": "Server configuration error."}), 500

        user_tfidf = vectorizer.transform([text_clean])
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)
        most_similar_idx = np.argmax(similarities[0])
        similarity_score = similarities[0][most_similar_idx]
        logger.debug(f"Most similar index: {most_similar_idx} with similarity score: {similarity_score}")

        if similarity_score < 0.35:
            response = "Maaf, saya tidak memahami pertanyaan Anda."
        else:
            try:
                response = df_utterances.iloc[most_similar_idx]['responses']
                logger.debug(f"Selected response: '{response}'")
            except IndexError:
                logger.error(f"No similar response found for index: {most_similar_idx}")
                response = "Maaf, saya tidak memahami pertanyaan Anda."

        # Tambahkan pesan ke sesi
        session_manager.add_message(session_id, {"user": user_text, "bot": response})

        logger.info(f"Sending chat response: intent='{intent}', entities={entities}, response='{response}'")

        return jsonify({"intent": intent, "entities": entities, "response": response}), 200

    except Exception as e:
        logger.exception(f"Error processing chat request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

