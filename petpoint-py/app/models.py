# app/models.py

import os
import pickle
import logging
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

from app.config import Config
from app.sessions.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Tentukan path absolut berdasarkan direktori file ini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, Config.DATA_JSON_PATH)

# Load df_utterances dari JSON
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df_utterances = df[['utterances', 'responses']].reset_index(drop=True)
    logger.info("df_utterances berhasil dimuat dari JSON.")
except FileNotFoundError:
    logger.error(f"File {data_path} tidak ditemukan.")
    df_utterances = pd.DataFrame()  # Atau handle sesuai kebutuhan
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat df_utterances: {e}")
    df_utterances = pd.DataFrame()

# Load Model Intent
try:
    model_intent = load_model(Config.MODEL_INTENT_PATH)
    logger.info("Model intent berhasil dimuat.")
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat model intent: {e}")
    model_intent = None

# Load Model NER
try:
    model_ner = load_model(Config.MODEL_NER_PATH)
    logger.info("Model NER berhasil dimuat.")
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat model NER: {e}")
    model_ner = None

# Load Tokenizer
try:
    with open(Config.TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    logger.info("Tokenizer berhasil dimuat.")
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat tokenizer: {e}")
    tokenizer = None

# Load Label Encoder
try:
    with open(Config.LABEL_ENCODER_PATH, 'rb') as handle:
        label_encoder = pickle.load(handle)
    logger.info("Label encoder berhasil dimuat.")
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat label encoder: {e}")
    label_encoder = None

# Load NER Label Encoder
try:
    with open(Config.NER_LABEL_ENCODER_PATH, 'rb') as handle:
        ner_label_encoder = pickle.load(handle)
    ner_label_decoder = {idx: label for label, idx in ner_label_encoder.items()}
    logger.info("NER label encoder berhasil dimuat.")
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat NER label encoder: {e}")
    ner_label_decoder = {}

# Load Vectorizer
try:
    with open(Config.VECTORIZER_PATH, 'rb') as handle:
        vectorizer = pickle.load(handle)
    logger.info("Vectorizer berhasil dimuat.")
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat vectorizer: {e}")
    vectorizer = None

# Load TF-IDF Matrix
try:
    with open(Config.TFIDF_MATRIX_PATH, 'rb') as handle:
        tfidf_matrix = pickle.load(handle)
    logger.info("TF-IDF matrix berhasil dimuat.")
except Exception as e:
    logger.error(f"Terjadi kesalahan saat memuat TF-IDF matrix: {e}")
    tfidf_matrix = None

# Initialize SessionManager
session_manager = SessionManager(timeout=Config.SESSION_TIMEOUT)

