# app/config.py

import os

class Config:
    MODEL_INTENT_PATH = os.path.join('app', 'models', 'model_intent.keras')
    MODEL_NER_PATH = os.path.join('app', 'models', 'model_ner.keras')
    TOKENIZER_PATH = os.path.join('app', 'utils', 'tokenizer.pickle')
    LABEL_ENCODER_PATH = os.path.join('app', 'utils', 'label_encoder.pickle')
    NER_LABEL_ENCODER_PATH = os.path.join('app', 'utils', 'ner_label_encoder.pickle')
    VECTORIZER_PATH = os.path.join('app', 'utils', 'vectorizer.pickle')
    TFIDF_MATRIX_PATH = os.path.join('app', 'utils', 'tfidf_matrix.pickle')
    DATA_JSON_PATH = os.path.join('dataset', 'dataaa.json')
    SESSION_TIMEOUT = 300

