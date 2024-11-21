# app/sessions/session_manager.py

import threading
import time
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, timeout=300):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.messages = {}
        self.last_activity = {}  # Pastikan atribut ini ada

    def add_message(self, session_id, message):
        with self.lock:
            if session_id not in self.messages:
                self.messages[session_id] = []
            self.messages[session_id].append(message)
            self.last_activity[session_id] = time.time()
            logger.info(f"Pesan ditambahkan ke sesi {session_id}: {message}")

    def reset_session(self, session_id):
        with self.lock:
            if session_id in self.messages:
                del self.messages[session_id]
            if session_id in self.last_activity:
                del self.last_activity[session_id]
            logger.info(f"Sesi {session_id} telah direset.")

