"""Session Manager: Tracks user performance and stores history in SQLite."""

import sqlite3
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages quiz sessions and persists performance data to SQLite."""

    def __init__(self, db_path: str = "tutor_sessions.db"):
        self.db_path = db_path
        self.session_id = str(uuid.uuid4())
        self.session_start = time.time()
        self._results: List[Dict] = []
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                total_questions INTEGER,
                correct INTEGER,
                partial INTEGER,
                incorrect INTEGER,
                avg_score REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS question_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question_id INTEGER,
                question TEXT,
                user_answer TEXT,
                verdict TEXT,
                score REAL,
                response_time REAL,
                timestamp REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Session DB initialized at {self.db_path}")

    def record_answer(self, question: dict, user_answer: str,
                      evaluation_result, response_time: float):
        """Record a single question result."""
        result = {
            "session_id": self.session_id,
            "question_id": question.get("question_id"),
            "question": question.get("question", ""),
            "user_answer": user_answer,
            "verdict": evaluation_result.verdict,
            "score": evaluation_result.score,
            "response_time": response_time,
            "timestamp": time.time(),
        }
        self._results.append(result)
        self._save_result(result)

    def _save_result(self, result: dict):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                INSERT INTO question_results
                (session_id, question_id, question, user_answer, verdict, score,
                 response_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result["session_id"], result["question_id"], result["question"],
                result["user_answer"], result["verdict"], result["score"],
                result["response_time"], result["timestamp"],
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def get_stats(self) -> dict:
        """Get current session statistics."""
        total = len(self._results)
        correct = sum(1 for r in self._results if r["verdict"] == "correct")
        partial = sum(1 for r in self._results if r["verdict"] == "partial")
        incorrect = sum(1 for r in self._results if r["verdict"] == "incorrect")
        avg_score = sum(r["score"] for r in self._results) / total if total > 0 else 0.0
        avg_time = sum(r["response_time"] for r in self._results) / total if total > 0 else 0.0

        return {
            "session_id": self.session_id,
            "total_questions": total,
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "avg_score": avg_score,
            "avg_response_time": avg_time,
        }

    def finalize_session(self):
        """Save session summary to DB."""
        stats = self.get_stats()
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, start_time, end_time, total_questions, correct,
                 partial, incorrect, avg_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id, self.session_start, time.time(),
                stats["total_questions"], stats["correct"], stats["partial"],
                stats["incorrect"], stats["avg_score"],
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to finalize session: {e}")

    def get_weak_topics(self) -> List[str]:
        """Return topics where user performed poorly."""
        topic_scores: Dict[str, List[float]] = {}
        for r in self._results:
            # We'd need topic info - for now return empty
            pass
        return []

    def get_history(self) -> List[Dict]:
        """Return all results for this session."""
        return list(self._results)
