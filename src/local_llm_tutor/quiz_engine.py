"""Quiz Engine: Loads and manages question banks."""

import json
import random
import time
from pathlib import Path
from typing import Optional


class QuizEngine:
    """Manages quiz questions from a JSON question bank."""

    def __init__(self, question_bank_path: str, mode: str = "sequential",
                 difficulty_filter: Optional[str] = None, time_limit: Optional[int] = None):
        self.questions = self._load_questions(question_bank_path)
        self.mode = mode
        self.difficulty_filter = difficulty_filter
        self.time_limit = time_limit
        self._filtered_questions = self._apply_filters()
        self._index = 0
        self._question_start_time: Optional[float] = None

    def _load_questions(self, path: str) -> list:
        with open(path, "r") as f:
            return json.load(f)

    def _apply_filters(self) -> list:
        questions = self.questions
        if self.difficulty_filter:
            questions = [q for q in questions if q.get("difficulty") == self.difficulty_filter]
        if self.mode == "random":
            questions = questions[:]
            random.shuffle(questions)
        return questions

    def reset(self):
        self._filtered_questions = self._apply_filters()
        self._index = 0

    def has_next(self) -> bool:
        return self._index < len(self._filtered_questions)

    def next_question(self) -> Optional[dict]:
        if not self.has_next():
            return None
        q = self._filtered_questions[self._index]
        self._index += 1
        self._question_start_time = time.time()
        return q

    def is_timed_out(self) -> bool:
        if self.time_limit is None or self._question_start_time is None:
            return False
        return (time.time() - self._question_start_time) > self.time_limit

    def get_elapsed_time(self) -> Optional[float]:
        if self._question_start_time is None:
            return None
        return time.time() - self._question_start_time

    def get_question_count(self) -> int:
        return len(self._filtered_questions)

    def get_current_index(self) -> int:
        return self._index
