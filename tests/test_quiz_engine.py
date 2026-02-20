"""Tests for the QuizEngine module."""
import json
import tempfile
import os
import pytest
from local_llm_tutor.quiz_engine import QuizEngine


SAMPLE_QUESTIONS = [
    {
        "question_id": 1,
        "question": "What is Python?",
        "expected_answer": "Python is a high-level programming language.",
        "key_concepts": ["high-level", "programming"],
        "difficulty": "easy",
        "topic": "Python"
    },
    {
        "question_id": 2,
        "question": "What is a list?",
        "expected_answer": "A list is an ordered collection of items.",
        "key_concepts": ["ordered", "collection"],
        "difficulty": "easy",
        "topic": "Python"
    },
    {
        "question_id": 3,
        "question": "What is machine learning?",
        "expected_answer": "Machine learning is a branch of AI that learns from data.",
        "key_concepts": ["AI", "data", "learns"],
        "difficulty": "medium",
        "topic": "ML"
    },
]


@pytest.fixture
def question_bank_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_QUESTIONS, f)
        path = f.name
    yield path
    os.unlink(path)


def test_load_questions(question_bank_file):
    engine = QuizEngine(question_bank_file)
    assert engine.get_question_count() == 3


def test_sequential_mode(question_bank_file):
    engine = QuizEngine(question_bank_file, mode="sequential")
    q1 = engine.next_question()
    q2 = engine.next_question()
    assert q1["question_id"] == 1
    assert q2["question_id"] == 2


def test_has_next(question_bank_file):
    engine = QuizEngine(question_bank_file)
    assert engine.has_next() is True
    engine.next_question()
    engine.next_question()
    engine.next_question()
    assert engine.has_next() is False


def test_difficulty_filter(question_bank_file):
    engine = QuizEngine(question_bank_file, difficulty_filter="medium")
    assert engine.get_question_count() == 1
    q = engine.next_question()
    assert q["difficulty"] == "medium"


def test_reset(question_bank_file):
    engine = QuizEngine(question_bank_file)
    engine.next_question()
    engine.next_question()
    engine.reset()
    assert engine.get_current_index() == 0
    assert engine.has_next() is True


def test_no_timeout_by_default(question_bank_file):
    engine = QuizEngine(question_bank_file)
    engine.next_question()
    assert engine.is_timed_out() is False


def test_random_mode_returns_all(question_bank_file):
    engine = QuizEngine(question_bank_file, mode="random")
    questions = []
    while engine.has_next():
        questions.append(engine.next_question())
    assert len(questions) == 3
