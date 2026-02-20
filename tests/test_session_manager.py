"""Tests for the SessionManager module."""
import pytest
from unittest.mock import MagicMock
from local_llm_tutor.session_manager import SessionManager
from local_llm_tutor.evaluation_engine import EvaluationResult


@pytest.fixture
def session(tmp_path):
    db_file = str(tmp_path / "test_sessions.db")
    return SessionManager(db_path=db_file)


def make_eval_result(verdict="correct", score=0.9):
    return EvaluationResult(
        verdict=verdict,
        score=score,
        exact_match=True,
        semantic_score=0.9,
        concept_coverage=1.0,
        matched_concepts=["concept1"],
        missing_concepts=[],
    )


SAMPLE_QUESTION = {
    "question_id": 1,
    "question": "What is Python?",
    "expected_answer": "Python is a high-level programming language.",
    "key_concepts": ["high-level"],
    "difficulty": "easy",
}


def test_session_init(session):
    assert session.session_id is not None
    assert len(session.session_id) > 0


def test_record_answer(session):
    result = make_eval_result("correct", 0.9)
    session.record_answer(SAMPLE_QUESTION, "Python is a language", result, 2.5)
    assert len(session.get_history()) == 1


def test_get_stats_empty(session):
    stats = session.get_stats()
    assert stats["total_questions"] == 0
    assert stats["correct"] == 0


def test_get_stats_with_answers(session):
    session.record_answer(SAMPLE_QUESTION, "answer", make_eval_result("correct"), 1.0)
    session.record_answer(SAMPLE_QUESTION, "answer", make_eval_result("partial", 0.5), 2.0)
    session.record_answer(SAMPLE_QUESTION, "answer", make_eval_result("incorrect", 0.1), 3.0)
    stats = session.get_stats()
    assert stats["total_questions"] == 3
    assert stats["correct"] == 1
    assert stats["partial"] == 1
    assert stats["incorrect"] == 1


def test_finalize_session(session):
    session.record_answer(SAMPLE_QUESTION, "answer", make_eval_result("correct"), 1.5)
    session.finalize_session()  # Should not raise


def test_multiple_sessions_different_ids(tmp_path):
    db_file = str(tmp_path / "test.db")
    s1 = SessionManager(db_path=db_file)
    s2 = SessionManager(db_path=db_file)
    assert s1.session_id != s2.session_id
