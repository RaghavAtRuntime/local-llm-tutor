"""Tests for the FeedbackGenerator module."""
import pytest
from unittest.mock import MagicMock
from local_llm_tutor.feedback_generator import FeedbackGenerator
from local_llm_tutor.evaluation_engine import EvaluationResult


def make_result(verdict, score=0.8, semantic_score=0.8, concept_coverage=1.0,
                matched=None, missing=None):
    return EvaluationResult(
        verdict=verdict,
        score=score,
        exact_match=verdict == "correct",
        semantic_score=semantic_score,
        concept_coverage=concept_coverage,
        matched_concepts=matched or [],
        missing_concepts=missing or [],
    )


SAMPLE_QUESTION = {
    "question_id": 1,
    "question": "What is Python?",
    "expected_answer": "Python is a high-level programming language.",
    "key_concepts": ["high-level", "programming"],
    "difficulty": "easy",
}


@pytest.fixture
def generator():
    return FeedbackGenerator()


def test_correct_feedback_not_empty(generator):
    result = make_result("correct")
    feedback = generator.generate(result, SAMPLE_QUESTION)
    assert len(feedback) > 0
    assert isinstance(feedback, str)


def test_partial_feedback_mentions_missing(generator):
    result = make_result("partial", missing=["programming", "high-level"])
    feedback = generator.generate(result, SAMPLE_QUESTION)
    assert len(feedback) > 0


def test_incorrect_feedback_explains(generator):
    result = make_result("incorrect", score=0.1, semantic_score=0.1, concept_coverage=0.0)
    feedback = generator.generate(result, SAMPLE_QUESTION)
    assert len(feedback) > 0


def test_generate_intro(generator):
    intro = generator.generate_intro(SAMPLE_QUESTION, 1, 5)
    assert "1" in intro
    assert "5" in intro
    assert SAMPLE_QUESTION["question"] in intro


def test_session_summary_perfect_score(generator):
    stats = {"total_questions": 5, "correct": 5, "partial": 0, "incorrect": 0, "avg_score": 1.0}
    summary = generator.generate_session_summary(stats)
    assert "5" in summary
    assert len(summary) > 0


def test_session_summary_poor_score(generator):
    stats = {"total_questions": 5, "correct": 1, "partial": 1, "incorrect": 3, "avg_score": 0.3}
    summary = generator.generate_session_summary(stats)
    assert "5" in summary


def test_llm_unavailable_uses_template(generator):
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = False
    generator_with_llm = FeedbackGenerator(llm_core=mock_llm)
    result = make_result("correct")
    feedback = generator_with_llm.generate(result, SAMPLE_QUESTION)
    assert len(feedback) > 0
    mock_llm.generate_feedback.assert_not_called()
