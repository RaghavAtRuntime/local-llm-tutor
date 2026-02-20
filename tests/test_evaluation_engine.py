"""Tests for the EvaluationEngine module."""
import pytest
from unittest.mock import patch
from local_llm_tutor.evaluation_engine import EvaluationEngine, EvaluationResult


SAMPLE_QUESTION = {
    "question_id": 1,
    "question": "What is Python?",
    "expected_answer": "Python is a high-level programming language.",
    "key_concepts": ["high-level", "programming", "language"],
    "difficulty": "easy",
}


@pytest.fixture
def engine():
    return EvaluationEngine(threshold_correct=0.75, threshold_partial=0.45)


def test_exact_match_correct(engine):
    assert engine.check_exact_match(
        "Python is a high-level programming language",
        "Python is a high-level programming language."
    ) is True


def test_exact_match_different(engine):
    assert engine.check_exact_match(
        "Python is a snake",
        "Python is a high-level programming language."
    ) is False


def test_fallback_similarity_identical(engine):
    score = engine._fallback_similarity("hello world", "hello world")
    assert score == 1.0


def test_fallback_similarity_different(engine):
    score = engine._fallback_similarity("apple", "orange")
    assert score < 0.5


def test_key_concept_detection_all_matched(engine):
    matched, missing, coverage = engine.detect_key_concepts(
        "Python is a high-level programming language",
        ["high-level", "programming", "language"]
    )
    assert set(matched) == {"high-level", "programming", "language"}
    assert missing == []
    assert coverage == 1.0


def test_key_concept_detection_partial(engine):
    matched, missing, coverage = engine.detect_key_concepts(
        "Python is a high-level language",
        ["high-level", "programming", "language"]
    )
    assert "programming" in missing
    assert coverage < 1.0


def test_key_concept_detection_empty(engine):
    matched, missing, coverage = engine.detect_key_concepts("anything", [])
    assert coverage == 1.0


def test_evaluate_correct_answer(engine):
    with patch.object(engine, "_get_embedder", return_value="fallback"):
        result = engine.evaluate(
            "Python is a high-level programming language",
            SAMPLE_QUESTION
        )
    assert isinstance(result, EvaluationResult)
    assert result.verdict in ("correct", "partial")  # might vary with fallback
    assert 0.0 <= result.score <= 1.0


def test_evaluate_incorrect_answer(engine):
    with patch.object(engine, "_get_embedder", return_value="fallback"):
        result = engine.evaluate("I don't know", SAMPLE_QUESTION)
    assert result.verdict in ("incorrect", "partial")


def test_evaluation_result_to_dict(engine):
    with patch.object(engine, "_get_embedder", return_value="fallback"):
        result = engine.evaluate("Python is a programming language", SAMPLE_QUESTION)
    d = result.to_dict()
    assert "verdict" in d
    assert "score" in d
    assert "semantic_score" in d
    assert "concept_coverage" in d
    assert "matched_concepts" in d
    assert "missing_concepts" in d
