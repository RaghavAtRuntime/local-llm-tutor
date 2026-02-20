"""Feedback Generator: Adaptive feedback based on evaluation results."""

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

CORRECT_TEMPLATES = [
    "Excellent! That's correct. {reinforcement}",
    "Great job! You got it right. {reinforcement}",
    "Perfect! {reinforcement}",
    "Well done! That's exactly right. {reinforcement}",
]

PARTIAL_TEMPLATES = [
    "You're on the right track! {missing_hint} {hint}",
    "Partially correct! {missing_hint} {hint}",
    "Good start! You mentioned some key ideas, but {missing_hint} {hint}",
]

INCORRECT_TEMPLATES = [
    "Not quite. {explanation}",
    "Let me help you understand this better. {explanation}",
    "That's not right. {explanation}",
]

REINFORCEMENTS = [
    "Keep it up!",
    "You're doing well!",
    "Excellent understanding!",
    "That shows great comprehension!",
]


class FeedbackGenerator:
    """Generates adaptive feedback based on evaluation results."""

    def __init__(self, llm_core=None):
        self._llm = llm_core

    def generate(self, evaluation_result, question: dict) -> str:
        """Generate feedback text based on evaluation result."""
        verdict = evaluation_result.verdict

        if verdict == "correct":
            return self._correct_feedback(evaluation_result, question)
        elif verdict == "partial":
            return self._partial_feedback(evaluation_result, question)
        else:
            return self._incorrect_feedback(evaluation_result, question)

    def _correct_feedback(self, result, question: dict) -> str:
        reinforcement = random.choice(REINFORCEMENTS)
        template = random.choice(CORRECT_TEMPLATES)
        # Try LLM-enhanced feedback
        if self._llm and self._llm.is_available():
            try:
                enhanced = self._llm.generate_feedback(
                    verdict="correct",
                    question=question.get("question", ""),
                    expected=question.get("expected_answer", ""),
                )
                if enhanced:
                    return enhanced
            except Exception as e:
                logger.debug(f"LLM feedback failed, using template: {e}")
        return template.format(reinforcement=reinforcement)

    def _partial_feedback(self, result, question: dict) -> str:
        missing = result.missing_concepts
        if missing:
            missing_hint = f"You missed these key concepts: {', '.join(missing[:3])}."
        else:
            missing_hint = "Your answer was close but needs more detail."

        expected = question.get("expected_answer", "")
        hint = f"The complete answer should mention: {expected[:100]}..." if len(expected) > 100 else f"The answer is: {expected}"

        if self._llm and self._llm.is_available():
            try:
                enhanced = self._llm.generate_feedback(
                    verdict="partial",
                    question=question.get("question", ""),
                    expected=expected,
                    missing_concepts=missing,
                )
                if enhanced:
                    return enhanced
            except Exception as e:
                logger.debug(f"LLM feedback failed, using template: {e}")

        template = random.choice(PARTIAL_TEMPLATES)
        return template.format(missing_hint=missing_hint, hint=hint)

    def _incorrect_feedback(self, result, question: dict) -> str:
        expected = question.get("expected_answer", "")
        explanation = f"The correct answer is: {expected}"

        if self._llm and self._llm.is_available():
            try:
                enhanced = self._llm.generate_feedback(
                    verdict="incorrect",
                    question=question.get("question", ""),
                    expected=expected,
                )
                if enhanced:
                    return enhanced
            except Exception as e:
                logger.debug(f"LLM feedback failed, using template: {e}")

        template = random.choice(INCORRECT_TEMPLATES)
        return template.format(explanation=explanation)

    def generate_intro(self, question: dict, question_num: int, total: int) -> str:
        """Generate intro text for a question."""
        return (
            f"Question {question_num} of {total}. "
            f"{question.get('question', '')}"
        )

    def generate_session_summary(self, session_stats: dict) -> str:
        """Generate end-of-session summary."""
        total = session_stats.get("total_questions", 0)
        correct = session_stats.get("correct", 0)
        partial = session_stats.get("partial", 0)
        incorrect = session_stats.get("incorrect", 0)
        avg_score = session_stats.get("avg_score", 0.0)

        pct = (correct / total * 100) if total > 0 else 0
        summary = (
            f"Session complete! You answered {total} questions. "
            f"{correct} correct, {partial} partially correct, and {incorrect} incorrect. "
            f"Your overall score was {avg_score:.0%}. "
        )
        if pct >= 80:
            summary += "Outstanding performance!"
        elif pct >= 60:
            summary += "Good work! Keep practicing."
        else:
            summary += "Keep studying, you'll improve with practice!"
        return summary
