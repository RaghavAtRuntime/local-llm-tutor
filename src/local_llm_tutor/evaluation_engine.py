"""Evaluation Engine: Multi-layer answer evaluation."""

import re
from difflib import SequenceMatcher
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Holds the result of answer evaluation."""

    def __init__(self, verdict: str, score: float, exact_match: bool,
                 semantic_score: float, concept_coverage: float,
                 matched_concepts: list, missing_concepts: list):
        self.verdict = verdict  # "correct", "partial", "incorrect"
        self.score = score  # 0.0 to 1.0
        self.exact_match = exact_match
        self.semantic_score = semantic_score
        self.concept_coverage = concept_coverage
        self.matched_concepts = matched_concepts
        self.missing_concepts = missing_concepts

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "exact_match": self.exact_match,
            "semantic_score": self.semantic_score,
            "concept_coverage": self.concept_coverage,
            "matched_concepts": self.matched_concepts,
            "missing_concepts": self.missing_concepts,
        }


class EvaluationEngine:
    """Evaluates user answers using exact match, semantic similarity, and concept detection."""

    def __init__(self, threshold_correct: float = 0.75, threshold_partial: float = 0.45):
        self.threshold_correct = threshold_correct
        self.threshold_partial = threshold_partial
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load sentence transformer for semantic similarity."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded sentence-transformers model.")
            except ImportError:
                logger.warning("sentence-transformers not available; using fallback similarity.")
                self._embedder = "fallback"
        return self._embedder

    def check_exact_match(self, user_answer: str, expected_answer: str) -> bool:
        """Check if user answer is an exact or near-exact match."""
        normalize = lambda s: re.sub(r"[^\w\s]", "", s.lower()).strip()
        return normalize(user_answer) == normalize(expected_answer)

    def compute_semantic_similarity(self, user_answer: str, expected_answer: str) -> float:
        """Compute semantic similarity between answers."""
        embedder = self._get_embedder()
        if embedder == "fallback":
            return self._fallback_similarity(user_answer, expected_answer)
        try:
            import numpy as np
            embeddings = embedder.encode([user_answer, expected_answer])
            # Cosine similarity
            a, b = embeddings[0], embeddings[1]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return self._fallback_similarity(user_answer, expected_answer)

    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def detect_key_concepts(self, user_answer: str, key_concepts: list) -> tuple:
        """Detect which key concepts are present in the user answer."""
        answer_lower = user_answer.lower()
        matched = []
        missing = []
        for concept in key_concepts:
            if concept.lower() in answer_lower:
                matched.append(concept)
            else:
                missing.append(concept)
        coverage = len(matched) / len(key_concepts) if key_concepts else 1.0
        return matched, missing, coverage

    def evaluate(self, user_answer: str, question: dict) -> EvaluationResult:
        """Perform full evaluation of a user answer against a question."""
        expected = question.get("expected_answer", "")
        key_concepts = question.get("key_concepts", [])

        exact = self.check_exact_match(user_answer, expected)
        semantic_score = self.compute_semantic_similarity(user_answer, expected)
        matched, missing, concept_coverage = self.detect_key_concepts(user_answer, key_concepts)

        # Combined score: weighted average
        combined = 0.6 * semantic_score + 0.4 * concept_coverage

        if exact or semantic_score >= self.threshold_correct:
            verdict = "correct"
            score = min(1.0, combined + 0.1)
        elif semantic_score >= self.threshold_partial or concept_coverage >= 0.5:
            verdict = "partial"
            score = combined
        else:
            verdict = "incorrect"
            score = combined * 0.5

        logger.info(
            f"Evaluation: verdict={verdict}, semantic={semantic_score:.2f}, "
            f"concepts={concept_coverage:.2f}, score={score:.2f}"
        )

        return EvaluationResult(
            verdict=verdict,
            score=score,
            exact_match=exact,
            semantic_score=semantic_score,
            concept_coverage=concept_coverage,
            matched_concepts=matched,
            missing_concepts=missing,
        )
