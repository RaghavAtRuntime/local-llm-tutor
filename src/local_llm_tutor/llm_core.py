"""LLM Core: Local LLM integration via Ollama."""

import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)


class LLMCore:
    """Interfaces with a local LLM via Ollama for enhanced feedback generation."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434",
                 enabled: bool = False):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ollama LLM is accessible."""
        if not self.enabled:
            return False
        if self._available is not None:
            return self._available
        try:
            import urllib.request
            req = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2)
            self._available = req.status == 200
        except Exception:
            self._available = False
        return self._available

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response from the LLM."""
        if not self.is_available():
            return ""
        try:
            import urllib.request
            import json

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"LLM generate error: {e}")
            self._available = False
            return ""

    def generate_feedback(self, verdict: str, question: str, expected: str,
                          missing_concepts: Optional[list] = None) -> str:
        """Generate tutor feedback using the LLM."""
        missing_str = ""
        if missing_concepts:
            missing_str = f"Missing concepts: {', '.join(missing_concepts)}."

        prompt = (
            f"You are an encouraging AI tutor. The student answered a question.\n"
            f"Question: {question}\n"
            f"Expected answer: {expected}\n"
            f"Verdict: {verdict}\n"
            f"{missing_str}\n"
            f"Generate brief (2-3 sentence), encouraging feedback appropriate for the verdict."
        )
        return self.generate(prompt)

    def generate_explanation(self, question: str, expected: str) -> str:
        """Generate a detailed explanation."""
        prompt = (
            f"Explain the following concept clearly and concisely:\n"
            f"Question: {question}\n"
            f"Answer: {expected}\n"
            f"Provide a 2-3 sentence explanation suitable for a student."
        )
        return self.generate(prompt)
