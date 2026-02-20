"""Main Tutor Orchestrator: Coordinates all modules for the full interaction loop."""

import logging
import time
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Tutor:
    """
    Main AI Tutor class that orchestrates the full quiz interaction loop.
    Coordinates TTS, STT, Quiz Engine, Evaluation, Feedback, and Session Management.
    """

    def __init__(
        self,
        quiz_engine,
        evaluation_engine,
        tts_module,
        stt_module,
        feedback_generator,
        session_manager,
        audio_controller=None,
        llm_core=None,
        text_only: bool = False,
    ):
        self.quiz = quiz_engine
        self.evaluator = evaluation_engine
        self.tts = tts_module
        self.stt = stt_module
        self.feedback = feedback_generator
        self.session = session_manager
        self.audio = audio_controller
        self.llm = llm_core
        self.text_only = text_only
        self._running = False

    def speak(self, text: str):
        """Output text via TTS or print if text-only mode."""
        print(f"\n[Tutor]: {text}")
        if not self.text_only:
            self.tts.speak(text)

    def listen(self) -> str:
        """Get user input via STT or keyboard if text-only mode."""
        if self.text_only:
            try:
                response = input("\n[You]: ").strip()
                return response
            except (EOFError, KeyboardInterrupt):
                return ""

        print("\n[Listening... speak your answer]")
        if self.audio:
            audio_data = self.audio.record_with_vad()
            result = self.stt.transcribe_audio(audio_data)
        else:
            result = self.stt.listen_and_transcribe()

        if result.text:
            print(f"\n[Transcribed]: {result.text}")
            logger.info(f"STT: '{result.text}' (confidence: {result.confidence:.2f})")
        return result.text

    def handle_special_commands(self, text: str, question: dict) -> Optional[str]:
        """Handle special voice commands like 'repeat', 'explain more', etc."""
        lower = text.lower().strip()
        if any(cmd in lower for cmd in ["repeat", "repeat the question", "say again"]):
            self.speak(f"Repeating: {question.get('question', '')}")
            return "repeat"
        if any(cmd in lower for cmd in ["explain more", "explain", "give example"]):
            if self.llm and self.llm.is_available():
                explanation = self.llm.generate_explanation(
                    question.get("question", ""),
                    question.get("expected_answer", "")
                )
                if explanation:
                    self.speak(explanation)
                    return "explained"
            self.speak(f"The answer to this question involves: {question.get('expected_answer', '')}")
            return "explained"
        if any(cmd in lower for cmd in ["skip", "next", "pass"]):
            return "skip"
        if any(cmd in lower for cmd in ["quit", "exit", "stop"]):
            return "quit"
        return None

    def run_question(self, question: dict, question_num: int, total: int) -> bool:
        """Run a single question interaction. Returns False if session should end."""
        intro = self.feedback.generate_intro(question, question_num, total)
        self.speak(intro)

        start_time = time.time()
        max_retries = 2

        for attempt in range(max_retries + 1):
            user_answer = self.listen()
            response_time = time.time() - start_time

            if not user_answer:
                if attempt < max_retries:
                    self.speak("I didn't catch that. Please try again.")
                    continue
                else:
                    self.speak("I couldn't hear your answer. Moving on.")
                    return True

            # Handle special commands
            command = self.handle_special_commands(user_answer, question)
            if command == "repeat":
                continue
            if command == "quit":
                return False
            if command == "skip":
                self.speak("Skipping this question.")
                return True
            if command == "explained":
                continue

            # Evaluate answer
            result = self.evaluator.evaluate(user_answer, question)
            logger.info(f"Evaluation: {result.to_dict()}")

            # Record result
            self.session.record_answer(question, user_answer, result, response_time)

            # Generate and speak feedback
            feedback_text = self.feedback.generate(result, question)
            self.speak(feedback_text)

            # Log scores
            print(f"  [Score: {result.score:.0%} | Verdict: {result.verdict} | "
                  f"Semantic: {result.semantic_score:.2f}]")
            return True

        return True

    def run(self):
        """Run the full tutoring session."""
        self._running = True
        total = self.quiz.get_question_count()

        self.speak(
            f"Welcome to the AI Tutor! Today's session has {total} questions. "
            f"Let's begin!"
        )

        question_num = 0
        while self.quiz.has_next() and self._running:
            question = self.quiz.next_question()
            if question is None:
                break
            question_num += 1

            should_continue = self.run_question(question, question_num, total)
            if not should_continue:
                self.speak("Ending session early. Goodbye!")
                break

            # Brief pause between questions
            if self.quiz.has_next():
                time.sleep(0.5)

        # Finalize and summarize
        self.session.finalize_session()
        stats = self.session.get_stats()
        summary = self.feedback.generate_session_summary(stats)
        self.speak(summary)

        logger.info(f"Session complete: {stats}")
        self._running = False
        return stats
