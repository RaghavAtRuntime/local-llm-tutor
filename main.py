#!/usr/bin/env python3
"""Demo entry point for the Local LLM Tutor."""

import argparse
import logging
import yaml
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Local LLM AI Tutor")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--questions", default="data/questions.json", help="Question bank path")
    parser.add_argument("--mode", choices=["sequential", "random", "difficulty"], default=None)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--text-only", action="store_true", help="Disable audio (text I/O only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    quiz_cfg = config.get("quiz", {})
    tts_cfg = config.get("tts", {})
    stt_cfg = config.get("stt", {})
    eval_cfg = config.get("evaluation", {})
    llm_cfg = config.get("llm", {})
    session_cfg = config.get("session", {})

    mode = args.mode or quiz_cfg.get("mode", "sequential")
    difficulty = args.difficulty or quiz_cfg.get("difficulty_filter")
    text_only = args.text_only

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from local_llm_tutor.quiz_engine import QuizEngine
    from local_llm_tutor.evaluation_engine import EvaluationEngine
    from local_llm_tutor.tts_module import TTSModule
    from local_llm_tutor.stt_module import STTModule
    from local_llm_tutor.feedback_generator import FeedbackGenerator
    from local_llm_tutor.session_manager import SessionManager
    from local_llm_tutor.audio_controller import AudioController
    from local_llm_tutor.llm_core import LLMCore
    from local_llm_tutor.tutor import Tutor

    quiz_engine = QuizEngine(
        question_bank_path=args.questions,
        mode=mode,
        difficulty_filter=difficulty,
        time_limit=quiz_cfg.get("time_limit"),
    )

    evaluation_engine = EvaluationEngine(
        threshold_correct=eval_cfg.get("semantic_threshold_correct", 0.75),
        threshold_partial=eval_cfg.get("semantic_threshold_partial", 0.45),
    )

    class DummyTTS:
        def speak(self, text, blocking=True): pass
        def stop(self): pass
        def is_speaking(self): return False

    if text_only:
        tts_module = DummyTTS()
    else:
        tts_module = TTSModule(
            rate=tts_cfg.get("rate", 150),
            volume=tts_cfg.get("volume", 1.0),
            voice_index=tts_cfg.get("voice_index", 0),
        )

    stt_module = STTModule(
        model_size=stt_cfg.get("model", "base"),
        language=stt_cfg.get("language", "en"),
        energy_threshold=stt_cfg.get("energy_threshold", 300),
    )

    llm_core = LLMCore(
        model=llm_cfg.get("model", "llama3.2"),
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        enabled=llm_cfg.get("enabled", False),
    )

    feedback_gen = FeedbackGenerator(llm_core=llm_core)
    session_mgr = SessionManager(db_path=session_cfg.get("db_path", "tutor_sessions.db"))
    audio_ctrl = AudioController(
        energy_threshold=stt_cfg.get("energy_threshold", 300),
    ) if not text_only else None

    tutor = Tutor(
        quiz_engine=quiz_engine,
        evaluation_engine=evaluation_engine,
        tts_module=tts_module,
        stt_module=stt_module,
        feedback_generator=feedback_gen,
        session_manager=session_mgr,
        audio_controller=audio_ctrl,
        llm_core=llm_core,
        text_only=text_only,
    )

    tutor.run()


if __name__ == "__main__":
    main()
