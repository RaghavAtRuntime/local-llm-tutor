"""Tests for the Tutor orchestrator module."""
import pytest
from unittest.mock import MagicMock, patch, call
from local_llm_tutor.tutor import Tutor
from local_llm_tutor.evaluation_engine import EvaluationResult


SAMPLE_QUESTION = {
    "question_id": 1,
    "question": "What is Python?",
    "expected_answer": "Python is a high-level programming language.",
    "key_concepts": ["high-level", "programming"],
    "difficulty": "easy",
}


def make_eval_result(verdict="correct", score=0.9):
    """Factory for EvaluationResult test objects with configurable verdict and score."""
    return EvaluationResult(
        verdict=verdict,
        score=score,
        exact_match=(verdict == "correct"),
        semantic_score=0.9,
        concept_coverage=1.0,
        matched_concepts=["high-level"],
        missing_concepts=[],
    )


@pytest.fixture
def tutor():
    quiz = MagicMock()
    evaluator = MagicMock()
    tts = MagicMock()
    stt = MagicMock()
    feedback = MagicMock()
    session = MagicMock()

    tutor_obj = Tutor(
        quiz_engine=quiz,
        evaluation_engine=evaluator,
        tts_module=tts,
        stt_module=stt,
        feedback_generator=feedback,
        session_manager=session,
        text_only=True,
    )
    tutor_obj.speak = MagicMock()
    return tutor_obj


# --- Tests for handle_special_commands ---

def test_handle_quit_command(tutor):
    assert tutor.handle_special_commands("quit", SAMPLE_QUESTION) == "quit"


def test_handle_exit_command(tutor):
    assert tutor.handle_special_commands("exit", SAMPLE_QUESTION) == "quit"


def test_handle_stop_command(tutor):
    assert tutor.handle_special_commands("stop", SAMPLE_QUESTION) == "quit"


def test_handle_next_command(tutor):
    assert tutor.handle_special_commands("next", SAMPLE_QUESTION) == "skip"


def test_handle_skip_command(tutor):
    assert tutor.handle_special_commands("skip", SAMPLE_QUESTION) == "skip"


def test_handle_pass_command(tutor):
    assert tutor.handle_special_commands("pass", SAMPLE_QUESTION) == "skip"


def test_handle_repeat_command(tutor):
    assert tutor.handle_special_commands("repeat", SAMPLE_QUESTION) == "repeat"


def test_handle_no_command(tutor):
    assert tutor.handle_special_commands("Python is a language", SAMPLE_QUESTION) is None


def test_handle_command_case_insensitive(tutor):
    assert tutor.handle_special_commands("QUIT", SAMPLE_QUESTION) == "quit"
    assert tutor.handle_special_commands("Next", SAMPLE_QUESTION) == "skip"
    assert tutor.handle_special_commands("EXIT", SAMPLE_QUESTION) == "quit"


def test_handle_command_with_punctuation(tutor):
    """Commands with trailing punctuation (e.g. from STT) should still match."""
    assert tutor.handle_special_commands("quit.", SAMPLE_QUESTION) == "quit"
    assert tutor.handle_special_commands("next.", SAMPLE_QUESTION) == "skip"
    assert tutor.handle_special_commands("exit!", SAMPLE_QUESTION) == "quit"


# --- Tests for run_question ---

def test_run_question_quit_command_ends_session(tutor):
    """Saying 'quit' should return False to end the session."""
    tutor.listen = MagicMock(return_value="quit")
    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)
    assert result is False


def test_run_question_exit_command_ends_session(tutor):
    """Saying 'exit' should return False to end the session."""
    tutor.listen = MagicMock(return_value="exit")
    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)
    assert result is False


def test_run_question_next_command_skips(tutor):
    """Saying 'next' should return True and speak a skip message."""
    tutor.listen = MagicMock(return_value="next")
    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)
    assert result is True
    tutor.speak.assert_any_call("Skipping this question.")


def test_run_question_skip_command_skips(tutor):
    """Saying 'skip' should return True and speak a skip message."""
    tutor.listen = MagicMock(return_value="skip")
    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)
    assert result is True
    tutor.speak.assert_any_call("Skipping this question.")


def test_run_question_normal_answer_evaluates(tutor):
    """A normal answer should be evaluated and feedback given."""
    tutor.listen = MagicMock(return_value="Python is a programming language")
    tutor.evaluator.evaluate.return_value = make_eval_result("correct")
    tutor.feedback.generate.return_value = "Great job!"
    tutor.feedback.generate_intro.return_value = "Question 1 of 3. What is Python?"

    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)

    assert result is True
    tutor.evaluator.evaluate.assert_called_once()
    tutor.session.record_answer.assert_called_once()
    tutor.speak.assert_any_call("Great job!")


def test_run_question_repeat_does_not_consume_retries(tutor):
    """
    Saying 'repeat' multiple times should NOT exhaust the empty-answer retry
    budget; commands are handled independently from empty-listen retries.
    After repeats, a normal answer is still accepted.
    """
    tutor.feedback.generate_intro.return_value = "Question 1 of 3. What is Python?"
    tutor.evaluator.evaluate.return_value = make_eval_result("correct")
    tutor.feedback.generate.return_value = "Correct!"

    # User says repeat 3 times (more than max_retries=2), then gives a real answer
    tutor.listen = MagicMock(side_effect=[
        "repeat", "repeat", "repeat", "Python is a language"
    ])
    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)
    assert result is True
    tutor.evaluator.evaluate.assert_called_once()


def test_run_question_empty_answers_exhaust_retries(tutor):
    """After max empty answers, the question is skipped with a 'moving on' message."""
    tutor.feedback.generate_intro.return_value = "Question 1 of 3. What is Python?"
    # 3 empty answers (1 initial + 2 retries)
    tutor.listen = MagicMock(return_value="")
    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)
    assert result is True
    tutor.speak.assert_any_call("I couldn't hear your answer. Moving on.")


def test_run_question_quit_after_repeats(tutor):
    """Saying 'quit' after some 'repeat' commands should still end the session."""
    tutor.feedback.generate_intro.return_value = "Question 1 of 3. What is Python?"
    tutor.listen = MagicMock(side_effect=["repeat", "repeat", "quit"])
    result = tutor.run_question(SAMPLE_QUESTION, 1, 3)
    assert result is False


# --- Tests for TTS reinitialization ---

def test_tts_speak_reinitializes_engine_each_call():
    """TTSModule.speak() should reinitialize the pyttsx3 engine on each call
    so that repeated calls keep working (avoids pyttsx3 reuse bug)."""
    from local_llm_tutor.tts_module import TTSModule

    mock_engine = MagicMock()
    mock_engine.getProperty.return_value = []  # no voices

    with patch("pyttsx3.init", return_value=mock_engine) as mock_init:
        tts = TTSModule()
        tts.speak("Hello")
        tts.speak("World")

    # pyttsx3.init should be called: once during __init__ + once per speak() call
    expected_init_calls = 1 + 2  # 1 in __init__, 1 per speak call
    assert mock_init.call_count == expected_init_calls
    assert mock_engine.say.call_count == 2
    assert mock_engine.runAndWait.call_count == 2


def test_tts_speak_does_not_stop_engine_after_each_call():
    """TTSModule.speak() must NOT call engine.stop() after a normal utterance.

    pyttsx3 caches the engine in an internal WeakValueDictionary keyed by
    driver name.  Dropping the strong reference (self._engine = None) is
    sufficient to let the WeakValueDictionary entry be garbage-collected so
    the next pyttsx3.init() call creates a truly fresh engine.

    Calling stop() after runAndWait() is harmful: stop() interrupts audio
    that is still buffered in the OS audio layer, causing longer explanations
    to be cut off even though runAndWait() has already returned.
    """
    from local_llm_tutor.tts_module import TTSModule

    mock_engine = MagicMock()
    mock_engine.getProperty.return_value = []  # no voices

    with patch("pyttsx3.init", return_value=mock_engine):
        tts = TTSModule()
        tts.speak("Hello")
        tts.speak("World")

    # stop() must NOT be called during normal speak() completion
    assert mock_engine.stop.call_count == 0


# --- Tests for voice gender selection ---

def _make_voice(name, vid, gender=None):
    """Helper to create a mock pyttsx3 voice object."""
    v = MagicMock()
    v.name = name
    v.id = vid
    v.gender = gender
    return v


def test_tts_selects_female_voice_by_gender_attribute():
    """_select_voice picks a voice whose gender attribute equals 'female'."""
    from local_llm_tutor.tts_module import TTSModule

    male_voice = _make_voice("David", "com.apple.speech.synthesis.voice.david", "male")
    female_voice = _make_voice("Samantha", "com.apple.speech.synthesis.voice.samantha", "female")
    voices = [male_voice, female_voice]

    mock_engine = MagicMock()
    mock_engine.getProperty.return_value = voices

    with patch("pyttsx3.init", return_value=mock_engine):
        tts = TTSModule(voice_gender="female")

    selected_id = mock_engine.setProperty.call_args_list[-1][0][1]
    assert selected_id == female_voice.id


def test_tts_selects_female_voice_by_name_keyword():
    """_select_voice falls back to name-based matching when gender attr is absent."""
    from local_llm_tutor.tts_module import TTSModule

    male_voice = _make_voice("David", "voice.david", None)
    female_voice = _make_voice("Microsoft Zira Desktop", "voice.zira.female", None)
    voices = [male_voice, female_voice]

    mock_engine = MagicMock()
    mock_engine.getProperty.return_value = voices

    with patch("pyttsx3.init", return_value=mock_engine):
        tts = TTSModule(voice_gender="female")

    selected_id = mock_engine.setProperty.call_args_list[-1][0][1]
    assert selected_id == female_voice.id


def test_tts_falls_back_to_voice_index_when_no_gender_match():
    """When no voice matches the requested gender, voice_index is used."""
    from local_llm_tutor.tts_module import TTSModule

    voice_a = _make_voice("Voice A", "voice.a", "male")
    voice_b = _make_voice("Voice B", "voice.b", "male")
    voices = [voice_a, voice_b]

    mock_engine = MagicMock()
    mock_engine.getProperty.return_value = voices

    with patch("pyttsx3.init", return_value=mock_engine):
        tts = TTSModule(voice_index=1, voice_gender="female")

    selected_id = mock_engine.setProperty.call_args_list[-1][0][1]
    assert selected_id == voice_b.id


def test_tts_default_voice_gender_is_none():
    """When voice_gender is not set, voice_index selection is used as before."""
    from local_llm_tutor.tts_module import TTSModule

    voice_a = _make_voice("Voice A", "voice.a", "male")
    voice_b = _make_voice("Voice B", "voice.b", "female")
    voices = [voice_a, voice_b]

    mock_engine = MagicMock()
    mock_engine.getProperty.return_value = voices

    with patch("pyttsx3.init", return_value=mock_engine):
        tts = TTSModule(voice_index=0)  # no voice_gender

    selected_id = mock_engine.setProperty.call_args_list[-1][0][1]
    assert selected_id == voice_a.id
