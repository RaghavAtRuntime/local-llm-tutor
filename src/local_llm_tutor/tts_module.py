"""Text-to-Speech Module: Local TTS using pyttsx3."""

import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TTSModule:
    """Handles text-to-speech conversion using pyttsx3."""

    def __init__(self, rate: int = 150, volume: float = 1.0, voice_index: int = 0,
                 voice_gender: Optional[str] = None):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.voice_gender = voice_gender
        self._engine = None
        self._speaking = False
        self._lock = threading.Lock()
        self._init_engine()

    def _select_voice(self, voices):
        """Return the best matching voice object.

        If *voice_gender* is set, the first voice whose ``gender`` attribute or
        whose name/ID contains the requested gender string is returned.  Falls
        back to the index-based selection when no gender match is found.
        """
        if self.voice_gender:
            gender = self.voice_gender.lower()
            for v in voices:
                v_gender = getattr(v, "gender", None)
                if v_gender and v_gender.lower() == gender:
                    return v
                name = (v.name or "").lower()
                vid = (v.id or "").lower()
                # Use word-boundary matching to avoid false positives
                # (e.g., "female" must not match "shemale").
                if re.search(r'\b' + re.escape(gender) + r'\b', name) or \
                        re.search(r'\b' + re.escape(gender) + r'\b', vid):
                    return v
        if self.voice_index < len(voices):
            return voices[self.voice_index]
        logger.warning(
            "voice_index %d is out of range (%d voices available); using voices[0].",
            self.voice_index, len(voices),
        )
        return voices[0]

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            voices = self._engine.getProperty("voices")
            if voices:
                self._engine.setProperty("voice", self._select_voice(voices).id)
            logger.info("TTS engine initialized.")
        except Exception as e:
            logger.error(f"TTS init error: {e}")
            self._engine = None

    def speak(self, text: str, blocking: bool = True):
        """Convert text to speech and play it."""
        if not text:
            return
        logger.info(f"TTS speaking: {text[:80]}...")
        with self._lock:
            self._speaking = True
            try:
                # Reinitialize engine on each call to avoid pyttsx3 reuse issues
                # where the event loop fails to restart after the first utterance.
                self._init_engine()
                if self._engine is None:
                    print(f"[TTS] {text}")
                    return
                self._engine.say(text)
                if blocking:
                    self._engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS speak error: {e}")
                print(f"[TTS] {text}")
            finally:
                # Release the engine reference so that pyttsx3's internal
                # WeakValueDictionary drops the stale instance.  The next
                # speak() call then gets a truly fresh engine via pyttsx3.init().
                # We intentionally do NOT call stop() here: stop() interrupts
                # any audio still buffered in the OS layer, which causes longer
                # explanations to be cut off even after runAndWait() returns.
                self._engine = None
                self._speaking = False

    def stop(self):
        """Stop current speech."""
        if self._engine and self._speaking:
            try:
                self._engine.stop()
            except Exception as e:
                logger.error(f"TTS stop error: {e}")
        self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking

    def set_rate(self, rate: int):
        self.rate = rate
        if self._engine:
            self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        self.volume = volume
        if self._engine:
            self._engine.setProperty("volume", volume)
