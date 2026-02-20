"""Text-to-Speech Module: Local TTS using pyttsx3."""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TTSModule:
    """Handles text-to-speech conversion using pyttsx3."""

    def __init__(self, rate: int = 150, volume: float = 1.0, voice_index: int = 0):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self._engine = None
        self._speaking = False
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            voices = self._engine.getProperty("voices")
            if voices and self.voice_index < len(voices):
                self._engine.setProperty("voice", voices[self.voice_index].id)
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
                # Stop and release the engine so that pyttsx3's internal cache
                # drops the stale instance.  The next speak() call then gets a
                # truly fresh engine via pyttsx3.init(), which is required for
                # the event loop to restart correctly on every platform.
                if self._engine is not None:
                    try:
                        self._engine.stop()
                    except Exception as cleanup_err:
                        logger.debug(f"TTS engine cleanup error (non-fatal): {cleanup_err}")
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
