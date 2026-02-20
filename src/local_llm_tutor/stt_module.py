"""Speech-to-Text Module: Local STT using faster-whisper."""

import logging
import tempfile
import os
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1


class STTResult:
    """Result from speech-to-text transcription."""

    def __init__(self, text: str, confidence: float, language: str = "en"):
        self.text = text
        self.confidence = confidence
        self.language = language

    def __bool__(self):
        return bool(self.text.strip())


class STTModule:
    """Handles speech-to-text using faster-whisper (fully local)."""

    def __init__(self, model_size: str = "base", language: str = "en",
                 energy_threshold: float = 300.0):
        self.model_size = model_size
        self.language = language
        self.energy_threshold = energy_threshold
        self._model = None

    def _get_model(self):
        """Lazy-load Whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                logger.info(f"Loaded faster-whisper model: {self.model_size}")
            except ImportError:
                try:
                    import whisper
                    self._model = whisper.load_model(self.model_size)
                    logger.info(f"Loaded openai-whisper model: {self.model_size}")
                except ImportError:
                    logger.warning("No Whisper library available.")
                    self._model = "unavailable"
        return self._model

    def transcribe_audio(self, audio_data: np.ndarray) -> STTResult:
        """Transcribe audio numpy array to text."""
        model = self._get_model()
        if model == "unavailable":
            return STTResult("", 0.0)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            import scipy.io.wavfile as wav
            wav.write(tmp_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
            return self._transcribe_file(tmp_path)
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return STTResult("", 0.0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _transcribe_file(self, audio_path: str) -> STTResult:
        """Transcribe an audio file."""
        model = self._get_model()
        try:
            from faster_whisper import WhisperModel
            if isinstance(model, WhisperModel):
                segments, info = model.transcribe(audio_path, language=self.language)
                text = " ".join(seg.text for seg in segments).strip()
                # Use avg_logprob as proxy for confidence
                confidence = 0.8  # Default
                return STTResult(text, confidence, info.language)
        except ImportError:
            pass

        try:
            import whisper
            result = model.transcribe(audio_path, language=self.language)
            text = result.get("text", "").strip()
            return STTResult(text, 0.8, self.language)
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return STTResult("", 0.0)

    def record_audio(self, duration: float = 5.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Record audio from microphone."""
        try:
            import sounddevice as sd
            logger.info(f"Recording audio for {duration} seconds...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                          channels=CHANNELS, dtype="float32")
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return np.zeros(int(duration * sample_rate))

    def listen_and_transcribe(self, duration: float = 5.0) -> STTResult:
        """Record audio and transcribe it."""
        audio = self.record_audio(duration)
        return self.transcribe_audio(audio)
