"""Audio I/O Controller: Manages audio input/output with interrupt detection."""

import logging
import threading
import time
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1


class AudioController:
    """
    Manages audio I/O with full-duplex support and interrupt detection.
    Supports:
    - Continuous listening
    - Voice Activity Detection (VAD) for auto-stop
    - Interrupt callbacks when speech is detected during TTS
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, energy_threshold: float = 300.0,
                 silence_duration: float = 1.5):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self._listening = False
        self._interrupt_callback: Optional[Callable] = None
        self._stream = None

    def set_interrupt_callback(self, callback: Callable):
        """Set callback to call when speech interrupts TTS."""
        self._interrupt_callback = callback

    def _compute_rms(self, data: np.ndarray) -> float:
        """Compute RMS energy of audio chunk."""
        return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

    def record_with_vad(self, max_duration: float = 30.0) -> np.ndarray:
        """
        Record audio until silence is detected or max_duration reached.
        Uses voice activity detection to determine when user stops speaking.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("sounddevice not available, using fixed-duration recording.")
            return self._fixed_duration_record(5.0)

        logger.info("Listening... (speak now)")
        frames = []
        speech_started = False
        silence_start = None
        start_time = time.time()

        with sd.InputStream(samplerate=self.sample_rate, channels=CHANNELS,
                           dtype="int16", blocksize=CHUNK_SIZE) as stream:
            while (time.time() - start_time) < max_duration:
                data, overflowed = stream.read(CHUNK_SIZE)
                rms = self._compute_rms(data.flatten())

                if rms > self.energy_threshold:
                    speech_started = True
                    silence_start = None
                    frames.append(data.flatten())
                elif speech_started:
                    frames.append(data.flatten())
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) > self.silence_duration:
                        logger.info("Silence detected, stopping recording.")
                        break

        if not frames:
            return np.zeros(1)

        audio = np.concatenate(frames).astype(np.float32) / 32767.0
        return audio

    def _fixed_duration_record(self, duration: float = 5.0) -> np.ndarray:
        """Fallback fixed-duration recording without VAD."""
        try:
            import sounddevice as sd
            audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate,
                          channels=CHANNELS, dtype="float32")
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return np.zeros(int(duration * self.sample_rate))

    def monitor_for_interrupt(self, tts_module, stop_event: threading.Event):
        """
        Run in background thread - monitors mic for speech while TTS is playing.
        If speech detected, signals TTS to stop.
        """
        try:
            import sounddevice as sd
            with sd.InputStream(samplerate=self.sample_rate, channels=CHANNELS,
                               dtype="int16", blocksize=CHUNK_SIZE) as stream:
                while not stop_event.is_set():
                    data, _ = stream.read(CHUNK_SIZE)
                    rms = self._compute_rms(data.flatten())
                    if rms > self.energy_threshold * 2:
                        logger.info("Interrupt detected during TTS.")
                        tts_module.stop()
                        if self._interrupt_callback:
                            self._interrupt_callback()
                        break
                    time.sleep(0.01)
        except Exception as e:
            logger.debug(f"Interrupt monitor stopped: {e}")
