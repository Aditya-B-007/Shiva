import io
import logging
from typing import Any, Dict
from .base import PerceptionDevice
from .permissions import PermissionManager
from .detector import get_current_os

logger = logging.getLogger("shiva.perception.microphone")

try:
    import sounddevice as sd
    import numpy as np
    from scipy.io import wavfile
    audio_libraries_available = True
except ImportError:
    audio_libraries_available = False

class MicrophoneDevice(PerceptionDevice):

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "Microphone"

    @property
    def description(self) -> str:
        return "Captures raw audio inputs from the microphone. No transcribing."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {
            "duration_seconds": {
                "type": "integer",
                "description": "Duration of audio to capture in seconds.",
                "required": True
            }
        }

    def capture(self, **kwargs: Any) -> bytes:
        if not self.permission_manager.has_permission("microphone"):
            success = self.permission_manager.request_permission("microphone")
            if not success:
                raise PermissionError("Microphone permission denied.")

        duration = kwargs.get("duration_seconds", 3)
        current_os = get_current_os()
        logger.info(f"Initiating microphone recording for {duration}s on: {current_os}")

        if current_os in ["windows", "macos", "linux"] and audio_libraries_available:
            try:
                sample_rate = 16000  # 16kHz
                # Record mono audio
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
                sd.wait()  # Wait until the recording is finished
                
                # Write to WAV in-memory bytes
                wav_io = io.BytesIO()
                wavfile.write(wav_io, sample_rate, recording)
                return wav_io.getvalue()
            except Exception as e:
                logger.warning(f"Failed to record audio via sounddevice: {e}")

        # Fallback to mock silent audio bytes (16kHz 16-bit mono PCM equivalent size)
        return b'\x00' * (duration * 16000 * 2)
