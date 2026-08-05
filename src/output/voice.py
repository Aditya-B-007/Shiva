import os
from typing import Optional

try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    KPipeline = None
    sf = None
    np = None

class VoiceGenerator:
    def __init__(
        self,
        voice: str = "af_heart",
        lang: str = "a",
        speed: float = 1.0,
        sample_rate: int = 24000
    ) -> None:
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.sample_rate = sample_rate
        self._pipeline: Optional[KPipeline] = None

    def synthesize(self, text: str, output_path: str) -> bool:
        if not KOKORO_AVAILABLE:
            print("[Voice Output] Warning: kokoro, soundfile, or numpy is not installed. Text-to-voice skipped.")
            return False

        try:
            if self._pipeline is None:
                self._pipeline = KPipeline(lang_code=self.lang)

            generator = self._pipeline(text, voice=self.voice, speed=self.speed)
            audio_segments = []

            for _, _, audio in generator:
                if audio is not None:
                    audio_segments.append(audio)

            if not audio_segments:
                print("[Voice Output] Kokoro generated empty audio.")
                return False
            combined_audio = np.concatenate(audio_segments)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sf.write(output_path, combined_audio, self.sample_rate)
            return True

        except Exception as e:
            print(f"[Voice Output] Speech synthesis failed: {e}")
            return False
