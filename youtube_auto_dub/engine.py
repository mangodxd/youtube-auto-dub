"""
Legacy AI Engine — consolidated pipeline component.

This module provides the ``Engine`` class which wraps transcription,
translation, and TTS into a single orchestration interface.
It is not used by the current ``cli.py`` / ``pipeline.py`` entry points
but is preserved for backward compatibility.
"""

import random
import time
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Union

import edge_tts

from youtube_auto_dub.config import (
    DEFAULT_VOICE,
    ConfigManager,
)
from youtube_auto_dub.exceptions import (
    ModelLoadError,
    TranslationError,
    TTSError,
    handle_error,
)
from youtube_auto_dub.googlev4 import GoogleTranslator
from youtube_auto_dub.transcriber import (
    ASR_MODEL,
    DeviceManager,
)


class PipelineComponent(ABC):
    """Base class for pipeline components with shared utilities."""

    def __init__(self, device_manager: DeviceManager, config_manager: ConfigManager):
        """Initialize pipeline component.

        Args:
            device_manager: Device management instance.
            config_manager: Configuration management instance.
        """
        self.device_manager = device_manager
        self.config_manager = config_manager
        self.device = device_manager.device

    def _validate_file_exists(self, file_path: Path, description: str = "File") -> None:
        """Validate that a file exists.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"{description} not found: {file_path}")

    def _ensure_directory(self, directory: Path) -> None:
        """Ensure directory exists, create if necessary."""
        directory.mkdir(parents=True, exist_ok=True)


class Engine(PipelineComponent):
    """Central AI/ML engine for the YouTube Auto Dub pipeline."""

    def __init__(self, device: Optional[str] = None):
        device_manager = DeviceManager(device)
        config_manager = ConfigManager()
        super().__init__(device_manager, config_manager)

        self._asr = None
        self.translator = GoogleTranslator()

        print("[+] AI Engine initialized successfully")

    @property
    def asr_model(self):
        """Lazy-load Whisper ASR model."""
        if not self._asr:
            print(f"[*] Loading Whisper model ({ASR_MODEL}) on {self.device}...")
            try:
                from faster_whisper import WhisperModel

                compute_type = "float16" if self.device == "cuda" else "int8"
                self._asr = WhisperModel(
                    ASR_MODEL, device=self.device, compute_type=compute_type
                )
                print("[+] Whisper model loaded successfully")
            except Exception as e:
                raise ModelLoadError(f"Failed to load Whisper model: {e}") from e
        return self._asr

    def _get_lang_config(self, lang: str) -> Dict:
        """Get language configuration."""
        return self.config_manager.get_language_config(lang)

    def _extract_voice_string(self, voice_data: Union[str, List[str], None]) -> str:
        """Extract voice string from data."""
        return self.config_manager.extract_voice(voice_data)

    def release_memory(self, component: Optional[str] = None) -> None:
        """Release VRAM and clean up GPU memory.

        Args:
            component: Specific component ('asr'). If None, releases all.
        """
        if component in (None, "asr") and self._asr:
            del self._asr
            self._asr = None
            print("[*] ASR VRAM cleared")
            self.device_manager.clear_cache()

    def transcribe_safe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio with automatic memory management.

        Returns:
            List of transcription segments with timing.
        """
        try:
            res = self.transcribe(audio_path)
            self.release_memory("asr")
            return res
        except Exception as e:
            handle_error(e, "transcription")
            raise

    def translate_safe(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate texts safely with memory management."""
        self.release_memory()
        return self.translate(texts, target_lang)

    def transcribe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio using Whisper model."""
        segments, _ = self.asr_model.transcribe(
            str(audio_path), word_timestamps=False, language=None
        )
        return [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
        ]

    def translate(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate texts to target language."""
        if not texts:
            return []
        results = []
        print(f"[*] Translating {len(texts)} segments to '{target_lang}'...")

        for i, text in enumerate(texts):
            try:
                if not text.strip():
                    results.append("")
                    continue

                translated = self.translator.translate(text, target=target_lang)
                if translated.startswith(("Error:", "Parse Error:")):
                    results.append(text)
                else:
                    results.append(translated)

                time.sleep(random.uniform(0.1, 0.5))
            except Exception as e:
                handle_error(e, "translation")
                raise TranslationError(f"Translation failed: {e}") from e

        return results

    def calc_rate(self, text: str, target_dur: float, original_text: str = "") -> str:
        """Calculate speech rate adjustment for TTS with dynamic limits.

        Returns:
            Rate adjustment string (e.g., '+10%', '-5%').
        """
        words = len(text.split())
        if words == 0 or target_dur <= 0:
            return "+0%"

        wps = words / target_dur
        estimated_time = words / wps

        if estimated_time <= target_dur:
            return "+0%"

        ratio = estimated_time / target_dur
        speed_percent = int((ratio - 1) * 100)

        if original_text:
            orig_len = len(original_text.split())
            trans_len = words

            if trans_len > orig_len * 1.5:
                speed_percent = max(-25, min(speed_percent, 90))
            elif trans_len < orig_len * 0.7:
                speed_percent = max(-15, min(speed_percent, 50))
            else:
                speed_percent = max(-10, min(speed_percent, 90))
        else:
            speed_percent = max(-10, min(speed_percent, 90))

        return f"{speed_percent:+d}%"

    async def synthesize(
        self,
        text: str,
        target_lang: str,
        out_path: Path,
        gender: str = "female",
        rate: str = "+0%",
    ) -> None:
        """Synthesize text to speech using Edge TTS."""
        if not text.strip():
            raise ValueError("Text empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            voice_pool = self.config_manager.get_voice_pool(target_lang, gender)
            voice = voice_pool[0] if voice_pool else DEFAULT_VOICE

            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            await communicate.save(str(out_path))

            if not out_path.exists() or out_path.stat().st_size < 1024:
                raise RuntimeError("TTS file invalid")

        except Exception as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            handle_error(e, "TTS synthesis")
            raise TTSError(f"TTS failed: {e}") from e
