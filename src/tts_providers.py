"""TTS Provider Abstraction for YouTube Auto Dub.

Provides a pluggable TTS backend system with support for:
- Edge TTS (Microsoft Neural Voices) - free, no API key needed
- CAMB AI TTS (MARS models via camb-sdk) - high-quality, voice cloning support
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core_utils import TTSError


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
LANG_MAP_FILE = BASE_DIR / "language_map.json"

try:
    with open(LANG_MAP_FILE, "r", encoding="utf-8") as f:
        _LANG_DATA = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    _LANG_DATA = {}

DEFAULT_VOICE = "en-US-AriaNeural"


# =============================================================================
# TTS PROVIDER INTERFACE
# =============================================================================

class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        target_lang: str,
        out_path: Path,
        rate: str = "+0%",
        gender: str = "female",
    ) -> None:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...


# =============================================================================
# EDGE TTS PROVIDER
# =============================================================================

class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS provider. Free, no API key required."""

    def get_name(self) -> str:
        return "Edge TTS"

    def _get_voice(self, lang_code: str, gender: str) -> str:
        lang_config = _LANG_DATA.get(lang_code, {})
        voices = lang_config.get("voices", {})
        pool = voices.get(gender, [DEFAULT_VOICE])
        if isinstance(pool, str):
            pool = [pool]
        return pool[0] if pool else DEFAULT_VOICE

    async def synthesize(
        self,
        text: str,
        target_lang: str,
        out_path: Path,
        rate: str = "+0%",
        gender: str = "female",
    ) -> None:
        if not text.strip():
            raise ValueError("Text empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import edge_tts
            voice = self._get_voice(target_lang, gender)
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            await communicate.save(str(out_path))

            if not out_path.exists() or out_path.stat().st_size < 1024:
                raise RuntimeError("TTS output file invalid or too small")
        except Exception as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            raise TTSError(f"Edge TTS synthesis failed: {e}") from e


# =============================================================================
# CAMB AI TTS PROVIDER
# =============================================================================

class CambAITTSProvider(TTSProvider):
    """CAMB AI TTS provider using camb-sdk.

    Uses the streaming tts() method which takes string language codes
    (e.g., "es-es") and returns Iterator[bytes].
    """

    def __init__(
        self,
        model: str = "mars-flash",
        voice_id: Optional[int] = None,
        voice_clone: bool = False,
    ):
        self.model = model
        self.voice_id = voice_id
        self.voice_clone = voice_clone
        self._cloned_voice_id: Optional[int] = None
        self._default_voice_id: Optional[int] = None

    def get_name(self) -> str:
        return f"CAMB AI ({self.model})"

    def _get_effective_voice_id(self) -> int:
        if self.voice_id is not None:
            return self.voice_id
        if self._cloned_voice_id is not None:
            return self._cloned_voice_id
        if self._default_voice_id is not None:
            return self._default_voice_id
        return self._fetch_default_voice_id()

    def _fetch_default_voice_id(self) -> int:
        from src.camb_client import get_client
        client = get_client()
        try:
            voices = client.voice_cloning.list_voices()
            if voices:
                v = voices[0]
                self._default_voice_id = v.id if hasattr(v, 'id') else v['id']
                print(f"[*] Using default voice: {getattr(v, 'voice_name', v.get('voice_name', 'unknown'))} (ID: {self._default_voice_id})")
                return self._default_voice_id
        except Exception as e:
            print(f"[!] Failed to list voices: {e}")
        raise TTSError(
            "No voices available. Provide --voice-id or clone a voice with --voice-clone."
        )

    def clone_voice_from_audio(self, audio_path: Path) -> int:
        """Clone a voice from a reference audio sample.

        Extracts a 20-second sample if the file exceeds 20MB (CAMB AI limit).
        """
        import subprocess
        from src.camb_client import get_client
        client = get_client()

        sample_path = audio_path
        # CAMB AI has a 20MB file size limit for voice cloning
        if audio_path.stat().st_size > 18 * 1024 * 1024:
            sample_path = audio_path.parent / f"{audio_path.stem}_voice_sample.wav"
            if not sample_path.exists():
                print(f"[*] Extracting 20s voice sample (original too large)...")
                subprocess.run([
                    'ffmpeg', '-y', '-v', 'error',
                    '-i', str(audio_path),
                    '-ss', '5', '-t', '20',
                    '-ar', '16000', '-ac', '1',
                    str(sample_path),
                ], check=True, timeout=30)

        print(f"[*] Cloning voice from: {sample_path.name}")
        try:
            with open(sample_path, "rb") as f:
                result = client.voice_cloning.create_custom_voice(
                    voice_name=f"cloned_{audio_path.stem}",
                    gender=0,  # NOT_KNOWN
                    file=f,
                )
            self._cloned_voice_id = result.voice_id
            print(f"[+] Voice cloned successfully (ID: {self._cloned_voice_id})")
            return self._cloned_voice_id
        except Exception as e:
            raise TTSError(f"Voice cloning failed: {e}") from e

    async def synthesize(
        self,
        text: str,
        target_lang: str,
        out_path: Path,
        rate: str = "+0%",
        gender: str = "female",
    ) -> None:
        if not text.strip():
            raise ValueError("Text empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from src.camb_client import get_client, resolve_language
            from camb.types.stream_tts_output_configuration import StreamTtsOutputConfiguration

            client = get_client()
            voice_id = self._get_effective_voice_id()
            camb_lang = resolve_language(target_lang)

            # tts() returns Iterator[bytes] - stream to file
            response = client.text_to_speech.tts(
                text=text,
                voice_id=voice_id,
                language=camb_lang,
                speech_model=self.model,
                output_configuration=StreamTtsOutputConfiguration(format="wav"),
            )

            # Write streamed bytes to file
            with open(out_path, "wb") as f:
                for chunk in response:
                    f.write(chunk)

            if not out_path.exists() or out_path.stat().st_size < 1024:
                raise RuntimeError("CAMB AI TTS output file invalid or too small")

        except Exception as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            raise TTSError(f"CAMB AI TTS synthesis failed: {e}") from e
