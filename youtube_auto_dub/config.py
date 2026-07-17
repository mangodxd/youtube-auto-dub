"""Configuration management and language data loading."""

import json
from typing import Any, Dict

from youtube_auto_dub.models import LANG_MAP_PATH

# Default voice fallback
DEFAULT_VOICE = "en-US-AriaNeural"

# Load language configuration
try:
    with open(LANG_MAP_PATH, "r", encoding="utf-8") as f:
        LANG_DATA = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    LANG_DATA = {}


class ConfigManager:
    """Centralized configuration access with validation."""

    def get_language_config(self, lang_code: str) -> Dict[str, Any]:
        """Get language configuration by ISO language code.

        Args:
            lang_code: ISO language code.

        Returns:
            Language configuration dictionary.
        """
        return LANG_DATA.get(lang_code, {})

    def extract_voice(self, voice_data, fallback_gender: str = "female") -> str:
        """Extract voice string from various data formats.

        Args:
            voice_data: Voice data in list, string, or other format.
            fallback_gender: Default gender if extraction fails.

        Returns:
            Voice string for TTS.
        """
        if isinstance(voice_data, list):
            return voice_data[0] if voice_data else DEFAULT_VOICE
        if isinstance(voice_data, str):
            return voice_data
        return DEFAULT_VOICE

    def get_voice_pool(self, lang_code: str, gender: str) -> list:
        """Get pool of available voices for a language and gender.

        Args:
            lang_code: ISO language code.
            gender: Voice gender ('male' or 'female').

        Returns:
            List of available voice name strings.
        """
        lang_config = self.get_language_config(lang_code)
        voices = lang_config.get("voices", {})
        pool = voices.get(gender, [DEFAULT_VOICE])

        if isinstance(pool, str):
            pool = [pool]

        return pool
