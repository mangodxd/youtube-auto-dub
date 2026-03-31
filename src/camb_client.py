"""CAMB AI Client Module.

Provides lazy initialization of the CAMB AI SDK client (camb-sdk)
and language code resolution (ISO 639-1 -> CAMB AI language codes).

Requires CAMB_API_KEY environment variable to be set.
"""

import os
from typing import Optional

_client = None

# Map ISO 639-1 codes to CAMB AI language codes used by tts()
CAMB_LANG_MAP = {
    "en": "en-us", "es": "es-es", "fr": "fr-fr", "de": "de-de",
    "it": "it-it", "pt": "pt-pt", "ja": "ja-jp", "ko": "ko-kr",
    "zh": "zh-cn", "hi": "hi-in", "ar": "ar-sa", "ru": "ru-ru",
    "nl": "nl-nl", "pl": "pl-pl", "fi": "fi-fi", "tr": "tr-tr",
    "vi": "vi-vn", "th": "th-th", "id": "id-id", "uk": "uk-ua",
    "cs": "cs-cz", "ro": "ro-ro", "el": "el-gr", "bn": "bn-in",
    "ta": "ta-in", "te": "te-in", "mr": "mr-in", "kn": "kn-in",
    "ml": "ml-in", "pa": "pa-in",
}


def get_client():
    """Get or create the CAMB AI client singleton.

    Returns:
        CambAI client instance.

    Raises:
        RuntimeError: If CAMB_API_KEY is not set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("CAMB_API_KEY")
        if not api_key:
            raise RuntimeError(
                "CAMB_API_KEY environment variable not set. "
                "Set it with: export CAMB_API_KEY='your-key' "
                "or use --tts-engine edge to use Edge TTS instead."
            )
        from camb.client import CambAI
        _client = CambAI(api_key=api_key)
    return _client


def resolve_language(iso_code: str) -> str:
    """Convert an ISO 639-1 language code to a CAMB AI language string.

    Args:
        iso_code: ISO 639-1 language code (e.g., "es", "fr").

    Returns:
        CAMB AI language string (e.g., "es-es").
    """
    lang = CAMB_LANG_MAP.get(iso_code.lower())
    if lang:
        return lang
    # Try constructing it (e.g., "af" -> "af-za" won't work, but "es" -> "es-es" does)
    # Fall back to the code repeated
    return f"{iso_code.lower()}-{iso_code.lower()}"
