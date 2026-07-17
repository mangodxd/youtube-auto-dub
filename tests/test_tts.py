"""Tests for TTS helper functions (no network calls)."""


import pytest

from youtube_auto_dub.tts import get_voice


class TestGetVoice:
    def test_get_voice_returns_string(self):
        """get_voice should return a voice name for a known language."""
        voice = get_voice("en", gender="female")
        assert isinstance(voice, str)
        assert len(voice) > 0

    def test_get_voice_male(self):
        voice = get_voice("en", gender="male")
        assert isinstance(voice, str)
        assert "Neural" in voice

    def test_get_voice_fallback_gender(self):
        """If the requested gender has no voices, fall back to the other."""
        # "am" has only female voices in the map
        voice = get_voice("am", gender="male")
        assert isinstance(voice, str)

    def test_get_voice_unknown_language(self):
        with pytest.raises(ValueError, match="not found"):
            get_voice("zz", gender="female")
