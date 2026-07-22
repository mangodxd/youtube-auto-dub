"""Tests for voice lookup functions (no network calls)."""

import pytest

from youtube_auto_dub.voice import pick_voice


class TestPickVoice:
    def test_returns_string(self):
        voice = pick_voice("en", gender="female")
        assert isinstance(voice, str)
        assert len(voice) > 0

    def test_male(self):
        voice = pick_voice("en", gender="male")
        assert isinstance(voice, str)

    def test_fallback_gender(self):
        voice = pick_voice("am", gender="male")
        assert isinstance(voice, str)

    def test_unknown_language(self):
        with pytest.raises(ValueError, match="not found|not in map"):
            pick_voice("zz", gender="female")
