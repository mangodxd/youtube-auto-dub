"""Tests for the GoogleTranslator class (no network calls)."""

import pytest

from youtube_auto_dub.googlev4 import GoogleTranslator


@pytest.mark.asyncio
async def test_source_equals_target_skips_translation():
    """When source == target, translate() should return the text unchanged."""
    translator = GoogleTranslator()
    text = "This is a test sentence."
    result = await translator.translate(text, source="en", target="en")
    assert result == text
    await translator.close()


@pytest.mark.asyncio
async def test_empty_text_returns_empty():
    translator = GoogleTranslator()
    assert await translator.translate("", source="en", target="vi") == ""
    await translator.close()


@pytest.mark.asyncio
async def test_batch_source_equals_target():
    translator = GoogleTranslator()
    texts = ["Hello", "World", "Test"]
    result = await translator.translate_batch(texts, source="en", target="en")
    assert result == texts
    await translator.close()
