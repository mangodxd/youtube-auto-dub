import json
import asyncio
import edge_tts
from pathlib import Path
from .models import LANG_MAP_PATH
from .ui import console


def get_voice(lang_code: str, gender: str = "male") -> str:
    if not LANG_MAP_PATH.exists():
        raise FileNotFoundError(f"Language map file not found: {LANG_MAP_PATH}")

    with open(LANG_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if lang_code not in data:
        raise ValueError(f"Language {lang_code} not found in lang_map.json")

    voices = data[lang_code]["voices"].get(gender, [])

    if not voices:
        # fallback: choose the other gender if available
        other_gender = "female" if gender == "male" else "male"
        voices = data[lang_code]["voices"].get(other_gender, [])
        if not voices:
            raise ValueError(f"Cannot find any voice for language '{lang_code}'.")
        console.warning(f"No {gender} voice found, switching to {other_gender}.")

    return voices[0]


async def tts(
    text: str,
    voice: str,
    output: Path,
    max_retries: int = 2,
    timeout: int = 60,
) -> None:
    """Generate TTS audio with retry logic and timeout protection.

    Args:
        text: Text to synthesize.
        voice: Voice name (e.g. 'en-US-AriaNeural').
        output: Output audio file path.
        max_retries: Number of retries on failure.
        timeout: Timeout in seconds for each attempt.

    Raises:
        TimeoutError: If all retries time out.
        RuntimeError: If TTS fails for other reasons.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            # The save() call is async but doesn't natively support timeout.
            # asyncio.wait_for wraps it with a deadline.
            await asyncio.wait_for(
                communicate.save(str(output)),
                timeout=timeout,
            )
            if not output.exists() or output.stat().st_size < 256:
                raise RuntimeError("TTS output file is empty or too small")
            return
        except asyncio.TimeoutError:
            last_error = TimeoutError(
                f"TTS timed out after {timeout}s (attempt {attempt + 1}/{max_retries + 1})"
            )
            console.warning(
                f"TTS timeout for '{voice}', attempt {attempt + 1}/{max_retries + 1}"
            )
        except Exception as e:
            last_error = e
            console.warning(
                f"TTS failed for '{voice}', attempt {attempt + 1}/{max_retries + 1}: {e}"
            )

        # Clean up partial output before retry
        if output.exists():
            output.unlink(missing_ok=True)

        if attempt < max_retries:
            await asyncio.sleep(2)

    raise last_error or RuntimeError("TTS failed after all retries")
