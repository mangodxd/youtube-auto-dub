"""Audio Separation Module using CAMB AI.

Separates vocal tracks from background music/SFX using the
CAMB AI audio separation API (camb-sdk).
"""

import time
from pathlib import Path
from typing import Tuple

import httpx

from src.core_utils import AudioProcessingError


def separate_audio(audio_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Separate audio into vocals and background tracks using CAMB AI.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save separated tracks.

    Returns:
        Tuple of (vocals_path, background_path).
    """
    from src.camb_client import get_client

    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / "vocals.wav"
    background_path = output_dir / "background.wav"

    if vocals_path.exists() and background_path.exists():
        print("[*] Using cached audio separation results")
        return vocals_path, background_path

    client = get_client()

    print("[*] Separating vocals from background audio via CAMB AI...")
    try:
        # Compress audio if over 18MB (CAMB AI limit is 20MB)
        upload_path = audio_path
        if audio_path.stat().st_size > 18 * 1024 * 1024:
            upload_path = output_dir / "audio_compressed.mp3"
            if not upload_path.exists():
                print(f"[*] Compressing audio for upload (original: {audio_path.stat().st_size / (1024*1024):.0f}MB)...")
                import subprocess as _sp
                _sp.run([
                    'ffmpeg', '-y', '-v', 'error',
                    '-i', str(audio_path),
                    '-b:a', '64k', '-ac', '1',
                    str(upload_path),
                ], check=True, timeout=60)
                print(f"[*] Compressed to {upload_path.stat().st_size / (1024*1024):.1f}MB")

        # Submit separation task
        with open(upload_path, "rb") as f:
            task = client.audio_separation.create_audio_separation(media_file=f)

        task_id = task.task_id
        print(f"[*] Audio separation task submitted (ID: {task_id})")

        # Poll for completion
        run_id = None
        for _ in range(60):  # Max ~10 minutes
            status = client.audio_separation.get_audio_separation_status(task_id)
            if status.status == "SUCCESS":
                run_id = status.run_id
                break
            elif status.status in ("FAILED", "ERROR"):
                raise RuntimeError(f"Audio separation failed: {status.message}")
            time.sleep(10)
        else:
            raise RuntimeError("Audio separation timed out after 10 minutes")

        # Get results
        result = client.audio_separation.get_audio_separation_run_info(run_id)

        # Download separated tracks
        _download_url(result.foreground_audio_url, vocals_path)
        _download_url(result.background_audio_url, background_path)

        print(f"[+] Audio separation complete")
        print(f"    Vocals: {vocals_path}")
        print(f"    Background: {background_path}")

        return vocals_path, background_path

    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(f"Audio separation failed: {e}") from e


def _download_url(url: str, dest: Path) -> None:
    """Download a URL to a local file."""
    with httpx.stream("GET", url) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
