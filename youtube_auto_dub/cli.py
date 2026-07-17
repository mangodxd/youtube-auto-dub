#!/usr/bin/env python3
"""Thin CLI adapter — parse arguments and delegate to the pipeline."""

import argparse
import asyncio
import shutil

from youtube_auto_dub.models import TEMP_DIR
from youtube_auto_dub.pipeline import run_pipeline
from youtube_auto_dub.ui import console


def main() -> None:
    """Entry point: parse CLI args, clean temp dir, and run the pipeline."""
    parser = argparse.ArgumentParser(description="YouTube Auto Sub/Dub Studio")
    parser.add_argument("url", help="YouTube video URL")

    parser.add_argument("--lang", "-l", help="General target language (Default: vi)")
    parser.add_argument("--lang_sub", "-ls", help="Subtitle language (Overrides --lang)")
    parser.add_argument("--lang_dub", "-ld", help="Dubbing language (Overrides --lang)")

    parser.add_argument("--mode", "-m", choices=["sub", "dub", "both"], default="both", help="Processing mode")
    parser.add_argument("--gender", "-g", choices=["male", "female"], default="female", help="Voice gender")

    parser.add_argument("--browser", "-b", help="Browser to extract cookies from (chrome, edge, firefox)")
    parser.add_argument("--whisper_model", "-wm", help="Whisper model (tiny, base, small, medium)")

    args = parser.parse_args()

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"\n[red]System Error: {e}[/red]")
