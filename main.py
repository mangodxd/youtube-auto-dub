#!/usr/bin/env python3
"""YouTube Auto Dub - Automated Video Dubbing & Subtitling Pipeline.

This module provides a command-line interface for automatically dubbing and
subtitling YouTube videos using AI/ML technologies.

Supports two modes:
- Subtitle-only (default): generates subtitles on the video
- Dubbing (--tts-engine): generates dubbed audio with TTS

Example:
    python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es
    python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es --tts-engine camb --voice-clone

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 2.0.0
License: MIT
"""

import argparse
import shutil
import subprocess
import time
import random
from pathlib import Path
from typing import Optional
import asyncio
import torch

# Load .env file if present (for CAMB_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import src.engines
import src.youtube
import src.media


def _checkDeps() -> None:
    """Verifies critical dependencies are installed and accessible.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        SystemExit: If any critical dependency is missing.
        
    Note:
        Checks for FFmpeg, FFprobe binaries and PyTorch installation.
    """
    from shutil import which
    
    missing = []
    if not which("ffmpeg"):
        missing.append("ffmpeg")
    if not which("ffprobe"):
        missing.append("ffprobe")
    
    if missing:
        print(f"[!] CRITICAL: Missing dependencies: {', '.join(missing)}")
        print("    Please install FFmpeg and add it to your System PATH.")
        print("    Download: https://ffmpeg.org/download.html")
        exit(1)

    try:
        import torch
        print(f"[*] PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("[!] CRITICAL: PyTorch not installed.")
        print("    Install with: pip install torch")
        exit(1)


def _cleanup() -> None:
    """Clean up temporary directory with retry mechanism for Windows file locks.
    
    Args:
        None
        
    Returns:
        None
        
    Note:
        Windows can lock files temporarily, especially after FFmpeg operations.
        Implements exponential backoff retry strategy.
        If cleanup fails after max retries, pipeline will continue.
    """
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            if src.engines.TEMP_DIR.exists():
                shutil.rmtree(src.engines.TEMP_DIR)
            src.engines.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            return
        except PermissionError:
            wait_time = 0.5 * (2 ** attempt)
            print(f"[-] File locked (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    print(f"[!] WARNING: Could not fully clean temp directory after {max_retries} attempts.")
    print(f"    Files may persist in: {src.engines.TEMP_DIR}")


def main() -> None:
    """Main entry point for the YouTube Auto Sub pipeline.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        SystemExit: On critical errors or user interruption.
        
    Note:
        Orchestrates the complete subtitling process:
        1. Dependency validation and environment setup
        2. Video/audio download from YouTube
        3. Speech transcription using Whisper
        4. Smart audio chunking for optimal processing
        5. Translation to target language
        6. Subtitle file generation
        7. Final video rendering with subtitles
    """
    parser = argparse.ArgumentParser(
        description="YouTube Auto Dub - Automated Video Dubbing & Subtitling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Subtitle only (default)
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es

  # Dub with Edge TTS
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es --tts-engine edge

  # Dub with CAMB AI (requires CAMB_API_KEY)
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es --tts-engine camb

  # Dub with CAMB AI voice cloning (clones original speaker)
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es --tts-engine camb --voice-clone

  # Dub with CAMB AI + background music preservation
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es --tts-engine camb --separate-audio
        """
    )

    parser.add_argument("url", help="YouTube video URL to dub/subtitle")
    parser.add_argument(
        "--lang", "-l",
        default="es",
        help="Target language ISO code (e.g., es, fr, ja, vi)."
    )
    parser.add_argument(
        "--browser", "-b",
        help="Browser to extract cookies from (chrome, edge, firefox). Close browser first!"
    )
    parser.add_argument(
        "--cookies", "-c",
        help="Path to cookies.txt file (Netscape format) for YouTube authentication"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for Whisper (requires CUDA)"
    )
    parser.add_argument(
        "--whisper_model", "-wm",
        help="Whisper model to use (tiny, base, small, medium, large-v3). Default: auto-select based on VRAM"
    )
    # TTS / Dubbing options
    parser.add_argument(
        "--tts-engine",
        choices=["edge", "camb"],
        default=None,
        help="TTS engine for dubbing. If omitted, subtitle-only mode is used."
    )
    parser.add_argument(
        "--camb-model",
        choices=["mars-flash", "mars-pro", "mars-instruct"],
        default="mars-flash",
        help="CAMB AI model to use (default: mars-flash)"
    )
    parser.add_argument(
        "--voice-clone",
        action="store_true",
        help="Clone the original speaker's voice for dubbing (requires --tts-engine camb)"
    )
    parser.add_argument(
        "--voice-id",
        type=int,
        default=None,
        help="Use a specific CAMB AI voice ID for dubbing"
    )
    parser.add_argument(
        "--separate-audio",
        action="store_true",
        help="Separate background music from vocals for cleaner dubbing (requires CAMB_API_KEY)"
    )
    
    args = parser.parse_args()

    # Validate flag combinations
    if args.voice_clone and args.tts_engine != "camb":
        parser.error("--voice-clone requires --tts-engine camb")
    if args.voice_id and args.tts_engine != "camb":
        parser.error("--voice-id requires --tts-engine camb")

    dubbing_mode = args.tts_engine is not None
    mode_label = f"DUBBING ({args.tts_engine.upper()})" if dubbing_mode else "SUBTITLE-ONLY"

    print("\n" + "="*60)
    print(f"YOUTUBE AUTO DUB - INITIALIZING [{mode_label}]")
    print("="*60)

    _checkDeps()
    _cleanup()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[*] Using device: {device.upper()}")

    # Set Whisper model based on user input or auto-selection
    if args.whisper_model:
        src.engines.ASR_MODEL = args.whisper_model
        print(f"[*] Using specified Whisper model: {args.whisper_model}")
    else:
        print(f"[*] Auto-selected Whisper model: {src.engines.ASR_MODEL} (based on VRAM)")

    engine = src.engines.Engine(device)

    # Initialize TTS provider if dubbing
    tts_provider = None
    if dubbing_mode:
        if args.tts_engine == "camb":
            from src.tts_providers import CambAITTSProvider
            tts_provider = CambAITTSProvider(
                model=args.camb_model,
                voice_id=args.voice_id,
                voice_clone=args.voice_clone,
            )
        else:
            from src.tts_providers import EdgeTTSProvider
            tts_provider = EdgeTTSProvider()
        print(f"[*] TTS Engine: {tts_provider.get_name()}")
    
    print(f"\n{'='*60}")
    print(f"STEP 1: DOWNLOADING CONTENT")
    print(f"{'='*60}")
    print(f"[*] Target URL: {args.url}")
    print(f"[*] Target Language: {args.lang.upper()}")

    try:
        videoPath = src.youtube.downloadVideo(
            args.url,
            browser=args.browser,
            cookies_file=args.cookies
        )
        audioPath = src.youtube.downloadAudio(
            args.url,
            browser=args.browser,
            cookies_file=args.cookies
        )
        print(f"[+] Video downloaded: {videoPath}")
        print(f"[+] Audio extracted: {audioPath}")
    except Exception as e:
        print(f"\n[!] DOWNLOAD FAILED: {e}")
        print("\n[-] TROUBLESHOOTING TIPS:")
        print("    1. Close all browser windows if using --browser")
        print("    2. Export fresh cookies.txt and use --cookies")
        print("    3. Check if video is private/region-restricted")
        print("    4. Verify YouTube URL is correct")
        return

    # Audio separation (optional, for background music preservation)
    background_audio = None
    transcribe_audio = audioPath
    if args.separate_audio:
        print(f"\n{'='*60}")
        print(f"STEP 1.5: AUDIO SEPARATION")
        print(f"{'='*60}")
        try:
            from src.audio_separation import separate_audio
            vocals_path, background_audio = separate_audio(audioPath, src.engines.TEMP_DIR)
            transcribe_audio = vocals_path
            print(f"[+] Using clean vocals for transcription")
        except Exception as e:
            print(f"[!] Audio separation failed: {e}")
            print(f"[-] Continuing with original audio...")
            transcribe_audio = audioPath

    # Voice cloning (extract sample from original audio before transcription)
    if dubbing_mode and args.voice_clone and args.tts_engine == "camb":
        print(f"\n[*] Extracting voice sample for cloning...")
        try:
            tts_provider.clone_voice_from_audio(audioPath)
        except Exception as e:
            print(f"[!] Voice cloning failed: {e}")
            print(f"[-] Falling back to default CAMB AI voice")

    print(f"\n{'='*60}")
    print(f"STEP 2: SPEECH TRANSCRIPTION")
    print(f"{'='*60}")
    print(f"[*] Transcribing audio with Whisper ({src.engines.ASR_MODEL})...")

    raw_segments = engine.transcribeSafe(transcribe_audio)
    print(f"[+] Transcription complete: {len(raw_segments)} segments")
    
    if len(raw_segments) > 0:
        print(f"[*] Sample segment: '{raw_segments[0]['text'][:50]}...'")
    
    print(f"\n{'='*60}")
    print(f"STEP 3: INTELLIGENT CHUNKING")
    print(f"{'='*60}")
    
    chunks = src.engines.smartChunk(raw_segments)
    print(f"[+] Optimized {len(raw_segments)} raw segments into {len(chunks)} chunks")
    print(f"[*] Average chunk duration: {sum(c['end']-c['start'] for c in chunks)/len(chunks):.2f}s")

    print(f"\n{'='*60}")
    print(f"STEP 4: TRANSLATION ({args.lang.upper()})")
    print(f"{'='*60}")
    
    texts = [c['text'] for c in chunks]
    print(f"[*] Translating {len(texts)} text segments...")
    
    translated_texts = engine.translateSafe(texts, args.lang)
    
    for i, chunk in enumerate(chunks):
        chunk['trans_text'] = translated_texts[i]
    
    print(f"[+] Translation complete")
    
    if len(chunks) > 0:
        original = chunks[0]['text'][:50]
        translated = chunks[0]['trans_text'][:50]
        print(f"[*] Sample: '{original}' -> '{translated}'")

    # TTS Synthesis step (only in dubbing mode)
    concat_file = None
    if dubbing_mode and tts_provider:
        print(f"\n{'='*60}")
        print(f"STEP 5: TTS SYNTHESIS ({tts_provider.get_name()})")
        print(f"{'='*60}")

        tts_dir = src.engines.TEMP_DIR / "tts_chunks"
        tts_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Synthesizing {len(chunks)} audio chunks...")
        for i, chunk in enumerate(chunks):
            text = chunk.get('trans_text', chunk['text'])
            if not text.strip():
                continue

            target_dur = chunk['end'] - chunk['start']
            rate = engine.calcRate(text, target_dur, original_text=chunk.get('text', ''))

            chunk_audio = tts_dir / f"chunk_{i:04d}.wav"
            try:
                asyncio.run(tts_provider.synthesize(
                    text=text,
                    target_lang=args.lang,
                    out_path=chunk_audio,
                    rate=rate,
                ))

                # Fit audio to target duration
                fitted = src.media.fit_audio(chunk_audio, target_dur)
                chunk['processed_audio'] = fitted
                print(f"    [{i+1}/{len(chunks)}] {text[:40]}...")
            except Exception as e:
                print(f"    [!] Chunk {i+1} synthesis failed: {e}")
                chunk['processed_audio'] = None

        # Create concat manifest
        silence_ref = tts_dir / "silence_ref.wav"
        concat_file = src.engines.TEMP_DIR / "concat.txt"
        src.media.create_concat_file(chunks, silence_ref, concat_file)
        print(f"[+] TTS synthesis complete. Concat manifest: {concat_file}")

    print(f"\n{'='*60}")
    print(f"STEP {'6' if dubbing_mode else '5'}: SUBTITLE GENERATION")
    print(f"{'='*60}")

    subtitle_path = src.engines.TEMP_DIR / "subtitles.srt"
    src.media.generate_srt(chunks, subtitle_path)
    print(f"[+] Subtitles generated: {subtitle_path}")

    print(f"\n{'='*60}")
    print(f"STEP {'7' if dubbing_mode else '6'}: FINAL VIDEO RENDERING")
    print(f"{'='*60}")

    try:
        video_name = videoPath.stem
        prefix = "dubbed" if dubbing_mode else "subtitled"
        out_name = f"{prefix}_{args.lang}_{video_name}.mp4"
        final_output = src.engines.OUTPUT_DIR / out_name

        if dubbing_mode:
            print(f"[*] Rendering dubbed video...")
        else:
            print(f"[*] Rendering video with subtitles...")
        print(f"    Source: {videoPath}")
        print(f"    Output: {final_output}")

        src.media.render_video(
            videoPath,
            concat_file,
            final_output,
            subtitle_path=subtitle_path,
            background_audio=background_audio,
        )

        if final_output.exists():
            file_size = final_output.stat().st_size / (1024 * 1024)
            print(f"\n[+] SUCCESS! Video rendered successfully.")
            print(f"    Output: {final_output}")
            print(f"    Size: {file_size:.1f} MB")

            # Copy SRT alongside the video for external subtitle players
            if subtitle_path.exists():
                import shutil as _shutil
                srt_output = final_output.with_suffix('.srt')
                _shutil.copy2(subtitle_path, srt_output)
                print(f"    Subtitles: {srt_output}")
        else:
            print(f"\n[!] ERROR: Output file not created at {final_output}")

    except Exception as e:
        print(f"\n[!] RENDERING FAILED: {e}")
        print("[-] This may be due to:")
        print("    1. Corrupted audio chunks")
        print("    2. FFmpeg compatibility issues")
        print("    3. Insufficient disk space")
        return

    finally:
        print(f"\n{'='*60}")
        print("YOUTUBE AUTO DUB - PIPELINE COMPLETE")
        print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n[!] UNEXPECTED ERROR: {e}")
        print("[-] Please report this issue with the full error message")
        exit(1)