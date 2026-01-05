#!/usr/bin/env python3
"""
YouTube Auto Dub - Automated Video Dubbing Pipeline.

This module provides a command-line interface for automatically dubbing YouTube videos
into different languages using AI/ML technologies. The pipeline handles:
- Video/audio download from YouTube
- Speech transcription using Whisper
- Translation using Google Translate
- Text-to-speech synthesis using Edge TTS
- Audio synchronization and video rendering

Example:
    python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es --gender female

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
License: MIT
"""

import argparse
import shutil
import subprocess
import time
import random
from pathlib import Path
from typing import Optional

# Local imports
import src.engines
import src.youtube
import src.media

def check_dependencies() -> None:
    """Verifies critical dependencies are installed and accessible.
    
    Checks for:
    - FFmpeg and FFprobe binaries (required for media processing)
    - PyTorch installation (required for Whisper model)
    
    Raises:
        SystemExit: If any critical dependency is missing.
    """
    from shutil import which
    
    # TODO: Add version checks for FFmpeg and PyTorch
    # TODO: Add support for local LLM translation models
    # TODO: Implement 4K rendering profile support
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
        # NOTE: CUDA availability significantly speeds up Whisper transcription
    except ImportError:
        print("[!] CRITICAL: PyTorch not installed.")
        print("    Install with: pip install torch")
        exit(1)

def cleanup() -> None:
    """Clean up temporary directory with retry mechanism for Windows file locks.
    
    Windows can lock files temporarily, especially after FFmpeg operations.
    This function implements an exponential backoff retry strategy.
    
    NOTE: If cleanup fails after max retries, pipeline will continue with
    existing temp files, which may cause unexpected behavior.
    """
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            if src.engines.TEMP_DIR.exists():
                shutil.rmtree(src.engines.TEMP_DIR)
            src.engines.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            return
        except PermissionError as e:
            # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
            wait_time = 0.5 * (2 ** attempt)
            print(f"[-] File locked (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    print(f"[!] WARNING: Could not fully clean temp directory after {max_retries} attempts.")
    print(f"    Files may persist in: {src.engines.TEMP_DIR}")
    print(f"    Consider manual cleanup if issues occur.")

def create_base_silence() -> Path:
    """Generate a base silence audio file for gap filling.
    
    Creates a 5-minute silence file that serves as a reference for
    filling gaps between audio segments during concatenation.
    
    Returns:
        Path: Path to the generated silence file.
        
    Raises:
        subprocess.CalledProcessError: If FFmpeg fails to generate silence.
    """
    path = src.engines.TEMP_DIR / "silence_base.wav"
    
    # Skip generation if file already exists to save time
    if path.exists():
        return path
        
    print(f"[*] Generating base silence file: {path}")
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono',
        '-t', '300',  # 5 minutes - should be longer than any video
        '-c:a', 'pcm_s16le',
        str(path)
    ]
    
    subprocess.run(cmd, check=True)
    return path

def main() -> None:
    """Main entry point for the YouTube Auto Dub pipeline.
    
    Orchestrates the complete dubbing process:
    1. Dependency validation and environment setup
    2. Video/audio download from YouTube
    3. Speech transcription using Whisper
    4. Smart audio chunking for optimal processing
    5. Translation to target language
    6. Text-to-speech synthesis with voice gender selection
    7. Audio duration fitting and synchronization
    8. Final video rendering with dubbed audio
    
    Raises:
        SystemExit: On critical errors or user interruption.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="YouTube Auto Dub - Automated Video Dubbing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang es
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang fr --gender male --gpu
  python main.py "https://youtube.com/watch?v=VIDEO_ID" --lang ja --browser chrome
        """
    )
    
    # Required arguments
    parser.add_argument("url", help="YouTube video URL to dub")
    
    # Language and voice options
    parser.add_argument(
        "--lang", "-l", 
        default="es", 
        help="Target language ISO code (e.g., es, fr, ja, vi). Default: es"
    )
    parser.add_argument(
        "--gender", "-g", 
        default="female", 
        choices=["male", "female"],
        help="Preferred voice gender for TTS. Default: female"
    )
    
    # Authentication options
    parser.add_argument(
        "--browser", "-b", 
        help="Browser to extract cookies from (chrome, edge, firefox). Close browser first!"
    )
    parser.add_argument(
        "--cookies", "-c", 
        help="Path to cookies.txt file (Netscape format) for YouTube authentication"
    )
    
    # Performance options
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Use GPU acceleration for Whisper (requires CUDA)"
    )
    
    # Subtitle options
    parser.add_argument(
        "--subtitle", "-s",
        action="store_true",
        help="Add subtitles into the video. WARNING: Creates slower render time."
    )
    
    args = parser.parse_args()

    # STEP 0: Environment Setup & Dependency Check
    print("\n" + "="*60)
    print("YOUTUBE AUTO DUB - INITIALIZING")
    print("="*60)
    
    check_dependencies()
    cleanup()
    
    # Configure processing device
    device = "cuda" if args.gpu else "cpu"
    print(f"[*] Using device: {device.upper()}")
    
    # Initialize AI engines
    engine = src.engines.Engine(device)
    
    # STEP 1: YouTube Content Download
    print(f"\n{'='*60}")
    print(f"STEP 1: DOWNLOADING CONTENT")
    print(f"{'='*60}")
    print(f"[*] Target URL: {args.url}")
    print(f"[*] Target Language: {args.lang.upper()}")
    print(f"[*] Voice Gender: {args.gender.upper()}")
    
    try:
        video_path = src.youtube.download_video(
            args.url, 
            browser=args.browser, 
            cookies_file=args.cookies
        )
        audio_path = src.youtube.download_audio(
            args.url, 
            browser=args.browser, 
            cookies_file=args.cookies
        )
        print(f"[+] Video downloaded: {video_path}")
        print(f"[+] Audio extracted: {audio_path}")
    except Exception as e:
        print(f"\n[!] DOWNLOAD FAILED: {e}")
        print("\n[-] TROUBLESHOOTING TIPS:")
        print("    1. Close all browser windows if using --browser")
        print("    2. Export fresh cookies.txt and use --cookies")
        print("    3. Check if video is private/region-restricted")
        print("    4. Verify YouTube URL is correct")
        return

    # STEP 2: Speech Transcription
    print(f"\n{'='*60}")
    print(f"STEP 2: SPEECH TRANSCRIPTION")
    print(f"{'='*60}")
    print(f"[*] Transcribing audio with Whisper ({src.engines.ASR_MODEL})...")
    
    raw_segments = engine.transcribe_safe(audio_path)
    print(f"[+] Transcription complete: {len(raw_segments)} segments")
    
    # DEBUG: Show first few segments for verification
    if len(raw_segments) > 0:
        print(f"[*] Sample segment: '{raw_segments[0]['text'][:50]}...'")
    
    # STEP 3: Smart Audio Chunking
    print(f"\n{'='*60}")
    print(f"STEP 3: INTELLIGENT CHUNKING")
    print(f"{'='*60}")
    
    # TODO: Make chunking parameters configurable
    chunks = src.engines.smart_chunk(raw_segments)
    print(f"[+] Optimized {len(raw_segments)} raw segments into {len(chunks)} chunks")
    print(f"[*] Average chunk duration: {sum(c['end']-c['start'] for c in chunks)/len(chunks):.2f}s")

    # STEP 4: Translation Processing
    print(f"\n{'='*60}")
    print(f"STEP 4: TRANSLATION ({args.lang.upper()})")
    print(f"{'='*60}")
    
    texts = [c['text'] for c in chunks]
    print(f"[*] Translating {len(texts)} text segments...")
    
    # NOTE: Translation uses Google Translate API via web scraping
    # Rate limiting is implemented to avoid IP bans
    translated_texts = engine.translate_safe(texts, args.lang)
    
    # Merge translations back into chunks
    for i, chunk in enumerate(chunks):
        chunk['trans_text'] = translated_texts[i]
    
    print(f"[+] Translation complete")
    
    # DEBUG: Show sample translation
    if len(chunks) > 0:
        original = chunks[0]['text'][:50]
        translated = chunks[0]['trans_text'][:50]
        print(f"[*] Sample: '{original}' -> '{translated}'")

    # STEP 5: Text-to-Speech Synthesis & Audio Fitting
    print(f"\n{'='*60}")
    print(f"STEP 5: TTS SYNTHESIS & AUDIO SYNC")
    print(f"{'='*60}")
    print(f"[*] Generating {args.gender} voice in {args.lang.upper()}...")
    
    # Progress tracking variables
    failed_tts = 0
    processed_chunks = 0
    
    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i:04d}.mp3"
        tts_path = src.engines.TEMP_DIR / filename
        
        try:
            # Generate TTS audio
            engine.synthesize(
                text=chunk['trans_text'], 
                target_lang=args.lang, 
                gender=args.gender, 
                out_path=tts_path
            )
            
            # IMPORTANT: Add jitter to prevent API rate limiting
            # Edge TTS has undocumented rate limits
            time.sleep(random.uniform(0.5, 1.5))
            
            # Fit audio to original timing
            slot_duration = chunk['end'] - chunk['start']
            final_audio = src.media.fit_audio(tts_path, slot_duration)
            chunk['processed_audio'] = final_audio
            
            processed_chunks += 1
            
        except Exception as e:
            print(f"\n[!] TTS failed for chunk {i}: {e}")
            failed_tts += 1
            # Continue with next chunk instead of failing completely
            continue
        
        # Progress update every 5 chunks
        if i % 5 == 0:
            progress = (i + 1) / len(chunks) * 100
            print(f"[-] Progress: {i+1}/{len(chunks)} ({progress:.1f}%)", end='\r')
    
    print(f"\n[+] TTS complete: {processed_chunks}/{len(chunks)} chunks processed")
    if failed_tts > 0:
        print(f"[!] WARNING: {failed_tts} chunks failed TTS synthesis")

    # STEP 6: Final Video Rendering
    print(f"\n{'='*60}")
    print(f"STEP 6: FINAL VIDEO RENDERING")
    print(f"{'='*60}")
    
    try:
        # Create base silence for gap filling
        silence_path = create_base_silence()
        concat_list_path = src.engines.TEMP_DIR / "concat_list.txt"
        
        print(f"[*] Creating concatenation manifest...")
        src.media.create_concat_file(chunks, silence_path, concat_list_path)
        
        # Handle subtitle generation if requested
        subtitle_path = None
        if args.subtitle:
            subtitle_path = src.engines.TEMP_DIR / "subtitles.srt"
            src.media.generate_srt(chunks, subtitle_path)
        
        # Generate output filename with subtitle suffix
        video_name = video_path.stem
        sub_suffix = "_sub" if args.subtitle else ""
        out_name = f"dubbed_{args.lang}_{args.gender}{sub_suffix}_{video_name}.mp4"
        final_output = src.engines.OUTPUT_DIR / out_name
        
        print(f"[*] Rendering final video...")
        print(f"    Source: {video_path}")
        print(f"    Output: {final_output}")
        if subtitle_path:
            print(f"    Subtitles: {subtitle_path} (Re-encoding required)")
        
        src.media.render_video(video_path, concat_list_path, final_output, subtitle_path=subtitle_path)
        
        # Verify output file was created
        if final_output.exists():
            file_size = final_output.stat().st_size / (1024 * 1024)  # MB
            print(f"\n[+] SUCCESS! Video rendered successfully.")
            print(f"    Output: {final_output}")
            print(f"    Size: {file_size:.1f} MB")
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
    # NOTE: Entry point with error handling for uncaught exceptions
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n[!] UNEXPECTED ERROR: {e}")
        print("[-] Please report this issue with the full error message")
        exit(1)