"""Media Processing Module for YouTube Auto Dub.

This module handles all audio/video processing operations using FFmpeg.
It provides functionality for:
- Audio duration detection and analysis
- Silence generation for gap filling
- Audio time-stretching and duration fitting (PADDING logic added)
- Video concatenation and rendering (Volume Mixing fixed)
- Audio synchronization and mixing

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.1.0 (Patched)
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from src.engines import SAMPLE_RATE, AUDIO_CHANNELS


def _get_duration(path: Path) -> float:
    """Get the duration of an audio/video file using FFprobe."""
    if not path.exists():
        print(f"[!] ERROR: Media file not found: {path}")
        return 0.0
    
    try:
        cmd = [
            'ffprobe', '-v', 'error', 
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            str(path)
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=60  # Increased from 30s to 60s for better reliability
        )
        
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        else:
            return 0.0
            
    except Exception as e:
        print(f"[!] ERROR: Getting duration failed for {path}: {e}")
        return 0.0


def _generate_silence_segment(duration: float, silence_ref: Path) -> Optional[Path]:
    """Generate a small silence segment for the concat list."""
    if duration <= 0:
        return None
    
    # Use the parent folder of the reference silence file
    output_path = silence_ref.parent / f"gap_{duration:.4f}.wav"
    
    if output_path.exists():
        return output_path

    try:
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'lavfi', '-i', f'anullsrc=r={SAMPLE_RATE}:cl=mono',
            '-t', f"{duration:.4f}",
            '-c:a', 'pcm_s16le',
            str(output_path)
        ]
        subprocess.run(cmd, check=True)
        return output_path
    except Exception:
        return None

def _analyze_audio_loudness(audio_path: Path) -> Optional[float]:
    """Analyze audio loudness using FFmpeg volumedetect filter.
    
    Args:
        audio_path: Path to audio file to analyze.
        
    Returns:
        Mean volume in dB, or None if analysis fails.
    """
    if not audio_path.exists():
        return None
        
    try:
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(audio_path),
            '-filter:a', 'volumedetect',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        
        # Parse mean volume from output
        for line in result.stderr.split('\n'):
            if 'mean_volume:' in line:
                # Extract dB value from line like: "mean_volume: -15.2 dB"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
        
        return None
    except Exception:
        return None


def fit_audio(audio_path: Path, target_dur: float) -> Path:
    if not audio_path.exists() or target_dur <= 0:
        return audio_path
    
    actual_dur = _get_duration(audio_path)
    if actual_dur == 0.0:
        return audio_path
    
    out_path = audio_path.parent / f"{audio_path.stem}_fit.wav"
    
    # Increased tolerance from 0.05s to 0.15s for more natural audio
    if actual_dur > target_dur + 0.15:
        ratio = actual_dur / target_dur
        filter_chain = []
        current_ratio = ratio
        
        # Dynamic speed limit: max 1.5x instead of 2.0x to avoid chipmunk effect
        max_speed_ratio = 1.5
        
        while current_ratio > max_speed_ratio:
            filter_chain.append(f"atempo={max_speed_ratio}")
            current_ratio /= max_speed_ratio
            
        if current_ratio > 1.0:
            filter_chain.append(f"atempo={current_ratio:.4f}")
        
        filter_complex = ",".join(filter_chain)
        
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(audio_path),
            '-filter:a', f"{filter_complex},aresample=24000",
            '-t', f"{target_dur:.4f}",
            '-c:a', 'pcm_s16le',
            str(out_path)
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(audio_path),
            '-filter:a', f"apad,aresample=24000",
            '-t', f"{target_dur:.4f}",
            '-c:a', 'pcm_s16le',
            str(out_path)
        ]
    print(f"Fiting {actual_dur:.4f}s to {target_dur:.4f}s")
    
    try:
        subprocess.run(cmd, check=True, timeout=120)
        return out_path
    except Exception:
        return audio_path

def create_concat_file(segments: List[Dict], silence_ref: Path, output_txt: Path) -> None:
    if not segments:
        return
    
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            current_timeline = 0.0
            
            for segment in segments:
                start_time = segment['start']
                end_time = segment['end']
                audio_path = segment.get('processed_audio')
                
                gap = start_time - current_timeline
                if gap > 0.01:
                    silence_gap = _generate_silence_segment(gap, silence_ref)
                    if silence_gap:
                        f.write(f"file '{silence_gap.resolve().as_posix()}'\n")
                        current_timeline += gap
                
                if audio_path and audio_path.exists():
                    f.write(f"file '{audio_path.resolve().as_posix()}'\n")
                    current_timeline += (end_time - start_time)
                else:
                    dur = end_time - start_time
                    silence_err = _generate_silence_segment(dur, silence_ref)
                    if silence_err:
                        f.write(f"file '{silence_err.resolve().as_posix()}'\n")
                    current_timeline += dur
                    
    except Exception as e:
        raise RuntimeError(f"Failed to create concat manifest: {e}")


def _check_subtitle_filter() -> bool:
    """Check if FFmpeg has the subtitles filter (requires libass)."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-filters'], capture_output=True, text=True, timeout=10
        )
        return 'subtitles' in result.stdout
    except Exception:
        return False


def render_video(video_path: Path, concat_file: Optional[Path], output_path: Path, subtitle_path: Optional[Path] = None, background_audio: Optional[Path] = None) -> None:
    """Render final video with Dynamic Volume Mixing.

    Supports three modes:
    - Dubbed: concat_file provided, mixes dubbed audio with original/background
    - Subtitle-only: concat_file is None, burns subtitles onto original video
    - Dubbed + separated BGM: concat_file + background_audio for clean mixing
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if concat_file is not None and not concat_file.exists():
        raise FileNotFoundError(f"Concat file not found: {concat_file}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[*] Rendering final video...")

        # Check if FFmpeg has subtitles filter support
        has_subtitle_filter = _check_subtitle_filter()

        # === SUBTITLE-ONLY MODE (no dubbed audio) ===
        if concat_file is None:
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(video_path),
                '-map', '0:v', '-map', '0:a',
                '-c:a', 'copy',
            ]

            if subtitle_path and subtitle_path.exists() and has_subtitle_filter:
                sub_path = str(subtitle_path.resolve())
                sub_path = sub_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
                cmd.extend(['-c:v', 'libx264', '-vf', f"subtitles='{sub_path}'"])
            else:
                cmd.extend(['-c:v', 'copy'])
                if subtitle_path and subtitle_path.exists() and not has_subtitle_filter:
                    print(f"[-] FFmpeg lacks subtitles filter (needs libass). SRT saved separately.")

            cmd.append(str(output_path))
            subprocess.run(cmd, check=True, timeout=None)

            if not output_path.exists():
                raise RuntimeError("Output file not created")
            print(f"[+] Video rendered successfully: {output_path}")
            return

        # === DUBBED MODE (with TTS audio) ===

        # Determine background audio source
        bg_source = background_audio if (background_audio and background_audio.exists()) else None

        # Dynamic volume mixing
        analyze_target = bg_source or video_path
        original_loudness = _analyze_audio_loudness(analyze_target)

        if bg_source:
            # When we have separated background, keep it at higher volume
            bg_volume = 0.5
            print(f"[*] Using separated background audio at {bg_volume*100:.0f}% volume")
        elif original_loudness is not None:
            if original_loudness > -10:
                bg_volume = 0.08
            elif original_loudness > -20:
                bg_volume = 0.15
            else:
                bg_volume = 0.25
            print(f"[*] Dynamic volume mixing: original={original_loudness:.1f}dB, bg_volume={bg_volume*100:.0f}%")
        else:
            bg_volume = 0.15
            print(f"[*] Using default volume mixing: bg_volume={bg_volume*100:.0f}%")

        # Build subtitle filter if needed (requires libass in FFmpeg)
        sub_filter = ""
        video_codec = "copy"
        if subtitle_path and subtitle_path.exists() and has_subtitle_filter:
            sub_path = str(subtitle_path.resolve())
            sub_path = sub_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
            sub_filter = f"[0:v]subtitles='{sub_path}'[outv]; "
            video_codec = "libx264"
            video_map = "[outv]"
        else:
            video_map = "0:v"
            if subtitle_path and subtitle_path.exists() and not has_subtitle_filter:
                print(f"[-] FFmpeg lacks subtitles filter (needs libass). SRT saved separately.")

        if bg_source:
            filter_complex = (
                f"{sub_filter}"
                f"[2:a]volume={bg_volume}[bg]; "
                "[bg][1:a]amix=inputs=2:duration=first:dropout_transition=0[outa]"
            )
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(video_path),
                '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                '-i', str(bg_source),
                '-filter_complex', filter_complex,
                '-map', video_map,
                '-map', '[outa]',
                '-c:v', video_codec,
                '-c:a', 'aac', '-b:a', '192k',
                '-ar', str(SAMPLE_RATE),
                '-ac', str(AUDIO_CHANNELS),
                '-shortest'
            ]
        else:
            filter_complex = (
                f"{sub_filter}"
                f"[0:a]volume={bg_volume}[bg]; "
                "[bg][1:a]amix=inputs=2:duration=first:dropout_transition=0[outa]"
            )
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(video_path),
                '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                '-filter_complex', filter_complex,
                '-map', video_map,
                '-map', '[outa]',
                '-c:v', video_codec,
                '-c:a', 'aac', '-b:a', '192k',
                '-ar', str(SAMPLE_RATE),
                '-ac', str(AUDIO_CHANNELS),
                '-shortest'
            ]

        cmd.append(str(output_path))

        subprocess.run(cmd, check=True, timeout=None)

        if not output_path.exists():
            raise RuntimeError("Output file not created")

        print(f"[+] Video rendered successfully: {output_path}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg rendering failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Rendering error: {e}")


def generate_srt(segments: List[Dict], output_path: Path) -> None:
    """Generate SRT subtitle file."""
    if not segments: return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start = _format_timestamp_srt(segment['start'])
                end = _format_timestamp_srt(segment['end'])
                text = segment.get('trans_text', '').strip()
                
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                
        print(f"[+] SRT subtitles generated")
    except Exception as e:
        print(f"[!] Warning: SRT generation failed: {e}")


def _format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"