"""
Media Processing Module for YouTube Auto Dub.

This module handles all audio/video processing operations using FFmpeg.
It provides functionality for:
- Audio duration detection and analysis
- Silence generation for gap filling
- Audio time-stretching and duration fitting
- Video concatenation and rendering
- Audio synchronization and mixing

All operations use FFmpeg as the backend for maximum compatibility
and quality. The module includes comprehensive error handling and
validation for all media operations.

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
"""

import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union

# Import configuration for audio parameters
from src.engines import SAMPLE_RATE, AUDIO_CHANNELS

def get_duration(path: Path) -> float:
    """Get the duration of an audio/video file using FFprobe.
    
    This function uses FFprobe to accurately determine the duration
    of media files. It handles various audio and video formats.
    
    Args:
        path: Path to the media file.
        
    Returns:
        Duration in seconds. Returns 0.0 if duration cannot be determined.
        
    Raises:
        FileNotFoundError: If the media file doesn't exist.
        
    Example:
        >>> duration = get_duration(Path("audio.mp3"))
        >>> print(f"Duration: {duration:.2f} seconds")
        
    NOTE: This function is tolerant of errors and returns 0.0
    instead of raising exceptions for robustness in the pipeline.
    """
    if not path.exists():
        print(f"[!] ERROR: Media file not found: {path}")
        return 0.0
    
    try:
        # FFprobe command to get duration
        cmd = [
            'ffprobe', '-v', 'error', 
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            str(path)
        ]
        
        # Run FFprobe and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=30  # 30 second timeout
        )
        
        # Parse duration from output
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        else:
            print(f"[!] WARNING: No duration data for {path}")
            return 0.0
            
    except subprocess.TimeoutExpired:
        print(f"[!] ERROR: FFprobe timeout for {path}")
        return 0.0
    except subprocess.CalledProcessError as e:
        print(f"[!] ERROR: FFprobe failed for {path}: {e}")
        return 0.0
    except ValueError as e:
        print(f"[!] ERROR: Invalid duration format for {path}: {e}")
        return 0.0
    except Exception as e:
        print(f"[!] ERROR: Unexpected error getting duration for {path}: {e}")
        return 0.0

def generate_silence(duration: float, out_path: Path) -> None:
    """Generate a silence audio file using FFmpeg.
    
    This function creates a silence audio file that can be used for
    gap filling in audio concatenation operations. The silence is
    generated at the project's sample rate and channel configuration.
    
    Args:
        duration: Duration of silence in seconds.
        out_path: Output path for the silence file.
        
    Raises:
        RuntimeError: If FFmpeg fails to generate silence.
        ValueError: If duration is invalid.
        
    Example:
        >>> generate_silence(5.0, Path("silence.wav"))
        >>> print("Silence file generated")
        
    NOTE: If the output file already exists, this function skips
    generation to save time. Use force=True to override.
    """
    # Validate inputs
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip generation if file already exists
    if out_path.exists():
        print(f"[*] Silence file already exists: {out_path}")
        return
    
    try:
        print(f"[*] Generating {duration:.1f}s silence: {out_path.name}")
        
        # FFmpeg command to generate silence
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'lavfi', 
            '-i', f'anullsrc=r={SAMPLE_RATE}:cl=mono',
            '-t', str(duration),
            '-c:a', 'pcm_s16le',  # 16-bit PCM WAV
            str(out_path)
        ]
        
        # Run FFmpeg
        subprocess.run(cmd, check=True, timeout=60)
        
        # Verify output file was created
        if not out_path.exists():
            raise RuntimeError(f"Silence file not created: {out_path}")
            
        # Verify file size is reasonable
        file_size = out_path.stat().st_size
        expected_size = SAMPLE_RATE * 2 * duration  # 16-bit mono
        if file_size < expected_size * 0.9:  # Allow 10% tolerance
            print(f"[!] WARNING: Silence file may be corrupted: {file_size} bytes")
        
        print(f"[+] Silence generated successfully: {out_path}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout generating silence: {out_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to generate silence: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error generating silence: {e}")

def fit_audio(audio_path: Path, target_dur: float, max_speedup: float = 1.8) -> Path:
    """Fit audio duration to target duration using time-stretching.
    
    This function adjusts the duration of an audio file to match a target
    duration. It uses FFmpeg's atempo filter for time-stretching without
    changing pitch. If the speedup ratio is too high, it clamps to prevent
    extreme audio distortion.
    
    Args:
        audio_path: Path to the input audio file.
        target_dur: Target duration in seconds.
        max_speedup: Maximum speedup ratio allowed (default: 1.8).
                    Higher values may cause significant audio quality degradation.
                    
    Returns:
        Path to the processed audio file. If no processing is needed,
        returns the original path.
        
    Raises:
        FileNotFoundError: If input audio file doesn't exist.
        RuntimeError: If audio processing fails.
        
    Example:
        >>> fitted_audio = fit_audio(Path("speech.mp3"), 5.0)
        >>> print(f"Audio fitted to 5.0 seconds")
        
    NOTE: This function prioritizes timing accuracy over audio quality.
    For extreme speedup ratios, audio may sound unnatural but will
    maintain synchronization with the video timeline.
    
    TODO: Add quality options for better audio preservation
    TODO: Add support for local voice cloning models
    TODO: Production: Investigate audio quality degradation at high speedup ratios
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if target_dur <= 0:
        raise ValueError(f"Target duration must be positive, got {target_dur}")
    
    # Get actual audio duration
    actual_dur = get_duration(audio_path)
    if actual_dur == 0.0:
        print(f"[!] WARNING: Cannot determine duration for {audio_path}, returning original")
        return audio_path
    
    # Tolerance for floating point errors (100ms)
    tolerance = 0.1
    
    # If audio is already within tolerance, return original
    if actual_dur <= target_dur + tolerance:
        print(f"[*] Audio duration {actual_dur:.2f}s already fits target {target_dur:.2f}s")
        return audio_path
    
    # Calculate speedup ratio
    ratio = actual_dur / target_dur
    
    # Clamp ratio to maximum allowed speedup
    if ratio > max_speedup:
        print(f"[!] WARNING: Speedup ratio {ratio:.2f}x exceeds max {max_speedup}x")
        print(f"    Audio will be longer than video slot (may overlap next segment)")
        ratio = max_speedup
    
    # Generate output path
    out_path = audio_path.parent / f"{audio_path.stem}_fit.wav"
    
    try:
        print(f"[*] Fitting audio: {actual_dur:.2f}s -> {target_dur:.2f}s ({ratio:.2f}x speed)")
        
        # Build atempo filter chain
        # FFmpeg atempo has limits of 0.5-2.0, so we chain for higher ratios
        filter_complex = f"atempo={ratio:.4f}"
        
        # FFmpeg command for time-stretching
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(audio_path),
            '-filter:a', filter_complex,
            '-t', str(target_dur),  # Strictly enforce target duration
            '-c:a', 'pcm_s16le',  # Output as WAV
            str(out_path)
        ]
        
        # Run FFmpeg
        subprocess.run(cmd, check=True, timeout=120)
        
        # Verify output file
        if not out_path.exists():
            raise RuntimeError(f"Fitted audio file not created: {out_path}")
        
        # Verify duration is close to target
        fitted_dur = get_duration(out_path)
        duration_diff = abs(fitted_dur - target_dur)
        
        if duration_diff > 0.1:  # More than 100ms difference
            print(f"[!] WARNING: Duration mismatch: got {fitted_dur:.2f}s, wanted {target_dur:.2f}s")
        else:
            print(f"[+] Audio fitted successfully: {fitted_dur:.2f}s")
        
        return out_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout fitting audio: {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"[!] ERROR: Audio fitting failed for {audio_path}: {e}")
        print(f"    Returning original audio (may cause timing issues)")
        return audio_path  # Fallback to original
    except Exception as e:
        raise RuntimeError(f"Unexpected error fitting audio: {e}")

def create_concat_file(segments: List[Dict], silence_ref: Path, output_txt: Path) -> None:
    """Create FFmpeg concatenation manifest for audio segments.
    
    This function generates a concat demuxer file that tells FFmpeg
    how to concatenate audio segments with proper timing. It handles
    gaps between segments by inserting silence and ensures precise
    timing alignment with the original video timeline.
    
    Args:
        segments: List of audio segments with timing information.
                 Each segment should have 'start', 'end', and 'processed_audio' keys.
        silence_ref: Path to a reference silence file for gap filling.
        output_txt: Path for the output concat manifest file.
        
    Raises:
        FileNotFoundError: If silence reference file doesn't exist.
        ValueError: If segment format is invalid.
        
    Example:
        >>> create_concat_file(chunks, silence_path, Path("concat.txt"))
        >>> print("Concat manifest created")
        
    NOTE: The concat file format uses absolute paths for reliability.
    Gaps smaller than 0.01 seconds are ignored to avoid excessive
    fragmentation in the final audio.
    
    TODO: Add support for crossfading between segments.
    """
    if not silence_ref.exists():
        raise FileNotFoundError(f"Silence reference file not found: {silence_ref}")
    
    if not segments:
        print(f"[!] WARNING: No segments provided for concatenation")
        return
    
    # Validate segment format
    required_keys = {'start', 'end'}
    for i, seg in enumerate(segments):
        if not required_keys.issubset(seg.keys()):
            raise ValueError(f"Segment {i} missing required keys: {required_keys - seg.keys()}")
        if seg['start'] >= seg['end']:
            raise ValueError(f"Segment {i} has invalid timing: start >= end")
    
    try:
        print(f"[*] Creating concat manifest for {len(segments)} segments")
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            current_time = 0.0
            gap_count = 0
            audio_count = 0
            
            for i, segment in enumerate(segments):
                start_time = segment['start']
                end_time = segment['end']
                audio_path = segment.get('processed_audio')
                
                # Calculate gap from previous segment
                gap = start_time - current_time
                
                # Insert silence for gaps (minimum threshold to avoid fragmentation)
                if gap > 0.01:  # 10ms minimum gap
                    f.write(f"file '{silence_ref.resolve().as_posix()}'\n")
                    f.write(f"inpoint 0\n")
                    f.write(f"outpoint {gap:.4f}\n")
                    gap_count += 1
                
                # Insert audio segment if available
                if audio_path and audio_path.exists():
                    f.write(f"file '{audio_path.resolve().as_posix()}'\n")
                    f.write(f"duration {end_time - start_time:.4f}\n")
                    audio_count += 1
                else:
                    # Fallback to silence if audio segment is missing
                    segment_duration = end_time - start_time
                    if segment_duration > 0:
                        f.write(f"file '{silence_ref.resolve().as_posix()}'\n")
                        f.write(f"inpoint 0\n")
                        f.write(f"outpoint {segment_duration:.4f}\n")
                        gap_count += 1
                        print(f"[!] WARNING: Missing audio for segment {i}, using silence")
                
                # Update current time
                current_time = end_time
            
            # Log statistics
            total_duration = segments[-1]['end'] - segments[0]['start']
            print(f"[+] Concat manifest created:")
            print(f"    Total duration: {total_duration:.2f}s")
            print(f"    Audio segments: {audio_count}")
            print(f"    Silence gaps: {gap_count}")
            print(f"    Output file: {output_txt}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to create concat manifest: {e}")

def render_video(video_path: Path, concat_file: Path, output_path: Path, subtitle_path: Optional[Path] = None) -> None:
    """Render the final dubbed video using FFmpeg with optional hard subtitles.
    
    This function renders the final video by combining the original video
    with the dubbed audio track. If subtitle_path is provided, it burns
    the subtitles into the video (hardsubs), which requires re-encoding.
    Without subtitles, it uses fast stream copying.
    
    Args:
        video_path: Path to the original video file.
        concat_file: Path to the audio concatenation manifest file.
        output_path: Path where the final video will be saved.
        subtitle_path: Optional path to SRT subtitle file for hardsubs.
        
    Raises:
        FileNotFoundError: If input files don't exist.
        RuntimeError: If video rendering fails.
        
    Example:
        >>> render_video(video, audio_file, output)
        >>> render_video(video, audio_file, output, subtitle_path)  # With hardsubs
        
    NOTE: Hardsubs (subtitle_path provided) require video re-encoding
    and will be significantly slower than stream copying.
    """
    # Validate input files
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not concat_file.exists():
        raise FileNotFoundError(f"Concat file not found: {concat_file}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[*] Rendering final video:")
        print(f"    Source video: {video_path}")
        print(f"    Audio concat: {concat_file}")
        if subtitle_path:
            print(f"    Subtitles: {subtitle_path} (Re-encoding required)")
        print(f"    Output: {output_path}")
        
        # Base FFmpeg command
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(video_path),        # Input 0: Video
            '-f', 'concat', '-safe', '0', 
            '-i', str(concat_file),       # Input 1: Audio concat
            '-map', '0:v',                # Use video from input 0
            '-map', '1:a',                # Use audio from input 1
        ]
        
        if subtitle_path:
            # --- HARD SUB MODE (Re-encode required) ---
            sub_path_posix = subtitle_path.resolve().as_posix()
            sub_path_escaped = sub_path_posix.replace(":", "\\:")
            
            # Subtitle style configuration
            style = "FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=20"
            
            cmd.extend([
                '-vf', f"subtitles='{sub_path_escaped}':force_style='{style}'",
                '-c:v', 'libx264',        # Re-encode video for hardsubs
                '-preset', 'fast',        # Balance speed and quality
                '-crf', '23'              # Quality factor (18-28 range)
            ])
        else:
            # --- FAST MODE (Stream copy) ---
            cmd.extend([
                '-c:v', 'copy',           # Copy video stream (no re-encoding)
            ])
        
        # Audio settings (same for both modes)
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', str(SAMPLE_RATE),
            '-ac', str(AUDIO_CHANNELS),
            '-shortest',
            str(output_path)
        ])
        
        # Run FFmpeg with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        # Monitor progress (simple implementation)
        print("    Processing...")
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Could parse FFmpeg progress here if needed
                pass
        
        # Check result
        return_code = process.poll()
        if return_code != 0:
            error_output = process.stderr.read()
            raise RuntimeError(f"FFmpeg failed with code {return_code}: {error_output}")
        
        # Verify output file
        if not output_path.exists():
            raise RuntimeError(f"Output video file not created: {output_path}")
        
        # Get file size for reporting
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"[+] Video rendering complete:")
        print(f"    Output file: {output_path}")
        print(f"    File size: {file_size:.1f} MB")
        
        # Verify output has both video and audio streams
        try:
            # Quick verification with FFprobe
            verify_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(output_path)
            ]
            video_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            verify_cmd[2] = '-select_streams'  # Change to audio
            verify_cmd[3] = 'a'
            audio_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            if video_result.returncode == 0 and audio_result.returncode == 0:
                video_codec = video_result.stdout.strip()
                audio_codec = audio_result.stdout.strip()
                print(f"    Video codec: {video_codec}")
                print(f"    Audio codec: {audio_codec}")
            else:
                print(f"[!] WARNING: Could not verify output codecs")
                
        except Exception as e:
            print(f"[!] WARNING: Output verification failed: {e}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout rendering video: {video_path}")
    except Exception as e:
        raise RuntimeError(f"Video rendering failed: {e}")


def generate_srt(segments: List[Dict], output_path: Path) -> None:
    """Generate SRT subtitle file from transcript segments.
    
    This function creates an SRT subtitle file from transcript segments
    with timing information. The subtitles are formatted according to
    the SRT standard and can be used for video rendering.
    
    Args:
        segments: List of transcript segments with 'start', 'end', 'trans_text' keys.
        output_path: Path where the SRT file will be saved.
        
    Raises:
        FileNotFoundError: If output directory doesn't exist.
        ValueError: If segment format is invalid.
        
    Example:
        >>> generate_srt(chunks, Path("subtitles.srt"))
        >>> print("SRT file generated")
        
    TODO: Add subtitle formatting options and style customization.
    """
    if not segments:
        print(f"[!] WARNING: No segments provided for subtitle generation")
        return
    
    # Validate segment format
    required_keys = {'start', 'end', 'trans_text'}
    for i, seg in enumerate(segments):
        if not required_keys.issubset(seg.keys()):
            raise ValueError(f"Segment {i} missing required keys: {required_keys - seg.keys()}")
        if seg['start'] >= seg['end']:
            raise ValueError(f"Segment {i} has invalid timing: start >= end")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[*] Generating SRT subtitles: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = segment['start']
                end_time = segment['end']
                text = segment.get('trans_text', segment.get('text', ''))
                
                # Format timestamps as HH:MM:SS,mmm
                start_srt = _format_timestamp_srt(start_time)
                end_srt = _format_timestamp_srt(end_time)
                
                # Write SRT block
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text.strip()}\n\n")
        
        print(f"[+] SRT subtitles generated: {output_path}")
        print(f"    Total subtitles: {len(segments)}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate SRT file: {e}") from e


def _format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"