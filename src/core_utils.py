"""
Core utilities and exceptions for YouTube Auto Dub.

This module consolidates shared utilities, exceptions, and helper functions
used across the entire pipeline to reduce code duplication.

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
"""

import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class YouTubeAutoDubError(Exception):
    """Base exception for all YouTube Auto Dub errors."""
    pass


class ModelLoadError(YouTubeAutoDubError):
    """Raised when AI/ML model fails to load."""
    pass


class AudioProcessingError(YouTubeAutoDubError):
    """Raised when audio processing operations fail."""
    pass


class TranscriptionError(YouTubeAutoDubError):
    """Raised when speech transcription fails."""
    pass


class TranslationError(YouTubeAutoDubError):
    """Raised when text translation fails."""
    pass


class TTSError(YouTubeAutoDubError):
    """Raised when text-to-speech synthesis fails."""
    pass


class VideoProcessingError(YouTubeAutoDubError):
    """Raised when video processing operations fail."""
    pass


class ConfigurationError(YouTubeAutoDubError):
    """Raised when configuration is invalid or missing."""
    pass


class DependencyError(YouTubeAutoDubError):
    """Raised when required dependencies are missing."""
    pass


class ValidationError(YouTubeAutoDubError):
    """Raised when input validation fails."""
    pass


class ResourceError(YouTubeAutoDubError):
    """Raised when system resources are insufficient."""
    pass


# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

def handle_error(error: Exception, context: str = "") -> None:
    """Centralized error handling with context.
    
    Args:
        error: The exception that occurred.
        context: Additional context about where the error occurred.
    """
    if context:
        print(f"[!] ERROR in {context}: {error}")
    else:
        print(f"[!] ERROR: {error}")
    
    print(f"    Full traceback: {traceback.format_exc()}")


def safe_execute(func, *args, error_type: type = YouTubeAutoDubError, context: str = "", **kwargs):
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute.
        *args: Function arguments.
        error_type: Type of exception to raise on failure.
        context: Context for error messages.
        **kwargs: Function keyword arguments.
        
    Returns:
        Result of function execution.
        
    Raises:
        error_type: Specified exception type on failure.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if isinstance(e, error_type):
            raise
        else:
            raise error_type(f"{context}: {str(e)}") from e


# =============================================================================
# AUDIO/MEDIA UTILITIES
# =============================================================================

def get_duration(path: Path) -> float:
    """Get the duration of an audio/video file using FFprobe.
    
    Args:
        path: Path to the media file.
        
    Returns:
        Duration in seconds. Returns 0.0 if duration cannot be determined.
    """
    if not path.exists():
        return 0.0
    
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(path)
        ]
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )
        
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        else:
            return 0.0
            
    except Exception:
        return 0.0


def run_ffmpeg_command(cmd: List[str], timeout: int = 300, description: str = "FFmpeg operation") -> None:
    """Run FFmpeg command with consistent error handling.
    
    Args:
        cmd: FFmpeg command to run.
        timeout: Command timeout in seconds.
        description: Description for error messages.
        
    Raises:
        RuntimeError: If FFmpeg command fails.
    """
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{description} timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{description} failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during {description}: {e}")


def validate_audio_file(file_path: Path, min_size: int = 1024) -> bool:
    """Validate that audio file exists and has minimum size.
    
    Args:
        file_path: Path to audio file.
        min_size: Minimum file size in bytes.
        
    Returns:
        True if file is valid, False otherwise.
    """
    if not file_path.exists():
        return False
    
    if file_path.stat().st_size < min_size:
        return False
    
    return True


def safe_file_delete(file_path: Path) -> None:
    """Safely delete file with error handling.
    
    Args:
        file_path: Path to file to delete.
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"[!] WARNING: Could not delete file {file_path}: {e}")


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, description: str = "Operation"):
    """Retry function with exponential backoff.
    
    Args:
        func: Function to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        description: Description for logging.
        
    Returns:
        Result of function execution.
        
    Raises:
        Last exception if all retries fail.
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                break
                
            wait_time = base_delay * (2 ** attempt)
            print(f"[-] {description} failed (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    raise last_exception


class ProgressTracker:
    """Simple progress tracking for long operations."""
    
    def __init__(self, total: int, description: str = "Processing", update_interval: int = 10):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process.
            description: Description for progress messages.
            update_interval: How often to update progress (every N items).
        """
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.current = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress counter.
        
        Args:
            increment: Number of items processed.
        """
        self.current += increment
        
        if self.current % self.update_interval == 0 or self.current >= self.total:
            progress = (self.current / self.total) * 100
            print(f"[-] {self.description}: {self.current}/{self.total} ({progress:.1f}%)", end='\r')
            
            if self.current >= self.total:
                print()  # New line when complete
