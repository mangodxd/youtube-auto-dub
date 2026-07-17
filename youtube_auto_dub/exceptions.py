"""Custom exceptions and centralized error handling for the pipeline."""

import traceback


class YouTubeAutoSubError(Exception):
    """Base exception for all YouTube Auto Sub errors."""
    pass


class ModelLoadError(YouTubeAutoSubError):
    """Raised when AI/ML model fails to load."""
    pass


class AudioProcessingError(YouTubeAutoSubError):
    """Raised when audio processing operations fail."""
    pass


class TranscriptionError(YouTubeAutoSubError):
    """Raised when speech transcription fails."""
    pass


class TranslationError(YouTubeAutoSubError):
    """Raised when text translation fails."""
    pass


class TTSError(YouTubeAutoSubError):
    """Raised when text-to-speech synthesis fails."""
    pass


class VideoProcessingError(YouTubeAutoSubError):
    """Raised when video processing operations fail."""
    pass


class ConfigurationError(YouTubeAutoSubError):
    """Raised when configuration is invalid or missing."""
    pass


class DependencyError(YouTubeAutoSubError):
    """Raised when required dependencies are missing."""
    pass


class ValidationError(YouTubeAutoSubError):
    """Raised when input validation fails."""
    pass


class ResourceError(YouTubeAutoSubError):
    """Raised when system resources are insufficient."""
    pass


def handle_error(error: Exception, context: str = "") -> None:
    """Centralized error handler that prints to stderr with traceback.

    Args:
        error: The exception that occurred.
        context: Additional context about where the error occurred.
    """
    if context:
        print(f"[!] ERROR in {context}: {error}")
    else:
        print(f"[!] ERROR: {error}")
    print(f"    Full traceback: {traceback.format_exc()}")
