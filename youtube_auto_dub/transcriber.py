"""Whisper-based speech transcription and device management."""

import gc
from pathlib import Path
from typing import Dict, List, Optional

import torch


def select_optimal_whisper_model(device: str = "cpu") -> str:
    """Select optimal Whisper model based on available VRAM and device.

    Args:
        device: Device type ('cuda' or 'cpu').

    Returns:
        Optimal Whisper model name.
    """
    if device == "cpu":
        return "base"

    try:
        if not torch.cuda.is_available():
            return "base"

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if gpu_memory < 4:
            return "tiny"
        elif gpu_memory < 8:
            return "base"
        elif gpu_memory < 12:
            return "small"
        elif gpu_memory < 16:
            return "medium"
        else:
            return "large-v3"
    except Exception:
        return "base"


ASR_MODEL = select_optimal_whisper_model(
    device="cuda" if torch.cuda.is_available() else "cpu"
)


class DeviceManager:
    """Centralized device detection and management."""

    def __init__(self, device: Optional[str] = None):
        """Initialize device manager.

        Args:
            device: Device type ('cuda', 'mps', or 'cpu'). If None, auto-detects.
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self._log_device_info()

    def _log_device_info(self) -> None:
        """Log device information to console."""
        print(f"[*] Device initialized: {self.device.upper()}")

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    GPU: {gpu_name} | VRAM: {gpu_memory:.1f} GB")

    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory usage information.

        Returns:
            Dictionary with allocated and reserved memory in GB.
        """
        if self.device != "cuda":
            return {"allocated": 0.0, "reserved": 0.0}

        return {
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3),
        }

    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def transcribe_with_whisper(
    audio_path: Path,
    model_size: str = ASR_MODEL,
    device: str = "cpu",
) -> List[Dict]:
    """Transcribe audio using Whisper model.

    Args:
        audio_path: Path to audio file.
        model_size: Whisper model size name.
        device: Device to run inference on.

    Returns:
        List of transcription segments with start/end times and text.
    """
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(audio_path), word_timestamps=False, language=None)
    result = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result
