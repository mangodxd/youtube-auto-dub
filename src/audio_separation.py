"""
Audio Source Separation Module for YouTube Auto Dub.

This module provides advanced audio processing capabilities including:
- Background/vocals separation using Demucs
- Audio ducking and sidechain compression
- Dynamic audio mixing with FFmpeg
- Source separation for professional dubbing workflows

The module integrates Meta's Demucs model for high-quality source separation,
enabling professional-grade audio ducking instead of using silence files.

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
"""

import subprocess
import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import os

# Local imports
from src.engines import SAMPLE_RATE, AUDIO_CHANNELS, DeviceManager, ConfigManager, PipelineComponent


class AudioSeparator(PipelineComponent):
    """Advanced audio source separation using Demucs."""
    
    def __init__(self, device_manager: Optional[DeviceManager] = None, device: Optional[str] = None):
        # Handle backward compatibility
        if device_manager is None:
            device_manager = DeviceManager(device)
        
        config_manager = ConfigManager()
        super().__init__(device_manager, config_manager)
        
        self._model = None
        self._is_loaded = False
        
        print(f"[*] Audio Separator initialized")
    
    @property
    def model_name(self) -> str:
        return "Demucs Audio Separator"
        
    @property
    def separator(self):
        """Lazy-loaded Demucs separator."""
        if not self._is_loaded:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Load Demucs model."""
        print(f"[*] Loading Demucs model (mdx_extra_q) on {self.device}...")
        
        try:
            import demucs.pretrained
            
            # Load the MDX extra quantized model (best quality/speed ratio)
            self._model = demucs.pretrained.get_model('mdx_extra_q')
            self._model.cpu()  # Start on CPU
            self._model.eval()
            
            # Move to target device if not CPU
            if self.device != "cpu":
                self._model.to(self.device)
            
            self._is_loaded = True
            print(f"[+] Demucs model loaded successfully")
            
            # Log model information
            memory_info = self.device_manager.get_memory_info()
            if memory_info['allocated'] > 0:
                print(f"    VRAM used: {memory_info['allocated']:.2f} GB")
                
        except ImportError as e:
            raise RuntimeError(f"Failed to import demucs: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Demucs model: {e}")
    
    def _unload_model(self) -> None:
        """Unload Demucs model."""
        if self._model:
            del self._model
            self._model = None
            self._is_loaded = False
    
    def release_memory(self) -> None:
        """Release VRAM and clean up GPU memory."""
        self._unload_model()
        self.device_manager.clear_cache()
        print(f"[*] Audio Separator VRAM cleared")

    def separate_audio(self, audio_path: Path, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
        """Separate audio into background music and vocals."""
        self._validate_file_exists(audio_path, "Audio file")
        
        # Set output directory
        if output_dir is None:
            output_dir = audio_path.parent
        self._ensure_directory(output_dir)
        
        try:
            print(f"[*] Separating audio: {audio_path.name}")
            
            # Run Demucs separation
            from demucs.apply import apply_model
            
            # Apply Demucs model
            sources = apply_model(
                self.separator,
                str(audio_path),
                device=self.device,
                split=True,
                overlap=0.25,
                progress=True
            )
            
            # Demucs returns: drums, bass, other, vocals
            drums, bass, other, vocals = sources
            
            # Combine drums + bass + other = background music (no vocals)
            bgm = drums + bass + other
            
            # Output paths
            bgm_path = output_dir / "no_vocals.wav"
            vocals_path = output_dir / "vocals.wav"
            
            # Save separated stems
            import torchaudio
            
            # Save BGM (drums + bass + other)
            torchaudio.save(str(bgm_path), bgm.cpu(), SAMPLE_RATE)
            
            # Save vocals
            torchaudio.save(str(vocals_path), vocals.cpu(), SAMPLE_RATE)
            
            print(f"[+] Audio separation complete:")
            print(f"    BGM (no vocals): {bgm_path}")
            print(f"    Vocals only: {vocals_path}")
            
            # Verify files were created
            if not bgm_path.exists() or not vocals_path.exists():
                raise RuntimeError("Separated audio files were not created")
            
            return bgm_path, vocals_path
            
        except Exception as e:
            raise RuntimeError(f"Audio separation failed: {e}") from e



def create_dynamic_mix(
    bgm_path: Path,
    vocals_path: Path,
    segments: List[Dict],
    output_path: Path,
    ducking_ratio: float = 0.2,
    attack_time: float = 0.1,
    release_time: float = 0.3
) -> Path:
    """Create dynamic audio mix with sidechain compression.
    
    This function creates a professional audio mix where the background
    music volume is automatically reduced (ducked) when vocals are present,
    using FFmpeg's sidechain compression filter.
    
    Args:
        bgm_path: Path to background music audio file.
        vocals_path: Path to vocals audio file.
        segments: List of audio segments with timing information.
        output_path: Path for the output mixed audio file.
        ducking_ratio: Volume reduction ratio during vocals (0.0-1.0).
        attack_time: Attack time for compression in seconds.
        release_time: Release time for compression in seconds.
        
    Returns:
        Path to the mixed audio file.
        
    Raises:
        FileNotFoundError: If input audio files don't exist.
        RuntimeError: If mixing fails.
        
    Example:
        >>> mixed_audio = create_dynamic_mix(
        ...     bgm_path, vocals_path, segments, 
        ...     output_path, ducking_ratio=0.15
        ... )
    """
    if not bgm_path.exists():
        raise FileNotFoundError(f"BGM file not found: {bgm_path}")
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals file not found: {vocals_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[*] Creating dynamic audio mix with sidechain compression...")
        print(f"    BGM: {bgm_path}")
        print(f"    Vocals: {vocals_path}")
        print(f"    Ducking ratio: {ducking_ratio:.1%}")
        
        # Create sidechain compression filter
        # This reduces BGM volume when vocals are detected
        sidechain_filter = (
            f"sidechaincompress="
            f"threshold=-30dB:"
            f"ratio=10:"
            f"attack={attack_time}:"
            f"release={release_time}:"
            f"makeup=0dB"
        )
        
        # Volume filter for ducked BGM
        volume_filter = f"volume={ducking_ratio}"
        
        # Build FFmpeg filter_complex
        # Input 0: BGM, Input 1: Vocals
        filter_complex = (
            f"[0:a]{sidechain_filter}[bgm_compressed];"
            f"[1:a]asplit=2[vocals_out][vocals_sidechain];"
            f"[bgm_compressed][vocals_sidechain]sidechaincompress=threshold=-40dB:ratio=20:attack=0.05:release=0.2[bgm_ducked];"
            f"[bgm_ducked][vocals_out]amix=inputs=2:weights=1 1[mixed]"
        )
        
        # FFmpeg command for dynamic mixing
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(bgm_path),        # Input 0: BGM
            '-i', str(vocals_path),      # Input 1: Vocals
            '-filter_complex', filter_complex,
            '-map', '[mixed]',
            '-c:a', 'pcm_s16le',         # Output as WAV
            '-ar', str(SAMPLE_RATE),
            '-ac', str(AUDIO_CHANNELS),
            str(output_path)
        ]
        
        # Run FFmpeg
        run_ffmpeg_command(cmd, description="Dynamic mix creation")
        
        # Verify output file
        if not output_path.exists():
            raise RuntimeError(f"Mixed audio file not created: {output_path}")
        
        # Get duration for reporting
        duration = get_duration(output_path)
        print(f"[+] Dynamic mix created successfully:")
        print(f"    Output: {output_path}")
        print(f"    Duration: {duration:.2f}s")
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating dynamic mix: {e}")


def create_adelay_mix(
    bgm_path: Path,
    tts_segments: List[Dict],
    output_path: Path,
    bgm_volume: float = 0.8
) -> Path:
    """Create audio mix using adelay for precise timing.
    
    This function creates a professional audio mix by placing TTS segments
    at exact timestamps using FFmpeg's adelay filter, with background
    music at a reduced volume level.
    
    Args:
        bgm_path: Path to background music audio file.
        tts_segments: List of TTS segments with 'start', 'end', 'processed_audio' keys.
        output_path: Path for the output mixed audio file.
        bgm_volume: Background music volume level (0.0-1.0).
        
    Returns:
        Path to the mixed audio file.
        
    Raises:
        FileNotFoundError: If BGM file doesn't exist.
        RuntimeError: If mixing fails.
        
    Example:
        >>> mixed_audio = create_adelay_mix(
        ...     bgm_path, tts_segments, output_path, bgm_volume=0.3
        ... )
    """
    if not bgm_path.exists():
        raise FileNotFoundError(f"BGM file not found: {bgm_path}")
    
    if not tts_segments:
        raise ValueError("No TTS segments provided for mixing")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[*] Creating adelay-based audio mix...")
        print(f"    BGM: {bgm_path}")
        print(f"    TTS segments: {len(tts_segments)}")
        print(f"    BGM volume: {bgm_volume:.1%}")
        
        # Calculate total duration
        total_duration = max(seg['end'] for seg in tts_segments)
        
        # Build input list and filter complex
        inputs = ['-i', str(bgm_path)]  # Input 0: BGM
        
        # Add TTS segments as inputs
        for i, segment in enumerate(tts_segments):
            audio_path = segment.get('processed_audio')
            if audio_path and audio_path.exists():
                inputs.extend(['-i', str(audio_path)])  # Inputs 1-N: TTS segments
        
        # Build filter complex
        filters = []
        
        # BGM with volume reduction
        filters.append(f"[0:a]volume={bgm_volume}[bgm]")
        
        # Add TTS segments with adelay
        for i, segment in enumerate(tts_segments):
            audio_path = segment.get('processed_audio')
            if audio_path and audio_path.exists():
                start_ms = int(segment['start'] * 1000)  # Convert to milliseconds
                input_idx = i + 1  # BGM is input 0
                filters.append(f"[{input_idx}:a]adelay={start_ms}|{start_ms}[tts_{i}]")
        
        # Mix all inputs
        input_labels = ['[bgm]'] + [f"[tts_{i}]" for i in range(len(tts_segments)) 
                                   if tts_segments[i].get('processed_audio') and 
                                      tts_segments[i]['processed_audio'].exists()]
        
        if len(input_labels) > 1:
            mix_filter = ''.join(input_labels) + f"amix=inputs={len(input_labels)}:duration=longest[mixed]"
            filters.append(mix_filter)
        else:
            # Only BGM available
            filters.append("[bgm]copy[mixed]")
        
        filter_complex = ';'.join(filters)
        
        # FFmpeg command
        cmd = ['ffmpeg', '-y', '-v', 'error'] + inputs + [
            '-filter_complex', filter_complex,
            '-map', '[mixed]',
            '-t', str(total_duration),
            '-c:a', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', str(AUDIO_CHANNELS),
            str(output_path)
        ]
        
        # Run FFmpeg
        run_ffmpeg_command(cmd, description="Adelay mix creation")
        
        # Verify output file
        if not output_path.exists():
            raise RuntimeError(f"Mixed audio file not created: {output_path}")
        
        print(f"[+] Adelay mix created successfully:")
        print(f"    Output: {output_path}")
        print(f"    Duration: {total_duration:.2f}s")
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating adelay mix: {e}")


from src.core_utils import get_duration, run_ffmpeg_command, ProgressTracker
