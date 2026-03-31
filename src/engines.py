"""
AI/ML Engines Module for YouTube Auto Dub.

This module provides the core AI/ML functionality including:
- Device and configuration management
- Whisper-based speech transcription  
- Google Translate integration
- Edge TTS synthesis
- Pipeline orchestration and chunking

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
"""

import torch
import asyncio
import edge_tts
import time
import random
import gc
import json
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

# Local imports
from src.googlev4 import GoogleTranslator
from src.core_utils import (
    ModelLoadError, TranscriptionError, TranslationError, TTSError, 
    AudioProcessingError, _handleError, _runFFmpegCmd, ProgressTracker, 
    _validateAudioFile, _safeFileDelete
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Working directories
CACHE_DIR = BASE_DIR / ".cache"
OUTPUT_DIR = BASE_DIR / "output"  
TEMP_DIR = BASE_DIR / "temp"

# Configuration files
LANG_MAP_FILE = BASE_DIR / "language_map.json"

# Ensure directories exist
for directory_path in [CACHE_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory_path.mkdir(parents=True, exist_ok=True)

# Audio processing settings
SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1

def _select_optimal_whisper_model(device: str = "cpu") -> str:
    """Select optimal Whisper model based on available VRAM and device.
    
    Args:
        device: Device type ('cuda' or 'cpu').
        
    Returns:
        Optimal Whisper model name.
    """
    if device == "cpu":
        return "base"  # CPU works best with base model
    
    try:
        import torch
        if not torch.cuda.is_available():
            return "base"
            
        # Get VRAM information
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        if gpu_memory < 4:
            return "tiny"  # < 4GB VRAM
        elif gpu_memory < 8:
            return "base"  # 4-8GB VRAM
        elif gpu_memory < 12:
            return "small"  # 8-12GB VRAM
        elif gpu_memory < 16:
            return "medium"  # 12-16GB VRAM
        else:
            return "large-v3"  # > 16GB VRAM - use latest large model
            
    except Exception:
        return "base"  # Fallback to base if detection fails

ASR_MODEL = _select_optimal_whisper_model(device="cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_VOICE = "en-US-AriaNeural"


# Load language configuration
try:
    with open(LANG_MAP_FILE, "r", encoding="utf-8") as f:
        LANG_DATA = json.load(f)
        print(f"[*] Loaded language configuration for {len(LANG_DATA)} languages")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"[!] WARNING: Could not load language map from {LANG_MAP_FILE}")
    LANG_DATA = {}


class DeviceManager:
    """Centralized device detection and management."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize device manager.
        
        Args:
            device: Device type ('cuda' or 'cpu'). If None, auto-detects.
        """
        if device is None:
            if torch.backends.mps.is_available(): #macOS
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self._logDeviceInfo()
    
    def _logDeviceInfo(self) -> None:
        """Log device information to console.
        
        Args:
            None
            
        Returns:
            None
        """
        print(f"[*] Device initialized: {self.device.upper()}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    GPU: {gpu_name} | VRAM: {gpu_memory:.1f} GB")
    
    def getMemoryInfo(self) -> Dict[str, float]:
        """Get GPU memory usage information.
        
        Args:
            None
            
        Returns:
            Dictionary with allocated and reserved memory in GB.
        """
        if self.device != "cuda":
            return {"allocated": 0.0, "reserved": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3)
        }
    
    def clearCache(self) -> None:
        """Clear GPU cache and run garbage collection.
        
        Args:
            None
            
        Returns:
            None
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


class ConfigManager:
    """Centralized configuration access with validation."""
    
    def getLanguageConfig(self, lang_code: str) -> Dict[str, Any]:
        """Get language configuration by language code.
        
        Args:
            lang_code: ISO language code.
            
        Returns:
            Language configuration dictionary.
        """
        return LANG_DATA.get(lang_code, {})
    
    def extractVoice(self, voice_data, fallback_gender: str = "female") -> str:
        """Extract voice string from various data formats.
        
        Args:
            voice_data: Voice data in list, string, or other format.
            fallback_gender: Default gender to use if extraction fails.
            
        Returns:
            Voice string for TTS.
        """
        if isinstance(voice_data, list):
            return voice_data[0] if voice_data else DEFAULT_VOICE
        if isinstance(voice_data, str):
            return voice_data
        return DEFAULT_VOICE
    
    def getVoicePool(self, lang_code: str, gender: str) -> list:
        """Get pool of available voices for language and gender.
        
        Args:
            lang_code: ISO language code.
            gender: Voice gender (male/female).
            
        Returns:
            List of available voice strings.
        """
        lang_config = self.getLanguageConfig(lang_code)
        voices = lang_config.get('voices', {})
        pool = voices.get(gender, [DEFAULT_VOICE])
        
        if isinstance(pool, str):
            pool = [pool]
        
        return pool


class PipelineComponent(ABC):
    """Base class for pipeline components with shared utilities."""
    
    def __init__(self, device_manager: DeviceManager, config_manager: ConfigManager):
        """Initialize pipeline component.
        
        Args:
            device_manager: Device management instance.
            config_manager: Configuration management instance.
        """
        self.device_manager = device_manager
        self.config_manager = config_manager
        self.device = device_manager.device
    
    def _validateFileExists(self, file_path: Path, description: str = "File") -> None:
        """Validate that a file exists.
        
        Args:
            file_path: Path to validate.
            description: Description for error messages.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"{description} not found: {file_path}")
    
    def _ensureDirectory(self, directory: Path) -> None:
        """Ensure directory exists, create if necessary.
        
        Args:
            directory: Directory path to ensure exists.
            
        Returns:
            None
        """
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN AI/ML ENGINE
# =============================================================================

class Engine(PipelineComponent):
    """Central AI/ML engine for YouTube Auto Dub pipeline."""
    
    def __init__(self, device: Optional[str] = None):
        device_manager = DeviceManager(device)
        config_manager = ConfigManager()
        super().__init__(device_manager, config_manager)
        
        self._asr = None
        self.translator = GoogleTranslator()
        
        print(f"[+] AI Engine initialized successfully")
            
    @property
    def asrModel(self):
        """Lazy-load Whisper ASR model.
        
        Returns:
            Loaded Whisper model instance.
            
        Raises:
            ModelLoadError: If model fails to load.
        """
        if not self._asr:
            print(f"[*] Loading Whisper model ({ASR_MODEL}) on {self.device}...")
            try:
                from faster_whisper import WhisperModel
                # faster-whisper only supports cuda and cpu (not mps)
                whisper_device = self.device if self.device == "cuda" else "cpu"
                compute_type = "float16" if whisper_device == "cuda" else "int8"
                self._asr = WhisperModel(ASR_MODEL, device=whisper_device, compute_type=compute_type)
                print(f"[+] Whisper model loaded successfully")
            except Exception as e:
                raise ModelLoadError(f"Failed to load Whisper model: {e}") from e
        return self._asr
    
    def _getLangConfig(self, lang: str) -> Dict:
        """Get language configuration.
        
        Args:
            lang: Language code.
            
        Returns:
            Language configuration dictionary.
        """
        return self.config_manager.getLanguageConfig(lang)

    def _extractVoiceString(self, voice_data: Union[str, List[str], None]) -> str:
        """Extract voice string from data.
        
        Args:
            voice_data: Voice data in various formats.
            
        Returns:
            Voice string for TTS.
        """
        return self.config_manager.extractVoice(voice_data)

    def releaseMemory(self, component: Optional[str] = None) -> None:
        """Release VRAM and clean up GPU memory.
        
        Args:
            component: Specific component to release ('asr').
                      If None, releases all components.
                      
        Returns:
            None
        """
        if component in [None, 'asr'] and self._asr:
            del self._asr
            self._asr = None
            print("[*] ASR VRAM cleared")
            self.device_manager.clearCache()

    def transcribeSafe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio with automatic memory management.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            List of transcription segments with timing.
            
        Raises:
            TranscriptionError: If transcription fails.
        """
        try:
            res = self.transcribe(audio_path)
            self.releaseMemory('asr')
            return res
        except Exception as e:
            _handleError(e, "transcription")
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def translateSafe(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate texts safely with memory management.
        
        Args:
            texts: List of text strings to translate.
            target_lang: Target language code.
            
        Returns:
            List of translated text strings.
        """
        self.releaseMemory()
        return self.translate(texts, target_lang)

    def transcribe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio using Whisper model.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            List of transcription segments with start/end times and text.
        """
        segments, _ = self.asrModel.transcribe(str(audio_path), word_timestamps=False, language=None)
        return [{'start': s.start, 'end': s.end, 'text': s.text.strip()} for s in segments]

    def translate(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate texts to target language.
        
        Args:
            texts: List of text strings to translate.
            target_lang: Target language code.
            
        Returns:
            List of translated text strings.
            
        Raises:
            TranslationError: If translation fails.
        """
        if not texts: return []
        results = []
        print(f"[*] Translating {len(texts)} segments to '{target_lang}'...")
        
        for i, text in enumerate(texts):
            try:
                if not text.strip():
                    results.append("")
                    continue
                
                translated = self.translator.translate(text, target=target_lang)
                if translated.startswith(("Error:", "Parse Error:")):
                    results.append(text)
                else:
                    results.append(translated)
                
                time.sleep(random.uniform(0.1, 0.5))
            except Exception as e:
                _handleError(e, "translation")
                raise TranslationError(f"Translation failed: {e}") from e
                
        return results

    def calcRate(self, text: str, target_dur: float, original_text: str = "") -> str:
        """Calculate speech rate adjustment for TTS with dynamic limits.
        
        Args:
            text: Text to be synthesized (translated text).
            target_dur: Target duration in seconds.
            original_text: Original text for length comparison (optional).
            
        Returns:
            Rate adjustment string (e.g., '+10%', '-5%').
        """
        words = len(text.split())
        if words == 0 or target_dur <= 0: return "+0%"
        
        # Base calculation
        wps = words / target_dur
        estimated_time = words / wps
        
        if estimated_time <= target_dur:
            return "+0%"
            
        ratio = estimated_time / target_dur
        speed_percent = int((ratio - 1) * 100)
        
        # Dynamic speed limits based on text length comparison
        if original_text:
            orig_len = len(original_text.split())
            trans_len = words
            
            # If translated text is significantly longer, allow more slowdown
            if trans_len > orig_len * 1.5:
                # Allow up to -25% slowdown for longer translations
                speed_percent = max(-25, min(speed_percent, 90))
            elif trans_len < orig_len * 0.7:
                # If translation is shorter, be more conservative with speedup
                speed_percent = max(-15, min(speed_percent, 50))
            else:
                # Normal case: -10% to 90%
                speed_percent = max(-10, min(speed_percent, 90))
        else:
            # Fallback to original limits
            speed_percent = max(-10, min(speed_percent, 90))
        
        return f"{speed_percent:+d}%"

    async def synthesize(
        self, 
        text: str, 
        target_lang: str, 
        out_path: Path,
        gender: str = "female",
        rate: str = "+0%"
    ) -> None:
        if not text.strip(): raise ValueError("Text empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            lang_cfg = self._getLangConfig(target_lang)
            voice_pool = self.config_manager.getVoicePool(target_lang, gender)
            voice = voice_pool[0] if voice_pool else DEFAULT_VOICE

            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            await communicate.save(str(out_path))
            
            if not out_path.exists() or out_path.stat().st_size < 1024:
                raise RuntimeError("TTS file invalid")
                
        except Exception as e:
            if out_path.exists(): out_path.unlink(missing_ok=True)
            _handleError(e, "TTS synthesis")
            raise TTSError(f"TTS failed: {e}") from e


def smartChunk(segments: List[Dict]) -> List[Dict]:
    n = len(segments)
    if n == 0: return []

    # Calculate segment durations and gaps for dynamic analysis
    durations = [s['end'] - s['start'] for s in segments]
    gaps = [segments[i]['start'] - segments[i-1]['end'] for i in range(1, n)]
    
    # Dynamic parameters based on actual video content
    avg_seg_dur = sum(durations) / n
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.5
    
    # Dynamic min/max duration based on content characteristics
    min_dur = max(1.0, avg_seg_dur * 0.5)  # Minimum 1s, or 50% of average
    max_dur = np.percentile(durations, 90) if n > 5 else min(15.0, avg_seg_dur * 3)
    max_dur = max(5.0, min(30.0, max_dur))  # Clamp between 5-30 seconds
    
    # Hard threshold for gap-based splitting (1.5x average gap)
    gap_threshold = max(0.4, avg_gap * 1.5)

    path = []
    curr_chunk_segs = [segments[0]]

    for i in range(1, n):
        prev = segments[i-1]
        curr = segments[i]
        gap = curr['start'] - prev['end']
        
        # Dynamic splitting criteria:
        # 1. Gap exceeds threshold (natural pause)
        # 2. Current chunk exceeds safe duration
        # 3. Dynamic lookback: consider context but don't go too far back
        current_dur = curr['end'] - curr_chunk_segs[0]['start']
        
        if gap > gap_threshold or current_dur > max_dur:
            # Close current chunk
            path.append({
                'start': curr_chunk_segs[0]['start'],
                'end': curr_chunk_segs[-1]['end'],
                'text': " ".join(s['text'] for s in curr_chunk_segs).strip()
            })
            curr_chunk_segs = [curr]
        else:
            curr_chunk_segs.append(curr)

    # Add final chunk
    if curr_chunk_segs:
        path.append({
            'start': curr_chunk_segs[0]['start'],
            'end': curr_chunk_segs[-1]['end'],
            'text': " ".join(s['text'] for s in curr_chunk_segs).strip()
        })

    print(f"[+] Smart chunking: {len(path)} chunks (Dynamic: min={min_dur:.1f}s, max={max_dur:.1f}s, gap_thr={gap_threshold:.2f}s)")
    return path