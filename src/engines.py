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
import os
import gc
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

# Local imports
from src.googlev4 import GoogleTranslator
from src.core_utils import (
    ModelLoadError, TranscriptionError, TranslationError, TTSError, 
    AudioProcessingError, handle_error, safe_execute, get_duration, 
    run_ffmpeg_command, ProgressTracker, validate_audio_file, safe_file_delete
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
ASR_MODEL = "base"
DEFAULT_VOICE = "en-US-AriaNeural"

# Load language configuration
try:
    with open(LANG_MAP_FILE, "r", encoding="utf-8") as f:
        LANG_DATA = json.load(f)
        print(f"[*] Loaded language configuration for {len(LANG_DATA)} languages")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"[!] WARNING: Could not load language map from {LANG_MAP_FILE}")
    LANG_DATA = {}

# =============================================================================
# DEVICE AND CONFIGURATION MANAGEMENT
# =============================================================================

class DeviceManager:
    """Centralized device detection and management."""
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self._log_device_info()
    
    def _log_device_info(self) -> None:
        print(f"[*] Device initialized: {self.device.upper()}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    GPU: {gpu_name} | VRAM: {gpu_memory:.1f} GB")
    
    def get_memory_info(self) -> Dict[str, float]:
        if self.device != "cuda":
            return {"allocated": 0.0, "reserved": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved": torch.cuda.memory_reserved(0) / (1024**3)
        }
    
    def clear_cache(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


class ConfigManager:
    """Centralized configuration access with validation."""
    
    def get_language_config(self, lang_code: str) -> Dict[str, Any]:
        return LANG_DATA.get(lang_code, {})
    
    def extract_voice(self, voice_data, fallback_gender: str = "female") -> str:
        if isinstance(voice_data, list):
            return voice_data[0] if voice_data else DEFAULT_VOICE
        if isinstance(voice_data, str):
            return voice_data
        return DEFAULT_VOICE
    
    def get_voice_pool(self, lang_code: str, gender: str) -> list:
        lang_config = self.get_language_config(lang_code)
        voices = lang_config.get('voices', {})
        pool = voices.get(gender, [DEFAULT_VOICE])
        
        if isinstance(pool, str):
            pool = [pool]
        
        return pool


class PipelineComponent(ABC):
    """Base class for pipeline components with shared utilities."""
    
    def __init__(self, device_manager: DeviceManager, config_manager: ConfigManager):
        self.device_manager = device_manager
        self.config_manager = config_manager
        self.device = device_manager.device
    
    def _validate_file_exists(self, file_path: Path, description: str = "File") -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"{description} not found: {file_path}")
    
    def _ensure_directory(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN AI/ML ENGINE
# =============================================================================

class Engine(PipelineComponent):
    """Central AI/ML engine for YouTube Auto Dub pipeline."""
    
    def __init__(self, device: Optional[str] = None, hf_token: Optional[str] = None):
        device_manager = DeviceManager(device)
        config_manager = ConfigManager()
        super().__init__(device_manager, config_manager)
        
        self._asr = None
        self._separator = None
        self._diarizer = None
        self.hf_token = hf_token or self._get_huggingface_token()
        self.translator = GoogleTranslator()
        
        print(f"[+] AI Engine initialized successfully")
    
    def _get_huggingface_token(self) -> Optional[str]:
        return os.getenv('HF_TOKEN')
            
    @property
    def asr_model(self):
        if not self._asr:
            print(f"[*] Loading Whisper model ({ASR_MODEL}) on {self.device}...")
            try:
                from faster_whisper import WhisperModel
                compute_type = "float16" if self.device == "cuda" else "int8"
                self._asr = WhisperModel(ASR_MODEL, device=self.device, compute_type=compute_type)
                print(f"[+] Whisper model loaded successfully")
            except Exception as e:
                raise ModelLoadError(f"Failed to load Whisper model: {e}") from e
        return self._asr

    @property
    def separator(self):
        if not self._separator:
            from src.audio_separation import AudioSeparator
            self._separator = AudioSeparator(device_manager=self.device_manager)
        return self._separator
    
    @property
    def diarizer(self):
        if not self._diarizer:
            from src.speaker_diarization import SpeakerDiarizer
            self._diarizer = SpeakerDiarizer(
                device_manager=self.device_manager, 
                hf_token=self.hf_token
            )
        return self._diarizer
    
    def _get_lang_config(self, lang: str) -> Dict:
        return self.config_manager.get_language_config(lang)

    def _extract_voice_string(self, voice_data: Union[str, List[str], None]) -> str:
        return self.config_manager.extract_voice(voice_data)

    def release_memory(self, component: Optional[str] = None) -> None:
        """Release VRAM and clean up GPU memory."""
        components = []
        if component in [None, 'asr'] and self._asr:
            components.append(('asr', self._asr))
            self._asr = None
        if component in [None, 'separator'] and self._separator:
            components.append(('separator', self._separator))
            self._separator = None
        if component in [None, 'diarizer'] and self._diarizer:
            components.append(('diarizer', self._diarizer))
            self._diarizer = None
        
        for name, obj in components:
            if hasattr(obj, 'release_memory'):
                obj.release_memory()
            elif name == 'asr':
                del obj
            print(f"[*] {name.title()} VRAM cleared")
        
        if components:
            self.device_manager.clear_cache()

    def transcribe_safe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio with automatic memory management."""
        try:
            res = self.transcribe(audio_path)
            self.release_memory('asr')
            return res
        except Exception as e:
            handle_error(e, "transcription")
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def translate_safe(self, texts: List[str], target_lang: str) -> List[str]:
        """Translate texts safely."""
        self.release_memory()
        return self.translate(texts, target_lang)

    def transcribe(self, audio_path: Path) -> List[Dict]:
        segments, _ = self.asr_model.transcribe(str(audio_path), word_timestamps=False, language=None)
        return [{'start': s.start, 'end': s.end, 'text': s.text.strip()} for s in segments]

    def translate(self, texts: List[str], target_lang: str) -> List[str]:
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
                handle_error(e, "translation")
                raise TranslationError(f"Translation failed: {e}") from e
                
        return results

    def synthesize(self, text: str, target_lang: str, gender: str, out_path: Path) -> None:
        """Synthesize speech. Handles both List and String voice configs."""
        if not text.strip(): raise ValueError("Text empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            lang_cfg = self._get_lang_config(target_lang)
            voices = lang_cfg.get('voices', {})
            
            raw_voice = voices.get(gender)
            voice = self._extract_voice_string(raw_voice)
            
            asyncio.run(edge_tts.Communicate(text, voice=voice).save(str(out_path)))
            
            if not validate_audio_file(out_path):
                raise AudioProcessingError("TTS file invalid")
                
        except Exception as e:
            safe_file_delete(out_path)
            handle_error(e, "TTS synthesis")
            raise TTSError(f"TTS failed: {e}") from e

    def synthesize_multi_speaker(
        self, 
        text: str, 
        target_lang: str, 
        speaker_id: str, 
        out_path: Path,
        voice_assignments: Optional[Dict[str, str]] = None
    ) -> None:
        """Synthesize speech with speaker assignments. Safe for List configs."""
        if not text.strip(): raise ValueError("Text empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            lang_cfg = self._get_lang_config(target_lang)
            voice = None
            
            # 1. Manual/Auto Assignment (Priority)
            if voice_assignments and speaker_id in voice_assignments:
                voice = voice_assignments[speaker_id]
            
            # 2. Legacy Multi-speaker Config Check
            if not voice:
                multi_speakers = lang_cfg.get('multi_speaker', {})
                spk_key = f"speaker_{int(speaker_id.split('_')[1]):02d}" if speaker_id.startswith('SPEAKER_') else None
                if spk_key and spk_key in multi_speakers:
                    voice = multi_speakers[spk_key]

            # 3. Fallback to Gender Pool (Fixed Logic)
            if not voice:
                voices = lang_cfg.get('voices', {})
                raw_voice = voices.get('female')
                voice = self._extract_voice_string(raw_voice)
                print(f"[*] Fallback voice for {speaker_id}: {voice}")

            asyncio.run(edge_tts.Communicate(text, voice=voice).save(str(out_path)))
            
            if not out_path.exists() or out_path.stat().st_size < 1024:
                raise RuntimeError("TTS file invalid")
                
        except Exception as e:
            if out_path.exists(): out_path.unlink()
            handle_error(e, "multi-speaker TTS synthesis")
            raise TTSError(f"Multi-speaker TTS failed: {e}") from e

    def analyze_speakers(self, audio_path: Path, min_speakers: int = 1, max_speakers: int = 8) -> Dict:
        """Wrapper for diarization."""
        try:
            return self.diarizer.diarize_audio(audio_path, min_speakers, max_speakers)
        finally:
            self.release_memory('diarizer')

    def separate_audio(self, audio_path: Path, output_dir: Optional[Path] = None) -> Dict:
        """Wrapper for separation."""
        try:
            return self.separator.separate_audio(audio_path, output_dir)
        finally:
            self.release_memory('separator')


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================

class TranscriptChunker:
    """Intelligent transcript segmentation for optimal TTS processing."""
    
    def __init__(self, 
                 min_duration: float = 1.5, 
                 max_duration: float = 15.0, 
                 merge_threshold: float = 0.8):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_threshold = merge_threshold
    
    def process(self, segments: List[Dict]) -> List[Dict]:
        """Process transcript segments into optimized chunks.
        
        Args:
            segments: List of transcript segments with start, end, text keys.
            
        Returns:
            List of optimized chunks.
        """
        if not segments:
            return []
        
        chunks = []
        current_chunk = segments[0].copy()
        
        for segment in segments[1:]:
            gap = segment['start'] - current_chunk['end']
            duration = current_chunk['end'] - current_chunk['start']
            
            # Merge if gap is small and current chunk is not too long
            if gap < self.merge_threshold and duration < self.max_duration:
                current_chunk['end'] = segment['end']
                current_chunk['text'] += ' ' + segment['text']
            else:
                chunks.append(current_chunk)
                current_chunk = segment.copy()
        
        chunks.append(current_chunk)
        return chunks


def smart_chunk(segments: List[Dict], max_dur: float = 10.0, min_gap: float = 0.5) -> List[Dict]:
    """Smart chunking logic."""
    if not segments: return []
    chunks = []
    curr = segments[0].copy()
    
    for next_seg in segments[1:]:
        gap = next_seg['start'] - curr['end']
        dur = curr['end'] - curr['start']
        
        if gap < min_gap and dur < max_dur:
            curr['end'] = next_seg['end']
            curr['text'] += " " + next_seg['text']
        else:
            chunks.append(curr)
            curr = next_seg.copy()
    
    chunks.append(curr)
    print(f"[*] Smart chunking: {len(segments)} -> {len(chunks)}")
    return chunks