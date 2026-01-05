"""
Speaker Diarization Module for YouTube Auto Dub.

This module provides advanced speaker identification and segmentation
capabilities using Pyannote.audio. It enables multi-speaker dubbing
by identifying "who spoke when" in the original audio.

Features:
- Speaker diarization with timestamp mapping
- Speaker counting and identification
- Voice assignment for multiple speakers
- Integration with Edge TTS voice mapping

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
"""

import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Local imports
from src.engines import DeviceManager, ConfigManager, PipelineComponent


class SpeakerDiarizer(PipelineComponent):
    """Speaker diarization using Pyannote.audio."""
    
    def __init__(self, device_manager: Optional[DeviceManager] = None, device: Optional[str] = None, hf_token: Optional[str] = None):
        # Handle backward compatibility
        if device_manager is None:
            device_manager = DeviceManager(device)
        
        config_manager = ConfigManager()
        super().__init__(device_manager, config_manager)
        
        self.hf_token = hf_token
        self._model = None
        self._is_loaded = False
        
        print(f"[*] Speaker Diarizer initialized")
        
        if not hf_token:
            print("[!] WARNING: No Hugging Face token provided")
            print("    Set HF_TOKEN environment variable or pass hf_token parameter")
            print("    Get token: https://huggingface.co/settings/tokens")
            print("    Request access: https://huggingface.co/pyannote/speaker-diarization-3.1")
    
    @property
    def model_name(self) -> str:
        return "Pyannote Speaker Diarization"
        
    @property
    def pipeline(self):
        """Lazy-loaded Pyannote diarization pipeline."""
        if not self._is_loaded:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Load Pyannote model."""
        if not self.hf_token:
            raise RuntimeError(
                "Hugging Face token required for Pyannote.audio. "
                "Set HF_TOKEN environment variable or pass hf_token parameter."
            )
        
        print(f"[*] Loading Pyannote diarization pipeline on {self.device}...")
        
        try:
            from pyannote.audio import Pipeline
            
            # Initialize the diarization pipeline
            self._model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            # Move to target device if not CPU
            if self.device != "cpu":
                self._model.to(self.device)
            
            self._is_loaded = True
            print(f"[+] Pyannote pipeline loaded successfully")
            
            # Log memory usage if on GPU
            memory_info = self.device_manager.get_memory_info()
            if memory_info['allocated'] > 0:
                print(f"    VRAM used: {memory_info['allocated']:.2f} GB")
                
        except ImportError as e:
            raise RuntimeError(f"Failed to import pyannote.audio: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Pyannote pipeline: {e}")
    
    def _unload_model(self) -> None:
        """Unload Pyannote model."""
        if self._model:
            del self._model
            self._model = None
            self._is_loaded = False
    
    def release_memory(self) -> None:
        """Release VRAM and clean up GPU memory."""
        self._unload_model()
        self.device_manager.clear_cache()
        print(f"[*] Speaker Diarizer VRAM cleared")

    def diarize_audio(self, audio_path: Path, min_speakers: int = 1, max_speakers: int = 8) -> Dict:
        """Perform speaker diarization on audio file."""
        self._validate_file_exists(audio_path, "Audio file")
        
        try:
            print(f"[*] Diarizing audio: {audio_path.name}")
            print(f"    Speaker range: {min_speakers}-{max_speakers}")
            
            # Run diarization
            diarization = self.pipeline(
                str(audio_path),
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Process results
            speakers = set()
            segments = []
            speaker_stats = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                
                segment = {
                    'speaker': speaker,
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'duration': float(turn.end - turn.start)
                }
                segments.append(segment)
                
                # Update speaker statistics
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = 0.0
                speaker_stats[speaker] += segment['duration']
            
            # Sort segments by start time
            segments.sort(key=lambda x: x['start'])
            
            # Convert speakers to sorted list
            speaker_list = sorted(list(speakers))
            
            # Calculate total duration
            total_duration = max(seg['end'] for seg in segments) if segments else 0.0
            
            result = {
                'speakers': speaker_list,
                'segments': segments,
                'speaker_stats': speaker_stats,
                'total_duration': total_duration,
                'num_speakers': len(speaker_list)
            }
            
            print(f"[+] Diarization complete:")
            print(f"    Speakers detected: {len(speaker_list)}")
            print(f"    Total segments: {len(segments)}")
            print(f"    Duration: {total_duration:.2f}s")
            
            # Print speaker statistics
            for speaker in speaker_list:
                speaking_time = speaker_stats[speaker]
                percentage = (speaking_time / total_duration * 100) if total_duration > 0 else 0
                print(f"    {speaker}: {speaking_time:.2f}s ({percentage:.1f}%)")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Speaker diarization failed: {e}") from e

    def assign_voices_to_speakers(
        self, 
        diarization_result: Dict, 
        target_lang: str,
        voice_assignments: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Assigns voices using round-robin rotation from voice pools."""
        
        speakers = diarization_result.get('speakers', [])
        if not speakers:
            return {}

        # 1. Get Voice Pools
        lang_config = self.config_manager.get_language_config(target_lang)
        voices_cfg = lang_config.get('voices', {})
        
        # Fallback to default if list is empty or missing
        male_pool = self.config_manager.get_voice_pool(target_lang, 'male')
        female_pool = self.config_manager.get_voice_pool(target_lang, 'female')

        assignments = {}
        
        # 2. State trackers for rotation
        male_idx = 0
        female_idx = 0
        use_female_next = True  # Start with female for better clarity usually

        print(f"[*] Assigning voices for {len(speakers)} speakers...")

        for speaker in speakers:
            # A. Manual Override
            if voice_assignments and speaker in voice_assignments:
                assignments[speaker] = voice_assignments[speaker]
                print(f"    {speaker} -> {assignments[speaker]} (Manual)")
                continue

            # B. Auto Assignment (Round Robin)
            if use_female_next:
                voice = female_pool[female_idx % len(female_pool)]
                female_idx += 1
            else:
                voice = male_pool[male_idx % len(male_pool)]
                male_idx += 1
            
            assignments[speaker] = voice
            print(f"    {speaker} -> {voice} (Auto)")
            
            # Toggle gender for next speaker to maximize contrast
            use_female_next = not use_female_next
        
        return assignments

    def merge_transcript_with_speakers(
        self, 
        transcript_segments: List[Dict], 
        diarization_result: Dict
    ) -> List[Dict]:
        """Merge transcript segments with speaker information.
        
        This method aligns transcript segments with speaker diarization
        results to assign speakers to each text segment.
        
        Args:
            transcript_segments: List of transcript segments with 'start', 'end', 'text'.
            diarization_result: Result from diarize_audio() method.
                       
        Returns:
            List of segments with added 'speaker' key.
            
        Example:
            >>> merged = diarizer.merge_transcript_with_speakers(transcript, diarization)
            >>> print(f"Merged {len(merged)} segments with speaker info")
        """
        diar_segments = diarization_result['segments']
        merged_segments = []
        
        for trans_seg in transcript_segments:
            trans_start = trans_seg['start']
            trans_end = trans_seg['end']
            
            # Find the speaker for this transcript segment
            speaker = "UNKNOWN"
            
            for diar_seg in diar_segments:
                # Check for overlap between transcript and diarization segments
                if (trans_start >= diar_seg['start'] and trans_start < diar_seg['end']) or \
                   (trans_end > diar_seg['start'] and trans_end <= diar_seg['end']) or \
                   (trans_start <= diar_seg['start'] and trans_end >= diar_seg['end']):
                    speaker = diar_seg['speaker']
                    break
            
            # Create merged segment
            merged_seg = trans_seg.copy()
            merged_seg['speaker'] = speaker
            merged_segments.append(merged_seg)
        
        print(f"[*] Merged {len(transcript_segments)} transcript segments with speakers")
        
        # Count speakers in merged segments
        speaker_counts = {}
        for seg in merged_segments:
            speaker = seg['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        for speaker, count in speaker_counts.items():
            print(f"    {speaker}: {count} segments")
        
        return merged_segments

    
    def save_diarization_result(self, result: Dict, output_path: Path) -> None:
        """Save diarization result to JSON file.
        
        Args:
            result: Diarization result dictionary.
            output_path: Path to save the JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"[+] Diarization result saved: {output_path}")
            
        except Exception as e:
            print(f"[!] WARNING: Failed to save diarization result: {e}")

    def load_diarization_result(self, input_path: Path) -> Dict:
        """Load diarization result from JSON file.
        
        Args:
            input_path: Path to the JSON file.
            
        Returns:
            Diarization result dictionary.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Diarization file not found: {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print(f"[+] Diarization result loaded: {input_path}")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization result: {e}") from e


def get_huggingface_token() -> Optional[str]:
    """Get Hugging Face token from environment variable.
    
    Returns:
        Hugging Face token if available, None otherwise.
    """
    import os
    return os.getenv('HF_TOKEN')


def validate_voice_assignments(voice_assignments: Dict[str, str], target_lang: str) -> bool:
    """Validate voice assignments against language configuration.
    
    Args:
        voice_assignments: Dictionary mapping speakers to voice names.
        target_lang: Target language code.
        
    Returns:
        True if all assignments are valid, False otherwise.
    """
    config_manager = ConfigManager()
    lang_config = config_manager.get_language_config(target_lang)
    available_voices = set(lang_config.get('voices', {}).values())
    available_voices.add('en-US-AriaNeural')  # Default fallback
    
    for speaker, voice in voice_assignments.items():
        if voice not in available_voices:
            print(f"[!] WARNING: Voice '{voice}' not in available voices for '{target_lang}'")
            return False
    
    return True
