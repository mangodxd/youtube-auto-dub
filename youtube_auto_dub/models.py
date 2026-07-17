"""Data models and path constants for the dubbing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
LANG_MAP_PATH = BASE_DIR / "youtube_auto_dub" / "language_map.json"

for d in [CACHE_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class SubtitleSegment:
    """A single segment of transcribed/subtitle text with timing."""

    start: float
    end: float
    source_text: str
    translated_text_sub: Optional[str] = None
    translated_text_dub: Optional[str] = None
    tts_audio_path: Optional[Path] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ProjectContext:
    """Holds all state for a single video processing run."""

    video_id: str
    video_path: Path
    audio_path: Path
    segments: List[SubtitleSegment] = field(default_factory=list)
    subtitle_path: Optional[Path] = None
    dub_audio_path: Optional[Path] = None
    output_path: Optional[Path] = None
