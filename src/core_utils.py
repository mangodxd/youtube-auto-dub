"""
Core Utilities - Day 03
Basic utility functions for the project
"""

import os
import re
from pathlib import Path

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['output', 'temp', '.cache']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Directory ready: {directory}")

def validate_url(url):
    """Basic URL validation for YouTube"""
    if not url:
        return False
    
    youtube_pattern = r'(youtube\.com|youtu\.be)'
    return bool(re.search(youtube_pattern, url))

def clean_temp_files():
    """Clean temporary files - placeholder for now"""
    print("Cleaning temp files...")
    # TODO: Implement actual file cleaning

def get_video_id(url):
    """Extract video ID from YouTube URL - placeholder"""
    print(f"Extracting video ID from: {url}")
    # TODO: Implement actual video ID extraction
    return "placeholder_id"

class Config:
    """Basic configuration class"""
    
    OUTPUT_DIR = "output"
    TEMP_DIR = "temp"
    CACHE_DIR = ".cache"
    
    # Audio settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    
    # Video settings
    VIDEO_FORMAT = "mp4"
    AUDIO_FORMAT = "wav"

class YouTubeError(Exception):
    """Custom exception for YouTube-related errors"""
    pass

class AudioError(Exception):
    """Custom exception for audio processing errors"""
    pass
