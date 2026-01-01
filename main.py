#!/usr/bin/env python3
"""
YouTube Auto Dub - Day 03
Added basic YouTube downloading functionality
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core_utils import setup_directories, validate_url, get_video_id
    from youtube import YouTubeDownloader
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def main():
    print("YouTube Auto Dub - Starting...")
    
    # Setup directories
    setup_directories()
    
    # TODO: Get YouTube URL from user
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Placeholder
    
    # Validate URL
    if not validate_url(youtube_url):
        print("Invalid URL!")
        return
    
    print(f"Processing URL: {youtube_url}")
    
    # Initialize YouTube downloader
    downloader = YouTubeDownloader()
    
    # Get video info
    video_info = downloader.get_video_info(youtube_url)
    if video_info:
        print(f"Video found: {video_info.get('title', 'Unknown')}")
    else:
        print("Failed to get video info!")
        return
    
    # TODO: Implement actual video download
    print("Ready to download video...")
    
    # TODO: Implement audio processing
    # TODO: Implement text translation
    # TODO: Implement voice synthesis
    
    print("YouTube Auto Dub - Finished!")

if __name__ == "__main__":
    main()
