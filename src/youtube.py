"""
YouTube Module - Day 03
Basic YouTube video downloading functionality
"""

import yt_dlp
from core_utils import YouTubeError, Config

class YouTubeDownloader:
    """Basic YouTube video downloader"""
    
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def get_video_info(self, url):
        """Get basic video information"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', 'Unknown'),
                }
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None
    
    def download_audio(self, url, output_path=None):
        """Download audio from YouTube video - placeholder"""
        print(f"Downloading audio from: {url}")
        print(f"Output path: {output_path or Config.TEMP_DIR}")
        
        # TODO: Implement actual audio download
        return f"{Config.TEMP_DIR}/audio.mp3"
    
    def download_video(self, url, output_path=None):
        """Download video from YouTube - placeholder"""
        print(f"Downloading video from: {url}")
        print(f"Output path: {output_path or Config.TEMP_DIR}")
        
        # TODO: Implement actual video download
        return f"{Config.TEMP_DIR}/video.mp4"
    
    def get_available_formats(self, url):
        """Get available formats for the video - placeholder"""
        print(f"Getting available formats for: {url}")
        
        # TODO: Implement actual format listing
        return ['mp4', 'webm', 'audio_only']
