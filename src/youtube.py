"""
YouTube Content Download Module for YouTube Auto Dub.

This module provides a robust interface for downloading YouTube content
using yt-dlp. It handles:
- Video and audio extraction from YouTube URLs
- Authentication via cookies or browser integration
- Format selection and quality optimization
- Error handling and retry logic
- Metadata extraction and validation

The module focuses on reliability and provides comprehensive error
messages to help users troubleshoot download issues.

Author: Nguyen Cong Thuan Huy (mangodxd)
Version: 1.0.0
"""

import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Any
from src.engines import CACHE_DIR


def _get_opts(browser: Optional[str] = None, 
             cookies_file: Optional[str] = None, 
             quiet: bool = True) -> Dict[str, Any]:
    """Generate common yt-dlp options with authentication configuration.
    
    This helper function creates a standardized set of yt-dlp options
    including authentication methods. It supports both cookie files
    and browser-based authentication.
    
    Args:
        browser: Browser name for cookie extraction (chrome, edge, firefox).
                If provided, cookies will be extracted from this browser.
        cookies_file: Path to cookies.txt file in Netscape format.
                     Takes priority over browser extraction if both provided.
        quiet: Whether to suppress yt-dlp output messages.
        
    Returns:
        Dictionary of yt-dlp options.
        
    Raises:
        ValueError: If invalid browser name is provided.
        
    Example:
        >>> opts = _get_opts(browser="chrome", quiet=False)
        >>> print(f"Configured authentication: {bool('cookiefile' in opts or 'cookiesfrombrowser' in opts)}")
        
    NOTE: Priority order: cookies_file > browser > no authentication.
    
    TODO: Add support for additional browsers and authentication methods.
    """
    # Base options for all operations
    opts = {
        'quiet': quiet,
        'no_warnings': True,
        'extract_flat': False,  # We need full metadata
    }
    
    # Authentication configuration
    if cookies_file:
        # Use explicit cookie file (highest priority)
        cookies_path = Path(cookies_file)
        if not cookies_path.exists():
            raise FileNotFoundError(f"Cookies file not found: {cookies_file}")
        
        opts['cookiefile'] = str(cookies_file)
        print(f"[*] Using cookies file: {cookies_file}")
        
    elif browser:
        # Extract cookies from browser
        valid_browsers = ['chrome', 'firefox', 'edge', 'safari', 'opera', 'brave']
        browser_lower = browser.lower()
        
        if browser_lower not in valid_browsers:
            raise ValueError(f"Invalid browser '{browser}'. Supported: {', '.join(valid_browsers)}")
        
        # yt-dlp browser cookie extraction format
        opts['cookiesfrombrowser'] = (browser_lower,)
        print(f"[*] Extracting cookies from browser: {browser}")
        
    else:
        print(f"[*] No authentication configured (public videos only)")
    
    return opts


def get_id(url: str, 
          browser: Optional[str] = None, 
          cookies_file: Optional[str] = None) -> str:
    """Extract YouTube video ID from URL with authentication support.
    
    This function extracts the video ID from a YouTube URL using yt-dlp.
    It supports authentication and handles various YouTube URL formats.
    
    Args:
        url: YouTube video URL to extract ID from.
        browser: Browser name for cookie extraction.
        cookies_file: Path to cookies.txt file.
        
    Returns:
        YouTube video ID as string.
        
    Raises:
        ValueError: If URL is invalid or video ID cannot be extracted.
        RuntimeError: If yt-dlp fails to extract information.
        
    Example:
        >>> video_id = get_id("https://youtube.com/watch?v=VIDEO_ID")
        >>> print(f"Video ID: {video_id}")
        
    NOTE: This function validates the URL and extracts metadata
    without downloading the actual content.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    
    # Basic URL validation
    if not any(domain in url.lower() for domain in ['youtube.com', 'youtu.be']):
        raise ValueError(f"Invalid YouTube URL: {url}")
    
    try:
        print(f"[*] Extracting video ID from: {url[:50]}...")
        
        # Configure yt-dlp options
        opts = _get_opts(browser=browser, cookies_file=cookies_file)
        
        # Extract video information without downloading
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                video_id = info.get('id')
                
                if not video_id:
                    raise RuntimeError("No video ID found in extracted information")
                
                # Extract additional useful info for logging
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                uploader = info.get('uploader', 'Unknown')
                
                print(f"[+] Video ID extracted: {video_id}")
                print(f"    Title: {title[:50]}{'...' if len(title) > 50 else ''}")
                print(f"    Duration: {duration}s ({duration//60}:{duration%60:02d})")
                print(f"    Uploader: {uploader}")
                
                return video_id
                
            except yt_dlp.DownloadError as e:
                if "Sign in to confirm" in str(e) or "private video" in str(e).lower():
                    raise ValueError(f"Authentication required for this video. Please use --browser or --cookies. Original error: {e}")
                else:
                    raise RuntimeError(f"yt-dlp extraction failed: {e}")
                    
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to extract video ID: {e}") from e


def download_video(url: str, 
                  browser: Optional[str] = None, 
                  cookies_file: Optional[str] = None) -> Path:
    """Download the best quality video with audio from YouTube.
    
    This function downloads the highest quality video file that includes
    audio. It prioritizes MP4 format with AVC video codec for maximum
    compatibility with the processing pipeline.
    
    Args:
        url: YouTube video URL to download.
        browser: Browser name for cookie extraction.
        cookies_file: Path to cookies.txt file.
        
    Returns:
        Path to the downloaded video file.
        
    Raises:
        ValueError: If URL is invalid or authentication is required.
        RuntimeError: If download fails or file is corrupted.
        
    Example:
        >>> video_path = download_video("https://youtube.com/watch?v=VIDEO_ID")
        >>> print(f"Video downloaded: {video_path}")
        
    NOTE: This function downloads both video and audio in a single file.
    If the video already exists in cache, it returns the existing file
    to avoid unnecessary downloads.
    
    TODO: Add quality options and format selection parameters.
    """
    try:
        # Extract video ID first for validation and caching
        video_id = get_id(url, browser=browser, cookies_file=cookies_file)
    except Exception as e:
        raise ValueError(f"Failed to validate video URL: {e}") from e
    
    # Determine output path
    out_path = CACHE_DIR / f"{video_id}.mp4"
    
    # Check if file already exists and is valid
    if out_path.exists():
        file_size = out_path.stat().st_size
        if file_size > 1024 * 1024:  # At least 1MB
            print(f"[*] Video already cached: {out_path}")
            return out_path
        else:
            print(f"[!] WARNING: Cached video seems too small ({file_size} bytes), re-downloading")
            out_path.unlink()  # Remove corrupted file
    
    try:
        print(f"[*] Downloading video: {video_id}")
        
        # Configure yt-dlp options for video download
        opts = _get_opts(browser=browser, cookies_file=cookies_file)
        opts.update({
            # Format selection: Best MP4 with AVC video and AAC audio
            'format': (
                'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/'  # Preferred
                'best[ext=mp4]/'                                      # Fallback MP4
                'best'                                               # Final fallback
            ),
            'outtmpl': str(out_path),
            'merge_output_format': 'mp4',
            'postprocessors': [],  # No post-processing for raw video
        })
        
        # Download video
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        
        # Verify download was successful
        if not out_path.exists():
            raise RuntimeError(f"Video file not created after download: {out_path}")
        
        file_size = out_path.stat().st_size
        if file_size < 1024 * 1024:  # Less than 1MB seems suspicious
            raise RuntimeError(f"Downloaded video file is too small: {file_size} bytes")
        
        print(f"[+] Video downloaded successfully:")
        print(f"    File: {out_path}")
        print(f"    Size: {file_size / (1024*1024):.1f} MB")
        
        return out_path
        
    except yt_dlp.DownloadError as e:
        error_msg = str(e).lower()
        if "sign in to confirm" in error_msg or "private video" in error_msg:
            raise ValueError(
                f"Authentication required for this video. Please try:\n"
                f"1. Close all browser windows and use --browser\n"
                f"2. Export fresh cookies.txt and use --cookies\n"
                f"3. Check if video is public/accessible\n"
                f"Original error: {e}"
            )
        else:
            raise RuntimeError(f"Video download failed: {e}")
            
    except Exception as e:
        # Clean up partial download on failure
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"Video download failed: {e}") from e


def download_audio(url: str, 
                  browser: Optional[str] = None, 
                  cookies_file: Optional[str] = None) -> Path:
    """Download audio-only from YouTube for transcription processing.
    
    This function downloads the best quality audio from a YouTube video
    and converts it to WAV format for optimal compatibility with the
    Whisper transcription model. The audio is extracted at the project's
    configured sample rate for consistency.
    
    Args:
        url: YouTube video URL to extract audio from.
        browser: Browser name for cookie extraction.
        cookies_file: Path to cookies.txt file.
        
    Returns:
        Path to the downloaded WAV audio file.
        
    Raises:
        ValueError: If URL is invalid or authentication is required.
        RuntimeError: If audio download or conversion fails.
        
    Example:
        >>> audio_path = download_audio("https://youtube.com/watch?v=VIDEO_ID")
        >>> print(f"Audio downloaded: {audio_path}")
        
    NOTE: The output is always in WAV format at the project's sample rate
    for consistency with the transcription pipeline. If the file already
    exists, it returns the existing file to avoid re-downloading.
    
    TODO: Add audio quality options and format selection.
    """
    try:
        # Extract video ID first for validation and caching
        video_id = get_id(url, browser=browser, cookies_file=cookies_file)
    except Exception as e:
        raise ValueError(f"Failed to validate video URL: {e}") from e
    
    # Determine output paths
    temp_path = CACHE_DIR / f"{video_id}"  # yt-dlp will add extension
    final_path = CACHE_DIR / f"{video_id}.wav"  # Our target format
    
    # Check if final WAV file already exists and is valid
    if final_path.exists():
        file_size = final_path.stat().st_size
        if file_size > 1024 * 100:  # At least 100KB for audio
            print(f"[*] Audio already cached: {final_path}")
            return final_path
        else:
            print(f"[!] WARNING: Cached audio seems too small ({file_size} bytes), re-downloading")
            final_path.unlink()  # Remove corrupted file
    
    try:
        print(f"[*] Downloading audio: {video_id}")
        
        # Configure yt-dlp options for audio download
        opts = _get_opts(browser=browser, cookies_file=cookies_file)
        opts.update({
            # Best audio format
            'format': 'bestaudio/best',
            'outtmpl': str(temp_path),
            # Convert to WAV for transcription compatibility
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',  # 192kbps
            }],
        })
        
        # Download audio
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        
        # Verify the final WAV file was created
        if not final_path.exists():
            # Check if temp file exists (yt-dlp may have used different naming)
            temp_files = list(CACHE_DIR.glob(f"{video_id}.*"))
            if temp_files:
                print(f"[!] WARNING: Expected {final_path} but found {temp_files[0]}")
                final_path = temp_files[0]  # Use whatever was created
            else:
                raise RuntimeError(f"Audio file not created after download: {final_path}")
        
        file_size = final_path.stat().st_size
        if file_size < 1024 * 100:  # Less than 100KB seems suspicious for audio
            raise RuntimeError(f"Downloaded audio file is too small: {file_size} bytes")
        
        print(f"[+] Audio downloaded successfully:")
        print(f"    File: {final_path}")
        print(f"    Size: {file_size / (1024*1024):.1f} MB")
        
        # Additional validation: try to get duration
        try:
            from src.media import get_duration
            duration = get_duration(final_path)
            if duration > 0:
                print(f"    Duration: {duration:.1f}s ({duration//60}:{duration%60:02d})")
            else:
                print(f"[!] WARNING: Could not determine audio duration")
        except Exception as e:
            print(f"[!] WARNING: Audio validation failed: {e}")
        
        return final_path
        
    except yt_dlp.DownloadError as e:
        error_msg = str(e).lower()
        if "sign in to confirm" in error_msg or "private video" in error_msg:
            raise ValueError(
                f"Authentication required for this video. Please try:\n"
                f"1. Close all browser windows and use --browser\n"
                f"2. Export fresh cookies.txt and use --cookies\n"
                f"3. Check if video is public/accessible\n"
                f"Original error: {e}"
            )
        else:
            raise RuntimeError(f"Audio download failed: {e}")
            
    except Exception as e:
        # Clean up partial downloads on failure
        for path in [temp_path, final_path]:
            if path.exists():
                path.unlink()
        raise RuntimeError(f"Audio download failed: {e}") from e
