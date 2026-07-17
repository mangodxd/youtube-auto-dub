"""FFmpeg-based rendering and audio mixing utilities."""

import subprocess
from typing import List


def run_ffmpeg_cmd(cmd: List[str], timeout: int = 300, description: str = "FFmpeg operation") -> None:
    """Run an FFmpeg command with consistent error handling.

    Args:
        cmd: FFmpeg command list to execute.
        timeout: Command timeout in seconds.
        description: Description for error messages.

    Raises:
        RuntimeError: If FFmpeg command fails or times out.
    """
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{description} timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{description} failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during {description}: {e}")
