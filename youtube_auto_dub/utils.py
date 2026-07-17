"""General-purpose utility functions and classes."""

from pathlib import Path


def safe_file_delete(file_path: Path) -> None:
    """Safely delete a file with error handling.

    Args:
        file_path: Path to file to delete.
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"[!] WARNING: Could not delete file {file_path}: {e}")


class ProgressTracker:
    """Simple progress tracking for long operations."""

    def __init__(self, total: int, description: str = "Processing", update_interval: int = 10):
        """Initialize progress tracker.

        Args:
            total: Total number of items to process.
            description: Description for progress messages.
            update_interval: How often to update progress (every N items).
        """
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.current = 0

    def update(self, increment: int = 1) -> None:
        """Update progress counter and print progress percentage."""
        self.current += increment

        if self.current % self.update_interval == 0 or self.current >= self.total:
            progress = (self.current / self.total) * 100
            print(f"[-] {self.description}: {self.current}/{self.total} ({progress:.1f}%)", end="\r")

            if self.current >= self.total:
                print()
