"""Rich console wrapper for consistent CLI output."""

from rich.align import Align
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "#93c5fd",
    "success": "#86efac",
    "warning": "#fde68a",
    "error": "#fca5a5",
    "debug": "#9ca3af",
    "loading": "#93c5fd",
    "step": "#86efac",
})


class Logger:
    """Simple console logger with styled output levels."""

    def __init__(self):
        self._console = Console(highlight=False, soft_wrap=True, theme=custom_theme)

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def status(self, msg: str, **kwargs):
        return self._console.status(f"[loading]{msg}[/loading]", spinner="dots", **kwargs)

    def info(self, message: str):
        self._console.print(f"[info]\\[INFO] {message}[/info]")

    def success(self, message: str):
        self._console.print(f"[success]\\[SUCCESS] {message}[/success]")

    def warning(self, message: str):
        self._console.print(f"[warning]\\[WARNING] {message}[/warning]")

    def error(self, message: str):
        self._console.print(f"[error]\\[ERROR] {message}[/error]")

    def debug(self, message: str):
        self._console.print(f"[debug]\\[DEBUG] {message}[/debug]")

    def step(self, message: str):
        self._console.print(f"[step]  - {message}[/step]")

    def header(self, title: str, center: bool = True):
        if center:
            self._console.print(Align.center(f"[bold]{title}[/bold]"))
            self._console.print(Align.center("[#6b7280]──────────────[/#6b7280]"))
        else:
            self._console.print(f"\n[bold]{title}[/bold]")
            self._console.print("[#6b7280]──────────────[/#6b7280]")


console = Logger()
