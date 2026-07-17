#!/usr/bin/env python3
"""Legacy entry point — delegates to ``youtube_auto_dub.cli:main``.

Kept for backward compatibility. Prefer::

    python -m youtube_auto_dub
    youtube-auto-dub
"""

from youtube_auto_dub.cli import main

if __name__ == "__main__":
    main()
