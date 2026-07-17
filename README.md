# youtube-auto-dub

[![CI](https://github.com/mangodxd/youtube-auto-dub/actions/workflows/ci.yml/badge.svg)](https://github.com/mangodxd/youtube-auto-dub/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python pipeline that takes a YouTube URL and spits out a dubbed/subtitled video. Feed it a link, pick a target language, and it handles the rest — downloading, transcribing, translating, synthesizing speech, and rendering the final video.

We built this because existing tools were either too manual, too expensive, or too locked-in. This one runs locally, stays free, and gives you full control over the output.

---

## How it works

```
YouTube URL → Download → Transcribe (Whisper) → Chunk → Translate → TTS → Mix → Render
```

1. **Download** — pulls video and audio via `yt-dlp`
2. **Transcribe** — runs Whisper ASR to get timestamped text
3. **Chunk** — splits transcript into natural speech segments (respects silence gaps, max 10s)
4. **Translate** — sends segments to Google Translate (RPC method, falls back to scraping)
5. **TTS** — synthesizes each segment using Edge TTS with the appropriate voice
6. **Mix** — overlays TTS audio on top of the original, ducked by 12dB
7. **Render** — burns subtitles and/or swaps the audio track via FFmpeg

---

## Getting started

### Prerequisites

- Python 3.10+
- FFmpeg in your PATH
- (Optional) CUDA for faster Whisper inference

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows — grab a build from https://ffmpeg.org/download.html
```

### Install

```bash
pip install youtube-auto-dub
```

Or install from source:

```bash
git clone https://github.com/mangodxd/youtube-auto-dub.git
cd youtube-auto-dub
pip install .
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

```bash
# Dub + subtitles in Vietnamese (default)
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID"

# Or via python -m
python -m youtube_auto_dub "https://youtube.com/watch?v=VIDEO_ID"

# Or legacy entry point (still works)
python main.py "https://youtube.com/watch?v=VIDEO_ID"

# Just subtitles, in Spanish
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" --mode sub --lang es

# Dubbing only, French, female voice
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" --mode dub --lang fr --gender female

# Different languages for subs and dub
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" --mode both --lang_sub en --lang_dub vi

# Age-restricted or private video — pull cookies from your browser
youtube-auto-dub "https://youtube.com/watch?v=VIDEO_ID" --lang es --browser chrome
```

### All options

| Flag | Short | Description |
|------|-------|-------------|
| `url` | | YouTube URL (required) |
| `--mode` | `-m` | `sub`, `dub`, or `both` (default: `both`) |
| `--lang` | `-l` | Target language for both sub and dub |
| `--lang_sub` | `-ls` | Override subtitle language |
| `--lang_dub` | `-ld` | Override dubbing language |
| `--gender` | `-g` | `male` or `female` voice (default: `female`) |
| `--whisper_model` | `-wm` | `tiny`, `base`, `small`, `medium` |
| `--browser` | `-b` | Cookie source: `chrome`, `edge`, `firefox` |

---

## Language & voice config

Voices are mapped in `language_map.json` inside the package. Edit it to change defaults or add new languages:

```json
{
  "es": {
    "name": "Spanish",
    "voices": {
      "female": ["es-ES-ElviraNeural"],
      "male": ["es-ES-JorgeNeural"]
    }
  }
}
```

Common codes: `es` · `fr` · `de` · `it` · `pt` · `ja` · `ko` · `zh` · `ar` · `hi` · `ru` · `vi` · `th`

---

## Project structure

```
youtube-auto-dub/
├── pyproject.toml           # Package config, dependencies, CLI entry
├── main.py                 # Legacy entry point (thin wrapper)
├── language_map.json       # (root copy kept for reference)
└── youtube_auto_dub/       # Installable Python package
    ├── __init__.py         # Package version
    ├── __main__.py         # python -m youtube_auto_dub support
    ├── cli.py              # Thin CLI adapter (argparse → pipeline)
    ├── pipeline.py         # Pipeline orchestration logic
    ├── models.py           # Dataclasses (SubtitleSegment, ProjectContext)
    ├── youtube.py          # yt-dlp wrapper, audio extraction
    ├── media.py            # Chunking, SRT, mixing, FFmpeg render
    ├── googlev4.py         # Google Translate (RPC + scrape fallback)
    ├── tts.py              # Edge TTS synthesis with retry
    ├── transcriber.py      # Whisper transcription + device manager
    ├── renderer.py         # FFmpeg helper utilities
    ├── segmenter.py        # Smart segmentation (dynamic chunking)
    ├── config.py           # Config manager, language data loading
    ├── exceptions.py       # Custom exception hierarchy
    ├── utils.py            # General-purpose utilities
    ├── engine.py           # Legacy AI Engine (preserved for BC)
    └── language_map.json   # Language → voice mappings (package data)
├── tests/
│   ├── test_models.py      # Model dataclass tests
│   ├── test_googlev4.py    # Translation shortcut tests (no network)
│   └── test_tts.py         # Voice lookup tests
.tests/
├── .github/workflows/
│   └── ci.yml              # Ruff lint + pytest on 3.10, 3.11, 3.12
├── .cache/                 # Downloaded videos (persists between runs)
├── output/                 # Final rendered videos
└── temp/                   # Intermediate files (cleared each run)
```

---

## Known issues & workarounds

**FFmpeg not found**
Add FFmpeg to your system PATH. Run `ffmpeg -version` to verify.

**CUDA out of memory**
Switch to a smaller model: `--whisper_model tiny` or `--whisper_model base`. The pipeline auto-selects `base` on CPU and `small` on GPU if you don't specify.

**Translation splitting wrong**
The batch translator joins segments with a delimiter and splits on it after. If the translated text collapses the delimiter, it falls back to translating each segment individually. Slower but accurate.

**YouTube rate-limited or auth errors**
Close your browser fully before running with `--browser chrome`. If that still fails, export a `cookies.txt` file via a browser extension and pass it with `yt-dlp`'s `--cookies` option directly (you'd need to patch `youtube.py` for now — PRs welcome).

**TTS file too small / silent audio**
Usually a bad language code or a voice that's region-restricted. Double check your `language_map.json` entry and try the other gender.

---

## What we're working on

- **Speaker diarization** — using `pyannote.audio` to detect multiple speakers and assign them distinct voices
- **Background music separation** — `Demucs` integration to preserve BGM while replacing only the vocals
- **Voice conversion (RVC)** — optional post-processing to match the original speaker's voice characteristics

On the backlog:
- Batch mode for playlists/channels
- Local LLM translation (Llama 3 / Mistral) for offline/private use
- Web UI via Gradio or Streamlit
- Better lip-sync timing (stretch/compress TTS clips to fit original segment duration)

---

## Contributing

Issues and PRs are open. If you're adding a new language to `language_map.json`, please include both a male and female voice where Edge TTS supports it.

```bash
# Dev install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .
```

---

## License

MIT. See [LICENSE](LICENSE).

---

*Built by Nguyen Cong Thuan Huy ([@mangodxd](https://github.com/mangodxd))*
