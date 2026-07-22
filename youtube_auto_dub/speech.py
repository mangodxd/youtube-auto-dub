"""Speech-to-text with Whisper — VAD, prompt conditioning, resegmentation."""

import re
from pathlib import Path
from typing import Optional

import torch

from youtube_auto_dub.models import (
    SR_WHISPER,
    VAD_GUARD_SECONDS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
    WHISPER_BATCH,
    WHISPER_BEAM,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_DEFAULT_MODEL,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_TEMPERATURES,
    VideoMetadata,
    pick_whisper_compute_type,
)
from youtube_auto_dub.subs import refine_segments


def build_hint(meta: Optional[VideoMetadata]) -> Optional[str]:
    if not meta or not meta.title:
        return None
    parts = [meta.title]
    if meta.description:
        para = meta.description.split("\n\n")[0].strip()
        parts.append(para[:200])
    if meta.tags:
        parts.append(", ".join(meta.tags[:15]))
    prompt = ". ".join(parts)
    return prompt[:800] if len(prompt) > 800 else prompt


# ── Text cleanup ────────────────────────────────────────────────────────

_PHANTOM = re.compile(r"ترجمة\s+نانسي\s+قنقر")
_REP_CHAR = re.compile(r"(.)\1{2,}")
_REP_WORD = re.compile(r"\b(\S+)(?:\s+\1){2,}\b")


def _scrub(text: str) -> str:
    text = _PHANTOM.sub("", text)
    text = _REP_CHAR.sub(r"\1", text)
    text = _REP_WORD.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ── Whisper model cache ─────────────────────────────────────────────────

_MODEL_CACHE = {}


def transcribe(
    audio: Path,
    model_name: str = WHISPER_DEFAULT_MODEL,
    device: str = "cpu",
    language: Optional[str] = None,
    hint: Optional[str] = None,
    use_vad: bool = True,
    beam: int = None,
    batch: int = None,
):
    if beam is None:
        beam = WHISPER_BEAM
    if batch is None:
        batch = WHISPER_BATCH

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    ct = pick_whisper_compute_type(device)
    key = f"{model_name}|{device}|{ct}"

    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=ct)
        except ValueError:
            ct = "int8" if device == "cpu" else "int8_float16"
            key = f"{model_name}|{device}|{ct}"
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=ct)
    model = _MODEL_CACHE[key]

    # Normalise to 16kHz mono WAV
    wav = audio
    if not (audio.suffix == ".wav" and _is_16k_mono(audio)):
        wav = audio.with_name(audio.stem + "_16k.wav")
        if not wav.exists():
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", str(SR_WHISPER),
                 "-sample_fmt", "s16", "-c:a", "pcm_s16le", str(wav)],
                check=True, capture_output=True,
            )

    # VAD
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sr = model.feature_extractor.sampling_rate
    samples = decode_audio(str(wav), sampling_rate=sr)
    total_samples = samples.shape[0]

    if use_vad:
        clips = get_speech_timestamps(samples, VadOptions(
            max_speech_duration_s=model.feature_extractor.chunk_length,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
            threshold=VAD_THRESHOLD,
        ))
        guard = int(VAD_GUARD_SECONDS * sr)
        if clips:
            if clips[0]["start"] > guard:
                clips.insert(0, {"start": 0, "end": clips[0]["start"]})
            if total_samples - clips[-1]["end"] > guard:
                clips.append({"start": clips[-1]["end"], "end": total_samples})
        else:
            clips = [{"start": 0, "end": total_samples}]
        clip_sec = [{"start": c["start"] / sr, "end": c["end"] / sr} for c in clips]
    else:
        clip_sec = [{"start": 0, "end": total_samples / sr}]

    pipe = BatchedInferencePipeline(model=model)
    kw = dict(
        batch_size=batch, language=language, beam_size=beam,
        word_timestamps=True, clip_timestamps=clip_sec,
        condition_on_previous_text=True,
        temperature=WHISPER_TEMPERATURES,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )
    if hint:
        kw["initial_prompt"] = hint

    segs_gen, info = pipe.transcribe(samples, **kw)
    detected = info.language or None

    raw = []
    for seg in segs_gen:
        d = {"start": seg.start, "end": seg.end, "text": _scrub(seg.text.strip())}
        if seg.words:
            d["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
        raw.append(d)

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    return refine_segments(raw), detected


def _is_16k_mono(path: Path) -> bool:
    try:
        import subprocess
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        parts = res.stdout.strip().split(",")
        return len(parts) == 2 and parts[0].strip() == str(SR_WHISPER) and parts[1].strip() == "1"
    except Exception:
        return False
