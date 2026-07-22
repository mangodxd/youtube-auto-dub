"""Audio processing pipeline: mixing, tempo alignment, loudness, background."""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from youtube_auto_dub.models import (
    AUDIO_CROSSFADE_IN_MS,
    AUDIO_CROSSFADE_OUT_MS,
    AUDIO_DEFAULT_AMBIENT_GAIN,
    AUDIO_DUB_GAIN_DB,
    AUDIO_FINAL_FADE_MS,
    AUDIO_FINAL_PAD_MS,
    AUDIO_HPSS_KERNEL,
    AUDIO_HPSS_MARGIN,
    AUDIO_LOUDNESS_CEIL,
    AUDIO_LOUDNESS_TARGET,
    AUDIO_SILENCE_FLOOR_DB,
    AUDIO_SILENCE_FRAME_MS,
    AUDIO_TRIM_BACKOFF,
    FFMPEG_AUDIO_CODEC,
    FFMPEG_MUX_VIDEO_CODEC,
    FFMPEG_VIDEO_CODEC,
    SEGMENT_GAP_THRESHOLD,
    SEGMENT_MAX_DURATION,
    SR_TTS,
    TEMPO_GAP_MS,
    TEMPO_MAX_SPEED,
    TEMPO_OVERBUDGET_RATIO,
    TEMPO_SLOWDOWN_DIVISOR,
    TEMPO_SLOWDOWN_FLOOR,
    TEMPO_TAIL_SECONDS,
    TEMPO_UNDERBUDGET_RATIO,
    SubtitleSegment,
)
from youtube_auto_dub.ui import console

log = logging.getLogger(__name__)
SR = SR_TTS


# ── Audio chunking ──────────────────────────────────────────────────────

def group_segments(raw: List[dict]) -> List[SubtitleSegment]:
    if not raw:
        return []
    out, buf = [], [raw[0]]
    for cur in raw[1:]:
        prev = buf[-1]
        gap = cur["start"] - prev["end"]
        dur = cur["end"] - buf[0]["start"]
        if gap > SEGMENT_GAP_THRESHOLD or dur > SEGMENT_MAX_DURATION:
            out.append(SubtitleSegment(
                start=buf[0]["start"],
                end=buf[-1]["end"],
                source_text=" ".join(s["text"] for s in buf).strip(),
            ))
            buf = [cur]
        else:
            buf.append(cur)
    if buf:
        out.append(SubtitleSegment(
            start=buf[0]["start"],
            end=buf[-1]["end"],
            source_text=" ".join(s["text"] for s in buf).strip(),
        ))
    console.step(f"Grouped {len(out)} segments")
    return out


# ── SRT output ──────────────────────────────────────────────────────────

def _stamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: List[SubtitleSegment], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            txt = seg.translated_text_sub or seg.source_text
            f.write(f"{i}\n{_stamp(seg.start)} --> {_stamp(seg.end)}\n{txt}\n\n")


# ── Simple overlay mixing ───────────────────────────────────────────────

def overlay_dub(source_audio: Path, segments: List[SubtitleSegment], output: Path):
    base = AudioSegment.from_file(source_audio) - abs(AUDIO_DUB_GAIN_DB)
    for seg in segments:
        if seg.tts_audio_path and seg.tts_audio_path.exists():
            clip = AudioSegment.from_file(seg.tts_audio_path)
            base = base.overlay(clip, position=int(seg.start * 1000))
    base.export(output, format="wav")


# ── Tempo alignment engine ──────────────────────────────────────────────

def _load_raw(wav_path: Path, sr: int) -> np.ndarray:
    res = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(res.stdout, dtype=np.float32)


def _stretch(wav_path: Path, factor: float, dst: Path, sr: int):
    filts = []
    r = factor
    while r > 100.0:
        filts.append("atempo=100.0")
        r /= 100.0
    while r < 0.5:
        filts.append("atempo=0.5")
        r /= 0.5
    filts.append(f"atempo={r:.6f}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-filter:a", ",".join(filts),
         "-ar", str(sr), "-ac", "1", str(dst)],
        capture_output=True, check=True,
    )
    return sf.read(dst, dtype="float32")[0]


def _crossfade(
    audio: np.ndarray,
    sr: int,
    fin_ms: int = AUDIO_CROSSFADE_IN_MS,
    fout_ms: int = AUDIO_CROSSFADE_OUT_MS,
) -> np.ndarray:
    a = audio.copy()
    n = len(a)
    fi = min(int(sr * fin_ms / 1000), n // 2)
    if fi >= 2:
        a[:fi] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fi))).astype(np.float32)
    fo = min(int(sr * fout_ms / 1000), n // 2)
    if fo >= 2:
        a[-fo:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo))).astype(np.float32)
    return a


def _find_trim_point(
    audio: np.ndarray,
    sr: int,
    budget: int,
    floor_db: float = AUDIO_SILENCE_FLOOR_DB,
) -> int:
    fl = int(sr * AUDIO_SILENCE_FRAME_MS / 1000)
    nf = len(audio) // fl
    if nf == 0:
        return budget
    eng = np.sqrt(np.mean(audio[:nf * fl].reshape(nf, fl) ** 2, axis=1))
    thresh = 10 ** (floor_db / 20.0)
    bf = min(budget // fl, len(eng))
    backoff_start = max(int(bf * AUDIO_TRIM_BACKOFF), 1)
    for i in range(bf - 1, backoff_start, -1):
        if eng[i] < thresh:
            return (i + 1) * fl
    region = eng[backoff_start:bf]
    if len(region) > 0:
        return (int(np.argmin(region)) + backoff_start + 1) * fl
    return budget


def align_segments(
    seg_info: List[dict],
    source_duration: float,
    output: Path,
    mode: str = "auto",
    max_speed: float = None,
    gap_ms: int = None,
) -> Path:
    sr = SR
    if max_speed is None:
        max_speed = TEMPO_MAX_SPEED
    if gap_ms is None:
        gap_ms = TEMPO_GAP_MS
    tail = TEMPO_TAIL_SECONDS
    total = int((source_duration + tail) * sr)
    timeline = np.zeros(total, dtype=np.float32)

    valid = sorted([s for s in seg_info if s.get("wav_path")], key=lambda s: s["start"])

    for i, s in enumerate(valid):
        raw = _load_raw(s["wav_path"], sr)
        actual = len(raw) / sr
        if actual <= 0:
            continue
        start_samp = int(s["start"] * sr)
        is_last = i + 1 >= len(valid)
        budget = max(
            (valid[i + 1]["start"] - s["start"] - gap_ms / 1000) if not is_last
            else (source_duration - s["start"]),
            s.get("target_dur", actual),
        )
        ratio = actual / budget

        if mode == "auto":
            if ratio > TEMPO_OVERBUDGET_RATIO:
                audio = _stretch(s["wav_path"], min(ratio, max_speed),
                                 Path(str(s["wav_path"]).replace(".wav", "_sped.wav")), sr)
            elif ratio < TEMPO_UNDERBUDGET_RATIO:
                target = max(ratio / TEMPO_SLOWDOWN_DIVISOR, TEMPO_SLOWDOWN_FLOOR)
                if target < 0.95:
                    audio = _stretch(s["wav_path"], target,
                                     Path(str(s["wav_path"]).replace(".wav", "_slow.wav")), sr)
                else:
                    audio = raw
            else:
                audio = raw
        else:
            audio = raw

        if is_last:
            clip = int((budget + tail) * sr)
            if len(audio) > clip:
                audio = audio[:_find_trim_point(audio, sr, clip)]
                audio = _crossfade(audio, sr, AUDIO_CROSSFADE_IN_MS, 150)
            else:
                audio = _crossfade(audio, sr, AUDIO_CROSSFADE_IN_MS, 200)
        else:
            audio = _crossfade(audio, sr)

        end = min(start_samp + len(audio), total)
        seg_len = end - start_samp
        if seg_len > 0:
            timeline[start_samp:end] += audio[:seg_len]

    orig = int(source_duration * sr)
    nz = np.nonzero(timeline)[0]
    ce = int(nz[-1]) + 1 if len(nz) else orig
    tf = min(int(sr * AUDIO_FINAL_FADE_MS / 1000), ce // 2)
    if tf >= 2:
        timeline[ce - tf:ce] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, tf))).astype(np.float32)
    pe = min(max(ce + int(sr * AUDIO_FINAL_PAD_MS / 1000), orig), total)
    timeline = timeline[:pe]
    pk = np.max(np.abs(timeline))
    if pk > 1.0:
        timeline /= pk
    sf.write(output, timeline, sr)
    return output


# ── Loudness matching ───────────────────────────────────────────────────

def _loudness(path: Path) -> float:
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    import re
    m = re.search(r'\{[^}]+"input_i"[^}]+\}', res.stderr, re.DOTALL)
    if m:
        return float(json.loads(m.group())["input_i"])
    return -24.0


def _isolate_ambient(audio_path: Path, out: Path, sr: int = SR) -> Optional[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import librosa
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        S = librosa.stft(y)
        H, P = librosa.decompose.hpss(np.abs(S), kernel_size=AUDIO_HPSS_KERNEL, margin=AUDIO_HPSS_MARGIN)
        bg = librosa.istft(S * (P / (H + P + 1e-10)), length=len(y))
        sf.write(out, bg, sr)
        return out
    except Exception as exc:
        log.warning("Background isolation failed: %s", exc)
        return None


def finalize_audio(
    dubbed: Path,
    original: Path,
    output: Path,
    match_loudness: bool = True,
    mix_ambient: bool = False,
    ambient_gain: float = AUDIO_DEFAULT_AMBIENT_GAIN,
) -> Path:
    """Apply loudness normalisation and optional background mixing."""
    if not match_loudness and not mix_ambient:
        if dubbed != output:
            shutil.copy2(dubbed, output)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    work = dubbed

    if match_loudness:
        target = max(_loudness(original), AUDIO_LOUDNESS_CEIL)
        tmp = output.with_name(output.name + ".loud.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(work),
             "-af", f"loudnorm=I={target:.1f}:{AUDIO_LOUDNESS_TARGET}",
             "-ar", str(SR), "-ac", "1", str(tmp)],
            capture_output=True, check=True,
        )
        work = tmp

    if mix_ambient:
        bg = output.with_name("ambient.wav")
        result = _isolate_ambient(original, bg, sr=SR)
        if result and result.exists():
            dur = sf.info(str(work)).duration
            tmp = output.with_name(output.name + ".amb.wav")
            filt = (
                f"[1:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,"
                f"volume={ambient_gain:.2f}[bg];"
                f"[0:a][bg]amix=inputs=2:duration=first:weights=1 {ambient_gain:.2f}[out]"
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(work), "-i", str(bg),
                 "-filter_complex", filt, "-map", "[out]",
                 "-ar", str(SR), "-ac", "1", str(tmp)],
                capture_output=True, check=True,
            )
            work = tmp

    if work != output:
        shutil.move(str(work), str(output))
    for s in (".loud.wav", ".amb.wav"):
        p = output.with_name(output.name + s)
        if p.exists() and p != output:
            p.unlink()
    return output


# ── Video muxing ────────────────────────────────────────────────────────

def mux_video(video: Path, audio: Path, output: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
         "-c:v", FFMPEG_MUX_VIDEO_CODEC, "-map", "0:v:0", "-map", "1:a:0",
         "-shortest", str(output)],
        check=True, capture_output=True,
    )


def render_video(
    video_path: Path,
    subtitle_path: Optional[Path],
    dub_audio_path: Optional[Path],
    output_path: Path,
):
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if dub_audio_path:
        cmd.extend(["-i", str(dub_audio_path)])
    vf = []
    if subtitle_path:
        sp = str(subtitle_path.resolve()).replace("\\", "/").replace(":", "\\:")
        vf.append(f"subtitles='{sp}'")
    if vf:
        cmd.extend(["-vf", ",".join(vf)])
    cmd.extend(["-c:v", FFMPEG_VIDEO_CODEC])
    if dub_audio_path:
        cmd.extend(["-c:a", FFMPEG_AUDIO_CODEC, "-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-c:a", "copy"])
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
