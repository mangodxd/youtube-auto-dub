"""Smart speech segmentation with dynamic parameters."""

from typing import Dict, List

import numpy as np


def smart_chunk_dynamic(segments: List[Dict]) -> List[Dict]:
    """Advanced smart chunking with dynamic parameters based on content.

    Analyzes segment durations and gaps to compute optimal chunk boundaries.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.

    Returns:
        List of merged chunk dicts with 'start', 'end', 'text' keys.
    """
    n = len(segments)
    if n == 0:
        return []

    durations = [s["end"] - s["start"] for s in segments]
    gaps = [segments[i]["start"] - segments[i - 1]["end"] for i in range(1, n)]

    avg_seg_dur = sum(durations) / n
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.5

    min_dur = max(1.0, avg_seg_dur * 0.5)
    max_dur = np.percentile(durations, 90) if n > 5 else min(15.0, avg_seg_dur * 3)
    max_dur = max(5.0, min(30.0, max_dur))

    gap_threshold = max(0.4, avg_gap * 1.5)

    chunks = []
    curr_chunk_segs = [segments[0]]

    for i in range(1, n):
        curr = segments[i]
        gap = curr["start"] - segments[i - 1]["end"]
        current_dur = curr["end"] - curr_chunk_segs[0]["start"]

        if gap > gap_threshold or current_dur > max_dur:
            chunks.append({
                "start": curr_chunk_segs[0]["start"],
                "end": curr_chunk_segs[-1]["end"],
                "text": " ".join(s["text"] for s in curr_chunk_segs).strip(),
            })
            curr_chunk_segs = [curr]
        else:
            curr_chunk_segs.append(curr)

    if curr_chunk_segs:
        chunks.append({
            "start": curr_chunk_segs[0]["start"],
            "end": curr_chunk_segs[-1]["end"],
            "text": " ".join(s["text"] for s in curr_chunk_segs).strip(),
        })

    print(
        f"[+] Smart chunking: {len(chunks)} chunks "
        f"(Dynamic: min={min_dur:.1f}s, max={max_dur:.1f}s, gap_thr={gap_threshold:.2f}s)"
    )
    return chunks
