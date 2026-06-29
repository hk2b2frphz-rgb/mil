"""Extract interactivity event segments from synthetic stereo training data.

Reads the manifest JSONL + sidecar timing JSON produced by the TTS pipeline,
runs Silero VAD on each channel, and emits per-axis segment files used as
GRPO prompts.

Segment types (matching Kyutai paper Section 3.2):
  pause       — ≥1s silence in user channel followed by speech
  turn_taking — speaker change (user → moshi)
  backchannel — short (≤1s) moshi speech during user speech
  interruption — user speech onset during moshi speech
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 24_000  # Moshi native


def _load_audio_channel(wav_path: Path, channel: int) -> np.ndarray:
    """Load one channel from a stereo WAV as float32 numpy array."""
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz: {wav_path}")
    if data.shape[1] < 2:
        raise ValueError(f"Expected stereo, got {data.shape[1]} channels: {wav_path}")
    return data[:, channel]


def _run_vad(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """Return speech segments as (start_sec, end_sec) via Silero VAD."""
    import torch

    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    tensor = torch.from_numpy(audio)
    stamps = get_speech_timestamps(tensor, model, sampling_rate=sr)
    return [(s["start"] / sr, s["end"] / sr) for s in stamps]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def extract_segments(
    wav_path: Path,
    sidecar: dict[str, Any] | None = None,
    min_prompt_sec: float = 3.0,
) -> dict[str, list[dict[str, Any]]]:
    """Extract event segments from one stereo WAV.

    Returns dict with keys: pause, turn_taking, backchannel, interruption.
    Each value is a list of dicts with at minimum:
      wav_path, start_sec, end_sec, axis, channel metadata.
    """
    # left = moshi (ch0), right = user (ch1)
    moshi_audio = _load_audio_channel(wav_path, 0)
    user_audio = _load_audio_channel(wav_path, 1)

    moshi_segs = _run_vad(moshi_audio)
    user_segs = _run_vad(user_audio)

    total_dur = len(moshi_audio) / SAMPLE_RATE
    results: dict[str, list[dict[str, Any]]] = {
        "pause": [],
        "turn_taking": [],
        "backchannel": [],
        "interruption": [],
    }

    # ── Pause: silence ≥1s in user channel, followed by user speech ──
    prev_end = 0.0
    for seg_start, seg_end in user_segs:
        gap = seg_start - prev_end
        if gap >= 1.0 and prev_end >= min_prompt_sec:
            results["pause"].append({
                "wav_path": str(wav_path),
                "start_sec": max(0, prev_end - min_prompt_sec),
                "end_sec": min(seg_end, total_dur),
                "pause_start": prev_end,
                "pause_end": seg_start,
                "axis": "pause",
            })
        prev_end = seg_end

    # ── Turn-taking: user speech end → moshi speech start ──
    for u_start, u_end in user_segs:
        if u_end - u_start < min_prompt_sec:
            continue
        for m_start, m_end in moshi_segs:
            if m_start >= u_end and m_start - u_end < 5.0:
                results["turn_taking"].append({
                    "wav_path": str(wav_path),
                    "start_sec": max(0, u_end - min_prompt_sec),
                    "end_sec": min(m_end, total_dur),
                    "user_turn_end": u_end,
                    "axis": "turn_taking",
                })
                break

    # ── Backchannel: short moshi speech during user speech ──
    for m_start, m_end in moshi_segs:
        if m_end - m_start > 1.0:
            continue
        for u_start, u_end in user_segs:
            if _overlap((m_start, m_end), (u_start, u_end)) > 0:
                ctx_start = max(0, u_start)
                results["backchannel"].append({
                    "wav_path": str(wav_path),
                    "start_sec": ctx_start,
                    "end_sec": min(u_end, total_dur),
                    "bc_time": m_start,
                    "axis": "backchannel",
                })
                break

    # ── Interruption: user speech onset during moshi speech ──
    for u_start, u_end in user_segs:
        for m_start, m_end in moshi_segs:
            if m_start < u_start < m_end and m_end - m_start > 1.0:
                results["interruption"].append({
                    "wav_path": str(wav_path),
                    "start_sec": max(0, m_start),
                    "end_sec": min(u_end + 2.0, total_dur),
                    "interruption_start": u_start,
                    "axis": "interruption",
                })
                break

    return results


def extract_from_manifest(
    manifest_path: Path,
    out_dir: Path,
    max_per_axis: int = 1000,
) -> dict[str, int]:
    """Walk a training manifest JSONL → extract segments → write per-axis JSONL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"pause": 0, "turn_taking": 0, "backchannel": 0, "interruption": 0}
    buffers: dict[str, list[dict[str, Any]]] = {k: [] for k in counts}

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            wav = Path(row.get("audio_path", row.get("path", "")))
            if not wav.is_file():
                wav = manifest_path.parent / wav
            if not wav.is_file():
                continue
            try:
                segs = extract_segments(wav)
            except Exception as exc:
                print(f"WARN: skipping {wav}: {exc}")
                continue
            for axis, items in segs.items():
                if counts[axis] < max_per_axis:
                    for item in items:
                        if counts[axis] >= max_per_axis:
                            break
                        buffers[axis].append(item)
                        counts[axis] += 1

    for axis, items in buffers.items():
        out_path = out_dir / f"{axis}_segments.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{axis}: {len(items)} segments -> {out_path}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-per-axis", type=int, default=1000)
    args = parser.parse_args()
    extract_from_manifest(args.manifest, args.out_dir, args.max_per_axis)


if __name__ == "__main__":
    main()
