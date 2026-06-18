from __future__ import annotations

import math
import sys
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: only PCM16 WAV is supported, got {sample_width * 8}-bit")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sample_rate


def write_wav_mono(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    raw = (clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)


def resample_linear(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or pcm.size == 0:
        return np.asarray(pcm, dtype=np.float32)
    target_length = max(1, int(round(len(pcm) * target_rate / source_rate)))
    source_x = np.linspace(0.0, 1.0, num=len(pcm), endpoint=True)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(target_x, source_x, pcm).astype(np.float32)


def energy_vad(
    pcm: np.ndarray,
    sample_rate: int,
    frame_sec: float = 0.03,
    hop_sec: float = 0.01,
    merge_gap_sec: float = 0.25,
    min_speech_sec: float = 0.08,
) -> list[list[float]]:
    pcm = np.asarray(pcm, dtype=np.float32)
    frame = max(1, int(round(frame_sec * sample_rate)))
    hop = max(1, int(round(hop_sec * sample_rate)))
    if len(pcm) < frame:
        return []
    rms = np.asarray(
        [
            math.sqrt(float(np.mean(pcm[start : start + frame] ** 2)) + 1e-12)
            for start in range(0, len(pcm) - frame + 1, hop)
        ],
        dtype=np.float32,
    )
    peak = float(rms.max(initial=0.0))
    if peak < 1e-4:
        return []
    floor = float(np.percentile(rms, 20))
    threshold = max(0.0015, peak * 0.08, floor * 3.0)
    active = rms >= threshold

    raw: list[list[float]] = []
    start_index: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            raw.append([start_index * hop_sec, index * hop_sec + frame_sec])
            start_index = None
    if start_index is not None:
        raw.append([start_index * hop_sec, len(pcm) / sample_rate])

    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        [round(start, 4), round(end, 4)]
        for start, end in merged
        if end - start >= min_speech_sec
    ]


@lru_cache(maxsize=1)
def _load_silero_model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def silero_vad(pcm: np.ndarray, sample_rate: int) -> list[list[float]]:
    try:
        import torch
        from silero_vad import get_speech_timestamps

        model = _load_silero_model()
    except Exception as exc:
        print(
            f"[fdb-vad] Silero unavailable ({exc}); using energy VAD fallback.",
            file=sys.stderr,
        )
        return energy_vad(pcm, sample_rate)

    pcm_16k = resample_linear(np.asarray(pcm, dtype=np.float32), sample_rate, 16000)
    try:
        timestamps = get_speech_timestamps(
            torch.from_numpy(pcm_16k),
            model,
            sampling_rate=16000,
            return_seconds=True,
        )
    except Exception as exc:
        print(
            f"[fdb-vad] Silero inference failed ({exc}); using energy VAD fallback.",
            file=sys.stderr,
        )
        return energy_vad(pcm, sample_rate)
    return [
        [round(float(item["start"]), 4), round(float(item["end"]), 4)]
        for item in timestamps
    ]


def interval_overlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def total_duration(intervals: list[list[float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)
