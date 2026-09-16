#!/usr/bin/env python3
"""Evaluate v1.5 acoustic adaptation before/after overlap events.

This is the deterministic server-side half of the v1.5 acoustic evaluation.
It deliberately emits per-case paired values; after local semantic judging,
``summarize_full_duplex_adaptation.py`` can restrict significance tests to a
chosen action class such as ``C_RESPOND``.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from full_duplex_audio import read_wav_mono, write_wav_mono

OVERLAP_TASKS = {"user_interruption", "user_backchannel", "talking_to_other", "background_speech"}
METRICS = ("utmosv2", "wpm", "mean_pitch", "std_pitch", "mean_intensity", "std_intensity", "cutoff_count")
_UTMOS_MODEL: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--with-utmos", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def event_end(metadata: dict[str, Any]) -> float | None:
    name = "interruption" if metadata.get("task") == "user_interruption" else "overlap"
    values = (metadata.get("events") or {}).get(name) or []
    if values and len(values[0]) == 2 and isinstance(values[0][1], (int, float)):
        return float(values[0][1])
    return None


def split_pcm(pcm: np.ndarray, sample_rate: int, split_sec: float) -> tuple[np.ndarray, np.ndarray]:
    split = max(0, min(len(pcm), int(round(split_sec * sample_rate))))
    return pcm[:split], pcm[split:]


def chunks_for_window(chunks: list[dict[str, Any]], start: float, end: float | None) -> list[dict[str, Any]]:
    values = []
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if not isinstance(ts, list) or len(ts) != 2 or not all(isinstance(x, (int, float)) for x in ts):
            continue
        chunk_start, chunk_end = float(ts[0]), float(ts[1])
        if chunk_end <= start or (end is not None and chunk_start >= end):
            continue
        values.append(chunk)
    return values


def speech_duration(chunks: list[dict[str, Any]]) -> float:
    intervals = []
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if isinstance(ts, list) and len(ts) == 2 and all(isinstance(x, (int, float)) for x in ts):
            intervals.append((float(ts[0]), float(ts[1])))
    if not intervals:
        return 0.0
    return max(0.0, max(end for _, end in intervals) - min(start for start, _ in intervals))


def cutoff_count(pcm: np.ndarray, sample_rate: int) -> float:
    frame = max(1, int(sample_rate * 0.03))
    if len(pcm) < frame * 2:
        return 0.0
    rms = np.array([math.sqrt(float(np.mean(pcm[i : i + frame] ** 2)) + 1e-12) for i in range(0, len(pcm) - frame + 1, frame)])
    db = 20.0 * np.log10(rms + 1e-10)
    floor = float(np.percentile(db, 10))
    return float(sum(1 for previous, current in zip(db, db[1:], strict=False) if current - previous <= -18 and previous >= floor + 6))


def pitch_stats(pcm: np.ndarray, sample_rate: int) -> tuple[float | None, float | None]:
    if len(pcm) < sample_rate // 4:
        return None, None
    try:
        import librosa

        f0, _, _ = librosa.pyin(pcm.astype(np.float64), fmin=50, fmax=600, sr=sample_rate)
        values = f0[np.isfinite(f0)]
        if not len(values):
            return None, None
        return float(np.mean(values)), float(np.std(values))
    except Exception:
        return None, None


def intensity_stats(pcm: np.ndarray, sample_rate: int) -> tuple[float | None, float | None]:
    frame = max(1, int(sample_rate * 0.03))
    if len(pcm) < frame:
        return None, None
    values = np.array([
        20.0 * math.log10(math.sqrt(float(np.mean(pcm[i : i + frame] ** 2)) + 1e-12) + 1e-10)
        for i in range(0, len(pcm) - frame + 1, frame)
    ])
    values = values[values > -80.0]
    if not len(values):
        return None, None
    return float(np.mean(values)), float(np.std(values))


def utmos_score(pcm: np.ndarray, sample_rate: int) -> float | None:
    global _UTMOS_MODEL
    if not len(pcm):
        return None
    try:
        import utmosv2

        if _UTMOS_MODEL is None:
            _UTMOS_MODEL = utmosv2.create_model(pretrained=True, device="cuda" if _cuda_available() else "cpu")
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            write_wav_mono(Path(handle.name), pcm, sample_rate)
            return float(_UTMOS_MODEL.predict(input_path=handle.name))
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def measure(pcm: np.ndarray, sample_rate: int, chunks: list[dict[str, Any]], with_utmos: bool) -> dict[str, float | None]:
    duration = speech_duration(chunks)
    mean_pitch, std_pitch = pitch_stats(pcm, sample_rate)
    mean_intensity, std_intensity = intensity_stats(pcm, sample_rate)
    return {
        "utmosv2": utmos_score(pcm, sample_rate) if with_utmos else None,
        "wpm": len(chunks) * 60.0 / duration if duration > 0 else None,
        "mean_pitch": mean_pitch,
        "std_pitch": std_pitch,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "cutoff_count": cutoff_count(pcm, sample_rate),
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    ordered = sorted(enumerate(pvalues), key=lambda value: value[1])
    adjusted = [1.0] * count
    running = 0.0
    for rank, (index, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalue))
        adjusted[index] = running
    return adjusted


def paired_summary(rows: list[dict[str, Any]], before: str, after: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    pending: list[tuple[str, float]] = []
    for metric in METRICS:
        pairs = [
            (row[before].get(metric), row[after].get(metric))
            for row in rows
            if isinstance(row.get(before, {}).get(metric), (int, float))
            and isinstance(row.get(after, {}).get(metric), (int, float))
        ]
        if len(pairs) < 2:
            results[metric] = {"n": len(pairs), "p_value": None}
            continue
        left, right = np.asarray(pairs, dtype=float).T
        statistic, pvalue = stats.ttest_rel(right, left)
        results[metric] = {
            "n": len(pairs),
            "mean_before": float(np.mean(left)),
            "mean_after": float(np.mean(right)),
            "mean_delta": float(np.mean(right - left)),
            "t_stat": float(statistic),
            "p_value": float(pvalue),
        }
        pending.append((metric, float(pvalue)))
    for (metric, _), adjusted in zip(pending, holm_adjust([value for _, value in pending]), strict=True):
        results[metric]["p_value_holm"] = adjusted
    return results


def main() -> int:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for trial in sorted(path.parent for path in args.run_dir.glob("**/output.meta.json")):
        metadata = load_json(trial / "metadata.json")
        if metadata.get("task") not in OVERLAP_TASKS or not (trial / "clean_output.wav").exists():
            continue
        split = event_end(metadata)
        if split is None:
            continue
        pcm, sample_rate = read_wav_mono(trial / "output.wav")
        clean_pcm, clean_rate = read_wav_mono(trial / "clean_output.wav")
        output = load_json(trial / "output.json")
        clean_output = load_json(trial / "clean_output.json")
        pre_pcm, post_pcm = split_pcm(pcm, sample_rate, split)
        pre_chunks = chunks_for_window(output.get("chunks", []), 0.0, split)
        post_chunks = chunks_for_window(output.get("chunks", []), split, None)
        records.append({
            "case_id": metadata["id"],
            "task": metadata["task"],
            "seed": load_json(trial / "output.meta.json")["seed"],
            "trial_dir": str(trial),
            "pre": measure(pre_pcm, sample_rate, pre_chunks, args.with_utmos),
            "post": measure(post_pcm, sample_rate, post_chunks, args.with_utmos),
            "clean": measure(clean_pcm, clean_rate, clean_output.get("chunks", []), args.with_utmos),
        })
    result = {
        "n": len(records),
        "with_utmos": args.with_utmos,
        "per_case": records,
        "paired_tests": {
            "pre_vs_post": paired_summary(records, "pre", "post"),
            "clean_vs_post": paired_summary(records, "clean", "post"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fdb-adaptation] evaluated {len(records)} overlap trials -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
