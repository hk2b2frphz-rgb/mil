#!/usr/bin/env python3
"""Full-Duplex-Bench v1.5 get_timing.py-compatible timing artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_duplex_audio import read_wav_mono, silero_vad

USER_MERGE_GAP = 0.6
MODEL_MERGE_GAP = 0.5


def merge(segments: list[list[float]], gap: float) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(segments):
        if result and start - result[-1][1] <= gap:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def overlaps(user: list[list[float]], model: list[list[float]]) -> list[list[float]]:
    values: dict[int, list[float]] = {}
    for user_start, user_end in user:
        for model_start, model_end in model:
            start, end = max(user_start, model_start), min(user_end, model_end)
            if end <= start:
                continue
            key = round(end * 1000)
            candidate = [round(start, 3), round(end, 3)]
            if key not in values or candidate[1] - candidate[0] < values[key][1] - values[key][0]:
                values[key] = candidate
    return [values[key] for key in sorted(values)]


def response_gaps(user: list[list[float]], model: list[list[float]]) -> list[list[float]]:
    values: dict[int, list[float]] = {}
    starts = [start for start, _ in model]
    for _, user_end in user:
        next_start = next((start for start in starts if start > user_end), None)
        if next_start is None:
            continue
        key = round(next_start * 1000)
        candidate = [round(user_end, 3), round(next_start, 3)]
        if key not in values or candidate[0] > values[key][0]:
            values[key] = candidate
    return [values[key] for key in sorted(values, key=lambda item: values[item][1])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    trials = sorted(path.parent for path in args.run_dir.glob("**/output.meta.json"))
    for trial in trials:
        input_pcm, input_sr = read_wav_mono(trial / "input.wav")
        output_pcm, output_sr = read_wav_mono(trial / "output.wav")
        user = merge(silero_vad(input_pcm, input_sr), USER_MERGE_GAP)
        model = merge(silero_vad(output_pcm, output_sr), MODEL_MERGE_GAP)
        artifact = {
            "latency_stop_list": overlaps(user, model),
            "latency_resp_list": response_gaps(user, model),
        }
        (trial / "latency_intervals.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[fdb-timing] wrote {len(trials)} latency_intervals.json files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
