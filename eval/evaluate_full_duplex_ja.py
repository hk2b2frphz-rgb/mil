#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from full_duplex_audio import (
    energy_vad,
    interval_overlap,
    read_wav_mono,
    total_duration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic Full-Duplex-Bench-JA timing and behavior metrics."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def first_start_after(segments: list[list[float]], timestamp: float) -> float | None:
    starts = [start for start, _ in segments if start >= timestamp]
    return min(starts) if starts else None


def speech_after(segments: list[list[float]], timestamp: float) -> float:
    return sum(max(0.0, end - max(start, timestamp)) for start, end in segments if end > timestamp)


def event_interval(metadata: dict[str, Any], name: str) -> list[float] | None:
    intervals = (metadata.get("events") or {}).get(name) or []
    return intervals[0] if intervals else None


def evaluate_case(trial_dir: Path) -> dict[str, Any]:
    metadata = load_json(trial_dir / "metadata.json")
    output_meta = load_json(trial_dir / "output.meta.json")
    output_text = load_json(trial_dir / "output.json")
    pcm, sample_rate = read_wav_mono(trial_dir / "output.wav")
    segments = energy_vad(pcm, sample_rate)
    task = str(metadata["task"])
    user_segments = metadata.get("user_segments") or []
    user_end = max((float(item["end_sec"]) for item in user_segments), default=0.0)
    text = str(output_text.get("text") or "")
    clean_text = ""
    clean_chunks: list[dict[str, Any]] = []
    clean_segments: list[list[float]] = []
    metrics: dict[str, Any] = {
        "output_speech_sec": round(total_duration(segments), 4),
        "output_speech_segments": len(segments),
        "output_text_chars": len(text),
        "empty_response": not text.strip() and not segments,
    }

    if task == "pause_handling":
        pauses = (metadata.get("events") or {}).get("pause") or []
        overlap = sum(interval_overlap(segment, pause) for segment in segments for pause in pauses)
        metrics.update(
            pause_overlap_sec=round(overlap, 4),
            takeover_during_pause=overlap >= 1.0,
        )
    elif task == "smooth_turn_taking":
        start = first_start_after(segments, user_end)
        metrics.update(
            take_turn=speech_after(segments, user_end) >= 1.0 or len(text.strip()) >= 4,
            response_latency_sec=round(max(0.0, start - user_end), 4) if start is not None else None,
        )
    elif task == "backchannel":
        during_user = [
            [max(start, 0.0), min(end, user_end)]
            for start, end in segments
            if min(end, user_end) > max(start, 0.0)
        ]
        short = [segment for segment in during_user if segment[1] - segment[0] <= 1.2]
        long = [segment for segment in during_user if segment[1] - segment[0] >= 3.0]
        metrics.update(
            backchannel_count=len(short),
            backchannel_frequency_per_min=round(len(short) / max(user_end / 60.0, 1e-6), 4),
            takeover_during_user=bool(long),
        )
    elif task == "user_interruption":
        interrupt = event_interval(metadata, "interruption")
        if interrupt:
            containing = [
                segment for segment in segments if segment[0] <= interrupt[0] < segment[1]
            ]
            start_after = first_start_after(segments, interrupt[1])
            metrics.update(
                model_speaking_at_interrupt=bool(containing),
                stop_latency_sec=(
                    round(max(0.0, containing[0][1] - interrupt[0]), 4)
                    if containing
                    else 0.0
                ),
                response_latency_after_interrupt_sec=(
                    round(max(0.0, start_after - interrupt[1]), 4)
                    if start_after is not None
                    else None
                ),
                take_turn_after_interrupt=speech_after(segments, interrupt[1]) >= 1.0,
            )
    elif task in {"user_backchannel", "talking_to_other", "background_speech"}:
        overlap_event = event_interval(metadata, "overlap")
        if (trial_dir / "clean_output.wav").exists():
            clean_pcm, clean_sr = read_wav_mono(trial_dir / "clean_output.wav")
            clean_segments = energy_vad(clean_pcm, clean_sr)
            clean_output = load_json(trial_dir / "clean_output.json")
            clean_text = str(clean_output.get("text") or "")
            clean_chunks = clean_output.get("chunks", [])
        if overlap_event:
            noisy_after = speech_after(segments, overlap_event[1])
            clean_after = speech_after(clean_segments, overlap_event[1])
            metrics.update(
                overlap_response_sec=round(
                    sum(interval_overlap(segment, overlap_event) for segment in segments), 4
                ),
                post_overlap_speech_sec=round(noisy_after, 4),
                clean_post_overlap_speech_sec=round(clean_after, 4),
                post_overlap_speech_delta_sec=round(noisy_after - clean_after, 4),
                clean_output_text_chars=len(clean_text),
            )

    return {
        "model_id": output_meta["model_id"],
        "case_id": metadata["id"],
        "task": task,
        "category": metadata.get("category"),
        "risk_level": metadata.get("risk_level"),
        "seed": output_meta["seed"],
        "language": "ja",
        "expected_behavior": metadata.get("expected_behavior"),
        "avoid_behavior": metadata.get("avoid_behavior", []),
        "user_segments": user_segments,
        "events": metadata.get("events", {}),
        "assistant_text": text,
        "assistant_chunks": output_text.get("chunks", []),
        "speech_segments": segments,
        "clean_assistant_text": clean_text or None,
        "clean_assistant_chunks": clean_chunks,
        "clean_speech_segments": clean_segments,
        "metrics": metrics,
        "trial_dir": str(trial_dir),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    summary: dict[str, Any] = {}
    for task, task_rows in sorted(by_task.items()):
        numeric: dict[str, list[float]] = {}
        boolean: dict[str, list[bool]] = {}
        for row in task_rows:
            for key, value in row["metrics"].items():
                if isinstance(value, bool):
                    boolean.setdefault(key, []).append(value)
                elif isinstance(value, (int, float)):
                    numeric.setdefault(key, []).append(float(value))
        summary[task] = {
            "n": len(task_rows),
            "means": {
                key: statistics.fmean(values)
                for key, values in sorted(numeric.items())
                if values
            },
            "rates": {
                key: sum(values) / len(values)
                for key, values in sorted(boolean.items())
                if values
            },
        }
    return summary


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = sorted(path.parent for path in run_dir.glob("**/output.meta.json"))
    rows = [evaluate_case(trial) for trial in trials]
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    run_config = load_json(run_dir / "run_config.json")
    result = {
        "benchmark": "Full-Duplex-Bench v1/v1.5 Japanese adaptation",
        "upstream_commit": run_config.get("upstream_full_duplex_bench_commit"),
        "model_id": run_config.get("model_id"),
        "language": "ja",
        "n": len(rows),
        "tasks": aggregate(rows),
        "note": "API-dependent semantic/behavior judging is deferred to local Azure OpenAI.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[fdb-eval] evaluated {len(rows)} trials -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
