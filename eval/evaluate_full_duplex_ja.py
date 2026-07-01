#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon

from full_duplex_audio import (
    interval_overlap,
    read_wav_mono,
    silero_vad,
    total_duration,
)

# Upstream Full-Duplex-Bench deterministic metric constants.
turn_duration_threshold = 1.0
turn_num_words_threshold = 3
window_size = 0.2
time_threshold = 3.0
epsilon = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic Full-Duplex-Bench-JA timing and behavior metrics."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--backchannel-gt",
        type=Path,
        help="Optional Japanese backchannel GT distribution JSON.",
    )
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


def chunk_interval(chunk: dict[str, Any]) -> list[float] | None:
    timestamp = chunk.get("timestamp")
    if not isinstance(timestamp, list) or len(timestamp) != 2:
        return None
    return [float(timestamp[0]), float(timestamp[1])]


def chunks_overlapping(
    chunks: list[dict[str, Any]], interval: list[float]
) -> list[dict[str, Any]]:
    return [
        chunk
        for chunk in chunks
        if (timestamp := chunk_interval(chunk)) is not None
        and interval_overlap(timestamp, interval) > 0.0
    ]


def chunks_after(
    chunks: list[dict[str, Any]], timestamp: float
) -> list[dict[str, Any]]:
    return [
        chunk
        for chunk in chunks
        if (interval := chunk_interval(chunk)) is not None and interval[0] >= timestamp
    ]


def take_over(chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0
    first = chunk_interval(chunks[0])
    last = chunk_interval(chunks[-1])
    if first is None or last is None:
        return 0
    duration = last[1] - first[0]
    if duration < turn_duration_threshold:
        return 0 if len(chunks) <= turn_num_words_threshold else 1
    return 1


def first_chunk_start(chunks: list[dict[str, Any]]) -> float | None:
    starts = [
        interval[0]
        for chunk in chunks
        if (interval := chunk_interval(chunk)) is not None
    ]
    return min(starts) if starts else None


def occurrence_distribution(
    predictions: list[list[float]], total_audio_sec: float
) -> np.ndarray:
    num_windows = max(1, int(np.ceil(total_audio_sec / window_size)))
    distribution = np.full(num_windows, epsilon, dtype=np.float64)
    for start, end in predictions:
        first_window = max(0, int(np.floor(start / window_size)))
        last_window = min(num_windows - 1, int(np.floor(end / window_size)))
        distribution[first_window : last_window + 1] += 1.0
    return distribution / distribution.sum()


def backchannel_jsd(
    predictions: list[list[float]],
    total_audio_sec: float,
    gt_distribution: list[float] | None,
) -> float | None:
    if gt_distribution is None:
        return None
    if not predictions:
        return 1.0
    prediction = occurrence_distribution(predictions, total_audio_sec)
    gt = np.asarray(gt_distribution, dtype=np.float64)
    if gt.size == 0:
        return None
    if gt.size != prediction.size:
        if gt.size == 1:
            gt = np.full(prediction.size, gt[0], dtype=np.float64)
        else:
            source_x = np.linspace(0.0, 1.0, num=gt.size)
            target_x = np.linspace(0.0, 1.0, num=prediction.size)
            gt = interp1d(
                source_x,
                gt,
                kind="linear",
                bounds_error=False,
                fill_value=(gt[0], gt[-1]),
            )(target_x)
    gt = np.maximum(gt, 0.0) + epsilon
    gt /= gt.sum()
    return float(jensenshannon(prediction, gt))


def select_gt_distribution(
    metadata: dict[str, Any], backchannel_gt: dict[str, Any] | None
) -> list[float] | None:
    if backchannel_gt is None:
        return None
    # "tts_speaker" is what build_full_duplex_ja_dataset.py always records;
    # "spk"/"speaker" are accepted for upstream-style or hand-authored metadata.
    speaker = metadata.get("spk") or metadata.get("speaker") or metadata.get("tts_speaker")
    if speaker is None:
        return None
    distribution = backchannel_gt.get(str(speaker))
    return distribution if isinstance(distribution, list) else None


GREETING_MATCH_THRESHOLD = 0.6
GREETING_WINDOW_TOLERANCE_SEC = 1.0  # late-chunk-timestamp slack


def normalize_for_greeting_match(text: str) -> str:
    return "".join(str(text).split())


def evaluate_opening_greeting(
    metadata: dict[str, Any], chunks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Checks whether the model actually said its trained fixed opening line
    (build_full_duplex_ja_dataset.py --opening-greeting) during the reserved
    lead-in, using output text chunks alone (no audio/ASR needed since we
    already have Moshi's own timestamped text stream)."""
    greeting = metadata.get("opening_greeting")
    if not isinstance(greeting, dict):
        return None
    expected = greeting.get("text")
    duration = greeting.get("duration_sec")
    if not expected or duration is None:
        return None
    window_end = float(duration) + GREETING_WINDOW_TOLERANCE_SEC
    actual = "".join(
        str(chunk.get("text", ""))
        for chunk in chunks
        if (interval := chunk_interval(chunk)) is not None and interval[0] < window_end
    )
    expected_norm = normalize_for_greeting_match(expected)
    actual_norm = normalize_for_greeting_match(actual)
    similarity = (
        difflib.SequenceMatcher(None, expected_norm, actual_norm).ratio()
        if expected_norm
        else 0.0
    )
    return {
        "expected_text": expected,
        "actual_text_in_window": actual,
        "similarity": round(similarity, 4),
        "matched": similarity >= GREETING_MATCH_THRESHOLD,
    }


def evaluate_case(
    trial_dir: Path, backchannel_gt: dict[str, Any] | None = None
) -> dict[str, Any]:
    metadata = load_json(trial_dir / "metadata.json")
    output_meta = load_json(trial_dir / "output.meta.json")
    output_text = load_json(trial_dir / "output.json")
    pcm, sample_rate = read_wav_mono(trial_dir / "output.wav")
    segments = silero_vad(pcm, sample_rate)
    task = str(metadata["task"])
    user_segments = metadata.get("user_segments") or []
    user_end = max((float(item["end_sec"]) for item in user_segments), default=0.0)
    text = str(output_text.get("text") or "")
    chunks = output_text.get("chunks") or []
    clean_text = ""
    clean_chunks: list[dict[str, Any]] = []
    clean_segments: list[list[float]] = []
    metrics: dict[str, Any] = {
        "output_speech_sec": round(total_duration(segments), 4),
        "output_speech_segments": len(segments),
        "output_text_chars": len(text),
        "empty_response": not text.strip() and not segments,
    }

    greeting = evaluate_opening_greeting(metadata, chunks)
    if greeting is not None:
        metrics["greeting_similarity"] = greeting["similarity"]
        metrics["greeting_matched"] = greeting["matched"]

    if task == "pause_handling":
        pauses = (metadata.get("events") or {}).get("pause") or []
        overlap = sum(interval_overlap(segment, pause) for segment in segments for pause in pauses)
        pause_chunks = [
            chunk
            for pause in pauses
            for chunk in chunks_overlapping(chunks, pause)
        ]
        tor = take_over(pause_chunks)
        metrics.update(
            TOR=tor,
            pause_overlap_sec=round(overlap, 4),
            takeover_during_pause=bool(tor),
        )
    elif task == "smooth_turn_taking":
        response_chunks = chunks_after(chunks, user_end)
        start = first_chunk_start(response_chunks)
        tor = take_over(response_chunks)
        metrics.update(
            TOR=tor,
            response_latency_sec=(
                round(max(0.0, start - user_end), 4)
                if tor and start is not None
                else None
            ),
        )
    elif task == "backchannel":
        tor = 0
        predictions: list[list[float]] = []
        for segment in segments:
            duration = segment[1] - segment[0]
            if duration > time_threshold:
                tor = 1
                break
            segment_chunks = chunks_overlapping(chunks, segment)
            num_words = len(segment_chunks)
            if num_words > turn_num_words_threshold:
                segment_tor = 1
            elif duration < turn_duration_threshold:
                segment_tor = 0 if num_words <= 2 else 1
            else:
                segment_tor = 1
            tor = max(tor, segment_tor)
            predictions.append(segment)
        total_audio_sec = len(pcm) / sample_rate
        gt_distribution = select_gt_distribution(metadata, backchannel_gt)
        metrics.update(
            TOR=tor,
            backchannel_frequency=round(
                len(predictions) / max(total_audio_sec, epsilon), 10
            ),
            jsd=backchannel_jsd(predictions, total_audio_sec, gt_distribution),
            backchannel_count=len(predictions),
            backchannel_prediction=predictions,
        )
    elif task == "user_interruption":
        interrupt = event_interval(metadata, "interruption")
        if interrupt:
            containing = [
                segment for segment in segments if segment[0] <= interrupt[0] < segment[1]
            ]
            response_chunks = chunks_after(chunks, interrupt[1])
            start_after = first_chunk_start(response_chunks)
            tor = take_over(response_chunks)
            metrics.update(
                TOR=tor,
                model_speaking_at_interrupt=bool(containing),
                stop_latency_sec=(
                    round(max(0.0, containing[0][1] - interrupt[0]), 4)
                    if containing
                    else 0.0
                ),
                response_latency_after_interrupt_sec=(
                    round(max(0.0, start_after - interrupt[1]), 4)
                    if tor and start_after is not None
                    else None
                ),
            )
    elif task in {"user_backchannel", "talking_to_other", "background_speech"}:
        overlap_event = event_interval(metadata, "overlap")
        if (trial_dir / "clean_output.wav").exists():
            clean_pcm, clean_sr = read_wav_mono(trial_dir / "clean_output.wav")
            clean_segments = silero_vad(clean_pcm, clean_sr)
            clean_output = load_json(trial_dir / "clean_output.json")
            clean_text = str(clean_output.get("text") or "")
            clean_chunks = clean_output.get("chunks", [])
        if overlap_event:
            noisy_after = speech_after(segments, overlap_event[1])
            clean_after = speech_after(clean_segments, overlap_event[1])
            containing = [
                segment for segment in segments if segment[0] <= overlap_event[0] < segment[1]
            ]
            metrics.update(
                stop_latency_sec=(
                    round(max(0.0, containing[0][1] - overlap_event[0]), 4)
                    if containing
                    else 0.0
                ),
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
        "assistant_chunks": chunks,
        "speech_segments": segments,
        "clean_assistant_text": clean_text or None,
        "clean_assistant_chunks": clean_chunks,
        "clean_speech_segments": clean_segments,
        "opening_greeting_check": greeting,
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
    backchannel_gt = load_json(args.backchannel_gt) if args.backchannel_gt else None
    trials = sorted(path.parent for path in run_dir.glob("**/output.meta.json"))
    rows = [evaluate_case(trial, backchannel_gt) for trial in trials]
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
