#!/usr/bin/env python3
"""Shared opening-greeting check, used both at inference time
(run_full_duplex_bench.py, to log a per-trial result while the job is
still running) and at evaluation time (evaluate_full_duplex_ja.py, to
record it in per_case.jsonl/summary.json)."""
from __future__ import annotations

import difflib
from typing import Any

GREETING_MATCH_THRESHOLD = 0.6
GREETING_WINDOW_TOLERANCE_SEC = 1.0  # late-chunk-timestamp slack


def chunk_interval(chunk: dict[str, Any]) -> list[float] | None:
    timestamp = chunk.get("timestamp")
    if not isinstance(timestamp, list) or len(timestamp) != 2:
        return None
    return [float(timestamp[0]), float(timestamp[1])]


def normalize_for_greeting_match(text: str) -> str:
    return "".join(str(text).split())


def evaluate_opening_greeting(
    metadata: dict[str, Any], chunks: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Checks whether the model actually said its trained fixed opening line
    (build_full_duplex_ja_dataset.py --opening-greeting) during the reserved
    lead-in, using output text chunks alone (no audio/ASR needed since we
    already have Moshi's own timestamped text stream). Returns None when the
    case has no opening_greeting metadata (e.g. FDB_OPENING_GREETING=0
    runs)."""
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
