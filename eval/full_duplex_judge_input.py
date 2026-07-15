"""Compact, event-aware context for semantic Full-Duplex-Bench judging."""
from __future__ import annotations

from typing import Any


ACTION_TASKS = {
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
}
ACTION_LABELS = (
    "C_RESPOND",
    "C_RESUME",
    "C_UNCERTAIN_HANDLING",
    "C_UNKNOWN",
)


def chunk_interval(chunk: dict[str, Any]) -> tuple[float, float] | None:
    value = chunk.get("timestamp")
    if not isinstance(value, list) or len(value) != 2:
        return None
    start, end = value
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return float(start), float(end)


def event_interval(row: dict[str, Any]) -> tuple[float, float] | None:
    events = row.get("event_timeline") or row.get("events") or {}
    name = "interruption" if row.get("task") == "user_interruption" else "overlap"
    values = events.get(name) or []
    if not values or not isinstance(values[0], list) or len(values[0]) != 2:
        return None
    start, end = values[0]
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return float(start), float(end)


def compact_event_segments(
    chunks: list[dict[str, Any]] | None,
    interval: tuple[float, float] | None,
) -> dict[str, str] | None:
    """Split timestamped ASR chunks into before/during/after event text.

    A chunk straddling an event is retained in ``during``. This keeps the
    semantic judge focused on the first genuinely post-event content without
    exporting the full high-volume timestamp timeline.
    """
    if interval is None:
        return None
    start, end = interval
    buckets = {"before": [], "during": [], "after": []}
    for chunk in chunks or []:
        text = str(chunk.get("text") or "").strip()
        timestamp = chunk_interval(chunk)
        if not text or timestamp is None:
            continue
        chunk_start, chunk_end = timestamp
        if chunk_end <= start:
            buckets["before"].append(text)
        elif chunk_start >= end:
            buckets["after"].append(text)
        else:
            buckets["during"].append(text)
    return {key: " ".join(value).strip() for key, value in buckets.items()}


def action_required(task: str | None) -> bool:
    return task in ACTION_TASKS
