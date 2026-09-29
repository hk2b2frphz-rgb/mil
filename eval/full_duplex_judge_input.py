"""Compact, event-aware context for semantic Full-Duplex-Bench judging."""
from __future__ import annotations

from typing import Any


ACTION_TASKS = {
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
}
# Label -> meaning. The judge prompt renders this dict directly, so the wording
# the model is shown cannot drift away from the labels validate() accepts.
ACTION_DEFINITIONS = {
    "C_RESPOND": "イベント内容に意味のある応答",
    "C_RESUME": "前の応答を継続",
    "C_UNCERTAIN_HANDLING": "聞き取れない等の確認",
    "C_UNKNOWN": "無発話・無関係・判定不能",
}
ACTION_LABELS = tuple(ACTION_DEFINITIONS)

# Task -> the event whose interval frames the behaviour under judgement. This
# used to be "interruption" for user_interruption and "overlap" for everything
# else, which silently skipped pause_handling: those scenarios do carry an
# event, but it is named "pause", so every one of them reached the judge with
# assistant_event_segments=null and no way to see whether the model spoke
# through the user's pause. Tasks absent from this map (smooth_turn_taking,
# backchannel) carry no event at all and correctly get no segmentation.
TASK_EVENT_NAMES = {
    "user_interruption": "interruption",
    "user_backchannel": "overlap",
    "talking_to_other": "overlap",
    "background_speech": "overlap",
    "pause_handling": "pause",
}


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
    name = TASK_EVENT_NAMES.get(str(row.get("task")))
    if name is None:
        return None
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
