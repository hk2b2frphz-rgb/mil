"""Turn a GRPO rollout into a Full-Duplex-Bench-JA packed row.

The Azure judge does not read audio: eval/pack_full_duplex_azure.py hands it a
row of timed user utterances, an event timeline, the assistant transcript split
around the event, and deterministic timing metrics. A GRPO rollout carries the
same information in a different shape (segment metadata + generated text events
+ VAD on the generated audio), so this module reshapes it. Producing the packed
row -- rather than a new prompt -- is what lets the reward reuse the evaluation
rubric verbatim.

Every timestamp here is relative to the start of the audio fed to Moshi, which
includes the sampled context prefix. Segment metadata is absolute within the
source WAV, so it is shifted by ``-segment_start_sec + context_offset_sec``.
"""
from __future__ import annotations

from typing import Any

from scripts.grpo.judge_prompt import compact_event_segments, event_interval

# The four GRPO axes correspond to Full-Duplex-Bench-JA tasks. The names matter:
# action_required(), TASK_EVENT_NAMES and the judge's validate() all key off
# them, so an axis mapped to the wrong task silently changes what is scored.
AXIS_TO_TASK = {
    "pause": "pause_handling",
    "turn_taking": "smooth_turn_taking",
    "backchannel": "backchannel",
    "interruption": "user_interruption",
}

# Gap between two text pieces above which they belong to separate utterances.
# Matches the window used by rewards._detect_backchannels so the assistant is
# chunked the same way the deterministic backchannel reward sees it.
CHUNK_GAP_SEC = 0.8


def group_text_events(
    text_events: list[dict[str, Any]],
    gap_sec: float = CHUNK_GAP_SEC,
) -> list[dict[str, Any]]:
    """Group per-frame text pieces into timestamped chunks.

    Produces the same ``{"text", "timestamp": [start, end]}`` shape that the
    evaluation runner's ASR chunks have, so compact_event_segments() can split
    them into before/during/after the event without special-casing.
    """
    chunks: list[dict[str, Any]] = []
    text = ""
    start: float | None = None
    last = 0.0

    def flush() -> None:
        nonlocal text, start
        if text.strip() and start is not None:
            chunks.append({"text": text, "timestamp": [start, last]})
        text = ""
        start = None

    for event in sorted(text_events, key=lambda e: float(e.get("time_sec", 0.0))):
        piece = str(event.get("piece", ""))
        t = float(event.get("time_sec", 0.0))
        if not piece.strip():
            flush()
            last = t
            continue
        if start is None:
            start = t
        elif t - last > gap_sec:
            flush()
            start = t
        text += piece
        last = t
    flush()
    return chunks


def _shift(value: float, segment: dict[str, Any], ctx_offset: float) -> float:
    return round(float(value) - float(segment["segment_start_sec"]) + ctx_offset, 4)


def _interval_overlap(a: tuple[float, float], b: list[float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def build_event_timeline(
    segment: dict[str, Any],
    ctx_offset: float,
) -> dict[str, list[list[float]]]:
    """Event intervals in rollout time, under the names the judge expects."""
    axis = segment["axis"]
    if axis == "pause":
        return {
            "pause": [
                [
                    _shift(p["start"], segment, ctx_offset),
                    _shift(p["end"], segment, ctx_offset),
                ]
                for p in segment.get("internal_pauses", [])
            ]
        }
    if axis == "interruption":
        start = _shift(segment["interruption_start_sec"], segment, ctx_offset)
        end = _shift(
            segment.get("interruption_end_sec", segment["interruption_start_sec"]),
            segment,
            ctx_offset,
        )
        # A zero-width interval would put every assistant chunk in "after" and
        # hide the overlap the judge is asked about, so give the event the
        # minimum width the before/during/after split needs.
        if end <= start:
            end = round(start + 1.0, 4)
        return {"interruption": [[start, end]]}
    # smooth_turn_taking and backchannel carry no event, and TASK_EVENT_NAMES
    # correctly gives them no segmentation.
    return {}


def build_user_timeline(
    segment: dict[str, Any],
    ctx_offset: float,
) -> list[dict[str, Any]]:
    """The user side as timed utterances.

    Segment extraction keeps one user utterance per segment plus a single
    sidecar transcript for it, so this is one entry. For the pause axis the
    utterance is split at the annotated internal pauses: without the split the
    judge reads one continuous line and cannot see the gap it is scoring.
    """
    seg_start = ctx_offset
    seg_end = _shift(segment["segment_end_sec"], segment, ctx_offset)
    text = str(segment.get("user_text") or "").strip()

    pauses = segment.get("internal_pauses") or []
    if segment["axis"] != "pause" or not pauses:
        return [{
            "start_sec": round(seg_start, 4),
            "end_sec": seg_end,
            "text": text,
            "kind": "speech",
        }]

    # Text is only available for the utterance as a whole, so it rides on the
    # first fragment; later fragments are left untranscribed rather than
    # repeating it, which would read as the user saying the same thing twice.
    bounds = [seg_start]
    for pause in pauses:
        bounds.append(_shift(pause["start"], segment, ctx_offset))
        bounds.append(_shift(pause["end"], segment, ctx_offset))
    bounds.append(seg_end)

    fragments: list[dict[str, Any]] = []
    for index in range(0, len(bounds) - 1, 2):
        start, end = bounds[index], bounds[index + 1]
        if end <= start:
            continue
        fragments.append({
            "start_sec": round(start, 4),
            "end_sec": round(end, 4),
            "text": text if not fragments else "",
            "kind": "speech",
        })
    return fragments or [{
        "start_sec": round(seg_start, 4),
        "end_sec": seg_end,
        "text": text,
        "kind": "speech",
    }]


def build_metrics(
    segment: dict[str, Any],
    ctx_offset: float,
    vad_segments: list[tuple[float, float]],
    chunks: list[dict[str, Any]],
    transcript: str,
) -> dict[str, Any]:
    """Deterministic timing metrics, under evaluate_full_duplex_ja.py's names.

    The judge prompt calls these "reference information", and the axes that
    depend on timing (interruption_handling, backchannel_naturalness) are
    unanswerable from text alone -- the judge sees no audio, so anything it
    knows about when the model spoke, it knows from here.
    """
    speech = list(vad_segments)
    metrics: dict[str, Any] = {
        "output_speech_sec": round(sum(e - s for s, e in speech), 4),
        "output_speech_segments": len(speech),
        "output_text_chars": len(transcript),
        "empty_response": not transcript.strip() and not speech,
    }
    axis = segment["axis"]
    events = build_event_timeline(segment, ctx_offset)

    if axis == "pause":
        pauses = events.get("pause", [])
        overlap = sum(
            _interval_overlap(seg, pause) for seg in speech for pause in pauses
        )
        metrics.update(
            pause_overlap_sec=round(overlap, 4),
            takeover_during_pause=bool(overlap > 0.0),
        )
    elif axis == "turn_taking":
        user_end = _shift(segment["user_turn_end_sec"], segment, ctx_offset)
        onset = next((s for s, _ in sorted(speech) if s >= user_end), None)
        metrics.update(
            response_latency_sec=(
                round(max(0.0, onset - user_end), 4) if onset is not None else None
            ),
            TOR=1 if onset is not None else 0,
        )
    elif axis == "backchannel":
        gt = segment.get("gt_backchannel_times") or []
        metrics.update(
            gt_backchannel_count=len(gt),
            predicted_chunk_count=len(chunks),
        )
    elif axis == "interruption":
        start = _shift(segment["interruption_start_sec"], segment, ctx_offset)
        containing = [seg for seg in speech if seg[0] <= start < seg[1]]
        after = next((s for s, _ in sorted(speech) if s >= start), None)
        metrics.update(
            model_speaking_at_interrupt=bool(containing),
            stop_latency_sec=(
                round(max(0.0, containing[0][1] - start), 4) if containing else 0.0
            ),
            response_latency_after_interrupt_sec=(
                round(max(0.0, after - start), 4) if after is not None else None
            ),
        )
    return metrics


def build_packed_row(
    rollout: dict[str, Any],
    segment: dict[str, Any],
    vad_segments: list[tuple[float, float]],
    row_id: str,
) -> dict[str, Any]:
    """One rollout as a row the Azure judge prompt can be applied to unchanged."""
    ctx_offset = float(rollout.get("context_offset_sec", 0.0))
    task = AXIS_TO_TASK[segment["axis"]]
    transcript = str(rollout.get("transcript", ""))
    chunks = group_text_events(rollout.get("text_events") or [])

    row: dict[str, Any] = {
        "id": row_id,
        "case_id": "{axis}:{wav}@{start}".format(
            axis=segment["axis"],
            wav=segment.get("wav_path", ""),
            start=segment.get("segment_start_sec"),
        ),
        "task": task,
        "category": "grpo_segment",
        "risk_level": segment.get("risk_level"),
        "expected_behavior": segment.get("expected_behavior"),
        "avoid_behavior": segment.get("avoid_behavior"),
        "user_timeline": build_user_timeline(segment, ctx_offset),
        "event_timeline": build_event_timeline(segment, ctx_offset),
        "assistant_text": transcript,
        "deterministic_metrics": build_metrics(
            segment, ctx_offset, vad_segments, chunks, transcript,
        ),
    }
    row["assistant_event_segments"] = compact_event_segments(
        chunks, event_interval(row),
    )
    return row
