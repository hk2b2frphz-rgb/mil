"""Deterministic reward functions for GRPO axes 1-4.

Axis 1: Pause Handling — binary -1/0
Axis 2: Turn-Taking — R = -d (response delay)
Axis 3: Backchannel — F1 (±1s window)
Axis 4: User Interruption — R = -d (stop delay)

All functions operate on VAD segments and text event lists produced by
Moshi inference, matching the Kyutai paper definitions exactly.
"""
from __future__ import annotations

import re
from typing import Any

# Japanese aizuchi patterns (from analyze_backchannels.py).
AIZUCHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("un", re.compile(r"う+ん+")),
    ("hai", re.compile(r"はい(?:はい)?")),
    ("ee", re.compile(r"え+ぇ*|ええ")),
    ("sou", re.compile(r"そう(?:そう|です(?:ね|よね)?|なん(?:です(?:ね)?|だ)?)?|そっか|そうか")),
    ("naruhodo", re.compile(r"なるほど(?:ね)?")),
    ("hee", re.compile(r"へ[ぇえー]+|ほ[ーお]+")),
    ("fun", re.compile(r"ふ[んー]+|ふんふん")),
    ("aa", re.compile(r"あ[ぁあーっ]+")),
    ("wakaru", re.compile(r"わかり(?:ます|ました)|わかる")),
    ("tashika", re.compile(r"たしかに|確かに")),
    ("oo", re.compile(r"お[おーっ]+")),
    ("warai", re.compile(r"(?:あは|うふ|えへ|ふふ)+|笑")),
]


# ── Axis 1: Pause Handling ──────────────────────────────────────────

def reward_pause_handling(
    moshi_vad_segments: list[tuple[float, float]],
    pause_start: float,
    pause_end: float,
    threshold_sec: float = 1.0,
) -> float:
    """Return -1 if Moshi produced a speech segment ≥ threshold during pause."""
    for seg_start, seg_end in moshi_vad_segments:
        overlap_start = max(seg_start, pause_start)
        overlap_end = min(seg_end, pause_end)
        if overlap_end - overlap_start >= threshold_sec:
            return -1.0
    return 0.0


# ── Axis 2: Turn-Taking ────────────────────────────────────────────

def reward_turn_taking(
    moshi_vad_segments: list[tuple[float, float]],
    user_turn_end: float,
) -> float:
    """R = -d where d = first Moshi speech onset after user finishes."""
    for seg_start, _ in sorted(moshi_vad_segments):
        if seg_start >= user_turn_end:
            return -(seg_start - user_turn_end)
    return -10.0  # no response at all


# ── Axis 3: Backchannel ────────────────────────────────────────────

def _detect_backchannels(
    text_events: list[dict[str, Any]],
    max_dur_sec: float = 1.0,
) -> list[float]:
    """Return timestamps of detected backchannel onsets in text events."""
    events = sorted(text_events, key=lambda e: e.get("time_sec", 0.0))
    hits: list[float] = []
    window_text = ""
    window_start: float | None = None
    last_t = 0.0

    def flush() -> None:
        nonlocal window_text, window_start
        if not window_text.strip() or window_start is None:
            window_text = ""
            window_start = None
            return
        dur = last_t - window_start
        if dur <= max_dur_sec:
            for _, pat in AIZUCHI_PATTERNS:
                if pat.search(window_text):
                    hits.append(window_start)
                    break
        window_text = ""
        window_start = None

    for ev in events:
        piece = str(ev.get("piece", ""))
        t = float(ev.get("time_sec", 0.0))
        if not piece.strip():
            flush()
            last_t = t
            continue
        if window_start is None:
            window_start = t
        if t - last_t > 0.8:
            flush()
            window_start = t
        window_text += piece
        last_t = t
    flush()
    return hits


def reward_backchannel_f1(
    moshi_text_events: list[dict[str, Any]],
    gt_backchannel_times: list[float],
    window_sec: float = 1.0,
    max_dur_sec: float = 1.0,
) -> float:
    """F1 between predicted and ground-truth backchannel timings."""
    pred_times = _detect_backchannels(moshi_text_events, max_dur_sec)
    if not gt_backchannel_times and not pred_times:
        return 1.0
    if not gt_backchannel_times or not pred_times:
        return 0.0

    gt_matched = [False] * len(gt_backchannel_times)
    pred_matched = [False] * len(pred_times)

    for i, pt in enumerate(pred_times):
        for j, gt in enumerate(gt_backchannel_times):
            if not gt_matched[j] and abs(pt - gt) <= window_sec:
                pred_matched[i] = True
                gt_matched[j] = True
                break

    tp = sum(pred_matched)
    precision = tp / len(pred_times) if pred_times else 0.0
    recall = tp / len(gt_backchannel_times) if gt_backchannel_times else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Axis 4: User Interruption ──────────────────────────────────────

def reward_user_interruption(
    moshi_vad_segments: list[tuple[float, float]],
    interruption_start: float,
) -> float:
    """R = -d where d = time for Moshi to stop after user interrupts."""
    for seg_start, seg_end in sorted(moshi_vad_segments):
        if seg_start <= interruption_start < seg_end:
            stop_delay = seg_end - interruption_start
            return -stop_delay
    return 0.0  # Moshi was already silent


# ── Reward aggregation ─────────────────────────────────────────────

def decoupling_normalize(
    rewards_per_axis: list[list[float]],
    weights: list[float] | None = None,
    min_std: float = 1e-4,
) -> list[float]:
    """Reward decoupling normalization across axes (Kyutai Eq. 5).

    Each axis is independently z-normalized within the group, then
    weighted and summed.

    An axis whose group spread is below ``min_std`` contributes nothing. This
    matters far more for the LLM judge than for the deterministic axes: the
    judge emits integers on a 1-5 scale, so a group of G rollouts from one
    segment routinely lands on the same score across the board. Dividing that
    axis by a floor-clamped std would turn float noise into a full-magnitude
    advantage and push the policy on a distinction the judge never drew.
    """
    n_axes = len(rewards_per_axis)
    if not rewards_per_axis or not rewards_per_axis[0]:
        return []
    group_size = len(rewards_per_axis[0])
    if weights is None:
        weights = [1.0] * n_axes

    combined = [0.0] * group_size
    for ax_idx, axis_rewards in enumerate(rewards_per_axis):
        mean = sum(axis_rewards) / len(axis_rewards)
        var = sum((r - mean) ** 2 for r in axis_rewards) / len(axis_rewards)
        std = var ** 0.5
        if std < min_std:
            continue
        w = weights[ax_idx]
        for i, r in enumerate(axis_rewards):
            combined[i] += w * (r - mean) / std
    return combined


# -- LLM judge reward -----------------------------------------------

def judge_reward_axes(
    judgements: list[dict[str, Any]],
    score_keys: tuple[str, ...],
    flag_keys: tuple[str, ...],
    flag_penalty: float = 1.0,
) -> tuple[list[list[float]], list[str]]:
    """Split judge output into one reward axis per rubric dimension.

    Returns (axes, names) where ``axes[i]`` holds one axis's reward for every
    rollout in the group, ready to hand to decoupling_normalize.

    The eight rubric scores stay separate rather than being averaged into one
    number. Averaging first would let a large swing on one axis dominate the
    normalization; keeping them apart means each dimension is z-normalized on
    its own spread, which is the whole point of reward decoupling -- and it is
    what lets a per-axis weight actually control that axis.

    Scores are mapped from 1-5 onto 0-1 so an axis weight of 1.0 means the same
    thing here as for the deterministic axes. Flags collapse into a single
    penalty axis: they are rare, correlated, and individually near-constant
    within a group, so as separate axes they would mostly be dropped by min_std
    and otherwise fire at full z-scale on a single outlier.
    """
    if not judgements:
        return [], []

    axes: list[list[float]] = []
    names: list[str] = []
    for key in score_keys:
        axes.append([
            (float(j["scores"][key]) - 1.0) / 4.0 for j in judgements
        ])
        names.append(f"judge_{key}")

    axes.append([
        -flag_penalty * sum(1.0 for key in flag_keys if j["flags"].get(key))
        for j in judgements
    ])
    names.append("judge_flags")
    return axes, names
