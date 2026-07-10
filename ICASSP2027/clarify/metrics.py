"""Offline metrics for the clarification benchmark.

Terminology (paper-facing):
  CRR   - Clarification Request Rate: fraction of trials whose first
          response turn is a clarification (targeted or generic).
  T-CRR - Targeted CRR: clarification that names the corrupted slot's
          WH-category (何時/どこ/誰...) or echo-questions the value.
  HCR   - Hallucinated Confirmation Rate: model proceeds (confirms) with a
          wrong or absent slot value without asking. The failure mode the
          paper is about.
  SSR   - Slot Success Rate: the gold slot value appears in the model's
          final committed turn (after at most one repair exchange).
  Decision quality - treat "should ask" (mask_* conditions, underspecified
          arm) vs "should act" (clean) as a binary detection problem:
          hit rate, false-alarm rate, balanced accuracy.
  Selective risk - among trials where the model acted (did not ask), the
          fraction with a wrong slot; coverage is the act rate. Reported
          per condition to draw risk-coverage curves across the SNR grid.

Every proportion carries a Wilson 95% interval; model-vs-model deltas use a
paired bootstrap over base utterances (both models see identical audio).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .detector import ModelTurn, classify_turn, first_turn_after, segment_turns
from .slots import slot_value_in_text

CLARIFY_LABELS = ("clarify_targeted", "clarify_generic")


# ---------------------------------------------------------------------------
# per-trial scoring
# ---------------------------------------------------------------------------

@dataclass
class TrialScore:
    case_id: str
    base_id: str
    condition: str
    arm: str
    expected_behavior: str
    seed: int
    first_label: str
    asked: bool
    targeted: bool
    repair_injected: bool
    final_text: str
    slot_correct: bool
    hallucinated_confirmation: bool
    no_response: bool
    response_latency_sec: float | None
    extra: dict[str, Any] = field(default_factory=dict)


def score_trial(
    case: dict[str, Any],
    seed: int,
    text_events: list[dict[str, Any]],
    policy: dict[str, Any],
    utterance_end_sec: float,
    turn_gap_sec: float = 1.2,
) -> TrialScore:
    """Recompute labels offline from the raw event stream.

    `case` is a BenchmarkCase.to_json() dict; `policy` is the driver's
    policy summary (used for repair timing, not for labels).
    """
    base = case["base"]
    slot_type = base["slot_type"]
    slot_value = base["slot_value"] or None

    turns = segment_turns(text_events, gap_sec=turn_gap_sec)
    first = first_turn_after(turns, utterance_end_sec)
    if first is None:
        return TrialScore(
            case_id=case["case_id"], base_id=base["base_id"],
            condition=case["condition"], arm=base["arm"],
            expected_behavior=case["expected_behavior"], seed=seed,
            first_label="no_response", asked=False, targeted=False,
            repair_injected=bool(policy.get("repair_injected")),
            final_text="", slot_correct=False,
            hallucinated_confirmation=False, no_response=True,
            response_latency_sec=None,
        )

    first_label_obj = classify_turn(first.text, slot_type, slot_value)
    asked = first_label_obj.is_clarification
    targeted = first_label_obj.label == "clarify_targeted"

    # Final committed turn: last turn in the trial (post-repair if repair
    # was injected, otherwise the first response turn itself).
    final = _final_turn(turns, first, policy)
    final_text = final.text if final else first.text
    slot_correct = bool(slot_value) and slot_value_in_text(
        slot_value, final_text
    )
    # For the underspecified arm the gold value only exists in the repair,
    # so slot_correct is only achievable by asking - by construction.
    hallucinated = (
        not asked
        and first_label_obj.label == "confirm"
        and not slot_correct
    )
    latency = max(0.0, first.start_sec - utterance_end_sec) \
        if first.start_sec >= utterance_end_sec else 0.0
    return TrialScore(
        case_id=case["case_id"], base_id=base["base_id"],
        condition=case["condition"], arm=base["arm"],
        expected_behavior=case["expected_behavior"], seed=seed,
        first_label=first_label_obj.label, asked=asked, targeted=targeted,
        repair_injected=bool(policy.get("repair_injected")),
        final_text=final_text, slot_correct=slot_correct,
        hallucinated_confirmation=hallucinated, no_response=False,
        response_latency_sec=latency,
        extra={"first_text": first.text,
               "online_turns": policy.get("turns", [])},
    )


def _final_turn(
    turns: list[ModelTurn], first: ModelTurn, policy: dict[str, Any]
) -> ModelTurn | None:
    repair_end_step = policy.get("repair_end_step")
    frame_rate = policy.get("frame_rate") or 12.5
    if repair_end_step:
        repair_end_sec = repair_end_step / frame_rate
        after = [t for t in turns if t.end_sec >= repair_end_sec]
        if after:
            return after[0]
    return turns[-1] if turns else first


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _prop(scores: list[TrialScore], pred) -> dict[str, Any]:
    n = len(scores)
    k = sum(1 for s in scores if pred(s))
    lo, hi = wilson_interval(k, n)
    return {"rate": k / n if n else None, "k": k, "n": n,
            "ci95": [round(lo, 4), round(hi, 4)]}


def aggregate_by_condition(scores: list[TrialScore]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_cond: dict[str, list[TrialScore]] = defaultdict(list)
    for s in scores:
        by_cond[s.condition].append(s)
    for cond, group in sorted(by_cond.items()):
        acted = [s for s in group if not s.asked and not s.no_response]
        latencies = [s.response_latency_sec for s in group
                     if s.response_latency_sec is not None]
        out[cond] = {
            "CRR": _prop(group, lambda s: s.asked),
            "T_CRR": _prop(group, lambda s: s.targeted),
            "HCR": _prop(group, lambda s: s.hallucinated_confirmation),
            "SSR": _prop(group, lambda s: s.slot_correct),
            "no_response": _prop(group, lambda s: s.no_response),
            "selective": {
                "coverage": round(len(acted) / len(group), 4) if group else None,
                "risk": (round(1 - sum(s.slot_correct for s in acted)
                               / len(acted), 4) if acted else None),
            },
            "latency_sec": {
                "mean": round(float(np.mean(latencies)), 3) if latencies else None,
                "p90": round(float(np.percentile(latencies, 90)), 3)
                if latencies else None,
            },
        }
    return out


def decision_quality(scores: list[TrialScore]) -> dict[str, Any]:
    """Binary ask/act detection over the unambiguous conditions."""
    should_ask = [s for s in scores if s.expected_behavior == "ask"]
    should_act = [s for s in scores if s.expected_behavior == "act"]
    hit = _prop(should_ask, lambda s: s.asked)
    fa = _prop(should_act, lambda s: s.asked)
    bal = None
    if hit["rate"] is not None and fa["rate"] is not None:
        bal = round((hit["rate"] + (1 - fa["rate"])) / 2, 4)
    return {"hit_rate": hit, "false_alarm_rate": fa,
            "balanced_accuracy": bal}


def paired_bootstrap_delta(
    scores_a: list[TrialScore],
    scores_b: list[TrialScore],
    metric,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap over base utterances for metric(s)->bool.

    Returns the observed delta (B - A), a 95% CI, and the fraction of
    bootstrap replicates where the delta sign flips (two-sided p proxy).
    """
    def by_base(scores: list[TrialScore]) -> dict[str, list[int]]:
        d: dict[str, list[int]] = defaultdict(list)
        for s in scores:
            d[s.base_id].append(int(metric(s)))
        return d

    a_map, b_map = by_base(scores_a), by_base(scores_b)
    common = sorted(set(a_map) & set(b_map))
    if not common:
        return {"error": "no common base utterances"}
    a_means = np.array([np.mean(a_map[b]) for b in common])
    b_means = np.array([np.mean(b_map[b]) for b in common])
    observed = float(np.mean(b_means - a_means))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(n_boot, len(common)))
    deltas = (b_means[idx] - a_means[idx]).mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_flip = float(np.mean(deltas <= 0) if observed > 0
                   else np.mean(deltas >= 0))
    return {
        "delta": round(observed, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_sign_flip": round(min(1.0, 2 * p_flip), 4),
        "n_bases": len(common),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_scores(path: Path, scores: Iterable[TrialScore]) -> None:
    import dataclasses

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in scores:
            fh.write(json.dumps(dataclasses.asdict(s), ensure_ascii=False)
                     + "\n")


def read_scores(path: Path) -> list[TrialScore]:
    scores = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                scores.append(TrialScore(**json.loads(line)))
    return scores
