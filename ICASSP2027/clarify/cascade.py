"""Cascade (ASR -> policy) baseline with an explicit confidence-threshold
ask/act rule.

This is the system class prior work optimizes (ASR confidence drives the
clarification decision). It bounds what an *explicit* uncertainty signal
achieves on our benchmark, and its threshold sweep traces a full
hit/false-alarm curve to plot the end-to-end models against.

Protocol parity with the closed-loop Moshi runs:
  * same corrupted input audio,
  * same one-repair budget: if the policy asks, the pre-synthesized repair
    audio is transcribed by the same ASR and the value re-extracted,
  * outputs are TrialScore records, so every aggregate in metrics.py works
    unchanged.

Turn-based timing (latency) is not comparable to full-duplex timing and is
therefore reported as null for this baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_oracle import WeakASR
from .metrics import TrialScore
from .slots import SLOT_WH_WORDS, slot_value_in_text


def ask_template(slot_type: str) -> str:
    wh = SLOT_WH_WORDS.get(slot_type, ("何",))[0]
    return f"すみません、{wh}でしたか？もう一度お願いします。"


def confirm_template(value_text: str) -> str:
    return f"{value_text}ですね。承知しました。"


def run_cascade_trial(
    case: dict[str, Any],
    dataset_dir: Path,
    asr: WeakASR,
    threshold: float,
) -> TrialScore:
    """One case under a fixed confidence threshold."""
    base = case["base"]
    slot_value = base["slot_value"]
    transcript, avg_lp = asr.transcribe(dataset_dir / case["audio_path"])
    heard = bool(slot_value) and slot_value_in_text(slot_value, transcript)
    confident = avg_lp is not None and avg_lp >= threshold
    # Underspecified arm: no slot in audio at all; a value can never be
    # heard, and a competent NLU notices the missing slot -> always ask.
    must_ask = base["arm"] == "underspecified"

    asked = must_ask or not confident
    if asked:
        repair_tx, _ = asr.transcribe(dataset_dir / case["repair_audio_path"])
        recovered = bool(slot_value) and slot_value_in_text(
            slot_value, repair_tx
        )
        final_text = confirm_template(repair_tx)
        slot_correct = recovered
        hallucinated = False
        label = "clarify_targeted"
    else:
        final_text = confirm_template(transcript)
        slot_correct = heard
        hallucinated = not heard
        label = "confirm"

    return TrialScore(
        case_id=case["case_id"], base_id=base["base_id"],
        condition=case["condition"], arm=base["arm"],
        expected_behavior=case["expected_behavior"], seed=0,
        first_label=label, asked=asked,
        targeted=asked,  # template asks are targeted by construction
        repair_injected=asked, final_text=final_text,
        slot_correct=slot_correct, hallucinated_confirmation=hallucinated,
        no_response=False, response_latency_sec=None,
        extra={"transcript": transcript, "avg_logprob": avg_lp,
               "threshold": threshold},
    )


def run_cascade(
    cases: list[dict[str, Any]],
    dataset_dir: Path,
    thresholds: list[float],
    model_size: str = "small",
    device: str = "cuda",
) -> dict[float, list[TrialScore]]:
    """Sweep ask thresholds; ASR results are computed once and reused."""
    asr = WeakASR(model_size=model_size, device=device)
    # Pre-transcribe every distinct wav once.
    tx_cache: dict[str, tuple[str, float | None]] = {}

    def cached(path_rel: str) -> tuple[str, float | None]:
        if path_rel not in tx_cache:
            tx_cache[path_rel] = asr.transcribe(dataset_dir / path_rel)
        return tx_cache[path_rel]

    results: dict[float, list[TrialScore]] = {t: [] for t in thresholds}
    for case in cases:
        base = case["base"]
        slot_value = base["slot_value"]
        transcript, avg_lp = cached(case["audio_path"])
        heard = bool(slot_value) and slot_value_in_text(slot_value, transcript)
        for threshold in thresholds:
            confident = avg_lp is not None and avg_lp >= threshold
            must_ask = base["arm"] == "underspecified"
            asked = must_ask or not confident
            if asked:
                repair_tx, _ = cached(case["repair_audio_path"])
                recovered = bool(slot_value) and slot_value_in_text(
                    slot_value, repair_tx
                )
                score = TrialScore(
                    case_id=case["case_id"], base_id=base["base_id"],
                    condition=case["condition"], arm=base["arm"],
                    expected_behavior=case["expected_behavior"], seed=0,
                    first_label="clarify_targeted", asked=True, targeted=True,
                    repair_injected=True,
                    final_text=confirm_template(repair_tx),
                    slot_correct=recovered, hallucinated_confirmation=False,
                    no_response=False, response_latency_sec=None,
                    extra={"transcript": transcript, "avg_logprob": avg_lp,
                           "threshold": threshold},
                )
            else:
                score = TrialScore(
                    case_id=case["case_id"], base_id=base["base_id"],
                    condition=case["condition"], arm=base["arm"],
                    expected_behavior=case["expected_behavior"], seed=0,
                    first_label="confirm", asked=False, targeted=False,
                    repair_injected=False,
                    final_text=confirm_template(transcript),
                    slot_correct=heard, hallucinated_confirmation=not heard,
                    no_response=False, response_latency_sec=None,
                    extra={"transcript": transcript, "avg_logprob": avg_lp,
                           "threshold": threshold},
                )
            results[threshold].append(score)
    return results
