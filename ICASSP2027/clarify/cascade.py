"""Cascade (ASR -> policy) baseline with an explicit confidence-threshold
ask/act rule, language-parameterized.

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
from .lang import get_pack
from .metrics import TrialScore
from .slots import slot_value_in_text


def ask_template(slot_type: str, lang: str) -> str:
    pack = get_pack(lang)
    return pack.ask_by_slot.get(slot_type, pack.ask_fallback)


def confirm_template(value_text: str, intent: str, lang: str) -> str:
    pack = get_pack(lang)
    tpl = pack.confirm_by_intent.get(intent, pack.confirm_fallback)
    return tpl.format(value=value_text)


def run_cascade(
    cases: list[dict[str, Any]],
    dataset_dir: Path,
    thresholds: list[float],
    model_size: str = "small",
    device: str = "cuda",
) -> dict[float, list[TrialScore]]:
    """Sweep ask thresholds; ASR results are computed once per wav.

    The benchmark language comes from each case (one WeakASR per language
    is instantiated lazily, so a mixed en/ja manifest also works).
    """
    asrs: dict[str, WeakASR] = {}
    tx_cache: dict[str, tuple[str, float | None]] = {}

    def transcribe(path_rel: str, lang: str) -> tuple[str, float | None]:
        key = f"{lang}::{path_rel}"
        if key not in tx_cache:
            if lang not in asrs:
                asrs[lang] = WeakASR(
                    model_size=model_size, device=device,
                    language=get_pack(lang).asr_language,
                )
            tx_cache[key] = asrs[lang].transcribe(dataset_dir / path_rel)
        return tx_cache[key]

    results: dict[float, list[TrialScore]] = {t: [] for t in thresholds}
    for case in cases:
        base = case["base"]
        lang = base.get("language", "ja")
        slot_value = base["slot_value"]
        intent = base["intent"]
        transcript, avg_lp = transcribe(case["audio_path"], lang)
        heard = bool(slot_value) and slot_value_in_text(
            slot_value, transcript, lang=lang
        )
        common = dict(
            case_id=case["case_id"], base_id=base["base_id"],
            condition=case["condition"], arm=base["arm"],
            expected_behavior=case["expected_behavior"], seed=0,
            no_response=False, response_latency_sec=None,
            language=lang, source=base.get("source", "massive"),
            audio_source=base.get("audio_source", "tts"),
        )
        for threshold in thresholds:
            confident = avg_lp is not None and avg_lp >= threshold
            # Underspecified arm: no slot in audio at all; a competent NLU
            # notices the missing slot -> always ask.
            must_ask = base["arm"] == "underspecified"
            asked = must_ask or not confident
            if asked:
                repair_tx, _ = transcribe(case["repair_audio_path"], lang)
                recovered = bool(slot_value) and slot_value_in_text(
                    slot_value, repair_tx, lang=lang
                )
                score = TrialScore(
                    **common,
                    first_label="clarify_targeted", asked=True, targeted=True,
                    repair_injected=True,
                    final_text=confirm_template(repair_tx, intent, lang),
                    slot_correct=recovered, hallucinated_confirmation=False,
                    extra={"transcript": transcript, "avg_logprob": avg_lp,
                           "threshold": threshold},
                )
            else:
                score = TrialScore(
                    **common,
                    first_label="confirm", asked=False, targeted=False,
                    repair_injected=False,
                    final_text=confirm_template(transcript, intent, lang),
                    slot_correct=heard, hallucinated_confirmation=not heard,
                    extra={"transcript": transcript, "avg_logprob": avg_lp,
                           "threshold": threshold},
                )
            results[threshold].append(score)
    return results
