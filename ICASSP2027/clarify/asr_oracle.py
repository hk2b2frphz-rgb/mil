"""Weak-ASR recoverability oracle.

For every benchmark case we transcribe the (corrupted) input audio with a
deliberately weak ASR model and record whether the gold slot value is
recoverable from the transcript, plus the average token log-probability
over the utterance. This gives:

  1. A manipulation check: corruption levels are meaningful only if they
     actually destroy the information for a listener of bounded ability.
  2. A graded per-case difficulty axis for calibration analysis, replacing
     the coarse condition label.
  3. The confidence signal for the cascade baseline's ask/act policy: the
     cascade *can* see this number, an end-to-end model cannot - that
     asymmetry is the point of the comparison.

Whisper-small is the default "weak listener". `faster-whisper` is already
a project dependency (used by the FDB cascade baseline).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .slots import slot_value_in_text


@dataclass
class OracleResult:
    case_id: str
    transcript: str
    slot_recovered: bool
    avg_logprob: float | None
    span_text_recovered: bool   # slot value found verbatim (normalized)


class WeakASR:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ja",
    ):
        from faster_whisper import WhisperModel  # lazy

        self.language = language
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def transcribe(self, wav_path: Path) -> tuple[str, float | None]:
        segments, _info = self.model.transcribe(
            str(wav_path),
            language=self.language,
            beam_size=1,           # weak: greedy
            vad_filter=False,
            condition_on_previous_text=False,
        )
        pieces, logprobs = [], []
        for seg in segments:
            pieces.append(seg.text)
            if seg.avg_logprob is not None:
                logprobs.append(seg.avg_logprob)
        text = "".join(pieces).strip()
        avg_lp = sum(logprobs) / len(logprobs) if logprobs else None
        return text, avg_lp


def run_oracle(
    cases: list[dict[str, Any]],
    dataset_dir: Path,
    out_path: Path,
    model_size: str = "small",
    device: str = "cuda",
) -> list[OracleResult]:
    """Annotate every case dict with weak-ASR recoverability; write JSONL."""
    asrs: dict[str, WeakASR] = {}
    results = []
    with out_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            lang = case["base"].get("language", "ja")
            if lang not in asrs:
                from .lang import get_pack

                asrs[lang] = WeakASR(
                    model_size=model_size, device=device,
                    language=get_pack(lang).asr_language,
                )
            slot_value = case["base"]["slot_value"]
            wav = dataset_dir / case["audio_path"]
            transcript, avg_lp = asrs[lang].transcribe(wav)
            recovered = bool(slot_value) and slot_value_in_text(
                slot_value, transcript, lang=lang
            )
            res = OracleResult(
                case_id=case["case_id"],
                transcript=transcript,
                slot_recovered=recovered,
                avg_logprob=avg_lp,
                span_text_recovered=recovered,
            )
            results.append(res)
            fh.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
    return results


def load_oracle(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["case_id"]] = rec
    return out
