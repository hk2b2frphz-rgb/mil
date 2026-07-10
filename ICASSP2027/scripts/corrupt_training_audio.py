#!/usr/bin/env python3
"""Apply the corruption plan to rendered training stereo WAVs.

Runs AFTER `scripts/generate_qwen3_tts_data.py` produced
`training_set/data_stereo/sample_*.{wav,json}` from the clarify dialogues.
For every dialogue with a non-null corruption plan, this script:

  1. finds the stereo WAV + JSON sidecar for that dialogue id,
  2. locates the user-channel segment whose text matches the plan's
     `target_text` (the slot span was emitted as its own user turn by
     clarify/train_data.py, so it appears as one aligned segment),
  3. applies the named corruption condition to the RIGHT channel (user)
     over that segment's span, in place (a `.clean.wav` backup is kept),
  4. writes `corruption_report.jsonl` with per-dialogue status.

The sidecar schema is matched tolerantly (segments/turns lists with
start/end seconds and text under common key spellings); a dialogue whose
span cannot be located is reported and left clean, and the job fails at
the end if the failure rate exceeds --max-failure-rate, so a silent
schema mismatch cannot produce an uncorrupted "corrupted" dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "ICASSP2027"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clarify.corruptions import get_condition  # noqa: E402
from clarify.scenario import case_seed  # noqa: E402


def read_stereo_wav(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 2:
            raise ValueError(f"{path} is not stereo")
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data[0::2].copy(), data[1::2].copy(), sr


def write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray,
                     sr: int) -> None:
    n = max(len(left), len(right))
    left = np.pad(left, (0, n - len(left)))
    right = np.pad(right, (0, n - len(right)))
    inter = np.empty(2 * n, dtype=np.float32)
    inter[0::2] = left
    inter[1::2] = right
    pcm16 = np.clip(inter * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _iter_segments(sidecar: dict[str, Any]):
    """Yield (speaker, text, start_sec, end_sec) from tolerant schemas."""
    for key in ("segments", "turns", "alignment", "events"):
        entries = sidecar.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            speaker = str(entry.get("speaker", "")).lower()
            text = entry.get("text") or ""
            start = entry.get("start_sec", entry.get("start"))
            end = entry.get("end_sec", entry.get("end"))
            if start is None or end is None:
                continue
            yield speaker, text, float(start), float(end)
        return


def _norm(text: str) -> str:
    return "".join(text.split()).strip("、。！？!?…")


def locate_target(sidecar: dict[str, Any], target_text: str
                  ) -> tuple[float, float] | None:
    target_norm = _norm(target_text)
    candidates = []
    for speaker, text, start, end in _iter_segments(sidecar):
        if speaker != "user":
            continue
        text_norm = _norm(text)
        if text_norm == target_norm or (
                target_norm and target_norm in text_norm
                and len(target_norm) >= max(2, len(text_norm) // 2)):
            candidates.append((start, end))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # Multiple matches: the slot turn is the shortest matching span.
        return min(candidates, key=lambda se: se[1] - se[0])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-set-dir", required=True, type=Path,
                        help="Directory containing data_stereo/.")
    parser.add_argument("--corruption-plan", required=True, type=Path)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stereo_dir = args.training_set_dir / "data_stereo"
    if not stereo_dir.is_dir():
        raise SystemExit(f"{stereo_dir} not found")

    plans: dict[str, dict[str, Any]] = {}
    with args.corruption_plan.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("corruption"):
                    plans[rec["id"]] = rec["corruption"]

    wavs = sorted(stereo_dir.glob("*.wav"))
    index: dict[str, Path] = {}
    for wav in wavs:
        if wav.name.endswith(".clean.wav"):
            continue
        # sample_001_<dialogue_id>.wav -> match by contained dialogue id.
        for dialogue_id in plans:
            if dialogue_id in wav.stem:
                index[dialogue_id] = wav

    report_path = args.training_set_dir / "corruption_report.jsonl"
    n_ok = n_fail = 0
    with report_path.open("w", encoding="utf-8") as report:
        for dialogue_id, plan in sorted(plans.items()):
            wav_path = index.get(dialogue_id)
            status: dict[str, Any] = {"id": dialogue_id, "plan": plan}
            if wav_path is None:
                status["status"] = "wav_not_found"
                n_fail += 1
                report.write(json.dumps(status, ensure_ascii=False) + "\n")
                continue
            sidecar_path = wav_path.with_suffix(".json")
            if not sidecar_path.exists():
                status["status"] = "sidecar_not_found"
                n_fail += 1
                report.write(json.dumps(status, ensure_ascii=False) + "\n")
                continue
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            span = locate_target(sidecar, plan["target_text"])
            if span is None:
                status["status"] = "span_not_found"
                n_fail += 1
                report.write(json.dumps(status, ensure_ascii=False) + "\n")
                continue
            status["span"] = [round(span[0], 3), round(span[1], 3)]
            if args.dry_run:
                status["status"] = "dry_run_ok"
                n_ok += 1
                report.write(json.dumps(status, ensure_ascii=False) + "\n")
                continue
            left, right, sr = read_stereo_wav(wav_path)
            cond = get_condition(plan["condition"])
            rng = np.random.default_rng(
                case_seed(dialogue_id, plan["condition"])
            )
            try:
                corrupted = cond.apply(right, sr, span[0], span[1], rng)
            except ValueError as exc:
                status["status"] = f"corruption_error: {exc}"
                n_fail += 1
                report.write(json.dumps(status, ensure_ascii=False) + "\n")
                continue
            backup = wav_path.with_suffix(".clean.wav")
            if not backup.exists():
                write_stereo_wav(backup, left, right, sr)
            write_stereo_wav(wav_path, left, corrupted, sr)
            status["status"] = "ok"
            n_ok += 1
            report.write(json.dumps(status, ensure_ascii=False) + "\n")

    total = n_ok + n_fail
    rate = (n_fail / total) if total else 0.0
    print(f"[corrupt] ok={n_ok} fail={n_fail} "
          f"(failure rate {rate:.1%}) -> {report_path}")
    if total and rate > args.max_failure_rate:
        print(f"[corrupt] FAILURE RATE ABOVE {args.max_failure_rate:.1%}; "
              "inspect corruption_report.jsonl (schema mismatch?)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
