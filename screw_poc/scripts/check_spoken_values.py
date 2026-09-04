#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transcribe synthesized screw dialogues and check the numbers survived.

Two different things can go wrong when a TTS reads "SSH-M10-SD-EL" or
"27ニュートンメートル", and they matter very differently:

  a reading varies   M10 as エムジュウ or エムテン, 8.8 as ハチテンハチ or ハッテンハチ
  a value changes    M10 as エムイチ, 27 as ニジュウ

The first is what real people do; the text label stays correct, so the model
gets pronunciation robustness out of it.  The second makes the audio and the
text say different things, and the model learns the wrong number.  Only the
second is a defect, and you cannot tell them apart by listening to ten files.

So: transcribe each channel separately (left is the assistant, right is the
worker -- no diarization needed), pull every number and part number out of both
the transcript and the script, and compare the sets.

  python screw_poc/scripts/check_spoken_values.py screw_poc/artifacts/tts_mix/training_set
  python screw_poc/scripts/check_spoken_values.py <dir> --model large-v3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

# 数値と品番。読み上げでの取り違えが致命的になるのはこの二つだけなので、
# 文全体の一致率ではなく、これらが保たれたかだけを見る。
VALUE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
PART_RE = re.compile(r"[A-Z]{2,}-?M?[0-9]*(?:-[A-Z]+)+")
SIZE_RE = re.compile(r"M[0-9]+")

# ASR は数を漢数字でもカタカナでも返す。比較する前に算用数字へ寄せる。
KANA_DIGITS = {
    "ゼロ": "0", "レイ": "0", "零": "0",
    "イチ": "1", "一": "1", "ニ": "2", "二": "2", "サン": "3", "三": "3",
    "ヨン": "4", "シ": "4", "四": "4", "ゴ": "5", "五": "5",
    "ロク": "6", "六": "6", "ナナ": "7", "シチ": "7", "七": "7",
    "ハチ": "8", "八": "8", "キュウ": "9", "ク": "9", "九": "9",
}
TENS = {"ジュウ": 10, "十": 10}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def kana_number_to_digits(text: str) -> str:
    """「ニジュウナナ」→「27」。桁の言い方だけを拾う素朴な変換。"""
    out = text
    for kana, digit in sorted(KANA_DIGITS.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(kana + "テン", digit + ".")
    # ジュウ を含む二桁。ニジュウナナ / ジュウヨン / ニジュウ
    def two_digit(match: re.Match[str]) -> str:
        tens_word, ones_word = match.group(1), match.group(2)
        tens = KANA_DIGITS.get(tens_word, "1") if tens_word else "1"
        ones = KANA_DIGITS.get(ones_word, "0") if ones_word else "0"
        return f"{int(tens) * 10 + int(ones)}"

    pattern = "|".join(sorted(KANA_DIGITS, key=len, reverse=True))
    for tens_word in TENS:
        out = re.sub(rf"({pattern})?{tens_word}({pattern})?", two_digit, out)
    for kana, digit in sorted(KANA_DIGITS.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(kana, digit)
    return out


def values_in(text: str) -> set[str]:
    body = kana_number_to_digits(normalize(text))
    found = set(VALUE_RE.findall(body))
    found |= {m.replace("-", "") for m in PART_RE.findall(body)}
    found |= set(SIZE_RE.findall(body))
    return found


def load_script(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """合成時に書かれた dialogues.jsonl から、原文の台本を引く。"""
    path = directory / "dialogues.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row.get("id") or row.get("dialogue_id") or ""): row["turns"] for row in rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="training_set ディレクトリ")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    script = load_script(args.run_dir)

    def transcribe(pcm: "np.ndarray", rate: int) -> str:
        if rate != 16000:
            index = np.arange(0, len(pcm), rate / 16000.0)
            pcm = np.interp(index, np.arange(len(pcm)), pcm).astype("float32")
        segments, _ = model.transcribe(pcm, language="ja", beam_size=5)
        return "".join(segment.text for segment in segments)

    total_expected = 0
    total_missing = 0
    rows: list[tuple[str, str, set[str], set[str]]] = []
    for wav_path in sorted(args.run_dir.glob("data_stereo/*.wav")):
        audio, rate = sf.read(wav_path, dtype="float32", always_2d=True)
        dialogue_id = wav_path.stem.split("_", 2)[-1]
        turns = script.get(dialogue_id, [])
        for channel, role, label in ((0, "moshi", "システム"), (1, "user", "作業者")):
            expected: set[str] = set()
            for turn in turns:
                if turn.get("speaker") == role:
                    expected |= values_in(str(turn.get("text") or ""))
            if not expected:
                continue
            heard = values_in(transcribe(audio[:, channel], rate))
            missing = expected - heard
            total_expected += len(expected)
            total_missing += len(missing)
            rows.append((wav_path.stem, label, expected, missing))

    print(f"{'対話':30s} {'側':8s} {'台本の値':28s} 聞き取れなかった値")
    print("-" * 100)
    for stem, label, expected, missing in rows:
        marker = "" if not missing else "  <-- "
        print(
            f"{stem:30s} {label:8s} {','.join(sorted(expected)):28s}"
            f"{marker}{','.join(sorted(missing))}"
        )
    kept = total_expected - total_missing
    rate = kept / total_expected * 100 if total_expected else 0.0
    print()
    print(f"値の保持率 {kept}/{total_expected} = {rate:.1f}%")
    print(
        "注意: ASR 自身の誤りも「聞き取れなかった値」に混ざる。"
        "同じ台本を別 backend で合成して比べるための数字であって、絶対値ではない。"
    )


if __name__ == "__main__":
    main()
