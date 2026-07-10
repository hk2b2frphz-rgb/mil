#!/usr/bin/env python3
"""Create upstream-style ASR-aligned JSON for every FDB input/output WAV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from full_duplex_audio import read_wav_mono
from local_baseline_common import LocalASR


WAV_TO_JSON = {
    "input.wav": "input.json",
    "clean_input.wav": "clean_input.json",
    "output.wav": "output.json",
    "clean_output.wav": "clean_output.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR-align Full-Duplex-Bench WAV files.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asr = LocalASR(args.asr_model, args.device, args.compute_type)
    asr.load()
    wavs = [path for path in sorted(args.run_dir.glob("**/*.wav")) if path.name in WAV_TO_JSON]
    for index, wav_path in enumerate(wavs, start=1):
        json_path = wav_path.with_name(WAV_TO_JSON[wav_path.name])
        if json_path.exists() and not args.overwrite:
            continue
        pcm, sample_rate = read_wav_mono(wav_path)
        text, chunks, wall_time = asr.transcribe_aligned(pcm, sample_rate)
        json_path.write_text(
            json.dumps(
                {
                    "text": text,
                    "chunks": chunks,
                    "source": "faster-whisper ASR alignment",
                    "language": "ja",
                    "asr_model": args.asr_model,
                    "asr_wall_time_sec": round(wall_time, 4),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[fdb-asr] {index}/{len(wavs)} {wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
