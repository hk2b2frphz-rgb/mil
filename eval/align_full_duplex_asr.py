#!/usr/bin/env python3
"""Create ASR-aligned JSON using the pinned upstream FDB v1.5 ASR model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


WAV_TO_JSON = {
    "input.wav": "input.json",
    "clean_input.wav": "clean_input.json",
    "output.wav": "output.json",
    "clean_output.wav": "clean_output.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR-align Full-Duplex-Bench WAV files.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--asr-model", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise SystemExit(
            "The official Full-Duplex-Bench v1.5 ASR requires "
            "nemo_toolkit[asr]. Run `uv sync` before evaluation."
        ) from exc
    # Exact upstream v1.5 model. `device` is retained only to make CPU smoke
    # tests possible; normal PBS evaluation runs this on CUDA.
    asr = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)
    if args.device.startswith("cuda"):
        asr = asr.cuda()
    wavs = [path for path in sorted(args.run_dir.glob("**/*.wav")) if path.name in WAV_TO_JSON]
    for index, wav_path in enumerate(wavs, start=1):
        json_path = wav_path.with_name(WAV_TO_JSON[wav_path.name])
        if json_path.exists() and not args.overwrite:
            continue
        waveform, sample_rate = sf.read(wav_path)
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        import tempfile
        import time

        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            sf.write(temporary.name, waveform, sample_rate)
            temporary_path = temporary.name
        try:
            result = asr.transcribe([temporary_path], timestamps=True)[0]
        finally:
            Path(temporary_path).unlink(missing_ok=True)
        chunks = [
            {"text": item["word"], "timestamp": [item["start"], item["end"]]}
            for item in result.timestamp["word"]
        ]
        text = " ".join(str(item["text"]) for item in chunks)
        wall_time = time.perf_counter() - started
        json_path.write_text(
            json.dumps(
                {
                    "text": text,
                    "chunks": chunks,
                    "source": "nvidia/parakeet-tdt-0.6b-v2 (upstream FDB v1.5 ASR)",
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
