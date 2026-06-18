#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from full_duplex_audio import resample_linear, write_wav_mono


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Japanese Full-Duplex-Bench profile.")
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--tts-speed", type=float, default=1.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc


def synthesize(text: str, sample_rate: int, speed: float) -> np.ndarray:
    try:
        import pyopenjtalk
    except ImportError as exc:
        raise SystemExit("pyopenjtalk is required to build the Japanese benchmark data.") from exc
    pcm, source_rate = pyopenjtalk.tts(text)
    pcm = np.asarray(pcm, dtype=np.float32)
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 1.0:
        pcm /= peak
    if speed <= 0:
        raise ValueError("--tts-speed must be > 0")
    if abs(speed - 1.0) > 1e-6 and pcm.size > 1:
        new_length = max(1, int(round(len(pcm) / speed)))
        pcm = np.interp(
            np.linspace(0.0, 1.0, new_length),
            np.linspace(0.0, 1.0, len(pcm)),
            pcm,
        ).astype(np.float32)
    return resample_linear(pcm, int(source_rate), sample_rate)


def render_scenario(
    scenario: dict[str, Any], sample_rate: int, speed: float
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    noisy_parts: list[np.ndarray] = []
    clean_parts: list[np.ndarray] = []
    cursor = 0.0
    user_segments: list[dict[str, Any]] = []
    events: dict[str, list[list[float]]] = {}

    for item in scenario["timeline"]:
        kind = str(item["type"])
        if kind == "silence":
            duration = float(item["duration_sec"])
            pcm = np.zeros(int(round(duration * sample_rate)), dtype=np.float32)
            start, end = cursor, cursor + duration
            noisy_parts.append(pcm)
            clean_parts.append(pcm.copy())
        elif kind in {"speech", "interrupt", "overlap_speech"}:
            text = str(item["text"])
            pcm = synthesize(text, sample_rate, speed)
            gain = float(item.get("gain", 1.0))
            noisy_pcm = np.clip(pcm * gain, -1.0, 1.0)
            start, end = cursor, cursor + len(pcm) / sample_rate
            noisy_parts.append(noisy_pcm)
            clean_parts.append(
                np.zeros_like(pcm) if kind == "overlap_speech" else noisy_pcm.copy()
            )
            user_segments.append(
                {
                    "start_sec": round(start, 4),
                    "end_sec": round(end, 4),
                    "text": text,
                    "kind": kind,
                }
            )
        else:
            raise ValueError(f"{scenario['id']}: unknown timeline type {kind!r}")

        event = item.get("event")
        if event:
            events.setdefault(str(event), []).append([round(start, 4), round(end, 4)])
        cursor = end

    noisy = np.concatenate(noisy_parts) if noisy_parts else np.zeros(1, np.float32)
    clean = np.concatenate(clean_parts) if scenario.get("paired") else None
    metadata = {
        **{key: value for key, value in scenario.items() if key != "timeline"},
        "language": "ja",
        "sample_rate": sample_rate,
        "duration_sec": round(len(noisy) / sample_rate, 4),
        "user_segments": user_segments,
        "events": events,
        "source_scenario": scenario,
    }
    return noisy, clean, metadata


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(args.scenarios))
    manifest = []
    for index, scenario in enumerate(rows, start=1):
        sample_dir = out_dir / str(scenario["task"]) / str(scenario["id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        noisy, clean, metadata = render_scenario(
            scenario, args.sample_rate, args.tts_speed
        )
        write_wav_mono(sample_dir / "input.wav", noisy, args.sample_rate)
        if clean is not None:
            write_wav_mono(sample_dir / "clean_input.wav", clean, args.sample_rate)
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (sample_dir / "input.txt").write_text(
            "".join(segment["text"] for segment in metadata["user_segments"]) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "id": scenario["id"],
                "task": scenario["task"],
                "path": str(sample_dir.relative_to(out_dir)),
                "paired": bool(clean is not None),
            }
        )
        print(f"[ja-data] {index}/{len(rows)} {scenario['task']}/{scenario['id']}")

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "Full-Duplex-Bench v1/v1.5 Japanese adaptation",
                "upstream_commit": "3e799c45a045256f47d5f1c9cda90157e2d2ec9e",
                "language": "ja",
                "count": len(manifest),
                "samples": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ja-data] wrote {len(manifest)} scenarios -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
