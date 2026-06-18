#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from full_duplex_audio import resample_linear, write_wav_mono

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from generate_qwen3_tts_data import Qwen3TTS, VALID_SPEAKERS
except ImportError as exc:
    Qwen3TTS = None  # type: ignore[assignment,misc]
    VALID_SPEAKERS: set[str] = set()
    QWEN3_IMPORT_ERROR: ImportError | None = exc
else:
    QWEN3_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Japanese Full-Duplex-Bench profile.")
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--tts-backend",
        choices=["qwen3", "pyopenjtalk", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    parser.add_argument("--tts-speaker", default="Ono_Anna")
    parser.add_argument("--tts-language", default="Japanese")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tts-instruct", default=None)
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.tts_speed <= 0:
        parser.error("--tts-speed must be > 0")
    if VALID_SPEAKERS and args.tts_speaker not in VALID_SPEAKERS:
        parser.error(
            f"--tts-speaker={args.tts_speaker!r} is invalid. "
            f"Choose from {sorted(VALID_SPEAKERS)}"
        )
    return args


def iter_jsonl(path: Path):
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc


def initialize_tts(args: argparse.Namespace) -> tuple[str, Any | None]:
    if args.tts_backend == "pyopenjtalk":
        return "pyopenjtalk", None
    if QWEN3_IMPORT_ERROR is not None or Qwen3TTS is None:
        if args.tts_backend == "auto":
            print(
                f"[ja-data] Qwen3-TTS unavailable ({QWEN3_IMPORT_ERROR}); "
                "falling back to pyopenjtalk."
            )
            return "pyopenjtalk", None
        raise SystemExit(f"Qwen3-TTS import failed: {QWEN3_IMPORT_ERROR}")

    engine = Qwen3TTS(
        model_id=args.tts_model,
        device=args.device,
        dtype_str=args.dtype,
        attn_impl="default",
        speaker_user=args.tts_speaker,
        speaker_moshi=args.tts_speaker,
        language=args.tts_language,
        instruct_user=args.tts_instruct,
        instruct_moshi=args.tts_instruct,
    )
    try:
        engine.load()
    except Exception as exc:
        if args.tts_backend == "auto":
            print(
                f"[ja-data] Qwen3-TTS load failed ({exc}); "
                "falling back to pyopenjtalk."
            )
            return "pyopenjtalk", None
        raise
    return "qwen3", engine


def synthesize(
    text: str,
    sample_rate: int,
    speed: float,
    tts_backend: str,
    tts_engine: Any | None,
    tts_speaker: str,
) -> np.ndarray:
    if tts_backend == "qwen3":
        if tts_engine is None:
            raise RuntimeError("Qwen3-TTS backend selected without a loaded engine.")
        pcm = tts_engine.synthesize(
            text,
            speaker_role="user",
            speaker_override=tts_speaker,
        )
        pcm = np.asarray(pcm, dtype=np.float32).squeeze()
        if pcm.ndim != 1:
            raise ValueError(f"Qwen3-TTS returned non-mono audio with shape {pcm.shape}")
        if abs(speed - 1.0) > 1e-6 and pcm.size > 1:
            new_length = max(1, int(round(len(pcm) / speed)))
            pcm = np.interp(
                np.linspace(0.0, 1.0, new_length),
                np.linspace(0.0, 1.0, len(pcm)),
                pcm,
            ).astype(np.float32)
        pcm = resample_linear(pcm, int(tts_engine.sample_rate), sample_rate)
        peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
        if peak > 1.0:
            pcm /= peak
        return pcm

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
    scenario: dict[str, Any],
    sample_rate: int,
    speed: float,
    tts_backend: str,
    tts_engine: Any | None,
    tts_model: str,
    tts_speaker: str,
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
            pcm = synthesize(
                text,
                sample_rate,
                speed,
                tts_backend,
                tts_engine,
                tts_speaker,
            )
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
        "tts_backend": tts_backend,
        "tts_model": tts_model,
        "tts_speaker": tts_speaker,
        "duration_sec": round(len(noisy) / sample_rate, 4),
        "user_segments": user_segments,
        "events": events,
        "source_scenario": scenario,
    }
    return noisy, clean, metadata


def main() -> int:
    args = parse_args()
    tts_backend, tts_engine = initialize_tts(args)
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
            scenario,
            args.sample_rate,
            args.tts_speed,
            tts_backend,
            tts_engine,
            args.tts_model,
            args.tts_speaker,
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
                "tts_backend": tts_backend,
                "tts_model": args.tts_model,
                "tts_speaker": args.tts_speaker,
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
