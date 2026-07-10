#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from full_duplex_audio import read_wav_mono, resample_linear, write_wav_mono

FDB_V15_BACKGROUND_PROFILE = {
    "speaker": "distinct",
    "gain_db": -15.0,
    "lowpass_hz": 3000.0,
    "echo_delay_sec": 0.1,
    "echo_gain_db": -10.0,
    "compression_threshold": 0.1,
    "compression_ratio": 4.0,
}
FDB_V15_PROTOCOL_NAME = "Full-Duplex-Bench v1.5 static overlap protocol"

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from generate_qwen3_tts_data import (
        OPENING_GREETING_TEXT,
        ForcedAligner,
        Qwen3TTS,
        VALID_SPEAKERS,
    )
except ImportError as exc:
    Qwen3TTS = None  # type: ignore[assignment,misc]
    ForcedAligner = None  # type: ignore[assignment,misc]
    VALID_SPEAKERS: set[str] = set()
    # Kept in sync by hand with generate_qwen3_tts_data.OPENING_GREETING_TEXT
    # for when torch/torchaudio aren't installed (e.g. local dev machines)
    # and that module can't be imported at all.
    OPENING_GREETING_TEXT = "もしもし、こちら孤独孤立相談窓口になります。"
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
    parser.add_argument(
        "--tts-background-speaker",
        default="Uncle_Fu",
        help="Distinct Qwen3-TTS speaker used only for background_speech announcements.",
    )
    parser.add_argument("--tts-language", default="Japanese")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tts-instruct", default=None)
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--whole-utterance",
        action="store_true",
        help=(
            "Synthesize consecutive same-speaker 'speech' timeline items in a "
            "single TTS call and slice them apart with MMS_FA forced alignment, "
            "matching the training-data whole-utterance pipeline "
            "(scripts/generate_qwen3_tts_data.py). Avoids the flat, "
            "sentence-final prosody that independent per-fragment synthesis "
            "produces across a scripted pause or backchannel gap. Falls back "
            "to per-fragment synthesis when forced alignment is unavailable."
        ),
    )
    parser.add_argument(
        "--whole-utterance-max-chars",
        type=int,
        default=150,
        help="Max characters concatenated into a single TTS call in --whole-utterance mode.",
    )
    parser.add_argument(
        "--opening-greeting",
        default=OPENING_GREETING_TEXT,
        help=(
            "Fixed line the model is trained to always open a session with "
            "(scripts/generate_qwen3_tts_data.py --opening-greeting). A matching "
            "silent lead-in is prepended to every scenario's input audio so the "
            "model has room to say it before the scripted user speech starts. "
            "Pass '' or --no-opening-greeting to disable."
        ),
    )
    parser.add_argument(
        "--no-opening-greeting",
        action="store_true",
        help="Do not reserve lead-in time for the opening greeting.",
    )
    parser.add_argument(
        "--opening-greeting-gap-sec",
        type=float,
        default=0.4,
        help="Silence after the greeting audio before scripted user speech starts (matches training --gap-sec).",
    )
    parser.add_argument(
        "--opening-greeting-cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / ".cache" / "opening_greeting",
        help=(
            "Reuse a cached greeting synthesis (keyed by text/backend/speaker/"
            "model/sample-rate/speed) across dataset builds instead of "
            "resynthesizing the fixed greeting every time. Pass '' to disable."
        ),
    )
    args = parser.parse_args()
    if args.tts_speed <= 0:
        parser.error("--tts-speed must be > 0")
    if VALID_SPEAKERS and args.tts_speaker not in VALID_SPEAKERS:
        parser.error(
            f"--tts-speaker={args.tts_speaker!r} is invalid. "
            f"Choose from {sorted(VALID_SPEAKERS)}"
        )
    if VALID_SPEAKERS and args.tts_background_speaker not in VALID_SPEAKERS:
        parser.error(
            f"--tts-background-speaker={args.tts_background_speaker!r} is invalid. "
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


def initialize_aligner(enabled: bool, device: str) -> Any | None:
    if not enabled:
        return None
    if ForcedAligner is None:
        print(
            f"[ja-data] --whole-utterance requested but forced alignment is "
            f"unavailable ({QWEN3_IMPORT_ERROR}); falling back to per-fragment synthesis."
        )
        return None
    aligner = ForcedAligner(device=device)
    try:
        aligner.load()
    except Exception as exc:
        print(
            f"[ja-data] --whole-utterance requested but ForcedAligner failed to "
            f"load ({exc}); falling back to per-fragment synthesis."
        )
        return None
    return aligner


def synthesize_raw(
    text: str,
    tts_backend: str,
    tts_engine: Any | None,
    tts_speaker: str,
    speaker_role: str = "user",
) -> tuple[np.ndarray, int]:
    """Synthesize *text* with no speed adjustment or resampling applied."""
    if tts_backend == "qwen3":
        if tts_engine is None:
            raise RuntimeError("Qwen3-TTS backend selected without a loaded engine.")
        pcm = tts_engine.synthesize(
            text,
            speaker_role=speaker_role,
            speaker_override=tts_speaker,
        )
        pcm = np.asarray(pcm, dtype=np.float32).squeeze()
        if pcm.ndim != 1:
            raise ValueError(f"Qwen3-TTS returned non-mono audio with shape {pcm.shape}")
        return pcm, int(tts_engine.sample_rate)

    try:
        import pyopenjtalk
    except ImportError as exc:
        raise SystemExit("pyopenjtalk is required to build the Japanese benchmark data.") from exc
    pcm, source_rate = pyopenjtalk.tts(text)
    return np.asarray(pcm, dtype=np.float32), int(source_rate)


def postprocess_audio(
    pcm: np.ndarray,
    source_rate: int,
    sample_rate: int,
    speed: float,
) -> np.ndarray:
    """Apply speed stretch, resample to *sample_rate*, and peak-normalize."""
    if abs(speed - 1.0) > 1e-6 and pcm.size > 1:
        new_length = max(1, int(round(len(pcm) / speed)))
        pcm = np.interp(
            np.linspace(0.0, 1.0, new_length),
            np.linspace(0.0, 1.0, len(pcm)),
            pcm,
        ).astype(np.float32)
    pcm = resample_linear(pcm, source_rate, sample_rate)
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 1.0:
        pcm = pcm / peak
    return pcm


def apply_fdb_v15_background_profile(
    pcm: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Apply the published v1.5 far-field background-speech profile.

    The paper specifies a different speaker, -15 dB attenuation, a 3 kHz
    low-pass filter, a 100 ms echo at -10 dB, and dynamic-range compression.
    The source release does not publish compressor coefficients, so we use a
    deterministic 4:1 static compressor above a 0.1 linear threshold and
    record every parameter in metadata.
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size == 0:
        return pcm
    cutoff = float(FDB_V15_BACKGROUND_PROFILE["lowpass_hz"])
    # Windowed-sinc low-pass filter: dependency-free and stable on offline PBS
    # nodes. 101 taps is sufficient at the benchmark's 24 kHz sample rate.
    taps = 101
    index = np.arange(taps, dtype=np.float32) - (taps - 1) / 2
    kernel = 2 * cutoff / sample_rate * np.sinc(2 * cutoff / sample_rate * index)
    kernel *= np.hamming(taps).astype(np.float32)
    kernel /= kernel.sum()
    half = taps // 2
    filtered = np.convolve(pcm, kernel, mode="full")[half : half + len(pcm)]
    filtered = filtered.astype(np.float32)

    echo_delay = int(round(float(FDB_V15_BACKGROUND_PROFILE["echo_delay_sec"]) * sample_rate))
    echo_gain = 10 ** (float(FDB_V15_BACKGROUND_PROFILE["echo_gain_db"]) / 20.0)
    echoed = filtered.copy()
    if echo_delay > 0 and echo_delay < len(echoed):
        echoed[echo_delay:] += filtered[:-echo_delay] * echo_gain

    threshold = float(FDB_V15_BACKGROUND_PROFILE["compression_threshold"])
    ratio = float(FDB_V15_BACKGROUND_PROFILE["compression_ratio"])
    magnitude = np.abs(echoed)
    compressed = np.sign(echoed) * np.where(
        magnitude <= threshold,
        magnitude,
        threshold + (magnitude - threshold) / ratio,
    )
    gain = 10 ** (float(FDB_V15_BACKGROUND_PROFILE["gain_db"]) / 20.0)
    return np.asarray(compressed * gain, dtype=np.float32)


def synthesize(
    text: str,
    sample_rate: int,
    speed: float,
    tts_backend: str,
    tts_engine: Any | None,
    tts_speaker: str,
    speaker_role: str = "user",
) -> np.ndarray:
    pcm, source_rate = synthesize_raw(
        text, tts_backend, tts_engine, tts_speaker, speaker_role
    )
    return postprocess_audio(pcm, source_rate, sample_rate, speed)


def greeting_cache_path(
    cache_dir: Path,
    text: str,
    tts_backend: str,
    tts_speaker: str,
    tts_model: str,
    sample_rate: int,
    speed: float,
) -> Path:
    key = "|".join(
        [text, tts_backend, tts_speaker, tts_model, str(sample_rate), str(speed)]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"opening_greeting_{digest}.wav"


def synthesize_greeting_cached(
    text: str,
    sample_rate: int,
    speed: float,
    tts_backend: str,
    tts_engine: Any | None,
    tts_speaker: str,
    tts_model: str,
    cache_dir: Path | None,
) -> np.ndarray:
    """The greeting is a fixed phrase synthesized identically on every
    dataset build; cache it on disk (keyed by every parameter that could
    change its audio) so repeated builds -- e.g. across --refresh-fdb-data
    reruns for different models -- skip resynthesizing it."""
    if cache_dir is None:
        return synthesize(text, sample_rate, speed, tts_backend, tts_engine, tts_speaker)
    path = greeting_cache_path(
        cache_dir, text, tts_backend, tts_speaker, tts_model, sample_rate, speed
    )
    if path.exists():
        pcm, cached_rate = read_wav_mono(path)
        print(f"[ja-data] opening greeting: cache hit -> {path}")
        return resample_linear(pcm, cached_rate, sample_rate)
    pcm = synthesize(text, sample_rate, speed, tts_backend, tts_engine, tts_speaker)
    write_wav_mono(path, pcm, sample_rate)
    print(f"[ja-data] opening greeting: cache miss, synthesized -> {path}")
    return pcm


def group_whole_utterance_runs(timeline: list[dict[str, Any]]) -> list[list[int]]:
    """Group indices of consecutive 'speech' timeline items that belong to the
    same speaker turn (only bridged by 'silence'), so they can be synthesized
    as one continuous utterance. A run breaks at 'interrupt'/'overlap_speech'
    items, which are a different voice/speaker and stay per-fragment."""
    runs: list[list[int]] = []
    current: list[int] = []
    for index, item in enumerate(timeline):
        kind = str(item["type"])
        if kind == "speech":
            current.append(index)
        elif kind == "silence":
            continue
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    return [run for run in runs if len(run) >= 2]


def align_whole_utterance_runs(
    timeline: list[dict[str, Any]],
    runs: list[list[int]],
    tts_backend: str,
    tts_engine: Any | None,
    tts_speaker: str,
    aligner: Any,
    max_chars: int,
) -> dict[int, tuple[np.ndarray, int, float, float]]:
    """Return {timeline_index: (raw_pcm, raw_sample_rate, start_sec, end_sec)}
    for each 'speech' item covered by a whole-utterance run."""
    boundaries: dict[int, tuple[np.ndarray, int, float, float]] = {}
    for run in runs:
        texts = [str(timeline[index]["text"]) for index in run]
        total_chars = sum(len(text) for text in texts)
        if max_chars > 0 and total_chars > max_chars:
            print(
                f"[ja-data] whole-utterance run at index {run[0]} has "
                f"{total_chars} chars (> --whole-utterance-max-chars={max_chars}); "
                "synthesizing anyway as a single call."
            )
        raw_pcm, raw_rate = synthesize_raw(
            "".join(texts), tts_backend, tts_engine, tts_speaker
        )
        expanded, _tight = aligner.align(raw_pcm, raw_rate, texts)
        for index, (start_sec, end_sec) in zip(run, expanded):
            boundaries[index] = (raw_pcm, raw_rate, start_sec, end_sec)
    return boundaries


def render_scenario(
    scenario: dict[str, Any],
    sample_rate: int,
    speed: float,
    tts_backend: str,
    tts_engine: Any | None,
    tts_model: str,
    tts_speaker: str,
    whole_utterance_boundaries: dict[int, tuple[np.ndarray, int, float, float]] | None = None,
    lead_in_sec: float = 0.0,
    opening_greeting: dict[str, Any] | None = None,
    tts_background_speaker: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    noisy_parts: list[np.ndarray] = []
    clean_parts: list[np.ndarray] = []
    cursor = 0.0
    user_segments: list[dict[str, Any]] = []
    events: dict[str, list[list[float]]] = {}
    whole_utterance_boundaries = whole_utterance_boundaries or {}

    if lead_in_sec > 0:
        # Silent room for the model to say its trained opening greeting
        # before the scripted user speech begins. Every deterministic
        # metric in evaluate_full_duplex_ja.py is anchored to specific
        # timeline events (pause/interruption/overlap intervals) or to
        # user_segments end times, not to absolute recording start, so
        # shifting everything later by lead_in_sec doesn't require any
        # changes there.
        lead_in = np.zeros(int(round(lead_in_sec * sample_rate)), dtype=np.float32)
        noisy_parts.append(lead_in)
        clean_parts.append(lead_in.copy())
        cursor = lead_in_sec

    for timeline_index, item in enumerate(scenario["timeline"]):
        kind = str(item["type"])
        if kind == "silence":
            duration = float(item["duration_sec"])
            pcm = np.zeros(int(round(duration * sample_rate)), dtype=np.float32)
            start, end = cursor, cursor + duration
            noisy_parts.append(pcm)
            clean_parts.append(pcm.copy())
        elif kind in {"speech", "interrupt", "overlap_speech"}:
            text = str(item["text"])
            boundary = whole_utterance_boundaries.get(timeline_index)
            if boundary is not None:
                raw_pcm, raw_rate, start_sec, end_sec = boundary
                start_sample = max(0, int(round(start_sec * raw_rate)))
                end_sample = min(raw_pcm.size, int(round(end_sec * raw_rate)))
                fragment = raw_pcm[start_sample:end_sample]
                if fragment.size == 0:
                    fragment = np.zeros(1, dtype=np.float32)
                pcm = postprocess_audio(fragment, raw_rate, sample_rate, speed)
            else:
                item_speaker = (
                    tts_background_speaker
                    if kind == "overlap_speech"
                    and scenario.get("task") == "background_speech"
                    and tts_background_speaker
                    else tts_speaker
                )
                pcm = synthesize(
                    text,
                    sample_rate,
                    speed,
                    tts_backend,
                    tts_engine,
                    item_speaker,
                )
            gain = float(item.get("gain", 1.0))
            if kind == "overlap_speech" and scenario.get("task") == "background_speech":
                pcm = apply_fdb_v15_background_profile(pcm, sample_rate)
                gain = 1.0
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
        "tts_background_speaker": tts_background_speaker,
        "background_acoustic_profile": (
            FDB_V15_BACKGROUND_PROFILE
            if scenario.get("task") == "background_speech"
            else None
        ),
        "speaker": tts_speaker,
        "tts_mode": "whole-utterance" if whole_utterance_boundaries else "per-fragment",
        "duration_sec": round(len(noisy) / sample_rate, 4),
        "user_segments": user_segments,
        "events": events,
        "opening_greeting": opening_greeting,
        "source_scenario": scenario,
    }
    return noisy, clean, metadata


def main() -> int:
    args = parse_args()
    tts_backend, tts_engine = initialize_tts(args)
    aligner = initialize_aligner(args.whole_utterance, args.device)
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opening_greeting = None
    lead_in_sec = 0.0
    if not args.no_opening_greeting and args.opening_greeting:
        # Synthesize once (same backend/speaker as the rest of the dataset)
        # purely to measure how long the model's trained greeting takes to
        # say, so every scenario reserves exactly that much silent lead-in
        # plus a turn-taking gap before scripted user speech starts. Cached
        # on disk since this exact phrase never changes between builds.
        greeting_cache_dir = (
            args.opening_greeting_cache_dir
            if str(args.opening_greeting_cache_dir)
            else None
        )
        greeting_pcm = synthesize_greeting_cached(
            args.opening_greeting,
            args.sample_rate,
            args.tts_speed,
            tts_backend,
            tts_engine,
            args.tts_speaker,
            args.tts_model,
            greeting_cache_dir,
        )
        greeting_duration_sec = round(len(greeting_pcm) / args.sample_rate, 4)
        lead_in_sec = greeting_duration_sec + args.opening_greeting_gap_sec
        opening_greeting = {
            "text": args.opening_greeting,
            "duration_sec": greeting_duration_sec,
            "gap_sec": args.opening_greeting_gap_sec,
        }
        print(
            f"[ja-data] opening greeting reserves {lead_in_sec:.2f}s lead-in "
            f"({greeting_duration_sec:.2f}s speech + {args.opening_greeting_gap_sec:.2f}s gap)"
        )

    rows = list(iter_jsonl(args.scenarios))
    manifest = []
    for index, scenario in enumerate(rows, start=1):
        sample_dir = out_dir / str(scenario["task"]) / str(scenario["id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        whole_utterance_boundaries = None
        if aligner is not None:
            runs = group_whole_utterance_runs(scenario["timeline"])
            if runs:
                whole_utterance_boundaries = align_whole_utterance_runs(
                    scenario["timeline"],
                    runs,
                    tts_backend,
                    tts_engine,
                    args.tts_speaker,
                    aligner,
                    args.whole_utterance_max_chars,
                )
        noisy, clean, metadata = render_scenario(
            scenario,
            args.sample_rate,
            args.tts_speed,
            tts_backend,
            tts_engine,
            args.tts_model,
            args.tts_speaker,
            whole_utterance_boundaries,
            lead_in_sec,
            opening_greeting,
            args.tts_background_speaker,
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
                "tts_background_speaker": args.tts_background_speaker,
                "protocol": {
                    "name": FDB_V15_PROTOCOL_NAME,
                    "background_speech": FDB_V15_BACKGROUND_PROFILE,
                    "paired_clean_input": True,
                },
                "whole_utterance": aligner is not None,
                "opening_greeting": opening_greeting,
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
