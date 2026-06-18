#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import response_recorder as recorder
from full_duplex_audio import write_wav_mono


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run time-synchronous Moshi inference for the Japanese Full-Duplex-Bench profile."
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hf-repo", default="llm-jp/llm-jp-moshi-v1")
    parser.add_argument("--moshi-weight")
    parser.add_argument("--mimi-weight")
    parser.add_argument("--tokenizer")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true", help="Use fp16; required on V100.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks or all.")
    parser.add_argument("--tail-sec", type=float, default=8.0)
    parser.add_argument("--max-gen-sec", type=float, default=180.0)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--temp-text", type=float)
    parser.add_argument("--cfg-coef", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        hf_repo=args.hf_repo,
        moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight,
        tokenizer=args.tokenizer,
        config=args.config,
        device=args.device,
        half=args.half,
        temp=args.temp,
        temp_text=args.temp_text,
        cfg_coef=args.cfg_coef,
    )


def write_text_output(
    path: Path, text_events: list[dict[str, Any]], frame_rate: float
) -> None:
    chunks = [
        {
            "text": event["piece"],
            "timestamp": [
                round(float(event["time_sec"]), 4),
                round(float(event["time_sec"]) + 1.0 / frame_rate, 4),
            ],
        }
        for event in text_events
    ]
    path.write_text(
        json.dumps(
            {
                "text": "".join(event["piece"] for event in text_events).strip(),
                "chunks": chunks,
                "source": "moshi_text_tokens",
                "language": "ja",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_aligned_output(
    output_wav: Path,
    result: dict[str, Any],
    mimi: Any,
    acoustic_delay: int,
    device: str,
) -> dict[str, Any]:
    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    total_samples = max(1, int(round(result["total_steps"] / frame_rate * sample_rate)))
    aligned = np.zeros(total_samples, dtype=np.float32)
    decoded = recorder.decode_audio(
        result["audio_frames"], acoustic_delay, mimi, device
    )
    audible_start_step = (
        result["first_audio_step"] + acoustic_delay
        if result["first_audio_step"] is not None
        else None
    )
    if decoded is not None and audible_start_step is not None:
        start = max(0, int(round(audible_start_step / frame_rate * sample_rate)))
        end = min(total_samples, start + len(decoded))
        if end > start:
            aligned[start:end] = decoded[: end - start]
    write_wav_mono(output_wav, aligned, sample_rate)
    return {
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "total_steps": result["total_steps"],
        "input_steps": result["input_steps"],
        "first_audio_step": result["first_audio_step"],
        "audible_response_start_step": audible_start_step,
        "audible_response_start_sec": (
            round(audible_start_step / frame_rate, 4)
            if audible_start_step is not None
            else None
        ),
        "first_response_step": result["first_response_step"],
        "first_response_sec": (
            round(result["first_response_step"] / frame_rate, 4)
            if result["first_response_step"] is not None
            else None
        ),
    }


def main() -> int:
    args = parse_args()
    if args.tail_sec < 0:
        raise ValueError("--tail-sec must be >= 0")
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    selected_tasks = None if args.tasks == "all" else {
        item.strip() for item in args.tasks.split(",") if item.strip()
    }

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = [
        item
        for item in manifest["samples"]
        if selected_tasks is None or item["task"] in selected_tasks
    ]
    if not samples:
        raise SystemExit("No benchmark samples matched --tasks.")

    mimi, tokenizer, lm_gen, lm_gen_cfg, acoustic_delay = recorder.load_models(
        model_args(args)
    )
    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    run_config = {
        "model_id": args.model_id,
        "hf_repo": args.hf_repo,
        "moshi_weight": str(Path(args.moshi_weight).resolve()) if args.moshi_weight else None,
        "mimi_weight": args.mimi_weight,
        "tokenizer": args.tokenizer,
        "config": args.config,
        "dtype": "float16" if args.half else "bfloat16",
        "device": args.device,
        "seeds": seeds,
        "tasks": sorted(selected_tasks) if selected_tasks else "all",
        "tail_sec": args.tail_sec,
        "upstream_full_duplex_bench_commit": manifest.get("upstream_commit"),
        "profile": manifest.get("profile"),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total = len(samples) * len(seeds)
    completed = 0
    for sample in samples:
        source_dir = dataset_dir / sample["path"]
        metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
        variants = ["input"]
        if (source_dir / "clean_input.wav").exists():
            variants.append("clean_input")
        for seed in seeds:
            trial_dir = out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / "metadata.json", trial_dir / "metadata.json")
            for variant in variants:
                input_wav = source_dir / f"{variant}.wav"
                output_stem = "clean_output" if variant == "clean_input" else "output"
                output_wav = trial_dir / f"{output_stem}.wav"
                if output_wav.exists() and not args.overwrite:
                    print(f"[fdb] skip existing {output_wav}")
                    continue
                shutil.copy2(input_wav, trial_dir / f"{variant}.wav")
                pcm = recorder.load_wav_mono(input_wav, sample_rate)
                started = time.perf_counter()
                result = recorder.run_trial(
                    pcm=pcm,
                    seed=seed,
                    lm_gen=lm_gen,
                    mimi=mimi,
                    text_tokenizer=tokenizer,
                    device=args.device,
                    silence_sec=args.tail_sec,
                    max_gen_sec=max(
                        args.max_gen_sec, len(pcm) / sample_rate + args.tail_sec
                    ),
                    acoustic_delay=acoustic_delay,
                )
                timing = save_aligned_output(
                    output_wav, result, mimi, acoustic_delay, args.device
                )
                write_text_output(
                    trial_dir / f"{output_stem}.json",
                    result["text_events"],
                    frame_rate,
                )
                meta = {
                    **timing,
                    "model_id": args.model_id,
                    "task": sample["task"],
                    "case_id": sample["id"],
                    "seed": seed,
                    "variant": variant,
                    "input_duration_sec": round(len(pcm) / sample_rate, 4),
                    "wall_time_sec": round(time.perf_counter() - started, 4),
                    "language": "ja",
                    "expected_behavior": metadata.get("expected_behavior"),
                }
                (trial_dir / f"{output_stem}.meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            completed += 1
            print(f"[fdb] {completed}/{total} {sample['task']}/{sample['id']} seed={seed}")

    print(f"[fdb] inference complete -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
