#!/usr/bin/env python3
from __future__ import annotations
"""
Run a local cascade (ASR->LLM) or SpeechLLM (audio->LLM) baseline over an
already-built Full-Duplex-Bench-JA dataset (eval/build_full_duplex_ja_dataset.py),
writing run_full_duplex_bench.py-compatible trial directories so
evaluate_full_duplex_ja.py works unchanged:

    # Build the dataset once WITHOUT the Moshi-only opening greeting --
    # a cascade/SpeechLLM baseline was never trained to say it, and has no
    # streaming behavior to reserve lead-in time for anyway.
    uv run python eval/build_full_duplex_ja_dataset.py \
        --scenarios eval_sets/full_duplex_ja/scenarios.jsonl \
        --out-dir data/full_duplex_ja_nogreeting --no-opening-greeting

    uv run python eval/run_local_baseline_full_duplex.py --system cascade \
        --dataset-dir data/full_duplex_ja_nogreeting \
        --out-dir eval_runs/full_duplex/cascade_gemma2b/inference \
        --model-id cascade_gemma2b

    uv run python eval/evaluate_full_duplex_ja.py \
        --run-dir eval_runs/full_duplex/cascade_gemma2b/inference \
        --out-dir eval_runs/full_duplex/cascade_gemma2b/benchmark_results

**Read this before trusting the numbers.** Cascade/SpeechLLM baselines are
turn-based: they only ever produce ONE response, starting after the ENTIRE
input.wav (including any embedded pause/interruption/overlap audio) has
been fully processed. There is no way for them to react mid-utterance, so:

- `pause_handling`/`smooth_turn_taking`/`user_interruption`: TOR and
  response-latency metrics will faithfully show these baselines never take
  over early and always respond late by roughly
  asr_wall + llm_wall + tts_wall seconds -- that IS the point of the
  comparison, not a bug.
- `backchannel`: the completed response is split into one pseudo-chunk per
  character (proportionally timestamped) so `TOR`'s word-count threshold
  still classifies real sentences vs. short acks correctly (see
  build_pseudo_chunks() below) -- but `backchannel_frequency`/`jsd` count
  *occurrences* of separate backchannel-like segments over time, and a
  turn-based system only ever produces one segment total (it cannot
  interject multiple times while the user keeps talking), so those two
  metrics stay structurally not meaningful for these systems even with the
  chunk fix; treat `TOR` and latency as the comparable numbers here.
- `user_backchannel`/`talking_to_other`/`background_speech`: no
  clean_output.wav is produced (there is no separate "ignore the overlay"
  processing path), so `clean_post_overlap_speech_sec` /
  `post_overlap_speech_delta_sec` will be computed against an empty
  baseline. `stop_latency_sec`/`overlap_response_sec` remain meaningful.

The Azure/LLM-judge dimensions (contextual_relevance, safety_boundary,
etc.) via judge_full_duplex_azure.py / judge_llmjp_style.py stay fully
meaningful regardless -- they judge response content, not turn-taking.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from local_baseline_common import (  # noqa: E402
    COUNSELOR_SYSTEM_PROMPT,
    GemmaLLM,
    LocalASR,
    SpeechLLM,
    init_baseline_tts,
)
from build_full_duplex_ja_dataset import synthesize  # noqa: E402
from full_duplex_audio import read_wav_mono, write_wav_mono, write_wav_stereo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local cascade or SpeechLLM baseline over a Full-Duplex-Bench-JA dataset."
    )
    parser.add_argument("--system", required=True, choices=["cascade", "speechllm"])
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks or all.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--tts-backend", choices=["qwen3", "pyopenjtalk", "auto"], default="auto")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--tts-speaker", default="Ono_Anna")
    parser.add_argument("--tts-language", default="Japanese")
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")

    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="float16")

    parser.add_argument("--llm-model", default="google/gemma-2-2b-it")
    parser.add_argument("--llm-python", default=None)
    parser.add_argument("--llm-device-map", default="auto")
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-new-tokens", type=int, default=200)
    parser.add_argument("--llm-timeout-sec", type=float, default=300.0)

    parser.add_argument("--speechllm-model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--speechllm-python", default=None)
    parser.add_argument("--speechllm-device-map", default="auto")
    parser.add_argument("--speechllm-dtype", default="auto")
    parser.add_argument("--speechllm-max-new-tokens", type=int, default=200)
    parser.add_argument("--speechllm-timeout-sec", type=float, default=300.0)

    return parser.parse_args()


def build_pseudo_chunks(
    text: str, start_sec: float, duration_sec: float
) -> list[dict[str, Any]]:
    """Split a completed batch response into one pseudo-chunk per character,
    proportionally timestamped across duration_sec.

    evaluate_full_duplex_ja.py's take_over()/TOR classification assumes
    Moshi-style per-token streamed chunks: it treats a response as a genuine
    turn (not a backchannel-like blip) once it spans >3 counting units. A
    cascade/SpeechLLM baseline only ever produces ONE completed string, so
    without this, any response under turn_duration_threshold (1.0s) would
    universally register as a non-turnover (TOR=0, response_latency_sec
    dropped) regardless of how much it actually says -- a modeling artifact
    of "one chunk", not a fact about the baseline. Splitting per character
    approximates real token-stream granularity closely enough that
    real-sentence responses correctly count as turns while short ack-like
    replies still correctly fall under the threshold.
    """
    chars = list(text)
    if not chars or duration_sec <= 0:
        return []
    per_char = duration_sec / len(chars)
    return [
        {
            "text": ch,
            "timestamp": [
                round(start_sec + index * per_char, 4),
                round(start_sec + (index + 1) * per_char, 4),
            ],
        }
        for index, ch in enumerate(chars)
    ]


def process_sample(
    sample: dict[str, Any],
    seed: int,
    dataset_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    tts_backend: str,
    tts_engine: Any | None,
    asr: LocalASR | None,
    llm: GemmaLLM | None,
    speechllm: SpeechLLM | None,
    model_id: str,
) -> None:
    source_dir = dataset_dir / sample["path"]
    metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    trial_dir = out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "metadata.json", trial_dir / "metadata.json")

    input_wav = source_dir / "input.wav"
    shutil.copy2(input_wav, trial_dir / "input.wav")
    pcm, sample_rate = read_wav_mono(input_wav)
    input_duration_sec = len(pcm) / sample_rate

    asr_text = None
    asr_wall = 0.0
    if args.system == "cascade":
        assert asr is not None and llm is not None
        asr_text, asr_wall = asr.transcribe(pcm, sample_rate)
        response_text, llm_wall = llm.respond(asr_text, COUNSELOR_SYSTEM_PROMPT)
    else:
        assert speechllm is not None
        response_text, llm_wall = speechllm.respond(pcm, sample_rate, COUNSELOR_SYSTEM_PROMPT)

    tts_started = time.perf_counter()
    has_response = bool(response_text.strip())
    response_pcm = (
        synthesize(response_text, sample_rate, args.tts_speed, tts_backend, tts_engine, args.tts_speaker)
        if has_response
        else np.zeros(1, dtype=np.float32)
    )
    tts_wall = time.perf_counter() - tts_started

    total_wall = asr_wall + llm_wall + tts_wall
    # Place the response at the real time it would actually start: after the
    # full input is heard AND all processing finishes. This makes
    # evaluate_full_duplex_ja.py's existing (unchanged) latency/TOR metrics,
    # which are derived from output.wav/output.json placement rather than
    # from output.meta.json, automatically reflect true cascade/SpeechLLM
    # latency instead of understating it.
    output_start_sec = input_duration_sec + total_wall
    silence = np.zeros(int(round(output_start_sec * sample_rate)), dtype=np.float32)
    aligned = np.concatenate([silence, response_pcm])
    write_wav_mono(trial_dir / "output.wav", aligned, sample_rate)
    # Left = input speech, right = model output, for side-by-side listening
    # review; not used by any deterministic/LLM-judge metric.
    write_wav_stereo(trial_dir / "output_stereo.wav", pcm, aligned, sample_rate)

    response_duration_sec = len(response_pcm) / sample_rate
    output_json = {
        "text": response_text,
        "chunks": (
            build_pseudo_chunks(response_text, output_start_sec, response_duration_sec)
            if has_response
            else []
        ),
        "source": f"{args.system}_text",
        "language": "ja",
    }
    (trial_dir / "output.json").write_text(
        json.dumps(output_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output_meta = {
        "model_id": model_id,
        "seed": seed,
        "task": sample["task"],
        "case_id": sample["id"],
        "input_duration_sec": round(input_duration_sec, 4),
        "wall_time_sec": round(total_wall, 4),
        "language": "ja",
        "expected_behavior": metadata.get("expected_behavior"),
        "system": args.system,
        "asr_transcript": asr_text,
        "asr_wall_time_sec": round(asr_wall, 4),
        "llm_wall_time_sec": round(llm_wall, 4),
        "tts_wall_time_sec": round(tts_wall, 4),
        "audible_response_start_sec": round(output_start_sec, 4),
    }
    (trial_dir / "output.meta.json").write_text(
        json.dumps(output_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    selected_tasks = None if args.tasks == "all" else {
        item.strip() for item in args.tasks.split(",") if item.strip()
    }

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    samples = [
        item
        for item in manifest["samples"]
        if selected_tasks is None or item["task"] in selected_tasks
    ]
    if not samples:
        raise SystemExit("No benchmark samples matched --tasks.")

    tts_backend, tts_engine = init_baseline_tts(
        args.tts_backend, args.tts_model, args.tts_speaker, args.tts_language,
        args.device, args.dtype,
    )

    asr = llm = speechllm = None
    if args.system == "cascade":
        asr = LocalASR(args.asr_model, args.asr_device, args.asr_compute_type)
        llm = GemmaLLM(
            args.llm_model, args.llm_python, args.llm_device_map, args.llm_dtype,
            args.llm_temperature, args.llm_top_p, args.llm_max_new_tokens,
            args.llm_timeout_sec,
        )
        default_model_id = f"cascade_{args.asr_model}_{args.llm_model.split('/')[-1]}"
    else:
        speechllm = SpeechLLM(
            args.speechllm_model, args.speechllm_python, args.speechllm_device_map,
            args.speechllm_dtype, args.speechllm_max_new_tokens, args.speechllm_timeout_sec,
        )
        default_model_id = f"speechllm_{args.speechllm_model.split('/')[-1]}"
    model_id = args.model_id or default_model_id

    run_config = {
        "model_id": model_id,
        "system": args.system,
        "seeds": seeds,
        "tasks": sorted(selected_tasks) if selected_tasks else "all",
        "upstream_full_duplex_bench_commit": manifest.get("upstream_commit"),
        "profile": manifest.get("profile"),
        "note": (
            "Turn-based local baseline, not full-duplex. See this script's "
            "module docstring for which metrics are/aren't meaningful."
        ),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    total = len(samples) * len(seeds)
    completed = 0
    for sample in samples:
        for seed in seeds:
            trial_dir = out_dir / sample["task"] / sample["id"] / f"seed_{seed}"
            if (trial_dir / "output.meta.json").exists() and not args.overwrite:
                print(f"[local-fdb] skip existing {trial_dir}")
                completed += 1
                continue
            process_sample(
                sample, seed, dataset_dir, out_dir, args, tts_backend, tts_engine,
                asr, llm, speechllm, model_id,
            )
            completed += 1
            print(f"[local-fdb] {completed}/{total} {sample['task']}/{sample['id']} seed={seed}")

    print(f"[local-fdb] inference complete -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
