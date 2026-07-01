#!/usr/bin/env python3
from __future__ import annotations
"""
Run a local cascade (ASR->LLM) or SpeechLLM (audio->LLM) baseline over
eval_sets/loneliness_support.jsonl, writing response_recorder.py-compatible
trial directories so the existing local judge pipeline works unchanged:

    uv run python eval/run_local_baseline_loneliness.py --system cascade \
        --out-dir results/cascade_gemma2b_eval --model-id cascade_gemma2b

    python eval/pack_response_recorder_results.py \
        --recorder-dir results/cascade_gemma2b_eval \
        --scenarios eval_sets/loneliness_support.jsonl \
        --system-id cascade_gemma2b \
        --out eval_runs/cascade_gemma2b_responses.jsonl

    # then judge_openai.py / judge_llmjp_style.py / summarize_eval.py exactly
    # as documented in docs/evaluation_plan.md, and compare summaries
    # side by side with a Moshi run packed the same way.

Both baselines are turn-based: they only produce a response after the whole
input utterance has been synthesized and (for cascade) transcribed. There is
no full-duplex behavior to test here -- this script exists to compare
response quality, latency, and safety against Moshi on the same eval_sets,
not turn-taking behavior (see run_local_baseline_full_duplex.py for that).
"""

import argparse
import json
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
from full_duplex_audio import write_wav_mono  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local cascade or SpeechLLM baseline over loneliness_support.jsonl."
    )
    parser.add_argument("--system", required=True, choices=["cascade", "speechllm"])
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("eval_sets/loneliness_support.jsonl"),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Label used in meta.json/model_repo. Defaults to '<system>_<llm-or-speechllm-model>'.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    # TTS (both for synthesizing the user's input utterance and the reply).
    parser.add_argument("--tts-backend", choices=["qwen3", "pyopenjtalk", "auto"], default="auto")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--tts-speaker", default="Ono_Anna")
    parser.add_argument("--tts-language", default="Japanese")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")

    # Cascade ASR (faster-whisper).
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--asr-compute-type", default="float16")

    # Cascade LLM (Gemma via gemma_dialogue_worker.py subprocess).
    parser.add_argument("--llm-model", default="google/gemma-2-2b-it")
    parser.add_argument("--llm-python", default=None)
    parser.add_argument("--llm-device-map", default="auto")
    parser.add_argument("--llm-dtype", default="auto")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-new-tokens", type=int, default=200)
    parser.add_argument("--llm-timeout-sec", type=float, default=300.0)

    # SpeechLLM (Qwen2-Audio via speechllm_worker.py subprocess).
    parser.add_argument("--speechllm-model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--speechllm-python", default=None)
    parser.add_argument("--speechllm-device-map", default="auto")
    parser.add_argument("--speechllm-dtype", default="auto")
    parser.add_argument("--speechllm-max-new-tokens", type=int, default=200)
    parser.add_argument("--speechllm-timeout-sec", type=float, default=300.0)

    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    scenarios = list(iter_jsonl(args.scenarios))
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    for index, scenario in enumerate(scenarios, start=1):
        case_id = scenario.get("id") or f"case_{index:04d}"
        user_text = str(scenario["user_text"])
        trial_dir = out_dir / case_id / "seed_0"
        if trial_dir.exists() and not args.overwrite:
            print(f"[local-baseline] {index}/{len(scenarios)} skip existing {trial_dir}")
            continue
        trial_dir.mkdir(parents=True, exist_ok=True)

        input_pcm = synthesize(
            user_text, args.sample_rate, args.tts_speed, tts_backend, tts_engine,
            args.tts_speaker,
        )
        input_duration_sec = len(input_pcm) / args.sample_rate
        write_wav_mono(trial_dir / "input.wav", input_pcm, args.sample_rate)

        asr_text = None
        asr_wall = 0.0
        if args.system == "cascade":
            assert asr is not None and llm is not None
            asr_text, asr_wall = asr.transcribe(input_pcm, args.sample_rate)
            response_text, llm_wall = llm.respond(asr_text, COUNSELOR_SYSTEM_PROMPT)
        else:
            assert speechllm is not None
            response_text, llm_wall = speechllm.respond(
                input_pcm, args.sample_rate, COUNSELOR_SYSTEM_PROMPT
            )

        tts_started = time.perf_counter()
        response_pcm = synthesize(
            response_text or "…", args.sample_rate, args.tts_speed, tts_backend,
            tts_engine, args.tts_speaker,
        ) if response_text.strip() else np.zeros(1, dtype=np.float32)
        tts_wall = time.perf_counter() - tts_started
        output_audio_sec = len(response_pcm) / args.sample_rate
        write_wav_mono(trial_dir / "response.wav", response_pcm, args.sample_rate)

        (trial_dir / "transcript.txt").write_text(response_text, encoding="utf-8")
        with (trial_dir / "transcript.jsonl").open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"step": 0, "time_sec": asr_wall + llm_wall, "piece": response_text},
                    ensure_ascii=False,
                )
                + "\n"
            )

        first_response_latency_sec = round(asr_wall + llm_wall, 4)
        total_wall = asr_wall + llm_wall + tts_wall
        meta = {
            "input_path": str((trial_dir / "input.wav").resolve()),
            "prompt_text": user_text,
            "input_duration_sec": round(input_duration_sec, 4),
            "silence_sec": 0.0,
            "seed": 0,
            "model_repo": model_id,
            "dtype": args.dtype,
            "device": args.device,
            "temp": args.llm_temperature if args.system == "cascade" else float("nan"),
            "temp_text": float("nan"),
            "cfg_coef": float("nan"),
            "sample_rate": args.sample_rate,
            # Cascade/SpeechLLM never stream: audio only starts once ASR (if
            # any) + LLM + TTS have all finished, so "first text" and
            # "audible" both land after the full processing wall time --
            # unlike Moshi, which streams text almost immediately and only
            # waits out a small acoustic delay for audio.
            "first_response_latency_sec": first_response_latency_sec,
            "audible_response_start_sec": round(input_duration_sec + total_wall, 4),
            "audible_start_after_input_sec": round(total_wall, 4),
            "wall_time_sec": round(total_wall, 4),
            "output_audio_sec": round(output_audio_sec, 4),
            "system": args.system,
            "asr_transcript": asr_text,
            "asr_wall_time_sec": round(asr_wall, 4),
            "llm_wall_time_sec": round(llm_wall, 4),
            "tts_wall_time_sec": round(tts_wall, 4),
        }
        (trial_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[local-baseline] {index}/{len(scenarios)} {case_id} "
            f"total={total_wall:.2f}s (asr={asr_wall:.2f}s llm={llm_wall:.2f}s tts={tts_wall:.2f}s) "
            f"-> {response_text[:40]!r}"
        )

    print(f"[local-baseline] wrote {len(scenarios)} case(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
