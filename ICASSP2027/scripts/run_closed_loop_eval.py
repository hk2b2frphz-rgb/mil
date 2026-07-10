#!/usr/bin/env python3
"""Closed-loop clarification evaluation for Moshi-family models.

For every benchmark case x seed:
  * stream the (corrupted) user request into the model frame-by-frame,
  * detect the model's response turn from its own text-token stream,
  * if it asks for clarification, inject the pre-synthesized repair turn,
  * record audio/text/events, score the trial, and pack it for the
    offline LLM judge.

Sharding: --shard K --num-shards N processes every Nth case (round robin)
so a PBS array job splits the benchmark across GPUs with no coordination.

Example (single V100):
  uv run python ICASSP2027/scripts/run_closed_loop_eval.py \
      --dataset-dir ICASSP2027/runs/bench_v1 \
      --out-dir ICASSP2027/runs/eval_base \
      --model-id base --half --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "eval", REPO_ROOT / "ICASSP2027"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clarify.driver import PolicyConfig, run_closed_loop_trial  # noqa: E402
from clarify.judge_pack import pack_trial, write_pack  # noqa: E402
from clarify.metrics import (  # noqa: E402
    aggregate_by_condition, decision_quality, score_trial, write_scores,
)
from clarify.scenario import read_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hf-repo", default="llm-jp/llm-jp-moshi-v1")
    parser.add_argument("--moshi-weight")
    parser.add_argument("--mimi-weight")
    parser.add_argument("--tokenizer")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--temp", type=float)
    parser.add_argument("--temp-text", type=float)
    parser.add_argument("--cfg-coef", type=float)
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Cap cases for smoke tests (0 = all).")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-total-sec", type=float, default=90.0)
    parser.add_argument("--turn-gap-sec", type=float, default=1.2)
    parser.add_argument("--no-response-timeout-sec", type=float, default=8.0)
    parser.add_argument("--save-audio", action="store_true",
                        help="Decode and save model output audio per trial.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        hf_repo=args.hf_repo, moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight, tokenizer=args.tokenizer,
        config=args.config, device=args.device, half=args.half,
        temp=args.temp, temp_text=args.temp_text, cfg_coef=args.cfg_coef,
    )


def load_wav(path: Path, target_sr: int) -> np.ndarray:
    import response_recorder as recorder

    return recorder.load_wav_mono(path, target_sr)


def main() -> int:
    args = parse_args()
    import response_recorder as recorder
    from full_duplex_audio import write_wav_mono, write_wav_stereo

    header, cases = read_manifest(args.dataset_dir)
    if args.conditions != "all":
        wanted = set(args.conditions.split(","))
        cases = [c for c in cases if c.condition in wanted]
    if args.num_shards > 1:
        cases = [c for i, c in enumerate(cases)
                 if i % args.num_shards == args.shard]
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no cases selected")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir: Path = args.out_dir
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = (f"_shard{args.shard}" if args.num_shards > 1 else "")

    print(f"[eval] model={args.model_id} cases={len(cases)} "
          f"seeds={seeds} shard={args.shard}/{args.num_shards}")
    mimi, text_tokenizer, lm_gen, lm_gen_cfg, acoustic_delay = \
        recorder.load_models(model_args(args))
    sample_rate = int(mimi.sample_rate)

    run_config = {
        "model_id": args.model_id, "hf_repo": args.hf_repo,
        "moshi_weight": args.moshi_weight, "dataset": str(args.dataset_dir),
        "dataset_profile": header, "seeds": seeds,
        "conditions": args.conditions,
        "turn_gap_sec": args.turn_gap_sec,
        "no_response_timeout_sec": args.no_response_timeout_sec,
        "shard": args.shard, "num_shards": args.num_shards,
    }
    (out_dir / f"run_config{shard_tag}.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    scores = []
    judge_records = []
    t_start = time.time()
    for idx, case in enumerate(cases):
        base_pcm = load_wav(args.dataset_dir / case.audio_path, sample_rate)
        repair_pcm = load_wav(
            args.dataset_dir / case.repair_audio_path, sample_rate
        )
        utt_end_sec = len(base_pcm) / sample_rate
        cfg = PolicyConfig(
            slot_type=case.base.slot_type,
            slot_value=case.base.slot_value or None,
            utterance_end_sec=utt_end_sec,
            turn_gap_sec=args.turn_gap_sec,
            no_response_timeout_sec=args.no_response_timeout_sec,
        )
        for seed in seeds:
            trial_dir = trials_dir / case.case_id / f"seed_{seed}"
            result_path = trial_dir / "result.json"
            if result_path.exists() and not args.overwrite:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                text_events = result["text_events"]
                policy = result["policy"]
            else:
                trial_dir.mkdir(parents=True, exist_ok=True)
                result = run_closed_loop_trial(
                    base_pcm, repair_pcm, cfg, seed, lm_gen, mimi,
                    text_tokenizer, args.device,
                    max_total_sec=args.max_total_sec,
                )
                text_events = result["text_events"]
                policy = result["policy"]
                if args.save_audio and result["audio_frames"]:
                    decoded = recorder.decode_audio(
                        result["audio_frames"], acoustic_delay, mimi,
                        args.device,
                    )
                    if decoded is not None:
                        write_wav_mono(trial_dir / "output.wav",
                                       decoded, sample_rate)
                        write_wav_stereo(
                            trial_dir / "stereo.wav",
                            result["input_pcm_final"], decoded, sample_rate,
                        )
                persisted = {
                    "case_id": case.case_id, "seed": seed,
                    "utterance_end_sec": utt_end_sec,
                    "text_events": text_events, "policy": policy,
                    "total_steps": result["total_steps"],
                    "first_response_step": result["first_response_step"],
                }
                result_path.write_text(
                    json.dumps(persisted, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            score = score_trial(
                case.to_json(), seed, text_events, policy, utt_end_sec,
                turn_gap_sec=args.turn_gap_sec,
            )
            scores.append(score)
            judge_records.append(pack_trial(
                case.to_json(), seed,
                first_turn_text=score.extra.get("first_text", ""),
                final_turn_text=score.final_text,
                repair_injected=score.repair_injected,
            ))
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"[eval] {idx + 1}/{len(cases)} cases "
                  f"({elapsed / (idx + 1):.1f}s/case)")

    write_scores(out_dir / f"scores{shard_tag}.jsonl", scores)
    write_pack(out_dir / f"judge_pack{shard_tag}.jsonl", judge_records)
    summary = {
        "model_id": args.model_id,
        "n_trials": len(scores),
        "by_condition": aggregate_by_condition(scores),
        "decision_quality": decision_quality(scores),
    }
    (out_dir / f"summary{shard_tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["decision_quality"], ensure_ascii=False,
                     indent=2))
    print(f"[eval] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
