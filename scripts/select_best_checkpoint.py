#!/usr/bin/env python3
"""Pick the checkpoint at the step with the lowest recorded eval loss.

Two training pipelines write eval metrics differently, so this script has two
modes:

  lora    kyutai moshi-finetune (scripts/run_experiment.sh). Eval points come
          from ``<run_dir>/metrics.eval.jsonl`` (one JSON object per eval,
          written by finetune.monitoring.metrics_logger.MetricsLogger).
          Checkpoints live at
          ``<checkpoints_dir>/checkpoint_<step:06d>/<checkpoint-filename>``.

  fullft  nu-dialogue moshi-finetune (scripts/run_nu_fullft_experiment.sh).
          Eval points come from ``MILTO_METRICS {...}`` JSON lines in the
          training stdout log (split="eval"). Checkpoints live at
          ``<checkpoints_dir>/step_<N>/<tag>/``.

Only steps that have BOTH a recorded eval metric AND a checkpoint still on
disk are eligible -- a checkpoint can't be fetched if it was already deleted
(kyutai's ``num_ckpt_keep`` prunes older checkpoints by creation time). If the
step with the single best eval metric has no surviving checkpoint, this
prints a warning to stderr naming that step and falls back to the best
metric among steps that do have one.

Usage:
    uv run python scripts/select_best_checkpoint.py --mode lora \\
        --metrics-jsonl experiments/exp001/checkpoints/<ts>/metrics.eval.jsonl \\
        --checkpoints-dir experiments/exp001/checkpoints/<ts>/checkpoints \\
        --checkpoint-filename consolidated/lora.safetensors \\
        --output-json best.json

    uv run python scripts/select_best_checkpoint.py --mode fullft \\
        --log-file experiments/_fullft_sweeps/<run>/run_nu_<ts>.log \\
        --checkpoints-dir experiments/_fullft_sweeps/<run>/checkpoints/nu_<ts> \\
        --output-json best.json

Prints a JSON object to stdout (and optionally --output-json):
    {"step": 300, "metric_key": "eval_loss", "metric_value": 1.234,
     "checkpoint_path": "...", "num_eval_points": 10,
     "num_checkpoints_available": 5}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MILTO_METRICS_RE = re.compile(r"MILTO_METRICS\s+(?P<payload>\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["lora", "fullft"])
    parser.add_argument("--checkpoints-dir", required=True, type=Path)
    parser.add_argument("--metric-key", default=None,
                        help="Eval metric to minimize. Default: 'eval_loss' (lora) / 'loss/total' (fullft).")
    parser.add_argument("--output-json", type=Path, default=None)

    # lora mode
    parser.add_argument("--metrics-jsonl", type=Path,
                        help="[lora] Path to metrics.eval.jsonl")
    parser.add_argument("--checkpoint-filename", default="consolidated/lora.safetensors",
                        help="[lora] Filename inside checkpoint_<step>/ to require as evidence the checkpoint exists")

    # fullft mode
    parser.add_argument("--log-file", type=Path,
                        help="[fullft] Path to the training stdout log containing MILTO_METRICS lines")
    parser.add_argument("--tag", default="pytorch_model",
                        help="[fullft] DeepSpeed checkpoint tag subdirectory to require as evidence the checkpoint exists")

    return parser.parse_args()


def load_lora_eval_points(metrics_jsonl: Path, metric_key: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    if not metrics_jsonl.exists():
        return points
    for lineno, line in enumerate(metrics_jsonl.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"WARNING: skipping malformed line {metrics_jsonl}:{lineno}: {exc}", file=sys.stderr)
            continue
        if metric_key not in row or "step" not in row:
            continue
        value = row[metric_key]
        if not isinstance(value, (int, float)):
            continue
        points.append((int(row["step"]), float(value)))
    return points


def load_fullft_eval_points(log_file: Path, metric_key: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    if not log_file.exists():
        return points
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MILTO_METRICS_RE.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        if payload.get("split") != "eval":
            continue
        metrics = payload.get("metrics") or {}
        value = metrics.get(metric_key)
        if not isinstance(value, (int, float)):
            continue
        points.append((int(payload["step"]), float(value)))
    return points


def lora_checkpoint_path(checkpoints_dir: Path, step: int, filename: str) -> Path:
    return checkpoints_dir / f"checkpoint_{step:06d}" / filename


def fullft_checkpoint_path(checkpoints_dir: Path, step: int) -> Path:
    return checkpoints_dir / f"step_{step}"


def main() -> None:
    args = parse_args()

    if args.mode == "lora":
        if not args.metrics_jsonl:
            print("ERROR: --metrics-jsonl is required for --mode lora", file=sys.stderr)
            sys.exit(2)
        metric_key = args.metric_key or "eval_loss"
        eval_points = load_lora_eval_points(args.metrics_jsonl, metric_key)

        def ckpt_path_for(step: int) -> Path:
            return lora_checkpoint_path(args.checkpoints_dir, step, args.checkpoint_filename)

        def ckpt_exists(step: int) -> bool:
            return ckpt_path_for(step).is_file()

    else:
        if not args.log_file:
            print("ERROR: --log-file is required for --mode fullft", file=sys.stderr)
            sys.exit(2)
        metric_key = args.metric_key or "loss/total"
        eval_points = load_fullft_eval_points(args.log_file, metric_key)

        def ckpt_path_for(step: int) -> Path:
            return fullft_checkpoint_path(args.checkpoints_dir, step)

        def ckpt_exists(step: int) -> bool:
            step_dir = ckpt_path_for(step)
            tag_dir = step_dir / args.tag
            return tag_dir.is_dir() and (
                any(tag_dir.glob("*zero_pp_rank*")) or any(tag_dir.glob("*model_states*"))
            )

    if not eval_points:
        print(f"ERROR: no eval points found with metric key {metric_key!r}", file=sys.stderr)
        sys.exit(1)

    # last write wins per step (metrics.eval.jsonl can contain repeats on resume)
    by_step: dict[int, float] = {}
    for step, value in eval_points:
        by_step[step] = value

    global_best_step = min(by_step, key=lambda s: by_step[s])

    eligible = {step: value for step, value in by_step.items() if ckpt_exists(step)}
    if not eligible:
        print(
            f"ERROR: {len(by_step)} eval point(s) recorded but no surviving checkpoint "
            f"matches any of them under {args.checkpoints_dir} "
            "(they were likely pruned by num_ckpt_keep). "
            "Re-run with a larger num_ckpt_keep / HP_NUM_CKPT_KEEP to keep more checkpoints.",
            file=sys.stderr,
        )
        sys.exit(1)

    best_step = min(eligible, key=lambda s: eligible[s])
    if best_step != global_best_step:
        print(
            f"WARNING: lowest {metric_key} was at step {global_best_step} "
            f"({by_step[global_best_step]:.6g}) but its checkpoint no longer exists; "
            f"falling back to the best surviving checkpoint at step {best_step} "
            f"({eligible[best_step]:.6g}).",
            file=sys.stderr,
        )

    result = {
        "step": best_step,
        "metric_key": metric_key,
        "metric_value": eligible[best_step],
        "checkpoint_path": str(ckpt_path_for(best_step)),
        "num_eval_points": len(by_step),
        "num_checkpoints_available": len(eligible),
        "global_best_step": global_best_step,
        "global_best_value": by_step[global_best_step],
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
