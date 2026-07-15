#!/usr/bin/env python3
"""Cascade (weak ASR + confidence-threshold policy) baseline evaluation.

Sweeps ask thresholds over the same benchmark and writes one scores file
per threshold plus a combined summary, tracing the hit/false-alarm curve
an explicit ASR-confidence signal achieves.

  uv run python ICASSP2027/scripts/run_cascade_eval.py \
      --dataset-dir ICASSP2027/runs/bench_v1 \
      --out-dir ICASSP2027/runs/eval_cascade
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "ICASSP2027"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clarify.cascade import run_cascade  # noqa: E402
from clarify.metrics import (  # noqa: E402
    aggregate_by_condition, decision_quality, write_scores,
)
from clarify.scenario import read_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds",
                        default="-1.2,-1.0,-0.8,-0.6,-0.4,-0.2")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    _header, cases = read_manifest(args.dataset_dir)
    if args.max_cases:
        cases = cases[: args.max_cases]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    case_dicts = [c.to_json() for c in cases]
    print(f"[cascade] {len(cases)} cases, thresholds={thresholds}")

    results = run_cascade(case_dicts, args.dataset_dir, thresholds,
                          model_size=args.asr_model, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined = {}
    for threshold, scores in results.items():
        tag = f"thr{threshold:+.1f}".replace(".", "p")
        write_scores(args.out_dir / f"scores_{tag}.jsonl", scores)
        combined[str(threshold)] = {
            "by_condition": aggregate_by_condition(scores),
            "decision_quality": decision_quality(scores),
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps({"model_id": f"cascade_whisper_{args.asr_model}",
                    "thresholds": combined},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for threshold in thresholds:
        dq = combined[str(threshold)]["decision_quality"]
        print(f"  thr={threshold:+.1f} hit={dq['hit_rate']['rate']} "
              f"fa={dq['false_alarm_rate']['rate']} "
              f"bal={dq['balanced_accuracy']}")
    print(f"[cascade] done: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
