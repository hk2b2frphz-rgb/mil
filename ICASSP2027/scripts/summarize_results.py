#!/usr/bin/env python3
"""Merge shards, aggregate, and compare models.

Reads every `scores*.jsonl` under each given run directory, merges shards,
recomputes aggregates, runs paired bootstrap deltas between models on the
key metrics, and emits `comparison.json` plus a paper-ready markdown table.

  uv run python ICASSP2027/scripts/summarize_results.py \
      --runs base=ICASSP2027/runs/eval_base \
             clarify_full=ICASSP2027/runs/eval_ft_full \
      --out-dir ICASSP2027/runs/comparison_v1
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

from clarify.metrics import (  # noqa: E402
    TrialScore, aggregate_by_condition, decision_quality,
    paired_bootstrap_delta, read_scores,
)

KEY_METRICS = {
    "asked": lambda s: s.asked,
    "slot_correct": lambda s: s.slot_correct,
    "hallucinated_confirmation": lambda s: s.hallucinated_confirmation,
}


def load_run(run_dir: Path) -> list[TrialScore]:
    scores: list[TrialScore] = []
    files = sorted(run_dir.glob("scores*.jsonl"))
    if not files:
        raise SystemExit(f"no scores*.jsonl under {run_dir}")
    seen: set[tuple[str, int]] = set()
    for path in files:
        for s in read_scores(path):
            key = (s.case_id, s.seed)
            if key in seen:
                continue  # shard overlap safety
            seen.add(key)
            scores.append(s)
    return scores


def markdown_table(models: dict[str, list[TrialScore]]) -> str:
    conditions = sorted({s.condition for scores in models.values()
                         for s in scores})
    lines = ["| model | " + " | ".join(conditions) + " |",
             "|---" * (len(conditions) + 1) + "|"]
    for name, scores in models.items():
        agg = aggregate_by_condition(scores)
        cells = []
        for cond in conditions:
            crr = agg.get(cond, {}).get("CRR", {}).get("rate")
            ssr = agg.get(cond, {}).get("SSR", {}).get("rate")
            cells.append(
                f"{crr:.2f}/{ssr:.2f}" if crr is not None else "-"
            )
        lines.append(f"| {name} (CRR/SSR) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="name=path pairs")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--baseline", default=None,
                        help="Model name used as bootstrap reference "
                             "(default: first).")
    args = parser.parse_args()

    models: dict[str, list[TrialScore]] = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--runs entries must be name=path: {spec}")
        models[name] = load_run(Path(path))
        print(f"[summ] {name}: {len(models[name])} trials")

    baseline = args.baseline or next(iter(models))
    comparison: dict[str, object] = {"baseline": baseline, "models": {}}
    for name, scores in models.items():
        entry = {
            "n_trials": len(scores),
            "by_condition": aggregate_by_condition(scores),
            "decision_quality": decision_quality(scores),
        }
        if name != baseline:
            deltas = {}
            for metric_name, fn in KEY_METRICS.items():
                deltas[metric_name] = paired_bootstrap_delta(
                    models[baseline], scores, fn
                )
            entry["delta_vs_baseline"] = deltas
        comparison["models"][name] = entry  # type: ignore[index]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    table = markdown_table(models)
    (args.out_dir / "table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"[summ] wrote {args.out_dir / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
