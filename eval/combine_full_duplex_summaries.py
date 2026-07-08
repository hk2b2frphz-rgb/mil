#!/usr/bin/env python3
"""Combine per-model Full-Duplex-Bench-JA summary.json files from a batch run
into one side-by-side comparison file.

Expects the layout produced by scripts/run_full_duplex_eval_batch.sh:
    <batch-dir>/<output_name>/benchmark_results/summary.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-model Full-Duplex-Bench-JA summaries into one comparison file."
    )
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_comparison(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for model_id, summary in models.items():
        for task, task_summary in (summary.get("tasks") or {}).items():
            task_cmp = comparison.setdefault(task, {"means": {}, "rates": {}})
            for metric, value in (task_summary.get("means") or {}).items():
                task_cmp["means"].setdefault(metric, {})[model_id] = value
            for metric, value in (task_summary.get("rates") or {}).items():
                task_cmp["rates"].setdefault(metric, {})[model_id] = value
    return comparison


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    summary_paths = sorted(batch_dir.glob("*/benchmark_results/summary.json"))
    if not summary_paths:
        raise SystemExit(f"No benchmark_results/summary.json found under {batch_dir}")

    models: dict[str, dict[str, Any]] = {}
    for path in summary_paths:
        model_id = path.parent.parent.name
        summary = load_json(path)
        models[model_id] = summary

    result = {
        "batch_dir": str(batch_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_models": len(models),
        "models": models,
        # Per-task, per-metric model_id -> value, for reading a single number
        # (e.g. TOR mean or greeting_matched rate) across every model at once
        # instead of opening each model's summary.json separately.
        "comparison": build_comparison(models),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[fdb-combine] combined {len(models)} model(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
