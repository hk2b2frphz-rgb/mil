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
    parser.add_argument(
        "--status-file",
        type=Path,
        help="Optional batch_status.jsonl written by run_full_duplex_eval_batch.sh.",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_statuses(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


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
    statuses = load_statuses(args.status_file)
    if not summary_paths and not statuses:
        raise SystemExit(f"No benchmark_results/summary.json found under {batch_dir}")

    summaries: dict[str, dict[str, Any]] = {}
    for path in summary_paths:
        model_id = path.parent.parent.name
        summaries[model_id] = load_json(path)

    status_by_output = {str(item["output_name"]): item for item in statuses}
    if statuses:
        successful_names = {
            name for name, item in status_by_output.items() if item.get("status") == "ok"
        }
        failed_names = set(status_by_output) - successful_names
        models = {
            name: summary for name, summary in summaries.items() if name in successful_names
        }
        partial_models = {
            name: summary for name, summary in summaries.items() if name in failed_names
        }
    else:
        models = summaries
        partial_models = {}

    failed_statuses = [item for item in statuses if item.get("status") != "ok"]
    missing_summary_names = [
        str(item["output_name"])
        for item in statuses
        if str(item["output_name"]) not in summaries
    ]

    result = {
        "batch_dir": str(batch_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_models_requested": len(statuses) if statuses else len(summaries),
        "n_models": len(models),
        "n_models_failed": len(failed_statuses),
        "models": models,
        "partial_models": partial_models,
        "failures": {
            "count": len(failed_statuses),
            "model_ids": [str(item["model_id"]) for item in failed_statuses],
            "output_names": [str(item["output_name"]) for item in failed_statuses],
            "missing_summary_count": len(missing_summary_names),
            "missing_summary_output_names": missing_summary_names,
            "details": failed_statuses,
        },
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
