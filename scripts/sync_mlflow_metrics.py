#!/usr/bin/env python3
"""Log moshi-finetune JSONL metrics to MLflow.

This keeps the training code unchanged. moshi-finetune writes
metrics.train.jsonl and metrics.eval.jsonl under run_dir; this script reads
those files and logs params, metrics, and selected artifacts to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync moshi-finetune metrics to MLflow")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "default"))
    parser.add_argument("--run-name", default=os.environ.get("MLFLOW_RUN_NAME"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id-file", type=Path, default=None)
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists():
        return
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            # 切り詰められた行などで全体が止まらないよう、スキップして継続する。
            print(f"WARNING: skipping malformed JSONL at {path}:{lineno}: {exc}",
                  file=sys.stderr)
            continue


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(name, item))
        return out
    return {prefix: value}


def main() -> None:
    args = parse_args()

    import mlflow
    import yaml

    run_dir = args.run_dir.resolve()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or run_dir.name
    run_id = None
    if args.run_id_file and args.run_id_file.exists():
        run_id = args.run_id_file.read_text().strip() or None

    with mlflow.start_run(run_id=run_id, run_name=run_name) as active_run:
        if args.run_id_file and not run_id:
            args.run_id_file.parent.mkdir(parents=True, exist_ok=True)
            args.run_id_file.write_text(active_run.info.run_id + "\n")

        mlflow.log_param("run_dir", str(run_dir))
        if args.config and args.config.exists():
            config = yaml.safe_load(args.config.read_text()) or {}
            for key, value in flatten("", config).items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    mlflow.log_param(key, value)
            mlflow.log_artifact(str(args.config), artifact_path="config")

        for tag in ("train", "eval"):
            metrics_path = run_dir / f"metrics.{tag}.jsonl"
            for row in iter_jsonl(metrics_path) or []:
                step = int(row.get("step", 0))
                metrics: dict[str, float] = {}
                for key, value in row.items():
                    if key in {"step", "at"}:
                        continue
                    if isinstance(value, (int, float)):
                        metrics[f"{tag}.{key}"] = float(value)
                if metrics:
                    mlflow.log_metrics(metrics, step=step)
            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        tb_dir = run_dir / "tb"
        if tb_dir.exists():
            mlflow.log_artifacts(str(tb_dir), artifact_path="tensorboard")


if __name__ == "__main__":
    main()
