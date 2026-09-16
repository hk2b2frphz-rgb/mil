#!/usr/bin/env python3
"""Sync nu-dialogue/moshi-finetune stdout metrics to MLflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(
    r"Steps:\s*(?P<step>\d+),\s*"
    r"LRs:\s*\{(?P<lrs>[^}]*)\},\s*"
    r"Loss:\s*(?P<loss>[-+0-9.eE]+)\s*"
    r"\(text:\s*(?P<text>[-+0-9.eE]+),\s*audio:\s*(?P<audio>[-+0-9.eE]+)\)"
)
LR_RE = re.compile(r"'(?P<name>tempformer|depformer)':\s*'(?P<value>[-+0-9.eE]+)'")
MILTO_METRICS_RE = re.compile(r"MILTO_METRICS\s+(?P<payload>\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync nu-dialogue training log to MLflow")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "default"))
    parser.add_argument("--run-name", default=os.environ.get("MLFLOW_RUN_NAME"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--artifact-root", default=os.environ.get("MLFLOW_ARTIFACT_ROOT"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id-file", type=Path, default=None)
    return parser.parse_args()


def ensure_sqlite_parent(tracking_uri: str | None) -> None:
    if not tracking_uri or not tracking_uri.startswith("sqlite:///"):
        return
    db_path = Path(tracking_uri.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(name, item))
        return out
    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False)}
    return {prefix: value}


def mlflow_metric_name(split: str, raw_key: str) -> str:
    if raw_key.startswith("learning_rate/"):
        return raw_key.replace("/", ".")
    return f"{split}.{raw_key.replace('/', '.')}"


def add_metric_aliases(split: str, raw_key: str, value: float, metrics: dict[str, float]) -> None:
    aliases = {
        "loss/total": f"{split}.loss",
        "loss/text_total": f"{split}.loss.text",
        "loss/audio_total": f"{split}.loss.audio",
    }
    alias = aliases.get(raw_key)
    if alias:
        metrics[alias] = value


def parse_miltoka_metrics(lines: list[str]) -> tuple[list[tuple[int, dict[str, float]]], set[tuple[str, int]]]:
    events: list[tuple[int, dict[str, float]]] = []
    seen: set[tuple[str, int]] = set()
    for line in lines:
        match = MILTO_METRICS_RE.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError as exc:
            print(f"WARNING: could not parse MILTO_METRICS line: {exc}", file=sys.stderr)
            continue
        split = str(payload.get("split", "")).strip()
        if split not in {"train", "eval"}:
            continue
        step = int(payload.get("step", 0))
        raw_metrics = payload.get("metrics") or {}
        if not isinstance(raw_metrics, dict):
            continue
        metrics: dict[str, float] = {}
        if isinstance(payload.get("epoch"), (int, float)):
            metrics[f"{split}.epoch"] = float(payload["epoch"])
        for raw_key, raw_value in raw_metrics.items():
            if not isinstance(raw_value, (int, float)):
                continue
            value = float(raw_value)
            metrics[mlflow_metric_name(split, str(raw_key))] = value
            add_metric_aliases(split, str(raw_key), value, metrics)
        if metrics:
            events.append((step, metrics))
            seen.add((split, step))
    return events, seen


def iter_training_metrics(log_file: Path):
    if not log_file.exists():
        return
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    miltoka_events, miltoka_seen = parse_miltoka_metrics(lines)
    for step, metrics in miltoka_events:
        yield step, metrics
    for line in lines:
        match = TRAIN_RE.search(line)
        if not match:
            continue
        step = int(match.group("step"))
        if ("train", step) in miltoka_seen:
            continue
        metrics = {
            "train.loss": float(match.group("loss")),
            "train.loss.text": float(match.group("text")),
            "train.loss.audio": float(match.group("audio")),
        }
        for lr_match in LR_RE.finditer(match.group("lrs")):
            metrics[f"learning_rate.{lr_match.group('name')}"] = float(lr_match.group("value"))
        yield step, metrics


def safe_log_param(mlflow: Any, key: str, value: Any) -> None:
    if value is None:
        value = "null"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (str, int, float, bool)):
        value = str(value)
    if isinstance(value, str) and len(value) > 500:
        value = value[:497] + "..."
    try:
        mlflow.log_param(key, value)
    except Exception as exc:  # MLflow rejects changed param values on repeat syncs.
        print(f"WARNING: could not log param {key}: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow

    output_dir = args.output_dir.resolve()
    log_file = args.log_file.resolve()
    ensure_sqlite_parent(args.tracking_uri)
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        mlflow.create_experiment(args.experiment, artifact_location=args.artifact_root)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or output_dir.name
    run_id = None
    if args.run_id_file and args.run_id_file.exists():
        run_id = args.run_id_file.read_text(encoding="utf-8").strip() or None

    with mlflow.start_run(run_id=run_id, run_name=run_name) as active_run:
        if args.run_id_file and not run_id:
            args.run_id_file.parent.mkdir(parents=True, exist_ok=True)
            args.run_id_file.write_text(active_run.info.run_id + "\n", encoding="utf-8")

        safe_log_param(mlflow, "nu_output_dir", str(output_dir))
        safe_log_param(mlflow, "nu_log_file", str(log_file))

        if args.config and args.config.exists():
            try:
                config = json.loads(args.config.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
            for key, value in flatten("", config).items():
                safe_log_param(mlflow, key, value)
            mlflow.log_artifact(str(args.config), artifact_path="config")

        for step, metrics in iter_training_metrics(log_file) or []:
            mlflow.log_metrics(metrics, step=step)

        if log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="logs")
        exp_dir = output_dir.parent.parent
        if exp_dir.exists():
            for health_json in sorted(exp_dir.glob("nu_dataset_health_*.json")):
                mlflow.log_artifact(str(health_json), artifact_path="dataset_health")
        nu_config = output_dir / "config.json"
        if nu_config.exists():
            mlflow.log_artifact(str(nu_config), artifact_path="config")


if __name__ == "__main__":
    main()
