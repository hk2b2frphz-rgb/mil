#!/usr/bin/env python3
"""Validate LoRA train/eval manifests before starting GPU training."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--eval-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def read_scalar(config: Path, key: str, cast: type) -> Any:
    text = config.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(key)}\s*:\s*([^#\n]+)", text, flags=re.M)
    if not match:
        raise ValueError(f"{config}: missing top-level {key}")
    raw = match.group(1).strip().strip("\"'")
    return cast(raw)


def inspect_manifest(path: Path, duration_sec: float, world_size: int) -> dict[str, Any]:
    errors: list[str] = []
    rows = 0
    estimated_chunks = 0
    total_duration = 0.0
    resolved_paths: set[Path] = set()

    if not path.is_file():
        return {
            "manifest": str(path),
            "rows": 0,
            "estimated_chunks": 0,
            "errors": [f"manifest does not exist: {path}"],
        }

    for lineno, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        rows += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON: {exc}")
            continue

        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            errors.append(f"line {lineno}: missing non-empty 'path'")
            continue
        audio_path = Path(rel_path)
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        audio_path = audio_path.resolve()
        resolved_paths.add(audio_path)
        if not audio_path.is_file():
            errors.append(f"line {lineno}: audio file does not exist: {audio_path}")

        try:
            duration = float(item["duration"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"line {lineno}: missing or invalid positive 'duration'")
            continue
        if not math.isfinite(duration) or duration <= 0:
            errors.append(f"line {lineno}: duration must be finite and > 0, got {duration!r}")
            continue
        total_duration += duration
        estimated_chunks += max(1, math.ceil(duration / duration_sec))

    per_rank_chunks = [
        max(0, (estimated_chunks - rank + world_size - 1) // world_size)
        for rank in range(world_size)
    ]
    if rows <= 0:
        errors.append("manifest contains no records")
    if estimated_chunks <= 0:
        errors.append("manifest produces no chunks")
    if world_size > 1 and min(per_rank_chunks, default=0) <= 0:
        errors.append(
            "at least one distributed rank receives no chunks from this split; "
            f"estimated_chunks={estimated_chunks}, world_size={world_size}"
        )

    return {
        "manifest": str(path.resolve()),
        "rows": rows,
        "unique_audio_paths": len(resolved_paths),
        "total_duration_sec": total_duration,
        "estimated_chunks": estimated_chunks,
        "estimated_chunks_per_rank": per_rank_chunks,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")

    duration_sec = read_scalar(args.config, "duration_sec", float)
    batch_size = read_scalar(args.config, "batch_size", int)
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    summary = {
        "config": str(args.config.resolve()),
        "duration_sec": duration_sec,
        "batch_size": batch_size,
        "world_size": args.world_size,
        "train": inspect_manifest(
            args.train_manifest.resolve(), duration_sec, args.world_size
        ),
        "eval": inspect_manifest(
            args.eval_manifest.resolve(), duration_sec, args.world_size
        ),
    }

    # The loader now flushes partial eval batches, so one valid chunk is enough
    # on a single process. In distributed mode every rank needs work to keep
    # collective model forwards aligned.
    summary["eval"]["estimated_batches_per_rank"] = [
        math.ceil(chunks / batch_size) if chunks else 0
        for chunks in summary["eval"]["estimated_chunks_per_rank"]
    ]

    errors = [
        f"{split}: {error}"
        for split in ("train", "eval")
        for error in summary[split]["errors"]
    ]
    summary["ok"] = not errors
    summary["errors"] = errors

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
