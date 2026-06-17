#!/usr/bin/env python3
"""Validate nu-dialogue full-FT parquet data before training."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check nu full-FT train/eval parquet health")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--min-length", required=True, type=int)
    parser.add_argument("--text-padding-id", type=int, default=3)
    parser.add_argument("--end-of-text-padding-id", type=int, default=0)
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def chunk_count(frame_count: int, max_length: int, min_length: int) -> int:
    if frame_count <= 0:
        return 0
    chunks = np.array_split(np.arange(frame_count), math.ceil(frame_count / max_length))
    return sum(1 for chunk in chunks if len(chunk) >= min_length)


def load_split(data_dir: Path, split: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(data_dir / "processed" / f"{split}-*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def summarize_split(
    data_dir: Path,
    split: str,
    max_length: int,
    min_length: int,
    text_padding_id: int,
    end_of_text_padding_id: int,
) -> dict[str, Any]:
    df = load_split(data_dir, split)
    if df is None:
        return {"present": False}

    frame_lengths: list[int] = []
    main_text_nonpad = 0
    main_text_content = 0
    other_text_nonpad = 0
    other_text_content = 0
    main_empty_text_rows = 0
    post_filter_chunks = 0

    for _, row in df.iterrows():
        main = np.asarray(row["A"], dtype=np.int64)
        other = np.asarray(row["B"], dtype=np.int64)
        if main.ndim != 2 or other.ndim != 2 or main.shape[0] != 9 or other.shape[0] != 9:
            raise ValueError(
                f"{split}: expected A/B shape [9, T], got A={main.shape}, B={other.shape}"
            )
        frames = int(main.shape[1])
        frame_lengths.append(frames)
        post_filter_chunks += chunk_count(frames, max_length, min_length)

        main_nonpad = int(np.count_nonzero(main[0] != text_padding_id))
        other_nonpad = int(np.count_nonzero(other[0] != text_padding_id))
        main_content = int(
            np.count_nonzero((main[0] != text_padding_id) & (main[0] != end_of_text_padding_id))
        )
        other_content = int(
            np.count_nonzero((other[0] != text_padding_id) & (other[0] != end_of_text_padding_id))
        )
        main_text_nonpad += main_nonpad
        other_text_nonpad += other_nonpad
        main_text_content += main_content
        other_text_content += other_content
        if main_nonpad == 0:
            main_empty_text_rows += 1

    frame_array = np.asarray(frame_lengths, dtype=np.int64)
    return {
        "present": True,
        "rows": int(len(df)),
        "post_filter_chunks": int(post_filter_chunks),
        "frames_min": int(frame_array.min()) if len(frame_array) else 0,
        "frames_p50": float(np.percentile(frame_array, 50)) if len(frame_array) else 0.0,
        "frames_p90": float(np.percentile(frame_array, 90)) if len(frame_array) else 0.0,
        "frames_max": int(frame_array.max()) if len(frame_array) else 0,
        "main_text_nonpad": int(main_text_nonpad),
        "main_text_content": int(main_text_content),
        "other_text_nonpad": int(other_text_nonpad),
        "other_text_content": int(other_text_content),
        "main_empty_text_rows": int(main_empty_text_rows),
    }


def validate(summary: dict[str, Any], require_eval: bool) -> list[str]:
    errors: list[str] = []
    for split in ("train", "eval"):
        item = summary.get(split, {})
        if split == "eval" and not require_eval and not item.get("present"):
            continue
        if not item.get("present"):
            errors.append(f"{split}: parquet files are missing")
            continue
        if item["rows"] <= 0:
            errors.append(f"{split}: no rows")
        if item["post_filter_chunks"] <= 0:
            errors.append(
                f"{split}: no chunks survive min_length/max_length "
                f"(min_length={summary['min_length']}, max_length={summary['max_length']})"
            )
        if item["main_text_nonpad"] <= 0:
            errors.append(f"{split}: main speaker A has no non-padding text tokens")
        if item["main_text_content"] <= 0:
            errors.append(f"{split}: main speaker A has no content text tokens")
        if split == "eval" and item.get("rows", 0) < 2:
            errors.append("eval: fewer than 2 rows; metrics will be unstable")
    return errors


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    summary = {
        "data_dir": str(data_dir),
        "max_length": args.max_length,
        "min_length": args.min_length,
        "text_padding_id": args.text_padding_id,
        "end_of_text_padding_id": args.end_of_text_padding_id,
        "train": summarize_split(
            data_dir,
            "train",
            args.max_length,
            args.min_length,
            args.text_padding_id,
            args.end_of_text_padding_id,
        ),
        "eval": summarize_split(
            data_dir,
            "eval",
            args.max_length,
            args.min_length,
            args.text_padding_id,
            args.end_of_text_padding_id,
        ),
    }
    errors = validate(summary, args.require_eval)
    summary["errors"] = errors

    print("[nu-data-check] " + json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    if errors:
        for error in errors:
            print(f"[nu-data-check] ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
