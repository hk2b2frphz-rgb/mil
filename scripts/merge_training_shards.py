#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge per-shard training_set manifests into one dataset manifest.

Each shard produced by run_full_duplex_training_data_100h.pbs writes its own
  <batch>/shard_<i>/training_set/synthetic_moshi_train.jsonl
plus data_stereo/*.wav and *.json sidecars next to the wavs.

This script concatenates every shard manifest into a single manifest whose
"path" entries are ABSOLUTE paths to the existing wavs. No audio is copied:
prepare_nu_fullft_dataset.py resolves each path with (manifest.parent / path),
and pathlib keeps an absolute right-hand side as-is, so the merged manifest can
live anywhere.

Usage:
  uv run python scripts/merge_training_shards.py \
      --batch-dir data/runs/<BATCH_ID> \
      --out-dir   data/runs/<BATCH_ID>/merged
The fine-tuning launchers then take SRC_RUN_DIR=data/runs/<BATCH_ID>/merged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge full-duplex training shards.")
    parser.add_argument(
        "--batch-dir",
        required=True,
        type=Path,
        help="Directory containing shard_* subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory; a training_set/ manifest is written here.",
    )
    parser.add_argument(
        "--manifest-name",
        default="synthetic_moshi_train.jsonl",
    )
    parser.add_argument(
        "--shard-glob",
        default="shard_*",
        help="Glob (relative to --batch-dir) selecting shard directories.",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=None,
        help="If set, fail unless exactly this many shard directories are found.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow merging despite missing shard manifests, missing wavs, or "
            "empty shards. Without this, any such problem makes the merge fail."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    out_training = (args.out_dir / "training_set").resolve()
    out_training.mkdir(parents=True, exist_ok=True)

    merged_manifest = out_training / args.manifest_name
    merged_dialogues = out_training / "dialogues.jsonl"

    shard_dirs = sorted(p for p in batch_dir.glob(args.shard_glob) if p.is_dir())
    if not shard_dirs:
        raise SystemExit(f"No shard directories matched {args.shard_glob} under {batch_dir}")

    total_rows = 0
    total_dur = 0.0
    missing = 0
    missing_manifests: list[str] = []
    empty_shards: list[str] = []
    seen_stems: set[str] = set()
    shard_summary: list[dict[str, object]] = []

    with merged_manifest.open("w", encoding="utf-8") as mf, \
            merged_dialogues.open("w", encoding="utf-8") as df:
        for shard in shard_dirs:
            training = shard / "training_set"
            manifest = training / args.manifest_name
            if not manifest.is_file():
                print(f"[merge] skip (no manifest): {shard.name}")
                missing_manifests.append(shard.name)
                continue
            shard_rows = 0
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                rel = str(row.get("path") or "")
                if not rel:
                    continue
                wav = (training / rel).resolve()
                if not wav.exists():
                    missing += 1
                    continue
                # Deduplicate by wav stem so accidental cross-shard collisions
                # do not enter the manifest twice.
                stem = wav.stem
                if stem in seen_stems:
                    stem_unique = f"{shard.name}__{stem}"
                else:
                    stem_unique = stem
                seen_stems.add(stem_unique)
                dur = float(row.get("duration") or 0.0)
                total_dur += dur
                mf.write(
                    json.dumps(
                        {"path": str(wav), "duration": dur},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                shard_rows += 1
                total_rows += 1

            dialogues = training / "dialogues.jsonl"
            if dialogues.is_file():
                for line in dialogues.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        df.write(line + "\n")
            if shard_rows == 0:
                empty_shards.append(shard.name)
            shard_summary.append({"shard": shard.name, "samples": shard_rows})
            print(f"[merge] {shard.name}: {shard_rows} samples")

    summary = {
        "batch_dir": str(batch_dir),
        "shards": shard_summary,
        "total_samples": total_rows,
        "total_audio_hours": round(total_dur / 3600.0, 3),
        "missing_wavs": missing,
        "missing_manifests": missing_manifests,
        "empty_shards": empty_shards,
        "manifest": str(merged_manifest),
    }
    (args.out_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[merge] wrote {total_rows} samples "
        f"({summary['total_audio_hours']} h) -> {merged_manifest}"
    )

    problems: list[str] = []
    if total_rows == 0:
        problems.append("no samples were merged")
    if missing:
        problems.append(f"{missing} manifest rows referenced missing wavs")
    if missing_manifests:
        problems.append(f"{len(missing_manifests)} shards had no manifest: {missing_manifests}")
    if empty_shards:
        problems.append(f"{len(empty_shards)} shards contributed 0 samples: {empty_shards}")
    if args.expected_shards is not None and len(shard_dirs) != args.expected_shards:
        problems.append(
            f"found {len(shard_dirs)} shard dirs, expected {args.expected_shards}"
        )

    if problems:
        prefix = "[merge] WARNING (--allow-partial): " if args.allow_partial else "[merge] ERROR: "
        for p in problems:
            print(prefix + p)
        if not args.allow_partial:
            print("[merge] Re-run failed shards, or pass --allow-partial to merge anyway.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
