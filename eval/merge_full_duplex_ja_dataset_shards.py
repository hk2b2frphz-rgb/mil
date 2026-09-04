#!/usr/bin/env python3
"""Merge disjoint builder shards into one normal FDB-JA dataset directory."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path, action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenario_order = [(row["task"], row["id"]) for row in jsonl(args.scenarios)]
    manifests = [
        json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
        for directory in args.shard_dir
    ]
    if not manifests:
        raise SystemExit("At least one --shard-dir is required.")
    expected_shards = manifests[0].get("shard", {}).get("count")
    if expected_shards != len(manifests):
        raise SystemExit(f"Expected {expected_shards} shard manifests, got {len(manifests)}.")
    reference_fingerprint = manifests[0].get("input_fingerprint")
    if not reference_fingerprint:
        raise SystemExit("Shard manifests must contain input_fingerprint; rebuild shards.")

    samples: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for manifest, directory in zip(manifests, args.shard_dir, strict=True):
        shard = manifest.get("shard", {})
        if shard.get("count") != expected_shards:
            raise SystemExit("Shard manifests disagree on shard count.")
        if manifest.get("input_fingerprint") != reference_fingerprint:
            raise SystemExit("Shard manifests disagree on benchmark input fingerprint.")
        for sample in manifest["samples"]:
            key = (sample["task"], sample["id"])
            if key in samples:
                raise SystemExit(f"Duplicate sample across shards: {key}")
            samples[key] = (sample, directory)
    if set(samples) != set(scenario_order):
        raise SystemExit("Shard samples do not exactly match the scenario file.")

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    ordered_samples = []
    for key in scenario_order:
        sample, source_dir = samples[key]
        source = source_dir / sample["path"]
        destination = out_dir / sample["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        ordered_samples.append(sample)

    merged = dict(manifests[0])
    merged.pop("shard", None)
    merged["count"] = len(ordered_samples)
    merged["samples"] = ordered_samples
    (out_dir / "manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ja-data] merged {len(ordered_samples)} scenarios -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
