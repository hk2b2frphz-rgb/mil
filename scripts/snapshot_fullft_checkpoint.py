#!/usr/bin/env python3
"""Copy a completed ZeRO step without changing the training checkpoint."""

import argparse
import math
import shutil
import stat
import time
from pathlib import Path


def inventory(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Checkpoint symlinks are not supported: {path}")
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"Unsupported checkpoint entry: {path}")
        result[path.relative_to(root)] = (
            info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino
        )
    return result


def snapshot(source: Path, destination: Path, out_dir: Path,
             tag: str = "pytorch_model", stable_seconds: float = 30) -> None:
    source, destination, out_dir = (
        p.expanduser().resolve() for p in (source, destination, out_dir)
    )
    if not math.isfinite(stable_seconds) or stable_seconds < 0:
        raise ValueError("stable_seconds must be finite and non-negative")
    if not tag or tag in (".", "..") or "/" in tag or "\\" in tag:
        raise ValueError("tag must be a single directory name")
    if not source.is_dir():
        raise ValueError(f"Checkpoint directory not found: {source}")
    # Protect the entire checkpoint tree, including other steps and latest.
    checkpoint_root = next(
        (p for p in source.parents if p.name == "checkpoints"), source.parent
    )
    intermediate = Path(str(out_dir) + "_ft")
    targets = (destination, out_dir, intermediate)
    for target in targets:
        if target.is_relative_to(checkpoint_root) or checkpoint_root.is_relative_to(target):
            raise ValueError(f"Destination overlaps training checkpoints: {target}")
        if target.exists():
            raise ValueError(f"Destination already exists; refusing overwrite: {target}")
    for i, target in enumerate(targets):
        for other in targets[i + 1:]:
            if target.is_relative_to(other) or other.is_relative_to(target):
                raise ValueError("Snapshot, output and intermediate directories must be separate")

    before = inventory(source)
    for pattern in ("*model_states.pt", "*optim_states.pt"):
        shards = list((source / tag).glob(pattern))
        if not shards or any(p.stat().st_size == 0 for p in shards):
            raise ValueError(f"Missing or empty ZeRO shards: {source / tag / pattern}")
    print(f"[snapshot] Checking stability for {stable_seconds:g}s: {source}", flush=True)
    time.sleep(stable_seconds)
    if inventory(source) != before:
        raise ValueError("Checkpoint is changing; select a completed step and retry")
    print(f"[snapshot] Copying to {destination}", flush=True)
    # copytree/copy2 creates independent files, never hard links to training data.
    shutil.copytree(source, destination)
    if inventory(source) != before:
        raise ValueError("Checkpoint changed during copy; snapshot must not be exported")
    copied = inventory(destination)
    expected_sizes = {p: v[1] for p, v in before.items() if stat.S_ISREG(v[0])}
    copied_sizes = {p: v[1] for p, v in copied.items() if stat.S_ISREG(v[0])}
    if expected_sizes != copied_sizes:
        raise ValueError("Copied checkpoint file sizes do not match the source")
    print(f"[snapshot] Ready: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", default="pytorch_model")
    parser.add_argument("--stable-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        snapshot(args.step_dir, args.snapshot_dir, args.out_dir,
                 args.tag, args.stable_seconds)
    except (ValueError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\nPartial snapshots are retained; use a new destination.\n")


if __name__ == "__main__":
    main()
