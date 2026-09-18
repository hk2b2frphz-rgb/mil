#!/usr/bin/env python3
"""Build a Japanese backchannel-timing distribution for
evaluate_full_duplex_ja.py --backchannel-gt.

No human-annotated Japanese backchannel corpus exists for this project yet,
unlike upstream Full-Duplex-Bench's English icc_gt_distribution.json. This
script mines the timing this project's own training-data pipeline designed
for aizuchi placement instead: every dialogue turn tagged
event=="model_backchannel" records start_after_previous_start_sec, i.e. how
many seconds after the user's turn begins the backchannel is scripted to
land. Those offsets are binned into the same 0.2s windows
evaluate_full_duplex_ja.py uses for model output, producing a distribution
shape comparable at compare-time via that script's length interpolation.

This is a design-target proxy, not ground truth. Treat jsd computed against
it as "distance from this project's own aizuchi placement design", not
"distance from authentic human backchannel timing". Replace it once real
Japanese backchannel-timing data is available, and regenerate whenever the
aizuchi placement design changes (see e.g. the "Diversify aizuchi" /
"energy valley detection" commits).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WINDOW_SEC = 0.2  # matches evaluate_full_duplex_ja.py `window_size`
EPSILON = 1e-10  # matches evaluate_full_duplex_ja.py `epsilon`

DEFAULT_PATTERNS = (
    "data/**/*dialogues*.jsonl",
    "eval_runs/**/*dialogues*.jsonl",
    "tests/fixtures/*dialogues*.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dialogues",
        action="append",
        default=[],
        help="Glob pattern for dialogue JSONL files (repeatable). "
        f"Defaults to: {', '.join(DEFAULT_PATTERNS)}",
    )
    parser.add_argument(
        "--speaker",
        default="Ono_Anna",
        help="Speaker key to write the distribution under (must match tts_speaker).",
    )
    parser.add_argument("--max-offset-sec", type=float, default=6.0)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def collect_offsets(patterns: list[str]) -> list[float]:
    repo_root = Path(__file__).resolve().parents[1]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(repo_root.glob(pattern))
    offsets: list[float] = []
    for path in sorted(paths):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            dialogue = json.loads(line)
            for turn in dialogue.get("turns", []):
                if turn.get("event") != "model_backchannel":
                    continue
                offset = turn.get("start_after_previous_start_sec")
                if isinstance(offset, (int, float)) and offset >= 0:
                    offsets.append(float(offset))
    return offsets


def main() -> int:
    args = parse_args()
    offsets = collect_offsets(args.dialogues or list(DEFAULT_PATTERNS))
    if not offsets:
        raise SystemExit(
            "No event=='model_backchannel' turns found under the searched "
            "dialogue files; cannot build a distribution."
        )
    num_windows = max(1, int(np.ceil(args.max_offset_sec / WINDOW_SEC)))
    distribution = np.full(num_windows, EPSILON, dtype=np.float64)
    dropped = 0
    for offset in offsets:
        if offset >= args.max_offset_sec:
            dropped += 1
            continue
        distribution[int(offset // WINDOW_SEC)] += 1.0
    distribution /= distribution.sum()

    out = {
        args.speaker: distribution.tolist(),
        "_meta": {
            "n_backchannel_events": len(offsets),
            "n_dropped_over_max_offset": dropped,
            "window_sec": WINDOW_SEC,
            "max_offset_sec": args.max_offset_sec,
            "source": (
                "project-generated training dialogue timing design "
                "(event=='model_backchannel' turns), NOT a human-annotated corpus"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[backchannel-gt] {len(offsets)} events ({dropped} dropped, "
        f">= {args.max_offset_sec}s) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
