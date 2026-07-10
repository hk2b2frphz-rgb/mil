#!/usr/bin/env python3
"""Generate clarify fine-tuning dialogues (one variant per invocation).

Writes `dialogues.jsonl` (input for scripts/generate_qwen3_tts_data.py)
and `corruption_plan.jsonl` (input for corrupt_training_audio.py).

Slot material comes from MASSIVE ja-JP *train* split (benchmark uses
*test*); --massive-jsonl allows offline pre-exported rows; --demo uses a
tiny built-in set for smoke tests.

  uv run python ICASSP2027/scripts/build_training_data.py \
      --variant clarify_full --out-dir ICASSP2027/runs/train_full \
      --num-items 1200 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "ICASSP2027"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from clarify import slots  # noqa: E402
from clarify.train_data import (  # noqa: E402
    VARIANTS, generate_training_dialogues, write_training_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--massive-jsonl", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--num-items", type=int, default=1200,
                        help="Acoustic base items to sample.")
    parser.add_argument("--ask-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.demo:
        sys.path.insert(0, str(REPO_ROOT / "ICASSP2027" / "scripts"))
        from build_benchmark import DEMO_ITEMS
        rows = DEMO_ITEMS
    elif args.massive_jsonl:
        rows = []
        with args.massive_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset

        ds = load_dataset("AmazonScience/massive", "ja-JP", split=args.split)
        names = ds.features["intent"].names
        rows = []
        for row in ds:
            row = dict(row)
            if isinstance(row.get("intent"), int):
                row["intent"] = names[row["intent"]]
            rows.append(row)

    items = slots.select_base_items(rows, max_items=args.num_items)
    under = slots.build_underspecified_items()
    if not items:
        raise SystemExit("no acoustic items selected")
    print(f"[train-data] {len(items)} acoustic items, "
          f"{len(under)} underspecified templates")

    rng = random.Random(args.seed)
    dialogues = generate_training_dialogues(
        items, under, args.variant, rng, ask_ratio=args.ask_ratio
    )
    d_path, p_path = write_training_files(dialogues, args.out_dir)
    behaviors: dict[str, int] = {}
    for d in dialogues:
        behaviors[d.behavior] = behaviors.get(d.behavior, 0) + 1
    (args.out_dir / "generation_config.json").write_text(
        json.dumps({
            "variant": args.variant, "seed": args.seed,
            "num_items": len(items), "ask_ratio": args.ask_ratio,
            "behaviors": behaviors,
            "source": ("demo" if args.demo
                       else str(args.massive_jsonl
                                or f"massive:{args.split}")),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[train-data] {len(dialogues)} dialogues ({behaviors}) -> "
          f"{d_path}, plan -> {p_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
