#!/usr/bin/env python3
"""Build independent Japanese full-duplex training use cases.

The fixed cases under eval_sets/full_duplex_ja are evaluation inputs.  This
script instead expands the existing loneliness-support use-case generator with
full-duplex behavior instructions so training content can be generated without
copying evaluation utterances.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

try:
    from scripts.build_use_cases import build_one
except ModuleNotFoundError:
    from build_use_cases import build_one


TASKS = [
    "pause_handling",
    "smooth_turn_taking",
    "backchannel",
    "user_interruption",
    "user_backchannel",
    "talking_to_other",
    "background_speech",
]

TASK_GUIDANCE = {
    "pause_handling": (
        "userの一つの発話を前半と後半に分け、その間に2〜3秒のsilenceを置く。"
        "moshiは続きが終わるまで長い返答を始めない。"
    ),
    "smooth_turn_taking": (
        "通常の交互対話にし、userの発話終了後にmoshiが自然に応答する。"
        "重なりイベントは入れない。"
    ),
    "backchannel": (
        "長めのuser発話中に、moshiの短い相槌をoverlap_previousで1回以上入れる。"
        "相槌でuserの話題を奪わず、その後もuserが話を続ける。"
    ),
    "user_interruption": (
        "moshiのやや長い発話中にuserが訂正または重要情報を割り込ませる。"
        "user割り込みはoverlap_previousにし、moshi側を短い遅延で打ち切った後、"
        "moshiが新しい情報へ応答する。"
    ),
    "user_backchannel": (
        "moshiの説明中にuserが「はい」「うん」など短い相槌を重ねる。"
        "moshiは打ち切らず、同じ話題を自然に続ける。"
    ),
    "talking_to_other": (
        "moshiの発話中に、userが近くの人へ向けた短い発話を重ねる。"
        "moshiは自分への依頼と誤認せず、元の相談を維持する。"
    ),
    "background_speech": (
        "moshiの発話中にニュースや駅案内などの背景発話を低音量で重ねる。"
        "moshiは背景内容へ反応せず、元の相談を維持する。"
    ),
}


def parse_tasks(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(TASKS)
    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; choices={TASKS}")
    if not tasks:
        raise ValueError("--tasks selected no tasks")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build use cases for Japanese full-duplex training-data generation."
    )
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument("--num", type=int, default=140)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()

    if args.num <= 0:
        parser.error("--num must be positive")
    try:
        selected_tasks = parse_tasks(args.tasks)
    except ValueError as exc:
        parser.error(str(exc))

    rng = random.Random(args.seed)
    task_order: list[str] = []
    while len(task_order) < args.num:
        cycle = list(selected_tasks)
        rng.shuffle(cycle)
        task_order.extend(cycle)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(task_order[: args.num], start=1):
            row = asdict(build_one(rng, index))
            row["id"] = f"fd_{task}_{index:04d}"
            row["source_profile"] = "full_duplex_training_ja_v1"
            row["duplex_task"] = task
            row["duplex_guidance"] = TASK_GUIDANCE[task]
            row["target_turns"] = max(int(row.get("target_turns", 8)), 8)
            if task == "pause_handling":
                row["silence_pattern"] = "occasional"
            elif task != "smooth_turn_taking":
                row["silence_pattern"] = "none"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {task: task_order[: args.num].count(task) for task in selected_tasks}
    print(f"wrote {args.num} full-duplex training use cases to {args.out_path}")
    print(f"task counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
