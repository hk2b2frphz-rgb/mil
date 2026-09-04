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
    parser.add_argument(
        "--listening-ratio",
        type=float,
        default=0.7,
        help=(
            "Fraction of cards that are free-form listening/chat dialogues "
            "(duplex_task=null). The rest are spread over the labeled tasks. "
            "Free-form cards receive frequent backchannels via the timing "
            "enrichment pass."
        ),
    )
    parser.add_argument(
        "--content-seeds",
        type=Path,
        default=None,
        help=(
            "Optional JSONL talking-point bank from build_content_seeds.py. "
            "When set, 1-2 concrete talking points are sampled into each card "
            "to diversify the actual spoken content."
        ),
    )
    parser.add_argument(
        "--seeds-per-card",
        type=int,
        default=2,
        help="Max talking points injected per card when --content-seeds is set.",
    )
    args = parser.parse_args()

    if args.num <= 0:
        parser.error("--num must be positive")
    if not 0.0 <= args.listening_ratio <= 1.0:
        parser.error("--listening-ratio must be between 0 and 1")
    try:
        selected_tasks = parse_tasks(args.tasks)
    except ValueError as exc:
        parser.error(str(exc))

    content_seeds: list[str] = []
    if args.content_seeds is not None:
        if not args.content_seeds.is_file():
            parser.error(f"--content-seeds file not found: {args.content_seeds}")
        for line in args.content_seeds.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(row.get("text") if isinstance(row, dict) else row).strip()
            if text:
                content_seeds.append(text)
        if not content_seeds:
            parser.error(f"--content-seeds file had no usable seeds: {args.content_seeds}")

    rng = random.Random(args.seed)

    num_listening = int(round(args.num * args.listening_ratio))
    num_tasked = args.num - num_listening

    # Labeled-task cards: cycle the selected tasks so each is well represented.
    tasked_order: list[str] = []
    while len(tasked_order) < num_tasked:
        cycle = list(selected_tasks)
        rng.shuffle(cycle)
        tasked_order.extend(cycle)
    tasked_order = tasked_order[:num_tasked]

    # "listening" sentinel marks free-form chat/listening cards.
    plan: list[str] = ["listening"] * num_listening + tasked_order
    rng.shuffle(plan)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"listening": 0}
    with args.out_path.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(plan, start=1):
            row = asdict(build_one(rng, index))
            row["source_profile"] = "full_duplex_training_ja_v2"
            if content_seeds and args.seeds_per_card > 0:
                k = min(args.seeds_per_card, len(content_seeds))
                row["talking_points"] = rng.sample(content_seeds, k=rng.randint(1, k))
            if task == "listening":
                # Free-form listening/chat. No duplex_task: the renderer treats
                # it as a normal dialogue and the enrichment pass adds aizuchi.
                row["id"] = f"chat_{row.get('conversation_type', 'smalltalk')}_{index:05d}"
                row["duplex_task"] = None
                row["duplex_guidance"] = (
                    "自然な雑談・傾聴の対話。相談に限らず、相手の話したいことに沿って"
                    "相づちと短い問い返しで会話を続ける。"
                )
                counts["listening"] += 1
            else:
                row["id"] = f"fd_{task}_{index:05d}"
                row["duplex_task"] = task
                row["duplex_guidance"] = TASK_GUIDANCE[task]
                row["target_turns"] = max(int(row.get("target_turns", 8)), 8)
                if task == "pause_handling":
                    row["silence_pattern"] = "occasional"
                elif task != "smooth_turn_taking":
                    row["silence_pattern"] = "none"
                counts[task] = counts.get(task, 0) + 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {args.num} full-duplex training use cases to {args.out_path}")
    print(f"card counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
