#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick the handful of dialogues worth listening to before synthesizing 1,000.

Taking the first N dialogues off the top gives you whatever the shuffle put
there, which is usually four two-turn lookups.  That tells you the voice works
and nothing else.  The things that actually go wrong in this domain are the
readings -- 品番, 呼び径, 強度区分の刻印, ピッチの小数 -- and the long
clarification chains where the same fixed question comes back three times.

So this picks on purpose: every flow, and every knowledge row whose answer
carries a number or a part number.

  python screw_poc/scripts/pick_listening_set.py
  python screw_poc/scripts/pick_listening_set.py --count 12 --out /tmp/listen.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

POC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = POC_ROOT / "artifacts" / "train_dialogues.jsonl"
DEFAULT_OUT = POC_ROOT / "artifacts" / "listening_set.jsonl"

# 読みを外しやすい知識を先に。品番、刻印、小数のピッチ、かける表記。
WANTED = (
    ("K18", "clarify"),      # SSH-M10-SD-EL と 27ニュートンメートル
    ("K15", "direct"),       # SSH-M3-SD-EL と 0.7ニュートンメートル
    ("K08", "direct"),       # M8×1 の「かける」
    ("K05", "varied"),       # 1.25ミリ と 確認のやりとり
    ("K14", "varied"),       # 8.8 の刻印
    ("K22", "clarify"),      # 答えられる項目と答えられない項目が混ざる長い筋
    ("K30", "clarify"),      # 訊いても出てこないままエスカレーション
    ("K27", "direct"),       # 作業を止める指示
    ("K01", "unresolved"),   # 三段階に噛み砕いても届かない
    ("K12", "direct"),       # 数字を含まない普通の応答（比較用）
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=len(WANTED))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    picked: list[dict[str, Any]] = []
    missing: list[str] = []
    for knowledge_id, flow in WANTED[: args.count]:
        match = next(
            (
                row
                for row in rows
                if row.get("knowledge_id") == knowledge_id and row.get("flow") == flow
            ),
            None,
        )
        if match is None:
            missing.append(f"{knowledge_id}/{flow}")
            continue
        picked.append(match)
    # 領域外の応答も一本は聴いておく。断り方の口調が変だと現場で角が立つ。
    out_of_scope = next((row for row in rows if row["flow"] == "out_of_scope"), None)
    if out_of_scope is not None and len(picked) < args.count + 1:
        picked.append(out_of_scope)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in picked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if missing:
        print("見つからなかった組み合わせ: " + ", ".join(missing))
    print(f"{len(picked)} 本を {args.out} に書きました")
    for row in picked:
        turns = len(row["turns"])
        print(f"  {row['id']:28s} {row['flow']:11s} {turns:2d}ターン")


if __name__ == "__main__":
    main()
