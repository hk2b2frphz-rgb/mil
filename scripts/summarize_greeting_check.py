#!/usr/bin/env python3
"""固定の冒頭挨拶が実際に出ているかを、評価結果から集計する。

eval/greeting_check.py が per_case.jsonl の各行へ入れる
opening_greeting_check を読み、
  - 一致率(matched の割合)
  - 類似度の分布
  - 実際にモデルが冒頭で何と言ったか(頻度順)
を出す。

「挨拶ができていない」を切り分けるための道具。同じことでも中身が違う。

  A. 冒頭窓に何も出ていない           -> 生成自体が始まっていない
  B. 冒頭窓に別のことを言っている      -> 挨拶が定着していないだけで生成は動く
  C. 似ているが閾値未満                -> 言い回しの揺れ。学習不足寄り

A と B/C では次の手が変わるので、まずここを見る。

使い方:
    uv run python scripts/summarize_greeting_check.py \\
        eval_runs/full_duplex/<run>/benchmark_results/per_case.jsonl

複数渡すと横に並べて比較できる(LoRA と full FT の対比など):
    uv run python scripts/summarize_greeting_check.py \\
        eval_runs/full_duplex/<lora_run>/benchmark_results/per_case.jsonl \\
        eval_runs/full_duplex/<fullft_run>/benchmark_results/per_case.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("per_case", nargs="+", type=Path,
                        help="benchmark_results/per_case.jsonl")
    parser.add_argument("--top", type=int, default=8,
                        help="実際の冒頭発話を頻度順に何件出すか")
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(path: Path, top: int) -> None:
    rows = load(path)
    checks = [
        row["opening_greeting_check"]
        for row in rows
        if isinstance(row.get("opening_greeting_check"), dict)
    ]

    print("=" * 70)
    print(path)
    print(f"  trials: {len(rows)}")

    if not checks:
        print("  挨拶チェックの記録がありません。")
        print("  FDB_OPENING_GREETING=0 で評価した可能性があります。その設定では")
        print("  冒頭挨拶ぶんの無音リードインを確保しないので、挨拶を学習した")
        print("  モデルの評価には使えません。FDB_OPENING_GREETING=1 で取り直して")
        print("  ください。")
        return

    matched = sum(1 for c in checks if c.get("matched"))
    similarities = [float(c.get("similarity", 0.0)) for c in checks]
    empty = sum(1 for c in checks if not str(c.get("actual_text_in_window", "")).strip())

    print(f"  挨拶チェックあり: {len(checks)}")
    print(f"  一致(matched):    {matched}/{len(checks)} = {matched / len(checks):.1%}")
    print(f"  類似度 mean/median/max: "
          f"{statistics.fmean(similarities):.3f} / "
          f"{statistics.median(similarities):.3f} / {max(similarities):.3f}")
    print(f"  冒頭窓が空(何も言っていない): {empty}/{len(checks)} = {empty / len(checks):.1%}")

    expected = next(
        (c.get("expected_text") for c in checks if c.get("expected_text")), None
    )
    print(f"  期待する挨拶: {expected!r}")

    counter = collections.Counter(
        str(c.get("actual_text_in_window", "")).strip() or "(無音)"
        for c in checks
    )
    print(f"  実際の冒頭発話(頻度順 上位 {top}):")
    for text, count in counter.most_common(top):
        share = count / len(checks)
        print(f"    {count:4d} ({share:5.1%})  {text[:60]!r}")

    # 切り分けの当たりを付ける。
    print("  読み方:")
    if empty / len(checks) > 0.5:
        print("    冒頭窓が空の trial が過半。挨拶以前に生成が始まっていない")
        print("    可能性が高い。学習の失敗か、リードインの想定ずれを疑う。")
    elif matched / len(checks) < 0.2:
        print("    生成はしているが挨拶になっていない。挨拶が定着していない側の")
        print("    問題。学習条件(学習対象の範囲・容量・step 数)を見る。")
    elif matched / len(checks) < 0.8:
        print("    部分的に出ている。言い回しの揺れか、定着が浅い。")
    else:
        print("    挨拶は定着している。")


def main() -> int:
    args = parse_args()
    for path in args.per_case:
        if not path.is_file():
            print(f"見つかりません: {path}")
            continue
        summarize(path, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
