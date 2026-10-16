#!/usr/bin/env python3
"""
実対話リプレイ評価の LLM-as-a-judge 入力を作る。

1 ケースにつき 2 行を書き出す。

  - モデルの応答         system_id = <model_id>
  - 相談員の実際の応答   system_id = "human_counselor"

同じ文脈・同じ項目立てで両方を採点させることで、相談員のスコアがそのまま
そのケースの実質的な上限になる。モデル単体のスコアだけを見ても「4.1 は高いのか」
が判断できないため、人間の点を同じ土俵で取っておく。

両者は構造的に同じ形(assistant_text / assistant_timeline)で入れてあり、
どちらが人間かを示す情報は本文側に入れていない。judge のプロンプトを組む際に
system_id を渡さないこと。渡すと人間側に高い点が付く方向のバイアスが乗る。

なお、これは pairwise ではなく絶対評価にしてある。実音声リプレイは開ループで、
文脈区間でモデルは実際の相談員とは違う発話をしている。つまり両者が見ていた
文脈は厳密には同一ではないので、「どちらが良いか」の直接対決は前提が崩れる。
「そのユーザー発話に対して妥当か」を各々独立に採点する形なら成立する。

サーバ上では API を呼ばない。この出力をローカル PC へ持ち帰り、
eval/judge_openai.py などに掛ける(既存の 2 フェーズ方針と同じ)。

使い方:
    uv run python eval/pack_real_dialogue_judge_input.py \\
        --per-case eval_runs/real/<run>/benchmark_results/per_case.jsonl \\
        --out eval_runs/real/<run>/real_judge_input.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-case", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--human-system-id", default="human_counselor")
    parser.add_argument("--skip-human", action="store_true",
                        help="相談員側の行を出さず、モデルの応答だけを採点対象にする")
    return parser.parse_args()


def load_metadata(row: dict[str, Any]) -> dict[str, Any] | None:
    trial_dir = row.get("trial_dir")
    if not trial_dir:
        return None
    path = Path(trial_dir) / "metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def human_timeline(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """相談員の実発話を、モデル出力の chunk と同じ形へ整える。"""
    reference = metadata.get("human_reference") or {}
    return [
        {
            "timestamp": [segment.get("start_sec"), segment.get("end_sec")],
            "text": segment.get("text", ""),
            "is_aizuchi": segment.get("is_aizuchi", False),
        }
        for segment in reference.get("segments") or []
    ]


def base_record(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    source = metadata.get("source") or {}
    return {
        "case_id": row["case_id"],
        "task": row["task"],
        "category": row.get("category"),
        "risk_level": row.get("risk_level"),
        "language": "ja",
        "protocol": "real_dialogue_replay_open_loop",
        "expected_behavior": row.get("expected_behavior"),
        "avoid_behavior": row.get("avoid_behavior", []),
        "user_timeline": row.get("user_segments", []),
        "event_timeline": row.get("events", {}),
        "eval_region_sec": [
            source.get("eval_start_sec"),
            source.get("eval_end_sec"),
        ],
        "context_note": (
            "user_timeline はユーザー側の実音声のみ。相談員側の発話は入力に "
            "含まれておらず、採点対象の話者がその役を務めている。"
        ),
    }


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.per_case.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    seen_cases: set[str] = set()
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            metadata = load_metadata(row)
            if metadata is None:
                skipped += 1
                continue

            model_record = base_record(row, metadata)
            model_record.update(
                {
                    "id": f"{row['model_id']}:{row['case_id']}:seed_{row['seed']}",
                    "system_id": row["model_id"],
                    "system_kind": "model",
                    "seed": row["seed"],
                    "assistant_text": row.get("assistant_text", ""),
                    "assistant_timeline": row.get("assistant_chunks", []),
                    "deterministic_metrics": row.get("metrics", {}),
                }
            )
            handle.write(json.dumps(model_record, ensure_ascii=False) + "\n")
            written += 1

            # 相談員側はシードに依存しないので、ケースごとに 1 行だけ出す。
            if args.skip_human or row["case_id"] in seen_cases:
                continue
            seen_cases.add(row["case_id"])
            reference = metadata.get("human_reference") or {}
            timeline = human_timeline(metadata)
            human_record = base_record(row, metadata)
            human_record.update(
                {
                    "id": f"{args.human_system_id}:{row['case_id']}",
                    "system_id": args.human_system_id,
                    "system_kind": "human",
                    "seed": None,
                    "assistant_text": "".join(item["text"] for item in timeline),
                    "assistant_timeline": timeline,
                    "deterministic_metrics": {
                        "response_latency_sec": reference.get("response_latency_sec"),
                        "backchannel_count": sum(
                            1 for item in timeline if item.get("is_aizuchi")
                        ),
                    },
                }
            )
            handle.write(json.dumps(human_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[real-judge] wrote {written} rows -> {args.out}")
    if skipped:
        print(f"[real-judge] WARNING: metadata.json を読めず {skipped} 行を飛ばしました。")
    print("[real-judge] サーバ上では API を呼んでいません。ローカル PC で judge してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
