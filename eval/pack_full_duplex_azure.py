#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_duplex_judge_input import action_required, compact_event_segments, event_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack Full-Duplex-Bench-JA outputs for later local Azure judging."
    )
    parser.add_argument("--per-case", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.per_case.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            interval = event_interval(row)
            packed = {
                "id": f"{row['model_id']}:{row['case_id']}:seed_{row['seed']}",
                "system_id": row["model_id"],
                "case_id": row["case_id"],
                "task": row["task"],
                "category": row.get("category"),
                "risk_level": row.get("risk_level"),
                "language": "ja",
                "expected_behavior": row.get("expected_behavior"),
                "avoid_behavior": row.get("avoid_behavior", []),
                "user_timeline": row.get("user_segments", []),
                "clean_user_timeline": [
                    segment
                    for segment in row.get("user_segments", [])
                    if segment.get("kind") != "overlap_speech"
                ],
                "event_timeline": row.get("events", {}),
                "assistant_text": row.get("assistant_text", ""),
                "assistant_timeline": row.get("assistant_chunks", []),
                "assistant_event_segments": compact_event_segments(
                    row.get("assistant_chunks", []), interval
                ),
                "clean_assistant_text": row.get("clean_assistant_text"),
                "clean_assistant_timeline": row.get("clean_assistant_chunks", []),
                "clean_assistant_event_segments": compact_event_segments(
                    row.get("clean_assistant_chunks", []), interval
                ),
                "overlap_action_required": action_required(row.get("task")),
                "deterministic_metrics": row.get("metrics", {}),
            }
            handle.write(json.dumps(packed, ensure_ascii=False) + "\n")
    print(f"[fdb-pack] wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
