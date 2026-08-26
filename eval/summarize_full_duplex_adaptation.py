#!/usr/bin/env python3
"""Filter acoustic adaptation statistics by the local semantic action label."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_full_duplex_adaptation import paired_summary
from full_duplex_judge_input import ACTION_LABELS


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation", required=True, type=Path)
    parser.add_argument("--judged", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--action", choices=list(ACTION_LABELS) + ["all"], default="C_RESPOND")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adaptation = json.loads(args.adaptation.read_text(encoding="utf-8-sig"))
    actions: dict[tuple[str, str], str | None] = {}
    for row in iter_jsonl(args.judged):
        actions[(str(row.get("case_id")), str(row.get("seed")))] = (row.get("judge") or {}).get("overlap_action")
    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, str]] = []
    for row in adaptation.get("per_case", []):
        key = (str(row.get("case_id")), str(row.get("seed")))
        action = actions.get(key)
        if action is None:
            missing.append(key)
            continue
        if args.action == "all" or action == args.action:
            rows.append({**row, "overlap_action": action})
    result = {
        "n": len(rows),
        "action_filter": args.action,
        "missing_judgments": missing,
        "per_case": rows,
        "paired_tests": {
            "pre_vs_post": paired_summary(rows, "pre", "post"),
            "clean_vs_post": paired_summary(rows, "clean", "post"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fdb-adaptation] action={args.action} n={len(rows)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
