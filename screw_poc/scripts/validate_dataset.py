#!/usr/bin/env python3
"""Validate generated screw PoC dialogues and their locked system responses."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .generate_dialogues import (
        DEFAULT_KNOWLEDGE,
        DEFAULT_POLICY,
        load_knowledge,
        load_policy,
        validate_inputs,
    )
except ImportError:  # Direct execution: python screw_poc/scripts/validate_dataset.py
    from generate_dialogues import (
        DEFAULT_KNOWLEDGE,
        DEFAULT_POLICY,
        load_knowledge,
        load_policy,
        validate_inputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dialogues", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = parse_args()
    policy = load_policy(args.policy)
    knowledge = load_knowledge(args.knowledge)
    validate_inputs(policy, knowledge)
    by_id = {item.knowledge_id: item for item in knowledge}
    rows = read_jsonl(args.dialogues)
    errors: list[str] = []
    if args.expected_count is not None and len(rows) != args.expected_count:
        errors.append(f"expected {args.expected_count} dialogues, got {len(rows)}")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("dialogue ids are not unique")

    system_turns = 0
    for row in rows:
        knowledge_id = row.get("knowledge_id")
        item = by_id.get(knowledge_id) if knowledge_id else None
        expected_answer = row.get("expected_answer")
        if item and expected_answer != item.final_text:
            errors.append(f"{row['id']}: expected_answer differs from knowledge DB")
        patterns = []
        for turn in row.get("turns", []):
            if turn.get("speaker") != "moshi":
                continue
            system_turns += 1
            pattern = turn.get("response_pattern")
            if not pattern:
                errors.append(f"{row['id']}: Moshi turn lacks response_pattern")
            patterns.append(pattern)
            if pattern in {"ANSWER", "STOP"} and item and turn.get("text") != item.final_text:
                errors.append(f"{row['id']}: unlocked technical answer detected")
        declared = row.get("expected_response_patterns") or []
        if Counter(patterns) != Counter(declared):
            errors.append(f"{row['id']}: response pattern metadata mismatch")

    if errors:
        raise SystemExit("dataset validation failed:\n- " + "\n- ".join(errors[:30]))
    print(
        json.dumps(
            {
                "ok": True,
                "dialogues": len(rows),
                "system_turns": system_turns,
                "knowledge_coverage": len(
                    {row["knowledge_id"] for row in rows if row.get("knowledge_id")}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
