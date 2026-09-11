#!/usr/bin/env python3
"""Build deterministic 100/300/1000-dialogue comparison subsets.

Every subset covers all 30 knowledge rows.  The smaller subsets are sampled
from the canonical 1,000-dialogue file, so knowledge answers and wording stay
identical between experiments.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = POC_ROOT / "artifacts" / "train_dialogues.jsonl"
DEFAULT_OUT = POC_ROOT / "artifacts" / "subsets"
TARGETS = (100, 300, 1000)
FLOW_ORDER = ("direct", "clarify", "varied", "unresolved")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def take_balanced(
    rows: list[dict[str, Any]], count: int, flow_order: tuple[str, ...] = FLOW_ORDER
) -> list[dict[str, Any]]:
    by_flow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["id"]):
        by_flow[row["flow"]].append(row)

    selected: list[dict[str, Any]] = []
    indexes = Counter()
    while len(selected) < count:
        progressed = False
        for flow in flow_order:
            index = indexes[flow]
            if index < len(by_flow[flow]):
                selected.append(by_flow[flow][index])
                indexes[flow] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"cannot select {count} rows from {len(rows)} candidates")
    return selected


def build_subset(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if target == len(rows):
        return list(rows)
    if target not in {100, 300}:
        raise ValueError(f"unsupported target: {target}")

    knowledge_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out_of_scope: list[dict[str, Any]] = []
    for row in rows:
        if row.get("knowledge_id"):
            knowledge_rows[row["knowledge_id"]].append(row)
        else:
            out_of_scope.append(row)
    if len(knowledge_rows) != 30:
        raise ValueError(f"expected 30 knowledge groups, got {len(knowledge_rows)}")

    # 100 = 3 x 30 knowledge + 10 OOD, 300 = 8 x 30 + 60 OOD.
    per_knowledge, ood_count = {100: (3, 10), 300: (8, 60)}[target]
    selected: list[dict[str, Any]] = []
    flow_order = (
        ("direct", "clarify", "unresolved") if target == 100 else FLOW_ORDER
    )
    for knowledge_id in sorted(knowledge_rows):
        selected.extend(
            take_balanced(knowledge_rows[knowledge_id], per_knowledge, flow_order)
        )
    selected.extend(sorted(out_of_scope, key=lambda item: item["id"])[:ood_count])
    if len(selected) != target:
        raise AssertionError(f"subset size mismatch: {len(selected)} != {target}")
    return sorted(selected, key=lambda item: item["id"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dialogues": len(rows),
        "knowledge_coverage": len(
            {row["knowledge_id"] for row in rows if row.get("knowledge_id")}
        ),
        "flows": dict(sorted(Counter(row["flow"] for row in rows).items())),
        "duplex": dict(
            sorted(Counter(row.get("duplex_task", "none") for row in rows).items())
        ),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if len(rows) != 1000:
        raise SystemExit(f"canonical input must contain 1000 rows, got {len(rows)}")
    summary: dict[str, Any] = {}
    for target in TARGETS:
        subset = build_subset(rows, target)
        path = args.out_dir / f"train_{target:04d}.jsonl"
        write_jsonl(path, subset)
        summary[str(target)] = summarize(subset)
    (args.out_dir / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
