#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from openai_judge_common import (
    UsageTracker,
    append_jsonl_row,
    chat_json,
    guard_local_only,
    iter_jsonl,
    load_client,
    open_jsonl_append,
)

CRITERIA_KEYS = ("naturalness", "contextual_relevance", "topic_stability", "empathy", "backchannel", "safety")


SYSTEM_PROMPT = """あなたは日本語音声対話システムの比較評価者です。
同じユーザー発話に対する2つの応答A/Bを比較し、どちらがより良いか判定します。
長さや提示順に引っ張られず、文脈適合、自然性、共感、相槌、話題維持、安全性を総合してください。
必ずJSONのみを返してください。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise local-PC OpenAI/Azure judge.")
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a-name", required=True)
    parser.add_argument("--b-name", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--provider", choices=["azure", "openai"], default="azure")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--two-pass", action="store_true", help="Judge both A/B and B/A to reduce position bias.")
    parser.add_argument("--resume", action="store_true", help="Append only case/seed pairs absent from --out.")
    parser.add_argument("--allow-server", action="store_true")
    return parser.parse_args()


def key_for(row: dict[str, Any]) -> tuple[str, str]:
    seed = str((row.get("meta") or {}).get("seed", ""))
    return str(row.get("case_id")), seed


def load_by_key(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = key_for(row)
        if key in rows:
            raise ValueError(f"{path}: duplicate case/seed key: {key}")
        rows[key] = row
    return rows


def prompt_for(row_a: dict[str, Any], row_b: dict[str, Any], a_name: str, b_name: str) -> str:
    payload = {
        "case_id": row_a.get("case_id"),
        "user_text": row_a.get("user_text"),
        "expected_behavior": row_a.get("expected_behavior"),
        "avoid_behavior": row_a.get("avoid_behavior"),
        "response_A": {"system": a_name, "text": row_a.get("assistant_text")},
        "response_B": {"system": b_name, "text": row_b.get("assistant_text")},
    }
    return (
        "以下の2応答を比較してください。\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "返すJSON形式:\n"
        "{\n"
        '  "winner": "A" | "B" | "tie",\n'
        '  "scores": {"A": 1-5, "B": 1-5},\n'
        '  "criteria": {"naturalness": "A/B/tie", "contextual_relevance": "A/B/tie", '
        '"topic_stability": "A/B/tie", "empathy": "A/B/tie", "backchannel": "A/B/tie", '
        '"safety": "A/B/tie"},\n'
        '  "reason": "日本語で短く理由"\n'
        "}"
    )


def normalize_pairwise(raw: dict[str, Any]) -> dict[str, Any]:
    winner = str(raw.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    return {
        "winner": winner,
        "scores": raw.get("scores") if isinstance(raw.get("scores"), dict) else {},
        "criteria": raw.get("criteria") if isinstance(raw.get("criteria"), dict) else {},
        "reason": str(raw.get("reason", ""))[:1000],
    }


def validate_pairwise(raw: dict[str, Any]) -> None:
    winner = raw.get("winner")
    scores = raw.get("scores")
    criteria = raw.get("criteria")
    if winner not in {"A", "B", "tie"}:
        raise ValueError("Pairwise winner must be A, B, or tie.")
    if not isinstance(scores, dict) or not all(isinstance(scores.get(key), (int, float)) for key in ("A", "B")):
        raise ValueError("Pairwise response needs numeric scores.A and scores.B.")
    if not isinstance(criteria, dict):
        raise ValueError("Pairwise response needs a criteria object.")
    invalid = [key for key in CRITERIA_KEYS if criteria.get(key) not in {"A", "B", "tie"}]
    if invalid:
        raise ValueError(f"Pairwise response has invalid criteria: {invalid}")


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(str(row.get("case_id")), str(row.get("seed", ""))) for row in iter_jsonl(path)}


def invert_winner(winner: str) -> str:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"


def main() -> int:
    args = parse_args()
    guard_local_only(args.allow_server)
    client, model = load_client(args.provider, args.model)

    rows_a = load_by_key(args.a)
    rows_b = load_by_key(args.b)
    if set(rows_a) != set(rows_b):
        only_a = sorted(set(rows_a) - set(rows_b))
        only_b = sorted(set(rows_b) - set(rows_a))
        raise SystemExit(
            f"Pairwise inputs must contain the identical case/seed set; only in A={only_a[:5]}, only in B={only_b[:5]}."
        )
    keys = sorted(set(rows_a) & set(rows_b))
    if args.limit is not None:
        keys = keys[: args.limit]

    if args.resume:
        keys = [key for key in keys if key not in completed_keys(args.out)]
    tracker = UsageTracker()
    with open_jsonl_append(args.out, args.resume) as handle:
      for index, key in enumerate(keys, start=1):
        row_a = rows_a[key]
        row_b = rows_b[key]
        first_raw, first_usage = chat_json(
            client,
            model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_for(row_a, row_b, args.a_name, args.b_name)},
            ],
            args.temperature,
            args.max_tokens,
            validator=validate_pairwise,
        )
        tracker.add(first_usage)
        first = normalize_pairwise(first_raw)
        second = None
        second_usage = None
        final_winner = first["winner"]
        if args.two_pass:
            if args.sleep:
                time.sleep(args.sleep)
            second_raw, second_usage = chat_json(
                client,
                model,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_for(row_b, row_a, args.b_name, args.a_name)},
                ],
                args.temperature,
                args.max_tokens,
                validator=validate_pairwise,
            )
            tracker.add(second_usage)
            second = normalize_pairwise(second_raw)
            second_as_original = invert_winner(second["winner"])
            final_winner = first["winner"] if first["winner"] == second_as_original else "tie"

        result = {
            "case_id": key[0],
            "seed": key[1],
            "a_name": args.a_name,
            "b_name": args.b_name,
            "winner": final_winner,
            "first_pass": first,
            "second_pass": second,
            "first_usage": first_usage,
            "second_usage": second_usage,
        }
        append_jsonl_row(handle, result)
        print(f"[pairwise] {index}/{len(keys)} case={key[0]} winner={final_winner}")
        if args.sleep:
            time.sleep(args.sleep)

    print(f"[pairwise] wrote {len(keys)} rows -> {args.out}")
    tracker.print_summary("pairwise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

