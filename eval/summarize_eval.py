#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from openai_judge_common import iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize judged evaluation JSONL.")
    parser.add_argument("--input", type=Path, default=None, help="Judged per-response JSONL.")
    parser.add_argument("--pairwise", type=Path, default=None, help="Pairwise judge JSONL.")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def collect_judge_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = {}
    flags: dict[str, int] = {}
    empty = 0
    for row in rows:
        if not str(row.get("assistant_text") or "").strip():
            empty += 1
        judge = row.get("judge") or {}
        for key, value in (judge.get("scores") or {}).items():
            if isinstance(value, (int, float)):
                scores.setdefault(key, []).append(float(value))
        for key, value in (judge.get("flags") or {}).items():
            if value:
                flags[key] = flags.get(key, 0) + 1
    n = len(rows)
    return {
        "n": n,
        "empty_response_rate": empty / n if n else None,
        "score_means": {key: mean(values) for key, values in sorted(scores.items())},
        "flag_rates": {key: count / n for key, count in sorted(flags.items())} if n else {},
    }


def summarize_judged(path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    lat_text = []
    lat_audio = []
    wall_time = []
    output_audio = []
    by_risk: dict[str, list[dict[str, Any]]] = {}
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        meta = row.get("meta") or {}
        for target, dest in (
            ("first_response_latency_sec", lat_text),
            ("audible_start_after_input_sec", lat_audio),
            ("wall_time_sec", wall_time),
            ("output_audio_sec", output_audio),
        ):
            value = meta.get(target)
            if isinstance(value, (int, float)):
                dest.append(float(value))
        by_risk.setdefault(str(row.get("risk_level") or "unknown"), []).append(row)
        by_category.setdefault(str(row.get("category") or "unknown"), []).append(row)

    overall = collect_judge_stats(rows)
    return {
        "type": "judged",
        "input": str(path),
        "n": overall["n"],
        "empty_response_rate": overall["empty_response_rate"],
        "first_response_latency_sec_p50": percentile(lat_text, 0.50),
        "first_response_latency_sec_p90": percentile(lat_text, 0.90),
        "audible_start_after_input_sec_p50": percentile(lat_audio, 0.50),
        "audible_start_after_input_sec_p90": percentile(lat_audio, 0.90),
        "wall_time_sec_p50": percentile(wall_time, 0.50),
        "wall_time_sec_p90": percentile(wall_time, 0.90),
        "output_audio_sec_p50": percentile(output_audio, 0.50),
        "output_audio_sec_p90": percentile(output_audio, 0.90),
        "score_means": overall["score_means"],
        "flag_rates": overall["flag_rates"],
        # Counseling-domain stratification: unsafe/safety numbers on
        # risk_level=high (crisis_signal etc.) are the headline; overall
        # averages dilute them with smalltalk-heavy case mixes.
        "by_risk_level": {
            risk: collect_judge_stats(risk_rows)
            for risk, risk_rows in sorted(by_risk.items())
        },
        "by_category": {
            category: collect_judge_stats(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
    }


def summarize_pairwise(path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path))
    wins: dict[str, int] = {}
    for row in rows:
        winner = str(row.get("winner", "tie"))
        wins[winner] = wins.get(winner, 0) + 1
    n = len(rows)
    return {
        "type": "pairwise",
        "input": str(path),
        "n": n,
        "wins": wins,
        "win_rates": {key: value / n for key, value in sorted(wins.items())} if n else {},
    }


def main() -> int:
    args = parse_args()
    if not args.input and not args.pairwise:
        raise SystemExit("Provide --input or --pairwise.")
    summary: dict[str, Any] = {}
    if args.input:
        summary["judged"] = summarize_judged(args.input)
    if args.pairwise:
        summary["pairwise"] = summarize_pairwise(args.pairwise)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

