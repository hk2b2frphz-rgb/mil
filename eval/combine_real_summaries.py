#!/usr/bin/env python3
"""
実データ応答評価のバッチ結果を 1 つの表にまとめる。

<batch-dir>/<output_name>/benchmark_results/summary.json を集めて、モデルを
横並びにした combined_summary.json を書く。

gold(実際の相談員)が含まれていれば、それを基準行として先頭に置く。gold は
モデルではなく人間なので、勝ち負けを競う相手ではなく**その指標で到達しうる
水準**として読む。

使い方:
    uv run python eval/combine_real_summaries.py \\
        --batch-dir eval_runs/real_batches/<batch> \\
        --status-file eval_runs/real_batches/<batch>/batch_status.jsonl \\
        --out eval_runs/real_batches/<batch>/combined_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GOLD_ID = "gold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_statuses(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    statuses: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        statuses[row["output_name"]] = row
    return statuses


def row_for(output_name: str, summary: dict[str, Any] | None,
            status: dict[str, Any]) -> dict[str, Any]:
    if summary is None:
        return {
            "output_name": output_name,
            "model_id": status.get("model_id"),
            "status": status.get("status", "FAILED"),
            "elapsed_sec": status.get("elapsed_sec"),
            "note": "summary.json がありません(失敗、または別トラックの評価)。",
        }
    latency = summary.get("response_latency_sec") or {}
    utmos = summary.get("utmos") or {}
    duration = summary.get("response_duration_sec") or {}
    return {
        "output_name": output_name,
        "model_id": summary.get("model_id") or status.get("model_id"),
        "status": status.get("status", "ok"),
        "elapsed_sec": status.get("elapsed_sec"),
        "trials": summary.get("trials"),
        "response_rate": summary.get("response_rate"),
        "no_response": summary.get("no_response"),
        "latency_mean_sec": latency.get("mean"),
        "latency_p50_sec": latency.get("p50"),
        "latency_p90_sec": latency.get("p90"),
        "response_duration_mean_sec": duration.get("mean"),
        "utmos_mean": utmos.get("mean"),
        "utmos_n": utmos.get("n"),
        "mos_backend": summary.get("mos_backend"),
    }


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    statuses = load_statuses(args.status_file)

    names = sorted(
        p.name for p in batch_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    rows: list[dict[str, Any]] = []
    for name in names:
        path = batch_dir / name / "benchmark_results" / "summary.json"
        summary = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
        status = statuses.get(name, {})
        if summary is None and name not in statuses:
            continue  # バッチと無関係なディレクトリ
        rows.append(row_for(name, summary, status))

    if not rows:
        raise SystemExit(f"{batch_dir} に集計できる結果がありません。")

    # gold を先頭に。以降は応答率の高い順、同率なら応答が速い順。
    def sort_key(row: dict[str, Any]) -> tuple:
        is_gold = (row.get("model_id") == GOLD_ID)
        rate = row.get("response_rate")
        latency = row.get("latency_p50_sec")
        return (
            0 if is_gold else 1,
            -(rate if rate is not None else -1),
            latency if latency is not None else float("inf"),
        )

    rows.sort(key=sort_key)
    gold = next((r for r in rows if r.get("model_id") == GOLD_ID), None)

    combined = {
        "batch_dir": str(batch_dir),
        "protocol": "real_dialogue_single_turn_response",
        "reference": (
            "gold は実際の相談員の応答。モデルと競わせる相手ではなく、この"
            "データで到達しうる水準として読む。"
            if gold else
            "gold 行がありません。マニフェストに gold を入れると、実際の相談員の"
            "応答が同じ指標で並びます。"
        ),
        "models": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    header = f"{'output_name':<24} {'応答率':>8} {'遅延p50':>9} {'遅延p90':>9} {'UTMOS':>7}  status"
    print("\n===== 実データ応答評価 まとめ =====")
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(value: Any, spec: str) -> str:
            return format(value, spec) if isinstance(value, (int, float)) else "-"
        print(
            f"{row['output_name']:<24} "
            f"{fmt(row.get('response_rate'), '>8.3f')} "
            f"{fmt(row.get('latency_p50_sec'), '>9.3f')} "
            f"{fmt(row.get('latency_p90_sec'), '>9.3f')} "
            f"{fmt(row.get('utmos_mean'), '>7.2f')}  {row.get('status')}"
        )
    if gold:
        print("\ngold = 実際の相談員。上限の目安として読むこと。")
    print(f"\ncombined: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
