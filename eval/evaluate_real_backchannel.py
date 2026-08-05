#!/usr/bin/env python3
"""
実対話の相槌正解に対する、モデル相槌の一致度を測る。

正解は「その場面で実際の相談員が打った相槌」(時刻と種類つき)。
eval/build_real_dialogue_dataset.py が metadata.backchannel_gt に埋め込む。
予測側は eval/evaluate_full_duplex_ja.py が per_case.jsonl に書く
speech_segments(Silero VAD)と assistant_chunks(時刻つきテキスト)から取る。

指標を 4 層に分けている。上の層ほど緩く、下の層ほど厳しい。どこで落ちたかが
分かる形にしないと「相槌が下手」の中身が読めないため。

  1. rate      : 相槌の頻度(件/分)。タイミングも種類も見ない。人間との比だけ。
  2. timing F1 : 正解相槌と予測相槌を開始時刻の近さで 1 対 1 対応させた F1。
                 許容窓 --tolerance-sec 以内に入れば一致とみなす。
  3. typed F1  : 上の一致に加えて相槌の種類(un/hai/sou/naruhodo ...)が
                 重なることを要求した F1。
  4. type acc  : timing で一致したペアのうち、種類まで一致した割合。
                 typed F1 が落ちたのがタイミングのせいか種類のせいかを分ける。

許容窓を置いているのは、実音声リプレイが開ループだからである。文脈区間で
モデルは実際の相談員とは違う発話をしているので、同じ場面でも相槌の位置は
厳密には一致しない。ミリ秒単位の一致を要求しても測っているのは運になる。

人間側のスコアについて:
  正解が相談員自身なので、相談員の timing F1 は定義上 1.0 になり意味がない。
  代わりに人間の記述統計(件数・頻度・種類分布)を同じ場面集合で出し、モデルの
  数値を並べて読めるようにしている。F1 の実質的な上限を知りたい場合は、
  別の相談員が同じ場面で打った相槌が要る(現状のデータには無い)。

使い方:
    uv run python eval/evaluate_real_backchannel.py \\
        --per-case eval_runs/real/<run>/benchmark_results/per_case.jsonl \\
        --out eval_runs/real/<run>/benchmark_results/backchannel_agreement.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_real_dialogue import is_aizuchi_text  # noqa: E402

# 相槌とみなす発話の最大長。Full-Duplex-Bench の time_threshold と揃える。
BACKCHANNEL_MAX_SEC = 3.0
# 相槌に重なるテキスト片がこの数を超えるものは、相槌ではなく発話の乗っ取り。
BACKCHANNEL_MAX_CHUNKS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-case", type=Path, required=True,
                        help="evaluate_full_duplex_ja.py が書いた per_case.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance-sec", type=float, default=1.5,
                        help="正解相槌と予測相槌を一致とみなす開始時刻の許容差")
    parser.add_argument("--tasks", default="backchannel",
                        help="対象タスク(カンマ区切り)。all で全タスク")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunk_interval(chunk: dict[str, Any]) -> tuple[float, float] | None:
    """assistant_chunks の 1 片から (開始, 終了) を取り出す。

    上流の表現ゆれを吸収する。timestamp が [start, end] の場合と、
    start_sec / end_sec を持つ場合の両方がある。
    """
    stamp = chunk.get("timestamp")
    if isinstance(stamp, (list, tuple)) and len(stamp) == 2:
        try:
            return float(stamp[0]), float(stamp[1])
        except (TypeError, ValueError):
            return None
    if "start_sec" in chunk and "end_sec" in chunk:
        try:
            return float(chunk["start_sec"]), float(chunk["end_sec"])
        except (TypeError, ValueError):
            return None
    return None


def chunk_text(chunk: dict[str, Any]) -> str:
    for key in ("text", "token", "piece", "word"):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def extract_predictions(
    row: dict[str, Any], region: tuple[float, float]
) -> list[dict[str, Any]]:
    """モデル出力から相槌らしい発話区間を拾い、種類ラベルを付ける。

    区間は Silero VAD の結果をそのまま使う(評価器と同じ音声区間定義)。
    そこへ重なる時刻つきテキスト片を連結して相槌語彙に照合する。
    """
    predictions: list[dict[str, Any]] = []
    chunks = row.get("assistant_chunks") or []
    for segment in row.get("speech_segments") or []:
        try:
            start, end = float(segment[0]), float(segment[1])
        except (TypeError, ValueError, IndexError):
            continue
        if overlap((start, end), region) <= 0.0:
            continue
        if end - start > BACKCHANNEL_MAX_SEC:
            continue
        pieces = [
            chunk_text(chunk)
            for chunk in chunks
            if (span := chunk_interval(chunk)) is not None
            and overlap(span, (start, end)) > 0.0
        ]
        if len(pieces) > BACKCHANNEL_MAX_CHUNKS:
            # 語数が多い = 相槌ではなく発話。乗っ取り側なので予測に入れない。
            continue
        text = "".join(pieces).strip()
        labels = is_aizuchi_text(text) if text else []
        predictions.append(
            {"start_sec": start, "end_sec": end, "text": text, "labels": labels}
        )
    return predictions


def match_pairs(
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    tolerance: float,
    require_type: bool,
) -> list[tuple[int, int]]:
    """開始時刻の近さで 1 対 1 に対応づける。

    候補ペアを距離昇順に見て、まだ使われていない正解と予測を貪欲に結ぶ。
    1 対 1 を守らないと、1 つの正解に複数の予測を当てて再現率を水増しできて
    しまう。
    """
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            distance = abs(float(p["start_sec"]) - float(g["start_sec"]))
            if distance > tolerance:
                continue
            if require_type and not (set(g.get("labels") or []) & set(p.get("labels") or [])):
                continue
            candidates.append((distance, gi, pi))
    candidates.sort()
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _distance, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi))
    return pairs


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def primary_label(item: dict[str, Any]) -> str:
    labels = item.get("labels") or []
    return labels[0] if labels else "(none)"


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.per_case)
    selected = None if args.tasks == "all" else {
        item.strip() for item in args.tasks.split(",") if item.strip()
    }

    totals = {"tp": 0, "fp": 0, "fn": 0}
    typed_totals = {"tp": 0, "fp": 0, "fn": 0}
    type_hits = 0
    type_total = 0
    confusion: dict[str, dict[str, int]] = {}
    per_case: list[dict[str, Any]] = []
    model_rates: list[float] = []
    human_rates: list[float] = []
    human_labels: dict[str, int] = {}
    model_labels: dict[str, int] = {}
    skipped_no_metadata = 0

    for row in rows:
        if selected is not None and row.get("task") not in selected:
            continue
        trial_dir = row.get("trial_dir")
        if not trial_dir:
            skipped_no_metadata += 1
            continue
        metadata_path = Path(trial_dir) / "metadata.json"
        if not metadata_path.is_file():
            skipped_no_metadata += 1
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = metadata.get("source") or {}
        region = (
            float(source.get("eval_start_sec", 0.0)),
            float(source.get("eval_end_sec", metadata.get("duration_sec", 0.0))),
        )
        region_sec = max(1e-9, region[1] - region[0])

        gt = [
            item for item in (metadata.get("backchannel_gt") or [])
            if overlap(
                (float(item["start_sec"]), float(item["end_sec"])), region
            ) > 0.0
        ]
        pred = extract_predictions(row, region)

        timing_pairs = match_pairs(gt, pred, args.tolerance_sec, require_type=False)
        typed_pairs = match_pairs(gt, pred, args.tolerance_sec, require_type=True)

        tp, fp, fn = len(timing_pairs), len(pred) - len(timing_pairs), len(gt) - len(timing_pairs)
        t_tp = len(typed_pairs)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        typed_totals["tp"] += t_tp
        typed_totals["fp"] += len(pred) - t_tp
        typed_totals["fn"] += len(gt) - t_tp

        for gi, pi in timing_pairs:
            gold = primary_label(gt[gi])
            guess = primary_label(pred[pi])
            confusion.setdefault(gold, {})
            confusion[gold][guess] = confusion[gold].get(guess, 0) + 1
            type_total += 1
            if set(gt[gi].get("labels") or []) & set(pred[pi].get("labels") or []):
                type_hits += 1

        for item in gt:
            human_labels[primary_label(item)] = human_labels.get(primary_label(item), 0) + 1
        for item in pred:
            model_labels[primary_label(item)] = model_labels.get(primary_label(item), 0) + 1

        model_rates.append(len(pred) / region_sec * 60.0)
        human_rates.append(len(gt) / region_sec * 60.0)

        per_case.append(
            {
                "case_id": row.get("case_id"),
                "task": row.get("task"),
                "seed": row.get("seed"),
                "region_sec": round(region_sec, 3),
                "gt_count": len(gt),
                "pred_count": len(pred),
                "timing_matched": tp,
                "typed_matched": t_tp,
                "gt": gt,
                "pred": pred,
            }
        )

    if not per_case:
        raise SystemExit(
            "対象ケースが 0 件でした。--tasks と per_case.jsonl の中身を確認してください。"
        )

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    model_rate = mean(model_rates)
    human_rate = mean(human_rates)
    report = {
        "per_case_input": str(args.per_case),
        "cases": len(per_case),
        "tolerance_sec": args.tolerance_sec,
        "tasks": sorted(selected) if selected else "all",
        "timing_f1": prf(totals["tp"], totals["fp"], totals["fn"]),
        "typed_f1": prf(typed_totals["tp"], typed_totals["fp"], typed_totals["fn"]),
        "type_accuracy": {
            "matched_pairs": type_total,
            "type_agreed": type_hits,
            "accuracy": round(type_hits / type_total, 4) if type_total else 0.0,
        },
        "rate_per_min": {
            "model": model_rate,
            "human": human_rate,
            "ratio": round(model_rate / human_rate, 4) if human_rate else None,
            "abs_log_ratio": (
                round(abs(math.log((model_rate + 1e-6) / (human_rate + 1e-6))), 4)
                if human_rate else None
            ),
        },
        "label_distribution": {
            "human": dict(sorted(human_labels.items(), key=lambda kv: -kv[1])),
            "model": dict(sorted(model_labels.items(), key=lambda kv: -kv[1])),
        },
        "confusion_primary_label": confusion,
        "human_baseline_note": (
            "正解が相談員自身のため、相談員の timing/typed F1 は定義上 1.0 であり "
            "意味を持たない。人間側は label_distribution.human と "
            "rate_per_min.human を記述統計として読むこと。F1 の実質的な上限を "
            "推定するには、同じ場面に対する別の相談員の相槌が必要。"
        ),
        "skipped_cases_without_metadata": skipped_no_metadata,
        "per_case": per_case,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[real-bc] cases={len(per_case)} -> {args.out}")
    print(
        f"[real-bc] timing F1={report['timing_f1']['f1']} "
        f"typed F1={report['typed_f1']['f1']} "
        f"type acc={report['type_accuracy']['accuracy']}"
    )
    print(f"[real-bc] rate/min model={model_rate} human={human_rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
