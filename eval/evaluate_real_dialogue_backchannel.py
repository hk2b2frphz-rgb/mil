#!/usr/bin/env python3
"""
実データ相槌評価。User が話している最中の相槌を、応答評価とは別軸で測る。

応答評価(eval/evaluate_real_response.py)は User の発話が終わった後の第一声だけ
を見る。相槌は発話の最中に何度も起きる 0〜n 個の集合で、応答速度のような 1 点の
時刻としては定義できない。同じ表の同じ列には入らないので、指標を分けている。

  評価区間  output.wav の 0 秒(入力再生の開始)から User の発話終了まで。
            応答評価が捨てている前半が、そのままこちらの対象になる。
  正解      metadata.backchannel_gt。人手アノテーションで User の発話中に
            収まっている相談員の発話。
  予測      output.wav の音声区間(RMS)に、時刻つきテキスト片を重ねたもの。

指標は 4 層。上ほど緩く、下ほど厳しい。どこで落ちたかが分からないと
「相槌が下手」の中身が読めない。

  1. rate      相槌の頻度(件/分)。タイミングも種類も見ない。
  2. timing F1 開始時刻の近さで 1 対 1 対応させた F1(--tolerance-sec 以内)。
  3. typed F1  上に加えて種類(un/hai/sou/naruhodo ...)の一致を要求した F1。
  4. type acc  timing で一致したペアのうち種類まで一致した割合。

gold について:
  正解が相談員自身なので、gold の timing F1 は定義上 1.0 になり意味がない。
  gold は件数・頻度・種類分布の記述統計として読むこと。F1 の実質的な上限を
  知るには別の相談員が同じ場面で打った相槌が要る(現状のデータには無い)。

使い方:
    uv run python eval/evaluate_real_dialogue_backchannel.py \\
        --run-dir eval_runs/real_response/<run>/inference \\
        --out eval_runs/real_response/<run>/benchmark_results/backchannel.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# マッチングと F1 は既存の相槌評価をそのまま使う。二重に書くと、片方だけ直った
# ときに数字が静かに食い違う。
_bc = _load("_miltoka_bc", REPO_ROOT / "eval" / "evaluate_real_backchannel.py")
overlap = _bc.overlap
match_pairs = _bc.match_pairs
prf = _bc.prf
primary_label = _bc.primary_label
BACKCHANNEL_MAX_SEC = _bc.BACKCHANNEL_MAX_SEC
BACKCHANNEL_MAX_CHUNKS = _bc.BACKCHANNEL_MAX_CHUNKS
is_aizuchi_text = _bc.is_aizuchi_text

# 音声区間の取り方は応答評価と揃える。同じ音を別の閾値で見ると、応答したのに
# 相槌軸では無音、といった説明のつかない食い違いが出る。
_resp = _load("_miltoka_resp", REPO_ROOT / "eval" / "evaluate_real_response.py")
speech_segments = _resp.speech_segments
SPEECH_RMS_THRESHOLD = _resp.SPEECH_RMS_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="run_full_duplex_bench.py / run_gold_reference.py の出力")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tolerance-sec", type=float, default=1.0,
                        help="開始時刻の一致とみなす許容窓")
    parser.add_argument("--min-segment-sec", type=float, default=0.1,
                        help="これより短い音声区間は相槌に数えない")
    return parser.parse_args()


def chunk_interval(chunk: dict[str, Any]) -> tuple[float, float] | None:
    stamp = chunk.get("timestamp")
    if not stamp or len(stamp) < 2:
        return None
    try:
        return float(stamp[0]), float(stamp[1])
    except (TypeError, ValueError):
        return None


def extract_predictions(
    audio: np.ndarray, sr: int, chunks: list[dict[str, Any]],
    region: tuple[float, float], min_sec: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for start, end in speech_segments(audio, sr, min_sec):
        after_turn = start >= region[1]
        if not after_turn and end > region[1]:
            # User の発話終了をまたいで続いている = 相槌ではなく応答(または
            # 割り込み)。正解側も同じ規則で境界をまたぐ発話を応答へ渡している
            # ので、予測側だけ相槌に数えると偽陽性になる。
            predictions.append({
                "start_sec": round(start, 4), "end_sec": round(end, 4),
                "text": "", "labels": [], "not_backchannel": True,
                "position": "crosses_turn_end",
            })
            continue
        if end - start > BACKCHANNEL_MAX_SEC:
            # 長い = 相槌ではなく発話の乗っ取り。相槌としては数えないが、
            # 件数だけ別に残す(後段で barge-in として読める)。
            predictions.append({
                "start_sec": round(start, 4), "end_sec": round(end, 4),
                "text": "", "labels": [], "not_backchannel": True,
                "position": "after_turn" if after_turn else "during_turn",
            })
            continue
        pieces = [
            (chunk.get("text") or "")
            for chunk in chunks
            if (span := chunk_interval(chunk)) is not None
            and overlap(span, (start, end)) > 0.0
        ]
        if len(pieces) > BACKCHANNEL_MAX_CHUNKS:
            predictions.append({
                "start_sec": round(start, 4), "end_sec": round(end, 4),
                "text": "".join(pieces).strip(), "labels": [], "not_backchannel": True,
                "position": "after_turn" if after_turn else "during_turn",
            })
            continue
        text = "".join(pieces).strip()
        labels = is_aizuchi_text(text) if text else []
        # 発話終了後は、相槌語彙で説明できるものだけを相槌に数える。あちら側は
        # 普通の応答が来る場所なので、短いだけで相槌とみなすと実応答を相槌に
        # 数えてしまう。正解側も同じ条件で拾っている。
        predictions.append({
            "start_sec": round(start, 4),
            "end_sec": round(end, 4),
            "text": text,
            "labels": labels,
            "not_backchannel": after_turn and not labels,
            "position": "after_turn" if after_turn else "during_turn",
        })
    return predictions


def per_minute(count: int, seconds: float) -> float:
    return round(count / max(1e-9, seconds) * 60.0, 4)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"実行ディレクトリがありません: {run_dir}")

    trials = sorted(p.parent for p in run_dir.glob("*/*/seed_*/output.meta.json"))
    if not trials:
        raise SystemExit(f"{run_dir} に評価対象がありません。")
    print(f"[real-bc] {len(trials)} 試行を評価します")

    totals = {"tp": 0, "fp": 0, "fn": 0}
    typed_totals = {"tp": 0, "fp": 0, "fn": 0}
    type_hits = 0
    type_total = 0
    confusion: dict[str, dict[str, int]] = {}
    gt_labels: Counter = Counter()
    pred_labels: Counter = Counter()
    model_rates: list[float] = []
    human_rates: list[float] = []
    barge_ins = 0
    per_case: list[dict[str, Any]] = []
    model_ids: set[str] = set()

    for trial_dir in trials:
        meta = json.loads((trial_dir / "output.meta.json").read_text(encoding="utf-8"))
        metadata_path = trial_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = metadata.get("source") or {}
        model_ids.add(str(meta.get("model_id")))

        # 評価区間 = 入力再生の開始から User の発話終了まで。
        region_end = float(
            source.get("user_end_rel_sec")
            or meta.get("input_duration_sec")
            or 0.0
        )
        if region_end <= 0.0:
            continue
        # 相槌は User の発話中だけでなく、発話終了後にも起きる。区間は出力全体
        # とし、region_end(発話終了)の前後で扱いを変える。
        region = (0.0, region_end)

        audio, sr = sf.read(trial_dir / "output.wav", dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        text_path = trial_dir / "output.json"
        chunks = (
            json.loads(text_path.read_text(encoding="utf-8")).get("chunks") or []
            if text_path.is_file() else []
        )

        gt = list(metadata.get("backchannel_gt") or [])
        raw_pred = extract_predictions(audio, sr, chunks, region, args.min_segment_sec)
        pred = [p for p in raw_pred if not p["not_backchannel"]]
        barge_ins += sum(1 for p in raw_pred if p["not_backchannel"])

        timing_pairs = match_pairs(gt, pred, args.tolerance_sec, require_type=False)
        typed_pairs = match_pairs(gt, pred, args.tolerance_sec, require_type=True)

        tp = len(timing_pairs)
        totals["tp"] += tp
        totals["fp"] += len(pred) - tp
        totals["fn"] += len(gt) - tp
        typed_tp = len(typed_pairs)
        typed_totals["tp"] += typed_tp
        typed_totals["fp"] += len(pred) - typed_tp
        typed_totals["fn"] += len(gt) - typed_tp

        for gi, pi in timing_pairs:
            g_label = primary_label(gt[gi])
            p_label = primary_label(pred[pi])
            type_total += 1
            if set(gt[gi].get("labels") or []) & set(pred[pi].get("labels") or []):
                type_hits += 1
            confusion.setdefault(g_label, {}).setdefault(p_label, 0)
            confusion[g_label][p_label] += 1

        for item in gt:
            gt_labels[primary_label(item)] += 1
        for item in pred:
            pred_labels[primary_label(item)] += 1

        # 頻度の分母は観測できた全区間。相槌は発話終了後にも起きるので、
        # 発話中の長さだけで割ると水増しになる。
        observed_sec = len(audio) / sr
        model_rates.append(per_minute(len(pred), observed_sec))
        human_rates.append(per_minute(len(gt), observed_sec))

        per_case.append({
            "case_id": meta.get("case_id"),
            "seed": meta.get("seed"),
            "trial_dir": str(trial_dir),
            "region_sec": round(region_end, 4),
            "gt_count": len(gt),
            "pred_count": len(pred),
            "gt_during_turn": sum(
                1 for g in gt if g.get("position", "during_turn") == "during_turn"
            ),
            "gt_after_turn": sum(1 for g in gt if g.get("position") == "after_turn"),
            "pred_during_turn": sum(
                1 for x in pred if x.get("position") == "during_turn"
            ),
            "pred_after_turn": sum(
                1 for x in pred if x.get("position") == "after_turn"
            ),
            "barge_in_count": sum(1 for p in raw_pred if p["not_backchannel"]),
            "timing_matched": tp,
            "typed_matched": typed_tp,
            "gt": gt,
            "pred": pred,
        })

    if not per_case:
        raise SystemExit("評価できた試行がありません。backchannel_gt が入っている"
                         "データセットか確認してください(作り直しが要ります)。")

    ids = sorted(i for i in model_ids if i)
    summary = {
        "model_id": ids[0] if len(ids) == 1 else ids,
        "run_dir": str(run_dir),
        "protocol": "real_dialogue_backchannel",
        "note": (
            "User 発話中の相槌のみ。応答評価(response_rate/latency/utmos)とは"
            "別軸なので、同じ表の同じ列に並べないこと。"
        ),
        "trials": len(per_case),
        "tolerance_sec": args.tolerance_sec,
        "rate_per_min": {
            "model": round(statistics.fmean(model_rates), 4) if model_rates else 0.0,
            "human": round(statistics.fmean(human_rates), 4) if human_rates else 0.0,
        },
        "counts": {
            "gt": sum(c["gt_count"] for c in per_case),
            "pred": sum(c["pred_count"] for c in per_case),
            "barge_in": barge_ins,
        },
        # 発話中に打ったのか、発話終了後に相槌だけ返したのかの内訳。両方とも
        # 相槌として数える(応答軸にも同じものが「黙っていない」証拠として入るが、
        # 測っているものが違う)。
        "position": {
            "gt_during_turn": sum(c["gt_during_turn"] for c in per_case),
            "gt_after_turn": sum(c["gt_after_turn"] for c in per_case),
            "pred_during_turn": sum(c["pred_during_turn"] for c in per_case),
            "pred_after_turn": sum(c["pred_after_turn"] for c in per_case),
        },
        "timing": prf(totals["tp"], totals["fp"], totals["fn"]),
        "typed": prf(typed_totals["tp"], typed_totals["fp"], typed_totals["fn"]),
        "type_accuracy": round(type_hits / type_total, 4) if type_total else None,
        "label_distribution": {
            "gt": dict(gt_labels.most_common()),
            "pred": dict(pred_labels.most_common()),
        },
        "confusion": confusion,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    per_case_path = args.out.with_name(args.out.stem + "_per_case.jsonl")
    with per_case_path.open("w", encoding="utf-8") as f:
        for row in per_case:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pos = summary["position"]
    print(
        f"[real-bc] 発話中 model={pos['pred_during_turn']} human={pos['gt_during_turn']} / "
        f"発話終了後 model={pos['pred_after_turn']} human={pos['gt_after_turn']}"
    )
    print(
        f"[real-bc] rate model={summary['rate_per_min']['model']}/min "
        f"human={summary['rate_per_min']['human']}/min  "
        f"timing F1={summary['timing']['f1']}  "
        f"typed F1={summary['typed']['f1']}  "
        f"type acc={summary['type_accuracy']}"
    )
    print(f"[real-bc] {args.out}")
    print(f"[real-bc] {per_case_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
