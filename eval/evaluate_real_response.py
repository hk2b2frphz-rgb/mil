#!/usr/bin/env python3
"""
実データ応答評価。応答率・応答速度・音声品質(自動MOS)を測る。

入力は eval/run_full_duplex_bench.py または eval/run_gold_reference.py が書いた
実行ディレクトリ。両者は同じ形状なので、gold もモデルと同じコードで処理される。

  応答率   response_rate
      その入力に対して聞き取れる応答を返したケースの割合。無音のまま終わった
      ものは失敗として数える。テキストだけ出て音が出ていない場合も失敗。

  応答速度 response_latency_sec
      User の発話が終わってから、応答音声が鳴り始めるまでの秒数。応答した
      ケースだけで集計する(応答していないものを 0 秒や無限大として混ぜると
      平均が意味を失うため)。

  音声品質 utmos
      UTMOS(自動MOS)。参照音声を必要としない推定器で、応答音声の区間だけを
      切り出して掛ける。無音を含めたまま掛けると値が無音側へ引っ張られる。

出力:
  <out-dir>/summary.json     モデル単位の集計
  <out-dir>/per_case.jsonl   ケース単位(LLM-as-a-judge 入力の素材でもある)

MOS が落ちても評価全体は止めない。ネットワークやモデル取得の失敗で応答率と
応答速度まで失うのは損なので、utmos は null にして続ける。

使い方:
    uv run python eval/evaluate_real_response.py \\
        --run-dir eval_runs/real_response/<run>/inference \\
        --out-dir eval_runs/real_response/<run>/benchmark_results
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# 応答したとみなす音量の下限(RMS)。デコード誤差やごく小さなノイズを応答と
# 数えないための足切り。
SPEECH_RMS_THRESHOLD = 1e-3
# 応答区間とみなす窓の長さ。この単位で RMS を見て、最初に閾値を超えた窓を
# 応答開始とする。
FRAME_SEC = 0.02
# UTMOS に掛ける最短秒数。短すぎると推定が不安定なので、これ未満は測らない。
MOS_MIN_SEC = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mos-backend", default="utmos",
                        choices=["utmos", "none"],
                        help="自動MOSの実装(既定: utmos)。none で音声品質を測らない")
    parser.add_argument("--mos-device", default="cpu",
                        help="UTMOS を動かすデバイス(既定: cpu)。"
                             "推論ジョブと GPU を取り合わないよう既定は cpu")
    parser.add_argument("--max-latency-sec", type=float, default=None,
                        help="これを超える応答開始は「応答しなかった」と扱う。"
                             "既定は無制限(tail 長で自然に切れる)")
    return parser.parse_args()


def rms_frames(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    win = max(1, int(FRAME_SEC * sr))
    pad = (-audio.size) % win
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
    frames = audio.reshape(-1, win)
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)), win


def crosses_input_end(
    audio: np.ndarray, sr: int, input_sec: float
) -> tuple[bool, float | None]:
    """入力終了をまたいで音が続いているかを返す。

    またいでいる = User がまだ話しているうちにモデルが喋り始めた(割り込み)。
    2 つめの戻り値は、入力終了より何秒前から鳴っていたか。
    """
    if input_sec <= 0.0:
        return False, None
    values, win = rms_frames(audio, sr)
    boundary = int(input_sec * sr / win)
    if boundary <= 0 or boundary >= len(values):
        return False, None
    if values[boundary] < SPEECH_RMS_THRESHOLD:
        return False, None
    if values[boundary - 1] < SPEECH_RMS_THRESHOLD:
        return False, None
    i = boundary - 1
    while i > 0 and values[i - 1] >= SPEECH_RMS_THRESHOLD:
        i -= 1
    return True, round(input_sec - i * win / sr, 4)


def speech_span(audio: np.ndarray, sr: int) -> tuple[float, float] | None:
    """音が鳴っている最初と最後の時刻を返す。全部無音なら None。"""
    values, win = rms_frames(audio, sr)
    loud = np.flatnonzero(values >= SPEECH_RMS_THRESHOLD)
    if loud.size == 0:
        return None
    return (loud[0] * win / sr, min(audio.size, (loud[-1] + 1) * win) / sr)


class UtmosScorer:
    """UTMOS(自動MOS)。参照音声を要らない推定器。

    取得に失敗したら黙って無効化し、以降 None を返す。応答率・応答速度は
    MOS と独立に意味を持つので、ここで評価全体を落とさない。
    """

    def __init__(self, device: str):
        self.device = device
        self._model = None
        self._failed = False

    def _load(self) -> None:
        if self._model is not None or self._failed:
            return
        try:
            import torch
            self._model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            ).to(self.device).eval()
            print(f"[real-eval] UTMOS を読み込みました (device={self.device})")
        except Exception as exc:  # noqa: BLE001 - 失敗理由は何であれ続行する
            self._failed = True
            print(
                f"[real-eval] WARNING: UTMOS を読み込めませんでした: {exc}\n"
                "[real-eval]   音声品質は null になります。応答率・応答速度は"
                "そのまま出ます。"
            )

    def score(self, audio: np.ndarray, sr: int) -> float | None:
        self._load()
        if self._model is None:
            return None
        if audio.size < int(MOS_MIN_SEC * sr):
            return None
        try:
            import torch
            wave = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                value = self._model(wave, sr)
            return round(float(np.asarray(value.cpu()).reshape(-1)[0]), 4)
        except Exception as exc:  # noqa: BLE001
            print(f"[real-eval] WARNING: UTMOS の推定に失敗: {exc}")
            return None


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": round(statistics.median(values), 4),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def evaluate_trial(
    trial_dir: Path, scorer: UtmosScorer, max_latency: float | None
) -> dict[str, Any] | None:
    meta_path = trial_dir / "output.meta.json"
    wav_path = trial_dir / "output.wav"
    if not meta_path.is_file() or not wav_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata_path = trial_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file() else {}
    )

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    input_sec = float(meta.get("input_duration_sec") or 0.0)
    # 入力が鳴っている間の出力は評価対象にしない。応答開始は入力終了以降で探す。
    offset = min(audio.size, int(input_sec * sr))
    span = speech_span(audio[offset:], sr)

    # User がまだ話している最中に喋り始め、そのまま入力終了をまたいだ場合。
    # 入力終了の直後は既に鳴っているので、素直に測ると応答速度 0.000 秒という
    # 最良の値が付く。割り込みが最速の応答として表彰されることになるので、
    # 応答速度の集計からは外して別に数える。
    barge_in, barge_in_lead = crosses_input_end(audio, sr, input_sec)

    responded = span is not None
    latency = None
    quality = None
    response_sec = None
    if span is not None:
        latency = round(span[0], 4)
        if max_latency is not None and latency > max_latency:
            responded = False
        else:
            lo = offset + int(span[0] * sr)
            hi = offset + int(span[1] * sr)
            response = audio[lo:hi]
            response_sec = round(response.size / sr, 4)
            quality = scorer.score(response, sr)

    text_path = trial_dir / "output.json"
    text = {}
    if text_path.is_file():
        text = json.loads(text_path.read_text(encoding="utf-8"))

    source = metadata.get("source") or {}
    reference = metadata.get("human_reference") or {}
    return {
        "model_id": meta.get("model_id"),
        "task": meta.get("task"),
        "case_id": meta.get("case_id"),
        "seed": meta.get("seed"),
        "trial_dir": str(trial_dir),
        "dialogue": source.get("dialogue"),
        "user_text": source.get("user_text"),
        "assistant_text": text.get("text", ""),
        "assistant_chunks": text.get("chunks", []),
        "metrics": {
            "responded": responded,
            # 割り込みは応答速度を持たない(起点より前から鳴っているため)。
            "response_latency_sec": (
                latency if responded and not barge_in else None
            ),
            "barge_in": barge_in,
            "barge_in_lead_sec": barge_in_lead,
            "response_duration_sec": response_sec if responded else None,
            "utmos": quality if responded else None,
        },
        "human_reference": {
            "responded": reference.get("responded"),
            "response_latency_sec": reference.get("response_latency_sec"),
        },
        "expected_behavior": metadata.get("expected_behavior"),
        "language": "ja",
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"実行ディレクトリがありません: {run_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scorer = (
        UtmosScorer(args.mos_device) if args.mos_backend == "utmos"
        else UtmosScorer("cpu")
    )
    if args.mos_backend == "none":
        scorer._failed = True  # 読み込みを試みずに常に None を返す

    trials = sorted(p.parent for p in run_dir.glob("*/*/seed_*/output.meta.json"))
    if not trials:
        raise SystemExit(
            f"{run_dir} に評価対象がありません(*/*/seed_*/output.meta.json)。"
        )
    print(f"[real-eval] {len(trials)} 試行を評価します")

    rows: list[dict[str, Any]] = []
    for i, trial_dir in enumerate(trials, 1):
        row = evaluate_trial(trial_dir, scorer, args.max_latency_sec)
        if row is None:
            continue
        rows.append(row)
        if i % 25 == 0 or i == len(trials):
            print(f"[real-eval] {i}/{len(trials)}")

    if not rows:
        raise SystemExit("評価できた試行がありません。")

    per_case = args.out_dir / "per_case.jsonl"
    with per_case.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    responded = [r for r in rows if r["metrics"]["responded"]]
    barge_ins = [r for r in rows if r["metrics"].get("barge_in")]
    latencies = [r["metrics"]["response_latency_sec"] for r in responded]
    qualities = [
        r["metrics"]["utmos"] for r in responded if r["metrics"]["utmos"] is not None
    ]
    human_lat = [
        r["human_reference"]["response_latency_sec"] for r in rows
        if r["human_reference"].get("response_latency_sec") is not None
    ]

    model_ids = sorted({r["model_id"] for r in rows if r["model_id"]})
    summary = {
        "model_id": model_ids[0] if len(model_ids) == 1 else model_ids,
        "run_dir": str(run_dir),
        "protocol": "real_dialogue_single_turn_response",
        "trials": len(rows),
        "response_rate": round(len(responded) / len(rows), 4),
        "responded": len(responded),
        "no_response": len(rows) - len(responded),
        # User の発話終了をまたいで喋っていたケース。応答速度の集計からは
        # 外してある。応答率と併せて読むこと。割り込みが多いモデルは、
        # 残った少数のケースだけで応答速度が良く見える。
        "barge_in": len(barge_ins),
        "barge_in_rate": round(len(barge_ins) / len(rows), 4),
        "barge_in_lead_sec": summarize(
            [r["metrics"]["barge_in_lead_sec"] for r in barge_ins
             if r["metrics"].get("barge_in_lead_sec") is not None]
        ),
        "response_latency_sec": summarize([v for v in latencies if v is not None]),
        "response_duration_sec": summarize(
            [r["metrics"]["response_duration_sec"] for r in responded
             if r["metrics"]["response_duration_sec"] is not None]
        ),
        "utmos": summarize(qualities),
        "mos_backend": args.mos_backend,
        "human_reference_latency_sec": summarize(human_lat),
    }
    if not qualities and args.mos_backend != "none":
        summary["utmos_note"] = (
            "音声品質を取得できませんでした。UTMOS の読み込みに失敗したか、"
            "応答が MOS の最短長に届いていません。ログを確認してください。"
        )

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n===== 実データ応答評価 =====")
    print(f"model:        {summary['model_id']}")
    print(f"trials:       {summary['trials']}")
    print(f"応答率:       {summary['response_rate']:.3f} "
          f"({summary['responded']}/{summary['trials']})")
    if summary["barge_in"]:
        print(f"割り込み:     {summary['barge_in_rate']:.3f} "
              f"({summary['barge_in']}/{summary['trials']}) "
              "-- 応答速度の集計からは除外")
    lat = summary["response_latency_sec"]
    if lat.get("n"):
        print(f"応答速度:     mean {lat['mean']:.3f}s / p50 {lat['p50']:.3f}s "
              f"/ p90 {lat['p90']:.3f}s")
    mos = summary["utmos"]
    if mos.get("n"):
        print(f"音声品質:     UTMOS mean {mos['mean']:.3f} (n={mos['n']})")
    else:
        print("音声品質:     取得できませんでした")
    ref = summary["human_reference_latency_sec"]
    if ref.get("n"):
        print(f"[参考] 相談員の応答速度: mean {ref['mean']:.3f}s / p50 {ref['p50']:.3f}s")
    print(f"\nsummary:   {args.out_dir / 'summary.json'}")
    print(f"per_case:  {per_case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
