#!/usr/bin/env python3
"""合成済み training_set の所要時間と発話重なりをまとめる。

TTS バックエンドを差し替えたときに見たいのは 2 つだけ:

  速度   1 対話あたり何秒かかるか（= 3000 本流したら何時間か）
  重なり 相槌が本当に相手の発話に重なっているか

前者は sidecar JSON の wall_time_sec、後者はステレオ WAV の左右チャンネルの
発話区間から出す。重なりの指標は KABURI-TTS 論文 (arXiv:2609.07200) の Table III
と同じ定義にしてあるので、論文の数字と直接比べられる:

  Overlap      両チャンネルが同時に発話しているフレームの割合
  Silence      どちらも発話していないフレームの割合
  Switches/min 単独で話している話者が入れ替わった回数 / 分

  参考値 (論文 Table III, Test chunk Eval):
    GT (人間)      overlap 0.076 / silence 0.314 / switches 35.4
    KABURI (pred)  overlap 0.096 / silence 0.267 / switches 38.6
    Irodori (stat) overlap 0.013 / silence 0.402 / switches 11.6
      ^ 発話ごとに合成して並べる方式 = このリポの Qwen3/Kokoro 経路に相当

使い方:
  uv run python scripts/report_tts_smoke.py --training-dir data/runs/<batch> --label kaburi
  uv run python scripts/report_tts_smoke.py --training-dir A --training-dir B  # 並べて比較
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FPS = 25
WINDOW_SEC = 0.02
# 論文と同じ「チャンネル内ピークエネルギーの 5%」を発話判定の閾値にする。
ENERGY_THRESHOLD_RATIO = 0.05


def voice_activity(wav, sample_rate: int):
    """(2, n) の波形 -> (2, frames) の bool。20ms 窓の短時間エネルギーで判定。"""
    import numpy as np

    window = max(1, int(round(WINDOW_SEC * sample_rate)))
    hop = max(1, int(round(sample_rate / FPS)))
    frames = 1 + max(0, (wav.shape[-1] - window) // hop)
    if frames <= 0:
        return np.zeros((wav.shape[0], 0), dtype=bool)
    activity = np.zeros((wav.shape[0], frames), dtype=bool)
    for channel in range(wav.shape[0]):
        signal = wav[channel]
        energy = np.empty(frames, dtype=np.float64)
        for i in range(frames):
            chunk = signal[i * hop : i * hop + window]
            energy[i] = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        peak = float(energy.max()) if frames else 0.0
        activity[channel] = energy >= ENERGY_THRESHOLD_RATIO * peak if peak > 0 else False
    return activity


def turn_taking_stats(activity) -> dict[str, float]:
    import numpy as np

    frames = activity.shape[-1]
    if frames == 0:
        return {"overlap": 0.0, "silence": 1.0, "switches_per_min": 0.0}
    left, right = activity[0], activity[1]
    overlap = float(np.mean(left & right))
    silence = float(np.mean(~left & ~right))

    # 単独で話している話者が入れ替わった回数。重なり・無音は現在の話者を
    # 保持したまま通過させる（論文の Switches/min と同じ数え方）。
    switches = 0
    holder: str | None = None
    for is_left, is_right in zip(left.tolist(), right.tolist()):
        if is_left and not is_right:
            current = "L"
        elif is_right and not is_left:
            current = "R"
        else:
            continue
        if holder is not None and current != holder:
            switches += 1
        holder = current
    minutes = frames / FPS / 60.0
    return {
        "overlap": overlap,
        "silence": silence,
        "switches_per_min": switches / minutes if minutes > 0 else 0.0,
    }


def read_sample(json_path: Path) -> dict[str, Any] | None:
    import numpy as np
    import torchaudio

    wav_path = json_path.with_suffix(".wav")
    if not wav_path.is_file():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        meta = payload["metadata"]
    except (OSError, ValueError, KeyError):
        return None

    wav, sample_rate = torchaudio.load(str(wav_path))
    if wav.shape[0] < 2:
        raise SystemExit(f"ステレオではありません: {wav_path} ({wav.shape[0]}ch)")
    stats = turn_taking_stats(voice_activity(np.asarray(wav, dtype=np.float64), int(sample_rate)))
    duration = float(meta.get("duration_sec") or wav.shape[-1] / sample_rate)
    wall = float(meta.get("wall_time_sec") or 0.0)
    return {
        "name": json_path.stem,
        "duration_sec": duration,
        "wall_time_sec": wall,
        "rtf": wall / duration if duration > 0 else 0.0,
        "backend": meta.get("tts_backend"),
        "mode": meta.get("mode"),
        "sample_rate": int(sample_rate),
        "n_chunks": len((meta.get("kaburi") or {}).get("chunks") or []) or None,
        **stats,
    }


def summarize(training_dir: Path, label: str | None) -> dict[str, Any]:
    data_dir = training_dir / "data_stereo"
    if not data_dir.is_dir():
        raise SystemExit(f"data_stereo がありません: {training_dir}")
    samples = [s for s in (read_sample(p) for p in sorted(data_dir.glob("*.json"))) if s]
    if not samples:
        raise SystemExit(f"合成済みサンプルがありません: {data_dir}")

    total_audio = sum(s["duration_sec"] for s in samples)
    total_wall = sum(s["wall_time_sec"] for s in samples)
    n = len(samples)
    return {
        "label": label or training_dir.name,
        "training_dir": str(training_dir),
        "backend": samples[0]["backend"],
        "mode": samples[0]["mode"],
        "sample_rate": samples[0]["sample_rate"],
        "dialogues": n,
        "audio_sec": total_audio,
        "wall_sec": total_wall,
        "sec_per_dialogue": total_wall / n,
        "audio_sec_per_dialogue": total_audio / n,
        "rtf": total_wall / total_audio if total_audio > 0 else 0.0,
        "overlap": sum(s["overlap"] for s in samples) / n,
        "silence": sum(s["silence"] for s in samples) / n,
        "switches_per_min": sum(s["switches_per_min"] for s in samples) / n,
        "samples": samples,
    }


def format_report(summaries: list[dict[str, Any]], projection: int | None) -> str:
    lines: list[str] = []
    for summary in summaries:
        lines.append("")
        lines.append(f"=== {summary['label']} ===")
        lines.append(
            f"backend={summary['backend']} mode={summary['mode']} "
            f"sr={summary['sample_rate']} dialogues={summary['dialogues']}"
        )
        lines.append(
            f"{'sample':<40} {'audio_s':>8} {'wall_s':>8} {'rtf':>6} "
            f"{'overlap':>8} {'silence':>8} {'sw/min':>7}"
        )
        for sample in summary["samples"]:
            lines.append(
                f"{sample['name'][:40]:<40} {sample['duration_sec']:>8.1f} "
                f"{sample['wall_time_sec']:>8.1f} {sample['rtf']:>6.2f} "
                f"{sample['overlap']:>8.3f} {sample['silence']:>8.3f} "
                f"{sample['switches_per_min']:>7.1f}"
            )
        lines.append(
            f"{'MEAN':<40} {summary['audio_sec_per_dialogue']:>8.1f} "
            f"{summary['sec_per_dialogue']:>8.1f} {summary['rtf']:>6.2f} "
            f"{summary['overlap']:>8.3f} {summary['silence']:>8.3f} "
            f"{summary['switches_per_min']:>7.1f}"
        )
        if projection:
            hours = summary["sec_per_dialogue"] * projection / 3600.0
            lines.append(
                f"projection: {projection} dialogues = {hours:.1f} GPU-hours "
                f"({hours / 24:.1f} x 24h jobs on 1 GPU, "
                f"{hours / 2 / 24:.1f} on 2 GPUs)"
            )
    lines.append("")
    lines.append(
        "reference (paper Table III): GT overlap 0.076 / silence 0.314 / sw 35.4, "
        "Irodori-style layout overlap 0.013 / silence 0.402 / sw 11.6"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-dir",
        type=Path,
        action="append",
        required=True,
        help="data_stereo/ を含むディレクトリ。複数回渡すと並べて比較する",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="--training-dir と同じ順の表示名",
    )
    parser.add_argument(
        "--projection",
        type=int,
        default=3000,
        help="この本数を流したときの所要時間を見積もる（0 で無効）",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    labels = args.label or []
    summaries = [
        summarize(directory, labels[i] if i < len(labels) else None)
        for i, directory in enumerate(args.training_dir)
    ]
    print(format_report(summaries, args.projection or None))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\njson: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
