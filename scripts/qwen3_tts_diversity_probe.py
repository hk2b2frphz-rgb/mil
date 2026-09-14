#!/usr/bin/env python3
"""相槌を1つの参照音声で大量に合成し、出力がどれだけ散らばるかを測る。

狙いは 1 つ。「相槌のように短くパターンの限られた発話を Qwen3-TTS で作り置き
しておいて、欲しいタイミング・韻律のものを検索で引くことは成立するか」。
成立するには、同じ参照音声・同じ設定でも出力が散らばっていないといけない。
散らばり方は 2 つに分けて測る:

  text 内のばらつき  同じ文字列を n 回投げたときのばらつき。サンプリング由来。
                     ここが 0 なら貯める意味は無い(全部同じ音が並ぶだけ)。
  text 間のへだたり  「うーん」と「うーーん」と「うぅん」のように、書き方を
                     変えたときのへだたり。ここが大きいなら、多様性は
                     入力テキスト側で作るのが正解ということになる。

出るのは 1 音あたり 尺 / F0 / 有声区間の数、および上の 2 つの比。
両方まとめて data/runs/diversity/<id>/report.md に出す。

  $VLLM_PYTHON scripts/qwen3_tts_diversity_probe.py \
      --ref-audio ref.wav --ref-text "..." \
      --out-dir data/runs/diversity/probe01 --repeats 10

vLLM-Omni は import が重いので、解析側の関数だけを import して使えるように
バックエンドの import は main() の中に置いてある(tests/ がそれを使う)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# 解析のフレーム。40ms 窓 / 10ms ホップ。自己相関で F0 を拾うので、窓は
# 下限 60Hz の 2 周期より長く取ってある。
FRAME_SEC = 0.04
HOP_SEC = 0.01
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
# 有声判定。正規化自己相関のピークと、そのフレームの RMS の両方を見る。
F0_PEAK_THRESHOLD = 0.35
# 無音判定は report_tts_smoke.py と同じ「ピークの 5%」。
SILENCE_RATIO = 0.05
# 有声区間をいくつに割るか。60ms 以上の無声が入ったら別の区間として数える。
# 「うん」は 1、「うんうん」は 2、「そっか」は促音で 2 になる、という粒度。
RUN_GAP_SEC = 0.06

# 既定の相槌リスト。実録音(116件)で上位を占めた 4 語 -- うん / うーん /
# そっか / うんうん -- を中心に、伸ばし方と表記を振ってある。今の合成語彙に
# 無い語ばかりなので、そもそも出せるのかの確認も兼ねる。
DEFAULT_TEXTS: tuple[str, ...] = (
    "うん",
    "うーん",
    "うーーん",
    "うぅん",
    "うんっ",
    "うん。",
    "うんうん",
    "うんうんうん",
    "そっか",
    "そっかー",
    "そっかぁ",
    "そうか",
    "なるほど",
    "なるほどー",
    "はい",
    "はーい",
    "あー",
    "あぁ",
    "ねえ",
    "へー",
)


# --------------------------------------------------------------------------
# 測る
# --------------------------------------------------------------------------
def frame_rms(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """10ms ホップの短時間 RMS。"""
    window = max(1, int(round(FRAME_SEC * sample_rate)))
    hop = max(1, int(round(HOP_SEC * sample_rate)))
    if signal.size < window:
        signal = np.pad(signal, (0, window - signal.size))
    frames = 1 + (signal.size - window) // hop
    out = np.empty(frames, dtype=np.float64)
    for i in range(frames):
        chunk = signal[i * hop : i * hop + window]
        out[i] = float(np.sqrt(np.mean(np.square(chunk))))
    return out


def speech_bounds(rms: np.ndarray) -> tuple[int, int]:
    """発話している最初と最後のフレーム番号 (end は排他)。無音なら (0, 0)。"""
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return 0, 0
    voiced = np.flatnonzero(rms >= SILENCE_RATIO * peak)
    if voiced.size == 0:
        return 0, 0
    return int(voiced[0]), int(voiced[-1]) + 1


def estimate_f0(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """フレームごとの F0[Hz]。無声フレームは 0。

    正規化自己相関のピークを取り、放物線補間で lag を細かくする。librosa も
    scipy も無い vLLM-Omni 環境で動かすため numpy だけで書いてある。相槌は
    1 秒前後しかないので、この程度の素朴さでも尺と上下は十分に拾える。
    """
    window = max(1, int(round(FRAME_SEC * sample_rate)))
    hop = max(1, int(round(HOP_SEC * sample_rate)))
    min_lag = max(2, int(math.floor(sample_rate / F0_MAX_HZ)))
    max_lag = min(window - 1, int(math.ceil(sample_rate / F0_MIN_HZ)))
    rms = frame_rms(signal, sample_rate)
    peak = float(rms.max()) if rms.size else 0.0
    floor = SILENCE_RATIO * peak
    padded = signal
    if padded.size < window:
        padded = np.pad(padded, (0, window - padded.size))

    f0 = np.zeros(rms.size, dtype=np.float64)
    if max_lag <= min_lag:
        return f0
    for i in range(rms.size):
        if rms[i] < floor or rms[i] <= 0.0:
            continue
        chunk = padded[i * hop : i * hop + window].astype(np.float64)
        chunk = chunk - chunk.mean()
        energy = float(np.dot(chunk, chunk))
        if energy <= 0.0:
            continue
        correlation = np.correlate(chunk, chunk, mode="full")[window - 1 :]
        correlation = correlation[: max_lag + 1]
        if correlation.size <= min_lag + 1:
            continue
        search = correlation[min_lag:] / energy
        best = int(np.argmax(search))
        score = float(search[best])
        if score < F0_PEAK_THRESHOLD:
            continue
        lag = min_lag + best
        # 放物線補間。端にいるときは補間せずそのまま。
        if 0 < best < search.size - 1:
            left, center, right = search[best - 1], search[best], search[best + 1]
            denominator = left - 2.0 * center + right
            if denominator != 0.0:
                lag = lag + 0.5 * (left - right) / denominator
        if lag > 0:
            f0[i] = sample_rate / lag
    return f0


def voiced_runs(f0: np.ndarray) -> list[tuple[int, int]]:
    """有声フレームの連なり。RUN_GAP_SEC 未満の途切れは繋いだまま数える。"""
    gap_frames = max(1, int(round(RUN_GAP_SEC / HOP_SEC)))
    runs: list[list[int]] = []
    for index in np.flatnonzero(f0 > 0):
        index = int(index)
        if runs and index - runs[-1][1] <= gap_frames:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    return [(start, end + 1) for start, end in runs]


def _semitones(values: np.ndarray, reference: float) -> np.ndarray:
    if reference <= 0:
        return np.zeros_like(values)
    return 12.0 * np.log2(np.maximum(values, 1e-6) / reference)


def measure(signal: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """1 音の測定値。report と feature_sequence の両方がここから出る。"""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    rms = frame_rms(signal, sample_rate)
    start, end = speech_bounds(rms)
    f0 = estimate_f0(signal, sample_rate)
    voiced = f0[f0 > 0]
    median = float(np.median(voiced)) if voiced.size else 0.0

    slope = 0.0
    if voiced.size >= 3:
        positions = np.flatnonzero(f0 > 0).astype(np.float64) * HOP_SEC
        slope = float(
            np.polyfit(positions, _semitones(voiced, median), 1)[0]
        )

    runs = voiced_runs(f0)
    return {
        "duration_sec": round(signal.size / sample_rate, 4),
        "speech_sec": round((end - start) * HOP_SEC, 4),
        "lead_silence_sec": round(start * HOP_SEC, 4),
        "trail_silence_sec": round(max(0, rms.size - end) * HOP_SEC, 4),
        "rms": round(float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0, 6),
        "peak": round(float(np.max(np.abs(signal))) if signal.size else 0.0, 6),
        "voiced_ratio": round(float(np.mean(f0 > 0)) if f0.size else 0.0, 4),
        "f0_median_hz": round(median, 2),
        "f0_p10_hz": round(float(np.percentile(voiced, 10)), 2) if voiced.size else 0.0,
        "f0_p90_hz": round(float(np.percentile(voiced, 90)), 2) if voiced.size else 0.0,
        "f0_range_semitones": (
            round(
                float(
                    _semitones(
                        np.asarray(np.percentile(voiced, 90)),
                        float(np.percentile(voiced, 10)),
                    )
                ),
                2,
            )
            if voiced.size and np.percentile(voiced, 10) > 0
            else 0.0
        ),
        "f0_slope_semitones_per_sec": round(slope, 2),
        "n_voiced_runs": len(runs),
        "voiced_run_sec": [round((run_end - run_start) * HOP_SEC, 3) for run_start, run_end in runs],
    }


def feature_sequence(signal: np.ndarray, sample_rate: int, max_frames: int = 200) -> np.ndarray:
    """DTW 用の (frames, 2) 特徴。[F0 の相対セミトーン, 正規化 log エネルギー]。

    どちらも自分自身で正規化してあるので、比べているのは音量でも声の高さでも
    なく「形」。同じ語を 2 回投げて形まで一致するなら、貯める価値は無い。
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    rms = frame_rms(signal, sample_rate)
    f0 = estimate_f0(signal, sample_rate)
    start, end = speech_bounds(rms)
    if end <= start:
        return np.zeros((0, 2), dtype=np.float64)
    rms = rms[start:end]
    f0 = f0[start:end]

    voiced = f0[f0 > 0]
    reference = float(np.median(voiced)) if voiced.size else 0.0
    pitch = np.where(f0 > 0, _semitones(f0, reference) if reference > 0 else 0.0, 0.0)
    energy = np.log(np.maximum(rms, 1e-8))
    spread = float(np.std(energy))
    energy = (energy - float(np.mean(energy))) / (spread if spread > 0 else 1.0)

    features = np.stack([pitch, energy], axis=1)
    if features.shape[0] > max_frames:
        index = np.linspace(0, features.shape[0] - 1, max_frames)
        features = np.stack(
            [np.interp(index, np.arange(features.shape[0]), features[:, c]) for c in range(2)],
            axis=1,
        )
    return features


def dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    """長さ正規化した DTW 距離。どちらかが空なら 0。"""
    if left.shape[0] == 0 or right.shape[0] == 0:
        return 0.0
    cost = np.sqrt(
        np.maximum(
            ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2),
            0.0,
        )
    )
    rows, columns = cost.shape
    accumulated = np.full(columns + 1, np.inf)
    accumulated[0] = 0.0
    for i in range(rows):
        previous = accumulated.copy()
        accumulated[0] = np.inf
        for j in range(columns):
            accumulated[j + 1] = cost[i, j] + min(
                previous[j], previous[j + 1], accumulated[j]
            )
    return float(accumulated[columns] / (rows + columns))


def mean_pairwise_dtw(features: Sequence[np.ndarray], cap: int = 6) -> float:
    """総当たりは高くつくので先頭 cap 本だけで平均を取る。"""
    subset = list(features)[:cap]
    distances = [
        dtw_distance(subset[i], subset[j])
        for i in range(len(subset))
        for j in range(i + 1, len(subset))
    ]
    return round(float(statistics.fmean(distances)), 4) if distances else 0.0


# --------------------------------------------------------------------------
# まとめる
# --------------------------------------------------------------------------
def _spread(values: Sequence[float]) -> dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "cv": round(sd / mean, 4) if mean else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize_group(text: str, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    durations = [s["measure"]["speech_sec"] for s in samples]
    pitches = [s["measure"]["f0_median_hz"] for s in samples if s["measure"]["f0_median_hz"] > 0]
    runs = [s["measure"]["n_voiced_runs"] for s in samples]
    digests = {s["audio_sha1"] for s in samples}
    return {
        "text": text,
        "n": len(samples),
        "unique_audio": len(digests),
        "speech_sec": _spread(durations),
        "f0_median_hz": _spread(pitches),
        "n_voiced_runs": _spread(runs),
        "within_dtw": mean_pairwise_dtw([s["features"] for s in samples]),
    }


def cross_group_dtw(groups: Sequence[dict[str, Any]], per_group: int = 3) -> float:
    """異なる text 同士の DTW 平均。text 間のへだたりの物差し。"""
    picked = [
        (group["text"], sample["features"])
        for group in groups
        for sample in group["samples"][:per_group]
    ]
    distances = [
        dtw_distance(left_features, right_features)
        for i, (left_text, left_features) in enumerate(picked)
        for right_text, right_features in picked[i + 1 :]
        if left_text != right_text
    ]
    return round(float(statistics.fmean(distances)), 4) if distances else 0.0


def build_report(
    grouped: Sequence[dict[str, Any]], context: dict[str, Any]
) -> dict[str, Any]:
    summaries = [summarize_group(g["text"], g["samples"]) for g in grouped]

    within_cv = [s["speech_sec"]["cv"] for s in summaries if s["speech_sec"]["mean"] > 0]
    within_dtw = [s["within_dtw"] for s in summaries]
    group_means = [s["speech_sec"]["mean"] for s in summaries if s["speech_sec"]["mean"] > 0]
    group_pitch = [s["f0_median_hz"]["mean"] for s in summaries if s["f0_median_hz"]["mean"] > 0]
    within_sd = [s["speech_sec"]["sd"] for s in summaries if s["speech_sec"]["mean"] > 0]

    total = sum(s["n"] for s in summaries)
    unique = sum(s["unique_audio"] for s in summaries)
    mean_within_sd = statistics.fmean(within_sd) if within_sd else 0.0
    between_sd = statistics.pstdev(group_means) if len(group_means) > 1 else 0.0
    mean_within_dtw = statistics.fmean(within_dtw) if within_dtw else 0.0
    between_dtw = cross_group_dtw(grouped)

    return {
        "context": context,
        "groups": summaries,
        "totals": {
            "samples": total,
            "unique_audio": unique,
            "duplicate_audio": total - unique,
            "texts": len(summaries),
            "within_text_duration_cv_mean": round(
                statistics.fmean(within_cv) if within_cv else 0.0, 4
            ),
            "within_text_duration_sd_mean_sec": round(mean_within_sd, 4),
            "between_text_duration_sd_sec": round(between_sd, 4),
            "duration_separation": round(between_sd / mean_within_sd, 3)
            if mean_within_sd > 0
            else 0.0,
            "within_text_dtw_mean": round(mean_within_dtw, 4),
            "between_text_dtw_mean": round(between_dtw, 4),
            "dtw_separation": round(between_dtw / mean_within_dtw, 3)
            if mean_within_dtw > 0
            else 0.0,
            "f0_median_spread_hz": _spread(group_pitch),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    context = report["context"]
    totals = report["totals"]
    lines: list[str] = []
    lines.append("# Qwen3-TTS diversity probe")
    lines.append("")
    for key in (
        "run_id",
        "model",
        "ref_audio",
        "ref_text",
        "clone_mode",
        "repeats",
        "batch_size",
        "sample_rate",
        "wall_sec",
        "sampling_defaults",
        "sampling_overrides",
    ):
        if key in context:
            lines.append(f"- {key}: {context[key]}")
    lines.append("")
    lines.append("## text ごと")
    lines.append("")
    lines.append(
        "| text | n | uniq | 尺 mean | sd | cv | F0 med | sd | 有声区間 | within DTW |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in report["groups"]:
        duration = group["speech_sec"]
        pitch = group["f0_median_hz"]
        runs = group["n_voiced_runs"]
        lines.append(
            "| {text} | {n} | {uniq} | {dm:.3f} | {ds:.3f} | {dc:.3f} | "
            "{pm:.1f} | {ps:.1f} | {rm:.1f} | {dtw:.3f} |".format(
                text=group["text"],
                n=group["n"],
                uniq=group["unique_audio"],
                dm=duration["mean"],
                ds=duration["sd"],
                dc=duration["cv"],
                pm=pitch["mean"],
                ps=pitch["sd"],
                rm=runs["mean"],
                dtw=group["within_dtw"],
            )
        )
    lines.append("")
    lines.append("## まとめ")
    lines.append("")
    lines.append(f"- 合成 {totals['samples']} 本 / text {totals['texts']} 種")
    lines.append(
        f"- 同一音声: {totals['duplicate_audio']} 本 "
        f"(uniq {totals['unique_audio']})  <- ここが大きいならサンプリングは効いていない"
    )
    lines.append(
        f"- text 内の尺のばらつき: cv {totals['within_text_duration_cv_mean']:.3f} "
        f"(sd {totals['within_text_duration_sd_mean_sec']:.3f}s)"
    )
    lines.append(f"- text 間の尺のへだたり: sd {totals['between_text_duration_sd_sec']:.3f}s")
    lines.append(
        f"- 尺の分離比 (text 間 sd / text 内 sd): {totals['duration_separation']:.2f}"
    )
    lines.append(
        f"- 形の分離比 (DTW, text 間 {totals['between_text_dtw_mean']:.3f} / "
        f"text 内 {totals['within_text_dtw_mean']:.3f}): {totals['dtw_separation']:.2f}"
    )
    lines.append("")
    lines.append("読み方: 分離比が 1 前後なら、書き方を変えても同じ語を投げ直すのと")
    lines.append("変わらない。大きいほど、多様性は入力テキスト側で作れるということ。")
    lines.append("text 内 cv が 0 に近ければ、同じ文字列をいくら投げても貯まらない。")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 合成して回す
# --------------------------------------------------------------------------
def write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> str:
    """16bit PCM で書き、その量子化後の sha1 を返す(同一出力の検出用)。"""
    clipped = np.clip(np.asarray(signal, dtype=np.float64).reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return hashlib.sha1(pcm.tobytes()).hexdigest()


def load_texts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    texts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not texts:
        raise SystemExit(f"text が 1 つも読めませんでした: {path}")
    return texts


def slug(text: str, index: int) -> str:
    safe = "".join(ch for ch in text if ch.isalnum())
    return f"{index:04d}_{safe or 'x'}"


def apply_sampling_overrides(backend: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """stage-0 の SamplingParams に上書きを差し込む。

    production の backend には手を入れない。ここで差すのは、既定値のままで
    散らばらなかったときに「temperature を上げれば散るのか」を 1 回の実行で
    確かめるため。効かなかった項目は報告に残す(黙って無視しない)。
    """
    original = backend._sampling_params_list
    applied: dict[str, Any] = {}
    rejected: list[str] = []

    def patched() -> list[Any] | None:
        params = original()
        if params is None:
            return None
        for name, value in overrides.items():
            if not hasattr(params[0], name):
                if name not in rejected:
                    rejected.append(name)
                continue
            setattr(params[0], name, value)
            applied[name] = value
        return params

    backend._sampling_params_list = patched  # type: ignore[method-assign]
    return {"applied": applied, "rejected": rejected}


def describe_defaults(backend: Any) -> Any:
    defaults = getattr(backend._omni, "default_sampling_params_list", None)
    if not defaults:
        return "(vLLM-Omni exposes none)"
    described = []
    for params in defaults:
        fields = {}
        for name in ("temperature", "top_p", "top_k", "seed", "repetition_penalty", "max_tokens"):
            if hasattr(params, name):
                fields[name] = getattr(params, name)
        described.append(fields)
    return described


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True, help="クローン参照 WAV(1 本)")
    parser.add_argument("--ref-text", default="", help="参照の書き起こし。in-context クローンでは必須")
    parser.add_argument("--x-vector-only", action="store_true", help="書き起こし無しのクローン")
    parser.add_argument("--texts-file", type=Path, default=None, help="1 行 1 text。既定は相槌 20 種")
    parser.add_argument("--repeats", type=int, default=10, help="同じ text を何本合成するか")
    parser.add_argument("--limit", type=int, default=0, help="合計本数の上限(0 で無制限)")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument(
        "--stage-config", type=Path, default=REPO_ROOT / "configs/qwen3_tts_v100_batch16.yaml"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512, help="相槌は短いので production より小さくてよい")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.ref_audio.is_file():
        raise SystemExit(f"参照 WAV がありません: {args.ref_audio}")
    if not args.x_vector_only and not args.ref_text.strip():
        raise SystemExit("in-context クローンには --ref-text が要ります(--x-vector-only なら不要)")
    if args.repeats < 1:
        raise SystemExit("--repeats は 1 以上")

    texts = load_texts(args.texts_file)
    # text ごとにまとめて投げると batch が 1 種類の text で埋まる。production と
    # 同じく混ざった batch にしたいので、text を一巡ずつ回しながら並べる。
    plan = [texts[i % len(texts)] for i in range(len(texts) * args.repeats)]
    if args.limit > 0:
        plan = plan[: args.limit]

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from qwen3_tts_vllm_backend import CloneReference, SynthesisRequest, VLLMQwen3TTS

    backend = VLLMQwen3TTS(
        model_id=args.model,
        dtype_str=args.dtype,
        speaker_user="Ono_Anna",
        speaker_moshi="Serena",
        language=args.language,
        instruct_user=None,
        instruct_moshi=None,
        batch_size=args.batch_size,
        stage_configs_path=args.stage_config,
        max_new_tokens=args.max_new_tokens,
        # 長さ順に並べ替えられると、どの text がどの batch に入るかが変わって
        # 再現できなくなる。ここでは投げた順のままにする。
        bucket_by_length=False,
        clone_refs={
            "moshi": CloneReference(
                ref_audio=str(args.ref_audio.resolve()),
                ref_text=args.ref_text.strip() or None,
            )
        },
        clone_x_vector_only=args.x_vector_only,
    )

    print(f"[diversity] loading {args.model}", flush=True)
    backend.load()
    defaults = describe_defaults(backend)
    print(f"[diversity] default sampling params: {defaults}", flush=True)

    overrides = {
        name: value
        for name, value in (
            ("temperature", args.temperature),
            ("top_p", args.top_p),
            ("top_k", args.top_k),
            ("seed", args.seed),
        )
        if value is not None
    }
    override_report: dict[str, Any] = {"applied": {}, "rejected": []}
    if overrides:
        override_report = apply_sampling_overrides(backend, overrides)

    requests = [SynthesisRequest(text=text, speaker_role="moshi") for text in plan]
    print(f"[diversity] synthesizing {len(requests)} samples", flush=True)
    started = time.perf_counter()
    audio = backend.synthesize_many(requests)
    wall_sec = time.perf_counter() - started
    sample_rate = backend.sample_rate
    print(f"[diversity] done in {wall_sec:.1f}s at {sample_rate} Hz", flush=True)

    out_dir = args.out_dir
    wav_dir = out_dir / "wav"
    grouped: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for index, (text, signal) in enumerate(zip(plan, audio)):
        path = wav_dir / f"{slug(text, index)}.wav"
        digest = write_wav(path, signal, sample_rate)
        entry = {
            "index": index,
            "text": text,
            "wav": str(path.relative_to(out_dir)),
            "audio_sha1": digest,
            "measure": measure(signal, sample_rate),
        }
        records.append(entry)
        group = grouped.setdefault(text, {"text": text, "samples": []})
        group["samples"].append(
            {**entry, "features": feature_sequence(signal, sample_rate)}
        )

    report = build_report(
        [grouped[text] for text in dict.fromkeys(plan)],
        {
            "run_id": out_dir.name,
            "model": args.model,
            "ref_audio": str(args.ref_audio),
            "ref_text": args.ref_text.strip() or "(x-vector only)",
            "clone_mode": "x-vector" if args.x_vector_only else "in-context",
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "sample_rate": sample_rate,
            "wall_sec": round(wall_sec, 2),
            "sampling_defaults": defaults,
            "sampling_overrides": override_report,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for entry in records:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text_report = format_report(report)
    (out_dir / "report.md").write_text(text_report, encoding="utf-8")
    print()
    print(text_report)
    print(f"[diversity] wav:    {wav_dir}")
    print(f"[diversity] report: {out_dir / 'report.md'}")

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
