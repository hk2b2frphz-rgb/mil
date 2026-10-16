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

温度を振ると、同じ text のまま group が分かれる(「うん @T1.3」)。同じ入力で
どこまで散らせるのかは、これで見る:

  $VLLM_PYTHON scripts/qwen3_tts_diversity_probe.py \
      --ref-audio ref.wav --ref-text "..." --out-dir <dir> \
      --texts うん --repeats 50 --temperature-sweep 0.7,1.0,1.3

測り方を変えたら、合成し直さずに測り直せる。作り直すと別の音になって前の結果と
比べられないので、wav はそのままにしておく。GPU も vLLM も要らない:

  uv run python scripts/qwen3_tts_diversity_probe.py --out-dir <dir> --reanalyze

vLLM-Omni は import が重いので、解析側の関数だけを import して使えるように
バックエンドの import は main() の中に置いてある(tests/ と --reanalyze がそれを使う)。
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
# ACF は真の周期の整数倍にもピークを作るので、1 フレームだけ見て最大値を取ると
# そこでオクターブが飛ぶ。初回の計測で「そっかー」の F0 sd が 66Hz まで開いた
# のがこれ。候補を複数残し、発話全体の代表 F0 から離れた候補に罰則を付けて
# 選び直す。罰則はスコア差に対して効かせるので、1 オクターブずれた候補は
# スコアが OCTAVE_PENALTY 以上勝っていないと選ばれない。
F0_CANDIDATES = 4
OCTAVE_PENALTY = 0.25
# text 内の F0 sd が平均のこの割合を超えたら、値ではなく計測を疑う目印を出す。
F0_SUSPECT_RATIO = 0.15
# group の中央値からこの倍率を超えて離れた尺は外れとして別勘定にする。相槌で
# 中央値の 2.5 倍というのは、散ったのではなく壊れている(低温での繰り返し、
# 無音の垂れ流し)方を先に疑う長さ。cv は外れ 1 本で簡単に跳ねるので、外れを
# 抜いた cv も並べて出す -- でないと「広がった」と「壊れた」を取り違える。
OUTLIER_RATIO = 2.5
# 無音判定は report_tts_smoke.py と同じ「ピークの 5%」。
SILENCE_RATIO = 0.05
# 有声区間をいくつに割るか。60ms 以上の無声が入ったら別の区間として数える。
# 「うん」は 1、「そっか」は促音で 2 になる、という粒度。
RUN_GAP_SEC = 0.06
# 拍の数。無声を挟まなくても、エネルギーが両隣の山のこの割合を下回るまで凹めば
# 別の拍として数える。「うんうん」がひと息の連続音になっていないか -- 有声区間
# だけでは 1 と出てしまう -- を見るための列。
BEAT_DIP_RATIO = 0.55
BEAT_FLOOR_RATIO = 0.30

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


def frame_f0_candidates(
    signal: np.ndarray, sample_rate: int
) -> list[list[tuple[float, float]]]:
    """フレームごとの F0 候補 [(Hz, 正規化自己相関スコア), ...]。降順。

    ピーク 1 つに決め打ちせず、局所最大をスコア順に F0_CANDIDATES 個まで残す。
    どれを採るかは estimate_f0 が発話全体を見てから決める。
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

    frames: list[list[tuple[float, float]]] = []
    for i in range(rms.size):
        if max_lag <= min_lag or rms[i] < floor or rms[i] <= 0.0:
            frames.append([])
            continue
        chunk = padded[i * hop : i * hop + window].astype(np.float64)
        chunk = chunk - chunk.mean()
        energy = float(np.dot(chunk, chunk))
        if energy <= 0.0:
            frames.append([])
            continue
        correlation = np.correlate(chunk, chunk, mode="full")[window - 1 :]
        search = correlation[min_lag : max_lag + 1] / energy
        if search.size < 3:
            frames.append([])
            continue

        maxima = [
            j
            for j in range(1, search.size - 1)
            if search[j] >= search[j - 1]
            and search[j] > search[j + 1]
            and search[j] >= F0_PEAK_THRESHOLD
        ]
        if not maxima:
            best = int(np.argmax(search))
            maxima = [best] if search[best] >= F0_PEAK_THRESHOLD else []
        maxima.sort(key=lambda j: float(search[j]), reverse=True)

        candidates: list[tuple[float, float]] = []
        for j in maxima[:F0_CANDIDATES]:
            lag = float(min_lag + j)
            if 0 < j < search.size - 1:
                left, center, right = search[j - 1], search[j], search[j + 1]
                denominator = left - 2.0 * center + right
                if denominator != 0.0:
                    lag += 0.5 * float(left - right) / float(denominator)
            if lag > 0:
                candidates.append((sample_rate / lag, float(search[j])))
        frames.append(candidates)
    return frames


def reference_f0(frames: Sequence[Sequence[tuple[float, float]]]) -> float:
    """発話の代表 F0。各フレームの最良候補の中央値。

    中央値なので、少数のフレームでオクターブが飛んでいても引きずられない。
    """
    best = [frame[0][0] for frame in frames if frame]
    return float(np.median(best)) if best else 0.0


def choose_candidate(
    candidates: Sequence[tuple[float, float]], reference: float
) -> float:
    """候補から 1 つ選ぶ。スコアから代表 F0 との隔たり(オクターブ)を引いて比べる。"""
    if not candidates:
        return 0.0
    if reference <= 0:
        return float(candidates[0][0])
    return float(
        max(
            candidates,
            key=lambda candidate: candidate[1]
            - OCTAVE_PENALTY * abs(math.log2(max(candidate[0], 1e-6) / reference)),
        )[0]
    )


def estimate_f0(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """フレームごとの F0[Hz]。無声フレームは 0。

    正規化自己相関で候補を出し、発話全体の代表 F0 に照らして選び直す。
    librosa も scipy も無い vLLM-Omni 環境で動かすため numpy だけで書いてある。
    """
    frames = frame_f0_candidates(signal, sample_rate)
    reference = reference_f0(frames)
    return np.array(
        [choose_candidate(frame, reference) for frame in frames], dtype=np.float64
    )


def energy_beats(rms: np.ndarray) -> int:
    """拍の数。無声を挟まない「うんうん」を 1 と 2 のどちらで出すかを決める列。

    有声区間 (voiced_runs) は無声が入らないと割れないので、ひと息で繋がった
    連続音は 1 になる。こちらはエネルギーの谷で割るので、繋がっていても
    山が 2 つあれば 2 と出る。
    """
    start, end = speech_bounds(rms)
    segment = rms[start:end]
    if segment.size == 0:
        return 0
    smooth = np.convolve(segment, np.ones(3) / 3.0, mode="same")
    peak = float(smooth.max())
    if peak <= 0.0:
        return 0
    floor = BEAT_FLOOR_RATIO * peak
    maxima = [
        i
        for i in range(1, smooth.size - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] >= floor
    ]
    if not maxima:
        return 1
    kept = [maxima[0]]
    for i in maxima[1:]:
        valley = float(smooth[kept[-1] : i + 1].min())
        if valley < BEAT_DIP_RATIO * min(float(smooth[kept[-1]]), float(smooth[i])):
            kept.append(i)
        elif smooth[i] > smooth[kept[-1]]:
            kept[-1] = i
    return len(kept)


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
    # 有声区間ごとの F0。「そっか」のように促音で 2 つに割れる語は、発話全体の
    # 中央値がどちらの島に付くかで大きく振れる -- それは声が暴れているのでは
    # なく要約の仕方の問題なので、いちばん長い島の F0 も併せて持っておく。
    run_f0 = [
        round(float(np.median(f0[run_start:run_end][f0[run_start:run_end] > 0])), 2)
        for run_start, run_end in runs
        if np.any(f0[run_start:run_end] > 0)
    ]
    # 先頭の島の F0。同じ語なら島の並び順は変わらないので、発話全体の中央値と
    # 違って draw をまたいで比べられる。「そっか」で振れているのが声なのか
    # 「そ」と「か」のどちらが長いかなのかは、この列が分ける。
    head = run_f0[0] if run_f0 else 0.0
    # 島から島への跳ね。日本語のアクセントはここに出る。
    step = 0.0
    if len(run_f0) >= 2:
        step = max(
            abs(float(_semitones(np.asarray(right), left)))
            for left, right in zip(run_f0, run_f0[1:])
            if left > 0
        )
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
        "f0_head_hz": round(head, 2),
        "f0_run_hz": run_f0,
        "f0_step_semitones": round(step, 2),
        "n_voiced_runs": len(runs),
        "n_beats": energy_beats(rms),
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


def listening_picks(samples: Sequence[dict[str, Any]]) -> dict[str, str]:
    """最短・中央・最長の wav。聴いて確かめるときに開く 3 本。"""
    ordered = sorted(samples, key=lambda s: s["measure"]["speech_sec"])
    if not ordered:
        return {}
    return {
        "shortest": ordered[0].get("wav", ""),
        "median": ordered[len(ordered) // 2].get("wav", ""),
        "longest": ordered[-1].get("wav", ""),
    }


def suspect_reason(samples: Sequence[dict[str, Any]], pitch: dict[str, float]) -> str:
    """F0 が開いた理由を、値そのものから当てる。

    オクターブなら各サンプルの F0 は group 中央値のちょうど 2 倍か半分の近くに
    溜まる。そうでなく、発話が 2 つ以上の有声島に割れていて島から島への跳ねが
    大きいなら、暴れているのは計測ではなくアクセント -- 発話全体の中央値が
    どちらの島に付くかで振れているだけで、F0 頭 の列を見れば落ち着いている。
    """
    if pitch["mean"] <= 0:
        return ""
    values = [s["measure"]["f0_median_hz"] for s in samples if s["measure"]["f0_median_hz"] > 0]
    if not values:
        return ""
    center = float(np.median(values))
    if center > 0 and any(
        abs(abs(math.log2(value / center)) - 1.0) < 0.15 for value in values
    ):
        return "octave"
    runs = statistics.fmean([s["measure"]["n_voiced_runs"] for s in samples])
    steps = statistics.fmean(
        [s["measure"].get("f0_step_semitones", 0.0) for s in samples]
    )
    if runs >= 1.8 and steps >= 3.0:
        return "accent"
    return "unstable"


def split_outliers(
    durations: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    """尺を (中心, 長すぎ, 短すぎ) に割る。基準は group 自身の中央値。"""
    usable = [value for value in durations if value > 0]
    if not usable:
        return [], [], []
    center = float(np.median(usable))
    if center <= 0:
        return list(durations), [], []
    long_tail = [value for value in durations if value > OUTLIER_RATIO * center]
    short_tail = [value for value in durations if value < center / OUTLIER_RATIO]
    kept = [
        value
        for value in durations
        if center / OUTLIER_RATIO <= value <= OUTLIER_RATIO * center
    ]
    return kept, long_tail, short_tail


def summarize_group(group: dict[str, Any]) -> dict[str, Any]:
    samples = group["samples"]
    durations = [s["measure"]["speech_sec"] for s in samples]
    kept, long_tail, short_tail = split_outliers(durations)
    pitches = [s["measure"]["f0_median_hz"] for s in samples if s["measure"]["f0_median_hz"] > 0]
    runs = [s["measure"]["n_voiced_runs"] for s in samples]
    beats = [s["measure"].get("n_beats", 0) for s in samples]
    digests = {s["audio_sha1"] for s in samples}
    heads = [s["measure"].get("f0_head_hz", 0.0) for s in samples]
    heads = [value for value in heads if value > 0]
    steps = [s["measure"].get("f0_step_semitones", 0.0) for s in samples]
    pitch = _spread(pitches)
    suspect = bool(pitch["mean"] > 0 and pitch["sd"] > F0_SUSPECT_RATIO * pitch["mean"])
    return {
        "key": group_key(group),
        "text": group["text"],
        "temperature": group.get("temperature"),
        "n": len(samples),
        "unique_audio": len(digests),
        "speech_sec": _spread(durations),
        # 外れを抜いた尺。cv の跳ねが本当の広がりなのか尾なのかはここで分かる。
        "speech_sec_trimmed": _spread(kept),
        "outlier_long": len(long_tail),
        "outlier_short": len(short_tail),
        "f0_median_hz": pitch,
        "f0_head_hz": _spread(heads),
        "f0_step_semitones": _spread(steps),
        # F0 が平均の F0_SUSPECT_RATIO を超えて開いたら、値をそのまま読まない。
        # 理由まで出すのは、オクターブの取り違え(計測の問題)と、アクセントで
        # 中央値が振れているだけ(声は落ち着いている)とで、次の手が違うため。
        "f0_suspect": suspect,
        "f0_suspect_reason": suspect_reason(samples, pitch) if suspect else "",
        "n_voiced_runs": _spread(runs),
        "n_beats": _spread(beats),
        "within_dtw": mean_pairwise_dtw([s["features"] for s in samples]),
        "listen": listening_picks(samples),
    }


def group_key(group: dict[str, Any]) -> str:
    temperature = group.get("temperature")
    return group["text"] if temperature is None else f"{group['text']} @T{temperature:g}"


def cross_group_dtw(groups: Sequence[dict[str, Any]], per_group: int = 3) -> float:
    """異なる text 同士の DTW 平均。text 間のへだたりの物差し。

    同じ text の温度ちがい同士は「text 間」ではないので除く。
    """
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


def _pooled_sd(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt((left["sd"] ** 2 + right["sd"] ** 2) / 2.0)


def add_nearest_separation(summaries: Sequence[dict[str, Any]]) -> None:
    """尺の平均がいちばん近い相手と、その隔たりを pooled sd で割った値を書き込む。

    全体の分離比 (text 間 sd / text 内 sd) だけだと、低い理由が「平均が近い」
    のか「text 内が広い」のか区別できない。こちらは 1 本引いたときに隣と
    見分けが付くかを直接表す: 1 を下回ったら分布は重なっている。
    """
    usable = [s for s in summaries if s["speech_sec"]["mean"] > 0]
    for summary in summaries:
        summary["nearest_text"] = ""
        summary["nearest_d"] = 0.0
    for summary in usable:
        others = [
            other for other in usable if other["key"] != summary["key"]
        ]
        if not others:
            continue
        nearest = min(
            others,
            key=lambda other: abs(
                other["speech_sec"]["mean"] - summary["speech_sec"]["mean"]
            ),
        )
        pooled = _pooled_sd(summary["speech_sec"], nearest["speech_sec"])
        summary["nearest_text"] = nearest["key"]
        summary["nearest_d"] = round(
            abs(nearest["speech_sec"]["mean"] - summary["speech_sec"]["mean"]) / pooled, 3
        ) if pooled > 0 else 0.0


def temperature_effect(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """同じ text を温度ちがいで並べたときの差。温度を振ったときだけ中身が入る。"""
    by_text: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        if summary.get("temperature") is None:
            continue
        by_text.setdefault(summary["text"], []).append(summary)
    rows: list[dict[str, Any]] = []
    for text, group in sorted(by_text.items()):
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda s: s["temperature"])
        rows.append(
            {
                "text": text,
                "temperatures": [s["temperature"] for s in group],
                "duration_cv": [s["speech_sec"]["cv"] for s in group],
                "duration_cv_trimmed": [
                    s.get("speech_sec_trimmed", s["speech_sec"])["cv"] for s in group
                ],
                "outliers": [
                    s.get("outlier_long", 0) + s.get("outlier_short", 0) for s in group
                ],
                "duration_mean": [s["speech_sec"]["mean"] for s in group],
                "duration_min": [s["speech_sec"]["min"] for s in group],
                "duration_max": [s["speech_sec"]["max"] for s in group],
                "within_dtw": [s["within_dtw"] for s in group],
                "unique_audio": [s["unique_audio"] for s in group],
                "n": [s["n"] for s in group],
            }
        )
    return rows


def build_report(
    grouped: Sequence[dict[str, Any]], context: dict[str, Any]
) -> dict[str, Any]:
    summaries = [summarize_group(group) for group in grouped]
    add_nearest_separation(summaries)

    within_cv = [s["speech_sec"]["cv"] for s in summaries if s["speech_sec"]["mean"] > 0]
    within_dtw = [s["within_dtw"] for s in summaries]
    group_means = [s["speech_sec"]["mean"] for s in summaries if s["speech_sec"]["mean"] > 0]
    group_pitch = [s["f0_median_hz"]["mean"] for s in summaries if s["f0_median_hz"]["mean"] > 0]
    within_sd = [s["speech_sec"]["sd"] for s in summaries if s["speech_sec"]["mean"] > 0]
    nearest = [s["nearest_d"] for s in summaries if s["nearest_d"] > 0]

    total = sum(s["n"] for s in summaries)
    unique = sum(s["unique_audio"] for s in summaries)
    mean_within_sd = statistics.fmean(within_sd) if within_sd else 0.0
    between_sd = statistics.pstdev(group_means) if len(group_means) > 1 else 0.0
    mean_within_dtw = statistics.fmean(within_dtw) if within_dtw else 0.0
    between_dtw = cross_group_dtw(grouped)

    return {
        "context": context,
        "groups": summaries,
        "temperature_effect": temperature_effect(summaries),
        "totals": {
            "samples": total,
            "unique_audio": unique,
            "duplicate_audio": total - unique,
            "outliers": sum(
                s.get("outlier_long", 0) + s.get("outlier_short", 0) for s in summaries
            ),
            "groups": len(summaries),
            "texts": len({s["text"] for s in summaries}),
            "within_group_duration_cv_mean": round(
                statistics.fmean(within_cv) if within_cv else 0.0, 4
            ),
            "within_group_duration_sd_mean_sec": round(mean_within_sd, 4),
            "between_group_duration_sd_sec": round(between_sd, 4),
            "duration_separation": round(between_sd / mean_within_sd, 3)
            if mean_within_sd > 0
            else 0.0,
            "nearest_d_median": round(statistics.median(nearest), 3) if nearest else 0.0,
            "nearest_d_overlapping": sum(1 for d in nearest if d < 1.0),
            "within_group_dtw_mean": round(mean_within_dtw, 4),
            "between_text_dtw_mean": round(between_dtw, 4),
            "dtw_separation": round(between_dtw / mean_within_dtw, 3)
            if mean_within_dtw > 0
            else 0.0,
            "f0_suspect_groups": [
                f"{s['key']}({s['f0_suspect_reason']})"
                for s in summaries
                if s["f0_suspect"]
            ],
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
        "temperatures",
        "batch_size",
        "sample_rate",
        "wall_sec",
        "sec_per_sample",
        "sampling_defaults",
        "sampling_overrides",
        "reanalyzed_from",
    ):
        if key in context:
            lines.append(f"- {key}: {context[key]}")
    lines.append("")

    lines.append("## 尺")
    lines.append("")
    lines.append("| group | n | mean | sd | cv | cv* | 外れ | min | max | 隣 | d |")
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
    )
    for group in report["groups"]:
        duration = group["speech_sec"]
        trimmed = group.get("speech_sec_trimmed", duration)
        outliers = group.get("outlier_long", 0) + group.get("outlier_short", 0)
        lines.append(
            "| {key} | {n} | {mean:.3f} | {sd:.3f} | {cv:.3f} | {cvt:.3f} | {out} | "
            "{low:.3f} | {high:.3f} | {near} | {d:.2f} |".format(
                key=group["key"],
                n=group["n"],
                mean=duration["mean"],
                sd=duration["sd"],
                cv=duration["cv"],
                cvt=trimmed["cv"],
                out=outliers,
                low=duration["min"],
                high=duration["max"],
                near=group["nearest_text"] or "-",
                d=group["nearest_d"],
            )
        )
    lines.append("")

    lines.append("## 韻律と形")
    lines.append("")
    lines.append(
        "| group | F0 med | sd | F0 頭 | sd | 跳ね(st) | 拍 | 有声区間 | uniq | within DTW | |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for group in report["groups"]:
        pitch = group["f0_median_hz"]
        head = group.get("f0_head_hz", {"mean": 0.0, "sd": 0.0})
        lines.append(
            "| {key} | {pm:.1f} | {ps:.1f} | {mm:.1f} | {ms:.1f} | {step:.1f} | "
            "{beats:.1f} | {runs:.1f} | {uniq} | {dtw:.3f} | {flag} |".format(
                key=group["key"],
                pm=pitch["mean"],
                ps=pitch["sd"],
                mm=head["mean"],
                ms=head["sd"],
                step=group.get("f0_step_semitones", {"mean": 0.0})["mean"],
                beats=group["n_beats"]["mean"],
                runs=group["n_voiced_runs"]["mean"],
                uniq=group["unique_audio"],
                dtw=group["within_dtw"],
                flag=(
                    f"要確認({group['f0_suspect_reason']})"
                    if group["f0_suspect"]
                    else ""
                ),
            )
        )
    lines.append("")

    if report.get("temperature_effect"):
        lines.append("## temperature を振ったとき (同じ text)")
        lines.append("")
        lines.append(
            "| text | T | n | 尺 mean | cv | cv* | 外れ | min | max | uniq | within DTW |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in report["temperature_effect"]:
            for i, temperature in enumerate(row["temperatures"]):
                lines.append(
                    "| {text} | {t:g} | {n} | {mean:.3f} | {cv:.3f} | {cvt:.3f} | "
                    "{out} | {low:.3f} | {high:.3f} | {uniq} | {dtw:.3f} |".format(
                        text=row["text"] if i == 0 else "",
                        t=temperature,
                        n=row["n"][i],
                        mean=row["duration_mean"][i],
                        cv=row["duration_cv"][i],
                        cvt=row["duration_cv_trimmed"][i],
                        out=row["outliers"][i],
                        low=row["duration_min"][i],
                        high=row["duration_max"][i],
                        uniq=row["unique_audio"][i],
                        dtw=row["within_dtw"][i],
                    )
                )
        lines.append("")

    lines.append("## 聴くなら")
    lines.append("")
    for group in report["groups"]:
        listen = group.get("listen") or {}
        if listen:
            lines.append(
                f"- {group['key']}: 最短 {listen.get('shortest','')} / "
                f"中央 {listen.get('median','')} / 最長 {listen.get('longest','')}"
            )
    lines.append("")

    lines.append("## まとめ")
    lines.append("")
    lines.append(
        f"- 合成 {totals['samples']} 本 / group {totals['groups']} 種 "
        f"(text {totals['texts']} 種)"
    )
    lines.append(
        f"- 同一音声: {totals['duplicate_audio']} 本 "
        f"(uniq {totals['unique_audio']})  <- ここが大きいならサンプリングは効いていない"
    )
    lines.append(
        f"- 尺の外れ: {totals['outliers']} 本 "
        f"(中央値の {OUTLIER_RATIO:g} 倍を超えた/下回った)  <- 壊れている疑い"
    )
    lines.append(
        f"- group 内の尺のばらつき: cv {totals['within_group_duration_cv_mean']:.3f} "
        f"(sd {totals['within_group_duration_sd_mean_sec']:.3f}s)"
    )
    lines.append(f"- group 間の尺のへだたり: sd {totals['between_group_duration_sd_sec']:.3f}s")
    lines.append(
        f"- 尺の分離比 (group 間 sd / group 内 sd): {totals['duration_separation']:.2f}"
    )
    lines.append(
        f"- 隣との隔たり d の中央値: {totals['nearest_d_median']:.2f} "
        f"(d<1 で重なっている group: {totals['nearest_d_overlapping']}/{totals['groups']})"
    )
    lines.append(
        f"- 形の分離比 (DTW, text 間 {totals['between_text_dtw_mean']:.3f} / "
        f"group 内 {totals['within_group_dtw_mean']:.3f}): {totals['dtw_separation']:.2f}"
    )
    if totals["f0_suspect_groups"]:
        lines.append(
            "- F0 要確認 (sd が平均の "
            f"{F0_SUSPECT_RATIO:.0%} 超): {', '.join(totals['f0_suspect_groups'])}"
        )
        lines.append(
            "  octave=計測の取り違え / accent=島から島への跳ねで中央値が振れている"
            "(F0 頭 の列を見る) / unstable=どちらでもない"
        )
    lines.append("")
    lines.append("読み方:")
    lines.append("- group 内 cv が 0 に近ければ、同じ入力をいくら投げても貯まらない。")
    lines.append(
        f"- cv* は中央値の {OUTLIER_RATIO:g} 倍を外れた尺を抜いた cv。cv が大きいのに"
    )
    lines.append("  cv* が小さければ、広がったのではなく尾を引いているだけ。外れの")
    lines.append("  列が立っている group は、数える前に聴いて壊れていないか見ること。")
    lines.append("- 分離比が低い理由は 2 つある: 平均が近いか、group 内が広いか。")
    lines.append("  区別は d の列で付く。d<1 は隣と分布が重なっている、つまり 1 本引いて")
    lines.append("  狙った長さは出ない -- 引いて測って選ぶ運用が要る、ということ。")
    lines.append("- 平均そのものは別に見る。表記を伸ばして mean が動いているなら、")
    lines.append("  重なっていても「あたり」を付ける手段としては効いている。")
    if totals["texts"] == 1:
        lines.append("- この実行は text が 1 種なので、上の分離比と d が比べているのは")
        lines.append("  書き方ではなく温度ちがい。書き方の効き目は別の実行で見ること。")
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
    # <語>\t<重み> の形（real_backchannel_dist.tsv）もそのまま渡せるように、
    # 1 列目だけ取る。重みはここでは使わない -- 何本ずつ作るかは --repeats で
    # 決めており、バンクは語ごとに同じ深さで持っておく方が引きやすい。
    texts = [
        line.split("\t")[0].strip()
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
            # A sweep rewrites the same key per pass; keep every value it used,
            # or the report would claim only the last one was ever applied.
            seen = applied.setdefault(name, [])
            if value not in seen:
                seen.append(value)
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


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16bit PCM の mono wav を [-1, 1] の float で読む。"""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(f"16bit PCM ではありません: {path}")
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
    signal = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    return signal, sample_rate


def group_samples(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """(text, temperature) ごとにまとめる。出てきた順を保つ。"""
    groups: dict[tuple[str, Any], dict[str, Any]] = {}
    for record in records:
        key = (record["text"], record.get("temperature"))
        group = groups.setdefault(
            key,
            {"text": record["text"], "temperature": record.get("temperature"), "samples": []},
        )
        group["samples"].append(record)
    return list(groups.values())


def run_reanalysis(out_dir: Path) -> int:
    """合成し直さずに測り直す。

    F0 の取り方を変えるたびに 200 本を作り直すのは無駄で、作り直したら別の
    音になってしまうので前の結果と比べられない。wav はそのまま、測り方だけを
    差し替えたいときはこちら。
    """
    samples_path = out_dir / "samples.jsonl"
    if not samples_path.is_file():
        raise SystemExit(f"samples.jsonl がありません: {samples_path}")

    records: list[dict[str, Any]] = []
    sample_rate = 0
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            wav_path = out_dir / entry["wav"]
            if not wav_path.is_file():
                raise SystemExit(f"wav がありません: {wav_path}")
            signal, rate = read_wav(wav_path)
            sample_rate = sample_rate or rate
            entry["measure"] = measure(signal, rate)
            entry["features"] = feature_sequence(signal, rate)
            records.append(entry)
    if not records:
        raise SystemExit(f"samples.jsonl が空です: {samples_path}")
    print(f"[diversity] re-measured {len(records)} wav from {out_dir}", flush=True)

    previous: dict[str, Any] = {}
    report_path = out_dir / "report.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8")).get("context", {})
    context = {
        **{k: v for k, v in previous.items() if k != "reanalyzed_from"},
        "run_id": out_dir.name,
        "reanalyzed_from": str(samples_path),
        "sample_rate": sample_rate,
    }
    write_outputs(out_dir, records, group_samples(records), context)
    return 0


def write_outputs(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    grouped: Sequence[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    report = build_report(grouped, context)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for entry in records:
            handle.write(
                json.dumps(
                    {k: v for k, v in entry.items() if k != "features"},
                    ensure_ascii=False,
                )
                + "\n"
            )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text_report = format_report(report)
    (out_dir / "report.md").write_text(text_report, encoding="utf-8")
    print()
    print(text_report)
    print(f"[diversity] wav:    {out_dir / 'wav'}")
    print(f"[diversity] report: {out_dir / 'report.md'}")
    return report


def parse_temperatures(raw: str) -> list[float | None]:
    if not raw.strip():
        return [None]
    values = [float(part) for part in raw.split(",") if part.strip()]
    if not values:
        return [None]
    return values


def run_synthesis(args: argparse.Namespace) -> int:
    if not args.ref_audio or not args.ref_audio.is_file():
        raise SystemExit(f"参照 WAV がありません: {args.ref_audio}")
    if not args.x_vector_only and not args.ref_text.strip():
        raise SystemExit("in-context クローンには --ref-text が要ります(--x-vector-only なら不要)")
    if args.repeats < 1:
        raise SystemExit("--repeats は 1 以上")

    if args.texts:
        texts = [text.strip() for text in args.texts.split(",") if text.strip()]
        if not texts:
            raise SystemExit("--texts が空です")
    else:
        texts = load_texts(args.texts_file)
    temperatures = parse_temperatures(args.temperature_sweep)

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

    # 1 つの dict を使い回し、温度ごとに書き換える。差し込みは呼ばれるたびに
    # この dict を読むので、エンジンは 1 回しか積まなくてよい。
    overrides: dict[str, Any] = {
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
    if overrides or any(t is not None for t in temperatures):
        override_report = apply_sampling_overrides(backend, overrides)

    out_dir = args.out_dir
    wav_dir = out_dir / "wav"
    records: list[dict[str, Any]] = []
    index = 0
    wall_sec = 0.0
    sample_rate = 0
    for temperature in temperatures:
        if temperature is not None:
            overrides["temperature"] = temperature
            print(f"[diversity] temperature={temperature:g}", flush=True)
        requests = [SynthesisRequest(text=text, speaker_role="moshi") for text in plan]
        print(f"[diversity] synthesizing {len(requests)} samples", flush=True)
        started = time.perf_counter()
        audio = backend.synthesize_many(requests)
        elapsed = time.perf_counter() - started
        wall_sec += elapsed
        sample_rate = backend.sample_rate
        print(f"[diversity] pass done in {elapsed:.1f}s at {sample_rate} Hz", flush=True)

        for text, signal in zip(plan, audio):
            path = wav_dir / f"{slug(text, index)}.wav"
            entry = {
                "index": index,
                "text": text,
                "temperature": temperature,
                "wav": str(path.relative_to(out_dir)),
                "audio_sha1": write_wav(path, signal, sample_rate),
                "measure": measure(signal, sample_rate),
                "features": feature_sequence(signal, sample_rate),
            }
            records.append(entry)
            index += 1

    context = {
        "run_id": out_dir.name,
        "model": args.model,
        "ref_audio": str(args.ref_audio),
        "ref_text": args.ref_text.strip() or "(x-vector only)",
        "clone_mode": "x-vector" if args.x_vector_only else "in-context",
        "repeats": args.repeats,
        "temperatures": [t for t in temperatures] if any(
            t is not None for t in temperatures
        ) else "(engine default)",
        "batch_size": args.batch_size,
        "sample_rate": sample_rate,
        "wall_sec": round(wall_sec, 2),
        "sec_per_sample": round(wall_sec / len(records), 3) if records else 0.0,
        "sampling_defaults": defaults,
        "sampling_overrides": override_report,
    }
    write_outputs(out_dir, records, group_samples(records), context)
    backend.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="合成せず、--out-dir の wav を測り直して報告だけ作り直す",
    )
    parser.add_argument("--ref-audio", type=Path, default=None, help="クローン参照 WAV(1 本)")
    parser.add_argument("--ref-text", default="", help="参照の書き起こし。in-context クローンでは必須")
    parser.add_argument("--x-vector-only", action="store_true", help="書き起こし無しのクローン")
    parser.add_argument("--texts-file", type=Path, default=None, help="1 行 1 text。既定は相槌 20 種")
    parser.add_argument("--texts", default="", help="カンマ区切りの text。--texts-file より優先")
    parser.add_argument("--repeats", type=int, default=10, help="同じ text を何本合成するか")
    parser.add_argument("--limit", type=int, default=0, help="1 温度あたりの本数の上限(0 で無制限)")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument(
        "--stage-config", type=Path, default=REPO_ROOT / "configs/qwen3_tts_v100_batch16.yaml"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512, help="相槌は短いので production より小さくてよい")
    parser.add_argument("--language", default="Japanese")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--temperature-sweep",
        default="",
        help="カンマ区切りの温度。同じ plan を温度ごとに 1 巡ずつ回す(例: 0.7,1.0,1.3)",
    )
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.reanalyze:
        return run_reanalysis(args.out_dir)
    return run_synthesis(args)


if __name__ == "__main__":
    raise SystemExit(main())
