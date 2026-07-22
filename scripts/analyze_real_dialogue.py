#!/usr/bin/env python3
"""
1ch ロールプレイ実対話 WAV の聴取確認用分析(ステージ1: 分離なし)。

pyannote ダイアライゼーション + faster-whisper 書き起こしで話者タイムラインを
作り、耳で確認できる WAV 群を出力する。音源分離(重畳区間の 2ch 化)は
ステージ2として別スクリプトに分け、まず solo トラックと発話ごとの切り出しを
聴いてダイアライゼーション精度を確認する想定。

入力 WAV ごとに <out_dir>/<wav_stem>/ 配下へ:
  - speaker_A_solo.wav   話者Aの区間だけ残して他を無音化した確認用トラック
  - speaker_B_solo.wav   同・話者B
  - stereo_diarized.wav  L=話者A / R=話者B (重畳区間は両chに混合が残る)
  - segments/A/, segments/B/
        発話ごとの切り出し WAV。ファイル名に通し番号・開始時刻・長さ・
        重畳(ov)/相槌(aizuchi)フラグ・ASRテキスト先頭を入れる。例:
        0042_07m31.2s_0.6s_ov_aizuchi_ええ.wav
  - timeline.jsonl       {speaker, start, end, text, overlap_sec, is_aizuchi}
  - stats.json           相槌頻度・応答gap・重畳長の分布(合成側パラメータ校正用)

使い方:
    uv run python scripts/analyze_real_dialogue.py 対話1.wav [対話2.wav ...] \
        --out-dir data/real_dialogue/pilot

ダイアライゼーションは nvidia/diar_streaming_sortformer_4spk-v2 (NeMo)。
gated ではないので HF の同意・トークンは不要。ライセンスは CC-BY-4.0(商用可)。
NeMo は既存依存 (nemo_toolkit[asr]) に含まれるため追加インストールも不要。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.analyze_backchannels import AIZUCHI_PATTERNS
except ImportError:  # スクリプト直接実行時（scripts/ が sys.path 先頭）
    from analyze_backchannels import AIZUCHI_PATTERNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Heartbeat:
    """内部進捗を出さない長い処理(diarize など)の生存確認用。

    with ブロックの間、interval 秒ごとに経過秒をログへ出す。数値が増え続けて
    いれば実行中、増えなければハングと判別できる。daemon スレッドなので本体の
    終了を妨げない。
    """

    def __init__(self, label: str, interval: float = 30.0):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Heartbeat":
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            logger.info("%s ... 実行中(%.0f 秒経過)", self.label, time.monotonic() - self._t0)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        logger.info("%s 完了(%.0f 秒)", self.label, time.monotonic() - self._t0)

COMPILED_AIZUCHI = [(name, re.compile(pat)) for name, pat in AIZUCHI_PATTERNS]

# 相槌判定: この秒数以下の短いセグメントのみ対象にする。
AIZUCHI_MAX_SEC = 3.0
# クリップ切り出し時に前後へ足すマージン(語頭・語尾の欠け防止)。
CLIP_MARGIN_SEC = 0.08

# 相槌パターン照合の前に除去する記号類。
_STRIP_RE = re.compile(r"[\s、。！？!?,.…・「」『』()（）]")
# Windows で使えない文字などファイル名に入れない文字。
_FNAME_RE = re.compile(r"[\\/:*?\"<>|\s]+")


@dataclass
class Segment:
    speaker: str          # "A" | "B"
    start: float
    end: float
    text: str = ""
    overlap_sec: float = 0.0
    is_aizuchi: bool = False
    aizuchi_labels: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def is_aizuchi_text(text: str) -> list[str]:
    """テキスト全体が相槌語彙の連結で説明できる場合、マッチしたラベル列を返す。"""
    s = _STRIP_RE.sub("", text)
    if not s:
        return []
    labels: list[str] = []
    pos = 0
    while pos < len(s):
        for name, pat in COMPILED_AIZUCHI:
            m = pat.match(s, pos)
            if m and m.end() > pos:
                labels.append(name)
                pos = m.end()
                break
        else:
            return []
    return labels


DIARIZATION_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"


DIARIZATION_SR = 16000  # Sortformer(NEST/FastConformer)は 16kHz モノラル入力。


def run_diarization(
    audio: np.ndarray, sr: int, device: str
) -> list[tuple[str, float, float]]:
    """Sortformer で (label, start, end) 列を得る。gated でなくトークン不要。

    audio はダウンミックス済みモノラル配列。Sortformer は 16kHz モノラルの
    (batch, time) を要求するため、16kHz へリサンプルした一時 WAV を書いて
    diarize に渡す(元ファイルを直接渡すとステレオのまま読まれ、
    "input shape found (1, T, 2)" で落ちる)。
    """
    import os
    import tempfile

    from nemo.collections.asr.models import SortformerEncLabelModel

    mono = audio
    if sr != DIARIZATION_SR:
        import librosa

        logger.info("diarize 用に %d→%d Hz へリサンプル中...", sr, DIARIZATION_SR)
        mono = librosa.resample(mono, orig_sr=sr, target_sr=DIARIZATION_SR)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, mono, DIARIZATION_SR)

    try:
        logger.info("%s をロード中(初回はダウンロードあり)...", DIARIZATION_MODEL)
        with Heartbeat("モデルロード", interval=15.0):
            model = SortformerEncLabelModel.from_pretrained(DIARIZATION_MODEL)
            model.eval()
            if device.startswith("cuda"):
                model = model.to(device)

        logger.info("diarize 実行中(長い音声はここが最も時間を要します)...")
        with Heartbeat("diarize"):
            outputs = model.diarize(audio=[tmp_path], batch_size=1)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # 出力は入力ファイルごとの ["<start> <end> speaker_k", ...]。
    lines = outputs[0] if outputs and isinstance(outputs[0], list) else outputs
    return parse_diarization_lines(lines)


def parse_diarization_lines(lines: list[str]) -> list[tuple[str, float, float]]:
    raw: list[tuple[str, float, float]] = []
    for line in lines:
        parts = str(line).split()
        if len(parts) != 3:
            logger.warning("解釈できない diarization 行を無視: %r", line)
            continue
        start, end, label = float(parts[0]), float(parts[1]), parts[2]
        if end > start:
            raw.append((label, start, end))
    return raw


def diarization_to_segments(
    raw: list[tuple[str, float, float]],
    min_duration: float,
    num_speakers: int,
) -> list[Segment]:
    """(label, start, end) 列 → Segment 列。話者は総発話時間の多い順に A, B, ...。

    Sortformer は最大4話者を検出しうるので、総発話時間上位 num_speakers 名
    以外のセグメントは捨てる(誤検出の断片であることが多い)。
    """
    totals: dict[str, float] = {}
    for label, s, e in raw:
        totals[label] = totals.get(label, 0.0) + (e - s)
    order = sorted(totals, key=lambda k: -totals[k])
    dropped = order[num_speakers:]
    if dropped:
        logger.warning(
            "話者 %d 名を検出。上位 %d 名以外を破棄: %s",
            len(order), num_speakers,
            {lab: round(totals[lab], 1) for lab in dropped},
        )
    rename = {label: chr(ord("A") + i) for i, label in enumerate(order[:num_speakers])}
    segments = [
        Segment(speaker=rename[label], start=s, end=e)
        for label, s, e in raw
        if label in rename and (e - s) >= min_duration
    ]
    segments.sort(key=lambda seg: seg.start)
    return segments


def annotate_overlaps(segments: list[Segment]) -> list[tuple[float, float]]:
    """異話者セグメント間の重畳区間を求め、各セグメントに overlap_sec を記録する。"""
    regions: list[tuple[float, float]] = []
    for i, a in enumerate(segments):
        for b in segments[i + 1 :]:
            if b.start >= a.end:
                break
            if a.speaker == b.speaker:
                continue
            lo, hi = max(a.start, b.start), min(a.end, b.end)
            if hi > lo:
                regions.append((lo, hi))
                a.overlap_sec += hi - lo
                b.overlap_sec += hi - lo
    regions.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in regions:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def transcribe_segments(
    audio: np.ndarray,
    sr: int,
    segments: list[Segment],
    model_name: str,
    device: str,
) -> None:
    from faster_whisper import WhisperModel

    compute_type = "float16" if device.startswith("cuda") else "int8"
    logger.info("faster-whisper %s (%s, %s) をロード中...", model_name, device, compute_type)
    model = WhisperModel(
        model_name, device="cuda" if device.startswith("cuda") else "cpu",
        compute_type=compute_type,
    )

    import librosa

    total = len(segments)
    logger.info("書き起こし対象 %d セグメント", total)
    t0 = time.monotonic()
    for idx, seg in enumerate(segments, 1):
        lo = max(0, int((seg.start - CLIP_MARGIN_SEC) * sr))
        hi = min(audio.size, int((seg.end + CLIP_MARGIN_SEC) * sr))
        clip = audio[lo:hi]
        if sr != 16000:
            clip = librosa.resample(clip, orig_sr=sr, target_sr=16000)
        results, _ = model.transcribe(
            clip, language="ja", beam_size=5, condition_on_previous_text=False
        )
        seg.text = "".join(r.text for r in results).strip()
        if idx % 25 == 0 or idx == total:
            logger.info("書き起こし %d/%d 完了(%.0f 秒経過)", idx, total, time.monotonic() - t0)


def mark_aizuchi(segments: list[Segment]) -> None:
    for seg in segments:
        if seg.duration <= AIZUCHI_MAX_SEC and seg.text:
            labels = is_aizuchi_text(seg.text)
            if labels:
                seg.is_aizuchi = True
                seg.aizuchi_labels = labels


def build_solo_tracks(
    audio: np.ndarray, sr: int, segments: list[Segment]
) -> dict[str, np.ndarray]:
    """話者ごとに、その話者の区間以外を無音化した確認用トラックを作る。"""
    tracks: dict[str, np.ndarray] = {}
    for seg in segments:
        if seg.speaker not in tracks:
            tracks[seg.speaker] = np.zeros_like(audio)
        lo = max(0, int(seg.start * sr))
        hi = min(audio.size, int(seg.end * sr))
        tracks[seg.speaker][lo:hi] = audio[lo:hi]
    return tracks


def segment_filename(index: int, seg: Segment, with_text: bool) -> str:
    """発話 WAV のファイル名を組み立てる。

    <通し番号>_<開始 mm'm'ss.s's'>_<長さ>s[_ov][_aizuchi]_<テキスト先頭>.wav
    with_text=False(ASR 前の先出し保存)では _aizuchi とテキストを省く
    (どちらも書き起こしが要る)。_ov は diarization だけで確定する。
    """
    flags = ""
    if seg.overlap_sec > 0:
        flags += "_ov"
    if with_text and seg.is_aizuchi:
        flags += "_aizuchi"
    name = (
        f"{index:04d}_{int(seg.start // 60):02d}m{seg.start % 60:04.1f}s"
        f"_{seg.duration:.1f}s{flags}"
    )
    if with_text:
        text = _FNAME_RE.sub("", _STRIP_RE.sub("", seg.text))[:12]
        if text:
            name += f"_{text}"
    return name + ".wav"


def export_segment_wavs(
    audio: np.ndarray, sr: int, segments: list[Segment], out_dir: Path,
    with_text: bool,
) -> list[Path]:
    """発話ごとの WAV を segments/<話者>/ に書き出し、書き出したパス列を返す。"""
    paths: list[Path] = []
    for i, seg in enumerate(segments):
        seg_dir = out_dir / "segments" / seg.speaker
        seg_dir.mkdir(parents=True, exist_ok=True)
        lo = max(0, int((seg.start - CLIP_MARGIN_SEC) * sr))
        hi = min(audio.size, int((seg.end + CLIP_MARGIN_SEC) * sr))
        path = seg_dir / segment_filename(i, seg, with_text)
        sf.write(path, audio[lo:hi], sr)
        paths.append(path)
    return paths


def rename_segment_wavs(
    segments: list[Segment], paths: list[Path]
) -> None:
    """ASR 後、テキスト・相槌フラグを含む最終ファイル名へリネームする。"""
    for i, (seg, old) in enumerate(zip(segments, paths)):
        new = old.parent / segment_filename(i, seg, with_text=True)
        if new != old:
            old.rename(new)


def write_timeline(out_dir: Path, segments: list[Segment]) -> None:
    with (out_dir / "timeline.jsonl").open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(asdict(seg), ensure_ascii=False) + "\n")


def write_stats(
    out_dir: Path, segments: list[Segment],
    overlaps: list[tuple[float, float]], total_sec: float,
) -> None:
    (out_dir / "stats.json").write_text(
        json.dumps(compute_stats(segments, overlaps, total_sec),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_stats(
    segments: list[Segment],
    overlaps: list[tuple[float, float]],
    total_sec: float,
) -> dict:
    def pct(values: list[float]) -> dict:
        if not values:
            return {}
        qs = statistics.quantiles(values, n=10) if len(values) >= 2 else [values[0]] * 9
        return {
            "n": len(values),
            "mean": round(statistics.fmean(values), 3),
            "p10": round(qs[0], 3),
            "p50": round(statistics.median(values), 3),
            "p90": round(qs[8], 3),
        }

    # 話者交替 gap: 直前の異話者セグメント終端から次セグメント開始までの秒数。
    # 負値 = 重畳開始(食い気味の応答)。相槌セグメントは gap の対象から除く。
    gaps: list[float] = []
    main = [s for s in segments if not s.is_aizuchi]
    for prev, cur in zip(main, main[1:]):
        if prev.speaker != cur.speaker:
            gaps.append(round(cur.start - prev.end, 3))

    aizuchi = [s for s in segments if s.is_aizuchi]
    label_counts: dict[str, int] = {}
    for seg in aizuchi:
        for lab in seg.aizuchi_labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1

    per_speaker = {
        sp: {
            "segments": sum(1 for s in segments if s.speaker == sp),
            "speech_sec": round(sum(s.duration for s in segments if s.speaker == sp), 1),
            "aizuchi": sum(1 for s in aizuchi if s.speaker == sp),
        }
        for sp in sorted({s.speaker for s in segments})
    }

    minutes = total_sec / 60 if total_sec else 1.0
    return {
        "total_sec": round(total_sec, 1),
        "per_speaker": per_speaker,
        "aizuchi": {
            "count": len(aizuchi),
            "per_min": round(len(aizuchi) / minutes, 2),
            "overlapped": sum(1 for s in aizuchi if s.overlap_sec > 0),
            "clean": sum(1 for s in aizuchi if s.overlap_sec == 0),
            "by_label": dict(sorted(label_counts.items(), key=lambda kv: -kv[1])),
            "duration_sec": pct([s.duration for s in aizuchi]),
        },
        "turn_gap_sec": pct(gaps),
        "overlap": {
            "regions": len(overlaps),
            "total_sec": round(sum(hi - lo for lo, hi in overlaps), 1),
            "ratio_of_audio": round(
                sum(hi - lo for lo, hi in overlaps) / total_sec, 4
            ) if total_sec else 0.0,
            "duration_sec": pct([round(hi - lo, 3) for lo, hi in overlaps]),
        },
    }


def process_wav(wav_path: Path, out_root: Path, args: argparse.Namespace) -> Path:
    out_dir = out_root / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        logger.warning("%s は %dch です。モノラルへダウンミックスします。", wav_path.name, audio.shape[1])
        audio = audio.mean(axis=1)
    total_sec = audio.size / sr
    logger.info("%s: %.1f 秒 (%d Hz)", wav_path.name, total_sec, sr)

    logger.info("ダイアライゼーション実行中...")
    raw = run_diarization(audio, sr, args.device)
    segments = diarization_to_segments(raw, args.min_segment_sec, args.num_speakers)
    overlaps = annotate_overlaps(segments)
    logger.info("セグメント %d 件 / 重畳 %d 区間", len(segments), len(overlaps))

    # --- 分離結果を ASR より先に保存(遅い書き起こしを待たずに耳チェックできる) ---
    tracks = build_solo_tracks(audio, sr, segments)
    for speaker, track in tracks.items():
        sf.write(out_dir / f"speaker_{speaker}_solo.wav", track, sr)
    left = tracks.get("A", np.zeros_like(audio))
    right = tracks.get("B", np.zeros_like(audio))
    sf.write(out_dir / "stereo_diarized.wav", np.stack([left, right], axis=1), sr)

    seg_paths = export_segment_wavs(audio, sr, segments, out_dir, with_text=False)
    write_timeline(out_dir, segments)
    write_stats(out_dir, segments, overlaps, total_sec)
    logger.info(
        "分離結果を先に保存(発話 WAV %d 件): %s ← ここで耳チェック可能",
        len(seg_paths), out_dir,
    )

    if args.skip_asr:
        logger.info("--skip-asr: 書き起こしと相槌判定を省略")
    else:
        logger.info("書き起こし実行中...")
        transcribe_segments(audio, sr, segments, args.whisper_model, args.device)
        mark_aizuchi(segments)
        # ASR 結果を反映: 発話 WAV をテキスト付き名にリネームし、timeline/stats を更新。
        rename_segment_wavs(segments, seg_paths)
        write_timeline(out_dir, segments)
        write_stats(out_dir, segments, overlaps, total_sec)
        logger.info("書き起こし・相槌ラベルを反映して更新完了: %s", out_dir)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wavs", nargs="+", type=Path, help="入力 1ch WAV(複数可)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/real_dialogue"),
                        help="出力先ルート(既定: data/real_dialogue)")
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--whisper-model", default=None,
                        help="faster-whisper モデル名(既定: cuda なら large-v3, cpu なら small)")
    parser.add_argument("--device", default=None, help="cuda / cpu(既定: 自動判定)")
    parser.add_argument("--min-segment-sec", type=float, default=0.15,
                        help="これ未満のダイアライゼーション断片は捨てる")
    parser.add_argument("--skip-asr", action="store_true",
                        help="書き起こしを省略して切り出しだけ素早く確認する")
    args = parser.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    if args.whisper_model is None:
        args.whisper_model = "large-v3" if args.device.startswith("cuda") else "small"
    logger.info("device=%s whisper=%s", args.device, args.whisper_model)

    out_dirs = []
    for wav in args.wavs:
        if not wav.is_file():
            logger.error("見つかりません: %s", wav)
            continue
        out_dirs.append(process_wav(wav, args.out_dir, args))

    print("\n=== 耳で確認 ===")
    for d in out_dirs:
        print(f"{d.resolve()}")
        print(f"  solo:     speaker_A_solo.wav / speaker_B_solo.wav")
        print(f"  stereo:   stereo_diarized.wav (L=A / R=B)")
        print(f"  発話ごと: segments\\A\\ / segments\\B\\")


if __name__ == "__main__":
    main()
