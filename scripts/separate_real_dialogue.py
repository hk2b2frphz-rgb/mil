#!/usr/bin/env python3
"""
1ch ロールプレイ実対話 WAV のステージ2: 音源分離。

ステージ1 (scripts/analyze_real_dialogue.py) は「その話者の区間以外を無音化」
して solo トラックを作るが、1ch 録音では重畳区間に両者の声が同じ標本へ混ざって
いるため、時間マスクでは相手の声が消えない。評価データとしてはこれが致命的で、
ユーザー側トラックに相談員の相槌(「ええ」「はい」)が残ってしまう。モデルには
自分が担う役割の正解が入力に混入した状態になる。

そこで本スクリプトが重畳を含む全区間を音源分離し、話者ごとの独立トラックを
作る。ステージ1 のダイアライゼーション結果を「どちらの分離チャネルがどちらの
話者か」の教師として使うので、チャネル入れ替わり(permutation problem)を
チャンク境界ごとに決定的に解決できる。

入力: ステージ1 と同じ WAV と --out-dir。<out_dir>/<wav_stem>/ に
      timeline.jsonl が存在している必要がある。

出力を <out_dir>/<wav_stem>/ 配下へ追加:
  - separated/A.wav, separated/B.wav
        話者ごとの分離済みフルレングストラック。これが評価データの素材。
  - separated/stereo.wav        L=A / R=B の確認用
  - separation_report.json      チャンクごとの割り当てと信頼度
  - timeline_separated.jsonl    分離トラックで再書き起こしした最終タイムライン
  - stats_separated.json        再書き起こし後の統計
  - segments_separated/<話者>/  分離トラックからの発話 WAV(耳チェック用)
  - <話者>_segments_review.tsv  人手で間引くためのレビュー表(keep 列)

使い方:
    uv run python scripts/separate_real_dialogue.py 対話1.wav \\
        --out-dir data/real_dialogue/pilot

分離器は既定で speechbrain/sepformer-whamr16k (Apache-2.0, 商用可)。
`--sep-model` で差し替え可能。日本語で学習されたモデルではないが、話者分離は
音響的手がかりが主で言語依存が小さいため実用範囲。採用前に
segments_separated/ を耳で確認すること。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.analyze_real_dialogue import (
        CLIP_MARGIN_SEC,
        Heartbeat,
        Segment,
        compute_stats,
        mark_aizuchi,
        segment_filename,
        transcribe_segments,
    )
except ImportError:  # スクリプト直接実行時（scripts/ が sys.path 先頭）
    from analyze_real_dialogue import (  # type: ignore[no-redef]
        CLIP_MARGIN_SEC,
        Heartbeat,
        Segment,
        compute_stats,
        mark_aizuchi,
        segment_filename,
        transcribe_segments,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 分離器が要求するサンプリングレート。whamr16k 系は 16kHz。
DEFAULT_SEP_MODEL = "speechbrain/sepformer-whamr16k"
DEFAULT_SEP_SR = 16000

# チャネル割り当ての信頼度がこの値を下回るチャンクは、直前チャンクの割り当てを
# 引き継ぐ。片方の話者しか喋っていないチャンクでは相関が退化するため。
ASSIGN_MARGIN_MIN = 0.05


def load_timeline(analysis_dir: Path) -> list[Segment]:
    path = analysis_dir / "timeline.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"ステージ1 の出力が見つかりません: {path}\n"
            "先に scripts/analyze_real_dialogue.py を同じ --out-dir で実行してください。"
        )
    segments: list[Segment] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            segments.append(
                Segment(
                    speaker=str(row["speaker"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    text=str(row.get("text", "")),
                    overlap_sec=float(row.get("overlap_sec", 0.0)),
                    is_aizuchi=bool(row.get("is_aizuchi", False)),
                    aizuchi_labels=list(row.get("aizuchi_labels", [])),
                )
            )
    segments.sort(key=lambda s: s.start)
    return segments


def build_speaker_masks(
    segments: list[Segment], speakers: list[str], num_samples: int, sr: int
) -> dict[str, np.ndarray]:
    """話者ごとの 0/1 時間マスク。分離チャネルの帰属判定に使う教師信号。"""
    masks = {sp: np.zeros(num_samples, dtype=np.float32) for sp in speakers}
    for seg in segments:
        if seg.speaker not in masks:
            continue
        lo = max(0, int(seg.start * sr))
        hi = min(num_samples, int(seg.end * sr))
        if hi > lo:
            masks[seg.speaker][lo:hi] = 1.0
    return masks


def energy_envelope(x: np.ndarray, win: int) -> np.ndarray:
    """短時間 RMS 包絡。マスクとの相関を取るために時間解像度を落とす。"""
    if win < 1:
        win = 1
    pad = (-len(x)) % win
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    frames = x.reshape(-1, win)
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))


def downsample_mask(mask: np.ndarray, win: int) -> np.ndarray:
    if win < 1:
        win = 1
    pad = (-len(mask)) % win
    if pad:
        mask = np.concatenate([mask, np.zeros(pad, dtype=mask.dtype)])
    return mask.reshape(-1, win).mean(axis=1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """ゼロ分散に強い相関係数。無音チャンクでは 0 を返す。"""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def assign_channels(
    sources: np.ndarray,
    masks: list[np.ndarray],
    env_win: int,
) -> tuple[list[int], float]:
    """分離チャネル -> 話者インデックスの割り当てを決める。

    2 話者なら取りうる並びは 2 通りだけなので総当たりでよい。戻り値の
    order[i] は「話者 i に対応する分離チャネル番号」。margin は 2 つの並びの
    スコア差で、割り当ての信頼度として使う。
    """
    envs = [energy_envelope(src, env_win) for src in sources]
    mask_envs = [downsample_mask(m, env_win) for m in masks]

    identity = _corr(envs[0], mask_envs[0]) + _corr(envs[1], mask_envs[1])
    swapped = _corr(envs[1], mask_envs[0]) + _corr(envs[0], mask_envs[1])
    if identity >= swapped:
        return [0, 1], float(identity - swapped)
    return [1, 0], float(swapped - identity)


class SepformerSeparator:
    """SpeechBrain SepFormer ラッパ。2 話者混合を 2 チャネルへ分離する。"""

    def __init__(self, model_name: str, device: str, savedir: Path | None = None):
        self.model_name = model_name
        self.device = device
        self.savedir = savedir
        self._model = None

    def load(self) -> None:
        from speechbrain.inference.separation import SepformerSeparation

        logger.info("%s をロード中(初回はダウンロードあり)...", self.model_name)
        with Heartbeat("分離モデルロード", interval=15.0):
            self._model = SepformerSeparation.from_hparams(
                source=self.model_name,
                savedir=str(self.savedir) if self.savedir else None,
                run_opts={"device": self.device},
            )

    def separate(self, chunk: np.ndarray) -> np.ndarray:
        """(T,) の 16kHz モノラル -> (2, T)。"""
        import torch

        if self._model is None:
            raise RuntimeError("load() を先に呼んでください。")
        with torch.no_grad():
            batch = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
            est = self._model.separate_batch(batch)  # (1, T, n_src)
        out = est.squeeze(0).transpose(0, 1).cpu().numpy()  # (n_src, T)
        if out.shape[0] < 2:
            raise RuntimeError(f"分離器が {out.shape[0]} 音源しか返しませんでした。")
        # 長さがモデル都合で前後することがあるので入力長に合わせる。
        if out.shape[1] < len(chunk):
            out = np.pad(out, ((0, 0), (0, len(chunk) - out.shape[1])))
        return out[:2, : len(chunk)]


def separate_full(
    mono16k: np.ndarray,
    masks16k: list[np.ndarray],
    separator: SepformerSeparator,
    chunk_sec: float,
    overlap_sec: float,
    sr: int,
) -> tuple[np.ndarray, list[dict]]:
    """長尺音声をチャンク分割して分離し、話者順に並べて overlap-add で繋ぐ。

    チャンクごとに分離器が返すチャネル順は不定なので、ダイアライゼーション
    マスクとの相関で毎回並べ直す。これをしないとチャンク境界で話者が入れ替わる。
    """
    total = len(mono16k)
    chunk = max(1, int(chunk_sec * sr))
    overlap = max(0, min(int(overlap_sec * sr), chunk // 2))
    hop = chunk - overlap
    env_win = max(1, sr // 50)  # 20ms 解像度で相関を取る

    out = np.zeros((2, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    report: list[dict] = []
    prev_order: list[int] | None = None

    starts = list(range(0, max(1, total - overlap), hop))
    t0 = time.monotonic()
    for idx, start in enumerate(starts, 1):
        stop = min(total, start + chunk)
        piece = mono16k[start:stop]
        if piece.size == 0:
            continue
        sources = separator.separate(piece)
        order, margin = assign_channels(
            sources, [m[start:stop] for m in masks16k], env_win
        )
        inherited = False
        if margin < ASSIGN_MARGIN_MIN and prev_order is not None:
            # 片方しか喋っていない/無音のチャンク。ここで相関に従うと
            # コイン投げになるので、直前の割り当てを引き継いで連続性を守る。
            order = prev_order
            inherited = True
        prev_order = order

        # 端をなだらかにして繋ぎ目のクリックを防ぐ。
        win = np.ones(stop - start, dtype=np.float32)
        ramp = min(overlap, (stop - start) // 2)
        if ramp > 0:
            fade = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            if start > 0:
                win[:ramp] = fade
            if stop < total:
                win[-ramp:] = fade[::-1]

        for speaker_index, channel in enumerate(order):
            out[speaker_index, start:stop] += sources[channel] * win
        weight[start:stop] += win

        report.append(
            {
                "chunk": idx,
                "start_sec": round(start / sr, 3),
                "end_sec": round(stop / sr, 3),
                "order": order,
                "margin": round(margin, 4),
                "inherited_assignment": inherited,
            }
        )
        if idx % 20 == 0 or idx == len(starts):
            logger.info(
                "分離 %d/%d チャンク完了(%.0f 秒経過)",
                idx, len(starts), time.monotonic() - t0,
            )

    np.divide(out, np.maximum(weight, 1e-6), out=out)
    return out, report


def export_separated_segments(
    tracks: dict[str, np.ndarray],
    sr: int,
    segments: list[Segment],
    out_dir: Path,
    with_text: bool,
) -> dict[int, Path]:
    """分離トラックから発話 WAV を切り出す。

    ファイル名の通し番号は timeline 全体での位置(話者をまたぐ通し番号)にする。
    レビュー表の index 列と一致していないと、耳で聴いた WAV と間引く行を
    突き合わせられない。
    """
    paths: dict[int, Path] = {}
    for index, seg in enumerate(segments):
        track = tracks.get(seg.speaker)
        if track is None:
            continue
        seg_dir = out_dir / "segments_separated" / seg.speaker
        seg_dir.mkdir(parents=True, exist_ok=True)
        lo = max(0, int((seg.start - CLIP_MARGIN_SEC) * sr))
        hi = min(track.size, int((seg.end + CLIP_MARGIN_SEC) * sr))
        path = seg_dir / segment_filename(index, seg, with_text)
        sf.write(path, track[lo:hi], sr)
        paths[index] = path
    return paths


def write_review_tsv(
    out_dir: Path,
    speaker: str,
    segments: list[Segment],
    seg_paths: dict[int, Path],
) -> Path:
    """人手で間引くためのレビュー表。keep 列を 0 にした行は評価対象から外れる。

    TSV にしているのは Excel でもエディタでも開けるため。文字コードは
    Excel が素直に開ける UTF-8 BOM 付きにする。
    """
    path = out_dir / f"{speaker}_segments_review.tsv"
    header = [
        "keep", "index", "start_sec", "end_sec", "duration_sec",
        "overlap_sec", "is_aizuchi", "aizuchi_labels", "text", "wav",
    ]
    lines = ["\t".join(header)]
    for index, seg in enumerate(segments):
        if seg.speaker != speaker:
            continue
        wav = seg_paths.get(index)
        lines.append(
            "\t".join(
                [
                    "1",
                    str(index),
                    f"{seg.start:.3f}",
                    f"{seg.end:.3f}",
                    f"{seg.duration:.3f}",
                    f"{seg.overlap_sec:.3f}",
                    "1" if seg.is_aizuchi else "0",
                    ",".join(seg.aizuchi_labels),
                    seg.text.replace("\t", " ").replace("\n", " "),
                    wav.name if wav else "",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def process_wav(wav_path: Path, out_root: Path, args: argparse.Namespace) -> Path:
    out_dir = out_root / wav_path.stem
    if not out_dir.is_dir():
        raise SystemExit(f"ステージ1 の出力ディレクトリがありません: {out_dir}")

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        logger.warning("%s は %dch です。モノラルへダウンミックスします。", wav_path.name, audio.shape[1])
        audio = audio.mean(axis=1)
    total_sec = audio.size / sr
    logger.info("%s: %.1f 秒 (%d Hz)", wav_path.name, total_sec, sr)

    segments = load_timeline(out_dir)
    speakers = sorted({seg.speaker for seg in segments})
    if len(speakers) != 2:
        raise SystemExit(
            f"2 話者を想定していますが {len(speakers)} 話者でした: {speakers}"
        )
    logger.info("ステージ1 のセグメント %d 件 / 話者 %s", len(segments), speakers)

    import librosa

    sep_sr = args.sep_sample_rate
    if sr != sep_sr:
        logger.info("分離用に %d→%d Hz へリサンプル中...", sr, sep_sr)
        mono16k = librosa.resample(audio, orig_sr=sr, target_sr=sep_sr)
    else:
        mono16k = audio

    masks = build_speaker_masks(segments, speakers, len(mono16k), sep_sr)
    separator = SepformerSeparator(
        args.sep_model, args.device,
        savedir=args.sep_cache_dir / wav_path.stem if args.sep_cache_dir else None,
    )
    separator.load()

    logger.info("音源分離実行中(ここが最も時間を要します)...")
    with Heartbeat("分離"):
        separated16k, report = separate_full(
            mono16k, [masks[sp] for sp in speakers], separator,
            args.chunk_sec, args.chunk_overlap_sec, sep_sr,
        )

    inherited = sum(1 for row in report if row["inherited_assignment"])
    logger.info(
        "分離完了: %d チャンク(うち %d チャンクは割り当てを継承)", len(report), inherited
    )

    # 評価データは元のサンプリングレートで扱うので戻す。
    sep_dir = out_dir / "separated"
    sep_dir.mkdir(parents=True, exist_ok=True)
    tracks: dict[str, np.ndarray] = {}
    for index, speaker in enumerate(speakers):
        track = separated16k[index]
        if sr != sep_sr:
            track = librosa.resample(track, orig_sr=sep_sr, target_sr=sr)
        # リサンプルで長さが 1 標本ずれることがあるので元長へ揃える。
        if len(track) < audio.size:
            track = np.pad(track, (0, audio.size - len(track)))
        track = track[: audio.size]
        tracks[speaker] = track.astype(np.float32)
        sf.write(sep_dir / f"{speaker}.wav", tracks[speaker], sr)
    sf.write(
        sep_dir / "stereo.wav",
        np.stack([tracks[speakers[0]], tracks[speakers[1]]], axis=1),
        sr,
    )

    (out_dir / "separation_report.json").write_text(
        json.dumps(
            {
                "source_wav": str(wav_path),
                "sample_rate": sr,
                "separation_sample_rate": sep_sr,
                "model": args.sep_model,
                "chunk_sec": args.chunk_sec,
                "chunk_overlap_sec": args.chunk_overlap_sec,
                "speakers": speakers,
                "chunks": len(report),
                "inherited_assignments": inherited,
                "assignments": report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --- 分離トラックで再書き起こし ---
    # ステージ1 の書き起こしは混合音に対するものなので、重畳区間では相手の声が
    # 混ざったまま ASR に入っていた。相槌の正解を取るにはここを作り直す必要がある。
    if args.skip_asr:
        logger.info("--skip-asr: 再書き起こしを省略")
    else:
        for speaker in speakers:
            targets = [seg for seg in segments if seg.speaker == speaker]
            logger.info("話者 %s を分離トラックで再書き起こし中(%d 件)...", speaker, len(targets))
            transcribe_segments(
                tracks[speaker], sr, targets, args.whisper_model, args.device
            )
        for seg in segments:
            seg.is_aizuchi = False
            seg.aizuchi_labels = []
        mark_aizuchi(segments)

    # 発話 WAV は分離トラックから切り出す(耳チェックの対象はこちら)。
    seg_paths = export_separated_segments(
        tracks, sr, segments, out_dir, with_text=not args.skip_asr
    )

    with (out_dir / "timeline_separated.jsonl").open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(asdict(seg), ensure_ascii=False) + "\n")

    overlaps = _overlap_regions(segments)
    (out_dir / "stats_separated.json").write_text(
        json.dumps(compute_stats(segments, overlaps, total_sec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_paths = [
        write_review_tsv(out_dir, speaker, segments, seg_paths) for speaker in speakers
    ]
    logger.info("レビュー表を出力: %s", ", ".join(p.name for p in review_paths))
    return out_dir


def _overlap_regions(segments: list[Segment]) -> list[tuple[float, float]]:
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
    regions.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in regions:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wavs", nargs="+", type=Path, help="ステージ1 に渡したものと同じ WAV")
    parser.add_argument("--out-dir", type=Path, default=Path("data/real_dialogue"),
                        help="ステージ1 と同じ出力ルート")
    parser.add_argument("--sep-model", default=DEFAULT_SEP_MODEL,
                        help=f"SpeechBrain 分離モデル(既定: {DEFAULT_SEP_MODEL})")
    parser.add_argument("--sep-sample-rate", type=int, default=DEFAULT_SEP_SR,
                        help="分離モデルが要求するサンプリングレート")
    parser.add_argument("--sep-cache-dir", type=Path, default=None,
                        help="SpeechBrain の savedir(未指定なら既定キャッシュ)")
    parser.add_argument("--chunk-sec", type=float, default=10.0,
                        help="分離のチャンク長。長すぎると VRAM を食う")
    parser.add_argument("--chunk-overlap-sec", type=float, default=2.0,
                        help="チャンク間の重なり(繋ぎ目のクロスフェードに使う)")
    parser.add_argument("--whisper-model", default=None,
                        help="再書き起こしの faster-whisper モデル名")
    parser.add_argument("--device", default=None, help="cuda / cpu(既定: 自動判定)")
    parser.add_argument("--skip-asr", action="store_true",
                        help="再書き起こしを省略して分離結果だけ先に聴く")
    args = parser.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    if args.whisper_model is None:
        args.whisper_model = "large-v3" if args.device.startswith("cuda") else "small"
    logger.info("device=%s sep=%s whisper=%s", args.device, args.sep_model, args.whisper_model)

    out_dirs = []
    for wav in args.wavs:
        if not wav.is_file():
            logger.error("見つかりません: %s", wav)
            continue
        out_dirs.append(process_wav(wav, args.out_dir, args))

    print("\n=== 耳で確認 → ラベル付け → 間引き ===")
    for d in out_dirs:
        print(f"{d.resolve()}")
        print("  分離トラック: separated\\A.wav / separated\\B.wav / separated\\stereo.wav")
        print("  発話ごと:     segments_separated\\A\\ / segments_separated\\B\\")
        print("  1) 相談員側の話者を決めて counselor_A.txt か counselor_B.txt を置く")
        print("  2) A_segments_review.tsv / B_segments_review.tsv の keep 列を 0 にして間引く")


if __name__ == "__main__":
    main()
