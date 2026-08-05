#!/usr/bin/env python3
"""
人手アノテーション TSV から User 側のテスト音声を切り出す。

ダイアライゼーションも ASR も使わない。正解は人が付けた TSV であり、この
スクリプトはそれを WAV に切り出すだけ。**TSV が正、WAV はその出力**。

アノテーション TSV は音声と同じ名前で拡張子だけ .tsv にしたもの:

    /path/dialogue1.wav  ->  /path/dialogue1.tsv

列は Speaker / Transcription / Start / End。話者ラベルは Staff と User。
時刻は 00:00:15.500 形式(HH:MM:SS.mmm)。ヘッダ行はあってもなくてもよく、
無ければこの順に並んでいるものとして読む。

    Speaker	Transcription	Start	End
    User	もしもし、あの、ちょっと相談したいことがあって	00:00:15.500	00:00:21.200
    Staff	はい	00:00:21.000	00:00:21.600

「はい」のような相槌も 1 発話として書かれているので、既定では**記号を除いた
文字数が 5 未満の発話を落とす**(--min-chars)。落としたものは dropped.tsv に
理由付きで残すので、閾値が妥当だったか後から確認できる。

出力(入力 WAV ごとに <out-dir>/<wav_stem>/ 配下):
  - User/0000_00m15.5s_5.7s_もしもしあのちょっと.wav
  - clips.tsv     切り出したもの一覧(index / 時刻 / 重畳秒 / 書き起こし / wav)
  - dropped.tsv   落としたもの一覧(理由付き)

使い方:
    uv run python scripts/build_test_wav_from_tsv.py 対話1.wav \\
        --out-dir data/real_dialogue/test

GPU も NeMo も Whisper も要らない(numpy と soundfile だけ)。ジョブ投入も不要。
TSV を直したらもう一度実行すれば作り直される。
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 切り出し時に前後へ足すマージン(語頭・語尾の欠け防止)。
CLIP_MARGIN_SEC = 0.08

# 文字数を数える前に除去する記号類。「はい、」を 2 文字と数えるため。
# 長音「ー」は入れないこと。記号ではなく実文字で、「えーっと」を 3 文字に
# 数えてしまうと相槌でない発話まで閾値で落ちる。
_STRIP_RE = re.compile(r"[\s、。！？!?,.…・「」『』()（）]")
# Windows で使えない文字などファイル名に入れない文字。
_FNAME_RE = re.compile(r"[\\/:*?\"<>|\s]+")

# ヘッダ行の判定と列の対応付け。表記ゆれを吸収する。
_COLUMN_ALIASES = {
    "speaker": "speaker",
    "spk": "speaker",
    "話者": "speaker",
    "transcription": "text",
    "transcript": "text",
    "text": "text",
    "書き起こし": "text",
    "start": "start",
    "starttime": "start",
    "start_time": "start",
    "開始": "start",
    "end": "end",
    "endtime": "end",
    "end_time": "end",
    "終了": "end",
}
_DEFAULT_ORDER = ["speaker", "text", "start", "end"]


@dataclass
class Utterance:
    speaker: str
    text: str
    start: float
    end: float
    lineno: int
    overlap_sec: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


class AnnotationError(Exception):
    """行番号付きで報告するためのエラー。"""


def parse_timestamp(value: str, lineno: int, column: str) -> float:
    """00:00:15.500 を秒に変換する。

    HH:MM:SS.mmm / MM:SS.mmm / SS.mmm のいずれも受ける。小数点はピリオドでも
    カンマでもよい(Excel やツールによって変わるため)。
    """
    raw = value.strip().replace(",", ".")
    if not raw:
        raise AnnotationError(f"{lineno} 行目: {column} が空です。")
    parts = raw.split(":")
    if len(parts) > 3:
        raise AnnotationError(f"{lineno} 行目: {column} の形式が不正です: {value!r}")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise AnnotationError(
            f"{lineno} 行目: {column} の形式が不正です: {value!r} "
            "(00:00:15.500 の形式で書いてください)"
        )
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    return seconds


def detect_columns(row: list[str]) -> dict[str, int] | None:
    """ヘッダ行なら {役割: 列番号} を返す。データ行なら None。"""
    mapping: dict[str, int] = {}
    for i, cell in enumerate(row):
        key = cell.strip().lower().replace(" ", "").lstrip("﻿")
        role = _COLUMN_ALIASES.get(key)
        if role and role not in mapping:
            mapping[role] = i
    if all(role in mapping for role in _DEFAULT_ORDER):
        return mapping
    return None


def read_annotation(path: Path) -> list[Utterance]:
    if not path.is_file():
        raise SystemExit(
            f"アノテーション TSV が見つかりません: {path}\n"
            "音声と同じ名前で拡張子を .tsv にしたファイルを置いてください。"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise SystemExit(f"{path}: 空です。")

    columns = detect_columns(rows[0])
    if columns is not None:
        body = rows[1:]
        first_lineno = 2
        logger.info("ヘッダ行を検出: %s", {k: rows[0][v] for k, v in columns.items()})
    else:
        columns = {role: i for i, role in enumerate(_DEFAULT_ORDER)}
        body = rows
        first_lineno = 1
        logger.info(
            "ヘッダ行が無いので Speaker / Transcription / Start / End の順とみなします。"
        )

    need = max(columns.values()) + 1
    utterances: list[Utterance] = []
    errors: list[str] = []
    for offset, row in enumerate(body):
        lineno = first_lineno + offset
        if len(row) < need:
            errors.append(
                f"{lineno} 行目: 列が {len(row)} 個しかありません({need} 個必要)。"
                f" 内容: {row}"
            )
            continue
        try:
            start = parse_timestamp(row[columns["start"]], lineno, "Start")
            end = parse_timestamp(row[columns["end"]], lineno, "End")
        except AnnotationError as exc:
            errors.append(str(exc))
            continue
        if end <= start:
            errors.append(
                f"{lineno} 行目: End <= Start です "
                f"({row[columns['start']]!r} -> {row[columns['end']]!r})."
            )
            continue
        utterances.append(
            Utterance(
                speaker=row[columns["speaker"]].strip(),
                text=row[columns["text"]].strip(),
                start=start,
                end=end,
                lineno=lineno,
            )
        )

    if errors:
        for message in errors:
            logger.error("%s", message)
        raise SystemExit(f"{path}: {len(errors)} 行に問題があります。直してから再実行してください。")

    utterances.sort(key=lambda u: u.start)
    return utterances


def annotate_overlaps(utterances: list[Utterance]) -> None:
    """異話者の発話と重なっている秒数を各発話に記録する。

    1ch 録音なので、重畳区間には相手の声が入ったまま切り出される。どれが
    該当するかを clips.tsv で見えるようにしておく。
    """
    for i, a in enumerate(utterances):
        for b in utterances[i + 1:]:
            if b.start >= a.end:
                break
            if a.speaker == b.speaker:
                continue
            span = min(a.end, b.end) - max(a.start, b.start)
            if span > 0:
                a.overlap_sec += span
                b.overlap_sec += span


def stripped_length(text: str) -> int:
    return len(_STRIP_RE.sub("", text))


def clip_filename(index: int, utt: Utterance) -> str:
    flags = "_ov" if utt.overlap_sec > 0 else ""
    name = (
        f"{index:04d}_{int(utt.start // 60):02d}m{utt.start % 60:04.1f}s"
        f"_{utt.duration:.1f}s{flags}"
    )
    text = _FNAME_RE.sub("", _STRIP_RE.sub("", utt.text))[:16]
    if text:
        name += f"_{text}"
    return name + ".wav"


def mute_other_speakers(
    clip: np.ndarray, utt: Utterance, others: list[Utterance], sr: int
) -> np.ndarray:
    """切り出した区間のうち、相手が喋っている部分を無音にする。

    1ch 録音なので重畳部の相手の声は消えない(同じ標本に混ざっている)。ここで
    できるのは「相手だけが喋っている部分」を落とすことまで。
    """
    clip = clip.copy()
    base = utt.start - CLIP_MARGIN_SEC
    for other in others:
        lo = max(0, int((other.start - base) * sr))
        hi = min(clip.size, int((other.end - base) * sr))
        if hi > lo:
            clip[lo:hi] = 0.0
    return clip


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Excel がそのまま開けるよう UTF-8 BOM 付きで書く。"""
    lines = ["\t".join(header)]
    lines += ["\t".join(cell.replace("\t", " ").replace("\n", " ") for cell in row)
              for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def process_wav(wav_path: Path, out_root: Path, args: argparse.Namespace) -> Path:
    annotation = args.annotation or wav_path.with_suffix(".tsv")
    utterances = read_annotation(annotation)

    labels = sorted({u.speaker for u in utterances})
    logger.info(
        "%s: %d 発話 / 話者ラベル %s", annotation.name, len(utterances), labels
    )

    matched = [u for u in utterances if u.speaker.lower() == args.speaker.lower()]
    if not matched:
        raise SystemExit(
            f"{annotation}: 話者 {args.speaker!r} の発話がありません。"
            f" TSV にあるラベルは {labels} です。\n"
            "  --speaker で対象を指定できます(大文字小文字は区別しません)。"
        )
    # 出力ディレクトリ名は引数の綴りではなく TSV 上の実ラベルにする。
    # --speaker staff と --speaker Staff で別フォルダができると混乱する。
    target = matched[0].speaker

    annotate_overlaps(utterances)

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        logger.warning("%s は %dch です。モノラルへダウンミックスします。",
                       wav_path.name, audio.shape[1])
        audio = audio.mean(axis=1)
    total_sec = audio.size / sr
    logger.info("%s: %.1f 秒 (%d Hz)", wav_path.name, total_sec, sr)

    last_end = max(u.end for u in utterances)
    if last_end > total_sec + 1.0:
        logger.warning(
            "TSV の最終時刻 %.1f 秒が音声長 %.1f 秒を超えています。"
            "音声と TSV の組み合わせが違う可能性があります。",
            last_end, total_sec,
        )

    out_dir = out_root / wav_path.stem
    clip_dir = out_dir / target
    clip_dir.mkdir(parents=True, exist_ok=True)
    for stale in clip_dir.glob("*.wav"):
        stale.unlink()

    others = [u for u in utterances if u.speaker.lower() != target.lower()]

    kept_rows: list[list[str]] = []
    dropped_rows: list[list[str]] = []
    for index, utt in enumerate(matched):
        chars = stripped_length(utt.text)
        reason = ""
        if chars < args.min_chars:
            reason = f"文字数 {chars} < {args.min_chars}"
        elif utt.duration < args.min_sec:
            reason = f"長さ {utt.duration:.2f}s < {args.min_sec}s"
        if reason:
            dropped_rows.append([
                str(index), f"{utt.start:.3f}", f"{utt.end:.3f}",
                f"{utt.duration:.3f}", str(chars), reason, utt.text,
            ])
            continue

        lo = max(0, int((utt.start - CLIP_MARGIN_SEC) * sr))
        hi = min(audio.size, int((utt.end + CLIP_MARGIN_SEC) * sr))
        if hi <= lo:
            dropped_rows.append([
                str(index), f"{utt.start:.3f}", f"{utt.end:.3f}",
                f"{utt.duration:.3f}", str(chars), "音声の範囲外", utt.text,
            ])
            continue

        clip = audio[lo:hi]
        if args.mute_other:
            clip = mute_other_speakers(clip, utt, others, sr)

        path = clip_dir / clip_filename(index, utt)
        sf.write(path, clip, sr)
        kept_rows.append([
            str(index), f"{utt.start:.3f}", f"{utt.end:.3f}", f"{utt.duration:.3f}",
            f"{utt.overlap_sec:.3f}", str(chars), utt.text, path.name,
        ])

    write_tsv(
        out_dir / "clips.tsv",
        ["index", "start_sec", "end_sec", "duration_sec", "overlap_sec",
         "chars", "text", "wav"],
        kept_rows,
    )
    write_tsv(
        out_dir / "dropped.tsv",
        ["index", "start_sec", "end_sec", "duration_sec", "chars", "reason", "text"],
        dropped_rows,
    )

    overlapped = sum(1 for row in kept_rows if float(row[4]) > 0)
    kept_sec = sum(float(row[3]) for row in kept_rows)
    logger.info(
        "%s: %s の発話 %d 件 -> 採用 %d 件(計 %.1f 秒) / 除外 %d 件",
        wav_path.name, target, len(matched), len(kept_rows), kept_sec,
        len(dropped_rows),
    )
    if overlapped:
        logger.warning(
            "採用のうち %d 件は相手の発話と重なっています(ファイル名の _ov)。"
            "1ch 録音なので相手の声が残ります。",
            overlapped,
        )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wavs", nargs="+", type=Path, help="入力 WAV(複数可)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/real_dialogue/test"),
                        help="出力先ルート(既定: data/real_dialogue/test)")
    parser.add_argument("--annotation", type=Path, default=None,
                        help="アノテーション TSV を明示指定(既定: WAV と同名 .tsv)。"
                             "WAV を 1 つだけ渡すときに使う")
    parser.add_argument("--speaker", default="User",
                        help="切り出す話者ラベル(既定: User。大文字小文字は区別しない)")
    parser.add_argument("--min-chars", type=int, default=5,
                        help="記号を除いた文字数がこれ未満の発話を落とす(既定: 5)。"
                             "相槌を除くため。0 で無効")
    parser.add_argument("--min-sec", type=float, default=0.0,
                        help="これより短い発話を落とす(既定: 0 = 無効)")
    parser.add_argument("--mute-other", action="store_true",
                        help="切り出した区間のうち相手だけが喋っている部分を無音化する")
    args = parser.parse_args()

    if args.annotation is not None and len(args.wavs) > 1:
        parser.error("--annotation は WAV を 1 つだけ渡すときに使ってください。")

    out_dirs = []
    for wav in args.wavs:
        if not wav.is_file():
            logger.error("見つかりません: %s", wav)
            continue
        out_dirs.append(process_wav(wav, args.out_dir, args))

    if not out_dirs:
        raise SystemExit("処理できた WAV がありません。")

    print("\n=== 出力 ===")
    for d in out_dirs:
        print(d.resolve())
        print(f"  テスト音声: {args.speaker}\\")
        print("  採用一覧:   clips.tsv")
        print("  除外一覧:   dropped.tsv  ← 閾値が妥当か確認できる")
    print("\nTSV を直したら、同じコマンドをもう一度実行すれば作り直されます。")


if __name__ == "__main__":
    main()
