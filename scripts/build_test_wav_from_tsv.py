#!/usr/bin/env python3
"""
人手アノテーション TSV から User 側のテスト音声を切り出す。

ダイアライゼーションも ASR も使わない。正解は人が付けた TSV であり、この
スクリプトはそれを WAV に切り出すだけ。**アノテーションが正、WAV はその出力**。

アノテーションは音声と同じ名前で拡張子だけ .tsv か .txt にしたもの:

    /path/dialogue1.wav  ->  /path/dialogue1.tsv  または  /path/dialogue1.txt

**Audacity のラベル書き出し(.txt)をそのまま読める。** 話者はコロンの前。

    15.500000	21.200000	User: もしもし、あの、ちょっと相談したいことがあって
    21.000000	21.600000	Staff: はい

TSV の列は Speaker / Transcription / Start / End。話者ラベルは Staff と User。
時刻は 00:00:15.500 形式(HH:MM:SS.mmm)。ヘッダ行はあってもなくてもよく、
無ければこの順に並んでいるものとして読む。

    Speaker	Transcription	Start	End
    User	もしもし、あの、ちょっと相談したいことがあって	00:00:15.500	00:00:21.200
    Staff	はい	00:00:21.000	00:00:21.600

どちらの形式かは拡張子ではなく**中身**で見分ける(先頭 2 列が数値ならラベル)。

「はい」のような相槌も 1 発話として書かれているので、既定では**記号を除いた
文字数が 5 未満の発話を落とす**(--min-chars)。落としたものは dropped.tsv に
理由付きで残すので、閾値が妥当だったか後から確認できる。

出力(入力 WAV ごとに <out-dir>/<wav_stem>/ 配下):
  - User/0000_00m15.5s_5.7s_もしもしあのちょっと.wav
  - clips.tsv     切り出したもの一覧(index / 時刻 / 重畳秒 / 書き起こし / wav)
  - dropped.tsv   落としたもの一覧(理由付き)
加えて <out-dir> 直下に manifest.json(全対話の内訳と合計)。

使い方(フォルダを渡すと中の WAV を全部処理する):
    uv run python scripts/build_test_wav_from_tsv.py /path/dialogues/

    # WAV を個別に指定してもよい
    uv run python scripts/build_test_wav_from_tsv.py /path/d1.wav /path/d2.wav

出力先の既定は本番用テストデータの置き場 (data/test_data/real_dialogue)。
テストデータは固定で使うものなので、既定を決め打ちにして毎回同じ場所へ出す。

GPU も NeMo も Whisper も要らない(numpy と soundfile だけ)。ジョブ投入も不要。
TSV を直したらもう一度実行すれば作り直される。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
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

# アノテーションとして受け付ける拡張子。.txt は Audacity のラベル書き出し。
ANNOTATION_SUFFIXES = (".tsv", ".txt")
# Audacity のラベルは「話者: 本文」。全角コロンも受ける。
_LABEL_COLONS = (":", "：")


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


def is_number(value: str) -> bool:
    try:
        float(value.strip().replace(",", "."))
    except ValueError:
        return False
    return True


def split_label(label: str) -> tuple[str, str]:
    """Audacity のラベル「話者: 本文」を割る。

    最初のコロンでしか切らないので、本文に入った「10:30 に」のようなコロンは
    そのまま残る。全角コロンも受ける。
    """
    found = [label.find(colon) for colon in _LABEL_COLONS if colon in label]
    if not found:
        return "", label.strip()
    at = min(found)
    return label[:at].strip(), label[at + 1:].strip()


def is_label_format(rows: list[list[str]]) -> bool:
    """Audacity のラベル書き出しかどうか。

    ラベルは「開始秒 終了秒 本文」の 3 列。アノテーション TSV は 1 列目が話者
    なので、先頭 2 列が数値かどうかで割り切れる。
    """
    for row in rows:
        if row and row[0].lstrip().startswith("\\"):
            continue  # スペクトル選択付きで書き出したときの付随行
        return len(row) >= 3 and is_number(row[0]) and is_number(row[1])
    return False


def read_label_annotation(
    numbered: list[tuple[int, list[str]]], path: Path
) -> list[Utterance]:
    """Audacity のラベルを Utterance にする。

        15.500000	21.200000	User: もしもし、あの、相談したいことがありまして

    話者はコロンの前。ここで拾えなかった行を黙って落とすと、テストデータが
    静かに数発話ぶん足りない状態になるので、行番号付きで報告して止める。
    """
    utterances: list[Utterance] = []
    errors: list[str] = []
    for lineno, row in numbered:
        if row and row[0].lstrip().startswith("\\"):
            continue  # 周波数の付随行
        if len(row) < 3:
            errors.append(
                f"{lineno} 行目: 列が {len(row)} 個しかありません(3 個必要)。 内容: {row}"
            )
            continue
        try:
            start = parse_timestamp(row[0], lineno, "開始")
            end = parse_timestamp(row[1], lineno, "終了")
        except AnnotationError as exc:
            errors.append(str(exc))
            continue
        if end <= start:
            errors.append(
                f"{lineno} 行目: 終了 <= 開始 です ({row[0]!r} -> {row[1]!r})。"
                "点ラベル(長さ 0)は区間にしてください。"
            )
            continue
        # ラベル本文にタブは入らないが、貼り付け由来で混ざっていても拾う。
        speaker, text = split_label(" ".join(cell for cell in row[2:] if cell))
        if not speaker:
            errors.append(
                f"{lineno} 行目: 話者がありません(「話者: 本文」の形): {row[2]!r}"
            )
            continue
        if not text:
            errors.append(f"{lineno} 行目: 書き起こしが空です: {row[2]!r}")
            continue
        utterances.append(
            Utterance(speaker=speaker, text=text, start=start, end=end, lineno=lineno)
        )

    if errors:
        for message in errors:
            logger.error("%s", message)
        raise SystemExit(
            f"{path}: {len(errors)} 行に問題があります。直してから再実行してください。"
        )
    logger.info("%s: Audacity のラベルとして読みました。", path.name)
    utterances.sort(key=lambda u: u.start)
    return utterances


def read_annotation(path: Path) -> list[Utterance]:
    if not path.is_file():
        raise SystemExit(
            f"アノテーションが見つかりません: {path}\n"
            "音声と同じ名前で拡張子を .tsv か .txt にしたファイルを置いてください。"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.reader(f, delimiter="\t"))

    # 行番号は元のファイルのものを保つ(空行を飛ばしても報告がずれないように)。
    numbered = [
        (lineno, row) for lineno, row in enumerate(raw_rows, start=1)
        if any(cell.strip() for cell in row)
    ]
    rows = [row for _, row in numbered]
    if not rows:
        raise SystemExit(f"{path}: 空です。")

    # 中身で見分ける。拡張子ではなく形式が正。
    if is_label_format(rows):
        return read_label_annotation(numbered, path)

    columns = detect_columns(rows[0])
    if columns is not None:
        body = numbered[1:]
        logger.info("ヘッダ行を検出: %s", {k: rows[0][v] for k, v in columns.items()})
    else:
        columns = {role: i for i, role in enumerate(_DEFAULT_ORDER)}
        body = numbered
        logger.info(
            "ヘッダ行が無いので Speaker / Transcription / Start / End の順とみなします。"
        )

    need = max(columns.values()) + 1
    utterances: list[Utterance] = []
    errors: list[str] = []
    for lineno, row in body:
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


DEFAULT_OUT_DIR = Path("data/test_data/real_dialogue")


def annotation_for(wav_path: Path) -> Path:
    """WAV に対応するアノテーション(同じ名前で拡張子だけ .tsv か .txt)。

    .txt は Audacity のラベル書き出し。両方あれば .tsv を採る。無いときは
    .tsv のパスを返す(見つからない旨の報告に使う)。
    """
    for suffix in ANNOTATION_SUFFIXES:
        candidate = wav_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return wav_path.with_suffix(ANNOTATION_SUFFIXES[0])


def collect_inputs(inputs: list[Path], recursive: bool) -> list[Path]:
    """ファイルとフォルダの混在を受けて、処理対象の WAV 一覧にする。

    フォルダを渡された場合は中の WAV を全部対象にする。テストデータを一式
    作り直すのが通常の使い方なので、フォルダ指定が主で個別指定が補助。
    """
    wavs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            pattern = "**/*.wav" if recursive else "*.wav"
            found = sorted(p for p in item.glob(pattern) if p.is_file())
            if not found:
                raise SystemExit(
                    f"{item}: WAV がありません"
                    f"{'(--recursive を付けても)' if recursive else ''}。"
                )
            logger.info("%s: WAV %d 件", item, len(found))
            wavs.extend(found)
        elif item.is_file():
            wavs.append(item)
        else:
            raise SystemExit(f"見つかりません: {item}")

    # 同じ WAV を二重に処理しない(フォルダと個別指定が重なった場合)。
    seen: set[Path] = set()
    unique: list[Path] = []
    for wav in wavs:
        resolved = wav.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(wav)
    return unique


def require_annotations(wavs: list[Path], override: Path | None) -> None:
    """対応する TSV が揃っているか先に全部見る。

    1 件ずつ処理しながら落ちると、テストデータが中途半端に出来た状態で
    止まる。欠けているものを全部挙げてから止める。
    """
    if override is not None:
        return
    missing = [wav for wav in wavs if not annotation_for(wav).is_file()]
    if not missing:
        return
    for wav in missing:
        logger.error(
            "%s に対応する %s がありません(.txt でも可)。",
            wav.name, annotation_for(wav).name,
        )
    raise SystemExit(
        f"アノテーションが {len(missing)} 件足りません。"
        "音声と同じ名前で拡張子を .tsv か .txt にしたファイルを隣に置いてください。"
    )


def process_wav(wav_path: Path, out_root: Path, args: argparse.Namespace) -> dict:
    annotation = args.annotation or annotation_for(wav_path)
    utterances = read_annotation(annotation)

    # 件数付きで出す。1 件しかないラベルは綴り間違い(Usre など)が疑わしい。
    counts = Counter(u.speaker for u in utterances)
    labels = sorted(counts)
    logger.info(
        "%s: %d 発話 / 話者ラベル %s",
        annotation.name, len(utterances),
        ", ".join(f"{label}:{counts[label]}" for label in labels),
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
    return {
        "name": wav_path.stem,
        "wav": str(wav_path),
        "annotation": str(annotation),
        "audio_sec": round(total_sec, 1),
        "speaker": target,
        "speaker_labels": labels,
        "utterances_total": len(utterances),
        "utterances_target": len(matched),
        "clips": len(kept_rows),
        "clips_sec": round(kept_sec, 1),
        "clips_overlapped": overlapped,
        "dropped": len(dropped_rows),
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="音声フォルダ、または WAV(混在可)。"
                             "フォルダなら中の WAV を全部処理する")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"出力先ルート(既定: {DEFAULT_OUT_DIR})")
    parser.add_argument("--recursive", action="store_true",
                        help="フォルダを再帰的に探す")
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

    wavs = collect_inputs(args.inputs, args.recursive)
    if not wavs:
        raise SystemExit("処理対象の WAV がありません。")
    if args.annotation is not None and len(wavs) > 1:
        parser.error("--annotation は WAV を 1 つだけ渡すときに使ってください。")

    # 途中で落ちてテストデータが中途半端に出来ないよう、先に全部確認する。
    require_annotations(wavs, args.annotation)

    summaries = [process_wav(wav, args.out_dir, args) for wav in wavs]

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "speaker": args.speaker,
        "filters": {
            "min_chars": args.min_chars,
            "min_sec": args.min_sec,
            "mute_other": args.mute_other,
        },
        "totals": {
            "dialogues": len(summaries),
            "clips": sum(s["clips"] for s in summaries),
            "clips_sec": round(sum(s["clips_sec"] for s in summaries), 1),
            "clips_overlapped": sum(s["clips_overlapped"] for s in summaries),
            "dropped": sum(s["dropped"] for s in summaries),
        },
        "dialogues": summaries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    totals = manifest["totals"]
    print("\n=== テストデータ ===")
    print(f"{args.out_dir.resolve()}")
    for summary in summaries:
        print(
            f"  {summary['name']:<24} {summary['clips']:>4} 件 "
            f"{summary['clips_sec']:>7.1f} 秒"
            f"  (除外 {summary['dropped']}, 重畳 {summary['clips_overlapped']})"
        )
    print(
        f"  {'合計':<24} {totals['clips']:>4} 件 {totals['clips_sec']:>7.1f} 秒"
        f"  (除外 {totals['dropped']}, 重畳 {totals['clips_overlapped']})"
    )
    print(f"\n内訳: {manifest_path}")
    print("各対話の clips.tsv / dropped.tsv で中身を確認できます。")
    print("TSV を直したら、同じコマンドをもう一度実行すれば作り直されます。")


if __name__ == "__main__":
    main()
