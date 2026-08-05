#!/usr/bin/env python3
"""
手で直した test_wav/turns_review.tsv から test_wav/ を作り直す。

analyze_real_dialogue.py の塊まとめは自動なので、境界がずれたり話者を取り違えたり
することがある。それを TSV 上で直して、この スクリプトで WAV に反映させる。
**TSV が正、WAV はその出力**という関係にする。

GPU も NeMo も Whisper も要らない(numpy と soundfile だけ)。切り出し元は
分析ディレクトリに既にある speaker_<話者>_solo.wav なので、元の WAV も要らない。

TSV でできること:
  - 間引く            keep 列を 0 にする
  - 境界を直す        start_sec / end_sec を書き換える
  - 話者を直す        speaker 列を A/B で書き換える
  - 塊を分ける        行をコピーして、それぞれの start_sec / end_sec を狭める
  - 塊を繋ぐ          片方の end_sec を伸ばし、もう片方を keep=0 にする
  - 書き起こしを直す  text 列を書き換える(ファイル名に反映される)

行は自由に足してよい。index は空でも重複していても構わない(その場合は
空いている番号を自動で振る)。並び順も見ない。

使い方:
    uv run python scripts/rebuild_test_wav.py data/real_dialogue/pilot/対話1
    uv run python scripts/rebuild_test_wav.py data/real_dialogue/pilot/*  --dry-run

既存の test_wav/<話者>/*.wav は消して作り直す。turns_review.tsv 自体は
書き換えない(手で直したものが正なので、上書きしてはいけない)。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.analyze_real_dialogue import CLIP_MARGIN_SEC, Turn, turn_filename
except ImportError:  # スクリプト直接実行時（scripts/ が sys.path 先頭）
    from analyze_real_dialogue import (  # type: ignore[no-redef]
        CLIP_MARGIN_SEC,
        Turn,
        turn_filename,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REVIEW_NAME = "turns_review.tsv"
# keep 列がこれらのいずれかなら除外。空欄も除外扱いにする(残すつもりなら 1 を
# 書くはずなので、空欄を採用にすると消し忘れが黙って混入する)。
DROP_VALUES = {"", "0", "0.0", "no", "false", "n", "x"}


class RowError(Exception):
    """行番号付きで報告するためのエラー。"""


def parse_bool_keep(value: str) -> bool:
    return value.strip().lower() not in DROP_VALUES


def read_review(path: Path) -> list[dict[str, str]]:
    """UTF-8 BOM 付き TSV を読む。Excel で保存し直しても壊れないようにする。"""
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"{path}: 空です。")
    header = lines[0].split("\t")
    required = {"keep", "speaker", "start_sec", "end_sec"}
    missing = required - set(header)
    if missing:
        raise SystemExit(
            f"{path}: 必要な列がありません: {sorted(missing)}\n"
            f"  見つかった列: {header}"
        )
    rows: list[dict[str, str]] = []
    for lineno, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        # Excel は末尾の空セルを落とすことがあるので、足りない分は空文字で埋める。
        cells += [""] * (len(header) - len(cells))
        row = dict(zip(header, cells))
        row["_lineno"] = str(lineno)
        rows.append(row)
    return rows


def row_to_turn(row: dict[str, str]) -> Turn:
    lineno = row.get("_lineno", "?")
    speaker = row["speaker"].strip()
    if not speaker:
        raise RowError(f"{lineno} 行目: speaker が空です。")
    try:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
    except ValueError as exc:
        raise RowError(f"{lineno} 行目: start_sec / end_sec が数値ではありません ({exc}).")
    if end <= start:
        raise RowError(f"{lineno} 行目: end_sec <= start_sec です ({start} -> {end}).")

    def as_int(key: str) -> int:
        try:
            return int(float(row.get(key, "") or 0))
        except ValueError:
            return 0

    def as_float(key: str) -> float:
        try:
            return float(row.get(key, "") or 0.0)
        except ValueError:
            return 0.0

    return Turn(
        speaker=speaker,
        start=start,
        end=end,
        text=row.get("text", "").strip(),
        n_segments=as_int("n_segments"),
        overlap_sec=as_float("overlap_sec"),
        aizuchi_count=as_int("aizuchi_count"),
    )


def assign_indices(rows: list[dict[str, str]]) -> list[int]:
    """TSV の index を尊重しつつ、空欄・重複には空き番号を振る。

    index を保つのは、手元で聴いていた WAV のファイル名が作り直しで変わると
    突き合わせられなくなるため。行を足した場合だけ新しい番号になる。
    """
    used: set[int] = set()
    parsed: list[int | None] = []
    for row in rows:
        raw = row.get("index", "").strip()
        try:
            value = int(float(raw))
        except ValueError:
            parsed.append(None)
            continue
        if value in used or value < 0:
            parsed.append(None)
            continue
        used.add(value)
        parsed.append(value)

    next_free = 0
    result: list[int] = []
    for value in parsed:
        if value is not None:
            result.append(value)
            continue
        while next_free in used:
            next_free += 1
        used.add(next_free)
        result.append(next_free)
    return result


def load_solo_tracks(analysis_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    tracks: dict[str, np.ndarray] = {}
    sr = 0
    for path in sorted(analysis_dir.glob("speaker_*_solo.wav")):
        speaker = path.stem[len("speaker_"): -len("_solo")]
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr and file_sr != sr:
            raise SystemExit(
                f"{path}: サンプリングレートが他と違います ({file_sr} != {sr})."
            )
        sr = file_sr
        tracks[speaker] = audio
    if not tracks:
        raise SystemExit(
            f"{analysis_dir}: speaker_*_solo.wav がありません。\n"
            "先に scripts/analyze_real_dialogue.py を実行してください。"
        )
    return tracks, sr


def clear_existing_clips(test_dir: Path) -> int:
    removed = 0
    for path in test_dir.glob("*/*.wav"):
        path.unlink()
        removed += 1
    return removed


def rebuild(analysis_dir: Path, dry_run: bool) -> int:
    review = analysis_dir / "test_wav" / REVIEW_NAME
    if not review.is_file():
        raise SystemExit(
            f"{review} がありません。\n"
            "先に scripts/analyze_real_dialogue.py を実行してください。"
        )

    rows = read_review(review)
    tracks, sr = load_solo_tracks(analysis_dir)

    kept: list[tuple[int, Turn]] = []
    errors: list[str] = []
    dropped = 0
    indices = assign_indices(rows)
    for row, index in zip(rows, indices):
        if not parse_bool_keep(row.get("keep", "")):
            dropped += 1
            continue
        try:
            turn = row_to_turn(row)
        except RowError as exc:
            errors.append(str(exc))
            continue
        if turn.speaker not in tracks:
            errors.append(
                f"{row['_lineno']} 行目: 話者 {turn.speaker!r} の solo トラックが"
                f"ありません(あるのは {sorted(tracks)})."
            )
            continue
        kept.append((index, turn))

    if errors:
        for message in errors:
            logger.error("%s", message)
        raise SystemExit(
            f"{review}: {len(errors)} 行に問題があります。直してから再実行してください。"
        )

    kept.sort(key=lambda pair: (pair[1].start, pair[1].speaker))
    total_sec = sum(turn.duration for _, turn in kept)
    logger.info(
        "%s: %d 行 -> 採用 %d 件 / 除外 %d 件 (計 %.1f 秒)",
        review, len(rows), len(kept), dropped, total_sec,
    )

    if dry_run:
        for index, turn in kept[:10]:
            logger.info(
                "  [dry-run] %s", turn_filename(index, turn)
            )
        if len(kept) > 10:
            logger.info("  [dry-run] ... 他 %d 件", len(kept) - 10)
        return len(kept)

    test_dir = analysis_dir / "test_wav"
    removed = clear_existing_clips(test_dir)
    logger.info("既存クリップ %d 件を削除して作り直します。", removed)

    written: list[dict] = []
    for index, turn in kept:
        track = tracks[turn.speaker]
        lo = max(0, int((turn.start - CLIP_MARGIN_SEC) * sr))
        hi = min(track.size, int((turn.end + CLIP_MARGIN_SEC) * sr))
        if hi <= lo:
            logger.warning(
                "index %d (%s %.2fs-%.2fs): 音声の範囲外なので飛ばします。",
                index, turn.speaker, turn.start, turn.end,
            )
            continue
        seg_dir = test_dir / turn.speaker
        seg_dir.mkdir(parents=True, exist_ok=True)
        path = seg_dir / turn_filename(index, turn)
        sf.write(path, track[lo:hi], sr)
        row = {
            "speaker": turn.speaker,
            "start": turn.start,
            "end": turn.end,
            "text": turn.text,
            "n_segments": turn.n_segments,
            "overlap_sec": turn.overlap_sec,
            "aizuchi_count": turn.aizuchi_count,
            "index": index,
            "wav": path.name,
        }
        written.append(row)

    turns_path = test_dir / "turns.jsonl"
    with turns_path.open("w", encoding="utf-8") as f:
        for row in written:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("書き出し %d 件: %s", len(written), test_dir)
    logger.info("メタデータ: %s", turns_path)
    return len(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "analysis_dirs", nargs="+", type=Path,
        help="分析ディレクトリ(test_wav/ を含むもの)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="WAV を書かずに、何件採用されるかだけ確認する",
    )
    args = parser.parse_args()

    total = 0
    for analysis_dir in args.analysis_dirs:
        if not analysis_dir.is_dir():
            raise SystemExit(f"ディレクトリではありません: {analysis_dir}")
        total += rebuild(analysis_dir, args.dry_run)

    if not args.dry_run:
        print(f"\n合計 {total} 件の test_wav クリップを作り直しました。")


if __name__ == "__main__":
    main()
