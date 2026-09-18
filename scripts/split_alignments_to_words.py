#!/usr/bin/env python3
"""既存 sidecar JSON の発話単位 alignments を単語単位に書き換える修復スクリプト。

音声（wav）の再合成は不要。sidecar JSON だけをインプレース更新する。
元の発話単位 alignments は同じ JSON の "alignments_utterance" キーに保存され、
変換済みファイルは再実行時にスキップされる（冪等）。

使い方:
    # 学習 run の data_stereo ディレクトリを直接指定
    uv run python scripts/split_alignments_to_words.py \
        --data-dir data/runs/<RUN_ID>/training_set/data_stereo

    # manifest (synthetic_moshi_train.jsonl) から辿る場合
    uv run python scripts/split_alignments_to_words.py \
        --manifest data/runs/<RUN_ID>/training_set/synthetic_moshi_train.jsonl

    # 変更内容の確認だけ（書き込みなし）
    uv run python scripts/split_alignments_to_words.py --data-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alignment_words import (
        DEFAULT_MAX_WORD_CHARS,
        get_segmenter_name,
        split_utterance_alignments,
    )
else:
    from .alignment_words import (
        DEFAULT_MAX_WORD_CHARS,
        get_segmenter_name,
        split_utterance_alignments,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="sidecar JSON (*.json) を含むディレクトリ（再帰探索）。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="synthetic_moshi_train.jsonl。各行の path から sidecar を解決する。",
    )
    parser.add_argument("--dry-run", action="store_true", help="書き込みせず統計のみ表示。")
    parser.add_argument(
        "--max-word-chars",
        type=int,
        default=DEFAULT_MAX_WORD_CHARS,
        help="1 エントリの上限文字数（超過は均等分割）。",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="変換例として表示するエントリ数。",
    )
    parser.add_argument(
        "--require-pyopenjtalk",
        action="store_true",
        help=(
            "pyopenjtalk 形態素分割が使えない環境ではエラー終了する。"
            "未指定時は正規表現チャンク分割（句読点境界+4文字）にフォールバック"
            "する。時刻はどちらも文字数比例内挿なので品質差は小さいが、"
            "学習 run 内で分割粒度を統一したい場合に指定する。"
        ),
    )
    return parser.parse_args()


def collect_sidecars(args: argparse.Namespace) -> list[Path]:
    paths: dict[Path, None] = {}
    if args.data_dir:
        root = args.data_dir.resolve()
        if not root.is_dir():
            raise SystemExit(f"ERROR: --data-dir not found: {root}")
        for path in sorted(root.rglob("*.json")):
            paths[path] = None
    if args.manifest:
        manifest = args.manifest.resolve()
        if not manifest.is_file():
            raise SystemExit(f"ERROR: --manifest not found: {manifest}")
        for lineno, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: {manifest.name} line {lineno}: invalid JSON: {exc}")
                continue
            rel = row.get("path")
            if not rel:
                continue
            wav = Path(rel)
            if not wav.is_absolute():
                wav = manifest.parent / wav
            paths[wav.resolve().with_suffix(".json")] = None
    if not paths:
        raise SystemExit("ERROR: --data-dir か --manifest のどちらかを指定してください。")
    return list(paths)


def convert_file(
    path: Path,
    max_word_chars: int,
    dry_run: bool,
    example_budget: int,
    segmenter: str,
) -> tuple[str, dict[str, int], list[str]]:
    """Returns (status, stats, examples). status: converted|skipped|no_alignments|error"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: {path}: 読み込み失敗: {exc}")
        return "error", {}, []
    if not isinstance(data, dict):
        return "no_alignments", {}, []
    alignments = data.get("alignments")
    if not isinstance(alignments, list) or not alignments:
        return "no_alignments", {}, []
    metadata = data.get("metadata")
    already = (
        "alignments_utterance" in data
        or (isinstance(metadata, dict) and metadata.get("alignments_granularity") == "word")
    )
    if already:
        return "skipped", {}, []

    word_alignments, stats = split_utterance_alignments(
        alignments, max_word_chars=max_word_chars
    )

    examples: list[str] = []
    if example_budget > 0:
        for entry in alignments:
            if example_budget <= 0:
                break
            if not (isinstance(entry, list) and len(entry) >= 3 and isinstance(entry[0], str)):
                continue
            if len(entry[0]) < 10:
                continue
            split_words = [
                w[0]
                for w in word_alignments
                if isinstance(w, list) and entry[1][0] <= w[1][0] < entry[1][1]
            ]
            examples.append(f"{entry[0]!r} -> {' / '.join(split_words)}")
            example_budget -= 1

    if not dry_run:
        data["alignments"] = word_alignments
        data["alignments_utterance"] = alignments
        if not isinstance(metadata, dict):
            metadata = {}
            data["metadata"] = metadata
        metadata["alignments_granularity"] = "word"
        metadata["alignments_word_split"] = {
            "version": 1,
            "method": "char-proportional",
            "segmenter": segmenter,
            "max_word_chars": max_word_chars,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tmp.replace(path)
    return "converted", stats, examples


def main() -> int:
    args = parse_args()
    segmenter = get_segmenter_name()
    if args.require_pyopenjtalk and segmenter != "pyopenjtalk":
        print(
            "ERROR: --require-pyopenjtalk が指定されましたが pyopenjtalk による"
            "形態素分割が使えません（未インストールか表層復元に失敗）。",
            file=sys.stderr,
        )
        return 2
    sidecars = collect_sidecars(args)

    totals: dict[str, Any] = {
        "files": len(sidecars),
        "converted": 0,
        "skipped": 0,
        "no_alignments": 0,
        "error": 0,
        "utterances": 0,
        "words": 0,
        "passthrough": 0,
    }
    shown_examples: list[str] = []
    for path in sidecars:
        budget = max(0, args.examples - len(shown_examples))
        status, stats, examples = convert_file(
            path, args.max_word_chars, args.dry_run, budget, segmenter
        )
        totals[status] += 1
        for key in ("utterances", "words", "passthrough"):
            totals[key] += stats.get(key, 0)
        shown_examples.extend(examples)

    mode = "dry-run" if args.dry_run else "in-place"
    print(f"[split-alignments] mode={mode} segmenter={segmenter}")
    print(
        f"[split-alignments] files={totals['files']} converted={totals['converted']} "
        f"skipped(already word-level)={totals['skipped']} "
        f"no_alignments={totals['no_alignments']} errors={totals['error']}"
    )
    if totals["converted"]:
        avg = totals["words"] / max(totals["utterances"], 1)
        print(
            f"[split-alignments] utterances={totals['utterances']} -> words={totals['words']} "
            f"(avg {avg:.1f} words/utterance, passthrough={totals['passthrough']})"
        )
    for example in shown_examples:
        print(f"[split-alignments] example: {example}")
    if totals["error"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
