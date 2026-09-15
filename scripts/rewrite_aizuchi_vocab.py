#!/usr/bin/env python3
"""対話 JSONL の相槌を、指定した語彙から引き直す。

作りたいのは「うんの入った対話」。いまの対話生成の語彙には うん が無く、
実録音で最も多かった 4 語（うん / うーん / そっか / うんうん = 76%）が
どれも入っていないので、生成し直すのではなく聞き手側の相槌だけを差し替える。

user 側のターンには触らない。同じ対話の同じ発話がそのまま残るので、
書き換え前のコーパスとそのまま比べられる。

  uv run python scripts/rewrite_aizuchi_vocab.py \\
      --dialogues-jsonl <in.jsonl> --out-jsonl <out.jsonl>

既定の語彙と重みは scripts/2026-09-15/real_backchannel_dist.tsv（実録音の実測）。
1 語だけにしたいなら --text うん。

出力の隣に <out>.vocab.txt を書く。差し替え後の対話に実際に現れた語の一覧で、
そのまま splice_aizuchi_bank.py の --aizuchi-vocab-file に渡せる（渡さないと、
差し替えたはずの相槌が「語彙外」と判定されて素通りする）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = REPO_ROOT / "scripts/2026-09-15/real_backchannel_dist.tsv"
# 相槌とみなす文字数の上限。今の語彙の最長が「はい、大丈夫ですよ。」で 10 文字。
DEFAULT_MAX_CHARS = 12
LISTENER = "moshi"
STRIP_CHARS = "。、．，!?！？…・ 　"


def load_distribution(path: Path) -> tuple[list[str], list[float]]:
    words: list[str] = []
    weights: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        words.append(parts[0].strip())
        weights.append(float(parts[1]) if len(parts) > 1 else 1.0)
    if not words:
        raise SystemExit(f"語彙が空です: {path}")
    return words, weights


def is_backchannel(text: str, max_chars: int) -> bool:
    return 0 < len(text.strip().strip(STRIP_CHARS)) <= max_chars


def rewrite(
    dialogues: Sequence[dict[str, Any]],
    words: Sequence[str],
    weights: Sequence[float],
    max_chars: int,
    rewrite_all: bool,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    out: list[dict[str, Any]] = []
    replaced: dict[str, int] = {}
    kept: dict[str, int] = {}
    for dialogue in dialogues:
        turns: list[dict[str, Any]] = []
        for turn in dialogue.get("turns") or []:
            text = str(turn.get("text", "")).strip()
            if str(turn.get("speaker", "")).strip().lower() != LISTENER or not text:
                turns.append(turn)
                continue
            if not rewrite_all and not is_backchannel(text, max_chars):
                kept[text] = kept.get(text, 0) + 1
                turns.append(turn)
                continue
            word = rng.choices(list(words), weights=list(weights), k=1)[0]
            replaced[word] = replaced.get(word, 0) + 1
            turns.append(
                {
                    **turn,
                    "text": word,
                    # tts_text は読みの指定。原文と同じ語にしておかないと、
                    # 書き換えた語と違う音が合成される。
                    "tts_text": word,
                    "original_text": text,
                }
            )
        out.append({**dialogue, "turns": turns})
    return out, replaced, kept


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dialogues-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--text", default="", help="この 1 語だけにする（--dist より優先）")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument(
        "--all-listener-turns",
        action="store_true",
        help="長さを問わず聞き手のターンを全部書き換える",
    )
    parser.add_argument(
        "--user-only-jsonl",
        type=Path,
        default=None,
        help=(
            "聞き手のターンを落としたものも書き出す。TTS はこちらに掛ける -- "
            "聞き手の音はバンクから来るので作る必要が無く、作ると捨てる音が "
            "user 側のタイムラインを押してしまう"
        ),
    )
    parser.add_argument("--num-dialogues", type=int, default=0, help="先頭 N 本だけ")
    parser.add_argument(
        "--keep-text",
        action="store_true",
        help=(
            "相槌を書き換えず、user のみの写しと語彙ファイルだけ作る。対話生成の"
            "時点ですでに実録音の語彙で作ってある場合はこちら"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.text:
        words, weights = [args.text], [1.0]
    else:
        words, weights = load_distribution(args.dist)

    dialogues: list[dict[str, Any]] = []
    with args.dialogues_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    if args.num_dialogues > 0:
        dialogues = dialogues[: args.num_dialogues]
    if not dialogues:
        raise SystemExit(f"対話がありません: {args.dialogues_jsonl}")

    if args.keep_text:
        rewritten = list(dialogues)
        replaced = {}
        kept = {}
        for dialogue in rewritten:
            for turn in dialogue.get("turns") or []:
                text = str(turn.get("text", "")).strip()
                if str(turn.get("speaker", "")).strip().lower() != LISTENER or not text:
                    continue
                if is_backchannel(text, args.max_chars):
                    stripped = text.strip(STRIP_CHARS)
                    replaced[stripped] = replaced.get(stripped, 0) + 1
                else:
                    kept[text] = kept.get(text, 0) + 1
    else:
        rewritten, replaced, kept = rewrite(
            dialogues, words, weights, args.max_chars,
            args.all_listener_turns, random.Random(args.seed),
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for dialogue in rewritten:
            handle.write(json.dumps(dialogue, ensure_ascii=False) + "\n")

    vocab_path = args.out_jsonl.with_suffix(args.out_jsonl.suffix + ".vocab.txt")
    vocab_path.write_text(
        "# rewrite_aizuchi_vocab.py が実際に入れた語。\n"
        "# splice_aizuchi_bank.py --aizuchi-vocab-file にそのまま渡す。\n"
        + "\n".join(sorted(replaced)) + "\n",
        encoding="utf-8",
    )

    if args.user_only_jsonl is not None:
        args.user_only_jsonl.parent.mkdir(parents=True, exist_ok=True)
        kept_turns = 0
        with args.user_only_jsonl.open("w", encoding="utf-8") as handle:
            for dialogue in rewritten:
                turns = [
                    turn
                    for turn in dialogue["turns"]
                    if str(turn.get("speaker", "")).strip().lower() != LISTENER
                ]
                kept_turns += len(turns)
                handle.write(
                    json.dumps({**dialogue, "turns": turns}, ensure_ascii=False) + "\n"
                )
        print(f"user のみ: {kept_turns} ターン -> {args.user_only_jsonl}")

    total = sum(replaced.values())
    verb = "そのまま数えました" if args.keep_text else "書き換えました"
    print(f"対話 {len(rewritten)} 本 / 相槌 {total} 箇所を{verb}")
    for word, count in sorted(replaced.items(), key=lambda kv: -kv[1]):
        share = 100.0 * count / total if total else 0.0
        print(f"  {word} x{count} ({share:.0f}%)")
    if kept:
        print(f"書き換えなかった聞き手のターン: {sum(kept.values())} 箇所（長すぎるもの）")
        for text, count in sorted(kept.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {text[:30]} x{count}")
    print(f"出力: {args.out_jsonl}")
    print(f"語彙: {vocab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
