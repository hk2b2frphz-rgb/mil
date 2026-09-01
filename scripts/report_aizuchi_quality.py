#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相づちだけモードの生成物を測る。

これまで生成のたびに手元で数えていたものを、そのままスクリプトにした。狙いは
「聞き手として成立しているか」を目で読む前に絞り込むこと。読まないと分からない
ことは残るが、壊れ方の多くは数字に出る。

  uv run python scripts/report_aizuchi_quality.py \
      data/runs/qwen_aizuchi_1000_normal_v9/llm_dialogues/dialogues.jsonl

複数渡せば横に並べて比べられる（プリセット間の比較はこれでやる）:

  uv run python scripts/report_aizuchi_quality.py data/runs/qwen_aizuchi_1000_*/llm_dialogues/dialogues.jsonl

秒数は「日本語の発話は 6 文字/秒」という仮定に基づく推定値で、合成した音声の
長さとは違う。プリセット間の比較や、修正の前後を見る用途に使う。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

CHARS_PER_SECOND = 6.0

# 話し手が聞き手の存在を確かめる言い方。
PROBE_RE = re.compile(
    r"もしもし|聞いて(い)?ま|通じて|繋がって|つながって|聞こえ|いますか|いますよね"
)
# 本文に残ってはいけないもの。
TAG_RE = re.compile(r"<|>|pause", re.IGNORECASE)
LATIN_RE = re.compile(r"[A-Za-z]{3,}")
# 日本語には出てこない簡体字・中国語の助詞（Qwen 由来の混入を拾う）。
CHINESE_RE = re.compile(r"[啊吧呢咱们这么吗儿说没对话时间里线呗]")
# 終話の言い回し。対話の最後以外に出てきたら、締めそこねている。
FAREWELL_RE = re.compile(r"失礼します|失礼いたします|さようなら|ご機嫌よう|また、?お電話|それでは、?また")


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def measure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    user_chars = 0
    silence_sec = 0.0
    silence_count = 0
    words: collections.Counter[str] = collections.Counter()
    fixed = 0
    probes = 0
    probes_max = 0
    moshi_runs = 0
    empty_user = 0
    tag_leak = 0
    latin = 0
    chinese = 0
    long_user = 0
    late_farewell = 0
    user_turns = 0

    for dialogue in rows:
        turns = dialogue.get("turns", [])
        probes_here = 0
        farewells_here = 0
        for index, turn in enumerate(turns):
            speaker = turn.get("speaker")
            note = turn.get("note") or ""
            text = str(turn.get("text") or "")
            if speaker == "silence":
                silence_sec += float(turn.get("duration_sec") or 0.0)
                silence_count += 1
                continue
            if speaker == "moshi":
                words[text] += 1
                if note.startswith("固定"):
                    fixed += 1
                if index and turns[index - 1].get("speaker") == "moshi":
                    moshi_runs += 1
                continue
            user_turns += 1
            user_chars += len(text)
            if not text.strip():
                empty_user += 1
            if TAG_RE.search(text):
                tag_leak += 1
            latin += len(LATIN_RE.findall(text))
            if CHINESE_RE.search(text):
                chinese += 1
            if len(text) > 60:
                long_user += 1
            if PROBE_RE.search(text) and len(text) < 45:
                probes_here += 1
            if FAREWELL_RE.search(text):
                farewells_here += 1
        probes += probes_here
        probes_max = max(probes_max, probes_here)
        # 締めを1回言って終わるのが自然。2回以上は言い直している。
        late_farewell += max(0, farewells_here - 1)

    speech = user_chars / CHARS_PER_SECOND
    reactions = sum(words.values()) - fixed
    n = max(1, len(rows))
    return {
        "dialogues": len(rows),
        "speech_sec": speech,
        "speech_per_dialogue": speech / n,
        "reactions": reactions,
        "gap_sec": speech / reactions if reactions else 0.0,
        "silence_pct": silence_sec / (speech + silence_sec) * 100 if speech else 0.0,
        "silence_per_dialogue": silence_count / n,
        "user_turn_chars": user_chars / max(1, user_turns),
        "vocab": len(words),
        "words": words,
        "probes_per_dialogue": probes / n,
        "probes_max": probes_max,
        "moshi_runs": moshi_runs,
        "empty_user": empty_user,
        "tag_leak": tag_leak,
        "latin": latin,
        "chinese": chinese,
        "long_user": long_user,
        "late_farewell": late_farewell,
    }


def print_table(reports: list[tuple[str, dict[str, Any]]]) -> None:
    rows = [
        ("相づち間隔", "{gap_sec:.1f}秒"),
        ("沈黙の割合", "{silence_pct:.0f}%"),
        ("沈黙/本", "{silence_per_dialogue:.1f}回"),
        ("発話/本", "{speech_per_dialogue:.0f}秒"),
        ("user1発話", "{user_turn_chars:.0f}字"),
        ("語の種類", "{vocab}"),
        ("確認/本", "{probes_per_dialogue:.1f}回"),
        ("確認の最多", "{probes_max}回"),
    ]
    faults = [
        ("moshi の連続", "moshi_runs"),
        ("空の user 発話", "empty_user"),
        ("タグの流出", "tag_leak"),
        ("英字", "latin"),
        ("簡体字", "chinese"),
        ("60字超の発話", "long_user"),
        ("締めの言い直し", "late_farewell"),
    ]
    width = max(12, max(len(name) for name, _ in reports) + 1)
    header = "".ljust(14) + "".join(name[:width].rjust(width) for name, _ in reports)
    print(header)
    print("-" * len(header))
    for label, fmt in rows:
        line = label.ljust(14)
        for _, report in reports:
            line += fmt.format(**report).rjust(width)
        print(line)
    print()
    print("壊れ方（0 が正常）".ljust(14) + "".join("".rjust(width) for _ in reports))
    for label, key in faults:
        line = label.ljust(14)
        for _, report in reports:
            value = report[key]
            line += (str(value) if value else "-").rjust(width)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="dialogues.jsonl")
    parser.add_argument(
        "--words", action="store_true", help="相づち語の分布も出す（偏りを見る）"
    )
    args = parser.parse_args()

    reports: list[tuple[str, dict[str, Any]]] = []
    for path in args.paths:
        if not path.is_file():
            print(f"見つかりません: {path}", file=sys.stderr)
            continue
        rows = load(path)
        if not rows:
            print(f"空です: {path}", file=sys.stderr)
            continue
        # data/runs/<batch>/llm_dialogues/dialogues.jsonl の <batch> を名前にする。
        name = path.parent.parent.name or path.stem
        reports.append((name.replace("qwen_aizuchi_1000_", ""), measure(rows)))

    if not reports:
        raise SystemExit("読めた対話がありません。")

    print_table(reports)

    if args.words:
        for name, report in reports:
            print(f"\n=== {name} の相づち")
            total = sum(report["words"].values())
            for word, count in report["words"].most_common():
                print(f"  {word:26s} {count:4d}  {count / total * 100:4.1f}%")


if __name__ == "__main__":
    main()
