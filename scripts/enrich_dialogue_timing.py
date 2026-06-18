#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma が生成した対話 JSONL に、日本語会話に近い相づち(あいづち)と
ターン構造を後付けする。

なぜ必要か:
  3h 微調整では「相づちが少なく遅い・応答が遅い」モデルになった。原因の一つは
  学習データのタイミング設計で、相づちが 1 ターンに 1 回・発話終了後に偏っていた
  こと。日本語の自然会話では相づちは非常に頻繁で、相手の発話の句切れごとに
  短く差し込まれる。本スクリプトはその構造をデータ側に作り込む。

日本語会話タイミングの根拠(文献):
  - ターン交替の間(gap)は言語間で日本語が最短クラス。平均はほぼ 0〜0.2 秒
    (Stivers et al. 2009, PNAS "Universals and cultural variation in turn-taking")。
    => 応答 gap はレンダラ側の --gap-sec を小さく(約 0.2)することで実現する。
  - 相づち(あいづち)は傾聴・雑談で頻繁。聞き手は相手発話のおよそ数秒ごと、
    句や節の切れ目で短い相づちを返す (Maynard 1989; Den et al. の研究)。
    => 長い user 発話を句切れで分割し、各句の終盤に短い相づちを重ねる。
  - 相づちは話題を奪わない短い語(うん/はい/そうですね 等)。

設計:
  - duplex_task が null の自由対話(雑談・傾聴)だけを加工する。
    ラベル付きタスク(backchannel / user_interruption など)は厳密な検証規則を
    壊さないため、そのまま通す。
  - 長い user 発話を 1〜3 個の句に分割し、句の終盤に moshi の相づちを
    overlap_previous で差し込む。相づちには event="model_backchannel" を付与。
  - 文字数から発話長を推定して相づち位置(秒)を決める。すべて seed で決定的。

使い方:
  uv run python scripts/enrich_dialogue_timing.py \
      --in  data/runs/<RUN>/gemma_dialogues/dialogues.jsonl \
      --out data/runs/<RUN>/gemma_dialogues/dialogues_enriched.jsonl \
      --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

# 短い相づち語(話題を奪わない)。weight で頻度を調整。
AIZUCHI_POOL: list[tuple[str, int]] = [
    ("うん。", 22),
    ("うんうん。", 10),
    ("はい。", 16),
    ("ええ。", 8),
    ("そうですね。", 10),
    ("そうなんですね。", 9),
    ("なるほど。", 7),
    ("へえ。", 5),
    ("うんうん、それで。", 4),
    ("ふんふん。", 3),
    ("わかります。", 6),
]

AIZUCHI_EMOTIONS = ["gentle", "warm", "empathetic", "reassuring"]

# 句切り文字。ここで分割して句の終盤に相づちを置く。
CLAUSE_SPLIT_RE = re.compile(r"(?<=[、。！？!?])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich Gemma dialogues with frequent, natural Japanese backchannels."
    )
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--chars-per-sec",
        type=float,
        default=6.5,
        help="Estimated Japanese speaking rate to convert text length to seconds.",
    )
    parser.add_argument(
        "--aizuchi-prob",
        type=float,
        default=0.75,
        help="Probability of adding backchannels to a qualifying user turn.",
    )
    parser.add_argument(
        "--min-turn-sec",
        type=float,
        default=2.2,
        help="User turns estimated shorter than this are not split for backchannels.",
    )
    parser.add_argument(
        "--sec-per-aizuchi",
        type=float,
        default=3.2,
        help="Roughly one backchannel per this many seconds of user speech.",
    )
    parser.add_argument(
        "--max-aizuchi-per-turn",
        type=int,
        default=3,
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc


def est_sec(text: str, chars_per_sec: float) -> float:
    chars = len(re.sub(r"\s+", "", text))
    return chars / max(1e-6, chars_per_sec)


def split_into_chunks(text: str, k: int) -> list[str]:
    """text を句切れで k 個程度のまとまりに分割する。"""
    if k <= 1:
        return [text]
    clauses = [c for c in CLAUSE_SPLIT_RE.split(text) if c.strip()]
    if len(clauses) <= 1:
        # 句読点が無い場合は文字数で等分
        n = len(text)
        size = max(1, n // k)
        return [text[i : i + size] for i in range(0, n, size)][:k] or [text]
    # clauses を k 個のチャンクへ均等にまとめる
    chunks: list[str] = []
    per = max(1, len(clauses) / k)
    acc = ""
    target = per
    for idx, clause in enumerate(clauses, start=1):
        acc += clause
        if idx >= round(target) and len(chunks) < k - 1:
            chunks.append(acc)
            acc = ""
            target += per
    if acc:
        chunks.append(acc)
    return [c for c in chunks if c] or [text]


def make_aizuchi(rng: random.Random, offset_sec: float) -> dict[str, Any]:
    text = rng.choices(
        [t for t, _ in AIZUCHI_POOL],
        weights=[w for _, w in AIZUCHI_POOL],
        k=1,
    )[0]
    return {
        "speaker": "moshi",
        "text": text,
        "emotion": rng.choice(AIZUCHI_EMOTIONS),
        "timing": "overlap_previous",
        "start_after_previous_start_sec": round(offset_sec, 3),
        "event": "model_backchannel",
    }


def enrich_user_turn(
    turn: dict[str, Any],
    rng: random.Random,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """1 つの user 発話を (分割 user 句 + 相づち) の列へ展開する。"""
    text = str(turn.get("text", "")).strip()
    if not text:
        return [turn]

    dur = est_sec(text, args.chars_per_sec)
    if dur < args.min_turn_sec or rng.random() > args.aizuchi_prob:
        return [turn]

    n_aizuchi = max(1, min(args.max_aizuchi_per_turn, int(dur / args.sec_per_aizuchi)))
    chunks = split_into_chunks(text, n_aizuchi)
    if len(chunks) == 1:
        # 分割できない短文。終盤に 1 回だけ相づちを重ねる。
        out = [dict(turn, text=chunks[0])]
        offset = est_sec(chunks[0], args.chars_per_sec) * rng.uniform(0.6, 0.85)
        out.append(make_aizuchi(rng, offset))
        return out

    out: list[dict[str, Any]] = []
    base_emotion = turn.get("emotion")
    for ci, chunk in enumerate(chunks):
        user_chunk = {
            "speaker": "user",
            "text": chunk,
        }
        if base_emotion:
            user_chunk["emotion"] = base_emotion
        # 元 turn の付帯フィールド(voice_role 等)は先頭句のみ引き継ぐ
        if ci == 0:
            for key in ("voice_role", "gain"):
                if key in turn and turn[key] not in (None, ""):
                    user_chunk[key] = turn[key]
        out.append(user_chunk)
        # 句の終盤に相づちを重ねる(最後の句の後は本応答が続くので確率低め)
        place = ci < len(chunks) - 1 or rng.random() < 0.4
        if place:
            chunk_sec = est_sec(chunk, args.chars_per_sec)
            offset = chunk_sec * rng.uniform(0.55, 0.85)
            out.append(make_aizuchi(rng, offset))
    return out


def enrich_dialogue(dialogue: dict[str, Any], rng: random.Random, args: argparse.Namespace) -> dict[str, Any]:
    # ラベル付きタスクは厳密な検証規則を壊さないため触らない
    if dialogue.get("duplex_task"):
        return dialogue
    turns = dialogue.get("turns")
    if not isinstance(turns, list):
        return dialogue

    new_turns: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip().lower()
        if speaker == "user" and str(turn.get("timing") or "sequential") == "sequential":
            new_turns.extend(enrich_user_turn(turn, rng, args))
        else:
            new_turns.append(turn)

    enriched = dict(dialogue)
    enriched["turns"] = new_turns
    enriched["timing_enriched"] = True
    return enriched


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = list(iter_jsonl(args.in_path))
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    n_enriched = 0
    n_aizuchi = 0
    with args.out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            before = sum(
                1
                for t in (row.get("turns") or [])
                if isinstance(t, dict) and t.get("event") == "model_backchannel"
            )
            out = enrich_dialogue(row, rng, args)
            after = sum(
                1
                for t in (out.get("turns") or [])
                if isinstance(t, dict) and t.get("event") == "model_backchannel"
            )
            if out.get("timing_enriched"):
                n_enriched += 1
                n_aizuchi += max(0, after - before)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        f"[enrich] {len(rows)} dialogues -> {args.out_path} "
        f"({n_enriched} enriched, +{n_aizuchi} backchannels)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
