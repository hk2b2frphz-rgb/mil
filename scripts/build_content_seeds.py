#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
具体的な「話の種」(talking points) を Gemma で大量に作り、バンク化する。

なぜ必要か:
  ペルソナ軸を増やしても、Gemma が話す「中身」は定型に収束しがち
  (「最近寂しくて」->「そうなんですね」)。そこで、平凡で具体的な出来事
  ("飼い猫が初めて膝に乗った" 等) を一度に大量生成しておき、対話生成時に
  各対話へ 1〜2 個ずつ注入することで、語彙と内容を強制的にばらけさせる。

  生成は一度きり(数千件)で済み、24,000 対話のコストに対して誤差。
  作ったバンクは build_full_duplex_training_use_cases.py の --content-seeds で
  読み込む。

使い方(GPU サーバで一度だけ):
  uv run python scripts/build_content_seeds.py \
      --out data/content_seeds/seeds.jsonl \
      --num 3000 \
      --gemma-model google/gemma-4-E2B-it \
      --gemma-dtype float16
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_synthetic_moshi_training_data import (  # noqa: E402
    GemmaDialogueGenerator,
)

# 話の種を広い生活ドメインに散らすためのヒント。各 brainstorm でランダムに使う。
DOMAINS = [
    "食べ物や料理", "天気や季節の移り変わり", "ペットや近所の動物",
    "テレビ・ラジオ・動画・配信", "散歩や近所の風景", "買い物や日用品",
    "睡眠や体調の小さな変化", "趣味や手仕事", "昔の思い出や懐かしいこと",
    "家族や親戚のちょっとした出来事", "音楽や本", "植物や園芸",
    "交通機関や移動中の出来事", "季節の行事やお祭り", "ご近所付き合い",
    "仕事や勉強の合間の小さな出来事", "スマホやネットで見たこと",
    "健康のために始めたこと", "ふと思い出した心配ごと", "最近うれしかった小さなこと",
]

PROMPT_TEMPLATE = """\
日本語の音声対話の学習データ用に、相談窓口に来た人が「ちょっと話したくなる」
ような、ごく平凡で具体的な出来事・話題を {n} 個考えてください。

テーマの中心: {domain}

条件:
- 1 件 12〜30 文字程度の、具体的で日常的な一言。
- 深刻すぎず、雑談・傾聴で自然に出てくる粒度。
- 固有名詞や具体的な細部(色・数・場所・人)を入れて、互いに似ないようにする。
- 説明や前置きは不要。JSON 配列のみを出力する。

出力形式(JSON 配列のみ):
["飼い猫が初めて膝に乗ってきた", "近所のパン屋が今月で閉店するらしい", ...]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Gemma talking-point seed bank.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num", type=int, default=3000, help="Target number of unique seeds.")
    parser.add_argument("--per-prompt", type=int, default=20, help="Items requested per Gemma call.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gemma-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--gemma-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--gemma-device-map", default="auto")
    parser.add_argument("--gemma-task", default="any-to-any")
    parser.add_argument("--gemma-temperature", type=float, default=1.0)
    parser.add_argument("--gemma-max-new-tokens", type=int, default=900)
    parser.add_argument("--gemma-timeout-sec", type=float, default=21600.0)
    parser.add_argument("--gemma-python", default=None)
    parser.add_argument(
        "--gemma-worker",
        type=Path,
        default=Path(__file__).with_name("gemma_dialogue_worker.py"),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def extract_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    n_prompts = max(1, (args.num + args.per_prompt - 1) // args.per_prompt)
    # 余分に作って重複除去後に目標数へ削るため 1.4 倍ほど投げる
    n_prompts = int(n_prompts * 1.4) + 1

    prompts: list[str] = []
    for _ in range(n_prompts):
        domain = rng.choice(DOMAINS)
        prompts.append(PROMPT_TEMPLATE.format(n=args.per_prompt, domain=domain))

    generator = GemmaDialogueGenerator(
        SimpleNamespace(
            gemma_backend="transformers-subprocess",
            gemma_python=args.gemma_python,
            gemma_worker=args.gemma_worker,
            gemma_model=args.gemma_model,
            gemma_task=args.gemma_task,
            gemma_device_map=args.gemma_device_map,
            gemma_dtype=args.gemma_dtype,
            gemma_temperature=args.gemma_temperature,
            gemma_max_new_tokens=args.gemma_max_new_tokens,
            gemma_timeout_sec=args.gemma_timeout_sec,
            trust_remote_code=args.trust_remote_code,
        )
    )

    print(f"[seeds] generating with {len(prompts)} Gemma prompts ...")
    texts = generator.generate_transformers_subprocess_batch(prompts)

    seen: set[str] = set()
    seeds: list[str] = []
    for text in texts:
        for item in extract_json_array(text):
            norm = re.sub(r"\s+", "", item)
            if norm and norm not in seen:
                seen.add(norm)
                seeds.append(item)

    rng.shuffle(seeds)
    seeds = seeds[: args.num]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in seeds:
            handle.write(json.dumps({"text": item}, ensure_ascii=False) + "\n")

    print(f"[seeds] wrote {len(seeds)} unique talking points -> {args.out}")
    if len(seeds) < args.num:
        print(
            f"[seeds] NOTE: got {len(seeds)} < requested {args.num}. "
            "Increase --num headroom or --per-prompt."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
