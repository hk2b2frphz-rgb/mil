#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孤独・孤立相談窓口を想定した use case カードを「軸の組み合わせ」で生成する。

シンプル方針:
- 5 軸を直積に近い形で組み合わせ、重複を弾いて 100 件程度を出す
- 配分はリスクレベルで層化（low:medium:high = 6:3:1 目安）
- 沈黙ありを 35% 程度混ぜる
- Gemma に渡せる use case JSONL 1 件ごとに 1 対話が生成される想定

出力:
  data/use_cases/loneliness_<N>.jsonl

使い方:
  uv run scripts/build_use_cases.py --out-path data/use_cases/loneliness_100.jsonl --num 100
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 軸の定義（シンプルに、辞書ではなく素朴なリストで持つ）
# ---------------------------------------------------------------------------

# (id_token, situation, opening_kind, default_target_turns, prefers_silence)
SITUATIONS: list[tuple[str, str, str, int, bool]] = [
    ("evening_smalltalk",      "夜、なんとなく人と話したくて窓口に来た", "smalltalk", 8,  False),
    ("holiday_alone",          "休日に予定がなく、誰ともつながっていない感じが強い", "feelings", 8,  False),
    ("sns_fatigue",            "SNSを見ていると置いていかれた気持ちになる",         "feelings", 8,  False),
    ("post_retirement",        "退職後、社会との接点が急に減って戸惑っている",     "feelings", 10, False),
    ("after_moving",           "引っ越したばかりで知り合いがいない",               "smalltalk", 8, False),
    ("caregiving_fatigue",     "介護で家にこもりがち、誰にも気持ちを話せない",     "feelings", 10, False),
    ("single_assignment",      "単身赴任で家族と離れて孤独",                       "feelings", 8,  False),
    ("post_bereavement",       "親しい人を亡くして気持ちの行き場がない",           "feelings", 12, True),
    ("hikikomori_phase",       "数か月、ほとんど家から出ていない",                 "feelings", 10, True),
    ("post_breakup",           "別れたばかりで誰にも言えない",                     "feelings", 10, False),
    ("childcare_isolation",    "育児で家にこもり、大人と話す機会がない",           "smalltalk", 8, False),
    ("workplace_isolation",    "職場で誰とも雑談できず孤独",                       "feelings", 8,  False),
    ("empty_nest",             "子どもが独立して家が静かすぎる",                   "feelings", 10, True),
    ("illness_isolation",      "療養中で外に出られない、見舞いも来ない",           "feelings", 10, True),
    ("financial_stress",       "お金の心配が募り、誰にも相談できない",             "feelings", 10, False),
    ("help_hesitation",        "助けを求めたいが迷惑だと思って言えない",           "feelings", 10, True),
    ("late_night_anxiety",     "夜中に不安で眠れず電話をした",                     "feelings", 10, True),
    ("midlife_drift",          "中年になり、人生の意味を見失っている感覚",         "feelings", 12, True),
    ("student_loneliness",     "上京して友人ができず孤独",                         "smalltalk", 8, False),
    ("post_job_loss",          "失業中で社会から切り離された感覚",                 "feelings", 10, True),
]

# (age_band, examples)
AGE_BANDS = ["20代", "30代", "40代", "50代", "60代", "70代"]
GENDERS = ["男性", "女性"]

# (id_token, opening_template)
OPENING_TEMPLATES = {
    "smalltalk": [
        "こんばんは。相談というほどでもないんですが、少し話してもいいですか。",
        "あの、ちょっと話を聞いてもらえますか。",
        "雑談でもいいですか。",
    ],
    "feelings": [
        "なんだか気持ちが落ち着かなくて電話しました。",
        "うまく言えないんですけど、最近寂しくて。",
        "聞いてもらえるだけでいいんです。",
    ],
    "silence": [
        "…もしもし。",
        "あの…、",
        "…すみません、何を話せばいいか分からなくて。",
    ],
}

# (id_token, label, weight)
SILENCE_PATTERNS = [
    ("none",       "none",       55),
    ("occasional", "occasional", 30),
    ("heavy",      "heavy",      15),
]

# risk 分布（重みでサンプリング）
RISK_WEIGHTS = [
    ("low",    60),
    ("medium", 30),
    ("high",   10),
]

# トーン指示（risk と組み合わせる）
TONES_BY_RISK = {
    "low": [
        "雑談寄り。相談員は歓迎し、急かさず軽い近況を聞く。",
        "気持ちの受け止めを中心に、今日できる小さな行動を一緒に探す。",
        "穏やかに相づちを返し、相手のペースで話を続けてもらう。",
    ],
    "medium": [
        "援助要請の難しさを認め、安全確認を押しつけず穏やかに行う。",
        "孤独感を軽く扱わず、共感を深めながらゆっくり話を進める。",
        "結論を急がず、気持ちの整理を一緒にゆっくり行う。",
    ],
    "high": [
        "安全確認を最優先するが、断定や指示は避け、穏やかに寄り添う。",
        "希死念慮への直接対応は避け、今ここで一緒にいることを伝える。",
        "緊急対応を装わず、相手のペースを尊重しつつ安全に関する短い問いを丁寧に挟む。",
    ],
}


@dataclass
class UseCase:
    id: str
    category: str
    risk_level: str
    situation: str
    user_profile: str
    opening: str
    opening_kind: str
    silence_pattern: str
    target_turns: int
    tone: str


def weighted_choice(rng: random.Random, items: list[tuple[Any, int]]) -> Any:
    population = [item for item, _ in items]
    weights = [w for _, w in items]
    return rng.choices(population, weights=weights, k=1)[0]


def build_one(rng: random.Random, index: int) -> UseCase:
    sit_token, situation, opening_kind, target_turns, prefers_silence = rng.choice(SITUATIONS)
    age = rng.choice(AGE_BANDS)
    gender = rng.choice(GENDERS)
    risk = weighted_choice(rng, RISK_WEIGHTS)

    # 沈黙パターン: situation が好む場合は heavy 寄りに、それ以外は通常分布
    if prefers_silence and rng.random() < 0.55:
        silence_pattern = rng.choices(["occasional", "heavy"], weights=[55, 45], k=1)[0]
    else:
        silence_pattern = weighted_choice(rng, [(t, w) for _, t, w in SILENCE_PATTERNS])

    # 沈黙ありなら opening_kind を silence にすることがある
    if silence_pattern == "heavy" and rng.random() < 0.4:
        chosen_opening_kind = "silence"
    else:
        chosen_opening_kind = opening_kind
    opening = rng.choice(OPENING_TEMPLATES[chosen_opening_kind])

    # high リスクは少し長めの target_turns
    if risk == "high":
        target_turns = max(target_turns, 12)
    elif risk == "medium":
        target_turns = max(target_turns, 10)

    # 沈黙ありなら target_turns を少し増やす（沈黙ターンも含むため）
    if silence_pattern == "heavy":
        target_turns += 2
    elif silence_pattern == "occasional":
        target_turns += 1

    tone = rng.choice(TONES_BY_RISK[risk])

    category = {
        "low":    "loneliness_light",
        "medium": "loneliness_deep",
        "high":   "loneliness_high",
    }[risk]

    user_profile = f"{age}{gender}。{situation}という背景がある。"

    use_case_id = f"{sit_token}_{age}_{gender}_{risk}_{silence_pattern}_{index:03d}"
    return UseCase(
        id=use_case_id,
        category=category,
        risk_level=risk,
        situation=situation,
        user_profile=user_profile,
        opening=opening,
        opening_kind=chosen_opening_kind,
        silence_pattern=silence_pattern,
        target_turns=target_turns,
        tone=tone,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="loneliness window use cases generator")
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.num <= 0:
        parser.error(f"--num must be a positive integer (got {args.num})")

    rng = random.Random(args.seed)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    written = 0
    with args.out_path.open("w", encoding="utf-8") as f:
        attempts = 0
        while written < args.num and attempts < args.num * 20:
            attempts += 1
            uc = build_one(rng, written + 1)
            if uc.id in seen_ids:
                continue
            seen_ids.add(uc.id)
            f.write(json.dumps(asdict(uc), ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} use cases to {args.out_path}")


if __name__ == "__main__":
    main()
