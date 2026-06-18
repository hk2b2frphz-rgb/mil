#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孤独・孤立相談窓口を想定した use case カードを「軸の組み合わせ」で生成する。

方針 (1000h 大規模生成向けに多様性を抜本拡張):
- 状況・話題・会話タイプ・職業・時間帯・季節を直交に近い形で組み合わせる
- 雑談/傾聴を基準に置き、相談だけに偏らない会話の「型」を持たせる
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
# 軸の定義
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
    # --- 拡張分 ---
    ("new_job_nervous",        "転職したばかりで職場になじめていない",             "feelings", 8,  False),
    ("commuter_solitude",      "毎日の通勤で誰とも話さず時間が過ぎていく",         "smalltalk", 8, False),
    ("weekend_routine",        "週末はいつも同じことの繰り返しで張り合いがない",   "smalltalk", 8, False),
    ("aging_parent_far",       "高齢の親が遠方にいて気がかりだが頼れる人がいない", "feelings", 10, False),
    ("divorce_adjusting",      "離婚してひとりの暮らしに慣れずにいる",             "feelings", 10, False),
    ("chronic_pain",           "持病の痛みがあり外出が億劫で人と疎遠になった",     "feelings", 10, True),
    ("night_shift_worker",     "夜勤続きで生活リズムが合わず人と会えない",         "smalltalk", 8, False),
    ("graduate_anxiety",       "卒業後の進路が見えず誰にも相談できない",           "feelings", 9,  False),
    ("freelancer_isolation",   "在宅フリーランスで一日中誰とも話さない",           "smalltalk", 8, False),
    ("returnee_culture",       "海外から帰国して周囲と話が合わない",               "smalltalk", 9, False),
    ("disaster_aftermath",     "災害のあと生活が一変し気持ちが落ち着かない",       "feelings", 11, True),
    ("language_barrier",       "地域に来て間もなく言葉や習慣の壁を感じる",         "smalltalk", 9, False),
    ("post_partum",            "出産後、社会から切り離された感覚がある",           "feelings", 10, False),
    ("exam_pressure",          "受験勉強で孤独に追い込まれている",                 "feelings", 9,  False),
    ("rural_isolation",        "地方在住で近くに同年代がいない",                   "smalltalk", 8, False),
    ("estranged_family",       "家族と疎遠で頼れる相手がいない",                   "feelings", 11, True),
    ("recovering_burnout",     "燃え尽きてしまい人と関わる気力がない",             "feelings", 10, True),
    ("lgbtq_isolation",        "周囲に本当の自分を話せず孤独を感じる",             "feelings", 11, True),
    ("widowed_senior",         "配偶者に先立たれ一人暮らしが長くなった",           "feelings", 12, True),
    ("job_hunting_fatigue",    "就職活動がうまくいかず気持ちが沈んでいる",         "feelings", 9,  False),
    ("hobby_lost",             "長く続けた趣味の仲間と離れてしまった",             "smalltalk", 8, False),
    ("relocation_for_work",    "転勤で見知らぬ土地に来て心細い",                   "feelings", 9,  False),
    ("social_anxiety",         "人と話すのが怖くて関係を築けない",                 "feelings", 10, True),
    ("digital_only_contact",   "家族とは連絡するが直接会う人がいない",             "smalltalk", 8, False),
    ("seasonal_blues",         "季節の変わり目で気分が落ち込みやすい",             "feelings", 9,  False),
    ("late_evening_quiet",     "仕事を終えた後の静けさが急に寂しくなる",           "smalltalk", 8, False),
]

AGE_BANDS = ["20代", "30代", "40代", "50代", "60代", "70代"]
GENDERS = ["男性", "女性"]

# 職業・立場（ペルソナの幅を広げる）
OCCUPATIONS = [
    "会社員", "パート勤め", "自営業", "学生", "求職中", "退職した人",
    "主婦・主夫", "医療・介護職", "技術職", "サービス業", "公務員", "農業に従事",
    "在宅ワーカー", "アルバイト", "経営者",
]

# 話題ドメイン（雑談・傾聴の中身を多様化）
TOPICS = [
    "今日あった小さな出来事", "最近の天気や季節の変化", "食事や料理のこと",
    "昔の思い出話", "ペットや動物の話", "テレビやドラマ、動画の話",
    "散歩や近所の風景", "睡眠や体調のこと", "仕事や勉強の近況",
    "家族や親しい人のこと", "趣味や好きなこと", "買い物や日用品の話",
    "音楽や本の話", "週末の過ごし方", "ちょっとした悩みごと",
    "将来やこれからのこと", "健康や運動のこと", "ニュースで気になったこと",
]

# 会話の「型」（相談一辺倒にしない）
# (id_token, label, guidance, weight)
CONVERSATION_TYPES = [
    ("smalltalk",   "雑談",     "肩の力を抜いた雑談。結論を求めず、近況や好きな話題を行き来する。", 30),
    ("listening",   "傾聴",     "相手の話をじっくり聴く。先回りせず、相づちと短い問い返しで話を促す。", 30),
    ("venting",     "愚痴",     "もやもやや不満を吐き出してもらう。否定せず受け止め、評価や助言を急がない。", 15),
    ("rambling",    "とりとめ", "話があちこち飛ぶのを自然に受け止め、流れに沿って相づちを返す。", 12),
    ("reflection",  "ふり返り", "今日や最近を一緒にふり返る。気持ちの整理を手伝うが結論は急がない。", 8),
    ("seeking",     "相談寄り", "軽い相談ごと。気持ちを受け止めつつ、本人のペースで一緒に考える。", 5),
]

TIME_OF_DAY = [
    ("morning",   "朝"),
    ("daytime",   "昼間"),
    ("evening",   "夕方"),
    ("night",     "夜"),
    ("late_night","深夜"),
]

# 話者の性格・気質（対話を通して一貫する）。話し方の幅を作る。
# (id_token, label, guidance)
PERSONALITIES: list[tuple[str, str, str]] = [
    ("quiet",       "物静かで口数が少ない",       "短い返答が多く、間が空きやすい。"),
    ("talkative",   "おしゃべりで話が長い",       "一つの話題を長く話し、脱線もする。"),
    ("cheerful",    "明るく前向き",               "つらい話でも時々冗談や笑いを交える。"),
    ("worrier",     "心配性で慎重",               "確認や言い直しが多く、不安を口にする。"),
    ("dependent",   "人懐っこく甘え上手",         "相手に頼り、相づちや反応を求める。"),
    ("reserved",    "遠慮がちで気を遣う",         "申し訳なさを口にし、本音を出すのが遅い。"),
    ("cynical",     "皮肉っぽく斜に構える",       "自虐や皮肉を交えつつ、本心は隠しがち。"),
    ("earnest",     "まじめで几帳面",             "順序立てて丁寧に話す。"),
    ("emotional",   "涙もろく感情的",             "感情の起伏が大きく、声に出やすい。"),
    ("easygoing",   "飄々としてマイペース",       "深刻ぶらず、淡々と自分のペースで話す。"),
    ("shorttemper", "短気で苛立ちやすい",         "苛立ちや不満が言葉に出やすい。"),
    ("calmnatured", "穏やかで落ち着いている",     "感情を荒げず、ゆったり話す。"),
]

# その日の感情状態・声の出し方。落ち着きを多数派にしつつ強い状態も混ぜる。
# (id_token, label, emotion_hint, weight)
EMOTIONAL_STATES: list[tuple[str, str, str, int]] = [
    ("steady",       "落ち着いている",   "neutral / relieved 寄り。",                    34),
    ("low",          "気分が沈んでいる", "sad / lonely / weary を多めに。",              22),
    ("anxious",      "不安が強い",       "anxious / hesitant を多めに。",                14),
    ("tearful",      "涙ぐんでいる",     "tearful を交え、強い場面で sobbing も使う。",  10),
    ("high_tension", "ハイテンション",   "high_tension / laughing を多めに、早口気味。", 8),
    ("agitated",     "気持ちが高ぶる",   "agitated / irritable を交える。",              7),
    ("withdrawn",    "ふさぎ込んでいる", "withdrawn を多めに、声が小さく途切れがち。",   5),
]

# (id_token, opening_template)
OPENING_TEMPLATES = {
    "smalltalk": [
        "こんばんは。相談というほどでもないんですが、少し話してもいいですか。",
        "あの、ちょっと話を聞いてもらえますか。",
        "雑談でもいいですか。",
        "特に用事はないんですけど、少しだけ話したくて。",
        "こんにちは。なんとなく声が聞きたくなって。",
        "ちょっとだけ、近況を話してもいいですか。",
        "暇つぶしみたいで申し訳ないんですけど、いいですか。",
    ],
    "feelings": [
        "なんだか気持ちが落ち着かなくて電話しました。",
        "うまく言えないんですけど、最近寂しくて。",
        "聞いてもらえるだけでいいんです。",
        "ちょっと気持ちが重たくて、話したくなって。",
        "誰かに話さないと、抱えきれない感じがして。",
        "理由はうまく言えないんですけど、つらくて。",
        "最近ずっと、心細い感じが続いているんです。",
    ],
    "silence": [
        "…もしもし。",
        "あの…、",
        "…すみません、何を話せばいいか分からなくて。",
        "…えっと。",
        "…うまく言えないんですけど。",
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
        "明るすぎず暗すぎず、自然な間で会話のキャッチボールを続ける。",
        "相手の好きな話題に乗り、興味を持って質問を返す。",
    ],
    "medium": [
        "援助要請の難しさを認め、安全確認を押しつけず穏やかに行う。",
        "孤独感を軽く扱わず、共感を深めながらゆっくり話を進める。",
        "結論を急がず、気持ちの整理を一緒にゆっくり行う。",
        "相手の言葉を言い換えて返し、理解していることを丁寧に伝える。",
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
    conversation_type: str = "smalltalk"
    conversation_guidance: str = ""
    topic: str = ""
    time_of_day: str = ""
    personality: str = ""
    personality_guidance: str = ""
    emotional_state: str = ""
    emotional_state_hint: str = ""


def weighted_choice(rng: random.Random, items: list[tuple[Any, int]]) -> Any:
    population = [item for item, _ in items]
    weights = [w for _, w in items]
    return rng.choices(population, weights=weights, k=1)[0]


def build_one(rng: random.Random, index: int) -> UseCase:
    sit_token, situation, opening_kind, target_turns, prefers_silence = rng.choice(SITUATIONS)
    age = rng.choice(AGE_BANDS)
    gender = rng.choice(GENDERS)
    occupation = rng.choice(OCCUPATIONS)
    risk = weighted_choice(rng, RISK_WEIGHTS)
    topic = rng.choice(TOPICS)
    time_token, time_label = rng.choice(TIME_OF_DAY)
    pers_token, pers_label, pers_guidance = rng.choice(PERSONALITIES)
    state_token, state_label, state_hint, _ = weighted_choice(
        rng, [(item, item[3]) for item in EMOTIONAL_STATES]
    )
    conv_token, conv_label, conv_guidance, _ = weighted_choice(
        rng, [(item, item[3]) for item in CONVERSATION_TYPES]
    )

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

    # 雑談・傾聴・とりとめは少し長めにして相づちの余地を作る
    if conv_token in {"listening", "rambling", "venting"}:
        target_turns += 2

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

    user_profile = (
        f"{age}{gender}、{occupation}。{situation}という背景がある。"
        f"性格は{pers_label}({pers_guidance})。"
        f"今日は{time_label}に話していて、{state_label}様子。"
    )

    use_case_id = (
        f"{sit_token}_{conv_token}_{pers_token}_{state_token}_"
        f"{age}_{gender}_{risk}_{silence_pattern}_{index:05d}"
    )
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
        conversation_type=conv_token,
        conversation_guidance=conv_guidance,
        topic=topic,
        time_of_day=time_label,
        personality=pers_label,
        personality_guidance=pers_guidance,
        emotional_state=state_label,
        emotional_state_hint=state_hint,
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
