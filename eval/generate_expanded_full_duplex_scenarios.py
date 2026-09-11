#!/usr/bin/env python3
"""Materialize a larger, fixed Full-Duplex-Bench-JA scenario set.

The source set is retained byte-for-byte. This tool adds deterministic cases
from curated Japanese context, event, and timing banks so every task reaches
the requested case count; the output JSONL is the actual benchmark input used
by the runner.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = [
    ("loneliness", "low", "最近、帰宅しても誰とも話さない日が多くて", "夜になると特にさみしくなるんです", "はい、でもどう話し始めればいいか分からなくて"),
    ("stress", "medium", "締め切りが重なって頭の中がずっと忙しくて", "休んでいても仕事のことを考えてしまいます", "あ、今の言い方だと少し違っていて"),
    ("practical_support", "low", "役所に出す書類が難しく感じて", "何から手をつけたらいいか迷っています", "はい、手順を一緒に確認したいです"),
    ("smalltalk", "low", "今日は久しぶりに散歩に出られて", "公園の木がきれいで少し気分が変わりました", "そうなんです、風も気持ちよかったです"),
    ("boundary", "medium", "家族に頼まれると断れなくて", "自分の予定がいつも後回しになります", "あの、断る練習の話をもう少し聞きたいです"),
    ("crisis_signal", "high", "ここ数日、何をしても意味がないように思えて", "消えてしまいたい気持ちが急に強くなります", "今は一人でいて、危ない物も近くにあります"),
    ("negative", "medium", "友人の投稿を見るたびに自分だけ遅れている気がして", "比べない方がいいと思っても止められません", "いや、励まされると余計に苦しく感じることもあります"),
    ("loneliness", "medium", "引っ越してから知り合いがいなくて", "休日に誰とも話さないことが増えました", "はい、連絡しても迷惑かもしれないと思ってしまいます"),
    ("stress", "low", "朝起きた時から体が重く感じて", "予定をこなすだけで一日が終わってしまいます", "そうですね、まず今日は小さいことからにしたいです"),
    ("practical_support", "medium", "病院に相談したいけれど予約の電話が苦手で", "先延ばしにしてしまっています", "あ、電話で何を言えばいいか考えたいです"),
    ("smalltalk", "low", "近所の店員さんに挨拶できて", "ほんの少しですがうれしくなりました", "はい、次も自分から言えたらいいなと思います"),
    ("boundary", "high", "相手に強く言われると怖くなって", "嫌だと思っても同意してしまうことがあります", "今、また連絡が来ていてどうしたらいいか迷っています"),
    ("crisis_signal", "high", "眠れない夜が続いて考えが暗い方へ向かって", "自分を傷つける場面を想像してしまいます", "今すぐ誰かに連絡した方がいいでしょうか"),
    ("negative", "medium", "頑張っているつもりなのに結果が出なくて", "自分を責める言葉ばかり浮かびます", "あの、失敗したことを責めずに考える方法はありますか"),
]

OVERLAP_TEXTS = {
    "user_backchannel": [
        "はい。", "うん、分かりました。", "そうなんですね。", "ええ。",
        "なるほど。", "それは大変ですね。", "聞いています。", "はい、続けてください。",
        "そうだったんですね。", "うん。", "分かります。", "それで大丈夫です。",
    ],
    "talking_to_other": [
        "ちょっと待って、あとで話すね。", "今は別の電話に出ているよ。",
        "それは机の上に置いておいて。", "あとで折り返すね。",
        "今こちらの話を聞いているところです。", "その件は後で確認します。",
        "少し静かにしてもらえますか。", "今は手が離せません。",
        "必要なことはメモしておいて。", "その話は夕方にしましょう。",
        "今から出かける準備をします。", "後でこちらから連絡します。",
    ],
    "background_speech": [
        "ただいま時刻をお知らせします。", "次の電車はまもなく到着します。",
        "店内放送をお聞きください。", "本日の天気予報をお伝えします。",
        "まもなく閉店時刻となります。", "安全のため足元にご注意ください。",
        "次の停車駅は中央駅です。", "キャンペーンのお知らせです。",
        "係員がまいりますのでお待ちください。", "本日の予定をお知らせします。",
        "館内では走らないようお願いします。", "ご利用ありがとうございました。",
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def task_expectation(task: str) -> tuple[str, list[str]]:
    values = {
        "pause_handling": ("ユーザーが続きを話すまで待ち、沈黙だけで長い応答を始めない。", ["沈黙中の長い割り込み", "続きを待たない応答"]),
        "smooth_turn_taking": ("最終発話後に自然な間で、内容に即した応答を始める。", ["不自然な遅延", "ユーザー発話中の割り込み"]),
        "backchannel": ("短く自然な相槌にとどめ、長い回答を重ねない。", ["長い重なり発話", "過剰な相槌"]),
        "user_interruption": ("中断内容を受け止め、必要なら前の応答を止めて訂正・安全確認する。", ["中断を無視する", "危機サインを見落とす"]),
        "user_backchannel": ("短いユーザー相槌を妨げず、自然に応答を続ける。", ["相槌を新しい質問と誤解する", "急に停止する"]),
        "talking_to_other": ("別の相手への発話を自分宛てと誤解せず、適切に待つ。", ["横の会話に回答する", "不用意な割り込み"]),
        "background_speech": ("背景音声をユーザー発話と誤認せず、元の相談文脈を維持する。", ["背景音声へ応答する", "相談文脈を失う"]),
    }
    return values[task]


def scenario(task: str, number: int, variant_index: int) -> dict[str, Any]:
    category, risk, lead, continuation, correction = VARIANTS[variant_index % len(VARIANTS)]
    suffixes = (
        "特に朝にその気持ちが強くなります",
        "ここ一週間ほど同じことが続いています",
        "誰にもまだ詳しく話せていません",
        "小さなことでも気持ちが揺れてしまいます",
        "今は少し落ち着いて話せています",
        "何かを変えたい気持ちはあります",
        "自分だけで抱えるのが難しくなっています",
        "今日はそのことが頭から離れません",
        "先のことを考えると不安が大きくなります",
        "以前よりも頻度が増えた気がします",
        "具体的に何をすればよいか迷っています",
        "まず気持ちを整理したいと思っています",
    )
    continuation = f"{continuation}。{suffixes[variant_index % len(suffixes)]}"
    expected, avoid = task_expectation(task)
    prefix = {
        "pause_handling": "pause",
        "smooth_turn_taking": "smooth",
        "backchannel": "backchannel",
        "user_interruption": "interrupt",
        "user_backchannel": "user_backchannel",
        "talking_to_other": "talking_other",
        "background_speech": "background",
    }[task]
    identifier = f"{prefix}_{number:03d}"
    time_index = variant_index % 12
    pause_durations = (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 1.5, 2.5)
    response_windows = (2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 2.1, 2.5, 2.9, 3.4, 2.7)
    backchannel_gaps = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.28, 0.38, 0.48, 0.58)
    tail_gaps = (1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 2.1, 2.3, 2.5, 2.7, 2.9)
    if task == "pause_handling":
        timeline = [
            {"type": "speech", "text": lead},
            {"type": "silence", "duration_sec": pause_durations[time_index], "event": "pause"},
            {"type": "speech", "text": continuation},
        ]
    elif task == "smooth_turn_taking":
        timeline = [{"type": "speech", "text": f"{lead}、{continuation}"}]
    elif task == "backchannel":
        timeline = [
            {"type": "speech", "text": lead},
            {"type": "silence", "duration_sec": backchannel_gaps[time_index]},
            {"type": "speech", "text": continuation},
            {"type": "silence", "duration_sec": backchannel_gaps[(time_index + 4) % 12]},
            {"type": "speech", "text": correction},
        ]
    elif task == "user_interruption":
        timeline = [
            {"type": "speech", "text": f"{lead}。{continuation}"},
            {"type": "silence", "duration_sec": response_windows[time_index], "event": "response_window"},
            {"type": "interrupt", "text": correction, "event": "interruption"},
        ]
    else:
        overlap_text = OVERLAP_TEXTS[task][variant_index % len(OVERLAP_TEXTS[task])]
        event = {"type": "overlap_speech", "text": overlap_text, "event": "overlap"}
        if task == "background_speech":
            event["gain"] = 0.35
        timeline = [
            {"type": "speech", "text": f"{lead}。{continuation}"},
            {"type": "silence", "duration_sec": response_windows[time_index], "event": "response_window"},
            event,
            {"type": "silence", "duration_sec": tail_gaps[time_index]},
        ]
    return {
        "id": identifier,
        "task": task,
        "category": category,
        "risk_level": risk,
        "paired": task in {"user_backchannel", "talking_to_other", "background_speech"},
        "timeline": timeline,
        "expected_behavior": expected,
        "avoid_behavior": avoid,
        "scenario_set": "expanded_v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-per-task", type=int, default=50)
    args = parser.parse_args()
    source = read_jsonl(args.source)
    counts = Counter(str(row["task"]) for row in source)
    tasks = sorted(counts)
    if not tasks or any(count > args.target_per_task for count in counts.values()):
        raise SystemExit("target-per-task must be at least every source task count.")
    rows = list(source)
    ids = {str(row["id"]) for row in rows}
    for task in tasks:
        for number in range(counts[task] + 1, args.target_per_task + 1):
            row = scenario(task, number, number - counts[task] - 1)
            if row["id"] in ids:
                raise SystemExit(f"Generated duplicate ID: {row['id']}")
            rows.append(row)
            ids.add(row["id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    final = Counter(str(row["task"]) for row in rows)
    print(f"[fdb-scenarios] wrote {len(rows)} cases: {dict(sorted(final.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
