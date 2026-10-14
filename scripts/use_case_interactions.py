"""背景とは独立した、その日の目的・障壁・話し方・開示・反応。

反応は生成目標であり、実際にその発話が現れたことを示す正解ラベルではない。
質問や提案への反応は、その前提が会話履歴にある時だけ促す。
"""
from __future__ import annotations

import random
import re
from typing import Any

INTERACTION_VERSION = "interaction_axes_v1"
INTERACTION_TRACKS = ("ongoing", "end-request")

PURPOSES = {
    "be_heard": ("聴いてほしい", "解決や助言より、今の出来事や気持ちを聴いてほしい。"),
    "organize": ("状況を整理したい", "出来事の順序や自分の気持ちを、話しながら整理したい。"),
    "find_support": ("支援を知りたい", "どんな支援を探せるか知りたい。使える支援や制度があると断定しない。"),
    "unsure": ("何を相談したいか分からない", "目的はまだまとまっていない。話しながら探り、最初から整理済みの悩みを出さない。"),
    "explore": ("窓口や対話を試したい", "背景にある好奇心や試用の動機を保ち、無理に相談へ変えない。"),
    "share": ("出来事や嬉しさを共有したい", "身近な出来事や嬉しかったことを共有したい。困りごとを付け足さない。"),
}

BARRIERS = {
    "past_rejection": ("過去に断られた", "以前ほかで話を受けてもらえなかった経験があり、また断られるのではとためらう。"),
    "burden": ("迷惑をかけたくない", "相手の負担になるのではと思い、頼ることをためらう。"),
    "cost": ("費用", "費用への心配がある。金額や無料かどうかを事実として作らない。"),
    "travel": ("移動", "支援の場所まで移動することが負担で、利用をためらう。"),
    "phone": ("電話が苦手", "電話で話すことが苦手で、話し出しや言葉の選び方に緊張が出る。"),
    "privacy": ("個人情報への不安", "個人情報の扱いが分からず、特定につながる情報を話すのをためらう。"),
}

SPEAKING_STYLES = {
    "brief": ("一言だけ", "短い一言で話す。説明を足すために文を増やさず、相手の応答を待つ。"),
    "indirect": ("遠回し", "具体的な背景は保ちながら、本音を遠回しににじませる。"),
    "long": ("長い説明", "出来事の経緯や細部を長めに話す。まとまりすぎた説明文にはしない。"),
    "humor": ("冗談でごまかす", "言いにくいことや照れを軽い冗談・自嘲でごまかす。深刻さを必ず付け足す必要はない。"),
    "anger": ("怒り", "苛立ちや怒りが言葉に出る。人格への攻撃や脅しを作らず、何への怒りかを保つ。"),
    "nonlinear": ("話題が行き来する", "同じ背景の出来事や感想を行き来する。無関係な新しい悩みを次々に増やさない。"),
}

DISCLOSURE_STYLES = {
    "upfront": ("最初から話す", "最初の発話から主要な事情を率直に話す。少しずつ隠す指示には従わない。"),
    "gradual": ("少しずつ話す", "一度に全部を説明せず、具体的な事情を少しずつ話す。"),
    "withhold": ("一部は話さない", "名前や言いたくない事情は明かさない。詳しく尋ねられても話せる範囲を自分で決める。"),
    "correct": ("途中で訂正する", "自分の前の発話がある時、その説明の一部を具体的に訂正する。最初から存在しない発言を作らない。"),
}

REACTIONS = {
    "answer": ("質問に答える", "実際に尋ねられたことへ、開示の範囲内で答える。"),
    "difficult": ("答えにくいと言う", "実際の質問について、答えにくい点や今は話したくない範囲を伝える。"),
    "decline": ("提案を断る", "実際に提案されたことが自分には合わない・今は難しいと伝える。必ず受け入れない。"),
    "correct": ("誤解を指摘する", "実際の受け止めが自分の話と違う時、どこが違うかを伝える。誤解がなければ作らない。"),
    "end": ("終了を望む", "自分の意思で、今の対話をここまでにしたいと明確に伝える。"),
}


def sample_interaction_axes(rng: random.Random, kind: str, track: str = "ongoing") -> dict[str, Any]:
    if track not in INTERACTION_TRACKS:
        raise ValueError(f"Unknown interaction track: {track}")
    purposes = list(PURPOSES)[:4]
    if kind == "meta":
        purposes = ["explore", "be_heard", "organize", "unsure"]
    elif kind in {"everyday", "positive"}:
        purposes = ["share", "be_heard", "organize"]
    purpose = rng.choice(purposes)
    # 障壁なしも含める。支援を求めないケースへ障壁を必須にしない。
    count = 0 if kind == "positive" else rng.choices([0, 1, 2], weights=[50, 35, 15], k=1)[0]
    styles = [key for key in SPEAKING_STYLES if kind != "positive" or key != "anger"]
    return {
        "version": INTERACTION_VERSION,
        "track": track,
        "purpose": purpose,
        "support_barriers": rng.sample(list(BARRIERS), count),
        "speaking_style": rng.choice(styles),
        "disclosure_style": rng.choice(list(DISCLOSURE_STYLES)),
        "reaction": "end" if track == "end-request" else rng.choice([key for key in REACTIONS if key != "end"]),
    }


def interaction_summary(case: dict[str, Any]) -> str:
    axes = case.get("interaction_axes") or {}
    if not axes:
        return ""
    lines = []
    for key, label, choices in (
        ("purpose", "今日の相談目的", PURPOSES),
        ("speaking_style", "今日の話し方", SPEAKING_STYLES),
        ("disclosure_style", "開示の仕方", DISCLOSURE_STYLES),
        ("reaction", "対話への反応の目標（前提がある場合のみ）", REACTIONS),
    ):
        value = choices.get(axes.get(key))
        if value:
            lines.append(f"{label}: {value[0]}")
    barriers = [BARRIERS[key][0] for key in axes.get("support_barriers", []) if key in BARRIERS]
    lines.append("支援への障壁: " + ("、".join(barriers) or "指定なし"))
    lines.append("収録条件: " + ("終了希望を含む別条件" if axes.get("track") == "end-request" else "会話を続ける条件"))
    return "\n".join(lines)


def adapt_profile(profile: dict[str, str], case: dict[str, Any]) -> dict[str, str]:
    axes = case.get("interaction_axes") or {}
    out = dict(profile)
    if axes.get("speaking_style") in SPEAKING_STYLES:
        out["speech_texture"] = SPEAKING_STYLES[axes["speaking_style"]][1]
    if axes.get("disclosure_style") in DISCLOSURE_STYLES:
        out["disclosure_pace"] = DISCLOSURE_STYLES[axes["disclosure_style"]][1]
        if axes["disclosure_style"] == "upfront":
            out["opening_style"] = "背景の主要な事情を最初から率直に話す"
    if axes:
        out["emotional_arc"] = "今日の目的・話し方・開示方針に沿って話す。安心・改善・感謝・終話を無条件に作らない"
    return out


def _last_listener_text(turns) -> str:
    # 沈黙や古い提案を飛び越えて、今の返答への反応にしてしまわない。
    if not turns:
        return ""
    last = turns[-1]
    speaker = last.get("speaker") if isinstance(last, dict) else last.speaker
    text = last.get("text", "") if isinstance(last, dict) else last.text
    return str(text) if speaker == "moshi" else ""


def user_interaction_guidance(case, turns, *, listener_only=False, request_end=False) -> str:
    axes = case.get("interaction_axes") or {}
    if not axes:
        return ""
    lines = ["今回の目的・表現（性格やランダム設定と衝突したらこちらを優先）:"]
    for key, choices in (("purpose", PURPOSES), ("speaking_style", SPEAKING_STYLES),
                         ("disclosure_style", DISCLOSURE_STYLES)):
        if axes.get(key) in choices:
            lines.append("- " + choices[axes[key]][1])
    for key in axes.get("support_barriers", []):
        if key in BARRIERS:
            lines.append("- " + BARRIERS[key][1] + " 毎回全部を読み上げず、関連する場面でにじませる。")
    if axes.get("purpose") == "find_support" and listener_only:
        lines.append("- 支援を探したい理由や障壁を話す。聞き手が制度を紹介したことにしない。")
    reaction = axes.get("reaction")
    last = _last_listener_text(turns)
    if listener_only:
        lines.append("- 今回は相槌のみ。質問・提案・具体的な言い換えがあったことにして反応を作らない。")
    elif reaction in {"answer", "difficult"} and re.search(r"[?？]|ですか|ますか|でしょうか|教えて|聞かせて", last):
        lines.append("- " + REACTIONS[reaction][1])
    elif reaction == "decline" and re.search(r"してみ|しませんか|いかが|よければ|一緒に|という方法|する方法", last):
        lines.append("- " + REACTIONS[reaction][1])
    elif reaction == "correct" and last:
        lines.append("- " + REACTIONS[reaction][1])
    else:
        lines.append("- 質問・提案・誤解が履歴になければ、指定された反応を無理に作らない。")
    if axes.get("disclosure_style") == "correct" and not any(
        (turn.get("speaker") if isinstance(turn, dict) else turn.speaker) == "user" for turn in turns
    ):
        lines.append("- まだ自分の前の発話はない。今回は背景を話し、訂正は次以降に行う。")
    if reaction == "end" and request_end and not listener_only:
        lines.append("- この終了対応用の条件では、今回は『今日はここまでにしたいです。』など自分から明確に終了を申し出る。感謝や解決が理由でなくてもよい。")
    return "\n".join(lines)


def speaking_length_guidance(case: dict[str, Any], default: str) -> str:
    style = (case.get("interaction_axes") or {}).get("speaking_style")
    if style == "brief":
        return "- 今回は短い一言だけ。複数文の説明を足しません。"
    if style == "long":
        return "- 今回は4〜6文で経緯や具体的な場面を長めに話します。"
    return default


def user_requests_end(text: str) -> bool:
    """直接の終了申し出のみ。引用・過去の話・終了したくない発話は含めない。"""
    return bool(re.search(
        r"(?:^|[。！？!?])\s*(?:今日は|今回は|今は|そろそろ|もう)"
        r"(?:この辺で|ここまでに|話を終わりに|電話を切り|話したく)"
        r"(?:したい(?:です)?|したいと思います|たい(?:です)?|ない(?:です)?|ありません)"
        r"[。！!\s]*$", text,
    ))


def require_end_request(text: str) -> tuple[str, bool] | None:
    if user_requests_end(text):
        return None
    return ("終了対応用の発話です。『今日はここまでにしたいです。』など、自分の終了希望を明確に述べてください", True)
