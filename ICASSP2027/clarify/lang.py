"""Language packs: every language-dependent piece of the pipeline.

The benchmark, detector, cascade, and training-data generator are
language-parameterized through a single `LanguagePack` so adding a locale
means adding one pack here. English (primary, matches Kyutai Moshi) and
Japanese (generality axis, matches llm-jp-moshi / J-Moshi) ship built-in.

Slot TYPE names are language-independent (MASSIVE and SLURP share them);
only surface material (WH words, lexicons, templates, normalization)
lives in the pack.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

# Slot types whose value the assistant must capture verbatim (shared
# across languages; MASSIVE inherited them from SLURP).
CRITICAL_SLOT_TYPES: tuple[str, ...] = (
    "time", "date", "person", "place_name", "event_name",
    "business_name", "artist_name", "food_type", "timeofday",
)

INTENT_WHITELIST: tuple[str, ...] = (
    "alarm_set",
    "calendar_set",
    "calendar_query",
    "play_music",
    "takeaway_order",
    "transport_query",
    "recommendation_events",
    "recommendation_locations",
    "qa_factoid",
)


@dataclass(frozen=True)
class LanguagePack:
    code: str                       # "en" | "ja"
    massive_locale: str             # HF MASSIVE config name
    tts_language: str               # Qwen3-TTS language hint
    tts_default_speaker: str
    asr_language: str               # faster-whisper language code

    slot_wh_words: dict[str, tuple[str, ...]]
    repeat_patterns: tuple[str, ...]
    nonunderstanding_patterns: tuple[str, ...]
    confirm_patterns: tuple[str, ...]
    question_suffixes: tuple[str, ...]

    # templates
    repair_template: Callable[[str, str], str]          # (slot_type, value)
    ask_by_slot: dict[str, str]                         # targeted ask turn
    ask_fallback: str
    confirm_by_intent: dict[str, str]                   # {value} placeholder
    confirm_fallback: str

    # normalization / matching
    normalize: Callable[[str], str]
    value_in_text: Callable[[str, str], bool]

    underspecified_templates: list[dict[str, str]] = field(
        default_factory=list
    )


# ===========================================================================
# Japanese
# ===========================================================================

_KANJI_DIGITS = {"〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
                 "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def _ja_kanji_number_to_int(text: str) -> str:
    def conv(match: re.Match[str]) -> str:
        s = match.group(0)
        total, current = 0, 0
        for ch in s:
            if ch in _KANJI_DIGITS:
                current = current * 10 + int(_KANJI_DIGITS[ch])
            elif ch == "十":
                total += (current or 1) * 10
                current = 0
            elif ch == "百":
                total += (current or 1) * 100
                current = 0
        total += current
        return str(total)

    return re.sub(r"[〇一二三四五六七八九十百]+", conv, text)


_JA_HOUR_RE = re.compile(r"(午前|午後|夜|朝|夕方)?\s*(\d{1,2})\s*時(半)?(\d{1,2}分)?")


def _ja_normalize_time(text: str) -> str:
    def conv(match: re.Match[str]) -> str:
        period, hour_s, han, minute_s = match.groups()
        hour = int(hour_s)
        minute = 30 if han else 0
        if minute_s:
            minute = int(minute_s.rstrip("分"))
        if period in ("午後", "夜", "夕方") and hour < 12:
            hour += 12
        return f"{hour:02d}:{minute:02d}"

    return _JA_HOUR_RE.sub(conv, text)


def ja_normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).lower()
    text = _ja_kanji_number_to_int(text)
    text = _ja_normalize_time(text)
    text = re.sub(r"[\s、。,．.!?！?・「」『』]", "", text)
    return text


def ja_value_in_text(slot_value: str, text: str) -> bool:
    norm_value = ja_normalize(slot_value)
    if not norm_value:
        return False
    return norm_value in ja_normalize(text)


JA_UNDERSPECIFIED: list[dict[str, str]] = [
    {"intent": "calendar_set", "slot_type": "time",
     "utterance": "明日、会議の予定をカレンダーに入れておいて。",
     "repair": "午後3時からです。", "gold": "午後3時"},
    {"intent": "alarm_set", "slot_type": "time",
     "utterance": "明日の朝、アラームをかけておいてほしいんだけど。",
     "repair": "7時にお願いします。", "gold": "7時"},
    {"intent": "calendar_set", "slot_type": "date",
     "utterance": "歯医者の予約をカレンダーに登録しておいて。",
     "repair": "来週の金曜日です。", "gold": "来週の金曜日"},
    {"intent": "play_music", "slot_type": "artist_name",
     "utterance": "いつもの好きな曲をかけてくれる？",
     "repair": "米津玄師の曲です。", "gold": "米津玄師"},
    {"intent": "transport_query", "slot_type": "place_name",
     "utterance": "そこまでの終電の時間を調べておいて。",
     "repair": "横浜駅までです。", "gold": "横浜駅"},
    {"intent": "takeaway_order", "slot_type": "food_type",
     "utterance": "夕飯の出前を頼んでおいてもらえる？",
     "repair": "ラーメンをお願いします。", "gold": "ラーメン"},
    {"intent": "calendar_set", "slot_type": "person",
     "utterance": "打ち合わせの予定を入れて、相手にも連絡しておいて。",
     "repair": "田中さんです。", "gold": "田中さん"},
    {"intent": "recommendation_events", "slot_type": "date",
     "utterance": "近くで何かいいイベントがないか探しておいて。",
     "repair": "今週の土曜日です。", "gold": "今週の土曜日"},
]

JA_PACK = LanguagePack(
    code="ja",
    massive_locale="ja-JP",
    tts_language="ja",
    tts_default_speaker="Ono_Anna",
    asr_language="ja",
    slot_wh_words={
        "time": ("何時", "いつ", "なんじ"),
        "date": ("いつ", "何日", "何曜日", "なんにち"),
        "person": ("誰", "どなた", "だれ", "お名前"),
        "place_name": ("どこ", "どちら", "場所"),
        "event_name": ("何の", "なんの", "どんな", "イベント"),
        "business_name": ("どこ", "どちら", "何という", "なんという", "お店"),
        "artist_name": ("誰", "どなた", "だれ", "アーティスト"),
        "food_type": ("何", "なに", "どんな", "料理"),
        "timeofday": ("何時", "いつ", "なんじ"),
    },
    repeat_patterns=(
        "もう一度", "もういちど", "もう一回", "もういっかい",
        "聞き取れ", "ききとれ", "聞こえ", "きこえ",
        "おっしゃいました", "言いました",
        "繰り返し", "くりかえし",
        "ゆっくり",
    ),
    nonunderstanding_patterns=(
        "わかりません", "分かりません",
        "うまく聞", "よく聞",
        "すみません、何", "すみません、なん",
        "えっと、何", "え、何",
    ),
    confirm_patterns=(
        "承知", "かしこまり", "わかりました", "分かりました",
        "了解", "設定し", "登録し", "予約し", "お伝えし", "メモし",
        "調べて", "かけますね", "再生し", "注文し", "手配し",
    ),
    question_suffixes=("?", "？", "ですか", "でしょうか", "ますか", "か"),
    repair_template=lambda slot_type, value: f"{value}です。",
    ask_by_slot={
        "time": "すみません、少し聞き取れませんでした。何時でしたか？",
        "date": "すみません、日にちがよく聞こえませんでした。いつでしたか？",
        "person": "すみません、お名前が聞き取れませんでした。どなたでしたか？",
        "place_name": "すみません、場所がよく聞こえませんでした。どちらでしたか？",
        "event_name": "すみません、聞き取れませんでした。何のご予定でしたか？",
        "business_name": "すみません、お店の名前が聞き取れませんでした。どちらでしたか？",
        "artist_name": "すみません、聞き取れませんでした。どなたの曲でしたか？",
        "food_type": "すみません、聞き取れませんでした。何のご注文でしたか？",
        "timeofday": "すみません、聞き取れませんでした。何時ごろでしたか？",
    },
    ask_fallback="すみません、もう一度お願いできますか？",
    confirm_by_intent={
        "alarm_set": "{value}ですね。アラームを設定しておきますね。",
        "calendar_set": "{value}ですね。カレンダーに登録しておきますね。",
        "calendar_query": "{value}のご予定ですね。確認しますね。",
        "play_music": "{value}ですね。かけますね。",
        "takeaway_order": "{value}ですね。注文しておきますね。",
        "transport_query": "{value}ですね。調べてみますね。",
        "recommendation_events": "{value}ですね。探してみますね。",
        "recommendation_locations": "{value}ですね。探してみますね。",
        "qa_factoid": "{value}ですね。調べてみますね。",
    },
    confirm_fallback="{value}ですね。承知しました。",
    normalize=ja_normalize,
    value_in_text=ja_value_in_text,
    underspecified_templates=JA_UNDERSPECIFIED,
)


# ===========================================================================
# English
# ===========================================================================

_EN_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50",
}

_EN_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE
)
_EN_OCLOCK_RE = re.compile(r"\b(\d{1,2})\s*o'?\s?clock\b", re.IGNORECASE)


def _en_words_to_digits(text: str) -> str:
    def conv(match: re.Match[str]) -> str:
        return _EN_NUMBER_WORDS[match.group(0)]

    pattern = r"\b(" + "|".join(_EN_NUMBER_WORDS) + r")\b"
    return re.sub(pattern, conv, text)


def _en_normalize_time(text: str) -> str:
    def conv(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3).replace(".", "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    text = _EN_TIME_RE.sub(conv, text)
    text = _EN_OCLOCK_RE.sub(lambda m: f"{int(m.group(1)):02d}:00", text)
    return text


def en_normalize(value: str) -> str:
    """Lowercase token sequence with number words as digits and clock
    times canonicalized to 24h HH:MM."""
    text = unicodedata.normalize("NFKC", value).lower()
    text = _en_words_to_digits(text)
    text = _en_normalize_time(text)
    tokens = re.findall(r"[a-z0-9:]+", text)
    return " ".join(tokens)


def en_value_in_text(slot_value: str, text: str) -> bool:
    """Token-boundary containment of the normalized value.

    Space padding prevents 'ten' matching inside 'tension'; a bare-hour
    fallback lets gold '5 pm' (-> 17:00) match a reply that says '17:00'
    or 'five pm' but not plain '5'.
    """
    norm_value = en_normalize(slot_value)
    if not norm_value:
        return False
    norm_text = f" {en_normalize(text)} "
    return f" {norm_value} " in norm_text


EN_UNDERSPECIFIED: list[dict[str, str]] = [
    {"intent": "calendar_set", "slot_type": "time",
     "utterance": "Can you put the meeting on my calendar for tomorrow?",
     "repair": "It's at 3 pm.", "gold": "3 pm"},
    {"intent": "alarm_set", "slot_type": "time",
     "utterance": "Set an alarm for me tomorrow morning, please.",
     "repair": "Seven o'clock, please.", "gold": "seven o'clock"},
    {"intent": "calendar_set", "slot_type": "date",
     "utterance": "Add my dentist appointment to the calendar.",
     "repair": "It's next Friday.", "gold": "next friday"},
    {"intent": "play_music", "slot_type": "artist_name",
     "utterance": "Play some of my favorite songs, will you?",
     "repair": "Songs by Taylor Swift.", "gold": "taylor swift"},
    {"intent": "transport_query", "slot_type": "place_name",
     "utterance": "Look up the last train to get there tonight.",
     "repair": "To Central Station.", "gold": "central station"},
    {"intent": "takeaway_order", "slot_type": "food_type",
     "utterance": "Can you order dinner for tonight?",
     "repair": "A large pizza, please.", "gold": "pizza"},
    {"intent": "calendar_set", "slot_type": "person",
     "utterance": "Schedule the meeting and let the other person know.",
     "repair": "It's with Sarah.", "gold": "sarah"},
    {"intent": "recommendation_events", "slot_type": "date",
     "utterance": "Find me something fun happening nearby.",
     "repair": "This Saturday.", "gold": "saturday"},
]

EN_PACK = LanguagePack(
    code="en",
    massive_locale="en-US",
    tts_language="en",
    tts_default_speaker="Ryan",
    asr_language="en",
    slot_wh_words={
        "time": ("what time", "when"),
        "date": ("what day", "what date", "when"),
        "person": ("who", "whose", "what name", "which person"),
        "place_name": ("where", "which place", "what place"),
        "event_name": ("what event", "which event", "what appointment"),
        "business_name": ("which place", "what's the name", "what is the name",
                          "which store", "which restaurant", "where"),
        "artist_name": ("which artist", "who", "whose"),
        "food_type": ("what food", "what would you like", "what did you want",
                      "what kind"),
        "timeofday": ("what time", "when"),
    },
    repeat_patterns=(
        "say that again", "say it again", "could you repeat",
        "can you repeat", "repeat that", "one more time",
        "didn't catch", "did not catch", "couldn't catch",
        "didn't hear", "did not hear", "couldn't hear",
        "come again", "pardon", "more slowly",
        "didn't get that", "did not get that", "missed that",
    ),
    nonunderstanding_patterns=(
        "i didn't understand", "i did not understand",
        "not sure i heard", "not sure what you said",
        "what was that", "sorry, what", "i'm not sure i caught",
        "hard to hear", "couldn't make out", "didn't quite",
    ),
    confirm_patterns=(
        "got it", "sure thing", "no problem", "will do", "all right",
        "alright", "i'll set", "setting", "i've set", "i have set",
        "scheduled", "i'll add", "adding", "i'll put", "i'll play",
        "playing", "i'll order", "ordering", "i'll check", "i'll look",
        "looking up", "noted", "done,", "reminder set", "alarm set",
        "on it",
    ),
    question_suffixes=("?",),
    repair_template=lambda slot_type, value: f"It's {value}.",
    ask_by_slot={
        "time": "Sorry, I didn't catch that. What time was it?",
        "date": "Sorry, I didn't catch the day. When was it?",
        "person": "Sorry, I didn't catch the name. Who was it?",
        "place_name": "Sorry, I didn't catch the place. Where was it?",
        "event_name": "Sorry, I didn't catch that. What event was it?",
        "business_name": "Sorry, I didn't catch the name. Which place was it?",
        "artist_name": "Sorry, I didn't catch that. Which artist was it?",
        "food_type": "Sorry, I didn't catch that. What would you like to order?",
        "timeofday": "Sorry, I didn't catch that. What time was it?",
    },
    ask_fallback="Sorry, could you say that again?",
    confirm_by_intent={
        "alarm_set": "{value}, got it. I'll set the alarm.",
        "calendar_set": "{value}, got it. I'll put it on your calendar.",
        "calendar_query": "{value}, let me check your calendar.",
        "play_music": "{value}, got it. I'll play that now.",
        "takeaway_order": "{value}, got it. I'll place the order.",
        "transport_query": "{value}, let me look that up.",
        "recommendation_events": "{value}, let me find something.",
        "recommendation_locations": "{value}, let me find something.",
        "qa_factoid": "{value}, let me look that up.",
    },
    confirm_fallback="{value}, got it.",
    normalize=en_normalize,
    value_in_text=en_value_in_text,
    underspecified_templates=EN_UNDERSPECIFIED,
)


# ===========================================================================
# registry
# ===========================================================================

_PACKS: dict[str, LanguagePack] = {"ja": JA_PACK, "en": EN_PACK}


def get_pack(code: str) -> LanguagePack:
    try:
        return _PACKS[code]
    except KeyError:
        raise KeyError(
            f"unknown language {code!r}; available: {sorted(_PACKS)}"
        ) from None
