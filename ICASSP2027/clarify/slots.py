"""MASSIVE ja-JP slot extraction and Japanese slot-value normalization.

MASSIVE (Amazon Science, CC BY 4.0) annotates each utterance with
``annot_utt`` where slots appear as ``[slot_type : surface]``. We select
voice-assistant intents whose task outcome hinges on one critical slot,
split the utterance into pre/slot/post text around that slot, and attach a
scripted repair utterance used by the closed-loop user simulator when the
model asks for clarification.

Loading the HF dataset needs `datasets` (cluster-side, proxied); everything
else in this module is pure-python and unit-tested locally.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from .scenario import BaseItem

# ---------------------------------------------------------------------------
# slot/intent selection
# ---------------------------------------------------------------------------

# Critical-slot whitelist: slot types whose value the assistant must capture
# verbatim for the task to succeed, with the WH-word set a *targeted*
# clarification about that slot would use.
SLOT_WH_WORDS: dict[str, tuple[str, ...]] = {
    "time": ("何時", "いつ", "なんじ"),
    "date": ("いつ", "何日", "何曜日", "なんにち"),
    "person": ("誰", "どなた", "だれ", "お名前"),
    "place_name": ("どこ", "どちら", "場所"),
    "event_name": ("何の", "なんの", "どんな", "イベント"),
    "business_name": ("どこ", "どちら", "何という", "なんという", "お店"),
    "artist_name": ("誰", "どなた", "だれ", "アーティスト"),
    "food_type": ("何", "なに", "どんな", "料理"),
    "timeofday": ("何時", "いつ", "なんじ"),
}

# Intents where the critical slot drives an action with a wrong-value cost.
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

_ANNOT_RE = re.compile(r"\[\s*([a-z_]+)\s*:\s*([^\]]+?)\s*\]")


@dataclass
class ParsedSlot:
    slot_type: str
    surface: str
    pre_text: str
    post_text: str


def parse_annot_utt(annot_utt: str) -> list[tuple[str, str]]:
    """Return [(slot_type, surface), ...] from a MASSIVE annot_utt string."""
    return [(m.group(1), m.group(2)) for m in _ANNOT_RE.finditer(annot_utt)]


def strip_annotations(annot_utt: str) -> str:
    """annot_utt -> plain utterance text."""
    return _ANNOT_RE.sub(lambda m: m.group(2), annot_utt)


def split_around_slot(annot_utt: str, slot_type: str) -> ParsedSlot | None:
    """Split the plain utterance into pre/slot/post around the FIRST
    occurrence of slot_type. Returns None if the slot is absent or appears
    more than once (ambiguous span)."""
    matches = [m for m in _ANNOT_RE.finditer(annot_utt)]
    targets = [m for m in matches if m.group(1) == slot_type]
    if len(targets) != 1:
        return None
    target = targets[0]
    pre_raw = annot_utt[: target.start()]
    post_raw = annot_utt[target.end():]
    return ParsedSlot(
        slot_type=slot_type,
        surface=target.group(2),
        pre_text=_ANNOT_RE.sub(lambda m: m.group(2), pre_raw),
        post_text=_ANNOT_RE.sub(lambda m: m.group(2), post_raw),
    )


# ---------------------------------------------------------------------------
# Japanese normalization for slot matching
# ---------------------------------------------------------------------------

_KANJI_DIGITS = {"〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
                 "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def _kanji_number_to_int(text: str) -> str:
    """Convert simple kanji numerals (十/百 compositions) to arabic digits."""

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


_HOUR_RE = re.compile(r"(午前|午後|夜|朝|夕方)?\s*(\d{1,2})\s*時(半)?(\d{1,2}分)?")


def _normalize_time(text: str) -> str:
    """Map Japanese clock expressions to a canonical HH:MM token."""

    def conv(match: re.Match[str]) -> str:
        period, hour_s, han, minute_s = match.groups()
        hour = int(hour_s)
        minute = 30 if han else 0
        if minute_s:
            minute = int(minute_s.rstrip("分"))
        if period in ("午後", "夜", "夕方") and hour < 12:
            hour += 12
        return f"{hour:02d}:{minute:02d}"

    return _HOUR_RE.sub(conv, text)


def normalize_slot_value(value: str) -> str:
    """Canonical form for slot comparison: NFKC, lowercase, kanji digits to
    arabic, clock expressions to HH:MM, punctuation/whitespace stripped."""
    text = unicodedata.normalize("NFKC", value).lower()
    text = _kanji_number_to_int(text)
    text = _normalize_time(text)
    text = re.sub(r"[\s、。,．.!?！?・「」『』]", "", text)
    return text


def slot_value_in_text(slot_value: str, text: str) -> bool:
    """Whether the normalized slot value appears in the normalized text.

    Time slots additionally match if the canonical HH:MM token appears, so
    「15時」 in the gold matches 「午後3時」 in the model's confirmation.
    """
    norm_value = normalize_slot_value(slot_value)
    norm_text = normalize_slot_value(text)
    if not norm_value:
        return False
    return norm_value in norm_text


# ---------------------------------------------------------------------------
# repair-utterance construction
# ---------------------------------------------------------------------------

def build_repair_text(slot_type: str, slot_value: str) -> str:
    """The scripted user turn injected when the model asks for clarification.

    Restates only the critical information, clearly, as a cooperative
    speaker would after being asked to repeat.
    """
    return f"{slot_value}です。"


# ---------------------------------------------------------------------------
# MASSIVE loading (cluster-side)
# ---------------------------------------------------------------------------

def load_massive_ja(split: str = "test") -> list[dict[str, Any]]:
    """Load MASSIVE ja-JP rows (requires `datasets`; run on cluster)."""
    from datasets import load_dataset  # lazy import

    ds = load_dataset("AmazonScience/massive", "ja-JP", split=split)
    return list(ds)


def select_base_items(
    rows: Iterable[dict[str, Any]],
    max_items: int = 60,
    min_utt_chars: int = 8,
    max_utt_chars: int = 40,
    min_slot_chars: int = 2,
    intents: tuple[str, ...] = INTENT_WHITELIST,
    id_key: str = "id",
) -> list[BaseItem]:
    """Select acoustic-arm base items from MASSIVE rows.

    One critical slot per utterance, appearing exactly once, long enough to
    carry a maskable span, inside a carrier utterance of reasonable length.
    Balanced across slot types by round-robin so no type dominates.
    """
    by_slot_type: dict[str, list[BaseItem]] = {t: [] for t in SLOT_WH_WORDS}
    for row in rows:
        intent = row.get("intent")
        if isinstance(intent, int):
            # HF datasets encodes intent as ClassLabel int; caller should
            # pass decoded rows, but tolerate raw ints by skipping.
            continue
        if intents and intent not in intents:
            continue
        annot = row.get("annot_utt") or ""
        utt = row.get("utt") or strip_annotations(annot)
        if not (min_utt_chars <= len(utt) <= max_utt_chars):
            continue
        slots = parse_annot_utt(annot)
        usable = [(t, v) for t, v in slots if t in SLOT_WH_WORDS
                  and len(v) >= min_slot_chars]
        if len(usable) != 1:
            continue  # exactly one critical slot keeps the span unambiguous
        slot_type, _value = usable[0]
        parsed = split_around_slot(annot, slot_type)
        if parsed is None:
            continue
        if strip_annotations(annot) != utt:
            # annot/utt disagree; skip rather than guess the span.
            continue
        base_id = f"massive_{row.get(id_key)}"
        item = BaseItem(
            base_id=base_id,
            arm="acoustic",
            intent=intent,
            slot_type=slot_type,
            slot_value=parsed.surface,
            utterance_text=utt,
            pre_text=parsed.pre_text,
            slot_text=parsed.surface,
            post_text=parsed.post_text,
            repair_text=build_repair_text(slot_type, parsed.surface),
            source="massive_ja",
            meta={"massive_id": row.get(id_key), "scenario": row.get("scenario")},
        )
        try:
            item.validate()
        except ValueError:
            continue
        by_slot_type[slot_type].append(item)

    # Round-robin across slot types for balance.
    selected: list[BaseItem] = []
    pools = [pool for pool in by_slot_type.values() if pool]
    idx = 0
    while len(selected) < max_items and pools:
        pool = pools[idx % len(pools)]
        if pool:
            selected.append(pool.pop(0))
        pools = [p for p in pools if p]
        idx += 1
    return selected


# ---------------------------------------------------------------------------
# underspecified arm (semantic ambiguity with clean audio)
# ---------------------------------------------------------------------------

UNDERSPECIFIED_TEMPLATES: list[dict[str, str]] = [
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


def build_underspecified_items() -> list[BaseItem]:
    items = []
    for i, tpl in enumerate(UNDERSPECIFIED_TEMPLATES):
        item = BaseItem(
            base_id=f"underspec_{i:03d}",
            arm="underspecified",
            intent=tpl["intent"],
            slot_type=tpl["slot_type"],
            slot_value=tpl["gold"],
            utterance_text=tpl["utterance"],
            pre_text="",
            slot_text="",
            post_text="",
            repair_text=tpl["repair"],
            source="template",
        )
        item.validate()
        items.append(item)
    return items
