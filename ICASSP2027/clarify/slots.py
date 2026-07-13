"""Slot extraction and value matching, language-parameterized.

MASSIVE and SLURP share the ``[slot_type : surface]`` annotation format
(MASSIVE localized SLURP), so one parser serves every corpus. Language-
specific material (WH words, normalization, templates) lives in
`clarify.lang.LanguagePack`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .lang import CRITICAL_SLOT_TYPES, INTENT_WHITELIST, get_pack
from .scenario import BaseItem

_ANNOT_RE = re.compile(r"\[\s*([a-z_]+)\s*:\s*([^\]]+?)\s*\]")


@dataclass
class ParsedSlot:
    slot_type: str
    surface: str
    pre_text: str
    post_text: str


def parse_annot_utt(annot_utt: str) -> list[tuple[str, str]]:
    """Return [(slot_type, surface), ...] from an annotated utterance."""
    return [(m.group(1), m.group(2)) for m in _ANNOT_RE.finditer(annot_utt)]


def strip_annotations(annot_utt: str) -> str:
    return _ANNOT_RE.sub(lambda m: m.group(2), annot_utt)


def split_around_slot(annot_utt: str, slot_type: str) -> ParsedSlot | None:
    """Split into pre/slot/post around the single occurrence of slot_type;
    None if absent or duplicated (ambiguous span)."""
    matches = list(_ANNOT_RE.finditer(annot_utt))
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
# language-delegating helpers (kept as functions for call-site brevity)
# ---------------------------------------------------------------------------

def normalize_slot_value(value: str, lang: str = "ja") -> str:
    return get_pack(lang).normalize(value)


def slot_value_in_text(slot_value: str, text: str, lang: str = "ja") -> bool:
    return get_pack(lang).value_in_text(slot_value, text)


def build_repair_text(slot_type: str, slot_value: str,
                      lang: str = "ja") -> str:
    return get_pack(lang).repair_template(slot_type, slot_value)


# ---------------------------------------------------------------------------
# base-item selection (corpus rows -> BaseItem)
# ---------------------------------------------------------------------------

def select_base_items(
    rows: Iterable[dict[str, Any]],
    language: str = "ja",
    max_items: int = 60,
    min_utt_chars: int | None = None,
    max_utt_chars: int | None = None,
    min_slot_chars: int | None = None,
    intents: tuple[str, ...] = INTENT_WHITELIST,
    id_key: str = "id",
) -> list[BaseItem]:
    """Select acoustic-arm base items from corpus rows.

    Rows need: id, intent (str), annot_utt, utt; optional audio_file (real
    recording path -> carried into meta, audio_source="real").
    One critical slot per utterance, appearing exactly once, long enough
    to carry a maskable span. Balanced across slot types by round-robin.

    Length bounds default per language (characters for ja, words for en).
    """
    if language == "ja":
        lo, hi, slot_min = (min_utt_chars or 8, max_utt_chars or 40,
                            min_slot_chars or 2)

        def length(text: str) -> int:
            return len(text)
    else:
        lo, hi, slot_min = (min_utt_chars or 4, max_utt_chars or 14,
                            min_slot_chars or 3)

        def length(text: str) -> int:
            return len(text.split())

    by_slot_type: dict[str, list[BaseItem]] = {
        t: [] for t in CRITICAL_SLOT_TYPES
    }
    for row in rows:
        intent = row.get("intent")
        if not isinstance(intent, str):
            continue
        if intents and intent not in intents:
            continue
        annot = row.get("annot_utt") or ""
        utt = row.get("utt") or strip_annotations(annot)
        if not (lo <= length(utt) <= hi):
            continue
        slots = parse_annot_utt(annot)
        usable = [(t, v) for t, v in slots if t in CRITICAL_SLOT_TYPES
                  and len(v) >= slot_min]
        if len(usable) != 1:
            continue
        slot_type, _value = usable[0]
        parsed = split_around_slot(annot, slot_type)
        if parsed is None:
            continue
        if strip_annotations(annot) != utt:
            continue
        meta: dict[str, Any] = {
            "corpus_id": row.get(id_key), "scenario": row.get("scenario"),
        }
        source = row.get("source") or "massive"
        if row.get("audio_file"):
            meta["audio_file"] = str(row["audio_file"])
        item = BaseItem(
            base_id=f"{source}_{row.get(id_key)}",
            arm="acoustic",
            intent=intent,
            slot_type=slot_type,
            slot_value=parsed.surface,
            utterance_text=utt,
            pre_text=parsed.pre_text,
            slot_text=parsed.surface,
            post_text=parsed.post_text,
            repair_text=build_repair_text(slot_type, parsed.surface,
                                          language),
            source=source,
            language=language,
            audio_source="real" if row.get("audio_file") else "tts",
            meta=meta,
        )
        try:
            item.validate()
        except ValueError:
            continue
        by_slot_type[slot_type].append(item)

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


def build_underspecified_items(lang: str = "ja") -> list[BaseItem]:
    """Semantic-ambiguity arm from the language pack's templates."""
    pack = get_pack(lang)
    items = []
    for i, tpl in enumerate(pack.underspecified_templates):
        item = BaseItem(
            base_id=f"underspec_{lang}_{i:03d}",
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
            language=lang,
        )
        item.validate()
        items.append(item)
    return items
