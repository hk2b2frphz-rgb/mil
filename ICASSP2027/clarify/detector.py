"""Clarification / confirmation detection from Moshi's own text stream.

Moshi emits time-aligned text tokens for its own speech, so no output ASR
is needed: turn segmentation and classification operate directly on the
token stream that `response_recorder.run_trial` collects.

Two consumers:
  * ONLINE  - the closed-loop driver uses `classify_turn` on the text
              accumulated since the user's utterance ended, to decide
              whether to inject the scripted repair turn.
  * OFFLINE - metrics recompute the same labels post-hoc; an LLM judge
              (see judge_pack.py) provides an independent second opinion,
              and disagreement between rule and judge is itself reported.

Labels:
  clarify_targeted - asks specifically about the corrupted/missing slot
                     (matching WH word or explicit repeat-request).
  clarify_generic  - signals non-understanding without targeting the slot.
  confirm          - proceeds with the task (echoes a value / accepts).
  other            - chit-chat, silence, or unrelated content.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .slots import SLOT_WH_WORDS, slot_value_in_text

# ---------------------------------------------------------------------------
# lexicons
# ---------------------------------------------------------------------------

# Explicit repeat/repair requests (strong clarification evidence regardless
# of question mark).
REPEAT_REQUEST_PATTERNS: tuple[str, ...] = (
    "もう一度", "もういちど", "もう一回", "もういっかい",
    "聞き取れ", "ききとれ", "聞こえ", "きこえ",
    "おっしゃいました", "言いました",
    "繰り返し", "くりかえし",
    "ゆっくり",
)

# Generic non-understanding markers.
NONUNDERSTANDING_PATTERNS: tuple[str, ...] = (
    "わかりません", "分かりません",
    "うまく聞", "よく聞",
    "すみません、何", "すみません、なん",
    "えっと、何", "え、何",
)

# Confirmation / task-progress markers. Deliberately excludes bare ですね
# (ubiquitous in chit-chat); a value echo is the primary confirm evidence.
CONFIRM_PATTERNS: tuple[str, ...] = (
    "承知", "かしこまり", "わかりました", "分かりました",
    "了解", "設定し", "登録し", "予約し", "お伝えし", "メモし",
    "調べて", "かけますね", "再生し", "注文し", "手配し",
)

_QUESTION_MARKS = ("?", "？", "ですか", "でしょうか", "ますか")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _contains_any(text: str, patterns: tuple[str, ...]) -> str | None:
    for pat in patterns:
        if pat in text:
            return pat
    return None


def _is_question(text: str) -> bool:
    stripped = text.rstrip("。 　\n")
    return any(mark in stripped[-14:] for mark in _QUESTION_MARKS) or \
        stripped.endswith(("か", "の？", "?"))


@dataclass
class TurnLabel:
    label: str                  # clarify_targeted|clarify_generic|confirm|other
    is_clarification: bool
    matched_pattern: str | None
    slot_value_echoed: bool     # gold slot value appears in the turn text
    is_question: bool


def classify_turn(
    text: str,
    slot_type: str,
    slot_value: str | None = None,
) -> TurnLabel:
    """Classify one model turn against the case's critical slot."""
    text_n = _normalize(text)
    question = _is_question(text_n)
    echoed = bool(slot_value) and slot_value_in_text(slot_value, text_n)

    wh_words = SLOT_WH_WORDS.get(slot_type, ())
    wh_hit = _contains_any(text_n, wh_words)
    repeat_hit = _contains_any(text_n, REPEAT_REQUEST_PATTERNS)
    nonund_hit = _contains_any(text_n, NONUNDERSTANDING_PATTERNS)
    confirm_hit = _contains_any(text_n, CONFIRM_PATTERNS)

    # Order matters. A turn like 「すみません、何時ですか？」 must be a
    # targeted clarification even though it contains no repeat keyword; a
    # turn like 「15時ですね、登録しますね」 must be a confirmation even
    # though ですね ends in ね.
    if (wh_hit and question) or (repeat_hit and wh_hit):
        return TurnLabel("clarify_targeted", True, wh_hit or repeat_hit,
                         echoed, question)
    if repeat_hit or nonund_hit:
        return TurnLabel("clarify_generic", True, repeat_hit or nonund_hit,
                         echoed, question)
    if echoed and question and not confirm_hit:
        # Echo question: 「15時、ですか？」 - treated as targeted
        # clarification (seeks verification of the heard value).
        return TurnLabel("clarify_targeted", True, "echo_question",
                         echoed, question)
    if confirm_hit or echoed:
        return TurnLabel("confirm", False, confirm_hit, echoed, question)
    if wh_hit and question:
        return TurnLabel("clarify_targeted", True, wh_hit, echoed, question)
    return TurnLabel("other", False, None, echoed, question)


# ---------------------------------------------------------------------------
# turn segmentation over the streamed text events
# ---------------------------------------------------------------------------

@dataclass
class ModelTurn:
    start_sec: float
    end_sec: float
    text: str


def segment_turns(
    text_events: list[dict],
    gap_sec: float = 1.2,
    min_chars: int = 2,
) -> list[ModelTurn]:
    """Group Moshi text events into turns split at silences >= gap_sec.

    `text_events` follow response_recorder's schema:
    {"step": int, "time_sec": float, "piece": str}.
    """
    turns: list[ModelTurn] = []
    current: list[dict] = []
    for event in text_events:
        piece = event.get("piece", "")
        if not piece.strip():
            continue
        if current and event["time_sec"] - current[-1]["time_sec"] >= gap_sec:
            turn = _finalize_turn(current, min_chars)
            if turn:
                turns.append(turn)
            current = []
        current.append(event)
    turn = _finalize_turn(current, min_chars)
    if turn:
        turns.append(turn)
    return turns


def _finalize_turn(events: list[dict], min_chars: int) -> ModelTurn | None:
    if not events:
        return None
    text = "".join(e["piece"] for e in events).strip()
    if len(text) < min_chars:
        return None
    return ModelTurn(
        start_sec=float(events[0]["time_sec"]),
        end_sec=float(events[-1]["time_sec"]),
        text=text,
    )


def first_turn_after(
    turns: list[ModelTurn], after_sec: float, overlap_slack_sec: float = 0.5
) -> ModelTurn | None:
    """First model turn whose *end* is after `after_sec`.

    Full-duplex models may begin responding slightly before the user
    finishes; a turn is still "the response" if it extends past the
    utterance end minus the slack.
    """
    for turn in turns:
        if turn.end_sec >= after_sec - overlap_slack_sec:
            return turn
    return None
