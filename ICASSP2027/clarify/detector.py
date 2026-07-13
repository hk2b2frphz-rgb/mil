"""Clarification / confirmation detection from Moshi's text stream.

Moshi-family models emit time-aligned text tokens for their own speech, so
turn segmentation and classification operate directly on the token stream
`response_recorder.run_trial` collects - no output ASR.

Language-parameterized: lexicons and normalization come from
`clarify.lang.LanguagePack` ("en" primary, "ja" generality axis).

Two consumers:
  * ONLINE  - the closed-loop driver classifies the accumulated turn to
              decide whether to inject the scripted repair.
  * OFFLINE - metrics recompute the same labels post-hoc; an LLM judge
              provides an independent second opinion (judge_pack.py), and
              rule/judge agreement is itself reported.

Labels:
  clarify_targeted - asks specifically about the corrupted/missing slot.
  clarify_generic  - signals non-understanding without targeting the slot.
  confirm          - proceeds with the task (echoes a value / accepts).
  other            - chit-chat, silence, or unrelated content.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .lang import get_pack


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _contains_any(text: str, patterns: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for pat in patterns:
        if pat in lowered:
            return pat
    return None


def _is_question(text: str, suffixes: tuple[str, ...]) -> bool:
    stripped = text.rstrip("。 　\n .!！")
    if "?" in stripped[-3:] or "？" in stripped[-3:]:
        return True
    tail = stripped[-14:]
    return any(suffix in tail
               for suffix in suffixes if suffix not in ("?", "？"))


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
    lang: str = "ja",
) -> TurnLabel:
    """Classify one model turn against the case's critical slot."""
    pack = get_pack(lang)
    text_n = _normalize(text)
    question = _is_question(text_n, pack.question_suffixes)
    echoed = bool(slot_value) and pack.value_in_text(slot_value, text_n)

    wh_words = pack.slot_wh_words.get(slot_type, ())
    wh_hit = _contains_any(text_n, wh_words)
    repeat_hit = _contains_any(text_n, pack.repeat_patterns)
    nonund_hit = _contains_any(text_n, pack.nonunderstanding_patterns)
    confirm_hit = _contains_any(text_n, pack.confirm_patterns)

    # Order matters. "Sorry, what time was it?" must be targeted even
    # without a repeat keyword; "5 pm, got it, setting the alarm" must be
    # a confirmation even though it echoes no WH word.
    if (wh_hit and question) or (repeat_hit and wh_hit):
        return TurnLabel("clarify_targeted", True, wh_hit or repeat_hit,
                         echoed, question)
    if repeat_hit or nonund_hit:
        return TurnLabel("clarify_generic", True, repeat_hit or nonund_hit,
                         echoed, question)
    if echoed and question and not confirm_hit:
        # Echo question ("3 pm, was it?" / 「15時、ですか？」): seeks
        # verification of the heard value -> targeted clarification.
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
    {"step": int, "time_sec": float, "piece": str}. Moshi tokenizers emit
    pieces carrying their own leading whitespace (English) or none
    (Japanese), so pieces are concatenated verbatim.
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
