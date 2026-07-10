"""Clarify fine-tuning dialogue generation.

Emits dialogues in the exact JSONL schema `scripts/generate_qwen3_tts_data.py
--dialogues-jsonl` consumes ({"id", "turns": [{speaker,text,emotion,...}]}),
plus a sidecar `corruption_plan.jsonl` that the audio post-processor
(`ICASSP2027/scripts/corrupt_training_audio.py`) uses to degrade the slot
span on the *user channel* of the rendered stereo WAVs.

Three training variants (the paper's ablation triangle):

  task_only       Every dialogue: user request (clean) -> assistant
                  confirms. Teaches the task format but never asking.
  clarify_lexical Adds ask-behavior ONLY where the ambiguity is lexical
                  (underspecified requests). Audio is always clean.
  clarify_full    Adds acoustically-corrupted requests whose target
                  response is a targeted clarification, then repair, then
                  confirmation. Contains minimal pairs: the same utterance
                  appears once clean (-> confirm) and once corrupted
                  (-> ask), differing ONLY in the user-channel audio, so
                  asking is learnable only from the acoustics.

Slot/value material is sampled from MASSIVE ja-JP *train* split (the
benchmark uses *test*), so surface forms never leak across the split.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scenario import BaseItem
from .slots import SLOT_WH_WORDS, build_repair_text

VARIANTS = ("task_only", "clarify_lexical", "clarify_full")

# Corruptions used on "ask" training dialogues, mirroring the benchmark's
# hard end of the grid (never its exact seeds; the benchmark stays unseen).
ASK_CORRUPTIONS = ("mask_silence", "mask_noise", "babble_snr-10",
                   "babble_snr-5")
# Mild corruption kept on some "confirm" dialogues: audible noise on the
# span that a listener still understands. This is the calibration signal -
# noise alone must not trigger asking; *information loss* must.
CONFIRM_MILD_CORRUPTIONS = ("babble_snr+5",)

WH_QUESTION_BY_SLOT: dict[str, str] = {
    "time": "すみません、少し聞き取れませんでした。何時でしたか？",
    "date": "すみません、日にちがよく聞こえませんでした。いつでしたか？",
    "person": "すみません、お名前が聞き取れませんでした。どなたでしたか？",
    "place_name": "すみません、場所がよく聞こえませんでした。どちらでしたか？",
    "event_name": "すみません、聞き取れませんでした。何のご予定でしたか？",
    "business_name": "すみません、お店の名前が聞き取れませんでした。どちらでしたか？",
    "artist_name": "すみません、聞き取れませんでした。どなたの曲でしたか？",
    "food_type": "すみません、聞き取れませんでした。何のご注文でしたか？",
    "timeofday": "すみません、聞き取れませんでした。何時ごろでしたか？",
}

CONFIRM_BY_INTENT: dict[str, str] = {
    "alarm_set": "{value}ですね。アラームを設定しておきますね。",
    "calendar_set": "{value}ですね。カレンダーに登録しておきますね。",
    "calendar_query": "{value}のご予定ですね。確認しますね。",
    "play_music": "{value}ですね。かけますね。",
    "takeaway_order": "{value}ですね。注文しておきますね。",
    "transport_query": "{value}ですね。調べてみますね。",
    "recommendation_events": "{value}ですね。探してみますね。",
    "recommendation_locations": "{value}ですね。探してみますね。",
    "qa_factoid": "{value}ですね。調べてみますね。",
}
CONFIRM_FALLBACK = "{value}ですね。承知しました。"


@dataclass
class TrainDialogue:
    dialogue_id: str
    variant: str
    behavior: str            # confirm | ask_acoustic | ask_lexical
    pair_id: str | None      # links clean/corrupted minimal pairs
    turns: list[dict[str, Any]]
    corruption: dict[str, Any] | None  # None => leave audio clean

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.dialogue_id,
            "variant": self.variant,
            "behavior": self.behavior,
            "pair_id": self.pair_id,
            "turns": self.turns,
            "corruption": self.corruption,
        }


def _confirm_text(intent: str, value: str) -> str:
    tpl = CONFIRM_BY_INTENT.get(intent, CONFIRM_FALLBACK)
    return tpl.format(value=value)


def _user_turns_split(item: BaseItem) -> list[dict[str, Any]]:
    """Split the user request into pre/slot/post user turns so the
    whole-utterance TTS + forced alignment pipeline recovers the slot-span
    boundaries; the corruption plan targets the middle turn by text."""
    turns = []
    for text in (item.pre_text, item.slot_text, item.post_text):
        if text:
            turns.append({"speaker": "user", "text": text,
                          "emotion": "neutral"})
    return turns


def build_confirm_dialogue(
    item: BaseItem, dialogue_id: str, variant: str,
    corruption: str | None = None, pair_id: str | None = None,
) -> TrainDialogue:
    turns = _user_turns_split(item) or [
        {"speaker": "user", "text": item.utterance_text, "emotion": "neutral"}
    ]
    turns.append({
        "speaker": "moshi",
        "text": _confirm_text(item.intent, item.slot_value),
        "emotion": "gentle",
    })
    plan = None
    if corruption:
        plan = {"condition": corruption, "target_text": item.slot_text,
                "channel": "user"}
    return TrainDialogue(dialogue_id, variant, "confirm", pair_id, turns, plan)


def build_ask_acoustic_dialogue(
    item: BaseItem, dialogue_id: str, variant: str,
    corruption: str, pair_id: str | None = None,
) -> TrainDialogue:
    ask = WH_QUESTION_BY_SLOT.get(
        item.slot_type, "すみません、もう一度お願いできますか？"
    )
    turns = _user_turns_split(item)
    turns += [
        {"speaker": "moshi", "text": ask, "emotion": "gentle"},
        {"speaker": "user",
         "text": build_repair_text(item.slot_type, item.slot_value),
         "emotion": "neutral"},
        {"speaker": "moshi",
         "text": _confirm_text(item.intent, item.slot_value),
         "emotion": "gentle"},
    ]
    plan = {"condition": corruption, "target_text": item.slot_text,
            "channel": "user"}
    return TrainDialogue(dialogue_id, variant, "ask_acoustic", pair_id,
                         turns, plan)


def build_ask_lexical_dialogue(
    item: BaseItem, dialogue_id: str, variant: str
) -> TrainDialogue:
    """Underspecified request (no slot in the utterance) -> ask -> repair."""
    wh = WH_QUESTION_BY_SLOT.get(
        item.slot_type, "すみません、もう少し詳しく教えていただけますか？"
    )
    turns = [
        {"speaker": "user", "text": item.utterance_text, "emotion": "neutral"},
        {"speaker": "moshi", "text": wh, "emotion": "gentle"},
        {"speaker": "user", "text": item.repair_text, "emotion": "neutral"},
        {"speaker": "moshi",
         "text": _confirm_text(item.intent, item.slot_value),
         "emotion": "gentle"},
    ]
    return TrainDialogue(dialogue_id, variant, "ask_lexical", None, turns,
                         None)


def generate_training_dialogues(
    acoustic_items: list[BaseItem],
    underspecified_items: list[BaseItem],
    variant: str,
    rng: random.Random,
    ask_ratio: float = 0.4,
    mild_noise_confirm_ratio: float = 0.25,
) -> list[TrainDialogue]:
    """Assemble the training mix for one variant.

    clarify_full assigns each acoustic item to either a minimal pair
    (clean-confirm + corrupted-ask, sharing pair_id) or a single role, so
    roughly `ask_ratio` of the acoustic dialogues are asks.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    dialogues: list[TrainDialogue] = []
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{variant}_{prefix}_{counter:05d}"

    for item in acoustic_items:
        item.validate()
        if variant == "clarify_full" and rng.random() < ask_ratio:
            pair_id = f"pair_{item.base_id}"
            corruption = rng.choice(ASK_CORRUPTIONS)
            dialogues.append(build_ask_acoustic_dialogue(
                item, next_id("ask"), variant, corruption, pair_id
            ))
            dialogues.append(build_confirm_dialogue(
                item, next_id("confirm"), variant, None, pair_id
            ))
        else:
            corruption = None
            if variant == "clarify_full" and \
                    rng.random() < mild_noise_confirm_ratio:
                corruption = rng.choice(CONFIRM_MILD_CORRUPTIONS)
            dialogues.append(build_confirm_dialogue(
                item, next_id("confirm"), variant, corruption
            ))

    if variant in ("clarify_lexical", "clarify_full"):
        for item in underspecified_items:
            item.validate()
            dialogues.append(build_ask_lexical_dialogue(
                item, next_id("asklex"), variant
            ))

    rng.shuffle(dialogues)
    return dialogues


def write_training_files(
    dialogues: list[TrainDialogue], out_dir: Path
) -> tuple[Path, Path]:
    """Write dialogues.jsonl (TTS input) and corruption_plan.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogues_path = out_dir / "dialogues.jsonl"
    plan_path = out_dir / "corruption_plan.jsonl"
    with dialogues_path.open("w", encoding="utf-8") as dfh, \
            plan_path.open("w", encoding="utf-8") as pfh:
        for d in dialogues:
            row = d.to_json()
            plan = row.pop("corruption")
            dfh.write(json.dumps(row, ensure_ascii=False) + "\n")
            pfh.write(json.dumps({
                "id": d.dialogue_id, "corruption": plan,
                "behavior": d.behavior, "pair_id": d.pair_id,
            }, ensure_ascii=False) + "\n")
    return dialogues_path, plan_path
