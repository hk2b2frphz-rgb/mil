#!/usr/bin/env python3
"""Generate deterministic screw-domain dialogues for Moshi fine-tuning.

The technical answer side is never authored by an LLM.  Every Moshi turn is
either an approved response template, an approved clarification question, or
the exact answer/action stored in the knowledge table.  User utterances are
varied independently so that the model learns a small and stable response
policy instead of many free-form technical answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


POC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = POC_ROOT / "config" / "dialogue_policy.yaml"
DEFAULT_KNOWLEDGE = POC_ROOT / "knowledge" / "screw_knowledge.csv"
DEFAULT_OUT = POC_ROOT / "artifacts"


@dataclass(frozen=True)
class Knowledge:
    knowledge_id: str
    category: str
    title: str
    intent: str
    required_slots: tuple[str, ...]
    slot_values: dict[str, str]
    answer_mode: str
    answer_text: str
    caution_text: str
    question_examples: tuple[str, ...]
    source_title: str
    source_url: str

    @property
    def final_text(self) -> str:
        return "".join(part for part in (self.answer_text, self.caution_text) if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--evaluation-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_policy(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"policy must be a mapping: {path}")
    return data


def parse_pairs(value: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in value.split("|"))):
        if "=" not in item:
            raise ValueError(f"slot value must use key=value: {item!r}")
        key, raw = item.split("=", 1)
        pairs[key.strip()] = raw.strip()
    return pairs


def load_knowledge(path: Path) -> list[Knowledge]:
    rows: list[Knowledge] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                Knowledge(
                    knowledge_id=raw["knowledge_id"].strip(),
                    category=raw["category"].strip(),
                    title=raw["title"].strip(),
                    intent=raw["intent"].strip(),
                    required_slots=tuple(
                        filter(None, (item.strip() for item in raw["required_slots"].split("|")))
                    ),
                    slot_values=parse_pairs(raw["slot_values"]),
                    answer_mode=raw["answer_mode"].strip(),
                    answer_text=raw["answer_text"].strip(),
                    caution_text=raw["caution_text"].strip(),
                    question_examples=tuple(
                        filter(None, (item.strip() for item in raw["question_examples"].split("||")))
                    ),
                    source_title=raw["source_title"].strip(),
                    source_url=raw["source_url"].strip(),
                )
            )
    return rows


def validate_inputs(policy: dict[str, Any], knowledge: list[Knowledge]) -> None:
    if len(knowledge) != 30:
        raise ValueError(f"expected exactly 30 knowledge rows, got {len(knowledge)}")
    ids = [item.knowledge_id for item in knowledge]
    if len(ids) != len(set(ids)):
        raise ValueError("knowledge_id values must be unique")
    slot_policy = policy.get("slots") or {}
    for item in knowledge:
        if item.answer_mode not in {"answer", "stop", "escalate"}:
            raise ValueError(f"{item.knowledge_id}: invalid answer_mode={item.answer_mode!r}")
        if not item.question_examples:
            raise ValueError(f"{item.knowledge_id}: question_examples is empty")
        if not item.answer_text or not item.source_title or not item.source_url:
            raise ValueError(f"{item.knowledge_id}: answer and source are required")
        for slot in item.required_slots:
            if slot not in slot_policy:
                raise ValueError(f"{item.knowledge_id}: undefined slot={slot!r}")
            if slot not in item.slot_values:
                raise ValueError(f"{item.knowledge_id}: no value for slot={slot!r}")
            questions = slot_policy[slot].get("questions") or []
            if len(questions) != 3:
                raise ValueError(f"slot {slot!r} must have exactly three question levels")
    voices = policy["voices"]
    if voices["system"] != "Ono_Anna":
        raise ValueError("the screw PoC system voice must be Ono_Anna")
    if "Ono_Anna" in voices["users"]:
        raise ValueError("Ono_Anna must not appear in the user voice pool")
    pronunciations = policy.get("tts_pronunciations") or []
    if not pronunciations:
        raise ValueError("tts_pronunciations must not be empty")
    for index, entry in enumerate(pronunciations, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"tts_pronunciations[{index}] must be a mapping")
        if not str(entry.get("written") or "") or not str(entry.get("spoken") or ""):
            raise ValueError(f"tts_pronunciations[{index}] needs written and spoken")


def tts_text(text: str, policy: dict[str, Any]) -> str:
    """Return the fixed, unambiguous pronunciation used only for synthesis."""
    rendered = text
    for entry in policy["tts_pronunciations"]:
        rendered = rendered.replace(str(entry["written"]), str(entry["spoken"]))
    return rendered


def attach_tts_text(dialogues: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            if turn.get("speaker") in {"user", "moshi"}:
                turn["tts_text"] = tts_text(str(turn["text"]), policy)


def user_turn(text: str, *, event: str | None = None, **extra: Any) -> dict[str, Any]:
    turn: dict[str, Any] = {"speaker": "user", "text": text, "emotion": "neutral"}
    if event:
        turn["event"] = event
    turn.update(extra)
    return turn


def moshi_turn(text: str, pattern: str, **extra: Any) -> dict[str, Any]:
    turn: dict[str, Any] = {
        "speaker": "moshi",
        "text": text,
        "emotion": "calm",
        "response_pattern": pattern,
    }
    turn.update(extra)
    return turn


def choose(rng: random.Random, values: Iterable[str]) -> str:
    options = tuple(values)
    if not options:
        raise ValueError("cannot choose from an empty collection")
    return options[rng.randrange(len(options))]


def slot_summary(item: Knowledge, policy: dict[str, Any]) -> str:
    parts = []
    for slot in item.required_slots:
        label = policy["slots"][slot]["label"]
        parts.append(f"{label}は{item.slot_values[slot]}")
    return "、".join(parts)


def start_question(
    item: Knowledge,
    policy: dict[str, Any],
    rng: random.Random,
    split: str,
    *,
    vague: bool,
) -> str:
    if vague:
        question = choose(rng, policy["vague_questions"][item.category])
    else:
        question = choose(rng, item.question_examples)
    wrapper = choose(rng, policy["question_templates"][split])
    text = wrapper.format(question=question)
    if not vague and item.required_slots:
        summary = slot_summary(item, policy)
        if summary and not all(value in text for value in item.slot_values.values()):
            text = f"{text} {summary}。"
    return text


def final_turn(item: Knowledge) -> dict[str, Any]:
    pattern = {"answer": "ANSWER", "stop": "STOP", "escalate": "ESCALATE"}[
        item.answer_mode
    ]
    return moshi_turn(item.final_text, pattern)


def ask_turn(slot: str, level: int, policy: dict[str, Any]) -> dict[str, Any]:
    return moshi_turn(policy["slots"][slot]["questions"][level], f"ASK_{slot.upper()}_L{level + 1}")


def reveal_turn(slot: str, value: str, policy: dict[str, Any], rng: random.Random, split: str) -> dict[str, Any]:
    text = choose(rng, policy["acknowledgements"][split]).format(value=value)
    return user_turn(text, provided_slot=slot, provided_value=value)


def build_core_dialogue(
    item: Knowledge,
    flow: str,
    sequence: int,
    policy: dict[str, Any],
    rng: random.Random,
    split: str,
) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    expected_patterns: list[str] = []

    if flow == "direct":
        turns.append(user_turn(start_question(item, policy, rng, split, vague=False)))
        turns.append(final_turn(item))
    elif flow == "clarify":
        turns.append(user_turn(start_question(item, policy, rng, split, vague=True)))
        for slot_index, slot in enumerate(item.required_slots):
            level = (sequence + slot_index) % 2
            turns.append(ask_turn(slot, level, policy))
            turns.append(reveal_turn(slot, item.slot_values[slot], policy, rng, split))
        turns.append(final_turn(item))
    elif flow == "varied":
        turns.append(user_turn(start_question(item, policy, rng, split, vague=True)))
        first, *remaining = item.required_slots
        level = 1 + (sequence % 2)
        turns.append(ask_turn(first, level, policy))
        turns.append(
            user_turn(
                f"たぶん{item.slot_values[first]}だと思います。",
                provided_slot=first,
                provided_value=item.slot_values[first],
            )
        )
        confirm = policy["response_patterns"]["CONFIRM"].format(
            slot_label=policy["slots"][first]["label"], value=item.slot_values[first]
        )
        turns.append(moshi_turn(confirm, "CONFIRM"))
        turns.append(user_turn(f"はい。見直しても{item.slot_values[first]}です。"))
        for slot_index, slot in enumerate(remaining):
            turns.append(ask_turn(slot, min(2, 1 + ((sequence + slot_index) % 2)), policy))
            turns.append(reveal_turn(slot, item.slot_values[slot], policy, rng, split))
        turns.append(final_turn(item))
    elif flow == "unresolved":
        turns.append(user_turn(start_question(item, policy, rng, split, vague=True)))
        target_slot = item.required_slots[sequence % len(item.required_slots)]
        for level in range(3):
            turns.append(ask_turn(target_slot, level, policy))
            turns.append(user_turn(choose(rng, policy["unknown_replies"][split])))
        turns.append(moshi_turn(policy["response_patterns"]["ESCALATE"], "ESCALATE"))
    else:
        raise ValueError(f"unknown flow: {flow}")

    expected_patterns.extend(
        turn["response_pattern"] for turn in turns if turn["speaker"] == "moshi"
    )
    return {
        "id": f"{split}_{item.knowledge_id.lower()}_{sequence:03d}",
        "category": item.category,
        "risk_level": "medium" if item.answer_mode in {"stop", "escalate"} else "low",
        "title": item.title,
        "source_use_case": item.knowledge_id,
        "knowledge_id": item.knowledge_id,
        "intent": item.intent,
        "flow": flow,
        "required_slots": list(item.required_slots),
        "slot_values": item.slot_values,
        "expected_answer": item.final_text,
        "expected_action": item.answer_mode,
        "expected_response_patterns": expected_patterns,
        "source": {"title": item.source_title, "url": item.source_url},
        "turns": turns,
    }


def build_out_of_scope(
    index: int, policy: dict[str, Any], rng: random.Random, split: str
) -> dict[str, Any]:
    question = choose(rng, policy["out_of_scope_questions"][split])
    answer = policy["response_patterns"]["OUT_OF_SCOPE"]
    return {
        "id": f"{split}_out_of_scope_{index:03d}",
        "category": "out_of_scope",
        "risk_level": "medium",
        "title": "ねじ領域外の問い合わせ",
        "source_use_case": "OUT_OF_SCOPE",
        "knowledge_id": None,
        "intent": "out_of_scope",
        "flow": "out_of_scope",
        "required_slots": [],
        "slot_values": {},
        "expected_answer": answer,
        "expected_action": "out_of_scope",
        "expected_response_patterns": ["OUT_OF_SCOPE"],
        "source": None,
        "turns": [user_turn(question), moshi_turn(answer, "OUT_OF_SCOPE")],
    }


def generate_train(
    knowledge: list[Knowledge], policy: dict[str, Any], rng: random.Random, count: int
) -> list[dict[str, Any]]:
    flows = policy["dataset"]["flows_per_knowledge"]
    dialogues: list[dict[str, Any]] = []
    for item in knowledge:
        sequence = 0
        for flow in ("direct", "clarify", "varied", "unresolved"):
            for _ in range(int(flows[flow])):
                dialogues.append(build_core_dialogue(item, flow, sequence, policy, rng, "train"))
                sequence += 1
    out_of_scope_count = count - len(dialogues)
    if out_of_scope_count < 0:
        raise ValueError(f"train-count {count} is smaller than core set {len(dialogues)}")
    dialogues.extend(
        build_out_of_scope(index, policy, rng, "train")
        for index in range(out_of_scope_count)
    )
    rng.shuffle(dialogues)
    attach_tts_text(dialogues, policy)
    return dialogues


def generate_evaluation(
    knowledge: list[Knowledge], policy: dict[str, Any], rng: random.Random, count: int
) -> list[dict[str, Any]]:
    flows = ("direct", "clarify", "varied", "unresolved", "clarify")
    dialogues = [
        build_core_dialogue(item, flow, index, policy, rng, "evaluation")
        for item in knowledge
        for index, flow in enumerate(flows)
    ]
    remaining = count - len(dialogues)
    if remaining < 0:
        raise ValueError(f"evaluation-count {count} is smaller than core set {len(dialogues)}")
    dialogues.extend(
        build_out_of_scope(index, policy, rng, "evaluation")
        for index in range(remaining)
    )
    rng.shuffle(dialogues)
    attach_tts_text(dialogues, policy)
    return dialogues


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dialogues": len(rows),
        "knowledge_coverage": len({row["knowledge_id"] for row in rows if row["knowledge_id"]}),
        "flows": dict(sorted(Counter(row["flow"] for row in rows).items())),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "turn_timing": {
            "sequential": sum(
                1
                for row in rows
                for turn in row["turns"]
                if turn.get("speaker") != "silence"
                and turn.get("timing", "sequential") == "sequential"
            )
        },
    }


def main() -> None:
    args = parse_args()
    policy = load_policy(args.policy)
    knowledge = load_knowledge(args.knowledge)
    validate_inputs(policy, knowledge)
    train_count = args.train_count or int(policy["dataset"]["train_dialogues"])
    evaluation_count = args.evaluation_count or int(policy["dataset"]["evaluation_dialogues"])
    seed = args.seed if args.seed is not None else int(policy["seed"])

    train = generate_train(knowledge, policy, random.Random(seed), train_count)
    evaluation = generate_evaluation(
        knowledge, policy, random.Random(seed + 1), evaluation_count
    )
    write_jsonl(args.out_dir / "train_dialogues.jsonl", train)
    write_jsonl(args.out_dir / "evaluation_dialogues.jsonl", evaluation)
    manifest = {
        "version": 1,
        "seed": seed,
        "knowledge_rows": len(knowledge),
        "voices": policy["voices"],
        "response_policy": "fixed-system-progressive-clarification",
        "train": summarize(train),
        "evaluation": summarize(evaluation),
    }
    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
