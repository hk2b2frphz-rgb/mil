"""Pack trial outputs for offline LLM judging (language-aware prompts).

The rule-based labels in detector.py gate the online repair injection and
drive the headline numbers; an LLM judge provides an independent second
opinion on (a) first-turn dialogue-act classification and (b) whether the
final turn commits to the correct slot value. The paper reports rule/judge
agreement (Cohen's kappa) and uses judge labels for the semantic-quality
axis that rules cannot capture (e.g., politeness-wrapped confirmations).

The pack is a provider-agnostic chat-payload JSONL:
  {"id": ..., "meta": {...}, "messages": [{"role","content"}, ...]}
run on the LOCAL PC with the existing Azure/OpenAI judge credentials
(cluster jobs never call external APIs, matching repo policy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT_EN = """You are an evaluator of spoken dialogue systems. You will assess an assistant's response to a user utterance in which a span may have been acoustically corrupted (unintelligible).

Output JSON only:
{
  "first_turn_act": "clarify_targeted" | "clarify_generic" | "confirm" | "other",
  "final_commits_value": true | false,
  "final_value_correct": true | false,
  "notes": "<under 20 words>"
}

Definitions:
- clarify_targeted: asks back about the specific missing information (time, place, name, ...)
- clarify_generic: signals non-understanding without targeting it ("say that again", "I didn't catch that")
- confirm: repeats back / accepts the content and proceeds with the task
- other: anything else (chit-chat, unrelated content)
- final_commits_value: the final turn commits to a concrete slot value
- final_value_correct: the committed value matches the gold value (paraphrases and clock-format variants count as matches)"""

USER_TEMPLATE_EN = """# Task context
Intent: {intent}
Critical slot: {slot_type}
Gold value: {slot_value}

# User utterance (transcript; the span marked with stars was the corruption target)
{marked_utterance}

# Assistant's first response turn
{first_turn}

# (If the assistant asked) the user's restatement
{repair}

# Assistant's final turn
{final_turn}

Output JSON only."""

SYSTEM_PROMPT_JA = """あなたは音声対話システムの評価者です。ユーザー発話(その一部が雑音で聞き取れない場合があります)に対するアシスタントの応答を評価します。

以下をJSONで出力してください:
{
  "first_turn_act": "clarify_targeted" | "clarify_generic" | "confirm" | "other",
  "final_commits_value": true | false,
  "final_value_correct": true | false,
  "notes": "<50字以内>"
}

定義:
- clarify_targeted: 欠落した情報(時刻・場所・人名など)を特定して聞き返している
- clarify_generic: 特定せず「もう一度」「聞き取れなかった」等の聞き返し
- confirm: 内容を確認・復唱してタスクを進めている
- other: 上記以外(雑談・無関係な応答など)
- final_commits_value: 最終ターンで具体的なスロット値を確定させている
- final_value_correct: 確定した値が正解値と一致する(言い換え・時刻表記ゆれは一致とみなす)"""

USER_TEMPLATE_JA = """# タスク文脈
意図: {intent}
重要スロット: {slot_type}
正解値: {slot_value}

# ユーザー発話(書き起こし; ★…★ 区間は音響劣化の対象)
{marked_utterance}

# アシスタントの最初の応答ターン
{first_turn}

# (聞き返しがあった場合)ユーザーの言い直し
{repair}

# アシスタントの最終ターン
{final_turn}

JSONのみを出力してください。"""

_PROMPTS = {
    "en": (SYSTEM_PROMPT_EN, USER_TEMPLATE_EN,
           {"missing": "(not present in the utterance; provided in the restatement)",
            "no_response": "(no response)", "none": "(none)"}),
    "ja": (SYSTEM_PROMPT_JA, USER_TEMPLATE_JA,
           {"missing": "(発話に含まれない; 言い直しで提示)",
            "no_response": "(応答なし)", "none": "(なし)"}),
}


def mark_utterance(case: dict[str, Any]) -> str:
    base = case["base"]
    if base["arm"] == "underspecified" or case["condition"] == "clean":
        return base["utterance_text"]
    return f"{base['pre_text']}★{base['slot_text']}★{base['post_text']}"


def pack_trial(
    case: dict[str, Any],
    seed: int,
    first_turn_text: str,
    final_turn_text: str,
    repair_injected: bool,
) -> dict[str, Any]:
    base = case["base"]
    lang = base.get("language", "ja")
    system, template, words = _PROMPTS.get(lang, _PROMPTS["en"])
    trial_id = f"{case['case_id']}__seed{seed}"
    user = template.format(
        intent=base["intent"],
        slot_type=base["slot_type"],
        slot_value=base["slot_value"] or words["missing"],
        marked_utterance=mark_utterance(case),
        first_turn=first_turn_text or words["no_response"],
        repair=base["repair_text"] if repair_injected else words["none"],
        final_turn=final_turn_text or words["none"],
    )
    return {
        "id": trial_id,
        "meta": {
            "case_id": case["case_id"], "seed": seed,
            "condition": case["condition"], "arm": base["arm"],
            "language": lang, "source": base.get("source", "massive"),
            "slot_type": base["slot_type"], "slot_value": base["slot_value"],
            "expected_behavior": case["expected_behavior"],
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def write_pack(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def rule_judge_agreement(
    rule_labels: dict[str, str], judge_labels: dict[str, str]
) -> dict[str, Any]:
    """Cohen's kappa between rule-based and judge first-turn labels."""
    common = sorted(set(rule_labels) & set(judge_labels))
    if not common:
        return {"kappa": None, "n": 0}
    labels = sorted(set(rule_labels[i] for i in common)
                    | set(judge_labels[i] for i in common))
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(common)
    obs = sum(1 for i in common if rule_labels[i] == judge_labels[i]) / n
    rule_marg = [0.0] * len(labels)
    judge_marg = [0.0] * len(labels)
    for i in common:
        rule_marg[idx[rule_labels[i]]] += 1 / n
        judge_marg[idx[judge_labels[i]]] += 1 / n
    exp = sum(r * j for r, j in zip(rule_marg, judge_marg))
    kappa = (obs - exp) / (1 - exp) if exp < 1 else None
    return {"kappa": round(kappa, 4) if kappa is not None else None,
            "observed_agreement": round(obs, 4), "n": n}
