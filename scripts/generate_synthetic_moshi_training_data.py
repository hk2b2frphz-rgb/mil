#!/usr/bin/env python3
from __future__ import annotations
"""
Generate a small synthetic Moshi fine-tuning dataset.

Pipeline:
1. LLM 4 generates Japanese dialogue scripts for loneliness/isolation
   support-window use cases.
2. In the default "scripted-moshi-tts" mode, each script turn is rendered
   with Kyutai/Moshi TTS and placed into a stereo dialogue file.
3. In "moshi-selfplay" mode, user turns are rendered as audio and
   llm-jp-moshi-v1 generates the counselor stream from that user audio context.
4. In "scripted-local-tts" mode, both speakers are rendered from the LLM
   script with local pyopenjtalk/pyttsx3 TTS for quick format checks.

Outputs follow the moshi-finetune dataset shape:
  out_dir/
    synthetic_moshi_train.jsonl
    dialogues.jsonl
    data_stereo/
      sample_001.wav
      sample_001.json
"""

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from response_recorder import (  # noqa: E402
    decode_audio,
    load_models,
    load_wav_mono,
    run_trial,
    synthesize_text_to_wav,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

LISTENING_DIALOGUES_PATH = REPO_ROOT / "tests" / "fixtures" / "listening_dialogues.jsonl"


def load_listening_dialogue_exemplars(
    path: Path = LISTENING_DIALOGUES_PATH,
) -> list[dict[str, Any]]:
    """Load the curated active-listening examples used by every prompt."""
    exemplars: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Listening-dialogue few-shot fixture not found: {path}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("turns"), list):
            raise ValueError(
                f"Listening exemplar at {path}:{line_number} must be a dialogue object."
            )
        exemplars.append(row)
    if len(exemplars) != 5:
        raise ValueError(
            f"Expected exactly 5 listening-dialogue exemplars in {path}, "
            f"found {len(exemplars)}."
        )
    return exemplars


LISTENING_DIALOGUE_EXEMPLARS = load_listening_dialogue_exemplars()

DIALOGUE_SYSTEM_INSTRUCTIONS = """
あなたは日本語の「孤独・孤立相談窓口」の対話データを作る。相談者 user と相談員 moshi の自然な会話を書く。
- moshi は丁寧語で傾聴中心。すぐ助言せず、言い換え・要約・感情の承認・短い相づちを織り交ぜる。相づちや文型を繰り返さない。
- moshi は「うん」「うんうん」などのくだけた相づちを避け、「はい」「ええ」「そうなんですね」「なるほど」を使う。
- user の感情に合わせて語調を変える。設定に沿った具体的な話題を、雑談から相談まで扱う。
- ターン制の交互会話にしない。日本語の自然な会話のように、相手が話している「途中」で短い相づちを重ね、
  どんどん反応を返す。聞き手の相づちは相手の発話を遮らず、話し手はそのまま続ける。
""".strip()


DEFAULT_USE_CASES: list[dict[str, Any]] = [
    {
        "id": "smalltalk_evening_001",
        "category": "smalltalk",
        "risk_level": "low",
        "situation": "夜、なんとなく人と話したくて窓口に来た。",
        "user_profile": "30代、一人暮らし。深刻な相談というほどではないが寂しさがある。",
        "opening": "こんばんは。相談というほどでもないんですが、少し話してもいいですか。",
        "target_turns": 8,
        "tone": "雑談寄り。相談員は歓迎し、急かさず、軽く近況を聞く。",
    },
    {
        "id": "holiday_loneliness_001",
        "category": "loneliness_light",
        "risk_level": "low",
        "situation": "休日に予定がなく、誰ともつながっていない感じが強い。",
        "user_profile": "20代後半。SNSを見ると置いていかれた気持ちになる。",
        "opening": "休日に予定がないと、自分だけ誰にも呼ばれてない気がします。",
        "target_turns": 8,
        "tone": "気持ちの受け止めを中心に、今日できる小さな行動を一緒に探す。",
    },
    {
        "id": "help_request_hesitation_001",
        "category": "loneliness_deep",
        "risk_level": "medium",
        "situation": "誰かに助けを求めたいが、迷惑だと思って言えない。",
        "user_profile": "40代。仕事と家庭の両方で孤立感がある。",
        "opening": "助けてって言いたいんですけど、迷惑だと思われそうで言えません。",
        "target_turns": 10,
        "tone": "援助要請の難しさを認め、安全確認を押しつけず穏やかに行う。",
    },
]


@dataclass
class DialogueTurn:
    speaker: str  # "user" | "moshi" | "silence"
    text: str = ""
    emotion: str | None = None
    duration_sec: float | None = None  # speaker=="silence" 用
    note: str | None = None
    timing: str = "sequential"  # "sequential" | "overlap_previous"
    start_after_previous_start_sec: float | None = None
    truncate_previous_after_sec: float | None = None
    gain: float = 1.0
    voice_role: str | None = None  # "user" | "other" | "background"
    event: str | None = None


@dataclass
class Dialogue:
    id: str
    category: str
    risk_level: str
    title: str
    source_use_case: str
    turns: list[DialogueTurn]
    generator_notes: str = ""
    duplex_task: str | None = None


@dataclass
class AudioSegment:
    speaker: str
    label: str
    text: str
    start_sec: float
    end_sec: float
    pcm: np.ndarray
    event: str | None = None
    voice_role: str | None = None


@dataclass(frozen=True)
class AgentPrompt:
    system: str
    user: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small LLM 4 + Moshi TTS synthetic stereo dataset "
            "for Moshi fine-tuning experiments. llm-jp-moshi can also be "
            "sampled with --mode moshi-selfplay."
        )
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for manifest, stereo WAVs, and metadata.",
    )
    parser.add_argument(
        "--num-dialogues",
        type=int,
        default=3,
        help="Number of test dialogues to generate. Default: 3.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=[
            "scripted-moshi-tts",
            "moshi-selfplay",
            "scripted-local-tts",
            "scripted-stereo",
            "dialogues-only",
        ],
        default="scripted-moshi-tts",
        help=(
            "scripted-moshi-tts: render the generated script with Kyutai/Moshi "
            "TTS into stereo WAVs. moshi-selfplay: local TTS creates the user "
            "stream and llm-jp-moshi generates the counselor stream. "
            "scripted-local-tts/scripted-stereo: local TTS renders both speakers. "
            "dialogues-only: LLM で対話 JSONL を生成して終了（音声合成は行わない）。"
        ),
    )
    parser.add_argument(
        "--use-cases-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL use-case cards. Defaults to three built-in cases.",
    )
    parser.add_argument(
        "--dialogues-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional pre-generated dialogue JSONL. If set, LLM is not loaded "
            "and these dialogues are rendered instead."
        ),
    )
    parser.add_argument(
        "--no-template-fallback",
        action="store_true",
        help="Deprecated; generation now fails unless --allow-template-fallback is set.",
    )
    parser.add_argument(
        "--allow-template-fallback",
        action="store_true",
        help="Use deterministic templates if LLM generation fails.",
    )

    # LLM generation
    parser.add_argument(
        "--llm-backend",
        choices=[
            "transformers-subprocess",
            "transformers",
            "openai-compatible",
            "template",
        ],
        default="transformers-subprocess",
        help=(
            "Dialogue generation backend. transformers-subprocess runs LLM "
            "from this script via a separate Python environment, avoiding "
            "dependency conflicts with Moshi. Default: transformers-subprocess."
        ),
    )
    parser.add_argument(
        "--llm-python",
        default=os.environ.get("LLM_PYTHON"),
        help=(
            "Python executable for --llm-backend transformers-subprocess. "
            "Defaults to gemma_runtime/.venv/bin/python if present, otherwise "
            "the current Python."
        ),
    )
    parser.add_argument(
        "--llm-worker",
        type=Path,
        default=Path(__file__).with_name("gemma_dialogue_worker.py"),
        help="Worker script used by --llm-backend transformers-subprocess.",
    )
    parser.add_argument(
        "--llm-api-base",
        default=os.environ.get("LLM_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8080/v1",
        help=(
            "OpenAI-compatible API base URL for LLM, e.g. "
            "http://127.0.0.1:8080/v1. Used with --llm-backend "
            "openai-compatible."
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="Optional API key for the OpenAI-compatible LLM endpoint.",
    )
    parser.add_argument(
        "--llm-openai-endpoint",
        choices=["chat", "completions"],
        default=os.environ.get("LLM_OPENAI_ENDPOINT") or "chat",
        help=(
            "OpenAI-compatible generation endpoint. chat uses "
            "/v1/chat/completions with messages. completions uses "
            "/v1/completions with a ChatML-style prompt. Default: chat."
        ),
    )
    parser.add_argument(
        "--llm-completions-no-think",
        action="store_true",
        default=(os.environ.get("LLM_COMPLETIONS_NO_THINK") == "1"),
        help=(
            "When using --llm-openai-endpoint completions, append /no_think "
            "to the final user message. This is needed for Qwen reasoning "
            "models because raw completions do not apply chat-template "
            "enable_thinking=false kwargs."
        ),
    )
    parser.add_argument(
        "--llm-timeout-sec",
        type=float,
        default=300.0,
        help="Timeout for LLM generation requests/subprocess calls.",
    )
    parser.add_argument(
        "--llm-model",
        default="google/gemma-2-2b-it",
        help=(
            "LLM model id/name. For OpenAI-compatible servers this is sent "
            "as the request model field."
        ),
    )
    parser.add_argument(
        "--llm-task",
        default="any-to-any",
        help=(
            "Transformers pipeline task, used with --llm-backend "
            "transformers or transformers-subprocess."
        ),
    )
    parser.add_argument(
        "--llm-device-map",
        default="auto",
        help='Device map passed to transformers.pipeline. Default: "auto".',
    )
    parser.add_argument(
        "--llm-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.8)
    parser.add_argument(
        "--llm-frequency-penalty",
        type=float,
        default=0.3,
        help="OpenAI-compatible frequency penalty. Default: 0.3.",
    )
    parser.add_argument(
        "--llm-presence-penalty",
        type=float,
        default=0.1,
        help="OpenAI-compatible presence penalty. Default: 0.1.",
    )
    parser.add_argument(
        "--llm-repetition-penalty",
        type=float,
        default=float(os.environ.get("LLM_REPETITION_PENALTY") or 1.0),
        help=(
            "vLLM repetition_penalty (>1.0 discourages repeats). Counters the "
            "model degenerating into endless newline/character repetition. "
            "Default: 1.0 (off). Only sent on the openai-compatible backend "
            "when > 1.0."
        ),
    )
    parser.add_argument("--llm-max-new-tokens", type=int, default=1400)
    parser.add_argument(
        "--llm-num-fewshot",
        type=int,
        default=int(os.environ.get("LLM_NUM_FEWSHOT") or 1),
        help=(
            "Number of few-shot exemplars to embed in the prompt (0-5). Fewer "
            "keeps the prompt short and the model under tighter control. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        choices=["low", "medium", "high"],
        default=os.environ.get("LLM_REASONING_EFFORT") or None,
        help=(
            "Reasoning effort for reasoning models (e.g. gpt-oss) on the "
            "openai-compatible backend. Lower values stop the model from "
            "spending the whole token budget on the analysis channel, so the "
            "final JSON channel is actually emitted. Default: unset (server "
            "default)."
        ),
    )
    parser.add_argument(
        "--llm-response-format",
        choices=["json_schema", "json_object", "text"],
        default=os.environ.get("LLM_RESPONSE_FORMAT") or None,
        help=(
            "OpenAI-compatible response_format. 'json_schema' (recommended) "
            "forces the exact dialogue structure via vLLM guided decoding, "
            "preventing newline wandering and truncated/broken JSON. "
            "'json_object' only requires valid JSON (no structure). "
            "Default: unset."
        ),
    )
    parser.add_argument(
        "--dialogue-generation-mode",
        choices=["single", "multi-agent"],
        default=os.environ.get("DIALOGUE_GENERATION_MODE") or "single",
        help=(
            "single: existing one-shot prompt. multi-agent: run separate user, "
            "moshi, completion-judge, and aizuchi agents. multi-agent currently "
            "requires --llm-backend openai-compatible (vLLM/OpenAI-style server)."
        ),
    )
    parser.add_argument(
        "--multi-agent-min-pairs",
        type=int,
        default=int(os.environ.get("MULTI_AGENT_MIN_PAIRS") or 3),
        help="Minimum user/moshi exchange pairs before the judge may stop.",
    )
    parser.add_argument(
        "--multi-agent-max-pairs",
        type=int,
        default=int(os.environ.get("MULTI_AGENT_MAX_PAIRS") or 5),
        help="Maximum user/moshi exchange pairs in multi-agent mode.",
    )
    parser.add_argument(
        "--multi-agent-user-temperature",
        type=float,
        default=float(os.environ.get("MULTI_AGENT_USER_TEMPERATURE") or 0.85),
        help="Temperature for userAI; this is where dialogue randomness lives.",
    )
    parser.add_argument(
        "--multi-agent-system-temperature",
        type=float,
        default=float(os.environ.get("MULTI_AGENT_SYSTEM_TEMPERATURE") or 0.2),
        help="Temperature for moshi systemAI responses.",
    )
    parser.add_argument(
        "--multi-agent-judge-temperature",
        type=float,
        default=float(os.environ.get("MULTI_AGENT_JUDGE_TEMPERATURE") or 0.0),
        help="Temperature for the completion judge.",
    )
    parser.add_argument(
        "--multi-agent-aizuchi-temperature",
        type=float,
        default=float(os.environ.get("MULTI_AGENT_AIZUCHI_TEMPERATURE") or 0.2),
        help="Temperature for the aizuchi insertion agent.",
    )
    parser.add_argument(
        "--multi-agent-aizuchi-min-chars",
        type=int,
        default=int(os.environ.get("MULTI_AGENT_AIZUCHI_MIN_CHARS") or 32),
        help="Only ask the aizuchi agent for user turns at least this long.",
    )
    parser.add_argument(
        "--multi-agent-max-aizuchi-per-user",
        type=int,
        default=int(os.environ.get("MULTI_AGENT_MAX_AIZUCHI_PER_USER") or 1),
        help="Maximum moshi backchannels inserted inside one user utterance.",
    )
    parser.add_argument(
        "--multi-agent-empty-policy",
        choices=["fail", "fallback"],
        default=os.environ.get("MULTI_AGENT_EMPTY_POLICY") or "fail",
        help=(
            "What to do when userAI/systemAI returns an empty utterance after "
            "retry. fail stops the job with trace context; fallback inserts a "
            "deterministic utterance. Default: fail."
        ),
    )
    parser.add_argument(
        "--multi-agent-aizuchi-mode",
        choices=["separate", "inline"],
        default=os.environ.get("MULTI_AGENT_AIZUCHI_MODE") or "separate",
        help=(
            "separate: add aizuchi to surface output only; userAI/systemAI/judgeAI "
            "see core turns without aizuchi. inline: add aizuchi to the running "
            "turn history too, so later agents see it. Default: separate."
        ),
    )
    parser.add_argument(
        "--no-multi-agent-aizuchi",
        action="store_true",
        help="Disable the separate aizuchi agent in multi-agent mode.",
    )
    parser.add_argument(
        "--multi-agent-concurrency",
        type=int,
        default=int(os.environ.get("MULTI_AGENT_CONCURRENCY") or 1),
        help=(
            "Number of dialogues to generate concurrently in multi-agent mode. "
            "Each dialogue is an independent chain of LLM calls, so running many "
            "in parallel keeps the vLLM server busy (KV cache otherwise sits near "
            "0%). 1 = sequential (legacy). Default: 1."
        ),
    )
    parser.add_argument(
        "--dump-agent-prompts",
        action="store_true",
        help=(
            "In multi-agent mode, write a prompt preview to "
            "<out-dir>/multi_agent_prompts.json."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers model loading.",
    )

    # Moshi generation
    parser.add_argument(
        "--moshi-hf-repo",
        default="llm-jp/llm-jp-moshi-v1",
        help="Moshi checkpoint used in moshi-selfplay mode.",
    )
    parser.add_argument("--moshi-weight", default=None)
    parser.add_argument("--mimi-weight", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--half", action="store_true", help="Use float16 instead of bfloat16.")
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--temp-text", type=float, default=None)
    parser.add_argument("--cfg-coef", type=float, default=None)
    parser.add_argument(
        "--moshi-silence-sec",
        type=float,
        default=18.0,
        help="Trailing silence fed after the user stream so Moshi can answer.",
    )
    parser.add_argument(
        "--max-total-sec",
        type=float,
        default=120.0,
        help="Safety cap for each Moshi streaming run.",
    )

    # Kyutai/Moshi TTS for script-faithful audio rendering
    parser.add_argument(
        "--moshi-tts-repo",
        default="kyutai/tts-1.6b-en_fr",
        help="HF repo for Kyutai/Moshi TTS in scripted-moshi-tts mode.",
    )
    parser.add_argument(
        "--moshi-tts-voice-repo",
        default=None,
        help="Optional HF repo for pre-computed Kyutai/Moshi TTS voices.",
    )
    parser.add_argument(
        "--moshi-tts-voice-user",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="Voice path/name for user turns in scripted-moshi-tts mode.",
    )
    parser.add_argument(
        "--moshi-tts-voice-moshi",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="Voice path/name for counselor turns in scripted-moshi-tts mode.",
    )
    parser.add_argument("--moshi-tts-temp", type=float, default=0.6)
    parser.add_argument("--moshi-tts-cfg-coef", type=float, default=2.0)
    parser.add_argument("--moshi-tts-n-q", type=int, default=32)

    # Local TTS
    parser.add_argument("--tts-rate", type=int, default=220)
    parser.add_argument(
        "--tts-speed-user",
        type=float,
        default=1.9,
        help="Speed multiplier for user-side local TTS.",
    )
    parser.add_argument(
        "--tts-speed-moshi",
        type=float,
        default=1.35,
        help="Speed multiplier for counselor TTS in scripted-stereo mode.",
    )
    parser.add_argument("--tts-voice-user", default=None)
    parser.add_argument("--tts-voice-moshi", default=None)
    parser.add_argument(
        "--no-trim-tts",
        action="store_true",
        help="Keep leading/trailing silence from local TTS outputs.",
    )

    # Timing layout
    parser.add_argument("--lead-in-sec", type=float, default=0.5)
    parser.add_argument("--turn-gap-sec", type=float, default=0.45)
    parser.add_argument(
        "--user-pause-sec",
        type=float,
        default=0.8,
        help="Pause after each user utterance before expected Moshi response time.",
    )
    parser.add_argument(
        "--assistant-gap-sec",
        type=float,
        default=0.6,
        help="Extra gap after estimated assistant response when scheduling user-only prompts.",
    )
    parser.add_argument(
        "--estimate-chars-per-sec",
        type=float,
        default=9.0,
        help="Used only to reserve response space in moshi-selfplay prompts.",
    )
    parser.add_argument(
        "--manifest-name",
        default="synthetic_moshi_train.jsonl",
        help="Name of the output manifest JSONL file.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def safe_stem(text: str, fallback: str) -> str:
    stem = re.sub(r"\s+", "_", text.strip())
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", stem)
    stem = re.sub(r"[^0-9A-Za-z_.\-\u3040-\u30ff\u3400-\u9fff]+", "_", stem)
    stem = stem[:60].strip("._-")
    return stem or fallback


def normalize_speaker(value: str) -> str:
    value = value.strip().lower()
    if value in {"moshi", "assistant", "counselor", "相談員", "窓口", "支援員"}:
        return "moshi"
    if value in {"user", "client", "相談者", "利用者", "来訪者"}:
        return "user"
    return value


def optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(text[start: end + 1])


def dialogue_response_schema() -> dict[str, Any]:
    """JSON schema for guided decoding on the openai-compatible backend.

    Free-form json_object mode lets an unsure model emit unbounded whitespace and
    never close the object, so it runs to max_tokens and yields broken JSON. A
    strict schema forces the exact structure and a closed object (xgrammar in
    vLLM), eliminating the newline wandering and truncation. maxItems also caps
    runaway turn counts. Fields mirror dialogue_from_mapping()."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "category": {"type": "string"},
            "risk_level": {"type": "string"},
            "duplex_task": {"type": "string"},
            "title": {"type": "string", "maxLength": 100},
            "turns": {
                "type": "array",
                "minItems": 2,
                "maxItems": 40,
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "enum": ["user", "moshi", "silence"],
                        },
                        # maxLength is the key guard against the model degenerating
                        # into endless "\n\n\n" inside a string: json_schema alone
                        # permits arbitrarily long strings, so a per-field cap forces
                        # the string (and thus the turn) to terminate.
                        "text": {"type": "string", "maxLength": 120},
                        "emotion": {"type": "string", "maxLength": 32},
                        "duration_sec": {"type": "number"},
                        "note": {"type": "string", "maxLength": 120},
                        "timing": {
                            "type": "string",
                            "enum": ["sequential", "overlap_previous"],
                        },
                        "start_after_previous_start_sec": {"type": "number"},
                        "truncate_previous_after_sec": {"type": "number"},
                        "gain": {"type": "number"},
                        "voice_role": {"type": "string", "maxLength": 32},
                        "event": {"type": "string", "maxLength": 64},
                    },
                    "required": ["speaker"],
                    "additionalProperties": False,
                },
            },
            "generator_notes": {"type": "string", "maxLength": 300},
        },
        "required": ["title", "turns"],
        "additionalProperties": False,
    }


# --- Plain-text transcript format -------------------------------------------
# Asking the model for JSON is fragile: a single unclosed brace or a degenerate
# "\n\n\n" run breaks the whole dialogue. Instead the model emits one turn per
# line, pipe-delimited, which it can produce far more reliably:
#
#     user|hesitant|発話テキスト
#     moshi|warm|応答テキスト
#     silence|2.5|沈黙の理由
#
# The model writes backchannels (相づち) itself, as plain short turns interleaved in
# the conversation (taught via the few-shot exemplars and the prompt). No overlap/
# timing meta is encoded in the text; the full-duplex overlap layer is designed
# separately. The labeled duplex tasks (not used in the default free-form run) still
# carry explicit timing in a leading <<...>> tag on the text field:
#
#     moshi|gentle|<<overlap=1.5 event=model_backchannel>>はい。
#
# We split with maxsplit=2 so text may itself contain "|".

_TRANSCRIPT_TAG_RE = re.compile(r"^<<(?P<tags>.*?)>>\s*(?P<text>.*)$", re.DOTALL)


def _parse_transcript_tags(tagstr: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for token in tagstr.replace(",", " ").split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "overlap":
            out["timing"] = "overlap_previous"
            out["start_after_previous_start_sec"] = value
        elif key == "truncate":
            out["truncate_previous_after_sec"] = value
        elif key in {"event", "voice_role"}:
            out[key] = value
        elif key == "gain":
            out["gain"] = value
    return out


_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\s*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop a reasoning model's thinking block from the output.

    Reasoning models (Qwen3, gpt-oss) sometimes leak a <think>...</think> block
    into the content channel despite enable_thinking=false / reasoning parsers.
    The real answer is whatever follows the final </think>; if only an unclosed
    <think> is present, everything is reasoning and there is no usable answer."""
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        return text[closes[-1].end():]
    opened = _THINK_OPEN_RE.search(text)
    if opened:
        return text[: opened.start()]
    return text


def parse_transcript(text: str) -> dict[str, Any]:
    """Parse the pipe-delimited transcript into a {"turns": [...]} mapping.

    Format is "話者|発話" (no emotion field -- emotion labels were error-prone and
    are no longer requested). Silence is "silence|秒数" (optional 3rd note field).
    Coercion/validation is deferred to dialogue_from_mapping."""
    text = strip_thinking(text)
    turns: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        speaker, sep, rest = line.partition("|")
        if not sep:
            continue
        speaker = speaker.strip().lower()
        rest = rest.strip()
        if speaker == "silence":
            dur, _, note = rest.partition("|")
            turns.append(
                {
                    "speaker": "silence",
                    "duration_sec": dur.strip(),
                    "note": note.strip(),
                }
            )
            continue
        turn: dict[str, Any] = {"speaker": speaker}
        match = _TRANSCRIPT_TAG_RE.match(rest)
        if match:
            turn.update(_parse_transcript_tags(match.group("tags")))
            rest = match.group("text").strip()
        turn["text"] = rest
        turns.append(turn)
    return {"turns": turns}


def dialogue_dict_to_transcript(dialogue: dict[str, Any]) -> str:
    """Serialize a stored dialogue dict to the "話者|発話" transcript (few-shot)."""
    lines: list[str] = []
    for turn in dialogue.get("turns", []):
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip().lower()
        if speaker == "silence":
            lines.append(f"silence|{turn.get('duration_sec', 2.0)}")
            continue
        tags = []
        if str(turn.get("timing") or "") == "overlap_previous":
            start = turn.get("start_after_previous_start_sec")
            if start is not None:
                tags.append(f"overlap={start}")
        if turn.get("truncate_previous_after_sec") is not None:
            tags.append(f"truncate={turn['truncate_previous_after_sec']}")
        for key in ("event", "voice_role"):
            if turn.get(key):
                tags.append(f"{key}={turn[key]}")
        if turn.get("gain") not in (None, "", 1.0):
            tags.append(f"gain={turn['gain']}")
        tag_str = f"<<{' '.join(tags)}>>" if tags else ""
        lines.append(f"{speaker}|{tag_str}{turn.get('text', '')}")
    return "\n".join(lines)


def dialogue_from_mapping(data: dict[str, Any], use_case: dict[str, Any]) -> Dialogue:
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("Dialogue JSON must contain a list field named 'turns'.")

    turns: list[DialogueTurn] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        speaker_raw = str(item.get("speaker", "")).strip().lower()
        if speaker_raw == "silence":
            dur = item.get("duration_sec")
            try:
                dur_f = float(dur) if dur is not None else 2.0
            except (TypeError, ValueError):
                dur_f = 2.0
            turns.append(DialogueTurn(
                speaker="silence",
                text="",
                duration_sec=dur_f,
                note=str(item.get("note") or "") or None,
            ))
            continue
        speaker = normalize_speaker(speaker_raw)
        text = str(item.get("text", "")).strip()
        if speaker not in {"user", "moshi"} or not text:
            continue
        emotion = item.get("emotion")
        emotion = str(emotion) if emotion else None
        timing = str(item.get("timing") or "sequential").strip().lower()
        if timing not in {"sequential", "overlap_previous"}:
            timing = "sequential"
        gain = optional_float(item.get("gain"), 1.0)
        turns.append(
            DialogueTurn(
                speaker=speaker,
                text=text,
                emotion=emotion,
                timing=timing,
                start_after_previous_start_sec=optional_float(
                    item.get("start_after_previous_start_sec")
                ),
                truncate_previous_after_sec=optional_float(
                    item.get("truncate_previous_after_sec")
                ),
                gain=max(0.0, float(gain if gain is not None else 1.0)),
                voice_role=str(item.get("voice_role") or "") or None,
                event=str(item.get("event") or "") or None,
            )
        )

    if not turns:
        raise ValueError("Dialogue contains no usable turns.")
    # 先頭が silence や moshi の場合は use_case の opening を user 発話として差し込む
    first_non_silence = next((t for t in turns if t.speaker != "silence"), None)
    if first_non_silence is None or first_non_silence.speaker != "user":
        turns.insert(
            0,
            DialogueTurn(
                speaker="user",
                text=str(use_case.get("opening", "少し話してもいいですか。")),
            ),
        )

    dialogue_id = str(data.get("id") or use_case.get("id") or "dialogue")
    return Dialogue(
        id=safe_stem(dialogue_id, "dialogue"),
        category=str(data.get("category") or use_case.get("category") or "unknown"),
        risk_level=str(data.get("risk_level") or use_case.get("risk_level") or "low"),
        title=str(data.get("title") or use_case.get("situation") or dialogue_id),
        source_use_case=str(use_case.get("id") or dialogue_id),
        turns=turns,
        generator_notes=str(data.get("generator_notes") or ""),
        duplex_task=str(
            data.get("duplex_task") or use_case.get("duplex_task") or ""
        )
        or None,
    )


class LLMDialogueGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipe = None
        self.active_task = args.llm_task
        # Guards the trace-file append so concurrent dialogue workers do not
        # interleave or corrupt lines in multi_agent_trace.jsonl.
        self._trace_lock = threading.Lock()
        self.trace_path: Path | None = None
        if getattr(args, "dialogue_generation_mode", "") == "multi-agent":
            self.trace_path = args.out_dir / "multi_agent_trace.jsonl"
            try:
                if self.trace_path.exists():
                    self.trace_path.unlink()
            except OSError:
                logger.warning("Could not reset trace file: %s", self.trace_path)

    def trace_event(self, event: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **event,
        }
        with self._trace_lock:
            append_jsonl(self.trace_path, payload)

    def _torch_dtype(self):
        if self.args.llm_dtype == "auto":
            return "auto"
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.args.llm_dtype]

    def load(self) -> None:
        if self.args.llm_backend in {
            "openai-compatible",
            "template",
            "transformers-subprocess",
        }:
            return
        if self.pipe is not None:
            return
        from transformers import pipeline

        kwargs: dict[str, Any] = {
            "model": self.args.llm_model,
            "device_map": self.args.llm_device_map,
            "trust_remote_code": self.args.trust_remote_code,
        }
        dtype = self._torch_dtype()
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype

        logger.info("Loading LLM pipeline task=%s model=%s", self.active_task, self.args.llm_model)
        try:
            self.pipe = pipeline(self.active_task, **kwargs)
        except Exception:
            if self.active_task == "text-generation":
                raise
            logger.warning("Falling back to text-generation pipeline.", exc_info=True)
            self.active_task = "text-generation"
            self.pipe = pipeline("text-generation", **kwargs)

    def generate(self, use_case: dict[str, Any], rng: random.Random) -> Dialogue:
        if self.args.llm_backend == "template":
            return template_dialogue(use_case)
        prompt = build_llm_prompt(use_case, rng, self.args.llm_num_fewshot)

        if self.args.llm_backend == "openai-compatible":
            text = self.generate_openai_compatible(prompt)
            data = parse_transcript(text)
            return dialogue_from_mapping(data, use_case)

        if self.args.llm_backend == "transformers-subprocess":
            text = self.generate_transformers_subprocess(prompt)
            data = parse_transcript(text)
            return dialogue_from_mapping(data, use_case)

        self.load()
        assert self.pipe is not None
        generation_kwargs = {
            "max_new_tokens": self.args.llm_max_new_tokens,
            "temperature": self.args.llm_temperature,
            "do_sample": True,
            "return_full_text": False,
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            result = self.pipe(messages, **generation_kwargs)
        except TypeError:
            result = self.pipe(prompt, **generation_kwargs)
        text = pipeline_result_to_text(result)
        data = parse_transcript(text)
        return dialogue_from_mapping(data, use_case)

    def resolve_served_model(self) -> str:
        """Verify the endpoint actually serves the configured model.

        These vLLM jobs serve a single model. We query /v1/models once and, if the
        configured model is NOT among the served ids, we raise instead of silently
        switching: a mismatch means requests are hitting the wrong server (e.g. a
        stale gpt-oss vLLM still bound to the same host:port as a Qwen job), and
        generating the whole dataset with the wrong model is far worse than failing
        fast. A benign id-format difference (e.g. trailing slash) is tolerated by
        basename comparison. If the lookup itself fails, fall back to the
        configured model rather than blocking on a transient network error."""
        cached = getattr(self, "_served_model", None)
        if cached is not None:
            return cached
        configured = self.args.llm_model
        models_url = openai_models_url(self.args.llm_api_base)
        try:
            request = urllib.request.Request(models_url, method="GET")
            if self.args.llm_api_key:
                request.add_header("Authorization", f"Bearer {self.args.llm_api_key}")
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            ids = [str(item["id"]) for item in data.get("data", []) if item.get("id")]
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            logger.warning(
                "Could not query %s (%s); using configured model %r.",
                models_url, exc, configured,
            )
            self._served_model = configured
            return configured

        def _base(name: str) -> str:
            return name.rsplit("/", 1)[-1].strip().lower()

        if ids and not any(_base(m) == _base(configured) for m in ids):
            raise RuntimeError(
                f"Endpoint {models_url} serves {ids} but the job is configured for "
                f"{configured!r}. Requests are hitting the wrong server -- likely a "
                f"stale vLLM still bound to this host:port. Stop the stale server or "
                f"use a distinct PORT for this job; refusing to generate with the "
                f"wrong model."
            )
        # Use the exact served id (handles benign id-format differences).
        served = next((m for m in ids if _base(m) == _base(configured)), configured)
        self._served_model = served
        return served

    def generate_openai_compatible_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: str | None = None,
        json_schema: dict[str, Any] | None = None,
        json_schema_name: str = "dialogue",
        role_name: str = "llm",
    ) -> str:
        url = openai_generation_url(
            self.args.llm_api_base,
            self.args.llm_openai_endpoint,
        )
        payload = {
            "model": self.resolve_served_model(),
            "temperature": self.args.llm_temperature if temperature is None else temperature,
            "frequency_penalty": (
                self.args.llm_frequency_penalty
                if frequency_penalty is None
                else frequency_penalty
            ),
            "presence_penalty": (
                self.args.llm_presence_penalty
                if presence_penalty is None
                else presence_penalty
            ),
            "max_tokens": self.args.llm_max_new_tokens if max_tokens is None else max_tokens,
        }
        if self.args.llm_openai_endpoint == "chat":
            payload["messages"] = messages
        else:
            payload["prompt"] = messages_to_completion_prompt(
                messages,
                no_think=self.args.llm_completions_no_think,
            )
            payload["stop"] = ["<|im_end|>"]
        # vLLM extension: multiplicative repetition penalty. Strongly discourages
        # the "\n\n\n..." degeneration that frequency/presence penalties alone do
        # not stop. Only send when enabled so other servers are unaffected.
        if self.args.llm_repetition_penalty and self.args.llm_repetition_penalty > 1.0:
            payload["repetition_penalty"] = self.args.llm_repetition_penalty
        # Reasoning models (gpt-oss) otherwise spend the whole token budget on the
        # analysis channel, leaving message.content empty -> we'd fall back to
        # reasoning_content, which is not JSON. Lower the effort to reach the
        # final channel.
        if self.args.llm_reasoning_effort:
            payload["reasoning_effort"] = self.args.llm_reasoning_effort
        # Force structured JSON output via vLLM's response_format support.
        # json_schema constrains the exact dialogue shape (no newline wandering,
        # always a closed object); json_object only requires syntactically valid
        # JSON.
        if (
            self.args.llm_openai_endpoint == "chat"
            and response_format == "json_schema"
            and json_schema is not None
        ):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema_name,
                    "schema": json_schema,
                },
            }
        elif self.args.llm_openai_endpoint == "chat" and response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.args.llm_api_key:
            headers["Authorization"] = f"Bearer {self.args.llm_api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            t0 = time.time()
            with urllib.request.urlopen(
                request,
                timeout=self.args.llm_timeout_sec,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - t0
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach the OpenAI-compatible LLM endpoint at "
                f"{url}. Start a LLM server separately, pass --llm-api-base, "
                "or use --dialogues-jsonl / --llm-backend template."
            ) from exc
        logger.info(
            "%s raw_response: %s",
            role_name,
            json.dumps(data, ensure_ascii=False),
        )
        usage = data.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens and elapsed > 0:
            logger.info(
                "%s throughput: %.1f tokens/sec (%d tokens in %.2fs)",
                role_name,
                completion_tokens / elapsed,
                completion_tokens,
                elapsed,
            )
        return openai_chat_response_to_text(data)

    def generate_openai_compatible(self, prompt: str) -> str:
        # The full instructions are already embedded at the top of `prompt`
        # (build_llm_prompt). Sending them again as a system message would
        # duplicate ~all of the guidance, so keep the system role minimal.
        return self.generate_openai_compatible_messages(
            [
                {"role": "system", "content": "日本語の対話データ作成アシスタント。"},
                {"role": "user", "content": prompt},
            ],
            response_format=self.args.llm_response_format,
            json_schema=dialogue_response_schema(),
            role_name="single-dialogue",
        )

    def call_agent(
        self,
        prompt: AgentPrompt,
        *,
        temperature: float,
        max_tokens: int,
        role_name: str,
    ) -> str:
        if self.args.llm_backend != "openai-compatible":
            raise RuntimeError(
                "multi-agent dialogue generation requires "
                "--llm-backend openai-compatible so each agent can be called "
                "through the same vLLM/OpenAI-compatible endpoint."
            )
        return self.generate_openai_compatible_messages(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format=None,
            role_name=role_name,
        )

    def call_agent_utterance(
        self,
        prompt: AgentPrompt,
        *,
        speaker: str,
        temperature: float,
        max_tokens: int,
        role_name: str,
        fallback_text: str,
        context: dict[str, Any],
    ) -> str:
        current = prompt
        last_raw = ""
        for attempt in range(2):
            raw = self.call_agent(
                current,
                temperature=temperature,
                max_tokens=max_tokens,
                role_name=role_name if attempt == 0 else f"{role_name}-retry",
            )
            last_raw = raw
            text = clean_agent_utterance(raw, speaker)
            self.trace_event(
                {
                    "event": "agent_utterance",
                    "role": role_name,
                    "speaker": speaker,
                    "attempt": attempt + 1,
                    "empty": not bool(text),
                    "raw_contains_think": bool(_THINK_OPEN_RE.search(raw)),
                    "prompt_system": current.system,
                    "prompt_user": current.user,
                    "raw_text": raw[:1000],
                    "cleaned_text": text,
                    **context,
                }
            )
            if text:
                return text
            logger.warning(
                "%s produced an empty utterance on attempt %d; raw=%r",
                role_name,
                attempt + 1,
                raw[:500],
            )
            current = AgentPrompt(
                system=prompt.system,
                user=(
                    prompt.user
                    + "\n\n前回の出力が空でした。必ず1文以上の発話本文だけを返してください。"
                ),
            )
        if self.args.multi_agent_empty_policy == "fallback":
            logger.warning("%s fallback utterance used: %s", role_name, fallback_text)
            self.trace_event(
                {
                    "event": "agent_fallback",
                    "role": role_name,
                    "speaker": speaker,
                    "fallback_text": fallback_text,
                    "last_raw_text": last_raw[:1000],
                    **context,
                }
            )
            return fallback_text
        self.trace_event(
            {
                "event": "agent_empty_failure",
                "role": role_name,
                "speaker": speaker,
                "last_raw_text": last_raw[:1000],
                **context,
            }
        )
        raise ValueError(
            f"{role_name} produced an empty utterance after retry "
            f"(case={context.get('case_id')} pair={context.get('pair_index')}). "
            f"See {self.trace_path} for raw output."
        )

    def generate_multi_agent(
        self,
        use_case: dict[str, Any],
        rng: random.Random,
    ) -> Dialogue:
        if self.args.llm_backend == "template":
            return template_dialogue(use_case)
        if self.args.llm_backend != "openai-compatible":
            raise RuntimeError(
                "--dialogue-generation-mode multi-agent currently requires "
                "--llm-backend openai-compatible (vLLM/OpenAI-style server)."
            )

        profile = build_user_agent_profile(use_case, rng)
        max_pairs = max(1, int(self.args.multi_agent_max_pairs))
        min_pairs = max(1, min(int(self.args.multi_agent_min_pairs), max_pairs))
        core_turns: list[DialogueTurn] = []
        surface_turns: list[DialogueTurn] = []
        case_id = str(use_case.get("id") or "multi_agent_dialogue")
        self.trace_event(
            {
                "event": "dialogue_start",
                "case_id": case_id,
                "profile": profile,
                "min_pairs": min_pairs,
                "max_pairs": max_pairs,
                "empty_policy": self.args.multi_agent_empty_policy,
                "aizuchi_mode": self.args.multi_agent_aizuchi_mode,
            }
        )

        for pair_index in range(max_pairs):
            context = {"case_id": case_id, "pair_index": pair_index + 1}
            user_prompt = build_user_agent_prompt(
                use_case, profile, core_turns, pair_index, max_pairs
            )
            user_text = self.call_agent_utterance(
                user_prompt,
                speaker="user",
                temperature=self.args.multi_agent_user_temperature,
                max_tokens=240,
                role_name="userAI",
                fallback_text=fallback_user_utterance(use_case, core_turns),
                context=context,
            )
            logger.info(
                "multi-agent case=%s pair=%d user=%s",
                case_id,
                pair_index + 1,
                user_text,
            )
            user_turn = DialogueTurn("user", user_text)
            core_turns.append(user_turn)

            moshi_prompt = build_moshi_agent_prompt(use_case, core_turns)
            moshi_text = self.call_agent_utterance(
                moshi_prompt,
                speaker="moshi",
                temperature=self.args.multi_agent_system_temperature,
                max_tokens=280,
                role_name="systemAI",
                fallback_text=fallback_moshi_utterance(core_turns),
                context=context,
            )
            moshi_turn = DialogueTurn("moshi", moshi_text)
            core_turns.append(moshi_turn)
            surface_pair_turns = self.add_aizuchi_for_user_turn(
                use_case=use_case,
                visible_turns=surface_turns,
                user_turn=user_turn,
                moshi_turn=moshi_turn,
                case_id=case_id,
                pair_index=pair_index + 1,
            )
            surface_pair_turns.append(moshi_turn)
            surface_turns.extend(surface_pair_turns)
            if self.args.multi_agent_aizuchi_mode == "inline":
                core_turns = list(surface_turns)
            logger.info(
                "multi-agent case=%s pair=%d moshi=%s",
                case_id,
                pair_index + 1,
                moshi_text,
            )
            self.trace_event(
                {
                    "event": "pair_complete",
                    "core_turn_count": len(core_turns),
                    "surface_turn_count": len(surface_turns),
                    "core_transcript": dialogue_turns_to_transcript(core_turns),
                    "surface_transcript": dialogue_turns_to_transcript(surface_turns),
                    **context,
                }
            )

            if pair_index + 1 >= min_pairs:
                judge_prompt = build_judge_agent_prompt(
                    use_case, core_turns, pair_index, min_pairs, max_pairs
                )
                judge_raw = self.call_agent(
                    judge_prompt,
                    temperature=self.args.multi_agent_judge_temperature,
                    max_tokens=120,
                    role_name="judgeAI",
                )
                complete = parse_completion_decision(judge_raw)
                self.trace_event(
                    {
                        "event": "judge_decision",
                        "prompt_system": judge_prompt.system,
                        "prompt_user": judge_prompt.user,
                        "raw_text": judge_raw[:1000],
                        "complete": complete,
                        **context,
                    }
                )
                if complete:
                    break

        dialogue_id = case_id
        self.trace_event(
            {
                "event": "dialogue_complete",
                "case_id": case_id,
                "core_turn_count": len(core_turns),
                "surface_turn_count": len(surface_turns),
                "core_transcript": dialogue_turns_to_transcript(core_turns),
                "surface_transcript": dialogue_turns_to_transcript(surface_turns),
            }
        )
        return Dialogue(
            id=safe_stem(dialogue_id, "multi_agent_dialogue"),
            category=str(use_case.get("category", "unknown")),
            risk_level=str(use_case.get("risk_level", "low")),
            title=str(use_case.get("situation") or dialogue_id),
            source_use_case=str(use_case.get("id") or dialogue_id),
            turns=surface_turns,
            generator_notes=(
                "multi_agent profile="
                + json.dumps(profile, ensure_ascii=False, sort_keys=True)
            ),
            duplex_task=str(use_case.get("duplex_task") or "") or None,
        )

    def add_aizuchi_for_user_turn(
        self,
        *,
        use_case: dict[str, Any],
        visible_turns: list[DialogueTurn],
        user_turn: DialogueTurn,
        moshi_turn: DialogueTurn,
        case_id: str,
        pair_index: int,
    ) -> list[DialogueTurn]:
        if (
            self.args.no_multi_agent_aizuchi
            or len(user_turn.text) < self.args.multi_agent_aizuchi_min_chars
            or self.args.multi_agent_max_aizuchi_per_user <= 0
        ):
            return [user_turn]
        clauses = split_text_into_clauses(user_turn.text)
        if len(clauses) < 2:
            return [user_turn]
        prompt = build_aizuchi_agent_prompt(
            use_case,
            visible_turns,
            user_turn.text,
            moshi_turn.text,
            self.args.multi_agent_max_aizuchi_per_user,
        )
        raw = self.call_agent(
            prompt,
            temperature=self.args.multi_agent_aizuchi_temperature,
            max_tokens=120,
            role_name="aizuchiAI",
        )
        insertions = parse_aizuchi_insertions(
            raw,
            clauses,
            self.args.multi_agent_max_aizuchi_per_user,
        )
        self.trace_event(
            {
                "event": "aizuchi_decision",
                "case_id": case_id,
                "pair_index": pair_index,
                "aizuchi_mode": self.args.multi_agent_aizuchi_mode,
                "prompt_system": prompt.system,
                "prompt_user": prompt.user,
                "raw_text": raw[:1000],
                "insertions": insertions,
            }
        )
        return user_turns_with_aizuchi(user_turn.text, insertions)

    def generate_transformers_subprocess(self, prompt: str) -> str:
        return self.generate_transformers_subprocess_batch([prompt])[0]

    def generate_transformers_subprocess_batch(self, prompts: list[str]) -> list[str]:
        python_exe = resolve_llm_python(self.args)
        worker = Path(self.args.llm_worker)
        if not worker.is_file():
            raise FileNotFoundError(f"LLM worker script not found: {worker}")

        command = [
            python_exe,
            str(worker),
            "--model",
            self.args.llm_model,
            "--task",
            self.args.llm_task,
            "--device-map",
            self.args.llm_device_map,
            "--dtype",
            self.args.llm_dtype,
            "--temperature",
            str(self.args.llm_temperature),
            "--max-new-tokens",
            str(self.args.llm_max_new_tokens),
        ]
        if self.args.trust_remote_code:
            command.append("--trust-remote-code")

        payload = json.dumps({"prompts": prompts}, ensure_ascii=False)
        try:
            proc = subprocess.run(
                command,
                input=payload,
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=None,  # stderrは親プロセスのターミナルにそのまま流す
                timeout=self.args.llm_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"LLM subprocess timed out after {self.args.llm_timeout_sec}s: "
                f"{' '.join(command)}"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                "LLM subprocess failed. Install the LLM runtime with "
                "`uv sync --project gemma_runtime`, or pass --llm-python to "
                "a Python environment that has Transformers/LLM installed.\n"
                f"command: {' '.join(command)}\n"
                "(stderr output is shown above)"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "LLM subprocess did not return JSON.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            ) from exc
        texts = data.get("texts")
        if not isinstance(texts, list):
            raise RuntimeError(f"LLM worker returned no 'texts' list: {data}")
        return [str(text) for text in texts]


def resolve_llm_python(args: argparse.Namespace) -> str:
    if args.llm_python:
        return str(args.llm_python)

    runtime_dir = REPO_ROOT / "gemma_runtime" / ".venv"
    candidates = [
        runtime_dir / "bin" / "python",
        runtime_dir / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def openai_v1_base_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/models"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def openai_models_url(api_base: str) -> str:
    return f"{openai_v1_base_url(api_base)}/models"


def openai_generation_url(api_base: str, endpoint: str) -> str:
    suffix = "chat/completions" if endpoint == "chat" else "completions"
    return f"{openai_v1_base_url(api_base)}/{suffix}"


def chat_completions_url(api_base: str) -> str:
    return openai_generation_url(api_base, "chat")


def messages_to_completion_prompt(
    messages: list[dict[str, str]],
    *,
    no_think: bool = False,
) -> str:
    last_user_index = -1
    if no_think:
        for index, message in enumerate(messages):
            if str(message.get("role", "")).strip() == "user":
                last_user_index = index
    parts: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if no_think and index == last_user_index and "/no_think" not in content:
            content = f"{content}\n\n/no_think".strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def openai_chat_response_to_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"OpenAI-compatible response has no choices: {data}")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            joined = "".join(parts)
            if joined:
                return joined
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            # The final channel was empty: the model spent its whole budget on
            # reasoning. This text is the analysis channel, not the requested
            # JSON. Surface it loudly instead of silently returning thinking.
            logger.warning(
                "Empty content channel; falling back to reasoning_content "
                "(not JSON). Lower --llm-reasoning-effort or raise "
                "--llm-max-new-tokens."
            )
            return reasoning_content
    text = choice.get("text") if isinstance(choice, dict) else None
    if isinstance(text, str):
        return text
    raise ValueError(f"Could not extract text from OpenAI-compatible response: {data}")


def pipeline_result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        if not result:
            return ""
        if len(result) == 1:
            return pipeline_result_to_text(result[0])
        return "\n".join(pipeline_result_to_text(item) for item in result)
    if isinstance(result, dict):
        for key in ("generated_text", "text", "content"):
            if key in result:
                return pipeline_result_to_text(result[key])
        return json.dumps(result, ensure_ascii=False)
    return str(result)


USER_AGENT_SYSTEM_PROMPT = """
あなたは合成対話データの userAI です。孤独・孤立相談窓口に来る相談者 user として発話します。
会話のランダム性、話題の出し方、ためらい、感情の揺れはあなたが担当します。
相談者らしく一度に全部を説明せず、少しずつ開示してください。
出力は user の次の発話本文だけ。話者名、JSON、説明、引用符、箇条書きは出力しません。
""".strip()


MOSHI_AGENT_SYSTEM_PROMPT = """
あなたは日本語の雑談傾聴システム moshi の聞き手です。徹底して傾聴します。
丁寧語で、相手の言葉や気持ちを受け止め、短く言い換えたり感情を承認したりして返します。
基本動作は「相手の言ったことを聞き返す・気持ちを言葉にして返す」こと。
1発話は必ず1文だけ。長く話しません。傾聴的な1文にします。
助言・解決策・評価・話題転換はしません。質問で促すより、まず受け止めることを優先します。
質問するとしても、相手が話しやすくなる短い問いをときどき添える程度にします。
「うん」「うんうん」などのくだけた相づちは使いません。
出力は moshi の次の発話本文だけ。話者名、JSON、説明、引用符、箇条書きは出力しません。
""".strip()


JUDGE_AGENT_SYSTEM_PROMPT = """
あなたは合成相談対話の外部判定器です。
会話が自然に一区切りしているかだけを判定します。
JSONだけを出力してください: {"complete": true/false, "reason": "短い理由"}
""".strip()


AIZUCHI_AGENT_SYSTEM_PROMPT = """
あなたは雑談傾聴システム moshi の聞き手です。相手が話している途中に短い相づちを打つ位置を決めます。
雑談を聞くときと同じように、相手の話に合わせてこまめに相づちを入れてください。
相づちは丁寧で短いものだけにします: はい。/ええ。/そうなんですね。/なるほど。
JSONだけを出力してください: {"insertions":[{"after_clause": 1, "text": "はい。"}]}
""".strip()


def case_brief(use_case: dict[str, Any]) -> str:
    talking_points = use_case.get("talking_points") or []
    if isinstance(talking_points, list):
        points = " / ".join(str(p) for p in talking_points if str(p).strip())
    else:
        points = str(talking_points)
    lines = [
        f"id: {use_case.get('id', 'unknown')}",
        f"category: {use_case.get('category', 'unknown')}",
        f"risk_level: {use_case.get('risk_level', 'low')}",
        f"situation: {use_case.get('situation', '')}",
        f"user_profile: {use_case.get('user_profile', '')}",
        f"opening_hint: {use_case.get('opening', '')}",
        f"topic: {use_case.get('topic', '')}",
        f"talking_points: {points}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


def build_user_agent_profile(
    use_case: dict[str, Any], rng: random.Random
) -> dict[str, str]:
    talking_points = use_case.get("talking_points") or []
    if isinstance(talking_points, list):
        ordered_points = [str(p) for p in talking_points if str(p).strip()]
        rng.shuffle(ordered_points)
    else:
        ordered_points = []
    return {
        "opening_style": rng.choice(
            [
                "最初は控えめに話してよいか確認する",
                "日常の出来事から自然に入り、後半で本音が出る",
                "少し明るく振る舞うが、途中で寂しさがにじむ",
                "言葉を探しながら短めに話し始める",
            ]
        ),
        "disclosure_pace": rng.choice(
            [
                "一度に説明しすぎず、質問されると少し詳しく話す",
                "感情より先に具体的な出来事を話す",
                "本題に触れたり戻ったりしながら話す",
                "相手の受け止めで少しずつ安心していく",
            ]
        ),
        "speech_texture": rng.choice(
            [
                "短い文が多く、少し言い淀む",
                "丁寧だが、ところどころ本音がこぼれる",
                "沈黙を避けようとして話題を足す",
                "感情を直接言うのが苦手で比喩的に話す",
            ]
        ),
        "emotional_arc": rng.choice(
            [
                "不安が少し和らぐが、完全には解決しない",
                "話せたことで少し整理される",
                "安心して、最後にまた相談してよいか確認する",
                "重さは残るが、聞かれたことへの感謝が出る",
            ]
        ),
        "topic_order": " / ".join(ordered_points[:4]) or "自然な流れで話題を選ぶ",
    }


def dialogue_turns_to_transcript(turns: list[DialogueTurn]) -> str:
    rows = [asdict(turn) for turn in turns]
    return dialogue_dict_to_transcript({"turns": rows})


def build_user_agent_prompt(
    use_case: dict[str, Any],
    profile: dict[str, str],
    turns: list[DialogueTurn],
    pair_index: int,
    max_pairs: int,
) -> AgentPrompt:
    transcript = dialogue_turns_to_transcript(turns) or "(まだ会話は始まっていません)"
    prompt = f"""
相談ケース:
{case_brief(use_case)}

今回の userAI のランダム設定:
- 出だし: {profile['opening_style']}
- 開示ペース: {profile['disclosure_pace']}
- 話し方: {profile['speech_texture']}
- 感情の流れ: {profile['emotional_arc']}
- 話題順の手がかり: {profile['topic_order']}

これまでの会話:
{transcript}

次に user として1発話だけ返してください。
条件:
- {pair_index + 1}/{max_pairs} 回目の user 発話です。
- 相談者として自然に、短すぎず長すぎず 1〜3 文で話します。
- 最終盤なら、少し区切りがつく発話や感謝も自然に入れてよいです。
- moshi の発話、話者名、相づち指示、JSONは書きません。
""".strip()
    return AgentPrompt(system=USER_AGENT_SYSTEM_PROMPT, user=prompt)


def build_moshi_agent_prompt(
    use_case: dict[str, Any], turns: list[DialogueTurn]
) -> AgentPrompt:
    transcript = dialogue_turns_to_transcript(turns)
    prompt = f"""
相談ケース:
{case_brief(use_case)}

これまでの会話:
{transcript}

次に moshi として1発話だけ返してください。
条件:
- 必ず1文だけ。短く、長く話しません。
- 直前の相手の発話を受け止め、その言葉や気持ちを短く言い換えて返すのを基本にします。
- 感情がにじむ発話には、その気持ちを言葉にして承認します（例:「さみしいですよね。」）。
- 助言・解決策・評価・話題転換はしません。受け止めだけで終えてかまいません。
- 問いを添えるのは、相手が話しやすくなるときだけ短く1つ。毎回は入れません。
- high/medium risk では断定せず、必要な場合だけ短く安全確認します。
- 「うん」「うんうん」は使いません。
- 話者名、JSON、説明は書きません。
""".strip()
    return AgentPrompt(system=MOSHI_AGENT_SYSTEM_PROMPT, user=prompt)


def build_judge_agent_prompt(
    use_case: dict[str, Any],
    turns: list[DialogueTurn],
    pair_index: int,
    min_pairs: int,
    max_pairs: int,
) -> AgentPrompt:
    transcript = dialogue_turns_to_transcript(turns)
    prompt = f"""
相談ケース:
{case_brief(use_case)}

これまでの会話:
{transcript}

判定基準:
- 相談者が少し受け止められた、整理できた、または次に話す余地を持って自然に一区切りしていれば complete=true。
- 唐突な終了、相談員の一方的な助言、相談者の強い未完了感が残る場合は complete=false。
- まだ {min_pairs} 往復未満なら complete=false。
- {max_pairs} 往復に達したら、破綻していなければ complete=true に寄せる。

現在の往復数: {pair_index + 1}
JSONだけを返してください。
""".strip()
    return AgentPrompt(system=JUDGE_AGENT_SYSTEM_PROMPT, user=prompt)


def split_text_into_clauses(text: str) -> list[str]:
    parts = re.findall(r".*?[、。！？!?]|.+$", text.strip())
    return [part.strip() for part in parts if part.strip()]


def build_aizuchi_agent_prompt(
    use_case: dict[str, Any],
    turns: list[DialogueTurn],
    user_text: str,
    moshi_text: str,
    max_insertions: int,
) -> AgentPrompt:
    transcript = dialogue_turns_to_transcript(turns) or "(まだ会話は始まっていません)"
    clauses = split_text_into_clauses(user_text)
    numbered = "\n".join(f"{i + 1}: {clause}" for i, clause in enumerate(clauses))
    prompt = f"""
相談ケース:
{case_brief(use_case)}

直前までの会話:
{transcript}

今回の user 発話を句ごとに分けたもの:
{numbered}

今回の moshi 本応答:
{moshi_text}

最大挿入数: {max_insertions}
判断:
- 雑談を聞くように、話の途中でこまめに相づちを打ちます。最大挿入数まで入れてよいです。
- user発話の途中に moshi の短い相づちを入れる位置を after_clause に句番号で指定します。
- 最後の句の後には挿入しません。
- moshi 本応答の直前に同じ語が重ならないようにします。
- text は必ず「はい。」「ええ。」「そうなんですね。」「なるほど。」のどれか。
JSONだけを返してください。
""".strip()
    return AgentPrompt(system=AIZUCHI_AGENT_SYSTEM_PROMPT, user=prompt)


def latest_turn_text(turns: list[DialogueTurn], speaker: str) -> str:
    for turn in reversed(turns):
        if turn.speaker == speaker and turn.text.strip():
            return turn.text.strip()
    return ""


def fallback_user_utterance(
    use_case: dict[str, Any], turns: list[DialogueTurn]
) -> str:
    if not any(turn.speaker == "user" for turn in turns):
        return str(use_case.get("opening") or "少し話してもいいですか。")
    return "はい。少しずつですが、もう少し話してみます。"


def fallback_moshi_utterance(turns: list[DialogueTurn]) -> str:
    last_user = latest_turn_text(turns, "user")
    if not last_user:
        return "来てくださってありがとうございます。少しずつで大丈夫です。"
    short = last_user
    if len(short) > 38:
        short = short[:38].rstrip("、。…") + "…"
    return (
        "話してくださってありがとうございます。"
        f"{short}ということなんですね。"
        "よければ、もう少し聞かせてください。"
    )


def clean_agent_utterance(text: str, speaker: str) -> str:
    text = strip_thinking(text).strip()
    text = re.sub(r"<\|im_start\|>\s*assistant\s*", "", text)
    text = text.replace("<|im_end|>", "").strip()
    lines = [
        line.strip().strip('"').strip("'")
        for line in text.splitlines()
        if line.strip()
        and not line.strip().startswith("```")
        and line.strip() not in {"<|im_start|>assistant", "<|im_end|>"}
    ]
    if not lines:
        return ""
    line = lines[0]
    if "|" in line:
        left, _, right = line.partition("|")
        if normalize_speaker(left.strip().lower()) == speaker:
            line = right.strip()
    line = re.sub(r"^(user|moshi|相談者|相談員)\s*[:：]\s*", "", line).strip()
    return line.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_thinking(text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found: {text}")
    return json.loads(cleaned[start : end + 1])


def parse_completion_decision(text: str) -> bool:
    try:
        data = extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        return False
    value = data.get("complete")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "complete"}
    return False


def parse_aizuchi_insertions(
    text: str,
    clauses: list[str],
    max_insertions: int,
) -> list[dict[str, Any]]:
    if len(clauses) < 2 or max_insertions <= 0:
        return []
    try:
        data = extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        return []
    raw = data.get("insertions")
    if not isinstance(raw, list):
        return []
    allowed = {"はい。", "ええ。", "そうなんですね。", "なるほど。"}
    out: list[dict[str, Any]] = []
    used: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            after_clause = int(item.get("after_clause"))
        except (TypeError, ValueError):
            continue
        text_value = str(item.get("text") or "").strip()
        if text_value not in allowed:
            continue
        if after_clause < 1 or after_clause >= len(clauses):
            continue
        if after_clause in used:
            continue
        used.add(after_clause)
        out.append({"after_clause": after_clause, "text": text_value})
        if len(out) >= max_insertions:
            break
    return sorted(out, key=lambda item: int(item["after_clause"]))


def user_turns_with_aizuchi(
    user_text: str,
    insertions: list[dict[str, Any]],
) -> list[DialogueTurn]:
    clauses = split_text_into_clauses(user_text)
    if not clauses or not insertions:
        return [DialogueTurn("user", user_text)]

    turns: list[DialogueTurn] = []
    start = 0
    for insertion in insertions:
        after_clause = int(insertion["after_clause"])
        chunk = "".join(clauses[start:after_clause]).strip()
        if chunk:
            turns.append(DialogueTurn("user", chunk))
        turns.append(DialogueTurn("moshi", str(insertion["text"])))
        start = after_clause
    rest = "".join(clauses[start:]).strip()
    if rest:
        turns.append(DialogueTurn("user", rest))
    return turns or [DialogueTurn("user", user_text)]


def multi_agent_prompt_preview(
    use_case: dict[str, Any],
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[str, Any]:
    profile = build_user_agent_profile(use_case, rng)
    max_pairs = max(1, args.multi_agent_max_pairs)
    turns: list[DialogueTurn] = []
    user_prompt = build_user_agent_prompt(use_case, profile, turns, 0, max_pairs)
    sample_user = DialogueTurn("user", str(use_case.get("opening", "少し話してもいいですか。")))
    moshi_prompt = build_moshi_agent_prompt(use_case, [sample_user])
    judge_prompt = build_judge_agent_prompt(
        use_case,
        [sample_user, DialogueTurn("moshi", "話してくださってありがとうございます。")],
        0,
        args.multi_agent_min_pairs,
        max_pairs,
    )
    aizuchi_prompt = build_aizuchi_agent_prompt(
        use_case,
        [],
        sample_user.text,
        "話してくださってありがとうございます。",
        args.multi_agent_max_aizuchi_per_user,
    )
    return {
        "profile": profile,
        "userAI": asdict(user_prompt),
        "systemAI": asdict(moshi_prompt),
        "judge": asdict(judge_prompt),
        "aizuchi": asdict(aizuchi_prompt),
    }


def build_duplex_prompt_section(use_case: dict[str, Any]) -> str:
    task = str(use_case.get("duplex_task") or "").strip()
    if not task:
        return ""

    # Timing for duplex turns is written as a leading <<...>> tag on the text,
    # e.g.  moshi|gentle|<<overlap=1.5 event=model_backchannel>>はい。
    task_rules = {
        "pause_handling": (
            "user発話を前半、2〜3秒のsilence、後半に分ける。"
            "silence行は silence|2.5 の形で書く。"
            "後半が終わる前にmoshiの長い返答を置かない。"
        ),
        "smooth_turn_taking": (
            "通常の交互対話にする。overlap タグは使わず、すべて素直に交互に並べる。"
        ),
        "backchannel": (
            "長めのuser発話の直後に短いmoshi相槌を置き、"
            "その相槌の text 先頭へ <<overlap=1.0〜2.5 event=model_backchannel>> を付ける。"
            "その後userが続きを話す。"
        ),
        "user_interruption": (
            "moshiのやや長い発話の直後にuserの訂正または重要情報を置き、"
            "そのuser行の text 先頭へ "
            "<<overlap=0.8〜1.8 truncate=0.15〜0.35 event=user_interruption>> を付ける。"
            "その次にmoshiが訂正内容へ応答する。"
        ),
        "user_backchannel": (
            "moshiの説明の直後にuserの「はい」「うん」などを置き、"
            "そのuser行の text 先頭へ <<overlap=0.8〜1.8 event=user_backchannel>> を付ける。"
            "truncate は付けず、次のmoshi発話で同じ説明を自然に続ける。"
        ),
        "talking_to_other": (
            "moshiの発話の直後にuserが近くの人へ向けた短い発話を置き、"
            "そのuser行の text 先頭へ "
            "<<overlap=0.8〜1.8 voice_role=user event=talking_to_other>> を付ける。"
            "次のmoshi発話は横の人への発話へ直接回答せず、元の相談を続ける。"
        ),
        "background_speech": (
            "moshiの発話の直後にニュースまたは駅案内の短い背景発話をuser行として置き、"
            "その背景行の text 先頭へ "
            "<<overlap=0.8〜1.8 voice_role=background gain=0.3 event=background_speech>> を付ける。"
            "次のmoshi発話は背景内容へ反応せず、元の相談を続ける。"
        ),
    }
    rule = task_rules.get(task)
    if rule is None:
        return ""
    return f"""

full-duplex学習パターン:
- duplex_task: {task}
- {rule}
- <<overlap=...>> は必ず直前の行に重なる。
- タグの無い行は通常の交互ターンとして扱う。
""".rstrip()


def build_llm_prompt(
    use_case: dict[str, Any], rng: random.Random, num_fewshot: int = 1
) -> str:
    turn_count = int(use_case.get("target_turns", rng.choice([6, 8, 10])))
    user_style = rng.choice(
        [
            "言い淀みが少しある",
            "短い文が多い",
            "最初は雑談っぽい",
            "気持ちをうまく言葉にできない",
            "具体的な出来事から話し始め、あとから気持ちが見えてくる",
            "明るく振る舞うが、ところどころ本音がにじむ",
            "話題が少し前後し、相談員が要点を拾う必要がある",
            "結論を急がず、日常の細部をぽつぽつ話す",
        ]
    )
    counselor_style = rng.choice(
        [
            "温かく、でも踏み込みすぎない",
            "丁寧な相づちを自然に入れる",
            "解決策を急がず、気持ちを整理する",
            "一問一答になりすぎず会話を続ける",
            "短い受け止め、言い換え、問いかけの順序を毎回変える",
            "相手の語彙を一部受け取り、感情を丁寧に言葉にする",
            "雑談では一緒に喜び、重い話では声量を落とすような語調にする",
        ]
    )
    opening_style = rng.choice(
        [
            "userが出来事を直接話し始める",
            "userが話してよいか確かめてから本題に入る",
            "何気ない近況や季節の話から本音へ移る",
            "userが感情ではなく身体感覚や生活の変化から話す",
            "userが嬉しかった小さな出来事を共有する",
            "userが言葉を探しながら断片的に話す",
        ]
    )
    response_structure = rng.choice(
        [
            "相槌→感情の言語化→短い開かれた問い",
            "言い換え→受容→余白を残す一言",
            "具体的内容の要約→感情への共感→必要なら確認",
            "短い共感→相手の言葉の反映→話したい方向を尋ねる",
            "驚きや喜びの共有→具体点への関心→自然な雑談の継続",
        ]
    )

    personality = str(use_case.get("personality") or "ふつう")
    personality_guidance = str(use_case.get("personality_guidance") or "自然に話す。")
    emotional_state = str(use_case.get("emotional_state") or "落ち着いている")
    emotional_state_hint = str(use_case.get("emotional_state_hint") or "")

    talking_points = use_case.get("talking_points") or []
    if isinstance(talking_points, list) and talking_points:
        points_line = (
            "- この対話で user が触れる具体的な話題(自然に会話へ織り込む。全部使わなくてよい): "
            + " / ".join(str(p) for p in talking_points)
            + "\n"
        )
    else:
        points_line = ""

    duplex_task = str(use_case.get("duplex_task") or "")
    silence_pattern = str(use_case.get("silence_pattern", "none"))
    if duplex_task == "pause_handling":
        silence_directive = (
            "- 2〜3秒の沈黙を silence|2.5 の形で入れ、その直後はuserが同じ発話を続ける。"
        )
    elif silence_pattern == "heavy":
        silence_directive = (
            "- 沈黙の行 silence|3〜6 を 3〜5 回挟む。"
            "沈黙の直後は必ず moshi が穏やかに声をかける（moshi が連続して話してもよい）。"
        )
    elif silence_pattern == "occasional":
        silence_directive = (
            "- 沈黙の行 silence|2〜4 を 1〜2 回挟む。沈黙の直後は moshi が穏やかに声をかける。"
        )
    else:
        silence_directive = "- 沈黙行は入れない。"
    duplex_section = build_duplex_prompt_section(use_case)
    schema_turn_examples = {
        "pause_handling": """
user|発話の前半
silence|2.4
user|同じ発話の続き
moshi|続きを受け止める応答
""".strip(),
        "smooth_turn_taking": """
user|相談者の発話
moshi|発話終了後の自然な応答
""".strip(),
        "backchannel": """
user|長めの相談者発話
moshi|<<overlap=1.5 event=model_backchannel>>はい。
user|相談者の続き
moshi|内容を受け止める応答
""".strip(),
        "user_interruption": """
user|相談者の発話
moshi|やや長い相談員の発話
user|<<overlap=1.2 truncate=0.25 event=user_interruption>>訂正または重要情報
moshi|割り込み内容への応答
""".strip(),
        "user_backchannel": """
user|相談者の発話
moshi|相談員の説明
user|<<overlap=1.0 event=user_backchannel>>はい。
moshi|同じ話題の自然な続き
""".strip(),
        "talking_to_other": """
user|相談者の発話
moshi|相談員の発話
user|<<overlap=1.0 voice_role=user event=talking_to_other>>少し待ってて。
moshi|元の相談を維持する応答
""".strip(),
        "background_speech": """
user|相談者の発話
moshi|相談員の発話
user|<<overlap=1.0 voice_role=background gain=0.32 event=background_speech>>まもなく電車が到着します。
moshi|元の相談を維持する応答
""".strip(),
    }
    schema_turn_example = schema_turn_examples.get(
        duplex_task,
        """
user|相談者の発話の前半、
moshi|はい。
user|発話の続き。
moshi|内容を受け止める応答。
""".strip(),
    )
    # Few-shot is the single biggest chunk of the prompt; keep it small so the
    # model stays under the explicit instructions rather than drowning in
    # examples. Count is configurable (LLM_NUM_FEWSHOT), default 1.
    n_fewshot = max(0, min(num_fewshot, len(LISTENING_DIALOGUE_EXEMPLARS)))
    few_shot_block = ""
    if n_fewshot > 0:
        examples = "\n\n".join(
            dialogue_dict_to_transcript(d)
            for d in LISTENING_DIALOGUE_EXEMPLARS[:n_fewshot]
        )
        few_shot_block = (
            "\n\n品質の参考（コピー禁止。moshiの丁寧語・言い換え・穏やかな深掘りを学ぶ）:\n"
            + examples
        )

    # Curated, minimal case facts -- each stated once. We deliberately do NOT dump
    # the full use_case JSON and do NOT show the concrete `opening` (it stays only
    # as a parser-side fallback). Diversity comes from the sampled fields.
    case_lines = [
        f"- 相談区分: {use_case.get('category', 'unknown')} / リスク {use_case.get('risk_level', 'low')}",
        f"- user 像: {use_case.get('user_profile', '')}",
        f"- user の性格: {personality}（{personality_guidance}）",
        f"- user の今日の状態: {emotional_state}。{emotional_state_hint}",
    ]
    topic = str(use_case.get("topic") or "").strip()
    if topic:
        case_lines.append(f"- 話題の手がかり: {topic}")
    case_summary = "\n".join(case_lines)

    return f"""
{DIALOGUE_SYSTEM_INSTRUCTIONS}

対話設定:
{case_summary}

出力形式（厳守）: 1行=1ターン「話者|発話」。話者は user か moshi の2つだけ。沈黙は「silence|秒数」。
感情ラベルやタイミングなどのメタ情報は付けない。話者と発話内容だけ。
JSON・説明文・見出し・番号・コードブロックは書かない。会話行だけを出力する。

相づち（あいづち）の入れ方 — ここが最重要。交互ターンにしない:
- 相手が長めに話している「途中」にも、聞き手の短い相づちの行をどんどん挟む。相づちの次の行で、話し手はそのまま自分の話を続ける。
- そのために、長い発話は句読点ごとに2〜3行に分け、その切れ目に相手の相づち行を入れる。
- moshi の相づちは丁寧語にする: はい／ええ／そうなんですね／なるほど／そうでしたか。単独の「うん」「うんうん」は使わない。
- user は自然な相談者として「はい」「うん」などで短く反応してよい（双方向）。
- moshi は相づちだけで終わらせすぎず、適度に「相手の言葉の反映」「感情の言語化」「短い確認」を入れて、聞いている感じを出す。
- 重い相談では相づちを控えめに、雑談では多めに。場面で密度を変える。
やってはいけない:
- 言い切りの短文の直後に相づちだけを置かない（浮く・意味が薄い）。相づちは「話の途中」に挟む。
- 「それで」など続きを促す相づちの直後は、必ず話し手が続ける。聞き手がそのまま本題を話し出さない。
- moshi にくだけた「うん」「うんうん」を言わせない。相談員として丁寧な距離感を保つ。
- 相づち行のすぐ後に、同じ話者の長い発話を続けない（相づち→本応答の連続を避け、間に相手の発話を挟む）。

ルール:
- 最初の行は user。user/moshi 合計 {turn_count} 行前後（相づちの行も数える）。本筋の発話は 1行 8〜45 文字。
- 話し方 user={user_style} / moshi={counselor_style}。出だしは毎回変え、定型にしない。
- 性格と今日の状態を語調に反映。気分は急変させず段階的に。無理に前向きにしない。
- high/medium risk は断定せず安全確認の短い問いを入れる。診断・説教・長い助言は避ける。
{points_line}{silence_directive}
{duplex_section}

この対話の形（例）:
{schema_turn_example}{few_shot_block}
""".strip()


def template_full_duplex_dialogue(use_case: dict[str, Any]) -> Dialogue:
    opening = str(use_case.get("opening", "少し話してもいいですか。"))
    case_id = str(use_case.get("id", "full_duplex_template"))
    task = str(use_case.get("duplex_task") or "smooth_turn_taking")

    if task == "pause_handling":
        turns = [
            DialogueTurn("user", opening.rstrip("。")),
            DialogueTurn("silence", duration_sec=2.4, note="userが言葉を探している"),
            DialogueTurn("user", "それで、夜になると少し不安が強くなるんです。"),
            DialogueTurn("moshi", "言葉を探しながら話してくれたんですね。ゆっくりで大丈夫です。"),
        ]
    elif task == "backchannel":
        turns = [
            DialogueTurn("user", "仕事から帰って部屋が静かになると、今日も誰とも話さなかったなって思って。"),
            DialogueTurn(
                "moshi",
                "はい。",
                timing="overlap_previous",
                start_after_previous_start_sec=1.5,
                event="model_backchannel",
            ),
            DialogueTurn("user", "その時間がいちばん寂しく感じるんです。"),
            DialogueTurn("moshi", "一日を振り返る静かな時間に、寂しさが強くなるんですね。"),
        ]
    elif task == "user_interruption":
        turns = [
            DialogueTurn("user", opening),
            DialogueTurn("moshi", "では、まず気分を変えるために外へ出る方法も考えてみましょうか。"),
            DialogueTurn(
                "user",
                "あ、外出の話ではなくて、今は誰かに気持ちを聞いてほしいんです。",
                timing="overlap_previous",
                start_after_previous_start_sec=1.1,
                truncate_previous_after_sec=0.25,
                event="user_interruption",
            ),
            DialogueTurn("moshi", "ごめんなさい、先を急いでしまいました。今の気持ちから聞かせてください。"),
        ]
    elif task == "user_backchannel":
        turns = [
            DialogueTurn("user", opening),
            DialogueTurn("moshi", "無理に答えを出さず、まず今の寂しさが強くなる時間を一緒に見ていきましょう。"),
            DialogueTurn(
                "user",
                "はい。",
                timing="overlap_previous",
                start_after_previous_start_sec=1.0,
                event="user_backchannel",
            ),
            DialogueTurn("moshi", "朝と夜では、どちらのほうがつらく感じますか。"),
        ]
    elif task == "talking_to_other":
        turns = [
            DialogueTurn("user", opening),
            DialogueTurn("moshi", "最近いちばん寂しさが強くなった場面から、話せる範囲で聞かせてください。"),
            DialogueTurn(
                "user",
                "ごめん、今電話しているから少し待ってて。",
                timing="overlap_previous",
                start_after_previous_start_sec=1.0,
                voice_role="user",
                event="talking_to_other",
            ),
            DialogueTurn("moshi", "大丈夫ですよ。落ち着いたら、さっきの続きからゆっくり話しましょう。"),
        ]
    elif task == "background_speech":
        turns = [
            DialogueTurn("user", opening),
            DialogueTurn("moshi", "ここでは急がなくて大丈夫です。話しやすいところから聞かせてください。"),
            DialogueTurn(
                "user",
                "まもなく二番線に電車が到着します。黄色い線までお下がりください。",
                timing="overlap_previous",
                start_after_previous_start_sec=1.0,
                gain=0.32,
                voice_role="background",
                event="background_speech",
            ),
            DialogueTurn("moshi", "今は、どんなことがいちばん心に残っていますか。"),
        ]
    else:
        turns = [
            DialogueTurn("user", opening),
            DialogueTurn("moshi", "来てくれてありがとうございます。少しずつで大丈夫ですよ。"),
            DialogueTurn("user", "夜になると、誰ともつながっていない感じがします。"),
            DialogueTurn("moshi", "夜の静かな時間に、その感じが強くなるんですね。"),
        ]

    return Dialogue(
        id=safe_stem(case_id, "full_duplex_template"),
        category=str(use_case.get("category", "unknown")),
        risk_level=str(use_case.get("risk_level", "low")),
        title=str(use_case.get("situation", case_id)),
        source_use_case=case_id,
        turns=turns,
        generator_notes="Full-duplex deterministic fallback template.",
        duplex_task=task,
    )


def template_dialogue(use_case: dict[str, Any]) -> Dialogue:
    if use_case.get("duplex_task"):
        return template_full_duplex_dialogue(use_case)

    opening = str(use_case.get("opening", "少し話してもいいですか。"))
    case_id = str(use_case.get("id", "template_dialogue"))
    if str(use_case.get("risk_level", "low")) == "medium":
        turns = [
            DialogueTurn("user", opening),
            DialogueTurn("moshi", "話してくれてありがとうございます。今はかなり抱えている感じでしょうか。"),
            DialogueTurn("user", "はい。誰かに言うだけでも迷惑かなって思います。"),
            DialogueTurn("moshi", "迷惑かもと思うと、声を出すだけでも苦しくなりますよね。"),
            DialogueTurn("user", "そうなんです。結局ひとりで考えてしまって。"),
            DialogueTurn("moshi", "今ここでは、急いで結論を出さなくて大丈夫です。少し一緒に整理しましょう。"),
            DialogueTurn("user", "少し安心しました。"),
            DialogueTurn("moshi", "よかったです。今夜はひとりで安全に過ごせそうかも、確認してもいいですか。"),
        ]
    else:
        turns = [
            DialogueTurn("user", opening),
            DialogueTurn("moshi", "来てくれてありがとうございます。少し話すだけでも大丈夫ですよ。"),
            DialogueTurn("user", "なんとなく、人とつながってない感じがして。"),
            DialogueTurn("moshi", "その感じ、ふと強くなると寂しいですよね。今日はどんな時間に強くなりましたか。"),
            DialogueTurn("user", "夜になってからです。部屋が静かで。"),
            DialogueTurn("moshi", "静かになると、気持ちだけが大きく聞こえることがありますね。"),
            DialogueTurn("user", "そうかもしれません。"),
            DialogueTurn("moshi", "ここで少し話していきましょう。急いで元気にならなくても大丈夫です。"),
        ]
    return Dialogue(
        id=safe_stem(case_id, "template_dialogue"),
        category=str(use_case.get("category", "unknown")),
        risk_level=str(use_case.get("risk_level", "low")),
        title=str(use_case.get("situation", case_id)),
        source_use_case=case_id,
        turns=turns,
        generator_notes="LLM fallback template.",
    )


class LocalTTS:
    def __init__(self, args: argparse.Namespace, sample_rate: int, tmp_dir: Path):
        self.args = args
        self.sample_rate = sample_rate
        self.tmp_dir = tmp_dir
        self.counter = 0
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, speaker: str) -> np.ndarray:
        self.counter += 1
        wav_path = self.tmp_dir / f"tts_{self.counter:05d}_{speaker}.wav"
        if speaker == "user":
            voice = self.args.tts_voice_user
            speed = self.args.tts_speed_user
        else:
            voice = self.args.tts_voice_moshi
            speed = self.args.tts_speed_moshi
        synthesize_text_to_wav(
            text=text,
            wav_path=wav_path,
            voice=voice,
            rate=self.args.tts_rate,
            speed=speed,
        )
        pcm = load_wav_mono(wav_path, self.sample_rate)
        if not self.args.no_trim_tts:
            pcm = trim_silence(pcm, self.sample_rate)
        return pcm.astype(np.float32)


class MoshiTTS:
    """Script-faithful text-to-speech using Kyutai/Moshi TTSModel."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tts_model = None
        self.sample_rate = 24_000

    def load(self) -> None:
        if self.tts_model is not None:
            return
        try:
            from moshi.models.loaders import CheckpointInfo  # type: ignore[import]
            from moshi.models.tts import (  # type: ignore[import]
                DEFAULT_DSM_TTS_VOICE_REPO,
                TTSModel,
            )
        except Exception as exc:
            raise RuntimeError(
                "Moshi TTS requires a moshi package that provides "
                "`moshi.models.tts.TTSModel`. Install/update moshi in the "
                "Moshi environment, then rerun."
            ) from exc

        logger.info("Loading Kyutai/Moshi TTS from %s ...", self.args.moshi_tts_repo)
        checkpoint_info = CheckpointInfo.from_hf_repo(self.args.moshi_tts_repo)
        self.tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            n_q=self.args.moshi_tts_n_q,
            temp=self.args.moshi_tts_temp,
            device=self.args.device,
        )
        if self.args.moshi_tts_voice_repo is None:
            self.args.moshi_tts_voice_repo = DEFAULT_DSM_TTS_VOICE_REPO
        self.sample_rate = int(self.tts_model.mimi.sample_rate)

    def synthesize(self, text: str, speaker: str) -> np.ndarray:
        self.load()
        assert self.tts_model is not None

        import torch

        voice = (
            self.args.moshi_tts_voice_user
            if speaker == "user"
            else self.args.moshi_tts_voice_moshi
        )
        entries = self.tts_model.prepare_script([text], padding_between=1)
        voice_path = self._voice_path(voice)
        condition_attributes = self.tts_model.make_condition_attributes(
            [voice_path],
            cfg_coef=self.args.moshi_tts_cfg_coef,
        )

        frame_count = 0

        def _on_frame(frame) -> None:
            nonlocal frame_count
            try:
                if (frame != -1).all():
                    frame_count += 1
            except Exception:
                frame_count += 1

        with torch.no_grad():
            result = self.tts_model.generate(
                [entries],
                [condition_attributes],
                on_frame=_on_frame,
            )

        pcms: list[np.ndarray] = []
        with self.tts_model.mimi.streaming(1), torch.no_grad():
            for frame in result.frames[self.tts_model.delay_steps:]:
                try:
                    if not bool((frame != -1).all().item()):
                        continue
                except Exception:
                    pass
                pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1.0, 1.0).astype(np.float32))

        if not pcms:
            logger.warning("Moshi TTS produced no audio for text: %s", text)
            return np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
        pcm_out = np.concatenate(pcms, axis=-1).astype(np.float32)
        if not self.args.no_trim_tts:
            pcm_out = trim_silence(pcm_out, self.sample_rate)
        logger.info(
            "Moshi TTS synthesized %.2fs for %s turn (%d frames)",
            pcm_out.size / self.sample_rate,
            speaker,
            frame_count,
        )
        return pcm_out

    def _voice_path(self, voice: str) -> str:
        assert self.tts_model is not None
        if voice.endswith(".safetensors") or Path(voice).is_file():
            return voice
        return self.tts_model.get_voice_path(voice)


def trim_silence(pcm: np.ndarray, sample_rate: int, threshold: float = 1e-4) -> np.ndarray:
    if pcm.size == 0:
        return pcm
    mask = np.abs(pcm) > threshold
    if not np.any(mask):
        return pcm
    idx = np.flatnonzero(mask)
    pad = int(0.06 * sample_rate)
    start = max(0, int(idx[0]) - pad)
    end = min(pcm.size, int(idx[-1]) + pad)
    return pcm[start:end]


def render_segments_to_stereo(
    segments: list[AudioSegment],
    sample_rate: int,
    tail_sec: float = 0.0,
) -> np.ndarray:
    duration_sec = max((segment.end_sec for segment in segments), default=0.0) + tail_sec
    total_samples = max(1, int(math.ceil(duration_sec * sample_rate)))
    stereo = np.zeros((2, total_samples), dtype=np.float32)
    for segment in segments:
        channel = 0 if segment.speaker == "moshi" else 1
        start = int(round(segment.start_sec * sample_rate))
        end = min(total_samples, start + segment.pcm.size)
        if end <= start:
            continue
        stereo[channel, start:end] += segment.pcm[: end - start]
    return normalize_stereo(stereo)


def render_segments_to_mono(
    segments: list[AudioSegment],
    sample_rate: int,
    tail_sec: float,
) -> np.ndarray:
    stereo = render_segments_to_stereo(segments, sample_rate, tail_sec=tail_sec)
    return stereo[1].copy()


def normalize_stereo(stereo: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 0.99:
        stereo = stereo / peak * 0.99
    return stereo.astype(np.float32)


def truncate_alignment_text(text: str, kept_samples: int, total_samples: int) -> str:
    if not text or total_samples <= 0 or kept_samples >= total_samples:
        return text
    ratio = max(0.0, min(1.0, kept_samples / total_samples))
    keep_chars = max(1, min(len(text), int(math.ceil(len(text) * ratio))))
    shortened = text[:keep_chars].rstrip("、。！？,.!? ")
    return (shortened or text[:1]) + "…"


def build_scripted_segments(
    dialogue: Dialogue,
    tts: Any,
    args: argparse.Namespace,
) -> list[AudioSegment]:
    cursor = args.lead_in_sec
    segments: list[AudioSegment] = []
    previous_segment: AudioSegment | None = None
    for turn in dialogue.turns:
        if turn.speaker == "silence":
            # 沈黙ターンは音声を作らず、duration_sec ぶんだけ無音を挟む。
            cursor += float(turn.duration_sec or 0.0)
            previous_segment = None
            continue
        pcm = tts.synthesize(turn.text, turn.speaker)
        gain = max(0.0, float(turn.gain))
        if gain != 1.0:
            pcm = np.asarray(pcm, dtype=np.float32) * gain
        if turn.timing == "overlap_previous" and previous_segment is not None:
            offset = max(0.0, float(turn.start_after_previous_start_sec or 0.0))
            latest_overlap_start = max(
                previous_segment.start_sec,
                previous_segment.end_sec - 0.05,
            )
            start = min(previous_segment.start_sec + offset, latest_overlap_start)
            if turn.truncate_previous_after_sec is not None:
                stop = min(
                    previous_segment.end_sec,
                    start + max(0.0, float(turn.truncate_previous_after_sec)),
                )
                original_samples = previous_segment.pcm.size
                keep = max(
                    1,
                    int(round((stop - previous_segment.start_sec) * tts.sample_rate)),
                )
                previous_segment.text = truncate_alignment_text(
                    previous_segment.text,
                    kept_samples=keep,
                    total_samples=original_samples,
                )
                previous_segment.pcm = previous_segment.pcm[:keep]
                previous_segment.end_sec = (
                    previous_segment.start_sec
                    + previous_segment.pcm.size / tts.sample_rate
                )
        else:
            start = cursor
        end = start + pcm.size / tts.sample_rate
        segment = AudioSegment(
            speaker=turn.speaker,
            label="SPEAKER_MAIN" if turn.speaker == "moshi" else "SPEAKER_USER",
            text=turn.text,
            start_sec=start,
            end_sec=end,
            pcm=pcm,
            event=turn.event,
            voice_role=turn.voice_role,
        )
        segments.append(segment)
        cursor = max(item.end_sec for item in segments) + args.turn_gap_sec
        previous_segment = segment
    return segments


def estimate_response_gap(text: str, args: argparse.Namespace) -> float:
    chars = len(re.sub(r"\s+", "", text))
    raw = chars / max(1.0, args.estimate_chars_per_sec)
    return max(1.2, min(8.0, raw)) + args.assistant_gap_sec


def build_user_prompt_segments(
    dialogue: Dialogue,
    tts: LocalTTS,
    args: argparse.Namespace,
) -> list[AudioSegment]:
    cursor = args.lead_in_sec
    segments: list[AudioSegment] = []
    for turn in dialogue.turns:
        if turn.speaker == "silence":
            # 沈黙ターンはスクリプト指定の duration_sec ぶんカーソルを進める。
            cursor += float(turn.duration_sec or 0.0)
        elif turn.speaker == "user":
            pcm = tts.synthesize(turn.text, "user")
            start = cursor
            end = start + pcm.size / tts.sample_rate
            segments.append(
                AudioSegment(
                    speaker="user",
                    label="SPEAKER_USER",
                    text=turn.text,
                    start_sec=start,
                    end_sec=end,
                    pcm=pcm,
                )
            )
            cursor = end + args.user_pause_sec
        else:
            cursor += estimate_response_gap(turn.text, args)
    return segments


def segments_to_alignments(segments: list[AudioSegment]) -> list[list[Any]]:
    return [
        [
            segment.text,
            [round(segment.start_sec, 4), round(segment.end_sec, 4)],
            segment.label,
        ]
        for segment in segments
    ]


def moshi_text_events_to_alignments(
    text_events: list[dict[str, Any]],
    acoustic_delay_sec: float,
    total_duration_sec: float,
) -> list[list[Any]]:
    alignments: list[list[Any]] = []
    current_text = ""
    current_start: Optional[float] = None
    current_last = 0.0

    def flush(end_hint: float) -> None:
        nonlocal current_text, current_start, current_last
        text = current_text.strip()
        if text and current_start is not None:
            end = min(total_duration_sec, max(current_start + 0.35, end_hint))
            alignments.append(
                [
                    text,
                    [round(current_start, 4), round(end, 4)],
                    "SPEAKER_MAIN",
                ]
            )
        current_text = ""
        current_start = None
        current_last = 0.0

    for event in text_events:
        piece = str(event.get("piece", ""))
        if not piece:
            continue
        t = float(event.get("time_sec", 0.0)) + acoustic_delay_sec
        t = max(0.0, min(total_duration_sec, t))
        if current_start is None:
            current_start = t
        current_text += piece
        current_last = t
        if re.search(r"[。！？!?]\s*$", piece) or len(current_text) >= 36:
            flush(current_last + 0.45)

    flush(current_last + 0.8)
    return alignments


def make_moshi_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hf_repo=args.moshi_hf_repo,
        moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight,
        tokenizer=args.tokenizer,
        config=args.config,
        device=args.device,
        half=args.half,
        temp=args.temp,
        temp_text=args.temp_text,
        cfg_coef=args.cfg_coef,
    )


class MoshiSelfplayRenderer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mimi = None
        self.text_tokenizer = None
        self.lm_gen = None
        self.lm_gen_cfg = None
        self.acoustic_delay = 0

    def load(self) -> None:
        if self.mimi is not None:
            return
        logger.info("Loading Moshi for selfplay: %s", self.args.moshi_hf_repo)
        (
            self.mimi,
            self.text_tokenizer,
            self.lm_gen,
            self.lm_gen_cfg,
            self.acoustic_delay,
        ) = load_models(make_moshi_args(self.args))

    @property
    def sample_rate(self) -> int:
        assert self.mimi is not None
        return int(self.mimi.sample_rate)

    @property
    def frame_rate(self) -> float:
        assert self.mimi is not None
        return float(self.mimi.frame_rate)

    def render(
        self,
        dialogue: Dialogue,
        tts: LocalTTS,
        seed: int,
        sample_dir: Path,
    ) -> tuple[np.ndarray, list[list[Any]], dict[str, Any]]:
        self.load()
        assert self.mimi is not None
        assert self.text_tokenizer is not None
        assert self.lm_gen is not None

        user_segments = build_user_prompt_segments(dialogue, tts, self.args)
        user_prompt_pcm = render_segments_to_mono(
            user_segments,
            self.sample_rate,
            tail_sec=0.0,
        )
        user_prompt_sec = user_prompt_pcm.size / self.sample_rate
        max_gen_sec = min(
            self.args.max_total_sec,
            user_prompt_sec + self.args.moshi_silence_sec,
        )

        logger.info(
            "Moshi selfplay id=%s user_prompt=%.2fs max_gen=%.2fs",
            dialogue.id,
            user_prompt_sec,
            max_gen_sec,
        )
        t0 = time.time()
        result = run_trial(
            pcm=user_prompt_pcm,
            seed=seed,
            lm_gen=self.lm_gen,
            mimi=self.mimi,
            text_tokenizer=self.text_tokenizer,
            device=self.args.device,
            silence_sec=self.args.moshi_silence_sec,
            max_gen_sec=max_gen_sec,
            acoustic_delay=self.acoustic_delay,
        )
        wall_time = time.time() - t0
        pcm_left = decode_audio(
            result["audio_frames"],
            self.acoustic_delay,
            self.mimi,
            self.args.device,
        )

        first_audio_step = result.get("first_audio_step")
        left_start_sec = (
            (first_audio_step + self.acoustic_delay) / self.frame_rate
            if first_audio_step is not None
            else 0.0
        )
        right_total = user_prompt_pcm.size + int(round(self.args.moshi_silence_sec * self.sample_rate))
        left_start = int(round(left_start_sec * self.sample_rate))
        left_len = 0 if pcm_left is None else int(pcm_left.size)
        total_samples = max(1, right_total, left_start + left_len)
        stereo = np.zeros((2, total_samples), dtype=np.float32)
        stereo[1, : user_prompt_pcm.size] = user_prompt_pcm
        if pcm_left is not None and pcm_left.size:
            stereo[0, left_start: left_start + pcm_left.size] = pcm_left
        stereo = normalize_stereo(stereo)

        acoustic_delay_sec = self.acoustic_delay / self.frame_rate
        total_duration_sec = total_samples / self.sample_rate
        moshi_alignments = moshi_text_events_to_alignments(
            result["text_events"],
            acoustic_delay_sec=acoustic_delay_sec,
            total_duration_sec=total_duration_sec,
        )
        alignments = sorted(
            moshi_alignments + segments_to_alignments(user_segments),
            key=lambda item: (item[1][0], item[2]),
        )
        transcript = "".join(event["piece"] for event in result["text_events"]).strip()
        meta = {
            "mode": "moshi-selfplay",
            "sample_rate": self.sample_rate,
            "duration_sec": round(total_duration_sec, 4),
            "moshi_hf_repo": self.args.moshi_hf_repo,
            "seed": seed,
            "wall_time_sec": round(wall_time, 3),
            "user_prompt_sec": round(user_prompt_sec, 4),
            "moshi_silence_sec": self.args.moshi_silence_sec,
            "first_audio_step": first_audio_step,
            "audible_response_start_sec": round(left_start_sec, 4) if first_audio_step is not None else None,
            "moshi_text_transcript": transcript,
            "transcript_source": "moshi_text_tokens_approx",
        }
        write_json(sample_dir / f"{dialogue.id}_moshi_result.json", result_for_json(result))
        return stereo, alignments, meta


def result_for_json(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "text_events": result.get("text_events", []),
        "total_steps": result.get("total_steps"),
        "input_steps": result.get("input_steps"),
        "first_audio_step": result.get("first_audio_step"),
        "first_response_step": result.get("first_response_step"),
        "audio_frame_count": len(result.get("audio_frames", [])),
    }


def save_wav(path: Path, stereo: np.ndarray, sample_rate: int) -> None:
    import sphn  # type: ignore[import]

    path.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(path), stereo.astype(np.float32), sample_rate)


def load_use_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.use_cases_jsonl is not None:
        return load_jsonl(args.use_cases_jsonl)
    return list(DEFAULT_USE_CASES)


def load_dialogues_from_jsonl(path: Path) -> list[Dialogue]:
    rows = load_jsonl(path)
    dialogues: list[Dialogue] = []
    for index, row in enumerate(rows, start=1):
        use_case = {"id": row.get("source_use_case", f"dialogue_{index:03d}")}
        dialogues.append(dialogue_from_mapping(row, use_case))
    return dialogues


def generate_dialogues(
    args: argparse.Namespace,
    on_generated: Optional[Callable[[Dialogue], None]] = None,
) -> list[Dialogue]:
    """Generate dialogues. If on_generated is given it is called for each dialogue
    as soon as it is produced, so callers can persist incrementally instead of
    losing everything if a long run is interrupted."""

    emit_lock = threading.Lock()

    def _emit(dialogue: Dialogue, sink: list[Dialogue]) -> None:
        # Locked so concurrent workers can append + persist incrementally
        # without racing on the sink list or the on_generated sink.
        with emit_lock:
            sink.append(dialogue)
            if on_generated is not None:
                on_generated(dialogue)

    if args.dialogues_jsonl is not None:
        logger.info("Loading dialogues from %s", args.dialogues_jsonl)
        loaded = load_dialogues_from_jsonl(args.dialogues_jsonl)[: args.num_dialogues]
        if on_generated is not None:
            for dialogue in loaded:
                on_generated(dialogue)
        return loaded

    rng = random.Random(args.seed)
    use_cases = load_use_cases(args)[: args.num_dialogues]
    generator = LLMDialogueGenerator(args)
    dialogues: list[Dialogue] = []
    if args.dialogue_generation_mode == "multi-agent":
        if args.dump_agent_prompts and use_cases:
            preview_rng = random.Random(args.seed)
            preview = multi_agent_prompt_preview(use_cases[0], preview_rng, args)
            write_json(args.out_dir / "multi_agent_prompts.json", preview)
            logger.info(
                "Wrote multi-agent prompt preview to %s",
                args.out_dir / "multi_agent_prompts.json",
            )
        def _one_dialogue(index: int, use_case: dict[str, Any]) -> Dialogue:
            # Per-case RNG keyed on the seed so output stays deterministic and
            # thread-safe regardless of concurrency (a shared random.Random is
            # neither). Profile selection is the only RNG consumer downstream.
            case_rng = random.Random(args.seed + index)
            try:
                dialogue = generator.generate_multi_agent(use_case, case_rng)
                logger.info("Generated dialogue with multi-agent LLM: %s", dialogue.id)
            except Exception as exc:
                if args.no_template_fallback or not args.allow_template_fallback:
                    raise
                logger.warning(
                    "Multi-agent generation failed for %s; using template fallback: %s",
                    use_case.get("id"),
                    exc,
                )
                dialogue = template_dialogue(use_case)
            return dialogue

        concurrency = max(1, int(getattr(args, "multi_agent_concurrency", 1)))
        if concurrency <= 1:
            for index, use_case in enumerate(use_cases):
                _emit(_one_dialogue(index, use_case), dialogues)
        else:
            logger.info(
                "Generating %d multi-agent dialogues with concurrency=%d",
                len(use_cases),
                concurrency,
            )
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(_one_dialogue, index, use_case)
                    for index, use_case in enumerate(use_cases)
                ]
                for future in futures:
                    _emit(future.result(), dialogues)
        return dialogues

    if args.llm_backend == "transformers-subprocess":
        prompts = [
            build_llm_prompt(use_case, rng, args.llm_num_fewshot)
            for use_case in use_cases
        ]
        try:
            texts = generator.generate_transformers_subprocess_batch(prompts)
            if len(texts) != len(use_cases):
                raise RuntimeError(
                    f"LLM returned {len(texts)} dialogues for "
                    f"{len(use_cases)} use cases."
                )
            for use_case, text in zip(use_cases, texts):
                data = parse_transcript(text)
                dialogue = dialogue_from_mapping(data, use_case)
                logger.info("Generated dialogue with LLM: %s", dialogue.id)
                _emit(dialogue, dialogues)
            return dialogues
        except Exception as exc:
            if args.no_template_fallback or not args.allow_template_fallback:
                raise
            logger.warning(
                "LLM batch generation failed; using template fallback: %s",
                exc,
            )
            fallback: list[Dialogue] = []
            for use_case in use_cases:
                _emit(template_dialogue(use_case), fallback)
            return fallback

    for use_case in use_cases:
        try:
            dialogue = generator.generate(use_case, rng)
            logger.info("Generated dialogue with LLM: %s", dialogue.id)
        except Exception as exc:
            if args.no_template_fallback or not args.allow_template_fallback:
                raise
            logger.warning(
                "LLM generation failed for %s; using template fallback: %s",
                use_case.get("id"),
                exc,
            )
            dialogue = template_dialogue(use_case)
        _emit(dialogue, dialogues)
    return dialogues


def dialogue_to_json(dialogue: Dialogue) -> dict[str, Any]:
    data = asdict(dialogue)
    data["turns"] = [asdict(turn) for turn in dialogue.turns]
    return data


def render_one_scripted(
    dialogue: Dialogue,
    args: argparse.Namespace,
    sample_dir: Path,
) -> tuple[np.ndarray, list[list[Any]], dict[str, Any]]:
    sample_rate = 24_000
    tts = LocalTTS(args=args, sample_rate=sample_rate, tmp_dir=sample_dir / "_tmp_tts")
    segments = build_scripted_segments(dialogue, tts, args)
    stereo = render_segments_to_stereo(segments, sample_rate, tail_sec=0.5)
    meta = {
        "mode": args.mode,
        "sample_rate": sample_rate,
        "duration_sec": round(stereo.shape[-1] / sample_rate, 4),
        "transcript_source": "llm_script",
    }
    return stereo, segments_to_alignments(segments), meta


def render_one_moshi_tts(
    dialogue: Dialogue,
    args: argparse.Namespace,
    tts: MoshiTTS,
) -> tuple[np.ndarray, list[list[Any]], dict[str, Any]]:
    segments = build_scripted_segments(dialogue, tts, args)
    stereo = render_segments_to_stereo(segments, tts.sample_rate, tail_sec=0.5)
    meta = {
        "mode": "scripted-moshi-tts",
        "sample_rate": tts.sample_rate,
        "duration_sec": round(stereo.shape[-1] / tts.sample_rate, 4),
        "moshi_tts_repo": args.moshi_tts_repo,
        "moshi_tts_voice_user": args.moshi_tts_voice_user,
        "moshi_tts_voice_moshi": args.moshi_tts_voice_moshi,
        "transcript_source": "llm_script",
        "audio_source": "kyutai_moshi_tts",
    }
    return stereo, segments_to_alignments(segments), meta


def render_dataset(args: argparse.Namespace, dialogues: list[Dialogue]) -> None:
    out_dir = args.out_dir
    data_dir = out_dir / "data_stereo"
    sample_meta_dir = out_dir / "sample_metadata"
    tmp_root = out_dir / "_tmp"
    for path in (data_dir, sample_meta_dir, tmp_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / args.manifest_name
    dialogues_path = out_dir / "dialogues.jsonl"
    run_meta_path = out_dir / "generation_run.json"
    for path in (manifest_path, dialogues_path):
        if path.exists():
            path.unlink()

    moshi_renderer: Optional[MoshiSelfplayRenderer] = None
    moshi_tts: Optional[MoshiTTS] = None
    if args.mode == "moshi-selfplay":
        moshi_renderer = MoshiSelfplayRenderer(args)
        moshi_renderer.load()
        sample_rate = moshi_renderer.sample_rate
    elif args.mode == "scripted-moshi-tts":
        moshi_tts = MoshiTTS(args)
        moshi_tts.load()
        sample_rate = moshi_tts.sample_rate
    else:
        sample_rate = 24_000

    run_meta = {
        "datetime": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "num_dialogues": len(dialogues),
        "llm_model": args.llm_model,
        "moshi_hf_repo": args.moshi_hf_repo,
        "sample_rate": sample_rate,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(run_meta_path, run_meta)

    for index, dialogue in enumerate(dialogues, start=1):
        stem = f"sample_{index:03d}_{safe_stem(dialogue.id, f'dialogue_{index:03d}')}"
        sample_dir = sample_meta_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(dialogues_path, dialogue_to_json(dialogue))

        if args.mode == "moshi-selfplay":
            assert moshi_renderer is not None
            tts = LocalTTS(
                args=args,
                sample_rate=moshi_renderer.sample_rate,
                tmp_dir=tmp_root / stem / "tts",
            )
            stereo, alignments, meta = moshi_renderer.render(
                dialogue=dialogue,
                tts=tts,
                seed=args.seed + index - 1,
                sample_dir=sample_dir,
            )
            sample_rate = moshi_renderer.sample_rate
        elif args.mode == "scripted-moshi-tts":
            assert moshi_tts is not None
            stereo, alignments, meta = render_one_moshi_tts(dialogue, args, moshi_tts)
            sample_rate = moshi_tts.sample_rate
        else:
            stereo, alignments, meta = render_one_scripted(dialogue, args, sample_dir)
            sample_rate = int(meta["sample_rate"])

        wav_path = data_dir / f"{stem}.wav"
        json_path = wav_path.with_suffix(".json")
        duration = stereo.shape[-1] / sample_rate
        save_wav(wav_path, stereo, sample_rate)
        write_json(
            json_path,
            {
                "alignments": alignments,
                "metadata": {
                    **meta,
                    "dialogue": dialogue_to_json(dialogue),
                    "left_channel": "moshi",
                    "right_channel": "user",
                },
            },
        )
        append_jsonl(
            manifest_path,
            {
                "path": str(wav_path.relative_to(out_dir)).replace("\\", "/"),
                "duration": duration,
            },
        )
        logger.info("Wrote %s duration=%.2fs", wav_path, duration)

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    logger.info("Done. Manifest: %s", manifest_path)


def write_dialogues_only(args: argparse.Namespace) -> None:
    """Generate and persist each dialogue as soon as it is produced.

    Appending per dialogue (rather than writing the whole list at the end) means
    an interrupted long run keeps every dialogue generated so far."""
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogues_path = out_dir / "dialogues.jsonl"
    if dialogues_path.exists():
        dialogues_path.unlink()

    count = 0

    def sink(dialogue: Dialogue) -> None:
        nonlocal count
        append_jsonl(dialogues_path, dialogue_to_json(dialogue))
        count += 1
        logger.info("saved dialogue %d -> %s (%s)", count, dialogue.id, dialogues_path)

    generate_dialogues(args, on_generated=sink)
    logger.info("dialogues-only mode: wrote %d dialogues to %s", count, dialogues_path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "dialogues-only":
        write_dialogues_only(args)
        return
    dialogues = generate_dialogues(args)
    render_dataset(args, dialogues)


if __name__ == "__main__":
    main()
