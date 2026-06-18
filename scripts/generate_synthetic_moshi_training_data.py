#!/usr/bin/env python3
from __future__ import annotations
"""
Generate a small synthetic Moshi fine-tuning dataset.

Pipeline:
1. Gemma 4 generates Japanese dialogue scripts for loneliness/isolation
   support-window use cases.
2. In the default "scripted-moshi-tts" mode, each script turn is rendered
   with Kyutai/Moshi TTS and placed into a stereo dialogue file.
3. In "moshi-selfplay" mode, user turns are rendered as audio and
   llm-jp-moshi-v1 generates the counselor stream from that user audio context.
4. In "scripted-local-tts" mode, both speakers are rendered from the Gemma
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
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small Gemma 4 + Moshi TTS synthetic stereo dataset "
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
            "dialogues-only: Gemma で対話 JSONL を生成して終了（音声合成は行わない）。"
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
            "Optional pre-generated dialogue JSONL. If set, Gemma is not loaded "
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
        help="Use deterministic templates if Gemma generation fails.",
    )

    # Gemma generation
    parser.add_argument(
        "--gemma-backend",
        choices=[
            "transformers-subprocess",
            "transformers",
            "openai-compatible",
            "template",
        ],
        default="transformers-subprocess",
        help=(
            "Dialogue generation backend. transformers-subprocess runs Gemma "
            "from this script via a separate Python environment, avoiding "
            "dependency conflicts with Moshi. Default: transformers-subprocess."
        ),
    )
    parser.add_argument(
        "--gemma-python",
        default=os.environ.get("GEMMA_PYTHON"),
        help=(
            "Python executable for --gemma-backend transformers-subprocess. "
            "Defaults to gemma_runtime/.venv/bin/python if present, otherwise "
            "the current Python."
        ),
    )
    parser.add_argument(
        "--gemma-worker",
        type=Path,
        default=Path(__file__).with_name("gemma_dialogue_worker.py"),
        help="Worker script used by --gemma-backend transformers-subprocess.",
    )
    parser.add_argument(
        "--gemma-api-base",
        default=os.environ.get("GEMMA_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8080/v1",
        help=(
            "OpenAI-compatible API base URL for Gemma, e.g. "
            "http://127.0.0.1:8080/v1. Used with --gemma-backend "
            "openai-compatible."
        ),
    )
    parser.add_argument(
        "--gemma-api-key",
        default=os.environ.get("GEMMA_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="Optional API key for the OpenAI-compatible Gemma endpoint.",
    )
    parser.add_argument(
        "--gemma-timeout-sec",
        type=float,
        default=300.0,
        help="Timeout for Gemma generation requests/subprocess calls.",
    )
    parser.add_argument(
        "--gemma-model",
        default="google/gemma-4-E2B-it",
        help=(
            "Gemma model id/name. For OpenAI-compatible servers this is sent "
            "as the request model field."
        ),
    )
    parser.add_argument(
        "--gemma-task",
        default="any-to-any",
        help=(
            "Transformers pipeline task, used with --gemma-backend "
            "transformers or transformers-subprocess."
        ),
    )
    parser.add_argument(
        "--gemma-device-map",
        default="auto",
        help='Device map passed to transformers.pipeline. Default: "auto".',
    )
    parser.add_argument(
        "--gemma-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--gemma-temperature", type=float, default=0.8)
    parser.add_argument("--gemma-max-new-tokens", type=int, default=1400)
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
        raise ValueError("No JSON object found in Gemma output.")
    return json.loads(text[start: end + 1])


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


class GemmaDialogueGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipe = None
        self.active_task = args.gemma_task

    def _torch_dtype(self):
        if self.args.gemma_dtype == "auto":
            return "auto"
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.args.gemma_dtype]

    def load(self) -> None:
        if self.args.gemma_backend in {
            "openai-compatible",
            "template",
            "transformers-subprocess",
        }:
            return
        if self.pipe is not None:
            return
        from transformers import pipeline

        kwargs: dict[str, Any] = {
            "model": self.args.gemma_model,
            "device_map": self.args.gemma_device_map,
            "trust_remote_code": self.args.trust_remote_code,
        }
        dtype = self._torch_dtype()
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype

        logger.info("Loading Gemma pipeline task=%s model=%s", self.active_task, self.args.gemma_model)
        try:
            self.pipe = pipeline(self.active_task, **kwargs)
        except Exception:
            if self.active_task == "text-generation":
                raise
            logger.warning("Falling back to text-generation pipeline.", exc_info=True)
            self.active_task = "text-generation"
            self.pipe = pipeline("text-generation", **kwargs)

    def generate(self, use_case: dict[str, Any], rng: random.Random) -> Dialogue:
        if self.args.gemma_backend == "template":
            return template_dialogue(use_case)
        prompt = build_gemma_prompt(use_case, rng)

        if self.args.gemma_backend == "openai-compatible":
            text = self.generate_openai_compatible(prompt)
            data = extract_json_object(text)
            return dialogue_from_mapping(data, use_case)

        if self.args.gemma_backend == "transformers-subprocess":
            text = self.generate_transformers_subprocess(prompt)
            data = extract_json_object(text)
            return dialogue_from_mapping(data, use_case)

        self.load()
        assert self.pipe is not None
        generation_kwargs = {
            "max_new_tokens": self.args.gemma_max_new_tokens,
            "temperature": self.args.gemma_temperature,
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
        data = extract_json_object(text)
        return dialogue_from_mapping(data, use_case)

    def generate_openai_compatible(self, prompt: str) -> str:
        url = chat_completions_url(self.args.gemma_api_base)
        payload = {
            "model": self.args.gemma_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.args.gemma_temperature,
            "max_tokens": self.args.gemma_max_new_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.args.gemma_api_key:
            headers["Authorization"] = f"Bearer {self.args.gemma_api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.args.gemma_timeout_sec,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach the OpenAI-compatible Gemma endpoint at "
                f"{url}. Start a Gemma server separately, pass --gemma-api-base, "
                "or use --dialogues-jsonl / --gemma-backend template."
            ) from exc
        return openai_chat_response_to_text(data)

    def generate_transformers_subprocess(self, prompt: str) -> str:
        return self.generate_transformers_subprocess_batch([prompt])[0]

    def generate_transformers_subprocess_batch(self, prompts: list[str]) -> list[str]:
        python_exe = resolve_gemma_python(self.args)
        worker = Path(self.args.gemma_worker)
        if not worker.is_file():
            raise FileNotFoundError(f"Gemma worker script not found: {worker}")

        command = [
            python_exe,
            str(worker),
            "--model",
            self.args.gemma_model,
            "--task",
            self.args.gemma_task,
            "--device-map",
            self.args.gemma_device_map,
            "--dtype",
            self.args.gemma_dtype,
            "--temperature",
            str(self.args.gemma_temperature),
            "--max-new-tokens",
            str(self.args.gemma_max_new_tokens),
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
                timeout=self.args.gemma_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Gemma subprocess timed out after {self.args.gemma_timeout_sec}s: "
                f"{' '.join(command)}"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                "Gemma subprocess failed. Install the Gemma runtime with "
                "`uv sync --project gemma_runtime`, or pass --gemma-python to "
                "a Python environment that has Transformers/Gemma installed.\n"
                f"command: {' '.join(command)}\n"
                "(stderr output is shown above)"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Gemma subprocess did not return JSON.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            ) from exc
        texts = data.get("texts")
        if not isinstance(texts, list):
            raise RuntimeError(f"Gemma worker returned no 'texts' list: {data}")
        return [str(text) for text in texts]


def resolve_gemma_python(args: argparse.Namespace) -> str:
    if args.gemma_python:
        return str(args.gemma_python)

    runtime_dir = REPO_ROOT / "gemma_runtime" / ".venv"
    candidates = [
        runtime_dir / "bin" / "python",
        runtime_dir / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def openai_chat_response_to_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"OpenAI-compatible response has no choices: {data}")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
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


def build_duplex_prompt_section(use_case: dict[str, Any]) -> str:
    task = str(use_case.get("duplex_task") or "").strip()
    if not task:
        return ""

    task_rules = {
        "pause_handling": (
            "user発話を前半、2〜3秒のsilence、後半に分ける。"
            "後半が終わる前にmoshiの長い返答を置かない。"
        ),
        "smooth_turn_taking": (
            "通常の交互対話にする。timingはすべてsequentialで、重なりを入れない。"
        ),
        "backchannel": (
            "長めのuser発話の直後に短いmoshi相槌を置き、"
            'その相槌へ "timing":"overlap_previous", '
            '"start_after_previous_start_sec":1.0〜2.5, '
            '"event":"model_backchannel" を付ける。その後userが続きを話す。'
        ),
        "user_interruption": (
            "moshiのやや長い発話の直後にuserの訂正または重要情報を置き、"
            'user側へ "timing":"overlap_previous", '
            '"start_after_previous_start_sec":0.8〜1.8, '
            '"truncate_previous_after_sec":0.15〜0.35, '
            '"event":"user_interruption" を付ける。その次にmoshiが訂正内容へ応答する。'
        ),
        "user_backchannel": (
            "moshiの説明の直後にuserの「はい」「うん」などを置き、"
            'user側へ "timing":"overlap_previous", '
            '"start_after_previous_start_sec":0.8〜1.8, '
            '"event":"user_backchannel" を付ける。truncate_previous_after_secは付けず、'
            "次のmoshi発話で同じ説明を自然に続ける。"
        ),
        "talking_to_other": (
            "moshiの発話の直後にuserが近くの人へ向けた短い発話を置き、"
            'user側へ "timing":"overlap_previous", '
            '"start_after_previous_start_sec":0.8〜1.8, '
            '"voice_role":"user", "event":"talking_to_other" を付ける。'
            "次のmoshi発話は横の人への発話へ直接回答せず、元の相談を続ける。"
        ),
        "background_speech": (
            "moshiの発話の直後にニュースまたは駅案内の短い背景発話をuser側として置き、"
            'その背景発話へ "timing":"overlap_previous", '
            '"start_after_previous_start_sec":0.8〜1.8, '
            '"voice_role":"background", "gain":0.25〜0.4, '
            '"event":"background_speech" を付ける。'
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
- overlap_previous は必ず直前の音声ターンに重ねる。
- timingを省略したターンはsequentialとして扱う。
- gainは通常1.0。背景発話以外を不必要に小さくしない。
""".rstrip()


def build_gemma_prompt(use_case: dict[str, Any], rng: random.Random) -> str:
    turn_count = int(use_case.get("target_turns", rng.choice([6, 8, 10])))
    user_style = rng.choice(
        [
            "言い淀みが少しある",
            "短い文が多い",
            "最初は雑談っぽい",
            "気持ちをうまく言葉にできない",
        ]
    )
    counselor_style = rng.choice(
        [
            "温かく、でも踏み込みすぎない",
            "相づちを自然に入れる",
            "解決策を急がず、気持ちを整理する",
            "一問一答になりすぎず会話を続ける",
        ]
    )

    personality = str(use_case.get("personality") or "ふつう")
    personality_guidance = str(use_case.get("personality_guidance") or "自然に話す。")
    emotional_state = str(use_case.get("emotional_state") or "落ち着いている")
    emotional_state_hint = str(use_case.get("emotional_state_hint") or "")

    duplex_task = str(use_case.get("duplex_task") or "")
    silence_pattern = str(use_case.get("silence_pattern", "none"))
    if duplex_task == "pause_handling":
        silence_directive = (
            '- 2〜3秒の沈黙ターン {"speaker":"silence","duration_sec":2〜3,'
            '"note":"userが言葉を探している"} を入れ、その直後はuserが同じ発話を続ける。'
        )
    elif silence_pattern == "heavy":
        silence_directive = (
            '- 沈黙のターン {"speaker":"silence","duration_sec":3〜6,"note":"..."} を 3〜5 回挟む。'
            "沈黙の直後は必ず moshi が穏やかに声をかける（moshi が連続して話してもよい）。"
        )
    elif silence_pattern == "occasional":
        silence_directive = (
            '- 沈黙のターン {"speaker":"silence","duration_sec":2〜4,"note":"..."} を 1〜2 回挟む。'
            "沈黙の直後は moshi が穏やかに声をかける。"
        )
    else:
        silence_directive = "- 沈黙ターンは入れない。"
    duplex_section = build_duplex_prompt_section(use_case)
    schema_turn_examples = {
        "pause_handling": """
    {"speaker": "user", "emotion": "hesitant", "text": "発話の前半"},
    {"speaker": "silence", "duration_sec": 2.4, "note": "userが言葉を探している"},
    {"speaker": "user", "emotion": "sad", "text": "同じ発話の続き"},
    {"speaker": "moshi", "emotion": "empathetic", "text": "続きを受け止める応答"}
""".strip(),
        "smooth_turn_taking": """
    {"speaker": "user", "emotion": "hesitant", "text": "相談者の発話"},
    {"speaker": "moshi", "emotion": "warm", "text": "発話終了後の自然な応答"}
""".strip(),
        "backchannel": """
    {"speaker": "user", "emotion": "sad", "text": "長めの相談者発話"},
    {"speaker": "moshi", "emotion": "gentle", "text": "うん。",
      "timing": "overlap_previous", "start_after_previous_start_sec": 1.5,
      "event": "model_backchannel"},
    {"speaker": "user", "emotion": "sad", "text": "相談者の続き"},
    {"speaker": "moshi", "emotion": "empathetic", "text": "内容を受け止める応答"}
""".strip(),
        "user_interruption": """
    {"speaker": "user", "emotion": "hesitant", "text": "相談者の発話"},
    {"speaker": "moshi", "emotion": "warm", "text": "やや長い相談員の発話"},
    {"speaker": "user", "emotion": "neutral", "text": "訂正または重要情報",
      "timing": "overlap_previous", "start_after_previous_start_sec": 1.2,
      "truncate_previous_after_sec": 0.25, "event": "user_interruption"},
    {"speaker": "moshi", "emotion": "empathetic", "text": "割り込み内容への応答"}
""".strip(),
        "user_backchannel": """
    {"speaker": "user", "emotion": "hesitant", "text": "相談者の発話"},
    {"speaker": "moshi", "emotion": "warm", "text": "相談員の説明"},
    {"speaker": "user", "emotion": "neutral", "text": "はい。",
      "timing": "overlap_previous", "start_after_previous_start_sec": 1.0,
      "event": "user_backchannel"},
    {"speaker": "moshi", "emotion": "warm", "text": "同じ話題の自然な続き"}
""".strip(),
        "talking_to_other": """
    {"speaker": "user", "emotion": "hesitant", "text": "相談者の発話"},
    {"speaker": "moshi", "emotion": "warm", "text": "相談員の発話"},
    {"speaker": "user", "emotion": "neutral", "text": "少し待ってて。",
      "timing": "overlap_previous", "start_after_previous_start_sec": 1.0,
      "voice_role": "user", "event": "talking_to_other"},
    {"speaker": "moshi", "emotion": "warm", "text": "元の相談を維持する応答"}
""".strip(),
        "background_speech": """
    {"speaker": "user", "emotion": "hesitant", "text": "相談者の発話"},
    {"speaker": "moshi", "emotion": "warm", "text": "相談員の発話"},
    {"speaker": "user", "emotion": "neutral", "text": "まもなく電車が到着します。",
      "timing": "overlap_previous", "start_after_previous_start_sec": 1.0,
      "gain": 0.32, "voice_role": "background", "event": "background_speech"},
    {"speaker": "moshi", "emotion": "warm", "text": "元の相談を維持する応答"}
""".strip(),
    }
    schema_turn_example = schema_turn_examples.get(
        duplex_task,
        """
    {"speaker": "user", "emotion": "hesitant", "text": "相談者の発話"},
    {"speaker": "moshi", "emotion": "warm", "text": "相談員の発話"},
    {"speaker": "silence", "duration_sec": 3.0, "note": "なぜ沈黙か簡単に"}
""".strip(),
    )

    return f"""
あなたは日本語の合成学習データを作る対話ライターです。
目的は「孤独・孤立相談窓口」で使う、自然な雑談寄りの相談会話を作ることです。

次のユースケースに基づいて、相談者 user と相談員 moshi の会話を作ってください。

use_case:
{json.dumps(use_case, ensure_ascii=False, indent=2)}

制約:
- 出力は JSON オブジェクトのみ。説明文や Markdown は入れない。
- speaker は "user" / "moshi" / "silence" のいずれか。
- 最初の turn は user。
- 通常ターン（user/moshi）は {turn_count} 件前後。沈黙は別カウント。
- user の話し方: {user_style}。
- moshi の話し方: {counselor_style}。
- user の性格: {personality}（{personality_guidance}）。この性格が話し方や語彙に一貫して出るようにする。
- user の今日の状態: {emotional_state}。{emotional_state_hint} 会話を通してこの状態を反映し、場面で感情が動くと emotion も変える。
- moshi は来訪を歓迎し、孤独感を軽く扱わず、すぐ解決策だけを出さない。
- high/medium risk のニュアンスがある場合は、断定せず安全確認につながる短い問いを入れる。
- 診断、説教、長い助言、緊急対応を装う表現は避ける。
- 1 turn は音声化しやすい長さ、だいたい 8〜45 文字。
- 各 user/moshi ターンには "emotion" を必ず付ける。候補:
  user 側: hesitant, sad, lonely, anxious, relieved, grateful, neutral,
           tearful, sobbing, high_tension, agitated, withdrawn, weary, irritable, laughing
  moshi 側: warm, gentle, empathetic, encouraging, concerned, reassuring, soothing, neutral
  user の emotion は上記「今日の状態」に沿って選ぶ（例: 涙ぐんでいる→tearful/sobbing、ハイテンション→high_tension/laughing）。
{silence_directive}
{duplex_section}

JSON schema:
{{
  "id": "{use_case.get("id", "dialogue")}",
  "category": "{use_case.get("category", "unknown")}",
  "risk_level": "{use_case.get("risk_level", "low")}",
  "duplex_task": "{use_case.get("duplex_task", "")}",
  "title": "短いタイトル",
  "turns": [
{schema_turn_example}
  ],
  "generator_notes": "多様性や注意点を一文で"
}}
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
                "うん。",
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
        generator_notes="Gemma fallback template.",
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


def generate_dialogues(args: argparse.Namespace) -> list[Dialogue]:
    if args.dialogues_jsonl is not None:
        logger.info("Loading dialogues from %s", args.dialogues_jsonl)
        return load_dialogues_from_jsonl(args.dialogues_jsonl)[: args.num_dialogues]

    rng = random.Random(args.seed)
    use_cases = load_use_cases(args)[: args.num_dialogues]
    generator = GemmaDialogueGenerator(args)
    dialogues: list[Dialogue] = []
    if args.gemma_backend == "transformers-subprocess":
        prompts = [build_gemma_prompt(use_case, rng) for use_case in use_cases]
        try:
            texts = generator.generate_transformers_subprocess_batch(prompts)
            if len(texts) != len(use_cases):
                raise RuntimeError(
                    f"Gemma returned {len(texts)} dialogues for "
                    f"{len(use_cases)} use cases."
                )
            for use_case, text in zip(use_cases, texts):
                data = extract_json_object(text)
                dialogue = dialogue_from_mapping(data, use_case)
                logger.info("Generated dialogue with Gemma: %s", dialogue.id)
                dialogues.append(dialogue)
            return dialogues
        except Exception as exc:
            if args.no_template_fallback or not args.allow_template_fallback:
                raise
            logger.warning(
                "Gemma batch generation failed; using template fallback: %s",
                exc,
            )
            return [template_dialogue(use_case) for use_case in use_cases]

    for use_case in use_cases:
        try:
            dialogue = generator.generate(use_case, rng)
            logger.info("Generated dialogue with Gemma: %s", dialogue.id)
        except Exception as exc:
            if args.no_template_fallback or not args.allow_template_fallback:
                raise
            logger.warning(
                "Gemma generation failed for %s; using template fallback: %s",
                use_case.get("id"),
                exc,
            )
            dialogue = template_dialogue(use_case)
        dialogues.append(dialogue)
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
        "transcript_source": "gemma_script",
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
        "transcript_source": "gemma_script",
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
        "gemma_model": args.gemma_model,
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


def write_dialogues_only(args: argparse.Namespace, dialogues: list[Dialogue]) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogues_path = out_dir / "dialogues.jsonl"
    if dialogues_path.exists():
        dialogues_path.unlink()
    for dialogue in dialogues:
        append_jsonl(dialogues_path, dialogue_to_json(dialogue))
    logger.info("dialogues-only mode: wrote %d dialogues to %s", len(dialogues), dialogues_path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dialogues = generate_dialogues(args)
    if args.mode == "dialogues-only":
        write_dialogues_only(args, dialogues)
        return
    render_dataset(args, dialogues)


if __name__ == "__main__":
    main()
