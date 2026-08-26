#!/usr/bin/env python3
from __future__ import annotations
"""
Shared components for local cascade / SpeechLLM comparison baselines.

Both baselines are turn-based (they only respond once the full input audio
is available), unlike the full-duplex Moshi system under test. They exist
to get genuinely comparable numbers -- latency, judge scores, safety -- by
reusing the *same* eval_sets and the *same* judge/summarize pipeline, not
to reimplement full-duplex behavior.

Pipeline pieces:
- ASR (cascade only): faster-whisper, in-process. faster-whisper has no
  transformers dependency, so it runs fine in the main project environment
  without conflicting with Moshi's pinned versions.
- LLM (cascade): reuses scripts/gemma_dialogue_worker.py via subprocess in
  the isolated gemma_runtime/ uv environment -- the same pattern
  generate_synthetic_moshi_training_data.py already uses to keep Moshi's
  huggingface-hub/transformers pins untouched.
- SpeechLLM: scripts/speechllm_worker.py (Qwen2-Audio), same subprocess
  pattern, taking raw audio instead of an ASR transcript.
- TTS: reuses Qwen3TTS via eval/build_full_duplex_ja_dataset.py's
  initialize_tts()/synthesize(), with the same backend and controlled preset
  voices used by the benchmark audio.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVAL_DIR = Path(__file__).resolve().parent
for _dir in (SCRIPTS_DIR, EVAL_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from build_full_duplex_ja_dataset import initialize_tts, synthesize  # noqa: E402
from full_duplex_audio import write_wav_mono  # noqa: E402

def _load_counselor_prompt() -> str:
    """相談員プロンプトの正本(scripts/counselor_prompts.py)を読む。

    学習データ生成が使っているものと同じ文言をカスケードにも与える。別々の
    プロンプトで比べると、測っているものがアーキテクチャの差ではなく指示文の
    差になる。長さの指示(1発話は必ず1文)もここに入っているので、
    max_new_tokens で長さを作る必要がない。
    """
    import importlib.util

    path = SCRIPTS_DIR / "counselor_prompts.py"
    spec = importlib.util.spec_from_file_location("_miltoka_counselor_prompts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MOSHI_AGENT_SYSTEM_PROMPT


COUNSELOR_SYSTEM_PROMPT = _load_counselor_prompt()

_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


def worker_environment() -> dict[str, str]:
    """Return a subprocess environment with a usable, normalized proxy only.

    Cascade launches an isolated Python runtime for Gemma/SpeechLLM.  Unlike
    the in-process Moshi path, that runtime eagerly initializes HTTP clients,
    so a proxy inherited from a PBS submission host must be normalized before
    it is passed to the child. The PBS wrapper's established proxy route is
    preserved; only malformed values are discarded.
    """
    environment = os.environ.copy()
    proxy_setting = next(
        ((key, environment[key]) for key in _PROXY_KEYS if environment.get(key)),
        None,
    )
    if proxy_setting is None:
        return environment
    proxy_source, raw_proxy = proxy_setting
    try:
        parsed = urlsplit(
            raw_proxy if "://" in raw_proxy else f"http://{raw_proxy}"
        )
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("missing http(s) scheme or hostname")
        _ = parsed.port
        normalized = parsed.geturl()
    except (OSError, ValueError) as exc:
        for key in (*_PROXY_KEYS, "all_proxy", "ALL_PROXY"):
            environment.pop(key, None)
        print(
            "[cascade] proxy from "
            f"{proxy_source} is malformed ({exc}); "
            "starting the worker without a proxy.",
            file=sys.stderr,
        )
        return environment
    for key in _PROXY_KEYS:
        environment[key] = normalized
    environment.pop("all_proxy", None)
    environment.pop("ALL_PROXY", None)
    return environment


def looks_truncated(text: str) -> bool:
    """生成が上限で打ち切られた可能性が高いかを、文末の形から推定する。

    ワーカーは終了理由を返さないので、これは推定でしかない。言い切っていない
    応答をそのまま TTS に流すと音声も途中で終わるため、件数だけは把握できる
    ようにしておく。
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] in "。．.!！?？」』）)…~〜ー":
        return False
    # 句点が無いだけの言い切り(「大変でしたね」)は珍しくないので、それだけでは
    # 打ち切りとみなさない。文が続くはずの形で終わっているものだけを拾う。
    if stripped[-1] in "、,":
        return True
    _CONNECTIVE_TAILS = (
        "て", "で", "が", "けど", "けれど", "ので", "から", "し", "たり",
        "とか", "など", "への", "への", "という", "といった", "ば",
        "を", "に", "へ", "と", "は", "も", "の",
    )
    return stripped.endswith(_CONNECTIVE_TAILS)


def validate_spoken_response(text: str) -> str:
    """Return assistant body text safe to pass to TTS, or fail on prompt leak."""
    cleaned = re.sub(r"<\|[^|>]+\|>", "", text).strip()
    counselor_parts = re.split(r"(?:相談員|assistant|アシスタント)\s*[:：]", cleaned, flags=re.I)
    if len(counselor_parts) > 1:
        cleaned = counselor_parts[-1].strip()
    cleaned = re.sub(
        r"^(?:相談員|assistant|アシスタント)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    if re.search(r"(?:利用者|ユーザー|user)\s*[:：]", cleaned, flags=re.I):
        raise RuntimeError("LLM response contains leaked user/prompt text; refusing TTS")
    return cleaned


def init_baseline_tts(
    tts_backend: str,
    tts_model: str,
    tts_speaker: str,
    tts_language: str,
    device: str,
    dtype: str,
) -> tuple[str, Any | None]:
    """Wraps build_full_duplex_ja_dataset.initialize_tts with a minimal
    args-like shim, so baselines synthesize with the identical TTS path
    already used to build eval input audio."""

    class _Args:
        pass

    ns = _Args()
    ns.tts_backend = tts_backend
    ns.tts_model = tts_model
    ns.tts_speaker = tts_speaker
    ns.tts_language = tts_language
    ns.device = device
    ns.dtype = dtype
    ns.tts_instruct = None
    return initialize_tts(ns)


def resolve_llm_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    runtime_dir = REPO_ROOT / "gemma_runtime" / ".venv"
    for candidate in (runtime_dir / "bin" / "python", runtime_dir / "Scripts" / "python.exe"):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def call_subprocess_worker(
    worker: Path,
    payload: dict[str, Any],
    extra_args: list[str],
    python_exe: str,
    timeout_sec: float,
) -> dict[str, Any]:
    if not worker.is_file():
        raise FileNotFoundError(f"Worker script not found: {worker}")
    command = [python_exe, str(worker), *extra_args]
    try:
        proc = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=None,  # let stderr flow to this process's terminal
            timeout=timeout_sec,
            check=False,
            env=worker_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Worker timed out after {timeout_sec}s: {' '.join(command)}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "Worker subprocess failed. Install its runtime with "
            "`uv sync --project gemma_runtime`, or pass a different --llm-python.\n"
            f"command: {' '.join(command)}\n"
            "(stderr output is shown above)"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Worker did not return JSON.\nstdout:\n{proc.stdout}"
        ) from exc


class GemmaLLM:
    """Cascade response generator: reuses scripts/gemma_dialogue_worker.py."""

    def __init__(
        self,
        model: str = "google/gemma-2-2b-it",
        python_exe: str | None = None,
        device_map: str = "auto",
        dtype: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 200,
        timeout_sec: float = 300.0,
    ):
        self.model = model
        self.python_exe = resolve_llm_python(python_exe)
        self.device_map = device_map
        self.dtype = dtype
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.timeout_sec = timeout_sec
        self.worker = SCRIPTS_DIR / "gemma_dialogue_worker.py"
        self._process: subprocess.Popen[str] | None = None
        self.startup_wall_time_sec = 0.0

    def _readline(self) -> str:
        assert self._process is not None and self._process.stdout is not None
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._process.stdout.readline)
        try:
            line = future.result(timeout=self.timeout_sec)
        except FutureTimeoutError as exc:
            self.close()
            raise RuntimeError(
                f"Persistent Gemma worker timed out after {self.timeout_sec}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if not line:
            return_code = self._process.poll()
            raise RuntimeError(
                f"Persistent Gemma worker exited unexpectedly (code={return_code})"
            )
        return line

    def load(self) -> None:
        if self._process is not None:
            return
        command = [
            self.python_exe,
            str(self.worker),
            "--model", self.model,
            "--task", "text-generation",
            "--device-map", self.device_map,
            "--dtype", self.dtype,
            "--temperature", str(self.temperature),
            "--top-p", str(self.top_p),
            "--max-new-tokens", str(self.max_new_tokens),
            "--serve",
        ]
        started = time.perf_counter()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=worker_environment(),
        )
        ready = json.loads(self._readline())
        if ready.get("ready") is not True:
            self.close()
            raise RuntimeError(f"Gemma worker did not become ready: {ready}")
        self.startup_wall_time_sec = time.perf_counter() - started

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def respond(
        self,
        user_text: str,
        system_prompt: str = COUNSELOR_SYSTEM_PROMPT,
        seed: int = 0,
    ) -> tuple[str, float]:
        prompt = f"{system_prompt}\n\n利用者: {user_text}\n相談員:"
        self.load()
        assert self._process is not None and self._process.stdin is not None
        started = time.perf_counter()
        self._process.stdin.write(
            json.dumps({"prompt": prompt, "seed": seed}, ensure_ascii=False) + "\n"
        )
        self._process.stdin.flush()
        result = json.loads(self._readline())
        if "error" in result:
            raise RuntimeError(f"Gemma worker generation failed: {result['error']}")
        wall_time = time.perf_counter() - started
        return validate_spoken_response(str(result.get("text", ""))), wall_time


class SpeechLLM:
    """SpeechLLM response generator: reuses scripts/speechllm_worker.py (Qwen2-Audio)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2-Audio-7B-Instruct",
        python_exe: str | None = None,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 200,
        timeout_sec: float = 300.0,
    ):
        self.model = model
        self.python_exe = resolve_llm_python(python_exe)
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.timeout_sec = timeout_sec
        self.worker = SCRIPTS_DIR / "speechllm_worker.py"

    def respond(
        self,
        pcm: np.ndarray,
        sample_rate: int,
        prompt: str = COUNSELOR_SYSTEM_PROMPT,
    ) -> tuple[str, float]:
        tmp_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            write_wav_mono(tmp_path, pcm, sample_rate)
            started = time.perf_counter()
            result = call_subprocess_worker(
                self.worker,
                {"audio_path": str(tmp_path), "prompt": prompt},
                [
                    "--model", self.model,
                    "--device-map", self.device_map,
                    "--dtype", self.dtype,
                    "--max-new-tokens", str(self.max_new_tokens),
                ],
                self.python_exe,
                self.timeout_sec,
            )
            wall_time = time.perf_counter() - started
        finally:
            tmp_path.unlink(missing_ok=True)
        return str(result.get("text", "")).strip(), wall_time


class LocalASR:
    """faster-whisper wrapper. Runs in-process: no transformers dependency,
    so no conflict with Moshi's pinned versions."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ja",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required for the cascade ASR step. It is a "
                "declared pyproject.toml dependency, so run `uv sync` (or launch "
                "via `uv run`) to install it into the project environment. A bare "
                "`pip install faster-whisper` may land in a different environment "
                "than the one `uv run` uses."
            ) from exc
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> tuple[str, float]:
        # 入力の書き起こしに要るのはテキストだけ。単語タイムスタンプは
        # faster-whisper でかなり重く、ここでは計算しても捨てていた。
        text, _chunks, wall_time = self.transcribe_aligned(
            pcm, sample_rate, word_timestamps=False
        )
        return text, wall_time

    def transcribe_aligned(
        self, pcm: np.ndarray, sample_rate: int, word_timestamps: bool = True
    ) -> tuple[str, list[dict[str, Any]], float]:
        """Return an ASR transcript with the word/segment timing contract
        consumed by the upstream Full-Duplex-Bench evaluators."""
        from full_duplex_audio import resample_linear

        self.load()
        assert self._model is not None
        pcm_16k = resample_linear(np.asarray(pcm, dtype=np.float32), sample_rate, 16000)
        started = time.perf_counter()
        segments, _info = self._model.transcribe(
            pcm_16k, language=self.language, vad_filter=True,
            word_timestamps=word_timestamps,
        )
        chunks: list[dict[str, Any]] = []
        for segment in segments:
            # Whisper's Japanese word timestamps are the closest available
            # alignment to the synthesized waveform.  A segment fallback is
            # retained only for a backend/model version that cannot expose
            # word timings; callers never manufacture character timestamps.
            words = getattr(segment, "words", None) or []
            word_chunks = [
                {
                    "text": str(word.word).strip(),
                    "timestamp": [round(float(word.start), 4), round(float(word.end), 4)],
                }
                for word in words
                if str(getattr(word, "word", "")).strip()
                and getattr(word, "start", None) is not None
                and getattr(word, "end", None) is not None
            ]
            if word_chunks:
                chunks.extend(word_chunks)
            elif segment.text.strip() and segment.end is not None:
                chunks.append(
                    {
                        "text": segment.text.strip(),
                        "timestamp": [round(float(segment.start), 4), round(float(segment.end), 4)],
                    }
                )
        text = "".join(chunk["text"] for chunk in chunks).strip()
        wall_time = time.perf_counter() - started
        return text, chunks, wall_time
