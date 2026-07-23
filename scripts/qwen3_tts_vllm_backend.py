"""Batched Qwen3-TTS CustomVoice inference through vLLM-Omni.

This module intentionally imports vLLM-Omni lazily.  The regular qwen-tts
pipeline remains usable in the project's normal environment, while production
jobs can opt into a separate vLLM-Omni Python 3.12 environment.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    speaker_role: str
    instruct: str | None = None
    speaker_override: str | None = None


class VLLMQwen3TTS:
    """Qwen3-TTS CustomVoice backend with true batched inference."""

    def __init__(
        self,
        *,
        model_id: str,
        dtype_str: str,
        speaker_user: str,
        speaker_moshi: str,
        language: str,
        instruct_user: str | None,
        instruct_moshi: str | None,
        batch_size: int,
        stage_configs_path: Path,
        max_new_tokens: int = 2048,
        bucket_by_length: bool = True,
    ) -> None:
        if batch_size < 1 or batch_size & (batch_size - 1):
            raise ValueError("vLLM-Omni batch_size must be a positive power of two")
        if dtype_str != "float16":
            raise ValueError(
                "The V100 production backend requires --dtype float16 "
                "(Volta does not support bfloat16)."
            )
        if "CustomVoice" not in model_id:
            raise ValueError("The vLLM backend currently supports Qwen3-TTS CustomVoice models only")

        self.model_id = model_id
        self.dtype_str = dtype_str
        self.speaker_user = speaker_user
        self.speaker_moshi = speaker_moshi
        self.language = language
        self.instruct_user = instruct_user
        self.instruct_moshi = instruct_moshi
        self.batch_size = batch_size
        self.stage_configs_path = Path(stage_configs_path)
        self.max_new_tokens = max_new_tokens
        self.bucket_by_length = bucket_by_length
        self.sample_rate = 0
        self._omni: Any = None
        self._tokenizer: Any = None
        self._talker_config: Any = None
        self._perf_batches = 0
        self._perf_requests = 0
        self._perf_audio_sec = 0.0
        self._perf_inference_sec = 0.0

    def load(self) -> None:
        if self._omni is not None:
            return
        if not self.stage_configs_path.is_file():
            raise FileNotFoundError(
                f"vLLM-Omni stage config not found: {self.stage_configs_path}"
            )

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from transformers import AutoTokenizer
            from vllm_omni import Omni
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
                Qwen3TTSConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "vLLM-Omni is required for --qwen-engine vllm-omni. "
                "Use the dedicated Python 3.12 environment documented in README.md."
            ) from exc

        logger.info(
            "Loading Qwen3-TTS with vLLM-Omni: model=%s batch=%d config=%s",
            self.model_id,
            self.batch_size,
            self.stage_configs_path,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        config = Qwen3TTSConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self._talker_config = getattr(config, "talker_config", None)
        self._omni = Omni(
            model=self.model_id,
            stage_configs_path=str(self.stage_configs_path),
            log_stats=False,
            batch_timeout=5,
            init_timeout=600,
            stage_init_timeout=600,
        )
        logger.info("Qwen3-TTS vLLM-Omni engine ready")

    def close(self) -> None:
        if self._omni is not None:
            close = getattr(self._omni, "close", None)
            if callable(close):
                close()
            self._omni = None

    def resolve_instruct(self, speaker_role: str, instruct: str | None) -> str | None:
        if instruct:
            return instruct
        return self.instruct_user if speaker_role == "user" else self.instruct_moshi

    def _to_prompt(self, request: SynthesisRequest) -> dict[str, Any]:
        assert self._tokenizer is not None
        voice = request.speaker_override or (
            self.speaker_user if request.speaker_role == "user" else self.speaker_moshi
        )
        additional_information = {
            "task_type": ["CustomVoice"],
            "text": [request.text],
            "language": [self.language],
            "speaker": [voice],
            "instruct": [self.resolve_instruct(request.speaker_role, request.instruct) or ""],
            "max_new_tokens": [self.max_new_tokens],
        }
        prompt_len = self._estimate_prompt_len(additional_information)
        return {
            "prompt_token_ids": [0] * prompt_len,
            "additional_information": additional_information,
        }

    def _estimate_prompt_len(self, additional_information: dict[str, Any]) -> int:
        try:
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
                Qwen3TTSPromptEmbedsBuilder,
            )

            return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type="CustomVoice",
                tokenize_prompt=lambda text: self._tokenizer(text, padding=False)["input_ids"],
                codec_language_id=getattr(self._talker_config, "codec_language_id", None),
                spk_is_dialect=getattr(self._talker_config, "spk_is_dialect", None),
                estimate_ref_code_len=lambda _audio: None,
            )
        except Exception as exc:
            logger.warning("Prompt length estimation failed; using 2048: %s", exc)
            return 2048

    @staticmethod
    def _request_output(stage_output: Any) -> Any:
        return getattr(stage_output, "request_output", stage_output)

    @staticmethod
    def _audio_from_output(request_output: Any) -> tuple[np.ndarray, int]:
        mm = request_output.outputs[0].multimodal_output
        if not isinstance(mm, dict) or "audio" not in mm or "sr" not in mm:
            raise RuntimeError("vLLM-Omni returned no audio multimodal output")

        sr_raw = mm["sr"]
        if isinstance(sr_raw, list):
            sr_raw = sr_raw[-1]
        sr = int(sr_raw.item() if hasattr(sr_raw, "item") else sr_raw)

        audio_raw = mm["audio"]
        chunks = audio_raw if isinstance(audio_raw, list) else [audio_raw]
        arrays: list[np.ndarray] = []
        for chunk in chunks:
            if hasattr(chunk, "detach"):
                chunk = chunk.detach()
            if hasattr(chunk, "float"):
                chunk = chunk.float()
            if hasattr(chunk, "cpu"):
                chunk = chunk.cpu()
            if hasattr(chunk, "numpy"):
                chunk = chunk.numpy()
            arrays.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        if not arrays:
            raise RuntimeError("vLLM-Omni returned an empty audio chunk list")
        return np.concatenate(arrays), sr

    def _generate_batch(self, requests: Sequence[SynthesisRequest]) -> list[np.ndarray]:
        assert self._omni is not None
        prompts = [self._to_prompt(request) for request in requests]
        started = time.perf_counter()
        stage_outputs = self._omni.generate(prompts, use_tqdm=False)
        ordered: list[np.ndarray | None] = [None] * len(requests)

        for stage_output in stage_outputs:
            output = self._request_output(stage_output)
            request_id = str(output.request_id)
            try:
                request_index = int(request_id.split("_", 1)[0])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Unexpected vLLM-Omni request id: {request_id!r}") from exc
            audio, sr = self._audio_from_output(output)
            if self.sample_rate == 0:
                self.sample_rate = sr
            elif sr != self.sample_rate:
                raise RuntimeError(
                    f"Inconsistent vLLM-Omni sample rate: {sr} != {self.sample_rate}"
                )
            ordered[request_index] = audio

        missing = [i for i, audio in enumerate(ordered) if audio is None]
        if missing:
            raise RuntimeError(f"vLLM-Omni produced no final audio for request(s): {missing}")
        result = [audio for audio in ordered if audio is not None]
        inference_sec = time.perf_counter() - started
        audio_sec = sum(audio.size for audio in result) / self.sample_rate
        self._perf_batches += 1
        self._perf_requests += len(result)
        self._perf_audio_sec += audio_sec
        self._perf_inference_sec += inference_sec
        audio_x = audio_sec / inference_sec if inference_sec > 0 else 0.0
        rtf = inference_sec / audio_sec if audio_sec > 0 else 0.0
        logger.info(
            "vLLM-Omni batch complete: requests=%d audio_sec=%.2f "
            "wall_sec=%.2f audio_x=%.3f rtf=%.3f",
            len(result),
            audio_sec,
            inference_sec,
            audio_x,
            rtf,
        )
        return result

    def performance_stats(self) -> dict[str, float | int]:
        """Return cumulative engine-only generation metrics for this process."""
        audio_x = (
            self._perf_audio_sec / self._perf_inference_sec
            if self._perf_inference_sec > 0
            else 0.0
        )
        rtf = (
            self._perf_inference_sec / self._perf_audio_sec
            if self._perf_audio_sec > 0
            else 0.0
        )
        return {
            "batches": self._perf_batches,
            "requests": self._perf_requests,
            "audio_sec": self._perf_audio_sec,
            "inference_wall_sec": self._perf_inference_sec,
            "audio_x": audio_x,
            "rtf": rtf,
        }

    def synthesize_many(self, requests: Sequence[SynthesisRequest]) -> list[np.ndarray]:
        self.load()
        if not requests:
            return []

        # Similar lengths share a batch to reduce padding and long-tail stalls.
        order = list(range(len(requests)))
        if self.bucket_by_length:
            order.sort(key=lambda i: len(requests[i].text))
        result: list[np.ndarray | None] = [None] * len(requests)

        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            batch = [requests[i] for i in indices]
            logger.info(
                "vLLM-Omni TTS batch: size=%d chars=%d..%d",
                len(batch),
                min(len(req.text) for req in batch),
                max(len(req.text) for req in batch),
            )
            for original_index, audio in zip(indices, self._generate_batch(batch)):
                result[original_index] = audio

        return [audio for audio in result if audio is not None]

    def synthesize(
        self,
        text: str,
        speaker_role: str,
        instruct: str | None = None,
        speaker_override: str | None = None,
    ) -> np.ndarray:
        return self.synthesize_many(
            [SynthesisRequest(text, speaker_role, instruct, speaker_override)]
        )[0]
