#!/usr/bin/env python3
from __future__ import annotations
"""
Small subprocess worker for the SpeechLLM comparison baseline (Qwen2-Audio).

Takes raw audio directly (no separate ASR step) and returns a text response,
mirroring the "typical" audio-in/text-out SpeechLLM architecture used for
comparison against full-duplex Moshi and the cascade (ASR->LLM) baseline.

Runs in the isolated gemma_runtime/ environment for the same reason
gemma_dialogue_worker.py does: keep the main Moshi environment's
huggingface-hub/transformers pins untouched.

Input:  JSON on stdin: {"audio_path": "<wav path>", "prompt": "<instruction>"}
Output: JSON on stdout: {"text": "..."}
"""

import argparse
import json
import sys
import wave
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def progress(msg: str) -> None:
    print(f"[speechllm] {msg}", file=sys.stderr, flush=True)


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: only PCM16 WAV is supported, got {sample_width * 8}-bit")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sample_rate


def resample_linear(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or pcm.size == 0:
        return pcm.astype(np.float32)
    target_length = max(1, int(round(len(pcm) * target_rate / source_rate)))
    source_x = np.linspace(0.0, 1.0, num=len(pcm), endpoint=True)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(target_x, source_x, pcm).astype(np.float32)


def main() -> None:
    args = parse_args()
    payload = json.loads(sys.stdin.read())
    audio_path = str(payload["audio_path"])
    prompt = str(payload.get("prompt", ""))

    progress(f"モデルをロード中: {args.model}")
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model_kwargs: dict[str, Any] = {"device_map": args.device_map}
    dtype = torch_dtype(args.dtype)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    model.eval()

    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    pcm, source_sr = read_wav_mono(audio_path)
    audio = resample_linear(pcm, source_sr, target_sr)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=text_prompt, audios=[audio], sampling_rate=target_sr,
        return_tensors="pt", padding=True,
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    progress("生成中...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    progress("完了")
    print(json.dumps({"text": text.strip()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
