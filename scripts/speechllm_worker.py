#!/usr/bin/env python3
from __future__ import annotations
"""
Small subprocess worker for the SpeechLLM comparison baseline (Gemma 4 audio).

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
from typing import Any
from pathlib import Path

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
    parser.add_argument(
        "--serve", action="store_true",
        help="Keep the model loaded and process one JSON request per stdin line.",
    )
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


def main() -> None:
    args = parse_args()
    progress(f"モデルをロード中: {args.model}")
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model_kwargs: dict[str, Any] = {"device_map": args.device_map}
    dtype = torch_dtype(args.dtype)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    def generate(payload: dict[str, Any]) -> dict[str, str]:
        audio_path = Path(str(payload["audio_path"])).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Input audio does not exist: {audio_path}")
        prompt = str(payload.get("prompt", ""))
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            # Gemma 4's current processor schema uses `url`, not the older
            # `audio` field.  A file:// URI reaches the generic URL/base64
            # loader and can fail with "Incorrect padding"; an absolute local
            # path is recognized as a local WAV by the audio loader instead.
            {"type": "audio", "url": str(audio_path)},
        ]}]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = inputs.to(model.device)
        progress("生成中...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
        text = processor.decode(generated_ids[0], skip_special_tokens=True)
        progress("完了")
        return {"text": text.strip()}

    if args.serve:
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                result: dict[str, str] = generate(json.loads(line))
            except Exception as exc:  # keep a batch's remaining cases runnable
                result = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    print(json.dumps(generate(json.loads(sys.stdin.read())), ensure_ascii=False))


if __name__ == "__main__":
    main()
