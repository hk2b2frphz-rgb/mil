#!/usr/bin/env python3
"""Persistent audio-in/audio-out worker for Qwen2.5-Omni.

One JSON request is read per stdin line and one JSON result is written per
stdout line.  Model stderr stays separate so the framing remains reliable.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from typing import Any

import numpy as np


QWEN_AUDIO_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="float16")
    parser.add_argument("--speaker", choices=["Chelsie", "Ethan"], default="Chelsie")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except ImportError as exc:
        raise SystemExit(
            "Qwen2.5-Omni dependencies are missing. Run `uv sync --project gemma_runtime` "
            "after pulling this change (it installs qwen-omni-utils), and use a Transformers "
            "version that provides Qwen2_5OmniForConditionalGeneration."
        ) from exc
    dtype: Any = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    print(f"[qwen25-omni] loading {args.model}", file=sys.stderr, flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            audio_path = str(request["audio_path"])
            counselor_prompt = str(request.get("prompt", ""))
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": f"{QWEN_AUDIO_SYSTEM}\n\n{counselor_prompt}"}]},
                {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False,
            ).to(model.device)
            # V100 must use fp16 rather than bf16; do not cast integer token
            # tensors, only floating point feature tensors.
            if args.dtype != "auto":
                target_dtype = getattr(torch, args.dtype)
                inputs = {key: value.to(target_dtype) if getattr(value, "is_floating_point", lambda: False)() else value
                          for key, value in inputs.items()}
            with torch.inference_mode():
                text_ids, audio = model.generate(
                    **inputs, return_audio=True, speaker=args.speaker,
                    max_new_tokens=args.max_new_tokens, use_audio_in_video=False,
                )
            reply = processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            audio_np = audio.reshape(-1).detach().float().cpu().numpy()
            pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            result = {"text": reply, "audio_pcm16_b64": base64.b64encode(pcm16).decode("ascii"), "sample_rate": 24000}
        except Exception as exc:  # keep the parent batch alive and report the actual failure
            result = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
