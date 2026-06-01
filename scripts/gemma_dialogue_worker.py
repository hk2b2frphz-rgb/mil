#!/usr/bin/env python3
from __future__ import annotations
"""
Small subprocess worker for Gemma dialogue generation.

This file intentionally imports Transformers only inside the worker process so
the main Moshi environment can keep Moshi's huggingface-hub constraints.
Input:  JSON on stdin: {"prompt": "..."}
Output: JSON on stdout: {"text": "..."}
"""

import argparse
import json
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="any-to-any")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--trust-remote-code", action="store_true")
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


def result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        if not result:
            return ""
        if len(result) == 1:
            return result_to_text(result[0])
        return "\n".join(result_to_text(item) for item in result)
    if isinstance(result, dict):
        for key in ("generated_text", "text", "content"):
            if key in result:
                return result_to_text(result[key])
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def main() -> None:
    args = parse_args()
    payload = json.loads(sys.stdin.read())
    if "prompts" in payload:
        prompts = [str(prompt) for prompt in payload["prompts"]]
    else:
        prompts = [str(payload["prompt"])]

    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "model": args.model,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    dtype = torch_dtype(args.dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype

    try:
        pipe = pipeline(args.task, **kwargs)
    except Exception:
        if args.task == "text-generation":
            raise
        pipe = pipeline("text-generation", **kwargs)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True,
        "return_full_text": False,
    }
    texts: list[str] = []
    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        try:
            result = pipe(messages, **generation_kwargs)
        except TypeError:
            result = pipe(prompt, **generation_kwargs)
        texts.append(result_to_text(result))

    if "prompts" in payload:
        print(json.dumps({"texts": texts}, ensure_ascii=False))
    else:
        print(json.dumps({"text": texts[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
