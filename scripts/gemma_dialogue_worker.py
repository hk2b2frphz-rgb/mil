#!/usr/bin/env python3
from __future__ import annotations
"""
Small subprocess worker for LLM dialogue generation.

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
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Load the model once, then process one JSON request per stdin line.",
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


def result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        if not result:
            return ""
        # Chat pipelines may return the complete conversation. Never join the
        # user prompt into the assistant text that will be sent to TTS.
        assistant_messages = [
            item
            for item in result
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if assistant_messages:
            return result_to_text(assistant_messages[-1].get("content", ""))
        if len(result) == 1:
            return result_to_text(result[0])
        return "\n".join(result_to_text(item) for item in result)
    if isinstance(result, dict):
        for key in ("generated_text", "text", "content"):
            if key in result:
                return result_to_text(result[key])
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def progress(msg: str) -> None:
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


def build_pipeline(args: argparse.Namespace) -> Any:
    progress(f"モデルをロード中: {args.model}")
    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "model": args.model,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        # SDPA は V100/A100 どちらでも使え、attention をネイティブ実装にして
        # 推論を 1.2〜1.5x 程度高速化する。
        "model_kwargs": {"attn_implementation": "sdpa"},
    }
    dtype = torch_dtype(args.dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype

    def _build_pipeline(task: str) -> Any:
        try:
            return pipeline(task, **kwargs)
        except (TypeError, ValueError):
            # 一部のモデルは model_kwargs/attn_implementation を受け付けない
            fallback_kwargs = {k: v for k, v in kwargs.items() if k != "model_kwargs"}
            return pipeline(task, **fallback_kwargs)

    try:
        result = _build_pipeline(args.task)
    except Exception:
        if args.task == "text-generation":
            raise
        result = _build_pipeline("text-generation")
    progress("モデルロード完了")
    return result


def generate_one(
    pipe: Any, prompt: str, args: argparse.Namespace, seed: int
) -> str:
    from transformers import set_seed

    set_seed(seed)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "return_full_text": False,
    }
    # Gemma is text-only. A plain string content is accepted across both the
    # v4 and v5 Transformers chat templates.
    messages = [{"role": "user", "content": prompt}]
    try:
        result = pipe(messages, **generation_kwargs)
    except (TypeError, ValueError):
        result = pipe(prompt, **generation_kwargs)
    return result_to_text(result).strip()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args)

    if args.serve:
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                text = generate_one(
                    pipe,
                    str(payload["prompt"]),
                    args,
                    int(payload.get("seed", 0)),
                )
                response = {"text": text}
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(response, ensure_ascii=False), flush=True)
        return

    payload = json.loads(sys.stdin.read())
    if "prompts" in payload:
        prompts = [str(prompt) for prompt in payload["prompts"]]
    else:
        prompts = [str(payload["prompt"])]
    seeds = payload.get("seeds")
    if seeds is None:
        seeds = [int(payload.get("seed", 0))] * len(prompts)
    if len(seeds) != len(prompts):
        raise ValueError("seeds must have the same length as prompts")

    texts: list[str] = []
    for prompt, seed in zip(prompts, seeds):
        texts.append(generate_one(pipe, prompt, args, int(seed)))

    if "prompts" in payload:
        print(json.dumps({"texts": texts}, ensure_ascii=False))
    else:
        print(json.dumps({"text": texts[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
