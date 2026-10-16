#!/usr/bin/env python3
"""Fail before generation when the V100 runtime or Japanese JSON probe is invalid."""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from urllib.request import Request, urlopen


def runtime_errors(torch, vllm_version: str, min_memory_gib: float = 30) -> list[str]:
    errors = []
    if torch.version.cuda != "12.6":
        errors.append(f"PyTorch CUDA must be 12.6, got {torch.version.cuda}")
    if "sm_70" not in torch.cuda.get_arch_list():
        errors.append("PyTorch has no sm_70 kernels")
    if vllm_version.split("+")[0] != "0.21.0":
        errors.append(f"vLLM must be 0.21.0, got {vllm_version}")
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")
        return errors
    count = torch.cuda.device_count()
    if count != 4:
        errors.append(f"Expected exactly 4 visible GPUs, got {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        if (props.major, props.minor) != (7, 0) or "V100" not in props.name:
            errors.append(f"GPU {i} must be a V100, got {props.name}")
        gib = props.total_memory / 1024**3
        if gib < min_memory_gib:
            errors.append(f"GPU {i} has {gib:.1f} GiB; this profile requires V100 32GB")
    return errors


def validate_chat_response(response: dict) -> None:
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Probe did not finish normally: {choice.get('finish_reason')}")
    message = choice["message"]
    if message.get("reasoning_content") or message.get("reasoning"):
        raise ValueError("Thinking was emitted despite enable_thinking=false")
    content = message.get("content")
    if not isinstance(content, str) or json.loads(content) != {"message": "こんにちは"}:
        raise ValueError("Japanese JSON probe returned unexpected content")


def probe_chat(api_base: str, model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content":
                      '次のJSONだけをそのまま返してください。説明は不要です。{"message":"こんにちは"}'}],
        "temperature": 0,
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(api_base.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as result:
        response = json.load(result)
    validate_chat_response(response)
    print("Japanese JSON probe: OK (thinking disabled)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime")
    chat = sub.add_parser("chat")
    chat.add_argument("--api-base", required=True)
    chat.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "runtime":
        import torch
        errors = runtime_errors(torch, version("vllm"))
        if errors:
            raise SystemExit("V100 preflight failed:\n" + "\n".join(errors))
        # Import the compiled extension too, rather than accepting metadata alone.
        import vllm._C  # noqa: F401
        for i in range(4):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"V100 runtime: OK (torch={torch.__version__}, CUDA=12.6, vLLM=0.21.0)")
    else:
        probe_chat(args.api_base, args.model)


if __name__ == "__main__":
    main()
