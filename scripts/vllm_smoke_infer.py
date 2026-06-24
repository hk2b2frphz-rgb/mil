#!/usr/bin/env python3
import os
import sys

from vllm import LLM, SamplingParams


def main() -> int:
    model = os.environ.get("SMOKE_MODEL")
    if not model:
        print("ERROR: SMOKE_MODEL is required.", file=sys.stderr)
        return 2

    kwargs = {
        "model": model,
        "tensor_parallel_size": int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "8192")),
        "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
        "enforce_eager": True,
    }
    max_num_batched_tokens = os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "")
    if max_num_batched_tokens:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)

    llm = LLM(**kwargs)
    outs = llm.chat(
        [{"role": "user", "content": "Hello, please introduce yourself briefly."}],
        SamplingParams(max_tokens=64, temperature=0.0),
    )
    text = outs[0].outputs[0].text
    print(text)
    if not text.strip():
        print(f"ERROR: empty completion for {model}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
