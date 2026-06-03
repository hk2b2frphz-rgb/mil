#!/usr/bin/env python3
"""Merge a LoRA checkpoint into the base model and save as a single safetensors file.

Usage:
    uv run --project ../moshi-finetune python scripts/merge_lora.py \
        --lora-ckpt experiments/exp001_lora_baseline/checkpoints/<ts>/checkpoint_000500/consolidated/lora.safetensors \
        --out merged_model/consolidated.safetensors

    # Then run with response_recorder:
    uv run python response_recorder.py \
        --moshi-weight merged_model/consolidated.safetensors \
        --inputs prompts/hello.wav --out-dir results/lora_merged/
"""
import argparse
import sys
from pathlib import Path

import safetensors.torch
import torch


def find_base_weights(hf_repo: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(hf_repo, "model.safetensors"))


def merge(base_state: dict[str, torch.Tensor],
          lora_state: dict[str, torch.Tensor],
          scaling: float) -> dict[str, torch.Tensor]:
    merged = dict(base_state)

    lora_a_keys = sorted(k for k in lora_state if "lora_A" in k)
    for a_key in lora_a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in lora_state:
            print(f"WARNING: {b_key} not found, skipping {a_key}", file=sys.stderr)
            continue

        # lora_A/lora_B key -> base weight key
        # e.g. "transformer.layers.0.self_attn.q_proj.lora_A" -> "transformer.layers.0.self_attn.q_proj.weight"
        weight_key = a_key.rsplit(".lora_A", 1)[0] + ".weight"
        if weight_key not in merged:
            print(f"WARNING: base key {weight_key} not found, skipping", file=sys.stderr)
            continue

        lora_a = lora_state[a_key].float()
        lora_b = lora_state[b_key].float()
        # LoRA: W' = W + B @ A * scaling
        delta = (lora_b @ lora_a) * scaling
        merged[weight_key] = (merged[weight_key].float() + delta).to(base_state[weight_key].dtype)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base Moshi model")
    parser.add_argument("--lora-ckpt", required=True, help="Path to lora.safetensors")
    parser.add_argument("--out", required=True, help="Output path for merged safetensors")
    parser.add_argument("--hf-repo", default="llm-jp/llm-jp-moshi-v1",
                        help="HF repo for base model weights")
    parser.add_argument("--scaling", type=float, default=None,
                        help="LoRA scaling (alpha/rank). If not set, read from checkpoint config.json")
    args = parser.parse_args()

    lora_path = Path(args.lora_ckpt)
    if not lora_path.exists():
        print(f"ERROR: LoRA checkpoint not found: {lora_path}", file=sys.stderr)
        sys.exit(1)

    # Try to read scaling from config.json next to lora.safetensors
    scaling = args.scaling
    if scaling is None:
        config_path = lora_path.parent / "config.json"
        if config_path.exists():
            import json
            config = json.loads(config_path.read_text())
            lora_cfg = config.get("lora", {})
            rank = lora_cfg.get("rank", 32)
            lora_scaling = lora_cfg.get("scaling", 2.0)
            scaling = lora_scaling
            print(f"[merge] config.json: rank={rank}, scaling={scaling}")
        else:
            scaling = 2.0
            print(f"[merge] config.json not found, using default scaling={scaling}")

    print(f"[merge] Loading base model from {args.hf_repo}...")
    base_path = find_base_weights(args.hf_repo)
    base_state = safetensors.torch.load_file(str(base_path))
    print(f"[merge]   base keys: {len(base_state)}")

    print(f"[merge] Loading LoRA from {lora_path}...")
    lora_state = safetensors.torch.load_file(str(lora_path))
    print(f"[merge]   lora keys: {len(lora_state)}")

    print(f"[merge] Merging with scaling={scaling}...")
    merged = merge(base_state, lora_state, scaling)
    print(f"[merge]   merged keys: {len(merged)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[merge] Saving to {out_path}...")
    safetensors.torch.save_file(merged, str(out_path))
    print(f"[merge] Done. Use with: uv run python response_recorder.py --moshi-weight {out_path}")


if __name__ == "__main__":
    main()
