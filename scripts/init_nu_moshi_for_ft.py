#!/usr/bin/env python3
"""Initialize a nu-dialogue MoshiForFinetuning model with optional HF config."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu-repo", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--moshi-lm-repo", default="llm-jp/llm-jp-moshi-v1")
    parser.add_argument("--moshi-lm-name", default="model.safetensors")
    parser.add_argument(
        "--moshi-lm-config-name",
        default="moshi_lm_kwargs.json",
        help="HF config file to pass through CheckpointInfo. Use an empty value to skip.",
    )
    parser.add_argument("--model-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--init-text-embeddings", action="store_true")
    parser.add_argument("--retain-text-token-ids", nargs="+", type=int, default=[0, 3, 32000])
    parser.add_argument("--extend-modules-for-user-stream", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nu_repo = args.nu_repo.resolve()
    if not (nu_repo / "models").is_dir():
        print(f"ERROR: nu-dialogue repo not found or incomplete: {nu_repo}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(nu_repo))

    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    from models import MoshiForFinetuning, extend_moshi_modules_for_user_stream

    def init_embedding_module(emb: nn.Embedding, retain_token_ids: list[int]) -> nn.Embedding:
        dtype = emb.weight.dtype
        emb_weights = emb.weight.data
        mean = emb_weights.mean(dim=0)
        vocab_size = emb_weights.size()[0]
        sigma = ((emb_weights - mean).T @ (emb_weights - mean)) / vocab_size
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mean.to(torch.float32),
            covariance_matrix=1e-5 * sigma.to(torch.float32),
        )
        new_emb_weights = torch.stack(
            tuple(dist.sample() for _ in range(vocab_size)),
            dim=0,
        ).to(dtype)
        for token_id in retain_token_ids:
            if token_id >= vocab_size:
                raise ValueError(f"Token id {token_id} is out of range of vocab size {vocab_size}")
            new_emb_weights[token_id] = emb_weights[token_id]
        emb.weight.data = new_emb_weights
        return emb

    save_dir = args.save_dir.resolve()
    if save_dir.exists() and args.overwrite:
        shutil.rmtree(save_dir)
    if (save_dir / "model.safetensors").exists() and (save_dir / "moshi_lm_kwargs.json").exists():
        print(f"[nu-init] model already initialized: {save_dir}")
        return 0

    config_name = args.moshi_lm_config_name.strip()
    if config_name:
        print(f"[nu-init] loading {args.moshi_lm_repo}/{args.moshi_lm_name} with config {config_name}")
        config_path = hf_hub_download(args.moshi_lm_repo, config_name)
        checkpoint = loaders.CheckpointInfo.from_hf_repo(
            args.moshi_lm_repo,
            moshi_weights=args.moshi_lm_name,
            config_path=config_path,
        )
        moshi_lm_kwargs = deepcopy(checkpoint.lm_config or loaders._lm_kwargs)
        moshi_lm = checkpoint.get_moshi(device="cpu", dtype=getattr(torch, args.model_dtype))
    else:
        print(f"[nu-init] loading {args.moshi_lm_repo}/{args.moshi_lm_name} with default Moshi config")
        moshi_lm_kwargs = deepcopy(loaders._lm_kwargs)
        moshi_lm = loaders.get_moshi_lm(
            hf_hub_download(args.moshi_lm_repo, args.moshi_lm_name),
            device="cpu",
            dtype=getattr(torch, args.model_dtype),
        )

    if args.init_text_embeddings:
        print("[nu-init] initializing text embeddings")
        init_embedding_module(moshi_lm.text_emb, args.retain_text_token_ids)
        init_embedding_module(moshi_lm.depformer_text_emb, args.retain_text_token_ids)

    if args.extend_modules_for_user_stream:
        print("[nu-init] extending modules for user stream")
        moshi_lm_kwargs.update({"dep_q": 16, "depformer_context": 16})
        moshi_lm = extend_moshi_modules_for_user_stream(moshi_lm)

    print("[nu-init] converting to MoshiForFinetuning")
    moshi_lm = MoshiForFinetuning.from_original_moshi_lm(
        moshi_lm=moshi_lm,
        moshi_lm_kwargs=moshi_lm_kwargs,
    )
    moshi_lm = moshi_lm.to(getattr(torch, args.model_dtype))

    print(f"[nu-init] saving initialized model to {save_dir}")
    moshi_lm.save_pretrained(save_dir)
    metadata = {
        "moshi_lm_repo": args.moshi_lm_repo,
        "moshi_lm_name": args.moshi_lm_name,
        "moshi_lm_config_name": config_name or None,
        "model_dtype": args.model_dtype,
        "extend_modules_for_user_stream": args.extend_modules_for_user_stream,
        "init_text_embeddings": args.init_text_embeddings,
    }
    (save_dir / "init_source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
