"""Stepwise sanity check for llm-jp/llm-jp-moshi-v1 + moshi-finetune.

Usage (from repo root):
    uv run --project ../moshi-finetune python scripts/check_llm_jp_config.py

Mirrors the loading steps train.py performs but stops *between* each step so
the failing one is obvious. Each step prints either [OK] or [FAIL] with a
short message. Exits non-zero on the first failure so it can be chained
into other scripts.

Steps:
    1. hf_hub_download(moshi_lm_kwargs.json)
    2. CheckpointInfo.from_hf_repo with the local config path
    3. raw_config presence / keys
    4. model file presence (moshi weights, mimi weights, tokenizer)
    5. meta-device model construction via get_moshi(...)
    6. safetensors weight load
    7. dtype conversion (bf16)
    8. state_dict apply (model.load_state_dict)
    9. model.cuda() (only if CUDA is available)

Run as part of debugging exitcode -11 / model-init segfaults to localize
the failure without launching torchrun.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

# moshi-finetune のソース (finetune パッケージ) を import 可能にする。
# 環境変数で上書き可能。デフォルトはリポジトリの兄弟ディレクトリ。
_MOSHI_FT_REPO = Path(
    os.environ.get("MOSHI_FT_REPO")
    or (Path(__file__).resolve().parent.parent.parent / "moshi-finetune")
).resolve()
if (_MOSHI_FT_REPO / "finetune").is_dir():
    sys.path.insert(0, str(_MOSHI_FT_REPO))


REPO_ID = "llm-jp/llm-jp-moshi-v1"
CONFIG_FILENAME = "moshi_lm_kwargs.json"

PASS = "\033[1;32m[ OK ]\033[0m"
FAIL = "\033[1;31m[FAIL]\033[0m"
INFO = "\033[1;34m[INFO]\033[0m"


def _step(name: str, fn: Callable[[], Any]) -> Any:
    """Run fn(); on success print [OK] name; on failure print [FAIL] + traceback and exit 1."""
    print(f"{INFO} step: {name}")
    try:
        result = fn()
    except SystemExit:
        raise
    except BaseException as exc:  # also catches signals raised as exceptions
        print(f"{FAIL} {name}")
        print(f"       {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        sys.exit(1)
    print(f"{PASS} {name}")
    return result


def main() -> int:
    print(f"{INFO} hf_repo_id : {REPO_ID}")
    print(f"{INFO} config file: {CONFIG_FILENAME}")
    print()

    # ------------------------------------------------------------------
    # 1. download config json
    # ------------------------------------------------------------------
    from huggingface_hub import hf_hub_download

    local_config = _step(
        "1. download moshi_lm_kwargs.json from HF",
        lambda: hf_hub_download(REPO_ID, CONFIG_FILENAME),
    )
    print(f"       -> {local_config}")

    # ------------------------------------------------------------------
    # 2. CheckpointInfo.from_hf_repo(config_path=<local>)
    # ------------------------------------------------------------------
    from moshi.models import loaders

    ci = _step(
        "2. CheckpointInfo.from_hf_repo(config_path=<local>)",
        lambda: loaders.CheckpointInfo.from_hf_repo(REPO_ID, config_path=local_config),
    )

    # ------------------------------------------------------------------
    # 3. raw_config attached?
    # ------------------------------------------------------------------
    def _check_raw_config() -> dict:
        raw = getattr(ci, "raw_config", None)
        if raw is None:
            raise RuntimeError(
                "raw_config is None — loader silently ignored the override. "
                "Inspect moshi.models.loaders to find the correct kwarg name."
            )
        if not raw:
            raise RuntimeError("raw_config is an empty dict.")
        return raw

    raw_config = _step("3. raw_config is non-empty dict", _check_raw_config)
    print(f"       -> {len(raw_config)} keys: {sorted(raw_config.keys())}")

    # ------------------------------------------------------------------
    # 4. model files reachable on disk
    # ------------------------------------------------------------------
    def _check_files() -> dict:
        paths = {
            "moshi_weights": getattr(ci, "moshi_weights", None),
            "mimi_weights": getattr(ci, "mimi_weights", None),
            "tokenizer": getattr(ci, "tokenizer", None)
            or getattr(ci, "tokenizer_path", None),
        }
        for name, p in paths.items():
            if p is None:
                raise RuntimeError(f"{name} is None on CheckpointInfo")
            if not os.path.exists(str(p)):
                raise RuntimeError(f"{name} path does not exist on disk: {p}")
            size = os.path.getsize(str(p))
            print(f"       -> {name}: {p} ({size/1e6:.1f} MB)")
        return paths

    _step("4. model files exist locally", _check_files)

    # ------------------------------------------------------------------
    # 5. meta-device model construction
    # ------------------------------------------------------------------
    import torch

    def _build_meta_model():
        with torch.device("meta"):
            model = ci.get_moshi(
                device="meta",
                dtype=torch.bfloat16,
                lm_kwargs_overrides={
                    "gradient_checkpointing": True,
                    "lora": True,
                    "lora_rank": 32,
                    "lora_scaling": 2.0,
                },
                load_weight=False,
            )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"       -> total params: {n_params/1e9:.2f} B")
        return model

    model = _step("5. get_moshi(device='meta', load_weight=False)", _build_meta_model)

    # ------------------------------------------------------------------
    # 6. safetensors weight load
    # ------------------------------------------------------------------
    import safetensors.torch as st

    def _load_weights():
        moshi_path = getattr(ci, "moshi_weights")
        sd = st.load_file(moshi_path)
        print(f"       -> {len(sd)} tensors loaded from safetensors")
        return sd

    state_dict = _step("6. safetensors.torch.load_file(moshi_weights)", _load_weights)

    # ------------------------------------------------------------------
    # 7. dtype conversion
    # ------------------------------------------------------------------
    def _to_bf16():
        for k in state_dict:
            state_dict[k] = state_dict[k].to(torch.bfloat16)
        return None

    _step("7. convert state_dict tensors to bf16", _to_bf16)

    # ------------------------------------------------------------------
    # 8. load_state_dict (shape mismatch があると C++ 側で死ぬ)
    #    LoRA 層は checkpoint に無いため meta に残るのは正常。次の 8b で初期化する。
    # ------------------------------------------------------------------
    def _apply_state_dict():
        missing_unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        try:
            missing = list(missing_unexpected.missing_keys)
            unexpected = list(missing_unexpected.unexpected_keys)
        except AttributeError:
            missing, unexpected = [], []
        print(f"       -> missing keys   : {len(missing)} (show up to 5: {missing[:5]})")
        print(f"       -> unexpected keys: {len(unexpected)} (show up to 5: {unexpected[:5]})")
        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        non_lora_meta = [n for n in meta_params if "lora" not in n.lower()]
        print(f"       -> meta params total: {len(meta_params)} (lora-only: {len(meta_params) - len(non_lora_meta)})")
        if non_lora_meta:
            raise RuntimeError(
                f"{len(non_lora_meta)} non-LoRA parameters are still on meta after "
                f"load_state_dict; first few: {non_lora_meta[:5]}\n"
                f"これは llm-jp の重みと moshi-finetune の LMModel のキー名がずれている兆候。"
            )
        return None

    _step("8. model.load_state_dict(state_dict, strict=False, assign=True)", _apply_state_dict)

    # ------------------------------------------------------------------
    # 8b. initialize LoRA layers (train.py / wrapped_model.py:147-150 と同じ処理)
    # ------------------------------------------------------------------
    from finetune.wrapped_model import initialize_lora_parameters

    def _init_lora():
        initialize_lora_parameters(model, torch.bfloat16)
        remaining = [n for n, p in model.named_parameters() if p.is_meta]
        if remaining:
            raise RuntimeError(
                f"After initialize_lora_parameters, {len(remaining)} params still on meta: "
                f"{remaining[:5]}"
            )
        return None

    _step("8b. initialize_lora_parameters(model, bfloat16)", _init_lora)

    # ------------------------------------------------------------------
    # 9. move to GPU (only if CUDA is visible)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        def _to_cuda():
            model.cuda()
            return None

        _step("9. model.cuda()", _to_cuda)
    else:
        print(f"{INFO} step: 9. model.cuda() — skipped (no CUDA visible)")

    print()
    print(f"{PASS} all checks passed. "
          f"If train.py still crashes, the cause is past model init "
          f"(data loader, mimi codec, optimizer init, NCCL collective, etc).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
