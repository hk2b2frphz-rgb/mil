"""Verify that moshi-finetune's loader picks up llm-jp's moshi_lm_kwargs.json.

Usage (from repo root):
    uv run --project ../moshi-finetune python scripts/check_llm_jp_config.py

Background:
    llm-jp/llm-jp-moshi-v1 ships its architecture config as
    `moshi_lm_kwargs.json` rather than the default `config.json` that
    moshi-finetune's loader looks for. If the loader does not pick it up,
    it falls back to the Kyutai English Moshi defaults (`loaders._lm_kwargs`)
    and the resulting shape mismatch makes the safetensors state_dict load
    crash natively (SIGSEGV / exitcode -11) during model initialization.

    This script reproduces the same loader path that train.py uses, but
    stops just after the config is read so we can confirm:
      - `raw_config is None`  -> loader silently ignored the override
      - `raw_config is dict`  -> config was correctly attached

Exit code:
    0 if `raw_config` is a dict with at least one key.
    1 otherwise (and prints a diagnostic message).
"""

from __future__ import annotations

import sys

from moshi.models import loaders


REPO_ID = "llm-jp/llm-jp-moshi-v1"
CONFIG_FILENAME = "moshi_lm_kwargs.json"


def main() -> int:
    print(f"[check] hf_repo_id : {REPO_ID}")
    print(f"[check] config_path: {CONFIG_FILENAME}")
    print()

    ci = loaders.CheckpointInfo.from_hf_repo(
        REPO_ID,
        config_path=CONFIG_FILENAME,
    )

    raw_config = getattr(ci, "raw_config", None)
    print(f"[check] raw_config is None? {raw_config is None}")
    if raw_config is not None:
        keys = sorted(raw_config.keys())
        print(f"[check] raw_config keys ({len(keys)}): {keys}")
    print()
    print(f"[check] moshi_weights: {ci.moshi_weights}")
    print(f"[check] mimi_weights : {ci.mimi_weights}")
    print(f"[check] tokenizer    : {getattr(ci, 'tokenizer', None) or getattr(ci, 'tokenizer_path', None)}")

    if raw_config is None:
        print()
        print(
            "[check] FAIL: loader did not pick up moshi_lm_kwargs.json.\n"
            "        The config_path argument may be wired differently in this\n"
            "        moshi version. Inspect the installed loaders.py and find\n"
            "        the correct way to point at moshi_lm_kwargs.json:\n"
            "          python -c \"import moshi.models.loaders as l, inspect; "
            "print(inspect.getsourcefile(l))\""
        )
        return 1

    if not raw_config:
        print()
        print("[check] FAIL: raw_config dict is empty.")
        return 1

    print()
    print("[check] OK: config attached. If model init still segfaults, the cause")
    print("       is elsewhere (state_dict key drift, sphn/mimi native init, etc).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
