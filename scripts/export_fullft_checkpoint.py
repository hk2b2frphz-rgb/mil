#!/usr/bin/env python3
"""Export a nu-dialogue full-FT checkpoint into an inference-ready weight.

The nu-dialogue runner saves checkpoints with ``accelerator.save_state()``,
i.e. a DeepSpeed ZeRO shard under ``step_<N>/pytorch_model/zero_pp_rank_*.pt``;
there is no single ``consolidated`` file. This wraps the two nu-repo tools into
one command:

    step_<N>/  (ZeRO shard)
       | (1) tools/zero_to_fp32.py  -> fp32 single file (MoshiForFinetuning fmt)
       v
    <ft_dir>/model.safetensors + moshi_lm_kwargs.json
       | (2) tools/clean_moshi.py   -> original LMModel format
       v
    <out_dir>/model.safetensors   (response_recorder can load this)

``--remove_modules_for_user_stream`` is decided automatically from the
checkpoint's ``moshi_lm_kwargs.json`` (``dep_q == 16 and depformer_context == 16``
means the model was trained with the user stream and must be reduced back to
``dep_q = 8``). Override with --remove-user-stream / --no-remove-user-stream.

Usage:
    uv run python scripts/export_fullft_checkpoint.py \
        --step-dir experiments/_fullft_sweeps/<RUN>_f01/checkpoints/nu_<ts>/step_120 \
        --out-dir  experiments/_fullft_sweeps/<RUN>_f01/exported/step_120_clean

    # Then run inference:
    uv run python response_recorder.py \
        --moshi-weight experiments/_fullft_sweeps/<RUN>_f01/exported/step_120_clean/model.safetensors \
        --inputs prompts/hello.wav --seeds 0,1,2 --out-dir results/f01_step120/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve_nu_repo(arg: str | None) -> Path:
    candidate = arg or os.environ.get("NU_MOSHI_FT_REPO") or str(
        REPO_ROOT.parent / "moshi-finetune-nu-dialogue"
    )
    repo = Path(candidate).expanduser().resolve()
    if not (repo / "tools" / "zero_to_fp32.py").exists():
        fail(f"nu-dialogue repo not found (no tools/zero_to_fp32.py under {repo}). "
             "Pass --nu-repo or set NU_MOSHI_FT_REPO.")
    return repo


def resolve_kwargs(arg: str | None, nu_repo: Path) -> Path:
    candidates: list[Path] = []
    if arg:
        candidates.append(Path(arg).expanduser())
    env_model_dir = os.environ.get("NU_MODEL_DIR")
    if env_model_dir:
        candidates.append(Path(env_model_dir).expanduser() / "moshi_lm_kwargs.json")
    candidates.append(
        nu_repo / "init_models" / "llm-jp-moshi-v1-both_streams-float32" / "moshi_lm_kwargs.json"
    )
    for c in candidates:
        if c.exists():
            return c.resolve()
    fail("Could not locate moshi_lm_kwargs.json. Pass --moshi-lm-kwargs, set "
         "NU_MODEL_DIR, or ensure the init model exists under the nu repo. "
         f"Tried: {', '.join(str(c) for c in candidates)}")


def decide_remove_user_stream(kwargs_path: Path, override: bool | None) -> bool:
    data = json.loads(kwargs_path.read_text(encoding="utf-8"))
    dep_q = data.get("dep_q")
    dep_ctx = data.get("depformer_context")
    auto = dep_q == 16 and dep_ctx == 16
    log(f"moshi_lm_kwargs: dep_q={dep_q} depformer_context={dep_ctx} "
        f"-> user-stream model: {auto}")
    if override is None:
        return auto
    if override and not auto:
        log("WARNING: --remove-user-stream forced but dep_q/depformer_context "
            "are not both 16; clean_moshi will likely assert.")
    return override


def run(cmd: list[str], cwd: Path) -> None:
    log("running: " + " ".join(cmd))
    log(f"  (cwd={cwd})")
    # The nu tools import the repo-root `models` package, but running
    # `python tools/<x>.py` puts tools/ on sys.path[0] instead of the repo
    # root (there is no tools/__init__.py). Put the repo root on PYTHONPATH so
    # `from models import ...` resolves.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + existing if existing else "")
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        fail(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a nu-dialogue full-FT ZeRO checkpoint into an "
                    "inference-ready Moshi weight.")
    parser.add_argument("--step-dir", required=True,
                        help="Path to the step_<N> checkpoint dir (parent of pytorch_model/).")
    parser.add_argument("--out-dir", required=True,
                        help="Output dir for the cleaned model.safetensors + kwargs.")
    parser.add_argument("--nu-repo", default=None,
                        help="nu-dialogue/moshi-finetune checkout "
                             "(default: $NU_MOSHI_FT_REPO or ../moshi-finetune-nu-dialogue).")
    parser.add_argument("--moshi-lm-kwargs", default=None,
                        help="Path to moshi_lm_kwargs.json "
                             "(default: $NU_MODEL_DIR/moshi_lm_kwargs.json or nu repo init model).")
    parser.add_argument("--tag", default="pytorch_model",
                        help="DeepSpeed checkpoint tag (accelerate default: pytorch_model).")
    parser.add_argument("--model-dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Output dtype for clean_moshi.")
    parser.add_argument("--intermediate-dir", default=None,
                        help="Where to write the stage-1 MoshiForFinetuning weight "
                             "(default: <out-dir>_ft). Must not already exist unless --overwrite.")
    remove_grp = parser.add_mutually_exclusive_group()
    remove_grp.add_argument("--remove-user-stream", dest="remove_user_stream",
                            action="store_true", default=None,
                            help="Force --remove_modules_for_user_stream in clean_moshi.")
    remove_grp.add_argument("--no-remove-user-stream", dest="remove_user_stream",
                            action="store_false",
                            help="Force-disable user-stream module removal.")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Keep the stage-1 intermediate dir instead of deleting it.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove an existing intermediate dir before stage 1.")
    args = parser.parse_args()

    step_dir = Path(args.step_dir).expanduser().resolve()
    if not step_dir.exists():
        fail(f"step dir not found: {step_dir}")
    tag_dir = step_dir / args.tag
    if not tag_dir.is_dir():
        fail(f"tag dir not found: {tag_dir}. Pass --tag if the checkpoint uses a "
             "different tag (expected the dir holding zero_pp_rank_*.pt).")
    if not any(tag_dir.glob("*zero_pp_rank*")) and not any(tag_dir.glob("*model_states*")):
        log(f"WARNING: no zero_pp_rank_*/model_states files found under {tag_dir}; "
            "proceeding anyway.")

    nu_repo = resolve_nu_repo(args.nu_repo)
    kwargs_path = resolve_kwargs(args.moshi_lm_kwargs, nu_repo)
    log(f"nu repo:        {nu_repo}")
    log(f"step dir:       {step_dir}")
    log(f"moshi kwargs:   {kwargs_path}")

    remove_user_stream = decide_remove_user_stream(kwargs_path, args.remove_user_stream)

    out_dir = Path(args.out_dir).expanduser().resolve()
    ft_dir = (Path(args.intermediate_dir).expanduser().resolve()
              if args.intermediate_dir else Path(str(out_dir) + "_ft"))

    # zero_to_fp32 does os.makedirs(output_dir) without exist_ok -> must not exist.
    if ft_dir.exists():
        if args.overwrite:
            log(f"--overwrite: removing existing intermediate dir {ft_dir}")
            shutil.rmtree(ft_dir)
        else:
            fail(f"intermediate dir already exists: {ft_dir} "
                 "(zero_to_fp32 requires a fresh dir; pass --overwrite or --intermediate-dir).")

    # ---- (1) ZeRO shard -> fp32 single file (MoshiForFinetuning format) -----
    log("stage 1/2: zero_to_fp32 (consolidate ZeRO shard)")
    run([
        "uv", "run", "python", "tools/zero_to_fp32.py",
        str(step_dir), str(ft_dir),
        "--tag", args.tag,
        "--safe_serialization",
        "--moshi_lm_kwargs_path", str(kwargs_path),
    ], cwd=nu_repo)

    # ---- (2) MoshiForFinetuning -> original LMModel format ------------------
    log("stage 2/2: clean_moshi (convert to original Moshi format)")
    clean_cmd = [
        "uv", "run", "python", "tools/clean_moshi.py",
        "--moshi_ft_dir", str(ft_dir),
        "--save_dir", str(out_dir),
        "--model_dtype", args.model_dtype,
    ]
    if remove_user_stream:
        clean_cmd.append("--remove_modules_for_user_stream")
    run(clean_cmd, cwd=nu_repo)

    if not args.keep_intermediate:
        log(f"removing intermediate dir {ft_dir}")
        shutil.rmtree(ft_dir, ignore_errors=True)

    final_weight = out_dir / "model.safetensors"
    if not final_weight.exists():
        fail(f"expected output weight not found: {final_weight}")

    log(f"done. exported weight: {final_weight}")
    log("run inference with:")
    log(f"  uv run python response_recorder.py \\")
    log(f"    --moshi-weight {final_weight} \\")
    log(f"    --inputs prompts/hello.wav --seeds 0,1,2 --out-dir results/")


if __name__ == "__main__":
    main()
