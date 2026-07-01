#!/usr/bin/env python3
"""Apply small runtime compatibility patches to nu-dialogue/moshi-finetune."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nu-repo", required=True, type=Path)
    return parser.parse_args()


def patch_tokenize_text(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_ROBUST_TEXT_ALIGNMENT" in src:
        return False

    pattern = re.compile(
        r"    # make token-level transcript by aligning the timestamps\n"
        r"    token_transcript = \[\]\n"
        r"    for i, token in enumerate\(tokens\):\n"
        r".*?"
        r"    assert not char_transcript, f\"Remaining characters: \{char_transcript\}\"\n",
        flags=re.S,
    )
    replacement = '''    # make token-level transcript by aligning the timestamps
    # AUTO_PATCH_ROBUST_TEXT_ALIGNMENT:
    # Synthetic data in miltoka has utterance-level Japanese timestamps. Some
    # SentencePiece outputs include word-boundary markers or decoded byte pieces
    # whose string length can exceed the remaining character transcript. Clamp
    # alignment instead of crashing at chars[0].
    token_transcript = []
    for token in tokens:
        token_text = token[1:] if token.startswith("▁") else token
        if not token_text:
            continue
        if not char_transcript:
            warnings.warn(f"Dropping token {token!r}: no characters remain for {text!r}")
            continue
        num_token_chars = max(1, len(token_text))
        if num_token_chars > len(char_transcript):
            warnings.warn(
                f"Clamping token/character alignment for token {token!r}; "
                f"remaining_chars={len(char_transcript)} text={text!r}"
            )
            chars = char_transcript
        else:
            chars = char_transcript[:num_token_chars]
        if not chars:
            continue
        token_transcript.append(
            {
                "speaker": chars[0]["speaker"],
                "start": chars[0]["start"],
                "end": chars[-1]["end"],
                "token": token,
            }
        )
        char_transcript = char_transcript[len(chars) :]
    if char_transcript:
        warnings.warn(f"Remaining characters after token alignment are ignored: {char_transcript}")
'''
    updated, count = pattern.subn(replacement, src, count=1)
    if count != 1:
        raise RuntimeError(f"Could not locate token alignment block in {path}")
    path.write_text(updated, encoding="utf-8")
    return True


def patch_prepare_dataset(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    updated = src.replace("np.concat(", "np.concatenate(")
    if updated == src:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def patch_finetune_tracking(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_WITH_TRACKING_DEFAULT" in src:
        return False

    old = """    if args.report_to is not None:
        args.with_tracking = True
"""
    new = """    # AUTO_PATCH_WITH_TRACKING_DEFAULT:
    # nu-dialogue/moshi-finetune references args.with_tracking even when
    # --report_to is omitted. Default it explicitly for non-W&B runs.
    args.with_tracking = False
    if args.report_to is not None:
        args.with_tracking = True
"""
    if old not in src:
        raise RuntimeError(f"Could not locate report_to tracking block in {path}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    return True


def patch_finetune_safe_means(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_SAFE_EMPTY_MEAN" in src:
        return False

    marker = 'logger = get_logger(__name__)\n'
    helper = '''logger = get_logger(__name__)


# AUTO_PATCH_SAFE_EMPTY_MEAN:
# Eval batches can contain streams with no valid text/user labels after
# utterance-level timestamp conversion and length filtering. torch.mean([]) is
# NaN, so use zero-valued tensors for logging-only empty means.
def _safe_mean(values):
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()
'''
    if marker not in src:
        raise RuntimeError(f"Could not locate logger declaration in {path}")
    src = src.replace(marker, helper, 1)

    replacements = {
        'text_accuracy[non_pad_indices].mean()': '_safe_mean(text_accuracy[non_pad_indices])',
        'text_accuracy[pad_indices].mean()': '_safe_mean(text_accuracy[pad_indices])',
        'audio_accuracy[..., 0][\n            audio_labels[..., 0] != moshi_lm.zero_token_id\n        ].mean()': '_safe_mean(audio_accuracy[..., 0][\n            audio_labels[..., 0] != moshi_lm.zero_token_id\n        ])',
        'audio_accuracy[..., 1:8][\n            audio_labels[..., 1:8] != moshi_lm.zero_token_id\n        ].mean()': '_safe_mean(audio_accuracy[..., 1:8][\n            audio_labels[..., 1:8] != moshi_lm.zero_token_id\n        ])',
        'audio_accuracy[..., 8][\n                    audio_labels[..., 8] != moshi_lm.zero_token_id\n                ].mean()': '_safe_mean(audio_accuracy[..., 8][\n                    audio_labels[..., 8] != moshi_lm.zero_token_id\n                ])',
        'audio_accuracy[..., 9:][\n                    audio_labels[..., 9:] != moshi_lm.zero_token_id\n                ].mean()': '_safe_mean(audio_accuracy[..., 9:][\n                    audio_labels[..., 9:] != moshi_lm.zero_token_id\n                ])',
        '    text_loss = 0.0': '    text_loss = tempformer_out.new_tensor(0.0)',
        'temp_result["non_pad_losses"].mean().detach()': '_safe_mean(temp_result["non_pad_losses"]).detach()',
        'temp_result["pad_losses"].mean().detach()': '_safe_mean(temp_result["pad_losses"]).detach()',
        'dep_result["semantic_losses"].mean().detach()': '_safe_mean(dep_result["semantic_losses"]).detach()',
        'dep_result["acoustic_losses"].mean().detach()': '_safe_mean(dep_result["acoustic_losses"]).detach()',
        'dep_result["semantic_losses_user"].mean().detach()': '_safe_mean(dep_result["semantic_losses_user"]).detach()',
        'dep_result["acoustic_losses_user"].mean().detach()': '_safe_mean(dep_result["acoustic_losses_user"]).detach()',
    }
    missing = []
    for old, new in replacements.items():
        if old not in src:
            missing.append(old)
        else:
            src = src.replace(old, new)
    if missing:
        raise RuntimeError(f"Could not locate safe-mean targets in {path}: {missing[:2]}")

    path.write_text(src, encoding="utf-8")
    return True


def patch_finetune_mlflow_stdout_metrics(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_MILTO_MLFLOW_STDOUT_METRICS_V2" in src:
        return False

    stdout_helper = '''# AUTO_PATCH_MILTO_MLFLOW_STDOUT_METRICS_V2:
# Emit machine-readable train/eval metrics to stdout so the miltoka launcher can
# sync them to MLflow even when nu-dialogue tracking integrations are disabled.
def _metric_buffer_to_scalars(logging_buffer, accelerator, strip_prefix=""):
    scalars = {}
    for key, values in logging_buffer.items():
        if not values:
            continue
        metric_key = key[len(strip_prefix):] if strip_prefix and key.startswith(strip_prefix) else key
        local_values = torch.stack(
            [value.detach().float().reshape(()) for value in values]
        ).to(accelerator.device)
        finite_values = local_values[torch.isfinite(local_values)]
        if finite_values.numel() == 0:
            local_sum_count = torch.zeros(2, device=accelerator.device)
        else:
            local_sum_count = torch.stack(
                [
                    finite_values.sum(),
                    finite_values.new_tensor(float(finite_values.numel())),
                ]
            )
        gathered_sum_count = accelerator.gather(local_sum_count).reshape(-1, 2)
        total_sum = gathered_sum_count[:, 0].sum()
        total_count = gathered_sum_count[:, 1].sum()
        if total_count.item() == 0:
            scalars[metric_key] = 0.0
        else:
            scalars[metric_key] = float((total_sum / total_count).item())
    return scalars


def _log_miltoka_metrics(split, step, epoch, metrics):
    payload = {
        "split": split,
        "step": int(step),
        "epoch": int(epoch),
        "metrics": {
            key: float(value)
            for key, value in sorted(metrics.items())
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        },
    }
    logger.info("MILTO_METRICS " + json.dumps(payload, sort_keys=True))
'''

    if "AUTO_PATCH_MILTO_MLFLOW_STDOUT_METRICS:" in src:
        pattern = re.compile(
            r"# AUTO_PATCH_MILTO_MLFLOW_STDOUT_METRICS:.*?\n\n# Parsing input arguments",
            flags=re.S,
        )
        updated, count = pattern.subn(stdout_helper + "\n\n# Parsing input arguments", src, count=1)
        if count != 1:
            raise RuntimeError(f"Could not upgrade stdout metrics helper in {path}")
        path.write_text(updated, encoding="utf-8")
        return True

    safe_mean_tail = '''def _safe_mean(values):
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()
'''
    metrics_helper = '''def _safe_mean(values):
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()


''' + stdout_helper
    if safe_mean_tail not in src:
        raise RuntimeError(f"Could not locate _safe_mean helper in {path}")
    src = src.replace(safe_mean_tail, metrics_helper, 1)

    old_train = '''                    if args.with_tracking:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in logging_buffer.items()
                            }
                        )
                        accelerator.log(
                            {
                                **{
                                    key: values.nanmean()
                                    for key, values in gathered_metrics.items()
                                },
                                "learning_rate/tempformer": optimizer.param_groups[0]["lr"],
                                "learning_rate/depformer": optimizer.param_groups[1]["lr"],
                            },
                            step=current_steps,
                        )
                    logging_buffer = collections.defaultdict(list)  # reset
'''
    new_train = '''                    metrics_for_log = _metric_buffer_to_scalars(
                        logging_buffer, accelerator, strip_prefix="training_"
                    )
                    metrics_for_log["learning_rate/tempformer"] = optimizer.param_groups[0]["lr"]
                    metrics_for_log["learning_rate/depformer"] = optimizer.param_groups[1]["lr"]
                    _log_miltoka_metrics("train", current_steps, epoch, metrics_for_log)
                    if args.with_tracking:
                        accelerator.log(
                            {
                                **{
                                    f"training_{key}": value
                                    for key, value in metrics_for_log.items()
                                    if not key.startswith("learning_rate/")
                                },
                                "learning_rate/tempformer": optimizer.param_groups[0]["lr"],
                                "learning_rate/depformer": optimizer.param_groups[1]["lr"],
                            },
                            step=current_steps,
                        )
                    logging_buffer = collections.defaultdict(list)  # reset
'''
    if old_train not in src:
        raise RuntimeError(f"Could not locate training metrics block in {path}")
    src = src.replace(old_train, new_train, 1)

    old_eval = '''                    if args.with_tracking:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in eval_logging_buffer.items()
                            }
                        )
                        accelerator.log(
                            {key: values.nanmean() for key, values in gathered_metrics.items()},
                            step=current_steps,
                        )
'''
    new_eval = '''                    eval_metrics_for_log = _metric_buffer_to_scalars(
                        eval_logging_buffer, accelerator, strip_prefix="evaluation_"
                    )
                    logger.info(
                        f"Eval Steps: {current_steps}, "
                        f"Loss: {eval_metrics_for_log.get('loss/total', 0.0):.5f} "
                        f"(text: {eval_metrics_for_log.get('loss/text_total', 0.0):.5f}, "
                        f"audio: {eval_metrics_for_log.get('loss/audio_total', 0.0):.5f})"
                    )
                    _log_miltoka_metrics("eval", current_steps, epoch, eval_metrics_for_log)
                    if args.with_tracking:
                        accelerator.log(
                            {f"evaluation_{key}": value for key, value in eval_metrics_for_log.items()},
                            step=current_steps,
                        )
'''
    if old_eval not in src:
        raise RuntimeError(f"Could not locate evaluation metrics block in {path}")
    src = src.replace(old_eval, new_eval, 1)

    path.write_text(src, encoding="utf-8")
    return True


def patch_finetune_keep_best_only(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "AUTO_PATCH_KEEP_BEST_ONLY" in src:
        return False

    if "import shutil\n" not in src:
        old_import = "import os\nfrom datetime import timedelta\n"
        new_import = "import os\nimport shutil\nfrom datetime import timedelta\n"
        if old_import not in src:
            raise RuntimeError(f"Could not locate import block in {path}")
        src = src.replace(old_import, new_import, 1)

    old_arg = '''    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every X updates steps.",
    )
'''
    new_arg = '''    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--keep_best_only",
        action="store_true",
        help="Delete every saved checkpoint except the one at the step with "
        "the lowest recorded eval loss (loss/total) and the most recently "
        "saved one (kept until it has been evaluated).",
    )
'''
    if old_arg not in src:
        raise RuntimeError(f"Could not locate --save_steps argparse block in {path}")
    src = src.replace(old_arg, new_arg, 1)

    prune_helper = '''# AUTO_PATCH_KEEP_BEST_ONLY: prune saved checkpoints down to the single best
# (by eval loss) plus the one just written, instead of keeping every step_<N>
# ZeRO shard. Full-FT checkpoints duplicate the whole model per rank, so
# unpruned runs can otherwise fill the disk quickly.
def _prune_checkpoints_keep_best(output_dir, current_steps, eval_loss_by_step, accelerator):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    if not os.path.isdir(output_dir):
        return

    step_dirs = {}
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if not os.path.isdir(full) or not name.startswith("step_"):
            continue
        try:
            step = int(name.split("_")[-1])
        except ValueError:
            continue
        step_dirs[step] = full

    keep_steps = {current_steps}
    scored = {
        step: loss
        for step, loss in eval_loss_by_step.items()
        if loss is not None and step in step_dirs
    }
    if scored:
        keep_steps.add(min(scored, key=scored.get))

    for step, step_path in step_dirs.items():
        if step in keep_steps:
            continue
        shutil.rmtree(step_path, ignore_errors=True)
        logger.info(f"[keep_best_only] deleted checkpoint step_{step}")
'''
    marker = "def _log_miltoka_metrics(split, step, epoch, metrics):"
    if marker not in src:
        raise RuntimeError(f"Could not locate _log_miltoka_metrics definition in {path}")
    src = src.replace(marker, prune_helper + "\n\n" + marker, 1)

    old_resume = '''    current_steps = 0
    starting_epoch = 0
'''
    new_resume = '''    current_steps = 0
    starting_epoch = 0
    # AUTO_PATCH_KEEP_BEST_ONLY: step -> eval loss (loss/total), used by
    # _prune_checkpoints_keep_best to decide which checkpoint to keep.
    eval_loss_by_step: dict = {}
'''
    if old_resume not in src:
        raise RuntimeError(f"Could not locate current_steps/starting_epoch block in {path}")
    src = src.replace(old_resume, new_resume, 1)

    old_eval_log = '''                    _log_miltoka_metrics("eval", current_steps, epoch, eval_metrics_for_log)
'''
    new_eval_log = '''                    _log_miltoka_metrics("eval", current_steps, epoch, eval_metrics_for_log)
                    eval_loss_by_step[current_steps] = eval_metrics_for_log.get("loss/total")
'''
    if old_eval_log not in src:
        raise RuntimeError(f"Could not locate eval metrics logging call in {path}")
    src = src.replace(old_eval_log, new_eval_log, 1)

    old_periodic_save = '''                # Save checkpoint
                if args.save_steps is not None and current_steps % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
                    accelerator.save_state(output_dir)
'''
    new_periodic_save = '''                # Save checkpoint
                if args.save_steps is not None and current_steps % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
                    accelerator.save_state(output_dir)
                    if getattr(args, "keep_best_only", False):
                        _prune_checkpoints_keep_best(
                            args.output_dir, current_steps, eval_loss_by_step, accelerator
                        )
'''
    if old_periodic_save not in src:
        raise RuntimeError(f"Could not locate periodic checkpoint save block in {path}")
    src = src.replace(old_periodic_save, new_periodic_save, 1)

    old_final_save = '''    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
    accelerator.save_state(output_dir)
'''
    new_final_save = '''    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
    accelerator.save_state(output_dir)
    if getattr(args, "keep_best_only", False):
        _prune_checkpoints_keep_best(args.output_dir, current_steps, eval_loss_by_step, accelerator)
'''
    if old_final_save not in src:
        raise RuntimeError(f"Could not locate final checkpoint save block in {path}")
    src = src.replace(old_final_save, new_final_save, 1)

    path.write_text(src, encoding="utf-8")
    return True


def patch_utils_data(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    updated = src.replace("np.concat(", "np.concatenate(")
    if updated == src:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    nu_repo = args.nu_repo.resolve()
    if not nu_repo.exists():
        raise SystemExit(f"nu repo not found: {nu_repo}")

    changes = []
    if patch_tokenize_text(nu_repo / "tools" / "tokenize_text.py"):
        changes.append("tools/tokenize_text.py")
    if patch_prepare_dataset(nu_repo / "tools" / "prepare_dataset.py"):
        changes.append("tools/prepare_dataset.py")
    if patch_finetune_tracking(nu_repo / "finetune.py"):
        changes.append("finetune.py")
    if patch_finetune_safe_means(nu_repo / "finetune.py"):
        changes.append("finetune.py:safe-means")
    if patch_finetune_mlflow_stdout_metrics(nu_repo / "finetune.py"):
        changes.append("finetune.py:mlflow-stdout-metrics")
    if patch_finetune_keep_best_only(nu_repo / "finetune.py"):
        changes.append("finetune.py:keep-best-only")
    if patch_utils_data(nu_repo / "utils" / "data.py"):
        changes.append("utils/data.py")

    if changes:
        print(f"[nu-patch] patched {', '.join(changes)}")
    else:
        print("[nu-patch] already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
