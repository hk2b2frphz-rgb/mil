#!/usr/bin/env python3
"""Apply local runtime patches to kyutai-labs/moshi-finetune.

The upstream LoRA training code can emit NaN eval loss when an eval segment has
no valid text/audio target positions. This script is intentionally idempotent so
it can run at every experiment launch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


LOSS_PY = '''import torch
from torch.nn import functional as F


def _target_and_weights(
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    return target, weights


def compute_loss_weight(
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
) -> torch.Tensor:
    _, weights = _target_and_weights(
        target,
        target_mask,
        mode,
        first_codebook_weight_multiplier,
        text_padding_weight,
        text_padding_ids,
    )
    return torch.sum(weights)


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    target, weights = _target_and_weights(
        target,
        target_mask,
        mode,
        first_codebook_weight_multiplier,
        text_padding_weight,
        text_padding_ids,
    )

    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    total_weight = torch.sum(weights)
    # AUTO_PATCH_SAFE_ZERO_WEIGHT_LOSS: avoid 0/0 on empty eval windows.
    safe_total_weight = torch.where(
        total_weight > 0.0,
        total_weight,
        torch.ones_like(total_weight),
    )
    mb_loss = torch.sum(mb_loss) / safe_total_weight

    return mb_loss
'''


EVAL_PY = '''import logging
from typing import Iterator

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from finetune.args import TrainArgs

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_weight, compute_loss_with_mask
from .utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    eval_data_loader: Iterator[Batch],
    state: TrainState,
    args: TrainArgs,
):
    # AUTO_PATCH_VALID_EVAL_COUNTS: average each eval component over batches that
    # actually contain valid targets. Padding-only windows otherwise make 0/0 NaN
    # or dilute the eval loss.
    max_eval_batches = max(1, 40 // get_world_size())
    num_samples = torch.tensor(0.0, device="cuda")
    text_count = torch.tensor(0.0, device="cuda")
    audio_count = torch.tensor(0.0, device="cuda")

    text_loss = torch.tensor(0.0, device="cuda")
    audio_loss = torch.tensor(0.0, device="cuda")
    model.eval()
    for batch in eval_data_loader:
        if int(num_samples.item()) >= max_eval_batches:
            break
        num_samples += 1.0
        with torch.no_grad():
            codes = batch.codes
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            output = model(codes=codes, condition_tensors=condition_tensors)
            text_target = codes[:, : model.audio_offset]
            audio_target = codes[:, model.audio_offset : model.audio_offset + model.dep_q]
            text_padding_ids = {
                model.text_padding_token_id,
                model.end_of_text_padding_id,
            }

            this_text_weight = compute_loss_weight(
                text_target,
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids=text_padding_ids,
            )
            this_audio_weight = compute_loss_weight(
                audio_target,
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            text_count += (this_text_weight > 0).float()
            audio_count += (this_audio_weight > 0).float()
            text_loss += compute_loss_with_mask(
                output.text_logits,
                text_target,
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids=text_padding_ids,
            )
            audio_loss += compute_loss_with_mask(
                output.logits,
                audio_target,
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

    main_logger_info("Eval finished!")

    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_loss, op=dist.ReduceOp.SUM)

    text_loss /= torch.clamp(text_count, min=1.0)
    audio_loss /= torch.clamp(audio_count, min=1.0)
    eval_loss = text_loss + audio_loss

    state.this_eval_loss = eval_loss.item()
    state.this_eval_perplexity = (2**eval_loss).item()
    state.this_audio_loss = audio_loss.item()
    state.this_text_loss = text_loss.item()
    state.this_eval_num_samples = num_samples.item()
    state.this_text_eval_batches = text_count.item()
    state.this_audio_eval_batches = audio_count.item()

    # train mode!
    model.train()
'''


def patch_loss(path: Path) -> bool:
    src = path.read_text()
    if "AUTO_PATCH_SAFE_ZERO_WEIGHT_LOSS" in src and "compute_loss_weight" in src:
        return False
    path.write_text(LOSS_PY)
    return True


def patch_eval(path: Path) -> bool:
    src = path.read_text()
    if "AUTO_PATCH_VALID_EVAL_COUNTS" in src:
        return False
    path.write_text(EVAL_PY)
    return True


def patch_metrics_logger(path: Path) -> bool:
    src = path.read_text()
    changed = False

    old_sig = '''def get_eval_logs(
    step: int,
    train_loss: float,
    perplexity: float | None = None,
    eval_loss: float | None = None,
    text_eval_loss: float | None = None,
    audio_eval_loss: float | None = None,
) -> dict[str, float | int]:
'''
    new_sig = '''def get_eval_logs(
    step: int,
    train_loss: float,
    perplexity: float | None = None,
    eval_loss: float | None = None,
    text_eval_loss: float | None = None,
    audio_eval_loss: float | None = None,
    eval_num_samples: float | int | None = None,
    text_eval_batches: float | int | None = None,
    audio_eval_batches: float | int | None = None,
) -> dict[str, float | int]:
'''
    if old_sig in src:
        src = src.replace(old_sig, new_sig, 1)
        changed = True

    old_body = '''    if audio_eval_loss is not None:
        eval_dict["audio_eval_loss"] = audio_eval_loss

    return eval_dict
'''
    new_body = '''    if audio_eval_loss is not None:
        eval_dict["audio_eval_loss"] = audio_eval_loss

    if eval_num_samples is not None:
        eval_dict["eval_num_samples"] = eval_num_samples

    if text_eval_batches is not None:
        eval_dict["text_eval_batches"] = text_eval_batches

    if audio_eval_batches is not None:
        eval_dict["audio_eval_batches"] = audio_eval_batches

    return eval_dict
'''
    if old_body in src:
        src = src.replace(old_body, new_body, 1)
        changed = True

    old_msg = '''        ("audio_eval_loss", ".3f", None),
    ]:
'''
    new_msg = '''        ("audio_eval_loss", ".3f", None),
        ("eval_num_samples", ".0f", None),
        ("text_eval_batches", ".0f", None),
        ("audio_eval_batches", ".0f", None),
    ]:
'''
    if old_msg in src:
        src = src.replace(old_msg, new_msg, 1)
        changed = True

    if changed:
        path.write_text(src)
    return changed


def patch_train(path: Path) -> bool:
    src = path.read_text()
    if 'getattr(state, "this_text_eval_batches", None)' in src:
        return False

    old = '''            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )
'''
    new = '''            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
                state.this_text_loss,
                state.this_audio_loss,
                getattr(state, "this_eval_num_samples", None),
                getattr(state, "this_text_eval_batches", None),
                getattr(state, "this_audio_eval_batches", None),
            )
'''
    if old not in src:
        raise RuntimeError(f"expected eval log block not found in {path}")
    path.write_text(src.replace(old, new, 1))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft-repo", required=True, type=Path)
    args = parser.parse_args()

    repo = args.ft_repo.resolve()
    required = [
        repo / "finetune" / "loss.py",
        repo / "finetune" / "eval.py",
        repo / "finetune" / "monitoring" / "metrics_logger.py",
        repo / "train.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing kyutai moshi-finetune files: " + ", ".join(missing))

    changed = []
    for label, func, path in [
        ("loss", patch_loss, required[0]),
        ("eval", patch_eval, required[1]),
        ("metrics_logger", patch_metrics_logger, required[2]),
        ("train", patch_train, required[3]),
    ]:
        if func(path):
            changed.append(label)

    if changed:
        print("[patch-kyutai] applied: " + ", ".join(changed))
    else:
        print("[patch-kyutai] already up to date")


if __name__ == "__main__":
    main()
