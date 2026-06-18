#!/usr/bin/env python3
"""Apply local runtime patches to kyutai-labs/moshi-finetune.

Patches applied (all idempotent, run at every experiment launch):
- optimizer: add a selectable warmup-then-constant learning-rate schedule while
  preserving the upstream OneCycleLR default.
- loss/eval/metrics: the upstream LoRA training code can emit NaN eval loss when
  an eval segment has no valid text/audio target positions; average each
  component only over batches with valid targets and expose eval batch counts.
- data_loader: the eval loader only yielded full batches and dropped the
  remainder, so an eval split smaller than batch_size produced zero eval batches
  (eval loss / eval_num_samples == 0). Flush the final partial batch for eval.
- train: rebuild the finite eval generator for every evaluation. The upstream
  code creates it once, so only the first evaluation sees data and every later
  evaluation silently records zero samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_args(path: Path) -> bool:
    src = path.read_text()
    marker = 'scheduler: str = "one_cycle"'
    if marker in src:
        return False

    old = '''class OptimArgs(Serializable):
    lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05
'''
    new = '''class OptimArgs(Serializable):
    lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05
    # "one_cycle" keeps the upstream behavior. "warmup_constant" linearly
    # warms up for pct_start * max_steps, then holds lr constant.
    scheduler: str = "one_cycle"

    def __post_init__(self) -> None:
        if self.scheduler not in {"one_cycle", "warmup_constant"}:
            raise ValueError(
                "optim.scheduler must be 'one_cycle' or 'warmup_constant', "
                f"got {self.scheduler!r}"
            )
        if not 0.0 <= self.pct_start <= 1.0:
            raise ValueError(f"optim.pct_start must be in [0, 1], got {self.pct_start}")
'''
    if old not in src:
        raise RuntimeError(f"expected OptimArgs block not found in {path}")
    path.write_text(src.replace(old, new, 1))
    return True


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
    nonfinite_loss_batches = torch.tensor(0.0, device="cuda")
    raw_nonfinite_logits_batches = torch.tensor(0.0, device="cuda")
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

            this_text_loss = compute_loss_with_mask(
                output.text_logits,
                text_target,
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids=text_padding_ids,
            )
            this_audio_loss = compute_loss_with_mask(
                output.logits,
                audio_target,
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            # AUTO_PATCH_MASKED_LOSS_FINITE_GUARD: validity is determined by the
            # masked loss, exactly as in training. Raw logits may contain
            # non-finite values at zero-weight positions; rejecting the whole
            # tensor in that case incorrectly drops an otherwise valid batch.
            text_logits_finite = bool(torch.isfinite(output.text_logits).all().item())
            audio_logits_finite = bool(torch.isfinite(output.logits).all().item())
            text_loss_finite = bool(torch.isfinite(this_text_loss).item())
            audio_loss_finite = bool(torch.isfinite(this_audio_loss).item())

            if not (text_logits_finite and audio_logits_finite):
                raw_nonfinite_logits_batches += 1.0
                main_logger_info(
                    f"[eval] raw logits contain non-finite values at step {state.step}: "
                    f"text_logits_finite={text_logits_finite} "
                    f"audio_logits_finite={audio_logits_finite} "
                    f"text_loss_finite={text_loss_finite} "
                    f"audio_loss_finite={audio_loss_finite} "
                    f"text_weight={float(this_text_weight.item()):.1f} "
                    f"audio_weight={float(this_audio_weight.item()):.1f}"
                )

            if not (text_loss_finite and audio_loss_finite):
                nonfinite_loss_batches += 1.0
                main_logger_info(
                    f"[eval] non-finite masked loss at step {state.step}: "
                    f"text_loss_finite={text_loss_finite} "
                    f"audio_loss_finite={audio_loss_finite} "
                    f"text_weight={float(this_text_weight.item()):.1f} "
                    f"audio_weight={float(this_audio_weight.item()):.1f}"
                )

            if text_loss_finite:
                text_count += (this_text_weight > 0).float()
                text_loss += this_text_loss
            if audio_loss_finite:
                audio_count += (this_audio_weight > 0).float()
                audio_loss += this_audio_loss

    main_logger_info("Eval finished!")

    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(nonfinite_loss_batches, op=dist.ReduceOp.SUM)
    dist.all_reduce(raw_nonfinite_logits_batches, op=dist.ReduceOp.SUM)

    # AUTO_PATCH_REQUIRE_NONEMPTY_EVAL: a zero-sample evaluation is never a
    # meaningful metric. This most commonly means the finite generator was
    # reused after exhaustion, the manifest resolved no audio, or the split is
    # too small for the distributed world size.
    if num_samples.item() <= 0:
        raise RuntimeError(
            "eval dataset yielded zero batches; check that eval_data contains "
            "valid audio paths and rebuild the eval loader for every evaluation"
        )
    if text_count.item() <= 0 or audio_count.item() <= 0:
        raise RuntimeError(
            "eval dataset produced no finite valid targets: "
            f"text_batches={int(text_count.item())}, "
            f"audio_batches={int(audio_count.item())}, "
            f"nonfinite_loss_batches={int(nonfinite_loss_batches.item())}, "
            f"raw_nonfinite_logits_batches={int(raw_nonfinite_logits_batches.item())}"
        )

    text_loss /= torch.clamp(text_count, min=1.0)
    audio_loss /= torch.clamp(audio_count, min=1.0)
    eval_loss = text_loss + audio_loss

    if nonfinite_loss_batches.item() > 0:
        main_logger_info(
            f"[eval] dropped {int(nonfinite_loss_batches.item())} batch(es) "
            "with non-finite masked loss; averaged over finite losses only."
        )
    if raw_nonfinite_logits_batches.item() > 0:
        main_logger_info(
            f"[eval] observed raw non-finite logits in "
            f"{int(raw_nonfinite_logits_batches.item())} batch(es), but retained "
            "batches whose masked text/audio losses were finite."
        )

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


DATA_LOADER_PY = '''from typing import Any, Iterator

from .args import DataArgs
from .dataset import build_dataset
from .interleaver import Batch


def build_data_loader(
    instruct_tokenizer: Any,
    args: DataArgs,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
) -> Iterator[Batch]:
    if is_eval:
        assert args.eval_data != "", "No eval data provided."
    pretrain_data = args.train_data if not is_eval else args.eval_data

    dataset = build_dataset(
        pretrain_data=pretrain_data,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle_pretrain=args.shuffle,
    )

    sample_list = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)

        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []

    # AUTO_PATCH_FLUSH_EVAL_PARTIAL_BATCH: the eval loader is finite, so emit the
    # remaining samples as a final (smaller) batch. Otherwise an eval split with
    # fewer than batch_size samples -- or any non-multiple remainder -- yields
    # zero batches and eval loss / eval_num_samples come out as 0. The train
    # loader is infinite, so this branch never fires for training.
    if is_eval and sample_list:
        yield Batch.collate(sample_list)
'''


def patch_data_loader(path: Path) -> bool:
    src = path.read_text()
    if "AUTO_PATCH_FLUSH_EVAL_PARTIAL_BATCH" in src:
        return False
    path.write_text(DATA_LOADER_PY)
    return True


def patch_loss(path: Path) -> bool:
    src = path.read_text()
    if "AUTO_PATCH_SAFE_ZERO_WEIGHT_LOSS" in src and "compute_loss_weight" in src:
        return False
    path.write_text(LOSS_PY)
    return True


def patch_eval(path: Path) -> bool:
    src = path.read_text()
    # Keyed on the latest marker so older patched checkouts are upgraded in
    # place (the whole file is rewritten from EVAL_PY).
    if "AUTO_PATCH_MASKED_LOSS_FINITE_GUARD" in src:
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
    changed = False

    if "AUTO_PATCH_SELECTABLE_LR_SCHEDULER" not in src:
        old_scheduler = '''    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )
'''
        new_scheduler = '''    # AUTO_PATCH_SELECTABLE_LR_SCHEDULER: retain the upstream
    # OneCycleLR default, while allowing a fair warmup-then-constant comparison.
    if args.optim.scheduler == "one_cycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.optim.lr,
            total_steps=args.max_steps,
            pct_start=args.optim.pct_start,
        )
    elif args.optim.scheduler == "warmup_constant":
        warmup_steps = int(args.max_steps * args.optim.pct_start)

        def lr_lambda(step: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            return min(float(step + 1) / float(warmup_steps), 1.0)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"unknown optimizer scheduler: {args.optim.scheduler!r}")
'''
        if old_scheduler not in src:
            raise RuntimeError(f"expected scheduler block not found in {path}")
        src = src.replace(old_scheduler, new_scheduler, 1)
        changed = True

    if "AUTO_PATCH_REBUILD_EVAL_LOADER" not in src:
        old_eval_loader = '''    if args.do_eval:
        eval_data_loader = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
        )
'''
        new_eval_loader = '''    if args.do_eval:
        # AUTO_PATCH_REBUILD_EVAL_LOADER: build_data_loader returns a finite
        # generator for eval. Return a fresh one for every evaluation instead
        # of reusing the exhausted generator from the first eval.
        def build_eval_data_loader():
            return build_data_loader(
                instruct_tokenizer=interleaved_tokenizer,
                args=args.data,
                batch_size=args.batch_size,
                seed=None,
                rank=get_rank(),  # DDP rank
                world_size=get_world_size(),  # DDP world_size
                is_eval=True,
            )
'''
        if old_eval_loader not in src:
            raise RuntimeError(f"expected eval data loader block not found in {path}")
        src = src.replace(old_eval_loader, new_eval_loader, 1)
        old_eval_call = "            evaluate(model, eval_data_loader, state, args)"
        new_eval_call = "            evaluate(model, build_eval_data_loader(), state, args)"
        if old_eval_call not in src:
            raise RuntimeError(f"expected evaluate call not found in {path}")
        src = src.replace(old_eval_call, new_eval_call, 1)
        changed = True

    if 'getattr(state, "this_text_eval_batches", None)' not in src:
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
        src = src.replace(old, new, 1)
        changed = True

    if changed:
        path.write_text(src)
    return changed


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
        repo / "finetune" / "data" / "data_loader.py",
        repo / "finetune" / "args.py",
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
        ("data_loader", patch_data_loader, required[4]),
        ("args", patch_args, required[5]),
    ]:
        if func(path):
            changed.append(label)

    if changed:
        print("[patch-kyutai] applied: " + ", ".join(changed))
    else:
        print("[patch-kyutai] already up to date")


if __name__ == "__main__":
    main()
