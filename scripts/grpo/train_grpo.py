#!/usr/bin/env python3
"""GRPO interactivity alignment training loop for Moshi.

Reproduces the Kyutai Multi-Faceted Interactivity Alignment paper
(arxiv 2606.11167) using segment-based training:

  1. Load per-axis segment datasets (D_pause, D_turn, D_bc, D_int)
  2. Each epoch: sample segments, optionally prepend context audio
  3. Feed user-side audio to Moshi → generate G rollouts
  4. Apply VAD to generated audio → compute axis-specific rewards
  5. GRPO policy update (text tokens only)

Launched via accelerate + DeepSpeed ZeRO3:
    accelerate launch --num_processes 2 --use_deepspeed \
        --deepspeed_config_file configs/deepspeed_grpo_zero3.json \
        scripts/grpo/train_grpo.py --config configs/grpo_loneliness.yaml
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.grpo.config import GRPOConfig
from scripts.grpo.llm_judge import LLMJudge
from scripts.grpo.rewards import (
    decoupling_normalize,
    reward_backchannel_f1,
    reward_pause_handling,
    reward_turn_taking,
    reward_user_interruption,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

AXES = ["pause", "turn_taking", "backchannel", "interruption"]
SAMPLE_RATE = 24_000


# ── Seed ─────────────────────────────────────────────────────────────

def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ── Segment dataset loading ─────────────────────────────────────────

def load_segment_datasets(
    segment_dir: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load per-axis segment JSONL files produced by segment_extractor.py."""
    seg_dir = Path(segment_dir)
    datasets: dict[str, list[dict[str, Any]]] = {}
    for axis in AXES:
        path = seg_dir / f"{axis}_segments.jsonl"
        items: list[dict[str, Any]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    items.append(json.loads(line.strip()))
        datasets[axis] = items
    return datasets


def load_segment_audio(
    segment: dict[str, Any],
    context_sec: float,
    context_prob: float,
    rng: random.Random,
) -> tuple[np.ndarray, float]:
    """Load user-side audio for a segment, optionally with context.

    Returns (user_pcm, segment_offset_sec) where segment_offset_sec
    is the time offset where the actual segment starts within the
    returned audio (= context length, 0 if no context).
    """
    import soundfile as sf

    wav_path = segment["wav_path"]
    seg_start = segment["segment_start_sec"]
    seg_end = segment["segment_end_sec"]

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import torchaudio
        t = torch.from_numpy(data.T)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        data = t.numpy().T
        sr = SAMPLE_RATE

    user_channel = data[:, 1]  # channel 1 = user (X)

    seg_start_sample = int(seg_start * sr)
    seg_end_sample = int(seg_end * sr)

    # Context: recording immediately preceding the segment (Sec 3.3)
    ctx_len_sec = 0.0
    if context_sec > 0 and rng.random() < context_prob:
        ctx_len_sec = rng.uniform(0, context_sec)
        ctx_start_sample = max(0, seg_start_sample - int(ctx_len_sec * sr))
        actual_ctx_sec = (seg_start_sample - ctx_start_sample) / sr
    else:
        ctx_start_sample = seg_start_sample
        actual_ctx_sec = 0.0

    pcm = user_channel[ctx_start_sample:seg_end_sample]
    return pcm.copy(), actual_ctx_sec


# ── Moshi model helpers ─────────────────────────────────────────────

def _getattr_chain(obj: Any, name: str, default: Any) -> Any:
    try:
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name)
    except AttributeError:
        return default


def _try_setattr(obj: Any, name: str, value: Any) -> None:
    try:
        if isinstance(obj, dict):
            obj[name] = value
        else:
            setattr(obj, name, value)
    except Exception:
        pass


def load_moshi_for_grpo(cfg: GRPOConfig):
    """Load Moshi to CPU (accelerator.prepare moves it to devices)."""
    from moshi.models import loaders
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(cfg.hf_repo)
    mimi = checkpoint_info.get_mimi(device="cpu")
    mimi.eval()
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm_model = checkpoint_info.get_moshi(device="cpu", dtype=torch.float32)
    return mimi, text_tokenizer, lm_model, checkpoint_info


def _ensure_lm_gen(lm_model, checkpoint_info, cfg: GRPOConfig):
    """Wrap lm_model in LMGen for streaming inference."""
    if hasattr(lm_model, "step") and hasattr(lm_model, "streaming"):
        return lm_model
    try:
        from moshi.models.lm import LMGen
    except ImportError:
        from moshi.models.lm_gen import LMGen

    lm_gen_cfg = checkpoint_info.lm_gen_config
    signature = inspect.signature(LMGen)
    supported = set(signature.parameters)
    kwargs = {}
    for name in ("temp", "temp_text", "top_k", "top_k_text", "cfg_coef"):
        value = _getattr_chain(lm_gen_cfg, name, None)
        if value is not None and name in supported:
            kwargs[name] = value
    lm_gen = LMGen(lm_model, **kwargs)
    _try_setattr(lm_gen, "temp", cfg.temp)
    _try_setattr(lm_gen, "temp_text", cfg.temp_text)
    return lm_gen


def _decode_text_piece(text_tokenizer: Any, text_id: int) -> str | None:
    if text_id in (0, 3) or text_id < 0:
        return None
    vocab_size = None
    for attr in ("get_piece_size", "vocab_size", "GetPieceSize"):
        method = getattr(text_tokenizer, attr, None)
        if callable(method):
            try:
                vocab_size = int(method())
                break
            except Exception:
                pass
    if vocab_size is not None and text_id >= vocab_size:
        return None
    for attr in ("id_to_piece", "IdToPiece", "decode"):
        method = getattr(text_tokenizer, attr, None)
        if callable(method):
            try:
                piece = method(text_id)
                if isinstance(piece, str):
                    return piece
                if isinstance(piece, list):
                    return "".join(piece)
            except Exception:
                pass
    return None


# ── Rollout generation ──────────────────────────────────────────────

def generate_rollout(
    user_pcm: np.ndarray,
    seed: int,
    lm_gen: Any,
    mimi: Any,
    text_tokenizer: Any,
    device: torch.device,
    context_offset_sec: float,
) -> dict[str, Any]:
    """Run Moshi inference on user audio, collect generated audio + text.

    The user_pcm is fed as Speaker X's input stream. Moshi generates
    Speaker Y's response in full-duplex mode.

    context_offset_sec: time offset where the actual segment starts
    (the preceding portion is context, masked from reward computation).
    """
    sample_rate = int(mimi.sample_rate)
    frame_rate = float(mimi.frame_rate)
    frame_size = int(sample_rate / frame_rate)
    n_steps = (len(user_pcm) + frame_size - 1) // frame_size

    _seed_all(seed)

    text_events: list[dict[str, Any]] = []
    text_token_ids: list[int] = []
    text_log_probs: list[float] = []
    audio_codes_in: list[torch.Tensor] = []
    audio_codes_out: list[torch.Tensor] = []

    with torch.no_grad():
        with lm_gen.streaming(1):
            with mimi.streaming(1):
                for step in range(n_steps):
                    start = step * frame_size
                    chunk_np = user_pcm[start: start + frame_size]
                    if len(chunk_np) < frame_size:
                        chunk_np = np.pad(chunk_np, (0, frame_size - len(chunk_np)))

                    chunk = (
                        torch.from_numpy(chunk_np)
                        .float()
                        .to(device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )

                    codes = mimi.encode(chunk)
                    if codes is None:
                        continue
                    audio_codes_in.append(codes.cpu())

                    out = lm_gen.step(codes)
                    if out is None:
                        continue

                    audio_codes_out.append(out.cpu())
                    text_id = int(out[0, 0, 0].item())
                    piece = _decode_text_piece(text_tokenizer, text_id)

                    if hasattr(lm_gen, "last_text_logits"):
                        logits = lm_gen.last_text_logits
                        if logits is not None:
                            log_prob = torch.log_softmax(logits, dim=-1)
                            text_log_probs.append(log_prob[0, text_id].item())
                            text_token_ids.append(text_id)

                    if piece is not None:
                        text_events.append({
                            "step": step,
                            "time_sec": round(step / frame_rate, 4),
                            "piece": piece,
                        })

    # Decode generated audio codes → PCM for VAD-based reward
    generated_pcm = _decode_audio_codes(audio_codes_out, mimi, device)

    return {
        "text_events": text_events,
        "text_token_ids": text_token_ids,
        "text_log_probs": text_log_probs,
        "audio_codes_in": audio_codes_in,
        "generated_pcm": generated_pcm,
        "context_offset_sec": context_offset_sec,
        "total_duration_sec": n_steps / frame_rate,
        "transcript": "".join(e["piece"] for e in text_events).strip(),
    }


def _decode_audio_codes(
    audio_codes_out: list[torch.Tensor],
    mimi: Any,
    device: torch.device,
) -> np.ndarray:
    """Decode Moshi's generated audio tokens back to PCM via mimi."""
    if not audio_codes_out:
        return np.zeros(0, dtype=np.float32)

    pcm_chunks = []
    with torch.no_grad():
        with mimi.streaming(1):
            for codes in audio_codes_out:
                # Extract audio codebook tokens (skip text token at index 0)
                audio_tokens = codes[:, 1:, :].to(device)
                try:
                    pcm = mimi.decode(audio_tokens)
                    if pcm is not None:
                        pcm_chunks.append(pcm.cpu().squeeze().numpy())
                except Exception:
                    pass

    if not pcm_chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pcm_chunks)


# ── VAD on generated audio ──────────────────────────────────────────

def _vad_on_generated(
    generated_pcm: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> list[tuple[float, float]]:
    """Run Silero VAD on Moshi's generated audio output."""
    if len(generated_pcm) < sr * 0.1:
        return []
    from scripts.grpo.segment_extractor import _run_vad
    utts = _run_vad(generated_pcm, sr)
    return [(u["start"], u["end"]) for u in utts]


# ── Axis-specific reward computation ────────────────────────────────

def compute_segment_reward(
    rollout: dict[str, Any],
    segment: dict[str, Any],
    cfg: GRPOConfig,
) -> float:
    """Compute the deterministic reward for a rollout based on its axis."""
    axis = segment["axis"]
    ctx_offset = rollout["context_offset_sec"]
    moshi_vad = _vad_on_generated(rollout["generated_pcm"])

    # Shift segment metadata times relative to context offset
    if axis == "pause":
        pauses = segment.get("internal_pauses", [])
        if not pauses:
            return 0.0
        total_r = 0.0
        for p in pauses:
            p_start = p["start"] - segment["segment_start_sec"] + ctx_offset
            p_end = p["end"] - segment["segment_start_sec"] + ctx_offset
            total_r += reward_pause_handling(
                moshi_vad, p_start, p_end, cfg.pause_threshold_sec,
            )
        return total_r / len(pauses)

    elif axis == "turn_taking":
        user_end = (
            segment["user_turn_end_sec"]
            - segment["segment_start_sec"]
            + ctx_offset
        )
        return reward_turn_taking(moshi_vad, user_end)

    elif axis == "backchannel":
        gt_times_abs = segment.get("gt_backchannel_times", [])
        gt_times = [
            t - segment["segment_start_sec"] + ctx_offset
            for t in gt_times_abs
        ]
        return reward_backchannel_f1(
            rollout["text_events"],
            gt_times,
            cfg.backchannel_window_sec,
            cfg.backchannel_max_dur_sec,
        )

    elif axis == "interruption":
        int_start = (
            segment["interruption_start_sec"]
            - segment["segment_start_sec"]
            + ctx_offset
        )
        return reward_user_interruption(moshi_vad, int_start)

    return 0.0


# ── Teacher-forced update pass ──────────────────────────────────────

def compute_update_log_probs(
    rollout: dict[str, Any],
    lm_gen: Any,
    mimi: Any,
    device: torch.device,
    context_steps: int,
) -> list[torch.Tensor]:
    """Re-run rollout in train mode for differentiable text log probs.

    Only collects log probs for steps AFTER context_steps (context is
    masked from loss, per Kyutai Section 3.3).
    """
    audio_codes = rollout["audio_codes_in"]
    text_token_ids = rollout["text_token_ids"]

    if not text_token_ids or not audio_codes:
        return []

    new_log_probs: list[torch.Tensor] = []
    text_idx = 0
    frame_rate = float(mimi.frame_rate)

    with lm_gen.streaming(1):
        with mimi.streaming(1):
            for step_idx, codes_cpu in enumerate(audio_codes):
                codes = codes_cpu.to(device)
                out = lm_gen.step(codes)
                if out is None:
                    continue

                if step_idx < context_steps:
                    continue

                if text_idx >= len(text_token_ids):
                    break

                if hasattr(lm_gen, "last_text_logits"):
                    logits = lm_gen.last_text_logits
                    if logits is not None:
                        log_prob = torch.log_softmax(logits, dim=-1)
                        tid = text_token_ids[text_idx]
                        new_log_probs.append(log_prob[0, tid])
                        text_idx += 1

    return new_log_probs


# ── GRPO loss ────────────────────────────────────────────────────────

def compute_grpo_loss(
    advantages: list[float],
    old_log_probs: list[list[float]],
    new_log_probs: list[list[Any]],
    ref_log_probs: list[list[float]],
    clip_epsilon: float,
    kl_beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Clipped surrogate GRPO loss with KL penalty (Kyutai Eq. 3-4).

    new_log_probs: differentiable tensors from teacher-forced pass.
    old_log_probs / ref_log_probs: detached floats from rollout.
    """
    losses: list[torch.Tensor] = []

    for i, adv in enumerate(advantages):
        if not old_log_probs[i] or not new_log_probs[i]:
            continue

        min_len = min(len(old_log_probs[i]), len(new_log_probs[i]))
        if min_len == 0:
            continue

        old_lp = torch.tensor(old_log_probs[i][:min_len], device=device)
        if isinstance(new_log_probs[i][0], torch.Tensor):
            new_lp = torch.stack(new_log_probs[i][:min_len]).to(device)
        else:
            new_lp = torch.tensor(new_log_probs[i][:min_len], device=device)

        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = -torch.min(ratio * adv, clipped * adv).mean()

        kl_loss = torch.tensor(0.0, device=device)
        if ref_log_probs[i]:
            ref_min = min(min_len, len(ref_log_probs[i]))
            ref_lp = torch.tensor(ref_log_probs[i][:ref_min], device=device)
            new_lp_kl = new_lp[:ref_min]
            kl = (torch.exp(new_lp_kl) * (new_lp_kl - ref_lp)).mean()
            kl_loss = kl_beta * kl

        losses.append(surrogate + kl_loss)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ── Main training loop ──────────────────────────────────────────────

def train(cfg: GRPOConfig) -> None:
    import deepspeed
    from accelerate import Accelerator
    from accelerate.utils import (
        DummyOptim,
        DummyScheduler,
        InitProcessGroupKwargs,
        set_seed,
    )

    process_group_kwargs = InitProcessGroupKwargs(
        timeout=__import__("datetime").timedelta(seconds=3600),
    )
    accelerator = Accelerator(
        kwargs_handlers=[process_group_kwargs],
        gradient_accumulation_steps=1,
    )
    is_main = accelerator.is_main_process
    device = accelerator.device

    if cfg.seed is not None:
        set_seed(cfg.seed)

    save_dir = Path(cfg.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_json(save_dir / "config.json")

    # ── Load segment datasets ──
    if not cfg.segment_dir:
        raise ValueError("segment_dir must be set in config")
    seg_datasets = load_segment_datasets(cfg.segment_dir)
    total_segs = sum(len(v) for v in seg_datasets.values())
    if is_main:
        for axis, items in seg_datasets.items():
            logger.info("  %s: %d segments", axis, len(items))
        logger.info("  Total: %d segments", total_segs)
    if total_segs == 0:
        raise ValueError(f"No segments found in {cfg.segment_dir}")

    # Build flat list of (axis, segment) for sampling
    all_segments: list[tuple[str, dict[str, Any]]] = []
    for axis, items in seg_datasets.items():
        for item in items:
            all_segments.append((axis, item))

    # ── Load Moshi to CPU ──
    if is_main:
        logger.info("Loading Moshi models to CPU...")
    mimi, text_tokenizer, lm_model, checkpoint_info = load_moshi_for_grpo(cfg)

    if hasattr(accelerator, "deepspeed_plugin") and accelerator.deepspeed_plugin is not None:
        ds_config = accelerator.deepspeed_plugin.hf_ds_config.config
        act_ckpt_kwargs = ds_config.get("activation_checkpointing", {})
        if act_ckpt_kwargs:
            deepspeed.checkpointing.configure(
                mpu_=None, deepspeed_config=None, **act_ckpt_kwargs,
            )
            if hasattr(lm_model, "enable_activation_checkpointing"):
                lm_model.enable_activation_checkpointing(
                    deepspeed.checkpointing.checkpoint,
                )

    for param in lm_model.parameters():
        param.requires_grad = True

    optimizer = DummyOptim(
        lm_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer.defaults = {"lr": cfg.lr}

    lr_scheduler = DummyScheduler(
        optimizer=optimizer,
        warmup_num_steps=0,
    )

    lm_model, optimizer, lr_scheduler = accelerator.prepare(
        lm_model, optimizer, lr_scheduler,
    )

    mimi = mimi.to(device)
    mimi.eval()

    unwrapped = accelerator.unwrap_model(lm_model)
    lm_gen = _ensure_lm_gen(unwrapped, checkpoint_info, cfg)

    judge = LLMJudge(
        model=cfg.judge_model,
        max_tokens=8,
        temperature=cfg.judge_temperature,
    )

    frame_rate = float(mimi.frame_rate)

    if is_main:
        logger.info("***** Starting GRPO training *****")
        logger.info("  Epochs: %d", cfg.epochs)
        logger.info("  Segments/epoch: %d", cfg.segments_per_epoch)
        logger.info("  Group size G: %d", cfg.group_size)
        logger.info("  LR: %s", cfg.lr)
        logger.info("  KL beta: %s", cfg.kl_beta)
        logger.info("  Context schedule: 0 → %.0fs", cfg.max_context_sec)
        logger.info("  Processes: %d", accelerator.num_processes)

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        context_sec = cfg.current_context_sec(epoch)
        if is_main:
            logger.info(
                "=== Epoch %d/%d | max_context=%.1fs ===",
                epoch + 1, cfg.epochs, context_sec,
            )

        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []

        # Deterministic segment sampling (all ranks same)
        rng = random.Random(cfg.seed + epoch)
        epoch_segments = rng.sample(
            all_segments, min(cfg.segments_per_epoch, len(all_segments)),
        )

        for s_idx, (axis, segment) in enumerate(epoch_segments):
            if is_main:
                logger.info(
                    "  Segment %d/%d [%s]: %s",
                    s_idx + 1, len(epoch_segments), axis,
                    Path(segment["wav_path"]).name,
                )

            # Load user audio + optional context
            user_pcm, ctx_offset = load_segment_audio(
                segment, context_sec, cfg.context_prob, rng,
            )
            context_steps = int(ctx_offset * frame_rate)

            # ── Phase 1: Generate G rollouts ──
            lm_model.eval()
            group_rollouts: list[dict[str, Any]] = []
            for g in range(cfg.group_size):
                seed = cfg.seed + epoch * cfg.group_size + g
                rollout = generate_rollout(
                    user_pcm=user_pcm,
                    seed=seed,
                    lm_gen=lm_gen,
                    mimi=mimi,
                    text_tokenizer=text_tokenizer,
                    device=device,
                    context_offset_sec=ctx_offset,
                )
                group_rollouts.append(rollout)
                if is_main:
                    logger.info(
                        "    Rollout %d: %s",
                        g, rollout["transcript"][:50],
                    )

            # ── Phase 2: Axis-specific deterministic rewards ──
            axis_rewards: list[float] = []
            for rollout in group_rollouts:
                r = compute_segment_reward(rollout, segment, cfg)
                axis_rewards.append(r)

            # ── Phase 3: LLM judge (turn_taking and interruption only) ──
            judge_scores: list[float] = [0.0] * cfg.group_size
            use_judge = axis in ("turn_taking", "interruption")

            if use_judge and is_main:
                logger.info("    Computing LLM judge for %s...", axis)
                user_text = segment.get("user_text", "")
                if not user_text:
                    user_text = f"[{axis}セグメント]"
                pairs = [
                    (user_text, r["transcript"])
                    for r in group_rollouts
                ]
                judge.load()
                judge_scores = judge.score_batch(pairs)
                judge.unload()
                torch.cuda.empty_cache()

            if use_judge:
                judge_tensor = torch.tensor(judge_scores, device=device)
                torch.distributed.broadcast(judge_tensor, src=0)
                judge_scores = judge_tensor.tolist()

            # ── Phase 4: Reward normalization ──
            # Per-axis z-normalization within the group, then combine.
            # For turn_taking/interruption: axis_reward + judge_reward
            rewards_to_normalize = [axis_rewards]
            weights = [1.0]
            if use_judge:
                rewards_to_normalize.append(judge_scores)
                weights.append(cfg.judge_weight)

            advantages = decoupling_normalize(rewards_to_normalize, weights)

            mean_reward = sum(advantages) / len(advantages) if advantages else 0.0
            epoch_rewards.append(mean_reward)

            if is_main:
                mean_axis = sum(axis_rewards) / max(len(axis_rewards), 1)
                msg = f"    R_{axis}={mean_axis:.3f}"
                if use_judge:
                    mean_judge = sum(judge_scores) / max(len(judge_scores), 1)
                    msg += f" R_judge={mean_judge:.3f}"
                msg += f" -> adv={mean_reward:.3f}"
                logger.info(msg)

            # ── Phase 5: GRPO update ──
            lm_model.train()
            old_log_probs = [r["text_log_probs"] for r in group_rollouts]
            ref_log_probs = old_log_probs

            if is_main:
                logger.info("    Teacher-forced update pass...")

            new_log_probs: list[list[torch.Tensor]] = []
            for rollout in group_rollouts:
                nlp = compute_update_log_probs(
                    rollout=rollout,
                    lm_gen=lm_gen,
                    mimi=mimi,
                    device=device,
                    context_steps=context_steps,
                )
                new_log_probs.append(nlp)

            loss = compute_grpo_loss(
                advantages=advantages,
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                ref_log_probs=ref_log_probs,
                clip_epsilon=cfg.clip_epsilon,
                kl_beta=cfg.kl_beta,
                device=device,
            )

            accelerator.backward(loss)

            epoch_losses.append(loss.item())
            if is_main:
                logger.info("    Loss: %.6f", loss.item())

        # ── Epoch summary ──
        elapsed = time.time() - epoch_start
        mean_r = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        mean_l = sum(epoch_losses) / max(len(epoch_losses), 1)
        if is_main:
            logger.info(
                "Epoch %d done in %.1fs | mean_reward=%.4f mean_loss=%.6f",
                epoch + 1, elapsed, mean_r, mean_l,
            )
            log_entry = {
                "epoch": epoch + 1,
                "context_sec": context_sec,
                "mean_reward": mean_r,
                "mean_loss": mean_l,
                "elapsed_sec": elapsed,
            }
            log_path = save_dir / "training_log.jsonl"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_dir = str(save_dir / f"checkpoint_epoch_{epoch + 1:04d}")
            accelerator.save_state(ckpt_dir)
            if is_main:
                logger.info("Saved checkpoint: %s", ckpt_dir)

    if is_main:
        final_dir = str(save_dir / f"checkpoint_epoch_{cfg.epochs:04d}")
        accelerator.save_state(final_dir)
        logger.info("Training complete. Final checkpoint: %s", final_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--segment-dir", type=str, default=None,
                        help="Override segment_dir from config")
    args = parser.parse_args()
    cfg = GRPOConfig.from_yaml(args.config)

    # Allow override via CLI arg or env var
    seg_dir = args.segment_dir or os.environ.get("GRPO_SEGMENT_DIR", "")
    if seg_dir:
        cfg.segment_dir = seg_dir

    train(cfg)


if __name__ == "__main__":
    main()
